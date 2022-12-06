//! This is an adaptation of the vector based architecture which implements all
//! the pruning techniques that I have proposed in my PhD thesis (RUB, LocB, EBPO).

use std::{sync::Arc, hash::Hash, collections::hash_map::Entry};

use fxhash::FxHashMap;

use crate::{NodeFlags, Decision, CutsetType, CompilationInput, Completion, Reason, CompilationType, Problem, LAST_EXACT_LAYER, DecisionDiagram, SubProblem, FRONTIER, Solution};

/// The identifier of a node: it indicates the position of the referenced node 
/// in the ’nodes’ vector of the mdd structure.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct NodeId(usize);

/// The identifier of an edge: it indicates the position of the referenced edge 
/// in the ’edges’ vector of the mdd structure.
#[derive(Debug, Clone, Copy)]
struct EdgeId(usize);

/// The identifier of an edge list: it indicates the position of an edge list
/// in the ’edgelists’ vector of the mdd structure.
#[derive(Debug, Clone, Copy)]
struct EdgesListId(usize);

/// The identifier of a layer: it indicates the position of the referenced layer 
/// in the 'layers' vector of the mdd structure.
#[derive(Debug, Clone, Copy)]
struct LayerId(usize);

/// Represents an effective node from the decision diagram
#[derive(Debug, Clone)]
struct Node<T> {
    /// The state associated to this node
    state: Arc<T>,
    /// The length of the longest path between the problem root and this
    /// specific node
    value_top: isize,
    /// The length of the longest path between this node and the terminal node.
    /// 
    /// ### Note
    /// This field is only ever populated after the MDD has been fully unrolled.
    value_bot: isize,
    /// The identifier of the last edge on the longest path between the problem 
    /// root and this node if it exists.
    best: Option<EdgeId>,
    /// The identifier of the latest edge having been added to the adjacency
    /// list of this node. (Edges, by themselves form a kind of linked structure)
    inbound: EdgesListId,
    // The rough upper bound associated to this node
    rub: isize,
    /// A group of flag telling if the node is an exact node, if it is a relaxed
    /// node (helps to determine if the best path is an exact path) and if the
    /// node is reachable in a backwards traversal of the MDD starting at the
    /// terminal node.
    flags: NodeFlags,
    /// The number of decisions that have been made since the problem root
    depth: usize,
}

/// Materializes one edge a.k.a arc from the decision diagram. It logically 
/// connects two nodes and annotates the link with a decision and a cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Edge {
    /// The identifier of the node at the ∗∗source∗∗ of this edge.
    from: NodeId,
    /// The identifier of the node at the ∗∗destinaition∗∗ of this edge.
    to: NodeId,
    /// This is the decision label associated to this edge. It gives the 
    /// information "what variable" is assigned to "what value".
    decision: Decision,
    /// This is the transition cost of making this decision from the state
    /// associated with the source node of this edge.
    cost: isize,
}

/// Represents a 'node' in the linked list that forms the adjacent edges list for a node 
#[derive(Debug, Clone, Copy)]
enum EdgesList {
    Cons {head: EdgeId, tail: EdgesListId},
    Nil
}

/// Represents a 'layer' in the decision diagram
#[derive(Debug, Clone, Copy)]
struct Layer {
    from: usize,
    to: usize,
}

/// The decision diagram in itself. This structure essentially keeps track
/// of the nodes composing the diagam as well as the edges connecting these
/// nodes in two vectors (enabling preallocation and good cache locality). 
/// In addition to that, it also keeps track of the path (root_pa) from the
/// problem root to the root of this decision diagram (explores a sub problem). 
/// The prev_l comprises information about the nodes that are currently being
/// expanded, next_l stores the information about the nodes from the next layer 
/// and cutset stores an exact cutset of the DD.
/// Depending on the type of DD compiled, different cutset types will be used:
/// - Exact: no cutset is needed since the DD is exact
/// - Restricted: the last exact layer is used as cutset
/// - Relaxed: either the last exact layer of the frontier cutset can be chosen
///            within the CompilationInput
#[derive(Debug, Clone)]
pub struct Mdd<T, const CUTSET_TYPE: CutsetType>
where
    T: Eq + PartialEq + Hash + Clone,
{
    /// This vector stores the information about the structure of all the layers
    /// in this decision diagram
    layers: Vec<Layer>,
    /// All the nodes composing this decision diagram. The vector comprises 
    /// nodes from all layers in the DD. A nice property is that all nodes
    /// belonging to one same layer form a sequence in the ‘nodes‘ vector.
    nodes: Vec<Node<T>>,
    /// This vector stores the information about all edges connecting the nodes 
    /// of the decision diagram.
    edges: Vec<Edge>,
    /// This vector stores the information about all edge lists consituting 
    /// linked lists between edges
    edgelists: Vec<EdgesList>,
    
    /// Contains the nodes of the layer which is currently being expanded.
    /// This collection is only used during the unrolling of transition relation,
    /// and when merging nodes of a relaxed DD.
    prev_l: Vec<NodeId>,
    /// The nodes from the next layer; those are the result of an application 
    /// of the transition function to a node in ‘prev_l‘.
    /// Note: next_l in itself is indexed on the state associated with nodes.
    /// The rationale being that two transitions to the same state in the same
    /// layer should lead to the same node. This indexation helps ensuring 
    /// the uniqueness constraint in amortized O(1).
    next_l: FxHashMap<Arc<T>, NodeId>,

    /// Keeps track of the decisions that have been taken to reach the root
    /// of this DD, starting from the problem root.
    path_to_root: Vec<Decision>,
    /// The identifier of the last exact layer (should this dd be inexact)
    lel: Option<LayerId>,
    /// The cutset of the decision diagram (only maintained for relaxed dd)
    cutset: Vec<NodeId>,
    /// The identifier of the best terminal node of the diagram (None when the
    /// problem compiled into this dd is infeasible)
    best_node: Option<NodeId>,
    /// A flag set to true when the longest r-t path of this decision diagram
    /// traverses no merged node (Exact Best Path Optimization aka EBPO).
    is_exact: bool,
}

const NIL: EdgesListId = EdgesListId(0);


// Tech note: WHY AM I USING MACROS HERE ? 
// ---> Simply to avoid the need to fight the borrow checker

/// These macro retrieve an element of the dd by its id
macro_rules! get {
    (    node     $id:expr, $dd:expr) => {&    $dd.nodes   [$id.0]};
    (mut node     $id:expr, $dd:expr) => {&mut $dd.nodes   [$id.0]};
    (    edge     $id:expr, $dd:expr) => {&    $dd.edges   [$id.0]};
    (mut edge     $id:expr, $dd:expr) => {&mut $dd.edges   [$id.0]};
    (    edgelist $id:expr, $dd:expr) => {&    $dd.edgelists[$id.0]};
    (mut edgelist $id:expr, $dd:expr) => {&mut $dd.edgelists[$id.0]};
    (    layer    $id:expr, $dd:expr) => {&    $dd.layers  [$id.0]};
    (mut layer    $id:expr, $dd:expr) => {&mut $dd.layers  [$id.0]};
}

/// This macro performs an action for each edge of a given node in the dd
macro_rules! foreach {
    (edge of $id:expr, $dd:expr, $action:expr) => {
        let mut list = get!(node $id, $dd).inbound;
        while let EdgesList::Cons{head, tail} = *get!(edgelist list, $dd) {
            let edge = *get!(edge head, $dd);
            $action(edge);
            list = tail;
        }
    };
}

/// This macro appends an edge to the list of edges adjacent to a given node
macro_rules! append_edge_to {
    ($dd:expr, $id:expr, $edge:expr) => {
        let new_eid = EdgeId($dd.edges.len());
        let lst_id  = EdgesListId($dd.edgelists.len());
        $dd.edges.push($edge);
        $dd.edgelists.push(EdgesList::Cons { head: new_eid, tail: get!(node $id, $dd).inbound });
        
        let parent = get!(node $edge.from, $dd);
        let parent_exact = parent.flags.is_exact();
        let value = parent.value_top.saturating_add($edge.cost);
        
        let node = get!(mut node $id, $dd);
        let exact = parent_exact & node.flags.is_exact();
        node.flags.set_exact(exact);
        node.inbound = lst_id;

        if value >= node.value_top {
            node.best = Some(new_eid);
            node.value_top = value;
        }
    };
}

impl<T, const CUTSET_TYPE: CutsetType> Default for Mdd<T, {CUTSET_TYPE}>
where
    T: Eq + PartialEq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CUTSET_TYPE: CutsetType> DecisionDiagram for Mdd<T, {CUTSET_TYPE}>
where
    T: Eq + PartialEq + Hash + Clone,
{
    type State = T;

    fn compile(&mut self, input: &CompilationInput<Self::State>) -> Result<Completion, Reason> {
        self._compile(input)
    }

    fn is_exact(&self) -> bool {
        self.is_exact
    }

    fn best_value(&self) -> Option<isize> {
        self._best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self._best_solution()
    }

    fn drain_cutset<F>(&mut self, func: F)
    where
        F: FnMut(SubProblem<Self::State>) {
        self._drain_cutset(func)
    }
}

impl<T, const CUTSET_TYPE: CutsetType> Mdd<T, {CUTSET_TYPE}>
where
    T: Eq + PartialEq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            layers: vec![],
            nodes: vec![],
            edges: vec![],
            edgelists: vec![],
            //
            prev_l: vec![],
            next_l: Default::default(),
            //
            path_to_root: vec![],
            lel: None,
            cutset: vec![],
            best_node: None,
            is_exact: true,
        }
    }
    
    fn _clear(&mut self) {
        self.layers.clear();
        self.nodes.clear();
        self.edges.clear();
        self.edgelists.clear();
        self.prev_l.clear();
        self.next_l.clear();
        self.path_to_root.clear();
        self.cutset.clear();
        self.lel = None;
        self.best_node = None;
        self.is_exact = true;
    }

    fn _best_value(&self) -> Option<isize> {
        self.best_node.map(|id| get!(node id, self).value_top)
    }

    fn _best_solution(&self) -> Option<Vec<Decision>> {
        self.best_node.map(|id| self._best_path(id))
    }

    fn _best_path(&self, id: NodeId) -> Vec<Decision> {
        Self::_best_path_partial_borrow(id, &self.path_to_root, &self.nodes, &self.edges)
    }

    fn _best_path_partial_borrow(
        id: NodeId,
        root_pa: &[Decision],
        nodes: &[Node<T>],
        edges: &[Edge],
    ) -> Vec<Decision> {
        let mut sol = root_pa.to_owned();
        let mut edge_id = nodes[id.0].best;
        while let Some(eid) = edge_id {
            let edge = edges[eid.0];
            sol.push(edge.decision);
            edge_id = nodes[edge.from.0].best;
        }
        sol
    }

    fn _compile(&mut self, input: &CompilationInput<T>) -> Result<Completion, Reason> {
        self._clear();
        self._initialize(input);
        
        let mut curr_l = vec![];
        while let Some(var) = input.problem.next_variable(&mut self.next_l.keys().map(|s| s.as_ref())) {
            // Did the cutoff kick in ?
            if input.cutoff.must_stop() {
                return Err(Reason::CutoffOccurred);
            }
            
            if !self._move_to_next_layer(input, &mut curr_l) {
                break;
            }

            for node_id in curr_l.iter() {
                let state = self.nodes[node_id.0].state.clone();
                let rub = input.relaxation.fast_upper_bound(state.as_ref());
                self.nodes[node_id.0].rub = rub;
                let ub = rub.saturating_add(self.nodes[node_id.0].value_top);
                if ub > input.best_lb {
                    input.problem.for_each_in_domain(var, state.as_ref(), &mut |decision| {
                        self._branch_on(*node_id, decision, input.problem)
                    })
                }
            }
        }

        self._finalize(input);

        Ok(Completion { 
            is_exact: self.is_exact, 
            best_value: self.best_node.map(|n| get!(node n, self).value_top) 
        })
    }

    fn _initialize(&mut self, input: &CompilationInput<T>) {
        self.path_to_root.extend_from_slice(&input.residual.path);
        self.edgelists.push(EdgesList::Nil);

        let root_node_id = NodeId(0);
        
        let root_node = Node { 
            state: input.residual.state.clone(), 
            value_top: input.residual.value, 
            value_bot: isize::MIN, 
            best: None, 
            inbound: NIL, 
            rub: input.residual.ub, 
            flags: NodeFlags::new_exact(), 
            depth: input.residual.depth,
        };

        self.nodes.push(root_node);
        self.next_l.insert(input.residual.state.clone(), root_node_id);
        self.edgelists.push(EdgesList::Nil);
    }

    fn _finalize(&mut self, input: &CompilationInput<T>) {
        self._finalize_layers();
        self._find_best_node();
        self._finalize_exact(input);
        self._finalize_cutset(input);
        self._compute_local_bounds(input);
    }


    fn _drain_cutset<F>(&mut self, mut func: F)
    where
        F: FnMut(SubProblem<T>),
    {
        if let Some(best_value) = self.best_value() {
            for id in self.cutset.drain(..) {
                let node = get!(node id, self);

                if node.flags.is_marked() {
                    let rub  = node.value_top.saturating_add(node.rub);
                    let locb = node.value_top.saturating_add(node.value_bot);
                    let ub = rub.min(locb).min(best_value);

                    func(SubProblem {
                        state: node.state.clone(),
                        value: node.value_top,
                        path: Self::_best_path_partial_borrow(
                            id,
                            &self.path_to_root,
                            &self.nodes,
                            &self.edges,
                        ),
                        ub,
                        depth: node.depth,
                    })
                }
            }
        }
    }

    fn _compute_local_bounds(&mut self, input: &CompilationInput<T>) {
        if !self.is_exact && input.comp_type == CompilationType::Relaxed {
            // initialize last layer
            let Layer { from, to } = *get!(layer LayerId(self.layers.len()-1), self);
            for node in &mut self.nodes[from..to] {
                node.value_bot = 0;
                node.flags.set_marked(true);
            }

            // traverse bottom-up
            let lel = self.lel.map(|l| l.0).unwrap_or(usize::MAX);
            for Layer{from, to} in self.layers.iter().skip(lel).rev().copied() {
                for id in from..to {
                    let id = NodeId(id);
                    let node = get!(node id, self);
                    let value = node.value_bot;
                    if node.flags.is_marked() {
                        foreach!(edge of id, self, |edge: Edge| {
                            let using_edge = value.saturating_add(edge.cost);
                            let parent = get!(mut node edge.from, self);
                            parent.flags.set_marked(true);
                            parent.value_bot = parent.value_bot.max(using_edge);
                        });
                    }
                }
            }
        }
    }

    fn _finalize_cutset(&mut self, input: &CompilationInput<T>) {
        if input.comp_type == CompilationType::Relaxed {
            match CUTSET_TYPE {
                LAST_EXACT_LAYER => {
                    if let Some(lel) = self.lel {
                        self._compute_last_exact_layer_cutset(lel);
                    }
                },
                FRONTIER => {
                    if let Some(lel) = self.lel {
                        self._compute_frontier_cutset(lel);
                    }
                },
                _ => {
                    panic!("Only LAST_EXACT_LAYER and FRONTIER are supported so far")
                }
            }
        }
    }

    fn _compute_last_exact_layer_cutset(&mut self, lel: LayerId) {
        let Layer { from, to } = *get!(layer lel, self);
        for (id, node) in self.nodes.iter_mut().enumerate().skip(from).take(to-from) {
            self.cutset.push(NodeId(id));
            node.flags.set_cutset(true);
        }
    }

    fn _compute_frontier_cutset(&mut self, lel: LayerId) {
        // traverse bottom-up
        for Layer{from, to} in self.layers.iter().skip(lel.0).rev().copied() {
            for id in from..to {
                let id = NodeId(id);
                let node = get!(node id, self);
                
                if !node.flags.is_exact() {
                    foreach!(edge of id, self, |edge: Edge| {
                        let parent = get!(mut node edge.from, self);
                        if parent.flags.is_exact() && !parent.flags.is_cutset() {
                            self.cutset.push(edge.from);
                            parent.flags.set_cutset(true);
                        }
                    });
                }
            }
        }
    }

    fn _finalize_layers(&mut self) {
        if !self.next_l.is_empty() {
            if self.layers.is_empty() {
                self.layers.push(Layer { from: 0, to: self.nodes.len() });
            } else {
                let id = LayerId(self.layers.len()-1);
                let layer = get!(layer id, self);
                self.layers.push(Layer { from: layer.to, to: self.nodes.len() });
            }
        }
    }

    fn _find_best_node(&mut self) {
        self.best_node = self
            .next_l
            .values()
            .copied()
            .max_by_key(|id| get!(node id, self).value_top);
    }

    fn _finalize_exact(&mut self, input: &CompilationInput<T>) {
        self.is_exact = self.lel.is_none()
            || (matches!(input.comp_type, CompilationType::Relaxed) && self._has_exact_best_path(self.best_node));
    }

    fn _has_exact_best_path(&self, node: Option<NodeId>) -> bool {
        if let Some(node_id) = node {
            let n = get!(node node_id, self);
            if n.flags.is_exact() {
                true
            } else {
                !n.flags.is_relaxed()
                    && self._has_exact_best_path(n.best.map(|e| get!(edge e, self).from))
            }
        } else {
            true
        }
    }

    fn _move_to_next_layer(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) -> bool {
        self.prev_l.clear();

        for id in curr_l.drain(..) {
            self.prev_l.push(id);
        }
        for (_, id) in self.next_l.drain() {
            curr_l.push(id);
        }

        if curr_l.is_empty() {
            self.layers.push(Layer { from: 0, to: 0 });
            false
        } else {
            self._squash_if_needed(input, curr_l);
            
            if self.layers.is_empty() {
                self.layers.push(Layer { from: 0, to: self.nodes.len() });
            } else {
                let id = LayerId(self.layers.len()-1);
                let layer = get!(layer id, self);
                self.layers.push(Layer { from: layer.to, to: self.nodes.len() });
            }
            true
        }
    }


    fn _branch_on(
        &mut self,
        from_id: NodeId,
        decision: Decision,
        problem: &dyn Problem<State = T>,
    ) {
        let state = get!(node from_id, self).state.as_ref();
        let next_state = Arc::new(problem.transition(state, decision));
        let cost = problem.transition_cost(state, decision);

        match self.next_l.entry(next_state.clone()) {
            Entry::Vacant(e) => {
                let parent = get!(node from_id, self);
                let node_id = NodeId(self.nodes.len());
                self.nodes.push(Node {
                    state: next_state,
                    value_top: parent.value_top.saturating_add(cost),
                    value_bot: isize::MIN,
                    //
                    best: None,
                    inbound: NIL,
                    //
                    rub: isize::MAX,
                    flags: parent.flags,
                    depth: parent.depth + 1,
                });
                append_edge_to!(self, node_id, Edge {
                    from: from_id,
                    to  : node_id,
                    decision,
                    cost,
                });
                e.insert(node_id);
            }
            Entry::Occupied(e) => {
                let node_id = *e.get();
                append_edge_to!(self, node_id, Edge {
                    from: from_id,
                    to  : node_id,
                    decision,
                    cost,
                });
            }
        }
    }


    fn _squash_if_needed(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        match input.comp_type {
            CompilationType::Exact => { /* do nothing: you want to explore the complete DD */ }
            CompilationType::Restricted => {
                if curr_l.len() > input.max_width {
                    self._maybe_save_lel();
                    self._restrict(input, curr_l)
                }
            },
            CompilationType::Relaxed => {
                if curr_l.len() > input.max_width && self.layers.len() > 1 {
                    self._maybe_save_lel();
                    self._relax(input, curr_l)
                }
            },
        }
    }
    fn _maybe_save_lel(&mut self) {
        if self.lel.is_none() {
            self.lel = Some(LayerId(self.layers.len()-1)); // lel was the previous layer
        }
    }

    fn _restrict(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self).value_top
                .cmp(&get!(node b, self).value_top)
                .then_with(|| input.ranking.compare(get!(node a, self).state.as_ref(), get!(node b, self).state.as_ref()))
                .reverse()
        }); // reverse because greater means more likely to be kept

        for drop_id in curr_l.iter().skip(input.max_width).copied() {
            get!(mut node drop_id, self).flags.set_deleted(true);
        }

        curr_l.truncate(input.max_width);
    }

    fn _relax(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self).value_top
                .cmp(&get!(node b, self).value_top)
                .then_with(|| input.ranking.compare(get!(node a, self).state.as_ref(), get!(node b, self).state.as_ref()))
                .reverse()
        }); // reverse because greater means more likely to be kept

        //--
        let (keep, merge) = curr_l.split_at_mut(input.max_width - 1);
        let merged = Arc::new(input.relaxation.merge(&mut merge.iter().map(|id| get!(node id, self).state.as_ref())));

        let recycled = keep.iter().find(|id| get!(node *id, self).state.eq(&merged)).copied();

        let merged_id = recycled.unwrap_or_else(|| {
            let node_id = NodeId(self.nodes.len());
            self.nodes.push(Node {
                state: merged.clone(),
                value_top: isize::MIN,
                value_bot: isize::MIN,
                best: None,    // yet
                inbound: NIL,  // yet
                //
                rub: isize::MAX,
                flags: NodeFlags::new_relaxed(),
                depth: get!(node merge[0], self).depth,
            });
            node_id
        });

        get!(mut node merged_id, self).flags.set_relaxed(true);

        for drop_id in merge {
            get!(mut node drop_id, self).flags.set_deleted(true);

            foreach!(edge of drop_id, self, |edge: Edge| {
                let src   = get!(node edge.from, self).state.as_ref();
                let dst   = get!(node edge.to,   self).state.as_ref();
                let rcost = input.relaxation.relax(src, dst, merged.as_ref(), edge.decision, edge.cost);

                append_edge_to!(self, merged_id, Edge {
                    from: edge.from,
                    to: merged_id,
                    decision: edge.decision,
                    cost: rcost
                });
            });
        }

        if recycled.is_some() {
            curr_l.truncate(input.max_width);
            let saved_id = curr_l[input.max_width - 1];
            self.nodes[saved_id.0].flags.set_deleted(false);
        } else {
            curr_l.truncate(input.max_width - 1);
            curr_l.push(merged_id);
        }
    }
}