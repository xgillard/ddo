//! This is an adaptation of the vector based architecture which implements all
//! the pruning techniques that I have proposed in my PhD thesis (RUB, LocB, EBPO).
//! It also implements the techniques we proposed in 
//! 
//! ``Decision Diagram-Based Branch-and-Bound with Caching
//! for Dominance and Suboptimality Detection''.

use std::{sync::Arc, hash::Hash, collections::{hash_map::Entry, BTreeMap}, fmt::Debug};

use fxhash::FxHashMap;

use crate::{NodeFlags, Decision, CompilationInput, Completion, Reason, CompilationType, Problem, DecisionDiagram, SubProblem, Solution, VizConfig, Variable, DominanceCheckResult};

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
struct Node<State> {
    /// The state associated to this node
    state: Arc<State>,
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
    /// A threshold value to be stored in the cache that conditions the
    /// re-exploration of other nodes with the same state.
    /// 
    /// ### Note
    /// This field is only ever populated after the MDD has been fully unrolled.
    theta: Option<isize>,
    /// A group of flag telling if the node is an exact node, if it is a relaxed
    /// node (helps to determine if the best path is an exact path) and if the
    /// node is reachable in a backwards traversal of the MDD starting at the
    /// terminal node.
    flags: NodeFlags,
    /// The number of decisions that have been made since the problem root
    /// 
    /// ### Note
    /// In this DD version, it is set when the node is expanded because only at
    /// that time do we know to which layer it belongs
    depth: usize,
}

/// Materializes one edge a.k.a arc from the decision diagram. It logically 
/// connects two nodes and annotates the link with a decision and a cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Edge {
    /// The identifier of the node at the ∗∗source∗∗ of this edge.
    from: NodeId,
    /// The identifier of the node at the ∗∗destination∗∗ of this edge.
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
#[derive(Debug, Clone)]
struct Layer {
    nodes: Vec<NodeId>
}

/// The decision diagram in itself. This structure essentially keeps track
/// of the nodes composing the diagram as well as the edges connecting these
/// nodes in two vectors (enabling preallocation and good cache locality). 
/// In addition to that, it also keeps track of the path (root_pa) from the
/// problem root to the root of this decision diagram (explores a sub problem). 
/// The prev_l comprises information about the nodes that are currently being
/// expanded, next_l stores the information about the nodes from the next layer 
/// and cut-set stores an exact cut-set of the DD.
/// Depending on the type of DD compiled, different cut-set types will be used:
/// - Exact: no cut-set is needed since the DD is exact
/// - Restricted: the last exact layer is used as cut-set
/// - Relaxed: either the last exact layer of the frontier cut-set can be chosen
///            within the CompilationInput
#[derive(Debug, Clone)]
pub struct Pooled<State>
where
    State: Eq + PartialEq + Hash + Clone,
{
    /// This map stores the information about the structure of all the layers
    /// in this decision diagram
    layers: BTreeMap<usize, Layer>,
    /// All the nodes composing this decision diagram. The vector comprises 
    /// nodes from all layers in the DD. A nice property is that all nodes
    /// belonging to one same layer form a sequence in the ‘nodes‘ vector.
    nodes: Vec<Node<State>>,
    /// This vector stores the information about all edges connecting the nodes 
    /// of the decision diagram.
    edges: Vec<Edge>,
    /// This vector stores the information about all edge lists constituting 
    /// linked lists between edges
    edgelists: Vec<EdgesList>,
    
    /// The id of the layer that must be expanded next
    curr_l: LayerId,
    /// The nodes that have not been expanded yet
    /// Note: pool in itself is indexed on the state associated with nodes.
    /// The rationale being that two transitions to the same state in the same
    /// layer should lead to the same node. This indexation helps ensuring 
    /// the uniqueness constraint in amortized O(1).
    pool: FxHashMap<Arc<State>, NodeId>,

    /// Keeps track of the decisions that have been taken to reach the root
    /// of this DD, starting from the problem root.
    path_to_root: Vec<Decision>,
    /// The cutset of the decision diagram (only maintained for relaxed dd)
    cutset: Vec<NodeId>,
    /// The identifier of the best terminal node of the diagram (None when the
    /// problem compiled into this dd is infeasible)
    best_node: Option<NodeId>,
    /// The identifier of the best exact terminal node of the diagram (None when
    /// no terminal node is exact)
    best_exact_node: Option<NodeId>,
    /// A flag set to true when no layer of the decision diagram has been
    /// restricted or relaxed
    is_exact: bool,
    /// A flag set to true when the longest r-t path of this decision diagram
    /// traverses no merged node (Exact Best Path Optimization aka EBPO).
    has_exact_best_path: bool,
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
    ($dd:expr, $edge:expr) => {
        let new_eid = EdgeId($dd.edges.len());
        let lst_id  = EdgesListId($dd.edgelists.len());
        $dd.edges.push($edge);
        $dd.edgelists.push(EdgesList::Cons { head: new_eid, tail: get!(node $edge.to, $dd).inbound });
        
        let parent = get!(node $edge.from, $dd);
        let parent_exact = parent.flags.is_exact();
        let value = parent.value_top.saturating_add($edge.cost);
        
        let node = get!(mut node $edge.to, $dd);
        let exact = parent_exact & node.flags.is_exact();
        node.flags.set_exact(exact);
        node.inbound = lst_id;

        if value >= node.value_top {
            node.best = Some(new_eid);
            node.value_top = value;
        }
    };
}

impl<State> Default for Pooled<State>
where
    State: Eq + PartialEq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<State> DecisionDiagram<State> for Pooled<State>
where
    State: Eq + PartialEq + Hash + Clone,
{
    fn compile(&mut self, input: &CompilationInput<State>) -> Result<Completion, Reason> {
        self._compile(input)
    }

    fn is_exact(&self) -> bool {
        self.is_exact || self.has_exact_best_path
    }

    fn best_value(&self) -> Option<isize> {
        self._best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self._best_solution()
    }

    fn best_exact_value(&self) -> Option<isize> {
        self._best_exact_value()
    }

    fn best_exact_solution(&self) -> Option<Solution> {
        self._best_exact_solution()
    }

    fn drain_cutset<F>(&mut self, func: F)
    where
        F: FnMut(SubProblem<State>) {
        self._drain_cutset(func)
    }
}

impl<T> Pooled<T>
where
    T: Eq + PartialEq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            layers: Default::default(),
            nodes: vec![],
            edges: vec![],
            edgelists: vec![],
            //
            curr_l: LayerId(0),
            pool: Default::default(),
            //
            path_to_root: vec![],
            cutset: vec![],
            best_node: None,
            best_exact_node: None,
            is_exact: true,
            has_exact_best_path: false,
        }
    }
    
    fn _clear(&mut self) {
        self.layers.clear();
        self.nodes.clear();
        self.edges.clear();
        self.edgelists.clear();
        self.pool.clear();
        self.path_to_root.clear();
        self.cutset.clear();
        self.best_node = None;
        self.best_exact_node = None;
        self.is_exact = true;
        self.has_exact_best_path = false;
    }

    fn _best_value(&self) -> Option<isize> {
        self.best_node.map(|id| get!(node id, self).value_top)
    }

    fn _best_solution(&self) -> Option<Vec<Decision>> {
        self.best_node.map(|id| self._best_path(id))
    }

    fn _best_exact_value(&self) -> Option<isize> {
        self.best_exact_node.map(|id| get!(node id, self).value_top)
    }

    fn _best_exact_solution(&self) -> Option<Vec<Decision>> {
        self.best_exact_node.map(|id| self._best_path(id))
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

        while let Some(var) = input.problem.next_variable(self.curr_l.0, &mut self.pool.keys().map(|s| s.as_ref())) {
            // Did the cutoff kick in ?
            if input.cutoff.must_stop() {
                return Err(Reason::CutoffOccurred);
            }

            if self.pool.is_empty() {
                break;
            }

            let curr_l = self._move_to_next_layer(input, var);

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

            self.curr_l.0 += 1;
        }

        self._finalize(input);

        Ok(Completion { 
            is_exact: self.is_exact(), 
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
            rub: isize::MAX, 
            theta: None,
            flags: NodeFlags::new_exact(), 
            depth: input.residual.depth,
        };

        self.nodes.push(root_node);
        self.pool.insert(input.residual.state.clone(), root_node_id);
        self.edgelists.push(EdgesList::Nil);
        self.curr_l = LayerId(input.residual.depth);
    }

    fn _finalize(&mut self, input: &CompilationInput<T>) {
        self._finalize_layers();
        self._find_best_node();
        self._finalize_exact(input);
        self._compute_frontier_cutset(input);
        self._compute_local_bounds(input);
        self._compute_thresholds(input);
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
    #[allow(clippy::redundant_closure_call)]
    fn _compute_local_bounds(&mut self, input: &CompilationInput<T>) {
        if !self.cutset.is_empty() && input.comp_type == CompilationType::Relaxed {
            // initialize last layer
            let (_, Layer { nodes }) = self.layers.last_key_value().unwrap();
            for id in nodes.iter() {
                let node = get!(mut node id, self);
                node.value_bot = 0;
                node.flags.set_marked(true);
            }

            // traverse bottom-up
            // note: cache requires that all nodes have an associated locb. not only those below cutset
            for Layer { nodes } in self.layers.values().rev() {
                for id in nodes.iter() {
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

    #[allow(clippy::redundant_closure_call)]
    fn _compute_thresholds(&mut self, input: &CompilationInput<T>) {
        if input.comp_type == CompilationType::Relaxed || self.is_exact {
            let mut best_known = input.best_lb;

            if let Some(best_exact_node) = self.best_exact_node {
                let best_exact_value = get!(mut node best_exact_node, self).value_top;
                best_known = best_known.max(best_exact_value);

                for id in self.pool.values() {
                    if self.nodes[id.0].flags.is_exact() {
                        self.nodes[id.0].theta = Some(best_known);
                    }
                }
            }


            for Layer { nodes } in self.layers.values().rev() {
                for id in nodes.iter() {
                    let node = get!(mut node id, self);

                    if node.flags.is_deleted() {
                        continue;
                    }

                    // ATTENTION: YOU WANT TO PROPAGATE THETA EVEN IF THE NODE WAS PRUNED BY THE CACHE
                    if !node.flags.is_pruned_by_cache() {
                        let tot_rub = node.value_top.saturating_add(node.rub);
                        if tot_rub <= best_known {
                            node.theta = Some(best_known.saturating_sub(node.rub));
                        } else if node.flags.is_cutset() {
                            let tot_locb = node.value_top.saturating_add(node.value_bot);
                            if tot_locb <= best_known {
                                let theta = node.theta.unwrap_or(isize::MAX);
                                node.theta = Some(theta.min(best_known.saturating_sub(node.value_bot)));
                            } else {
                                node.theta = Some(node.value_top);
                            }
                        } else if node.flags.is_exact() && node.theta.is_none() { // large theta for dangling nodes
                            node.theta = Some(isize::MAX);
                        }

                        Self::_maybe_update_cache(node, input);
                    }
                    // only propagate if you have an actual threshold
                    if let Some(my_theta) = node.theta {
                        foreach!(edge of id, self, |edge: Edge| {
                            let parent = get!(mut node edge.from, self);
                            let theta  = parent.theta.unwrap_or(isize::MAX); 
                            parent.theta = Some(theta.min(my_theta.saturating_sub(edge.cost)));
                        });
                    }
                }
            }
        }
    }

    fn _maybe_update_cache(node: &Node<T>, input: &CompilationInput<T>) {
        // A node can only be added to the cache if it belongs to the cutset or is above it
        if let Some(theta) = node.theta {
            if node.flags.is_above_cutset() {
                input.cache.update_threshold(
                    node.state.clone(), 
                    node.depth, 
                    theta, 
                    !node.flags.is_cutset()) // if it is in the cutset it has not been explored !
            }
        }
    }
    
    #[allow(clippy::redundant_closure_call)]
    fn _compute_frontier_cutset(&mut self, input: &CompilationInput<T>) {
        if input.comp_type == CompilationType::Relaxed || self.is_exact {
            // traverse bottom-up
            for Layer { nodes } in self.layers.values().rev() {
                for id in nodes.iter() {
                    let node = get!(mut node id, self);
                    
                    if node.flags.is_exact() {
                        node.flags.set_above_cutset(true);
                    } else {
                        foreach!(edge of id, self, |edge: Edge| {
                            let parent = get!(mut node edge.from, self);
                            if parent.flags.is_exact() && !parent.flags.is_cutset() {
                                if !self.is_exact {
                                    self.cutset.push(edge.from);
                                }
                                parent.flags.set_cutset(true);
                            }
                        });
                    }
                }
            }
        }
    }

    fn _finalize_layers(&mut self) {
        let mut last_l = vec![];
        self.pool.values().copied().for_each(|id| {
            let node = get!(mut node id, self);
            last_l.push(id);
            node.depth = self.curr_l.0;
        });
        self.layers.insert(self.curr_l.0, Layer { nodes: last_l });
    }

    fn _find_best_node(&mut self) {
        self.best_node = self
            .pool
            .values()
            .copied()
            .max_by_key(|id| get!(node id, self).value_top);
        self.best_exact_node = self
            .pool
            .values()
            .filter(|id| get!(node id, self).flags.is_exact())
            .copied()
            .max_by_key(|id| get!(node id, self).value_top);
    }

    fn _finalize_exact(&mut self, input: &CompilationInput<T>) {
        self.has_exact_best_path = matches!(input.comp_type, CompilationType::Relaxed) && self._has_exact_best_path(self.best_node);

        if self.has_exact_best_path {
            self.best_exact_node = self.best_node;
        }
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

    fn _move_to_next_layer(&mut self, input: &CompilationInput<T>, var: Variable) -> Vec<NodeId> {
        let mut curr_l: Vec<NodeId> = self.pool.values().copied().collect();
        let mut to_remove = vec![];
        curr_l.retain(|node_id| {
            let node = get!(mut node node_id, self);
            let state = node.state.as_ref();
            if input.problem.is_impacted_by(var, state) {
                node.depth = self.curr_l.0;
                to_remove.push(node.state.clone());
                true
            } else {
                false
            }
        });

        to_remove.drain(..).for_each(|s| { self.pool.remove(s.as_ref()); });
        
        let mut to_expand = curr_l.clone(); // need to preserve layer to remember nodes pruned by cache
        if !self.layers.is_empty() {
            self._filter_with_cache(input, &mut to_expand);
        }
        self._filter_with_dominance(input, &mut to_expand);

        let len = self.nodes.len(); // but need to add the potential merged node
        self._squash_if_needed(input, &mut to_expand);
        if self.nodes.len() > len {
            curr_l.push(NodeId(len));
        }

        if !curr_l.is_empty() {
            self.layers.insert(self.curr_l.0, Layer { nodes: curr_l });
        }

        to_expand
    }

    fn _filter_with_dominance(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        curr_l.sort_unstable_by(|a,b| input.dominance.cmp(get!(node a, self).state.as_ref(), get!(node a, self).value_top, get!(node b, self).state.as_ref(), get!(node b, self).value_top).reverse());
        curr_l.retain(|id| {
            let node = get!(mut node id, self);
            if node.flags.is_exact() {
                let DominanceCheckResult { dominated, threshold } = input.dominance.is_dominated_or_insert(node.state.clone(), node.depth, node.value_top);
                if dominated {
                    node.theta = threshold; // set theta for later propagation
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
    }

    fn _filter_with_cache(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        curr_l.retain(|id| {
            let node = get!(mut node id, self);
            let threshold = input.cache.get_threshold(node.state.as_ref(), node.depth);
            if let Some(threshold) = threshold {
                if node.value_top > threshold.value {
                    true
                } else {
                    node.flags.set_pruned_by_cache(true);
                    node.theta = Some(threshold.value); // set theta for later propagation
                    false
                }
            } else {
                true
            }
        });
    }

    fn _branch_on(
        &mut self,
        from_id: NodeId,
        decision: Decision,
        problem: &dyn Problem<State = T>,
    ) {
        let state = get!(node from_id, self).state.as_ref();
        let next_state = Arc::new(problem.transition(state, decision));
        let cost = problem.transition_cost(state, next_state.as_ref(), decision);

        match self.pool.entry(next_state.clone()) {
            Entry::Vacant(e) => {
                let parent = get!(node from_id, self);
                let node_id = NodeId(self.nodes.len());
                let mut flags = NodeFlags::new_exact();
                flags.set_exact(parent.flags.is_exact());

                self.nodes.push(Node {
                    state: next_state,
                    value_top: parent.value_top.saturating_add(cost),
                    value_bot: isize::MIN,
                    //
                    best: None,
                    inbound: NIL,
                    //
                    rub: isize::MAX,
                    theta: None,
                    flags,
                    depth: parent.depth + 1, // value will be updated when expanded
                });
                append_edge_to!(self, Edge {
                    from: from_id,
                    to  : node_id,
                    decision,
                    cost,
                });
                e.insert(node_id);
            }
            Entry::Occupied(e) => {
                let node_id = *e.get();
                append_edge_to!(self, Edge {
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
                    self._restrict(input, curr_l)
                }
            },
            CompilationType::Relaxed => {
                if curr_l.len() > input.max_width && self.layers.len() >= 2 {
                    self._relax(input, curr_l)
                }
            },
        }
    }

    fn _restrict(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        self.is_exact = false;
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

    #[allow(clippy::redundant_closure_call)]
    fn _relax(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        self.is_exact = false;
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
                theta: None,
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

                append_edge_to!(self, Edge {
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

impl <T> Pooled<T> 
where T: Debug + Eq + PartialEq + Hash + Clone {

    /// This is the method you will want to use in order to create the output image you would like.
    /// Note: the output is going to be a string of (not compiled) 'dot'. This makes it easier for
    /// me to code and gives you the freedom to fiddle with the graph if needed.
    pub fn as_graphviz(&self, config: &VizConfig) -> String {
        let mut out = String::new();

        out.push_str("digraph {\n\tranksep = 3;\n\n");

        // Show all nodes
        for (id, _) in self.nodes.iter().enumerate() {
            let node = get!(node NodeId(id), self);
            if !config.show_deleted && node.flags.is_deleted() {
                continue;
            }
            out.push_str(&self.node(id, config));
            out.push_str(&self.edges_of(id));
        }

        // Show clusters if requested
        if config.show_deleted && config.group_merged {
            let mut merged_by_layer = BTreeMap::default();
            for id in 0..self.nodes.len() {
                let id = NodeId(id);
                let node = get!(node id, self);
                if node.flags.is_deleted() || node.flags.is_relaxed() {
                    merged_by_layer.entry(node.depth)
                        .or_insert(vec![])
                        .push(format!("{}", id.0));
                }
            }
            for (i, merged) in merged_by_layer.iter() {
                if !merged.is_empty() {
                    out.push_str(&format!("\tsubgraph cluster_{i} "));
                    out.push_str("{\n");
                    out.push_str("\t\tstyle=filled;\n");
                    out.push_str("\t\tcolor=purple;\n");
                    out.push_str(&format!("\t\t{}\n", merged.join(";")));
                    out.push_str("\t};\n");
                }
            }
        }

        // Finish the graph with a terminal node
        out.push_str(&self.add_terminal_node());

        out.push_str("}\n");
        out
    }

    /// Creates a string representation of one single node
    fn node(&self, id: usize, config: &VizConfig) -> String {
        let attributes = self.node_attributes(id, config);
        format!("\t{id} [{attributes}];\n")
    }
    
    #[allow(clippy::redundant_closure_call)]
    /// Creates a string representation of the edges incident to one node
    fn edges_of(&self, id: usize) -> String {
        let mut out = String::new();
        foreach!(edge of NodeId(id), self, |edge: Edge| {
            let Edge{from, to, decision, cost} = edge;
            let best = get!(node NodeId(id), self).best;
            let best = best.map(|eid| *get!(edge eid, self));
            out.push_str(&Self::edge(from.0, to.0, decision, cost, Some(edge) == best));
        });
        out
    }
    /// Adds a terminal node (if the DD is feasible) and draws the edges entering that node from
    /// all the nodes of the terminal layer.
    fn add_terminal_node(&self) -> String {
        let mut out = String::new();
        let (_, Layer { nodes }) = self.layers.last_key_value().unwrap();
        if !nodes.is_empty() {
            let terminal = "\tterminal [shape=\"circle\", label=\"\", style=\"filled\", color=\"black\", group=\"terminal\"];\n";
            out.push_str(terminal);

            let vmax = nodes.iter().map(|id| self.nodes[id.0].value_top).max().unwrap_or(isize::MAX);
            for id in nodes.iter() {
                let value = self.nodes[id.0].value_top;
                if value == vmax {
                    out.push_str(&format!("\t{} -> terminal [penwidth=3];\n", id.0));
                } else {
                    out.push_str(&format!("\t{} -> terminal;\n", id.0));
                }
            }
        }
        out
    }
    /// Creates a string representation of one edge
    fn edge(from: usize, to: usize, decision: Decision, cost: isize, is_best: bool) -> String {
        let width = if is_best { 3 } else { 1 };
        let variable = decision.variable.0;
        let value = decision.value;
        let label = format!("(x{variable} = {value})\\ncost = {cost}");

        format!("\t{from} -> {to} [penwidth={width},label=\"{label}\"];\n")
    }
    /// Creates the list of attributes that are used to configure one node
    fn node_attributes(&self, id: usize, config: &VizConfig) -> String {
        let node = &self.nodes[id];
        let merged = node.flags.is_relaxed();
        let state = node.state.as_ref();
        let restricted= node.flags.is_deleted();

        let shape = Self::node_shape(merged, restricted);
        let color = Self::node_color(node, merged);
        let peripheries = Self::node_peripheries(node);
        let group = self.node_group(node);
        let label = Self::node_label(node, state, config);

        format!("shape={shape},style=filled,color={color},peripheries={peripheries},group=\"{group}\",label=\"{label}\"")
    }
    /// Determines the group of a node based on the last branching decision leading to it
    fn node_group(&self, node: &Node<T>) -> String {
        if let Some(eid) = node.best {
            let edge = self.edges[eid.0];
            format!("{}", edge.decision.variable.0)
        } else {
            "root".to_string()
        }
    }
    /// Determines the shape to use when displaying a node
    fn node_shape(merged: bool, restricted: bool) -> &'static str {
        if merged || restricted {
            "square"
        } else {
            "circle"
        }
    }
    /// Determines the number of peripheries to draw when displaying a node.
    fn node_peripheries(node: &Node<T>) -> usize {
        if node.flags.is_cutset() {
            4
        } else {
            1
        }
    }
    /// Determines the color of peripheries to draw when displaying a node.
    fn node_color(node: &Node<T>, merged: bool) -> &str {
        if node.flags.is_cutset() {
            "red"
        } else if node.flags.is_exact() {
            "\"#99ccff\""
        } else if merged {
            "yellow"
        } else {
            "lightgray"
        }
    }
    /// Creates text label to place inside of the node when displaying it
    fn node_label(node: &Node<T>, state: &T, config: &VizConfig) -> String {
        let mut out = format!("{state:?}");

        if config.show_value {
            out.push_str(&format!("\\nval: {}", node.value_top));
        }
        if config.show_locb {
        out.push_str(&format!("\\nlocb: {}", Self::extreme(node.value_bot)));
        }
        if config.show_rub {
            out.push_str(&format!("\\nrub: {}", Self::extreme(node.rub)));
        }
        if config.show_threshold {
            out.push_str(&format!("\\ntheta: {}", Self::extreme(node.theta.unwrap_or(isize::MAX))));
        }

        out
    }
    /// An utility method to replace extreme values with +inf and -inf
    fn extreme(x: isize) -> String {
        match x {
            isize::MAX => "+inf".to_string(),
            isize::MIN => "-inf".to_string(),
            _ => format!("{x}")
        }
    }
}

// ############################################################################
// #### TESTS #################################################################
// ############################################################################


#[cfg(test)]
mod test_default_mdd {
    use std::cmp::Ordering;
    use std::sync::Arc;

    use fxhash::FxHashMap;

    use crate::{Variable, DecisionDiagram, SubProblem, CompilationInput, Problem, Decision, Relaxation, StateRanking, NoCutoff, CompilationType, Cutoff, Reason, DecisionCallback, EmptyCache, SimpleCache, Cache, Pooled, VizConfigBuilder, Threshold, EmptyDominanceChecker};

    type DefaultMDD<State>    = Pooled<State>;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let mdd = Pooled::<usize>::new();

        assert!(mdd.is_exact());
    }

    #[test]
    fn root_remembers_the_pa_from_the_fringe_node() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let mut input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  3,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 1, value: 42}), 
                value: 42, 
                path:  vec![Decision{variable: Variable(0), value: 42}], 
                ub:    isize::MAX,
                depth: 1,
            },
            cache: &cache,
            dominance: &dominance,
        };

        let mut mdd = DefaultMDD::new();
        assert!(mdd.compile(&input).is_ok());
        assert_eq!(mdd.path_to_root, vec![Decision{variable: Variable(0), value: 42}]);

        input.comp_type = CompilationType::Relaxed;
        assert!(mdd.compile(&input).is_ok());
        assert_eq!(mdd.path_to_root, vec![Decision{variable: Variable(0), value: 42}]);

        input.comp_type = CompilationType::Restricted;
        assert!(mdd.compile(&input).is_ok());
        assert_eq!(mdd.path_to_root, vec![Decision{variable: Variable(0), value: 42}]);
    }
    
    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();

        assert!(mdd.compile(&input).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), Some(6));
        assert_eq!(mdd.best_solution().unwrap(),
                   vec![
                       Decision{variable: Variable(2), value: 2},
                       Decision{variable: Variable(1), value: 2},
                       Decision{variable: Variable(0), value: 2},
                   ]
        );
    }

    #[test]
    fn restricted_drops_the_less_interesting_nodes() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();

        assert!(mdd.compile(&input).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value().unwrap(), 6);
        assert_eq!(mdd.best_solution().unwrap(),
                   vec![
                       Decision{variable: Variable(2), value: 2},
                       Decision{variable: Variable(1), value: 2},
                       Decision{variable: Variable(0), value: 2},
                   ]
        );
    }

    #[test]
    fn exact_no_cutoff_completion_must_be_coherent_with_outcome() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);

        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, mdd.best_value());
    }
    #[test]
    fn restricted_no_cutoff_completion_must_be_coherent_with_outcome_() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, mdd.best_value());
    }
    #[test]
    fn relaxed_no_cutoff_completion_must_be_coherent_with_outcome() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, mdd.best_value());
    }
    
    #[derive(Debug, Clone, Copy)]
    struct CutoffAlways;
    impl Cutoff for CutoffAlways {
        fn must_stop(&self) -> bool { true }
    }
    #[test]
    fn exact_fails_with_cutoff_when_cutoff_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &CutoffAlways,
            max_width:  1,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }

    #[test]
    fn restricted_fails_with_cutoff_when_cutoff_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &CutoffAlways,
            max_width:  1,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    #[test]
    fn relaxed_fails_with_cutoff_when_cutoff_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &CutoffAlways,
            max_width:  1,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }

    #[test]
    fn relaxed_merges_the_less_interesting_nodes() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);

        assert!(result.is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value().unwrap(), 24);
        assert_eq!(mdd.best_solution().unwrap(),
                   vec![
                       Decision{variable: Variable(2), value: 2},
                       Decision{variable: Variable(1), value: 0}, // that's a relaxed edge
                       Decision{variable: Variable(0), value: 2},
                   ]
        );
    }

    #[test]
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        let mut cutset = vec![];
        mdd.drain_cutset(|n| cutset.push(n));
        assert_eq!(cutset.len(), 3); // L1 was not squashed even though it was 3 wide
    }

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  10,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occurred() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(!mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  10,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occurred() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  1,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(!mdd.is_exact())
    }
    #[test]
    fn when_the_problem_is_infeasible_there_is_no_solution() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyInfeasibleProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn when_the_problem_is_infeasible_there_is_no_best_value() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyInfeasibleProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_value().is_none())
    }
    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    1000,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    1000,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    1000,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn exact_skips_nodes_with_a_value_less_than_known_threshold() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 0}), 1, 0, true);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 1}), 1, 1, true);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 2}), 1, 2, true);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn relaxed_skips_nodes_with_a_value_less_than_known_threshold() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 0}), 1, 0, true);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 1}), 1, 1, true);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 2}), 1, 2, true);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn restricted_skips_nodes_with_a_value_less_than_known_threshold() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 0}), 1, 0, true);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 1}), 1, 1, true);
        cache.update_threshold(Arc::new(DummyState{depth: 1, value: 2}), 1, 2, true);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  usize::MAX,
            best_lb:    isize::MIN,
            residual: &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_mdd_computes_thresholds_when_exact() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  10,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(mdd.is_exact());

        let expected = vec![
            (DummyState{depth: 0, value: 0}, Some(Threshold {value: 0, explored: true})),
            (DummyState{depth: 1, value: 0}, Some(Threshold {value: 2, explored: true})),
            (DummyState{depth: 1, value: 1}, Some(Threshold {value: 2, explored: true})),
            (DummyState{depth: 1, value: 2}, Some(Threshold {value: 2, explored: true})),
            (DummyState{depth: 2, value: 0}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 1}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 2}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 3}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 4}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 3, value: 0}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 1}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 2}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 3}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 4}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 5}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 6}, Some(Threshold {value: 6, explored: true})),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    #[test]
    fn relaxed_mdd_computes_thresholds_when_exact() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  10,
            best_lb:    isize::MIN,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(mdd.is_exact());

        let expected = vec![
            (DummyState{depth: 0, value: 0}, Some(Threshold {value: 0, explored: true})),
            (DummyState{depth: 1, value: 0}, Some(Threshold {value: 2, explored: true})),
            (DummyState{depth: 1, value: 1}, Some(Threshold {value: 2, explored: true})),
            (DummyState{depth: 1, value: 2}, Some(Threshold {value: 2, explored: true})),
            (DummyState{depth: 2, value: 0}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 1}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 2}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 3}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 2, value: 4}, Some(Threshold {value: 4, explored: true})),
            (DummyState{depth: 3, value: 0}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 1}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 2}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 3}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 4}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 5}, Some(Threshold {value: 6, explored: true})),
            (DummyState{depth: 3, value: 6}, Some(Threshold {value: 6, explored: true})),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    #[test]
    fn restricted_mdd_computes_thresholds_when_all_pruned() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  10,
            best_lb:    15,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(mdd.is_exact());

        let expected = vec![
            (DummyState{depth: 0, value: 0}, Some(Threshold {value: 1, explored: true})),
            (DummyState{depth: 1, value: 0}, Some(Threshold {value: 3, explored: true})),
            (DummyState{depth: 1, value: 1}, Some(Threshold {value: 3, explored: true})),
            (DummyState{depth: 1, value: 2}, Some(Threshold {value: 3, explored: true})),
            (DummyState{depth: 2, value: 0}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 1}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 2}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 3}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 4}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 3, value: 0}, None),
            (DummyState{depth: 3, value: 1}, None),
            (DummyState{depth: 3, value: 2}, None),
            (DummyState{depth: 3, value: 3}, None),
            (DummyState{depth: 3, value: 4}, None),
            (DummyState{depth: 3, value: 5}, None),
            (DummyState{depth: 3, value: 6}, None),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    #[test]
    fn relaxed_mdd_computes_thresholds_when_all_pruned() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &DummyProblem,
            relaxation: &DummyRelax,
            ranking:    &DummyRanking,
            cutoff:     &NoCutoff,
            max_width:  10,
            best_lb:    15,
            residual:  &SubProblem { 
                state: Arc::new(DummyState{depth: 0, value: 0}), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        
        assert!(mdd.is_exact());

        let expected = vec![
            (DummyState{depth: 0, value: 0}, Some(Threshold {value: 1, explored: true})),
            (DummyState{depth: 1, value: 0}, Some(Threshold {value: 3, explored: true})),
            (DummyState{depth: 1, value: 1}, Some(Threshold {value: 3, explored: true})),
            (DummyState{depth: 1, value: 2}, Some(Threshold {value: 3, explored: true})),
            (DummyState{depth: 2, value: 0}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 1}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 2}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 3}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 2, value: 4}, Some(Threshold {value: 5, explored: true})),
            (DummyState{depth: 3, value: 0}, None),
            (DummyState{depth: 3, value: 1}, None),
            (DummyState{depth: 3, value: 2}, None),
            (DummyState{depth: 3, value: 3}, None),
            (DummyState{depth: 3, value: 4}, None),
            (DummyState{depth: 3, value: 5}, None),
            (DummyState{depth: 3, value: 6}, None),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    /// The example problem and relaxation for the local bounds should generate
    /// the following relaxed MDD in which the layer 'a','b' is the LEL.
    ///
    /// ```plain
    ///                      r
    ///                   /     \
    ///                10        7
    ///               /           |
    ///             a              b
    ///             |     +--------+-------+
    ///             |     |        |       |
    ///             2     3        6       5
    ///              \   /         |       |
    ///                M           e       f
    ///                |           |     /   \
    ///                4           0   1      2
    ///                |           |  /        \
    ///                g            h           i
    ///                |            |           |
    ///                0            0           0
    ///                +------------+-----------+
    ///                             t
    /// ```
    ///
    #[derive(Copy, Clone)]
    struct LocBoundsAndThresholdsExamplePb;
    impl Problem for LocBoundsAndThresholdsExamplePb {
        type State = char;
        fn nb_variables (&self) -> usize {  4  }
        fn initial_state(&self) -> char  { 'r' }
        fn initial_value(&self) -> isize {  0  }
        fn next_variable(&self, _: usize, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
            match next_layer.next().copied().unwrap_or('z') {
                'r' => Some(Variable(0)),
                'a' => Some(Variable(1)),
                'b' => Some(Variable(1)),
                // c, d are merged into M
                'c' => Some(Variable(2)),
                'd' => Some(Variable(2)),
                'M' => Some(Variable(2)),
                'e' => Some(Variable(2)),
                'f' => Some(Variable(2)),
                'g' => Some(Variable(0)),
                'h' => Some(Variable(0)),
                'i' => Some(Variable(0)),
                _   => None,
            }
        }
        fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback) {
            /* do nothing, just consider that all domains are empty */
            (match *state {
                'r' => vec![10, 7],
                'a' => vec![2],
                'b' => vec![3, 6, 5],
                // c, d are merged into M
                'M' => vec![4],
                'e' => vec![0],
                'f' => vec![1, 2],
                'g' => vec![0],
                'h' => vec![0],
                'i' => vec![0],
                _   => vec![],
            })
            .iter()
            .copied()
            .for_each(&mut |value| f.apply(Decision{variable, value}))
        }

        fn transition(&self, state: &char, d: Decision) -> char {
            match (*state, d.value) {
                ('r', 10) => 'a',
                ('r',  7) => 'b',
                ('a',  2) => 'c', // merged into M
                ('b',  3) => 'd', // merged into M
                ('b',  6) => 'e',
                ('b',  5) => 'f',
                ('M',  4) => 'g',
                ('e',  0) => 'h',
                ('f',  1) => 'h',
                ('f',  2) => 'i',
                _         => 't'
            }
        }

        fn transition_cost(&self, _: &char, _: &Self::State, d: Decision) -> isize {
            d.value
        }
    }

    #[derive(Copy, Clone)]
    struct LocBoundsAndThresholdsExampleRelax;
    impl Relaxation for LocBoundsAndThresholdsExampleRelax {
        type State = char;
        fn merge(&self, _: &mut dyn Iterator<Item=&char>) -> char {
            'M'
        }

        fn relax(&self, _: &char, _: &char, _: &char, _: Decision, cost: isize) -> isize {
            cost
        }

        fn fast_upper_bound(&self, state: &char) -> isize {
            match *state {
                'r' => 30,
                'a' => 20,
                'b' => 20,
                // c, d are merged into M
                'M' => 10,
                'e' => 10,
                'f' => 10,
                'g' => 0,
                'h' => 0,
                'i' => 0,
                _   => 0,
            }
        }
    }

    #[derive(Clone, Copy)]
    struct CmpChar;
    impl StateRanking for CmpChar {
        type State = char;
        fn compare(&self, a: &char, b: &char) -> Ordering {
            a.cmp(b)
        }
    }

    #[test]
    fn relaxed_computes_local_bounds_and_thresholds() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking:    &CmpChar,
            cutoff:     &NoCutoff,
            max_width:  3,
            best_lb:    0,
            residual: &SubProblem { 
                state: Arc::new('r'), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(!mdd.is_exact());
        assert_eq!(Some(16), mdd.best_value());

        let mut v = FxHashMap::<char, isize>::default();
        mdd.drain_cutset(|n| {v.insert(*n.state, n.ub);});

        assert_eq!(16, v[&'a']);
        assert_eq!(14, v[&'b']);
        assert_eq!(13, v[&'h']);
        assert_eq!(14, v[&'i']);
        assert_eq!(4, v.len());

        assert!(cache.get_threshold(&'r', 0).is_some());
        assert!(cache.get_threshold(&'a', 1).is_some());
        assert!(cache.get_threshold(&'b', 1).is_some());
        assert!(cache.get_threshold(&'M', 2).is_none());
        assert!(cache.get_threshold(&'e', 2).is_some());
        assert!(cache.get_threshold(&'f', 2).is_some());
        assert!(cache.get_threshold(&'g', 3).is_none());
        assert!(cache.get_threshold(&'h', 3).is_some());
        assert!(cache.get_threshold(&'i', 3).is_some());
        assert!(cache.get_threshold(&'t', 4).is_none());

        let mut threshold = cache.get_threshold(&'r', 0).unwrap();
        assert_eq!(0, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'a', 1).unwrap();
        assert_eq!(10, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'b', 1).unwrap();
        assert_eq!(7, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'e', 2).unwrap();
        assert_eq!(13, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'f', 2).unwrap();
        assert_eq!(12, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'h', 3).unwrap();
        assert_eq!(13, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'i', 3).unwrap();
        assert_eq!(14, threshold.value);
        assert!(!threshold.explored);
    }

    #[test]
    fn relaxed_computes_local_bounds_and_thresholds_with_pruning() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem:    &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking:    &CmpChar,
            cutoff:     &NoCutoff,
            max_width:  3,
            best_lb:    15,
            residual: &SubProblem { 
                state: Arc::new('r'), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(!mdd.is_exact());
        assert_eq!(Some(16), mdd.best_value());

        let mut v = FxHashMap::<char, isize>::default();
        mdd.drain_cutset(|n| {v.insert(*n.state, n.ub);});

        assert_eq!(16, v[&'a']);
        assert_eq!(14, v[&'b']);
        assert_eq!(2, v.len());

        assert!(cache.get_threshold(&'r', 0).is_some());
        assert!(cache.get_threshold(&'a', 1).is_some());
        assert!(cache.get_threshold(&'b', 1).is_some());
        assert!(cache.get_threshold(&'M', 2).is_none());
        assert!(cache.get_threshold(&'e', 2).is_some());
        assert!(cache.get_threshold(&'f', 2).is_some());
        assert!(cache.get_threshold(&'g', 3).is_none());
        assert!(cache.get_threshold(&'h', 3).is_some());
        assert!(cache.get_threshold(&'i', 3).is_some());
        assert!(cache.get_threshold(&'t', 4).is_none());

        let mut threshold = cache.get_threshold(&'r', 0).unwrap();
        assert_eq!(0, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'a', 1).unwrap();
        assert_eq!(10, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'b', 1).unwrap();
        assert_eq!(8, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'e', 2).unwrap();
        assert_eq!(15, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'f', 2).unwrap();
        assert_eq!(13, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'h', 3).unwrap();
        assert_eq!(15, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'i', 3).unwrap();
        assert_eq!(15, threshold.value);
        assert!(threshold.explored);
    }

    #[test]
    fn test_default_visualisation() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type:  crate::CompilationType::Relaxed,
            problem:    &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking:    &CmpChar,
            cutoff:     &NoCutoff,
            max_width:  3,
            best_lb:    0,
            residual: &SubProblem { 
                state: Arc::new('r'), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let _ = mdd.compile(&input);
        
        let dot = include_str!("../../../../resources/visualisation_tests/default_viz.dot");
        let config = VizConfigBuilder::default().build().unwrap();            
        let s = mdd.as_graphviz(&config); 
        //println!("{}", s)
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    #[test]
    fn test_terse_visualisation() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type:  crate::CompilationType::Relaxed,
            problem:    &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking:    &CmpChar,
            cutoff:     &NoCutoff,
            max_width:  3,
            best_lb:    0,
            residual: &SubProblem { 
                state: Arc::new('r'), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let _ = mdd.compile(&input);
        
        let dot = include_str!("../../../../resources/visualisation_tests/terse_viz.dot");
        let config = VizConfigBuilder::default()
            .show_value(false)
            .show_deleted(false)
            .show_rub(false)
            .show_locb(false)
            .show_threshold(false)
            .build().unwrap();           
        let s = mdd.as_graphviz(&config); 
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    #[test]
    fn test_show_deleted_viz() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type:  crate::CompilationType::Relaxed,
            problem:    &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking:    &CmpChar,
            cutoff:     &NoCutoff,
            max_width:  3,
            best_lb:    0,
            residual: &SubProblem { 
                state: Arc::new('r'), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let _ = mdd.compile(&input);
        
        let dot = include_str!("../../../../resources/visualisation_tests/deleted_viz.dot");
        let config = VizConfigBuilder::default()
            .show_value(false)
            .show_deleted(true)
            .show_rub(false)
            .show_locb(false)
            .show_threshold(false)
            .build().unwrap();         
        let s = mdd.as_graphviz(&config); 
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    #[test]
    fn test_show_group_merged() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type:  crate::CompilationType::Relaxed,
            problem:    &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking:    &CmpChar,
            cutoff:     &NoCutoff,
            max_width:  3,
            best_lb:    0,
            residual: &SubProblem { 
                state: Arc::new('r'), 
                value: 0, 
                path:  vec![], 
                ub:    isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let _ = mdd.compile(&input);
        
        let dot = include_str!("../../../../resources/visualisation_tests/clusters_viz.dot");
        let config = VizConfigBuilder::default()
            .show_value(false)
            .show_deleted(true)
            .show_rub(false)
            .show_locb(false)
            .show_threshold(false)
            .group_merged(true)
            .build().unwrap();
        let s = mdd.as_graphviz(&config);
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    fn strip_format(s: &str) -> String {
        s.lines().map(|l| l.trim()).collect()
    }

    #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
    struct DummyState {
        value: isize,
        depth: usize,
    }

    #[derive(Copy, Clone)]
    struct DummyProblem;
    impl Problem for DummyProblem {
        type State = DummyState;

        fn nb_variables(&self)  -> usize { 3 }
        fn initial_value(&self) -> isize { 0 }
        fn initial_state(&self) -> Self::State {
            DummyState {
                value: 0,
                depth: 0,
            }
        }

        fn transition(&self, state: &Self::State, decision: crate::Decision) -> Self::State {
            DummyState {
                value: state.value + decision.value,
                depth: 1 + state.depth
            }
        }

        fn transition_cost(&self, _: &Self::State, _: &Self::State, decision: crate::Decision) -> isize {
            decision.value
        }

        fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
            -> Option<crate::Variable> {
            if depth < self.nb_variables() {
                Some(Variable(depth))
            } else {
                None
            }
        }

        fn for_each_in_domain(&self, var: crate::Variable, _: &Self::State, f: &mut dyn DecisionCallback) {
            for d in 0..=2 {
                f.apply(Decision {variable: var, value: d})
            }
        }
    }

    #[derive(Clone,Copy)]
    struct DummyInfeasibleProblem;
    impl Problem for DummyInfeasibleProblem {
        type State = DummyState;

        fn nb_variables(&self)  -> usize { 3 }
        fn initial_value(&self) -> isize { 0 }
        fn initial_state(&self) -> Self::State {
            DummyState {
                value: 0,
                depth: 0,
            }
        }

        fn transition(&self, state: &Self::State, decision: crate::Decision) -> Self::State {
            DummyState {
                value: state.value + decision.value,
                depth: 1 + state.depth
            }
        }

        fn transition_cost(&self, _: &Self::State, _: &Self::State, decision: crate::Decision) -> isize {
            decision.value
        }

        fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
            -> Option<crate::Variable> {
            if depth < self.nb_variables() {
                Some(Variable(depth))
            } else {
                None
            }
        }

        fn for_each_in_domain(&self, _: crate::Variable, _: &Self::State, _: &mut dyn DecisionCallback) {
            /* do nothing, just consider that all domains are empty */
        }
    }

    #[derive(Copy, Clone)]
    struct DummyRelax;
    impl Relaxation for DummyRelax {
        type State = DummyState;

        fn merge(&self, s: &mut dyn Iterator<Item=&Self::State>) -> Self::State {
            s.next().map(|s| {
                DummyState {
                    value: 100,
                    depth: s.depth
                }
            }).unwrap()
        }
        fn relax(&self, _: &Self::State, _: &Self::State, _: &Self::State, _: Decision, _: isize) -> isize {
            20
        }
        fn fast_upper_bound(&self, state: &Self::State) -> isize {
            (DummyProblem.nb_variables() - state.depth) as isize * 10
        }
    }

    #[derive(Copy, Clone)]
    struct DummyRanking;
    impl StateRanking for DummyRanking {
        type State = DummyState;

        fn compare(&self, a: &Self::State, b: &Self::State) -> Ordering {
            a.value.cmp(&b.value).reverse()
        }
    }
}