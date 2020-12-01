// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! This module implements an MDD that behaves as a PooledMDD and at the same
//! time, benefits from the local bound computation.

use std::rc::Rc;
use std::sync::Arc;
use std::hash::Hash;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use crate::{Decision, NodeFlags, PartialAssignment, Config, MDDType, MDD, Completion, FrontierNode, Solution, Reason, Variable, VarSet, SelectableNode};
use crate::implementation::mdd::deep::deep_pooled::NodeRef::{Virtual, Actual};

/// This is a typesafe identifier for a node in the mdd graph. In practice,
/// it maps to the position of the given node in the `nodes` field of the
/// pooled mdd.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct NodeId(usize);

/// This is a typesafe identifier for an edge in the mdd graph. In practice,
/// it maps to the position of the given edge in the `edges` field of the
/// pooled mdd.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct EdgeId(usize);

/// This is a typesafe identifier for a layer in the mdd. In practice, it maps
/// to the position of the layer in the pooled mdd.
///
/// # Warning
/// Because the pooled mdd contains so called long arcs, a layer may not contain
/// *all* the nodes it ought to normally contain.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct LayerId(usize);

/// This is the concrete description of an edge in the mdd. It retains the
/// source node from which this edge stretches, the weight of the edge and
/// the decition that led to the drawing of this edge.
///
/// # Note
/// For the sake of simplicity, the edge constitute some kind of simply linked
/// list: each edge may possibly contain the identifier of the next edge in the
/// list. This makes it simple and efficient to store the list of inbound edges
/// inside of a node.
#[derive(Debug, Copy, Clone)]
struct EdgeData {
    /// This is the source node from which the edge originates
    src: NodeId,
    /// This is the weight of the edge
    weight: isize,
    /// This is the decision that led to the stretching of this edge.
    /// # Note:
    /// In general all edges should have an associated decision. However,
    /// it was decided that in order to efficiently maintain the frontier
    /// cutset of the pooled mdd while retaining the ability to easily compute
    /// local bounds for cutset nodes; these would be stored _between_ the layer
    /// demarcations, and connected to the node they have been merged into via
    /// a zero-weighted edge having no associated decision.
    decision: Option<Decision>,
    /// This is the 'pointer' to the next edge in the linked list.
    next: Option<EdgeId>
}
/// This is the concrete description of a node in the mdd. It retains the state
/// of the nodes, its inbound edges (which form a linked list) as well as some
/// meta data.
#[derive(Clone)]
struct NodeData<T> {
    /// This is the state of the node.
    state: Rc<T>,
    /// This is the length of the longest path between the root and this node.
    from_top: isize,
    /// This is the length of the longest path between this node and the
    /// terminal node of this mdd (or seen an other way, the length of the
    /// longest path between this node and any node from the last layer of the
    /// mdd).
    ///
    /// # Note
    /// This field is only ever populated after the MDD has been fully unrolled.
    from_bot: isize,
    /// A group of flag telling if the node is an exact node, if it is a relaxed
    /// node (helps to determine if the best path is an exact path) and if the
    /// node is reachable in a backwards traversal of the MDD starting at the
    /// terminal node.
    flags: NodeFlags,
    /// If present, this is the index of the head of the edge list of this node.
    /// This node stores the incoming edges entering the node.
    inbound: Option<EdgeId>,
    /// If present, this is the index of the parent sitting on the longest path
    /// between the root and this node (the path whose length is `lp_from_top`.
    best_edge: Option<EdgeId>,

    /// This is an unique identifier for a NodeData object. Its purpose is to
    /// perform the reconciliation between "merged nodes" and their exact
    /// ancestors belonging to the cutset.
    ///
    /// The MDD is in charge of making sure that each instance receives an
    /// unique transient identifier.
    transient_id : usize,
}
/// This structure stores basic information about a given layer: the index of
/// the first node belonging to this layer and the first index after the end
/// of it.
#[derive(Debug, Copy, Clone)]
struct LayerData {
    /// This is the index of the first node belonging to this layer.
    begin: usize,
    /// The position *after* the last node of this layer.
    /// Because `end` points *after* the end of the layer, it must be the case
    /// that the layer is empty whenever `start == end`.
    end: usize
}

/// This encapsulates the notion of a "reference to some nodata" which may
/// or may not yet have been added to the set of 'nodes'.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum NodeRef {
    /// This variant encapsulates the case where the referenced node has already
    /// been added to the nodes of the MDD. In this case, the referenced data
    /// has an actual Node identifier
    Actual(NodeId),
    /// This variant encapsulates the case where the referenced node has not yet
    /// been added to the nodes of the MDD. In this case, the referenced data
    /// is identified using its transient id.
    Virtual(usize)
}
/// A cutset node comprises two parts:
///
/// 1. The actual information about the cutset node. Note, because we are
///    implementing a scheme where the cutset nodes reside *between* the layers,
///    the actual data is identified with a node id.
///
/// 2. The referenced node data which designates the inexact node standing for
///    the node from the cutset. (It designates the node in which this cutset
///    node has been merged).
///
#[derive(Debug, Copy, Clone)]
struct CutsetNode {
    /// The actual information about the cutset node. Note, because we are
    /// implementing a scheme where the cutset nodes reside *between* the layers,
    /// the actual data is identified with a node id.
    the_node: NodeId
}

/// A convenient type alias to denote the cutset
type CutSet = HashMap<NodeRef, Vec<CutsetNode>>;

/// This structure provides an implementation of a pooled mdd (one that may
/// feature long arcs) while at the same time being a deep mdd (one that
/// materializes the complete mdd graph). It implements rough upper bound and
/// local bounds to strengthen the pruning achieved by the branch and bound
/// algorithm. It uses the given configuration to develop the (approximate) MDD.
#[derive(Clone)]
pub struct PooledDeepMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone,
{
    /// This object holds the 'configuration' of the MDD: the problem, relaxation
    /// and heuristics to use.
    config : C,
    /// Because the same instance can be used to develop an exact,
    /// restricted or relaxed MDD, we use the `mddtype` to remember the type
    /// of MDD we are currently developing or have developed.
    mddtype: MDDType,
    /// This is the maximum width allowed for a layer of the MDD. It is
    /// determined once at the beginning of the MDD derivation.
    max_width: usize,
    /// This is the pool of candidate nodes that might possibly participate in
    /// the next layer.
    pool: HashMap<Rc<T>, NodeData<T>>,
    /// This is the complete list with all the nodes data of the graph.
    /// The position a `NodeId` refers to is to be understood as a position
    /// in this vector.
    ///
    /// # Important Note:
    /// A node is only present in this vector iff it has been withdrawn from the
    /// pool **and it belongs to the final mdd**.
    nodes: Vec<NodeData<T>>,
    /// This is the complete list with all the edges data of the graph.
    /// The position an `EdgeId` refers to is to be understood as a position
    /// in this vector.
    ///
    /// # Notes
    /// 1. When one merges nodes, no edge is lost. They are relaxed and
    ///    redirected towards the merged node.
    /// 2. When one deletes a node (restrict), it would be nice if we could
    ///    recycle the dangling edges (not done so far).
    /// 3. This version of the code does not use "intermediate frontier cutset".
    ///    The frontier consists of the true nodes.
    edges: Vec<EdgeData>,
    /// This vector contains the identifiers of all nodes beloning to the
    /// frontier cutset of the mdd.
    cutset: CutSet,
    /// This is the complete list with all the layers of the graph.
    /// The position a `LayerId` refers to is to be understood as a position
    /// in this vector.
    layers: Vec<LayerData>,
    /// If present, this is the identifier of the last exact layer.
    /// # Warning
    /// Even though the last layer is easily spotted in a pooled mdd, nothing
    /// guarantees that the layer data associated to the LEL is sufficient to
    /// retrieve all nodes that ought to belong to LEL. (This is due to the
    /// long arcs in the mdd graph).
    lel: Option<LayerId>,
    /// If present, this is a shared reference to the partial assignment describing
    /// the path between the exact root of the problem and the root of this
    /// (possibly approximate) sub-MDD.
    root_pa: Option<Arc<PartialAssignment>>,
    /// This is the best known lower bound at the time of the MDD unrolling.
    /// This field is set once before developing mdd.
    best_lb: isize,
    /// This field memoizes the best node of the MDD. That is, the node of this
    /// mdd having the longest path from root.
    best_node: Option<NodeId>,
    /// A flag indicating whether this mdd is exact
    is_exact : bool,

    /// This is the counter which is used to give an unique transient identifier
    /// to all node data.
    transient_cnt: usize,
}

/// As the name suggests, `PooledDeepMDD` is an implementation of the `MDD` trait.
/// See the trait definiton for the documentation related to these methods.
impl <T, C> MDD<T, C> for PooledDeepMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn config(&self) -> &C {
        &self.config
    }
    fn config_mut(&mut self) -> &mut C {
        &mut self.config
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.develop(MDDType::Exact, root, best_lb, ub)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.develop(MDDType::Restricted, root, best_lb, ub)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.develop(MDDType::Relaxed, root, best_lb, ub)
    }

    fn is_exact(&self) -> bool {
        self.is_exact
    }
    fn best_value(&self) -> isize {
        self.best_node.map_or(isize::min_value(), |nid| self.nodes[nid.0].from_top)
    }
    fn best_solution(&self) -> Option<Solution> {
        self.best_node.map(|index| {
            Solution::new(Arc::new(self.best_partial_assignment_for(index)))
        })
    }
    fn for_each_cutset_node<F>(&self, mut func: F) where F: FnMut(FrontierNode<T>) {
        for (referenced, cnodes) in self.cutset.iter() {
            if let Actual(nid) = referenced {
                //if self.nodes[nid.0].flags.is_feasible() {
                    for cnode in cnodes.iter() {
                        func(self.node_to_frontier_node(cnode.the_node, *nid));
                    }
                //}
            }
        }
    }
}

/// Private functions
impl <T, C> PooledDeepMDD<T, C>
where T: Eq + Hash + Clone,
      C: Config<T> + Clone
{
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(c: C) -> Self {
        Self {
            config           : c,
            mddtype          : MDDType::Exact,
            max_width        : usize::max_value(),
            pool             : Default::default(),
            nodes            : vec![],
            edges            : vec![],
            cutset           : Default::default(),
            layers           : vec![],
            lel              : None,
            root_pa          : None,
            best_lb          : isize::min_value(),
            best_node        : None,
            is_exact         : true,

            transient_cnt    : 1,
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    fn clear(&mut self) {
        self.config.clear();
        self.mddtype = MDDType::Exact;
        self.max_width = usize::max_value();
        self.pool.clear();
        self.nodes.clear();
        self.edges.clear();
        self.cutset.clear();
        self.layers.clear();
        self.lel       = None;
        self.root_pa   = None;
        self.best_lb   = isize::min_value();
        self.best_node = None;
        self.is_exact  = true;

        self.transient_cnt = 1;
    }
    /// Returns a shared reference to the partial assignment describing
    /// the path between the exact root of the problem and the root of this
    /// (possibly approximate) sub-MDD.
    fn root_pa(&self) -> Arc<PartialAssignment> {
        self.root_pa.as_ref()
            .map_or(Arc::new(PartialAssignment::Empty), |refto| Arc::clone(&refto))
    }
    /// Returns the best partial assignment leading to the node identified by
    /// the `node` index in the graph.
    fn best_partial_assignment_for(&self, node: NodeId) -> PartialAssignment {
        PartialAssignment::FragmentExtension {
            fragment: self.longest_path(node),
            parent  : self.root_pa()
        }
    }
    /// Returns the longest path from root node to the destination node.
    fn longest_path(&self, mut nid: NodeId) -> Vec<Decision> {
        let mut path = vec![];
        while let Some(eid) = self.nodes[nid.0].best_edge {
            let edge = self.edges[eid.0];
            path.push(edge.decision.unwrap());
            nid = edge.src;
        }
        path
    }
    /// Converts an internal node into an equivalent frontier node
    fn node_to_frontier_node(&self, nid: NodeId, merged_into: NodeId) -> FrontierNode<T> {
        let node   = &self.nodes[nid.0];
        //let mrg_nd = &self.nodes[merged_into.0];
        //let ub_bot = mrg_nd.from_top.saturating_add(mrg_nd.from_bot);
        let ub_est = node.from_top.saturating_add(self.config.estimate(node.state.as_ref()));
        FrontierNode {
            state : Arc::new(node.state.as_ref().clone()),
            path  : Arc::new(self.best_partial_assignment_for(nid)),
            lp_len: node.from_top,
            ub    : ub_est,//ub_bot.min(ub_est)
        }
    }
    /// Develops/Unrolls the requested type of MDD, starting from a given root
    /// node. It only considers nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`).
    fn develop(&mut self, mddtype: MDDType, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.clear();

        let mut free_vars  = self.config.load_variables(root);
        self.mddtype       = mddtype;
        self.max_width     = self.config.max_width(&free_vars);
        self.root_pa       = Some(Arc::clone(&root.path));
        self.best_lb       = best_lb;

        let root_data      = NodeData::from(root);
        self.config.upon_node_insert(root.state.as_ref());
        self.pool.insert(Rc::clone(&root_data.state), root_data);

        let mut current= vec![];
        while let Some(var) = self.next_var(&free_vars, &current) {
            // Did the cutoff kick in ?
            if self.config.must_stop(best_lb, ub) {
                return Err(Reason::CutoffOccurred);
            }

            free_vars.remove(var);
            self.select_relevant_nodes(var, &mut current);
            self.config.upon_new_layer(var, &mut current.iter().map(|n|n.state.as_ref()));
            self.squash_if_needed(&mut current);
            let this_layer = self.add_layer(&mut current);

            for nid in this_layer.begin..this_layer.end {
                let src_state = Rc::clone(&self.nodes[nid].state);
                for val in self.config.domain_of(src_state.as_ref(), var) {
                    let decision = Decision { variable: var, value: val };
                    let state    = self.config.transition(src_state.as_ref(), &free_vars, decision);
                    let weight   = self.config.transition_cost(src_state.as_ref(), &free_vars, decision);

                    self.branch(NodeId(nid), state, decision, weight)
                }
            }
        }

        Ok(self.finalize())
    }
    /// Finalizes the computation of the MDD: it identifies the best terminal
    /// node, checks if the MDD is exact and computes the local bounds of the
    /// cutset nodes.
    fn finalize(&mut self) -> Completion {
        let pool = &mut self.pool;
        let nodes= &mut self.nodes;
        let layers=&mut self.layers;

        // add the nodes remaining in the pool as final layer
        let prev_end = nodes.len();
        pool.drain().for_each(|(_, v)| nodes.push(v));
        let last_layer = LayerData{begin: prev_end, end: nodes.len()};
        layers.push(last_layer);

        let terminal_nodes = nodes.iter().enumerate().skip(last_layer.begin);
        self.best_node = terminal_nodes
            .max_by_key(|(_id, node)| node.from_top)
            .map(|(id, _node)|NodeId(id));

        self.compute_is_exact();
        //if self.mddtype == MDDType::Relaxed {
        //    self.compute_local_bounds()
        //}

        Completion{
            is_exact     : self.is_exact,
            best_value   : self.best_node.map(|nid| self.nodes[nid.0].from_top)
        }
    }
    /// Returns the next variable to branch on (according to the configured
    /// branching heuristic) or None if all variables have been assigned a value.
    fn next_var(&self, free_vars: &VarSet, current: &[NodeData<T>]) -> Option<Variable> {
        let mut curr_it = current.iter().map(|n| n.state.as_ref());
        let mut next_it = self.pool.keys().map(|k| k.as_ref());

        self.config.select_var(free_vars, &mut curr_it, &mut next_it)
    }
    /// Selects and moves the set of nodes from the `current` pool that are
    /// relevant to build the next layer (knowing that the next variable that
    /// will be decided upon will be `var`).
    fn select_relevant_nodes(&mut self, var: Variable, current: &mut Vec<NodeData<T>>) {
        current.clear();

        // Add all selected nodes to the next layer
        for (s, n) in self.pool.iter() {
            if self.config.impacted_by(s, var) {
                current.push(n.clone());
            }
        }

        // Remove all nodes that belong to the current layer from the pool
        for node in current.iter() {
            self.pool.remove(&node.state);
        }
    }
    /// This method records the branching from the given `node` with the given
    /// `decision`. It creates a fresh node for the `dest` state (or reuses one
    /// if `dest` already belongs to the pool) and draws an edge of of the given
    /// `weight` between `src` and the new node.
    ///
    /// ### Note:
    /// In case where this branching would create a new longest path to an
    /// already existing node, the length and best parent of the pre-existing
    /// node are updated.
    fn branch(&mut self, src: NodeId, dest: T, decision: Decision, weight: isize) {
        let parent = &self.nodes[src.0];
        let edge   = Self::_new_edge(src, weight, Some(decision), &mut self.edges);
        let dst_node = NodeData {
            state    : Rc::new(dest),
            from_top : parent.from_top.saturating_add(weight),
            from_bot : isize::min_value(),
            flags    : parent.flags, // if its inexact, it will be or relaxed it will be considered inexact or relaxed too
            inbound  : Some(edge),
            best_edge: Some(edge),
            transient_id: self.transient_cnt,
        };
        self.transient_cnt += 1;
        self.add_to_pool(dst_node)
    }

    /// Inserts the given node in the next layer or updates it if needed.
    fn add_to_pool(&mut self, node: NodeData<T>) {
        let best_lb = self.best_lb;
        let config  = &mut self.config;
        let pool    = &mut self.pool;
        let nodes   = &mut self.nodes;
        let edges   = &mut self.edges;
        let cutset  = &mut self.cutset;

        match pool.entry(Rc::clone(&node.state)) {
            Entry::Vacant(re) => {
                let est = config.estimate(node.state.as_ref());
                let ub  = node.from_top.saturating_add(est);
                if ub > best_lb {
                    config.upon_node_insert(node.state.as_ref());
                    re.insert(node);
                }
            },
            Entry::Occupied(mut re) => {
                let old = re.get_mut();
                Self::merge(old, node, nodes, edges, cutset);
            }
        }
    }

    /// This function adds the given node to the cutset, attaching it to the
    /// transient id `t_id`
    fn add_to_cutset(node: &NodeData<T>,
                     t_id: usize,
                     nodes: &mut Vec<NodeData<T>>,
                     cutset:&mut CutSet) {
        let nid = NodeId(nodes.len());
        nodes.push(node.clone());

        let cnode = CutsetNode{ the_node: nid };
        cutset.entry(Virtual(t_id))
            .and_modify(|re| re.push(cnode))
            .or_insert_with(|| vec![cnode]);
    }
    /// This function 'merges' the given transient ids.
    /// In practice, this means that all cutset nodes attached to the node
    /// `from` will now be attached to the node `to`.
    fn merge_transient_ids(from: usize, to: usize, cutset:&mut CutSet) {
        let from = cutset.remove(&Virtual(from));
        if let Some(mut from) = from {
            cutset.entry(Virtual(to))
                .and_modify(|v| v.append(&mut from))
                .or_insert(from);
        }
    }
    /// This function is called upon insertion of a node in the set of `nodes'
    /// from the MDD. This is used to tell all cutset nodes attached to the
    /// transient id of the given node that they must now be attached to its
    /// actual id 'nid'.
    fn upon_node_insert(node: &NodeData<T>, nid: NodeId, cutset:&mut CutSet) {
        let from = cutset.remove(&Virtual(node.transient_id));
        if let Some(from) = from {
            cutset.insert(Actual(nid), from);
        }
    }

    /// This method ensures that a node be effectively merged with the 2nd one
    /// even though it is shielded behind a shared ref. This method makes sure
    /// to keep the cutset consistent as needed.
    ///
    /// # Tech note:
    /// This was developed as an associated function to get over a multiple
    /// mutable borrows error issued by the warning. This is also the reason
    /// why `nodes`, `edges` and `cutset` are passed as arguments.
    fn merge(old: &mut NodeData<T>, new: NodeData<T>,
             nodes: &mut Vec<NodeData<T>>,
             edges: &mut Vec<EdgeData>,
             cutset:&mut CutSet) {

        // maintain the frontier cutset
        if old.flags.is_exact() && !new.flags.is_exact() {
            Self::add_to_cutset(old, old.transient_id, nodes, cutset);
        } else if new.flags.is_exact() && !old.flags.is_exact() {
            Self::add_to_cutset(&new, old.transient_id, nodes, cutset);
        }

        Self::merge_transient_ids(new.transient_id, old.transient_id, cutset);

        // concatenate edges lists
        let mut next_edge = new.inbound;
        while let Some(eid) = next_edge {
            let edge = &mut edges[eid.0];
            next_edge   = edge.next;
            edge.next   = old.inbound;
            old.inbound = Some(eid);
        }

        // merge flags
        old.flags.set_exact(old.flags.is_exact() && new.flags.is_exact());
        if new.from_top > old.from_top {
            old.from_top  = new.from_top;
            old.best_edge = new.best_edge;
        }
    }
    /// Possibly restricts or relaxes the current layer.
    fn squash_if_needed(&mut self, current: &mut Vec<NodeData<T>>) {
        match self.mddtype {
            MDDType::Exact => {},
            MDDType::Restricted =>
                if current.len() > self.max_width {
                    self.restrict(current);
                },
            MDDType::Relaxed =>
                if current.len() > self.max_width {
                    self.relax(current);
                }
        }
    }
    /// This method adds a new layer to the graph comprising all the nodes from
    /// the `current` list.
    ///
    /// This method must be called after all transitions of all nodes of the
    /// current layer have been unrolled.
    fn add_layer(&mut self, current: &mut Vec<NodeData<T>>) -> LayerData {
        let begin      = self.nodes.len();
        let end        = begin + current.len();
        let this_layer = LayerData {begin, end};

        self.layers.push(this_layer);

        //self.nodes.append(current);
        for node in current.drain(..) {
            let nid = NodeId(self.nodes.len());
            Self::upon_node_insert(&node, nid, &mut self.cutset);
            self.nodes.push(node);
        }

        this_layer
    }
    /// This is the 'method' way to add an edge to the graph and return its
    /// `EdgeId`.
    ///
    /// # Note:
    /// This function does not esures that the edge will actually is connected
    /// to a node from the graph.
    fn _new_edge(src: NodeId, weight: isize, decision: Option<Decision>, edges: &mut Vec<EdgeData>) -> EdgeId {
        let eid = EdgeId(edges.len());
        edges.push(EdgeData { src, weight, decision, next: None });
        eid
    }
    /// This method restricts the current layer to make sure
    /// it fits within the maximum "width" size.
    fn restrict(&mut self, current: &mut Vec<NodeData<T>>) {
        if self.lel.is_none() {
            self.lel = Some(LayerId(self.layers.len()-1));
        }
        current.sort_unstable_by(|a, b| self.config.compare(a, b).reverse());
        // todo: i should recycle edges
        // then kill all the nodes that must be dropped
        current.truncate(self.max_width);
    }
    /// This method relaxes the current layer to make sure it fits within the
    /// maximum "width" size. The overdue nodes are merged according to the
    /// configured strategy.
    ///
    /// # Warning
    /// It is your responsibility to make sure the layer is broad enough to be
    /// relaxed. Failing to do so would result in a panic!
    ///
    /// # Warning
    /// This function will panic if you request a relaxation that would leave
    /// zero nodes in the current layer.
    fn relax(&mut self, current: &mut Vec<NodeData<T>>) {
        if self.lel.is_none() {
            self.lel = Some(LayerId(self.layers.len()-1));
        }

        // Select the nodes to be merged
        current.sort_unstable_by(|a, b| self.config.compare(a, b).reverse());
        let (_keep, squash) = current.split_at_mut(self.max_width - 1);

        // Merge the states of the nodes that must go
        let merged = self.config.merge_states(&mut squash.iter().map(|n| n.state.as_ref()));
        let merged = Rc::new(merged);

        // .. make a default node
        let mut merged_node = NodeData {
            state      : Rc::clone(&merged),
            flags      : NodeFlags::new_relaxed(),
            from_top   : isize::min_value(),
            from_bot   : isize::min_value(),
            inbound    : None,
            best_edge  : None,
            transient_id: self.transient_cnt
        };
        self.transient_cnt += 1;

        // Maintain the frontier cutset
        for node in squash.iter() {
            if node.flags.is_exact() {
                Self::add_to_cutset(node, merged_node.transient_id, &mut self.nodes, &mut self.cutset);
            }
        }

        // Relax all edges and transfer them to the new merged node (op. gamma)
        for node in squash.iter() {
            let mut it = node.inbound;
            while let Some(eid) = it {
                let edge      = &mut self.edges[eid.0];
                if let Some(decision) = edge.decision { // you only want to proceed for real edges
                    let parent    = &self.nodes[edge.src.0];

                    let src_state = parent.state.as_ref();
                    let dst_state = node.state.as_ref();
                    let mrg_state = merged_node.state.as_ref();

                    edge.weight = self.config.relax_edge(src_state, dst_state, mrg_state, decision, edge.weight);

                    // update the merged node if the relaxed edge improves longest path
                    if parent.from_top.saturating_add(edge.weight) > merged_node.from_top {
                        merged_node.from_top  = parent.from_top.saturating_add(edge.weight);
                        merged_node.best_edge = Some(eid);
                    }
                }
                it        = edge.next;
                edge.next = merged_node.inbound;
                merged_node.inbound = Some(eid);
            }
        }

        // Save the result of the merger
        current.truncate(self.max_width - 1);
        // Determine the identifier of the new merged node. If the merged state
        // is already known, combine the known state with the merger
        let pos = current.iter().enumerate()
            .find(|(_i, nd)| nd.state == merged).map(|(i,_)|i);
        if let Some(pos) = pos {
            Self::merge(&mut current[pos], merged_node, &mut self.nodes, &mut self.edges, &mut self.cutset);
        } else {
            current.push(merged_node);
        }
    }
    /// Checks if the mdd is exact or if the best terminal node has an exact
    /// best path from the root.
    fn compute_is_exact(&mut self) {
        self.is_exact = self.lel.is_none()
            || (self.mddtype == MDDType::Relaxed && self.has_exact_best_path(self.best_node))
    }
    /// Returns true iff the longest r-t path of the MDD traverses no relaxed
    /// node.
    fn has_exact_best_path(&self, node: Option<NodeId>) -> bool {
        if let Some(node_id) = node {
            let n = &self.nodes[node_id.0];
            if n.flags.is_exact() {
                true
            } else {
                !n.flags.is_relaxed() && self.has_exact_best_path(n.best_edge.map(|e| self.edges[e.0].src))
            }
        } else {
            true
        }
    }
    /// Computes the local bounds of the cutset nodes.
    fn compute_local_bounds(&mut self) {
        if !self.is_exact { // if it's exact, there is nothing to be done
            let lel = self.lel.unwrap();

            let mut depth = self.layers.len()-1;
            let mut layer = self.layers[depth];
            // all the nodes from the last layer have a lp_from_bot of 0
            for node in self.nodes[layer.begin..layer.end].iter_mut() {
                node.from_bot = 0;
                node.flags.set_feasible(true);
            }

            while depth > lel.0 {
                for node in self.nodes[layer.begin..layer.end].iter() {
                    if node.flags.is_feasible() {
                        let mut inbound = node.inbound;
                        while let Some(edge_id) = inbound {
                            let edge = self.edges[edge_id.0];

                            let from_bot_using_edge = node.from_bot.saturating_add(edge.weight);
                            let parent = unsafe {
                                let ptr = &self.nodes[edge.src.0] as *const NodeData<T> as *mut NodeData<T>;
                                &mut *ptr
                            };

                            parent.from_bot = parent.from_bot.max(from_bot_using_edge);
                            parent.flags.set_feasible(true);

                            inbound = edge.next;
                        }
                    }
                }

                depth-= 1;
                layer = self.layers[depth];
            }
        }
    }
}

impl <T, C> From<C> for PooledDeepMDD<T, C>
where T: Eq + Hash + Clone,
      C: Config<T> + Clone
{
    fn from(c: C) -> Self {
        Self::new(c)
    }
}
impl <T:Clone> From<&FrontierNode<T>> for NodeData<T> {
    fn from(n: &FrontierNode<T>) -> Self {
        let node_state = Rc::new(n.state.as_ref().clone());
        let node_value = n.lp_len;
        NodeData{
            state    : node_state,
            from_top : node_value,
            from_bot : 0,
            flags    : Default::default(),
            inbound  : None,
            best_edge: None,
            transient_id: 0
        }
    }
}
impl <T> SelectableNode<T> for NodeData<T> {
    fn state(&self) -> &T {
        self.state.as_ref()
    }
    fn value(&self) -> isize {
        self.from_top
    }
    fn is_exact(&self) -> bool {
        self.flags.is_exact()
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
mod test_pooled_deep_mdd {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::mdd::{MDD, Config};
    use crate::common::{Decision, Domain, FrontierNode, PartialAssignment, Reason, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::mdd::MDDType;
    use crate::test_utils::{MockConfig, MockCutoff, Proxy};
    use mock_it::Matcher;
    use crate::{VariableHeuristic, NaturalOrder, PooledDeepMDD, NodeSelectionHeuristic, SelectableNode};
    use std::collections::HashMap;
    use std::cmp::Ordering;

    type DD<T, C> = PooledDeepMDD<T, C>;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd = DD::from(config);

        assert_eq!(MDDType::Exact, mdd.mddtype);
    }
    #[test]
    fn mdd_type_changes_depending_on_the_requested_type_of_mdd() {
        let root_n = FrontierNode {
            state: Arc::new(0),
            lp_len: 0,
            ub: 24,
            path: Arc::new(PartialAssignment::Empty)
        };

        let config = MockConfig::default();
        let mut mdd = DD::from(config);

        assert!(mdd.relaxed(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Relaxed, mdd.mddtype);

        assert!(mdd.restricted(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Restricted, mdd.mddtype);

        assert!(mdd.exact(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Exact, mdd.mddtype);
    }

    #[derive(Copy, Clone)]
    struct DummyProblem;

    impl Problem<usize> for DummyProblem {
        fn nb_vars(&self) -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize { 0 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..=2).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> isize {
            d.value
        }
    }

    #[derive(Copy, Clone)]
    struct DummyRelax;

    impl Relaxation<usize> for DummyRelax {
        fn merge_states(&self, _: &mut dyn Iterator<Item=&usize>) -> usize {
            100
        }
        fn relax_edge(&self, _: &usize, _: &usize, _: &usize, _: Decision, _: isize) -> isize {
            20
        }
        fn estimate(&self, _state: &usize) -> isize {
            50
        }
    }
    #[test]
    fn exact_no_cutoff_completion_must_be_coherent_with_outcome() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build());

        let root   = mdd.config().root_node();
        let result = mdd.exact(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    #[test]
    fn restricted_no_cutoff_completion_must_be_coherent_with_outcome_() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build());

        let root   = mdd.config().root_node();
        let result = mdd.restricted(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    #[test]
    fn relaxed_no_cutoff_completion_must_be_coherent_with_outcome() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build());

        let root   = mdd.config().root_node();
        let result = mdd.relaxed(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    #[test]
    fn exact_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build());

        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.exact(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    #[test]
    fn restricted_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build());

        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.restricted(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    #[test]
    fn relaxed_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build());

        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.relaxed(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build());

        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 2 },
                       Decision { variable: Variable(1), value: 2 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn restricted_drops_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build());

        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 2 },
                       Decision { variable: Variable(1), value: 2 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn relaxed_merges_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build());

        let root = mdd.config().root_node();
        assert!(mdd.relaxed(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 42);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 2 },
                       Decision { variable: Variable(1), value: 2 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build());

        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build());
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build());
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 0, 1000).is_ok());
        assert_eq!(false, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_populates_frontier_cutset() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build());
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 0, 1000).is_ok());

        let mut cut = vec![];
        mdd.for_each_cutset_node(|n| cut.push(*n.state.as_ref()));

        cut.sort_unstable();
        assert_eq!(vec![0, 1, 2], cut);
    }

    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build());
        let root = mdd.config().root_node();
        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build());
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert_eq!(false, mdd.is_exact())
    }

    #[derive(Clone, Copy)]
    struct DummyInfeasibleProblem;

    impl Problem<usize> for DummyInfeasibleProblem {
        fn nb_vars(&self) -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize { 0 }
        #[allow(clippy::reversed_empty_ranges)]
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..0).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> isize {
            d.value
        }
    }

    #[test]
    fn when_the_problem_is_infeasible_there_is_no_solution() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).build());
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).build());
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert_eq!(isize::min_value(), mdd.best_value())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).build());
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).build());
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = DD::from(mdd_builder(&pb, rlx).build());
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[derive(Copy, Clone, Default)]
    struct DummyIncrementalVarHeu {
        cleared : usize,
        inserted: usize,
        layers  : usize,
    }
    impl VariableHeuristic<usize> for DummyIncrementalVarHeu {
        fn next_var(&self, free_vars: &VarSet, current_layer: &mut dyn Iterator<Item=&usize>, next_layer: &mut dyn Iterator<Item=&usize>) -> Option<Variable> {
            NaturalOrder.next_var(free_vars, current_layer, next_layer)
        }

        fn upon_new_layer(&mut self, _var: Variable, _current_layer: &mut dyn Iterator<Item=&usize>) {
            self.layers += 1;
        }

        fn upon_node_insert(&mut self, _state: &usize) {
            self.inserted += 1;
        }

        fn clear(&mut self) {
            self.cleared += 1;
        }
    }
    #[test]
    fn config_is_cleared_before_developing_any_mddtype() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut heu = DummyIncrementalVarHeu::default();

        let config = mdd_builder(&pb, rlx)
            .with_branch_heuristic(Proxy::new(&mut heu))
            .build();

        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 100, 1000).is_ok());
        assert_eq!(1, heu.cleared);

        assert!(mdd.relaxed(&root, 100, 1000).is_ok());
        assert_eq!(2, heu.cleared);

        assert!(mdd.exact(&root, 100, 1000).is_ok());
        assert_eq!(3, heu.cleared);
    }

    #[test]
    fn upon_layer_is_called_whenever_a_new_layer_is_created() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut heu = DummyIncrementalVarHeu::default();

        let config = mdd_builder(&pb, rlx)
            .with_branch_heuristic(Proxy::new(&mut heu))
            .build();

        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 100, 1000).is_ok());
        assert_eq!(3, heu.layers);

        assert!(mdd.relaxed(&root, 100, 1000).is_ok());
        assert_eq!(6, heu.layers);

        assert!(mdd.exact(&root, 100, 1000).is_ok());
        assert_eq!(9, heu.layers);
    }

    #[test]
    fn upon_insert_is_called_whenever_a_non_existing_node_is_added_to_next_layer() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut heu = DummyIncrementalVarHeu::default();

        let config = mdd_builder(&pb, rlx)
            .with_branch_heuristic(Proxy::new(&mut heu))
            .build();

        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        // Exact mdd comprises 16 nodes (includes the root)
        assert!(mdd.exact(&root, -100, 1000).is_ok());
        assert_eq!(16, heu.inserted);

        assert!(mdd.restricted(&root, -100, 1000).is_ok());
        assert_eq!(30, heu.inserted);

        assert!(mdd.relaxed(&root, -100, 1000).is_ok());
        assert_eq!(46, heu.inserted);
    }



    /// The example problem and relaxation for the local bounds should generate
    /// the following relaxed MDD in which the layer 'a','b' is the LEL.
    ///
    /// ```plain
    ///                      r
    ///                   /     \
    ///                10        0
    ///               /           |
    ///             a              b
    ///             |     +--------+-------+
    ///             |     |        |       |
    ///             2    100       7       5
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
    struct LocBoundsExamplePb;
    impl Problem<char> for LocBoundsExamplePb {
        fn nb_vars(&self)       -> usize {  4  }
        fn initial_state(&self) -> char  { 'r' }
        fn initial_value(&self) -> isize {  0  }

        fn domain_of<'a>(&self, state: &'a char, _: Variable) -> Domain<'a> {
            (match *state {
                'r' => vec![10, 0],
                'a' => vec![2],
                'b' => vec![5, 7, 100],
                // c, d are merged into M
                'M' => vec![4],
                'e' => vec![0],
                'f' => vec![1, 2],
                _   => vec![0],
            }).into()
        }

        fn transition(&self, state: &char, _: &VarSet, d: Decision) -> char {
            match (*state, d.value) {
                ('r', 10) => 'a',
                ('r',  0) => 'b',
                ('a',  2) => 'c', // merged into M
                ('b',100) => 'd', // merged into M
                ('b',  7) => 'e',
                ('b',  5) => 'f',
                ('M',  4) => 'g',
                ('e',  0) => 'h',
                ('f',  1) => 'h',
                ('f',  2) => 'i',
                _         => 't'
            }
        }

        fn transition_cost(&self, _: &char, _: &VarSet, d: Decision) -> isize {
            d.value
        }
    }

    #[derive(Copy, Clone)]
    struct LocBoundExampleRelax;
    impl Relaxation<char> for LocBoundExampleRelax {
        fn merge_states(&self, _: &mut dyn Iterator<Item=&char>) -> char {
            'M'
        }

        fn relax_edge(&self, _: &char, _: &char, _: &char, _: Decision, cost: isize) -> isize {
            cost
        }
    }

    #[derive(Clone, Copy)]
    struct CmpChar;
    impl NodeSelectionHeuristic<char> for CmpChar {
        fn compare(&self, a: &dyn SelectableNode<char>, b: &dyn SelectableNode<char>) -> Ordering {
            a.state().cmp(b.state())
        }
    }

    #[test]
    fn relaxed_computes_local_bounds() {
        let mut mdd = DD::from(
            mdd_builder(&LocBoundsExamplePb, LocBoundExampleRelax)
                .with_nodes_selection_heuristic(CmpChar)
                .with_max_width(FixedWidth(3))
                .build());

        let root = mdd.config.root_node();
        assert!(mdd.relaxed(&root, 0, 1000).is_ok());

        assert_eq!(false, mdd.is_exact());
        assert_eq!(104,   mdd.best_value());

        let mut v = HashMap::<char, isize>::default();
        mdd.for_each_cutset_node(|n| {v.insert(*n.state, n.ub);});

        assert_eq!(104, v[&'c']); // because we cannot distinguish between c and d when they are reconciled with M
        assert_eq!(104, v[&'d']);
    }

}
