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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct NodeId(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct EdgeId(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct LayerId(usize);

#[derive(Debug, Copy, Clone)]
struct EdgeData {
    src     : NodeId,
    weight  : isize,
    decision: Option<Decision>,
    next    : Option<EdgeId>
}
#[derive(Clone)]
struct NodeData<T> {
    state    : Rc<T>,
    from_top : isize,
    from_bot : isize,
    flags    : NodeFlags,
    inbound  : Option<EdgeId>,
    best_edge: Option<EdgeId>
}
#[derive(Debug, Copy, Clone)]
struct LayerData {
    begin    : usize,
    end      : usize
}
#[derive(Clone)]
pub struct PooledDeepMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone,
{
    config : C,
    mddtype: MDDType,
    max_width: usize,
    //
    pool: HashMap<Rc<T>, NodeData<T>>,
    // Un noeud n'est ici que ssi on l'a sorti du pool et qu'il appartient au
    // mdd final.
    nodes: Vec<NodeData<T>>,
    // * Qd on fusionne des noeuds, on ne perd pas d'edge (elles sont relaxed et
    //   redirigées vers le noeud fusionné).
    // * Qd on supprime un noeud (restrict), on devrait recycler des edges qui
    //   seraient dangling.
    // * Les noeuds qui font partie du frontier cutset doivent être ajoutés au
    //   graphe (de sorte qu'on puisse les retraverser lors du bottom up)
    //   **et** il faut ajouter une edge de poids 0 entre le noeud du cutset
    //   et le noeud 'fusion' qu'il représente. (concrètement, on doit ajouter
    //   une edge dont la 'src' est l'id du noeud du cutset au nodedata du noeud
    //   fusion).
    //   ==> Les noeuds qui appartiennent au cutset se trouvent **entre** les
    //       démarcations formées par les différents layers.
    edges: Vec<EdgeData>,
    // Cela ne contient que les identifiants des noeuds qui font partie du cutset.
    cutset   : Vec<NodeId>,
    layers   : Vec<LayerData>,
    lel      : Option<LayerId>,
    //
    root_pa  : Option<Arc<PartialAssignment>>,
    best_lb  : isize,
    best_node: Option<NodeId>,
    is_exact : bool,
}

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
        self.cutset.iter()
            .filter(|nid| self.nodes[nid.0].flags.is_feasible())
            .for_each(|nid| func(self.node_to_frontier_node(*nid)));
    }
}

impl <T, C> PooledDeepMDD<T, C>
where T: Eq + Hash + Clone,
      C: Config<T> + Clone
{
    /// Creates a new mdd from the given configuration
    pub fn new(c: C) -> Self {
        Self {
            config           : c,
            mddtype          : MDDType::Exact,
            max_width        : usize::max_value(),
            pool             : Default::default(),
            nodes            : vec![],
            edges            : vec![],
            cutset           : vec![],
            layers           : vec![],
            lel              : None,
            root_pa          : None,
            best_lb          : isize::min_value(),
            best_node        : None,
            is_exact         : true,
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
    fn node_to_frontier_node(&self, nid: NodeId) -> FrontierNode<T> {
        let node   = &self.nodes[nid.0];
        let ub_bot = node.from_top.saturating_add(node.from_bot);
        let ub_est = node.from_top.saturating_add(self.config.estimate(node.state.as_ref()));
        FrontierNode {
            state : Arc::new(node.state.as_ref().clone()),
            path  : Arc::new(self.best_partial_assignment_for(nid)),
            lp_len: node.from_top,
            ub    : ub_bot.min(ub_est)
        }
    }

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
        if self.mddtype == MDDType::Relaxed {
            self.compute_local_bounds()
        }

        Completion{
            is_exact     : self.is_exact,
            best_value   : self.best_node.map(|nid| self.nodes[nid.0].from_top)
        }
    }

    fn next_var(&self, free_vars: &VarSet, current: &[NodeData<T>]) -> Option<Variable> {
        let mut curr_it = current.iter().map(|n| n.state.as_ref());
        let mut next_it = self.pool.keys().map(|k| k.as_ref());

        self.config.select_var(free_vars, &mut curr_it, &mut next_it)
    }
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
    fn branch(&mut self, src: NodeId, dest: T, decision: Decision, weight: isize) {
        let parent = &self.nodes[src.0];
        let edge   = Self::_new_edge(src, weight, Some(decision), &mut self.edges);
        let dst_node = NodeData {
            state    : Rc::new(dest),
            from_top : parent.from_top.saturating_add(weight),
            from_bot : isize::min_value(),
            flags    : parent.flags, // if its inexact, it will be or relaxed it will be considered inexact or relaxed too
            inbound  : Some(edge),
            best_edge: Some(edge)
        };
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

    fn merge(old: &mut NodeData<T>, new: NodeData<T>,
             nodes: &mut Vec<NodeData<T>>,
             edges: &mut Vec<EdgeData>,
             cutset:&mut Vec<NodeId>) {

        // maintain the frontier cutset
        let cutset_end = cutset.len();
        if old.flags.is_exact() && !new.flags.is_exact() {
            let nid = NodeId(nodes.len());
            nodes.push(old.clone());
            cutset.push(nid);
        } else if new.flags.is_exact() && !old.flags.is_exact() {
            let nid = NodeId(nodes.len());
            nodes.push(new.clone());
            cutset.push(nid);
        }

        // concatenate edges lists
        let mut next_edge = new.inbound;
        while let Some(eid) = next_edge {
            let edge = &mut edges[eid.0];
            next_edge   = edge.next;
            edge.next   = old.inbound;
            old.inbound = Some(eid);
        }

        // connect edges from potential cutset nodes to the merged node
        for nid in cutset.iter().skip(cutset_end).copied() {
            let eid     = Self::_new_edge(nid, 0, None, edges);
            let edge    = &mut edges[eid.0];
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
    fn add_layer(&mut self, current: &mut Vec<NodeData<T>>) -> LayerData {
        let begin      = self.nodes.len();
        let end        = begin + current.len();
        let this_layer = LayerData {begin, end};

        self.layers.push(this_layer);
        self.nodes.append(current);

        this_layer
    }
    fn new_edge(&mut self, src: NodeId, weight: isize, decision: Option<Decision>) -> EdgeId {
        Self::_new_edge(src, weight, decision, &mut self.edges)
    }
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

        // from that point on, all the subsequent cutset ids have been added
        // to the cutset because of **this** relaxation.
        let added_to_cutset = self.cutset.len();
        // Maintain the frontier cutset
        for node in squash.iter() {
            if node.flags.is_exact() {
                let nid = NodeId(self.nodes.len());
                self.nodes.push(node.clone());
                self.cutset.push(nid);
            }
        }

        // .. make a default node
        let mut merged_node = NodeData {
            state      : Rc::clone(&merged),
            flags      : NodeFlags::new_relaxed(),
            from_top   : isize::min_value(),
            from_bot   : isize::min_value(),
            inbound    : None,
            best_edge  : None,
        };

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

        // Add an edge connecting all nodes that have been added to the frontier
        // cutset to the merged node
        let cutset_len = self.cutset.len();
        for cutsetpos in added_to_cutset..cutset_len {
            let nid = self.cutset[cutsetpos];
            let eid = self.new_edge(nid, 0, None);
            self.edges[eid.0].next = merged_node.inbound;
            merged_node.inbound = Some(eid);
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
            best_edge: None
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

        assert_eq!(16,  v[&'c']);
        assert_eq!(104, v[&'d']);
    }

}
