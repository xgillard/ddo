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

//! This module provides the implementation of the `DeepMDD` structure. This is
//! an MDD (implements the `MDD` trait)  which materializes the complete dd graph
//! (see `mddgraph`). It implements the rough upper bound and local bounds
//! techniques in order to strengthen the pruning of the global branch and bound
//! algorithm. In practice, the bulk of the work is accomplished by the
//! `implementation::mdd::deep::mddgraph::Graph` structure (but you should
//! probably not care about it) and `DeepMDD` is a convenient shim around it
//! that implements the local bounds and adapts the `Graph` structure to the
//! `MDD` trait. All in all, the most important method implemented by the
//! DeepMDD structure is `develop()`. This is the method which actually unrolls
//! the transistion and transition cost relations to develop an (approximate) mdd.

use std::hash::Hash;
use std::sync::Arc;

use crate::abstraction::mdd::{Config, MDD};
use crate::common::{Decision, FrontierNode, PartialAssignment, Solution, Variable, VarSet};
use crate::common::PartialAssignment::Empty;
use crate::implementation::mdd::deep::mddgraph::{Graph, LayerData, LayerIndex, NodeData, NodeIndex};
use crate::implementation::mdd::MDDType;

/// MiniNode is a private structure used to remember _some_ information about
/// the nodes from the current layer. The very reason for the existence of this
/// structure comes from the fact that one cannot use the iterator on the nodes
/// of the current layer (immutable borrow) while at the same time modifying
/// the graph (as is needed by a call to `branch` for instance).
/// Hence, the useful information is stored in these mini-nodes which are in
/// turn used to perform the calls to branch without causing any memory-related
/// trouble.
struct MiniNode<T> {
    id     : NodeIndex,
    state  : Arc<T>,
    lp_len : isize,
}

/// This structure provides an implementation of a deep mdd (one that materializes
/// the complete mdd graph). It implements rough upper bound and local bounds to
/// strengthen the pruning achieved by the branch and bound algorithm.
/// It uses the given configuration to develop the (approximate) MDD.
#[derive(Clone)]
pub struct DeepMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// Because the same `DeepMDD` structure can be used to develop an exact,
    /// restricted or relaxed MDD, we use the `mddtype` to remember the type
    /// of MDD we are currently developing or have developed.
    mddtype: MDDType,
    /// This is the actual representation of the MDD graph.
    graph: Graph<T>,
    /// This object holds the 'configuration' of the MDD: the problem, relaxation
    /// and heuristics to use.
    config: C,

    /// This is the maximum width allowed for a layer of the MDD. It is determined
    /// once at the beginning of the MDD derivation.
    max_width: usize,
    /// If present, this is a shared reference to the partial assignment describing
    /// the path between the exact root of the problem and the root of this
    /// (possibly approximate) sub-MDD.
    root: Option<Arc<PartialAssignment>>,
    /// If present, this is the index of the node (in the `graph`) of the best
    /// terminal node. See the documentation of
    /// `ddo::implementation::mdd::deep::mddgraph::Graph` for more details.
    best: Option<NodeIndex>,
    /// A flag indicating whether or not this MDD is 'exact'.
    is_exact: bool,
}

/// As the name suggests, `DeepMDD` is an implementation of the `MDD` trait.
/// See the trait definiton for the documentation related to these methods.
impl <T, C> MDD<T, C> for DeepMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn config(&self) -> &C {
        &self.config
    }

    fn exact(&mut self, node: &FrontierNode<T>, best_lb: isize) {
        self.clear();

        let init_state = Arc::clone(&node.state);
        let init_value = node.lp_len;
        let free_vars  = self.config.load_variables(&node);

        self.mddtype   = MDDType::Exact;
        self.root      = Some(Arc::clone(&node.path));
        self.max_width = usize::max_value();

        self.develop(init_state, init_value, free_vars, best_lb)
    }

    fn restricted(&mut self, node: &FrontierNode<T>, best_lb: isize) {
        self.clear();

        let init_state = Arc::clone(&node.state);
        let init_value = node.lp_len;
        let free_vars  = self.config.load_variables(&node);

        self.mddtype   = MDDType::Restricted;
        self.root      = Some(Arc::clone(&node.path));
        self.max_width = self.config.max_width(&free_vars);

        self.develop(init_state, init_value, free_vars, best_lb)
    }

    fn relaxed(&mut self, node: &FrontierNode<T>, best_lb: isize) {
        self.clear();

        let init_state = Arc::clone(&node.state);
        let init_value = node.lp_len;
        let free_vars  = self.config.load_variables(&node);

        self.mddtype   = MDDType::Relaxed;
        self.root      = Some(Arc::clone(&node.path));
        self.max_width = self.config.max_width(&free_vars);

        self.develop(init_state, init_value, free_vars, best_lb)
    }

    fn is_exact(&self) -> bool {
        self.is_exact
    }
    fn best_value(&self) -> isize {
        self.best.map_or(isize::min_value(), |nid| self.graph.nodes[nid.0].lp_from_top)
    }
    fn best_solution(&self) -> Option<Solution> {
        self.best.map(|index| {
            let node = &self.graph.nodes[index.0];
            Solution::new(Arc::new(self.best_partial_assignment_for(node.my_id)))
        })
    }

    fn for_each_cutset_node<F>(&self, mut func: F)
        where F: FnMut(FrontierNode<T>)
    {
        let lel = self.graph.lel.unwrap();
        let lel = self.graph.layers[lel.0];

        self.graph.nodes[lel.start..lel.end].iter()
            .filter(|node| node.is_feasible())
            .for_each(|node| func({
                let ub_bot = node.lp_from_top.saturating_add(node.lp_from_bot);
                let ub_est = node.lp_from_top.saturating_add(self.config.estimate(node.state.as_ref()));
                FrontierNode {
                    state: Arc::clone(&node.state),
                    lp_len: node.lp_from_top,
                    ub: ub_bot.min(ub_est),
                    path: Arc::new(self.best_partial_assignment_for(node.my_id))
                }
            }));
    }
}

/// Private functions
impl <T, C> DeepMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(conf: C) -> Self {
        DeepMDD {
            mddtype  : MDDType::Exact,
            config   : conf,
            graph    : Graph::new(),
            max_width: usize::max_value(),

            root     : None,
            best     : None,
            is_exact : true
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    fn clear(&mut self) {
        self.mddtype   = MDDType::Exact;
        self.max_width = usize::max_value();
        self.root      = None;
        self.best      = None;
        self.is_exact  = true;
        self.graph.clear();
    }

    /// Returns the best partial assignment leading to the node identified by
    /// the `node` index in the graph.
    fn best_partial_assignment_for(&self, node: NodeIndex) -> PartialAssignment {
        PartialAssignment::FragmentExtension {
            fragment: self.graph.longest_path(node),
            parent  : self.root_pa()
        }
    }
    /// Returns a shared reference to the partial assignment describing
    /// the path between the exact root of the problem and the root of this
    /// (possibly approximate) sub-MDD.
    fn root_pa(&self) -> Arc<PartialAssignment> {
        self.root.as_ref().map_or(Arc::new(Empty), |refto| Arc::clone(&refto))
    }
    /// Returns an iterator over the states of the nodes from the given layer
    /// index (when one is given). In case `None` is provided (before the root),
    /// the method returns an empty iterator.
    ///
    /// # Technical note
    /// The choice of using an option to pass the layer index might seem a little
    /// bit odd at first, but it actually simplifies the writing and clarifies
    /// the intent. Indeed, this method is used as a means to pass an iterator
    /// over layer nodes to the variable selection heuristic (which takes
    /// an iterator to the nodes of the current layer, and one to the nodes of
    /// the next layer). But, when we are at layer 0 (only the root is present
    /// in the graph), we cant pass any useful iterator. We could pass an
    /// iterator to the current layer and an empty one to the next layer. But
    /// if we chose to do so, it would mean that we would never be able to
    /// pass an useful iterator to the next layer. However, utilising the option
    /// allows us to use a dummy (empty) iterator when we are at layer 0 and
    /// no next layer has been expanded, and to return something useful as soon
    /// as a next layer has been unrolled.
    fn layer_states_iter(&self, layer: Option<LayerIndex>) -> impl Iterator<Item=&T> {
        let layer = if let Some(id) = layer {
            self.graph.layers[id.0]
        } else {
            LayerData {my_id: LayerIndex(0), start: 0, end: 0} // dummy empty layer (before the root)
        };
        let slice = &self.graph.nodes[layer.start..layer.end];
        slice.iter().map(|n| n.state.as_ref())
    }
    /// Returns the optional index of the "current layer" assuming that the next
    /// layer has already been developed. When no next layer has been developed,
    /// this method returns None.
    ///
    /// # Technical note
    /// The choice of using an option to return the layer index might seem a little
    /// bit odd at first, but it actually simplifies the writing and clarifies
    /// the intent. Indeed, this method is used as a means to pass an iterator
    /// over layer nodes to the variable selection heuristic (which takes
    /// an iterator to the nodes of the current layer, and one to the nodes of
    /// the next layer). But, when we are at layer 0 (only the root is present
    /// in the graph), we cant pass any useful iterator. We could pass an
    /// iterator to the current layer and an empty one to the next layer. But
    /// if we chose to do so, it would mean that we would never be able to
    /// pass an useful iterator to the next layer. However, utilising the option
    /// allows us to use a dummy (empty) iterator when we are at layer 0 and
    /// no next layer has been expanded, and to return something useful as soon
    /// as a next layer has been unrolled.
    fn current_layer_index(&self) -> Option<LayerIndex> {
        if self.graph.layers.len() >= 2 {
            Some(LayerIndex(self.graph.layers.len() - 2))
        } else {
            None
        }
    }
    /// Returns the optional index of the "next layer" assuming that the next
    /// layer has already been developed. When no next layer has been developed,
    /// this method returns None.
    ///
    /// # Technical note
    /// The choice of using an option to return the layer index might seem a little
    /// bit odd at first, but it actually simplifies the writing and clarifies
    /// the intent. Indeed, this method is used as a means to pass an iterator
    /// over layer nodes to the variable selection heuristic (which takes
    /// an iterator to the nodes of the current layer, and one to the nodes of
    /// the next layer). But, when we are at layer 0 (only the root is present
    /// in the graph), we cant pass any useful iterator. We could pass an
    /// iterator to the current layer and an empty one to the next layer. But
    /// if we chose to do so, it would mean that we would never be able to
    /// pass an useful iterator to the next layer. However, utilising the option
    /// allows us to use a dummy (empty) iterator when we are at layer 0 and
    /// no next layer has been expanded, and to return something useful as soon
    /// as a next layer has been unrolled.
    fn next_layer_index(&self) -> Option<LayerIndex> {
        Some(LayerIndex(self.graph.layers.len() - 1))
    }
    /// Returns the next variable to branch on (according to the configured
    /// branching heuristic) or None if all variables have been assigned a value.
    fn next_var(&self, vars: &VarSet) -> Option<Variable> {
        let curr    = self.current_layer_index();
        let next    = self.next_layer_index();
        let mut curr_it = self.layer_states_iter(curr);
        let mut next_it = self.layer_states_iter(next);

        self.config.select_var(vars, &mut curr_it, &mut next_it)
    }
    /// Develops/Unrolls the requested type of MDD, starting from a root node
    /// whose initial state (`init_state`) and value (`init_val`) are given.
    /// It only considers nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`) and assigns a value to the variables of the specified
    /// VarSet (`vars`).
    fn develop(&mut self, init_state: Arc<T>, init_val: isize, mut vars: VarSet, best_lb: isize) {
        self.graph.add_root(init_state, init_val);

        while let Some(var) = self.next_var(&vars) {
            let current = self.graph.current_layer();

            self.graph.add_layer();
            vars.remove(var);

            // unroll layer
            let states = self.graph.layer_nodes(current.my_id).iter()
                .map(|n| MiniNode {id: n.my_id, state: n.state_ref(), lp_len: n.lp_from_top})
                .collect::<Vec<MiniNode<T>>>();

            for keys in states {
                let src_state = keys.state.as_ref();

                let est = self.config.estimate(src_state);
                if keys.lp_len.saturating_add(est) > best_lb {
                    for val in self.config.domain_of(src_state, var) {
                        let decision = Decision { variable: var, value: val };
                        let state = self.config.transition(src_state, &vars, decision);
                        let cost = self.config.transition_cost(src_state, &vars, decision);
                        self.graph.branch(keys.id, state, decision, cost);
                    }
                }
            }

            // squash layer if needed
            match self.mddtype {
                MDDType::Exact => {},
                MDDType::Restricted => if self.graph.current_layer().width() > self.max_width {
                    let w = self.max_width;
                    let c = &self.config;
                    let g = &mut self.graph;
                    g.restrict_last(w, c);
                },
                MDDType::Relaxed => {
                    let current_layer = self.graph.current_layer();
                    if current_layer.my_id.0 > 1 && current_layer.width() > self.max_width {
                        let w = self.max_width;
                        let c = &self.config;
                        let g = &mut self.graph;
                        g.relax_last(w, c);
                    }
                }
            }
        }

        self.finalize()
    }
    /// Finalizes the computation of the MDD: it identifies the best terminal
    /// node, checks if the MDD is exact and computes the local bounds of the
    /// cutset nodes.
    fn finalize(&mut self) {
        self.best = self.graph.find_best_terminal_node();
        self.compute_is_exact();

        if self.mddtype == MDDType::Relaxed {
            self.compute_local_bounds()
        }
    }
    /// Checks if the mdd is exact or if the best terminal node has an exact
    /// best path from the root.
    fn compute_is_exact(&mut self) {
        self.is_exact = self.graph.is_exact()
            || (self.mddtype == MDDType::Relaxed && self.graph.has_exact_best_path(self.best))
    }
    /// Computes the local bounds of the cutset nodes.
    fn compute_local_bounds(&mut self) {
        if !self.is_exact { // if it's exact, there is nothing to be done
            let lel = self.graph.lel.unwrap();

            let mut layer = self.graph.current_layer();
            // all the nodes from the last layer have a lp_from_bot of 0
            for node in self.graph.nodes[layer.start..layer.end].iter_mut() {
                node.lp_from_bot = 0;
                node.set_feasible(true);
            }

            while layer.my_id.0 > lel.0 {
                for node in self.graph.nodes[layer.start..layer.end].iter() {
                    if node.is_feasible() {
                        let mut inbound = node.inbound;
                        while let Some(edge_id) = inbound {
                            let edge = self.graph.edges[edge_id.0];

                            let lp_from_bot_using_edge = node.lp_from_bot.saturating_add(edge.state.weight);
                            let parent = unsafe {
                                let ptr = &self.graph.nodes[edge.state.src.0] as *const NodeData<T> as *mut NodeData<T>;
                                &mut *ptr
                            };

                            parent.lp_from_bot = parent.lp_from_bot.max(lp_from_bot_using_edge);
                            parent.set_feasible(true);

                            inbound = edge.next;
                        }
                    }
                }

                layer = self.graph.layers[layer.my_id.0 - 1];
            }
        }
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################


#[cfg(test)]
mod test_deepmdd {
    use std::cmp::Ordering;
    use std::sync::Arc;

    use metrohash::MetroHashMap;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::heuristics::{NodeSelectionHeuristic, SelectableNode};
    use crate::abstraction::mdd::{Config, MDD};
    use crate::common::{Decision, Domain, FrontierNode, PartialAssignment, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::mdd::deep::mdd::DeepMDD;
    use crate::implementation::mdd::MDDType;
    use crate::test_utils::MockConfig;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd    = DeepMDD::new(config);

        assert_eq!(MDDType::Exact, mdd.mddtype);
    }
    #[test]
    fn mdd_type_changes_depending_on_the_requested_type_of_mdd() {
        let root_n  = FrontierNode {
            state : Arc::new(0),
            lp_len: 0,
            ub    : 24,
            path  : Arc::new(PartialAssignment::Empty)};

        let config  = MockConfig::default();
        let mut mdd = DeepMDD::new(config);

        mdd.relaxed(&root_n, 0);
        assert_eq!(MDDType::Relaxed, mdd.mddtype);

        mdd.restricted(&root_n, 0);
        assert_eq!(MDDType::Restricted, mdd.mddtype);

        mdd.exact(&root_n, 0);
        assert_eq!(MDDType::Exact, mdd.mddtype);
    }
    #[test]
    fn root_remembers_the_pa_from_the_frontier_node() {
        let root_n  = FrontierNode {
            state : Arc::new(0),
            lp_len: 0,
            ub    : 24,
            path  : Arc::new(PartialAssignment::Empty)};

        let config  = MockConfig::default();
        let mut mdd = DeepMDD::new(config);

        mdd.exact(&root_n, 0);
        assert!(mdd.root.is_some());
        assert!(std::ptr::eq(root_n.path.as_ref(), mdd.root.as_ref().unwrap().as_ref()));

        mdd.restricted(&root_n, 0);
        assert!(mdd.root.is_some());
        assert!(std::ptr::eq(root_n.path.as_ref(), mdd.root.as_ref().unwrap().as_ref()));

        mdd.relaxed(&root_n, 0);
        assert!(mdd.root.is_some());
        assert!(std::ptr::eq(root_n.path.as_ref(), mdd.root.as_ref().unwrap().as_ref()));
    }

    #[derive(Copy, Clone)]
    struct DummyProblem;
    impl Problem<usize> for DummyProblem {
        fn nb_vars(&self)       -> usize { 3 }
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
    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_deep();

        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision{variable: Variable(2), value: 2},
                       Decision{variable: Variable(1), value: 2},
                       Decision{variable: Variable(0), value: 2},
                   ]
        );
    }
    #[test]
    fn restricted_drops_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_deep();

        let root = mdd.config().root_node();

        mdd.restricted(&root, 0);
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision{variable: Variable(2), value: 2},
                       Decision{variable: Variable(1), value: 2},
                       Decision{variable: Variable(0), value: 2},
                   ]
        );
    }
    #[test]
    fn relaxed_merges_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_deep();

        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 42);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision{variable: Variable(2), value: 2},
                       Decision{variable: Variable(1), value: 2},
                       Decision{variable: Variable(0), value: 2},
                   ]
        );
    }
    #[test]
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_deep();

        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);

        let mut cutset = vec![];
        mdd.for_each_cutset_node(|n| cutset.push(n));
        assert_eq!(cutset.len(), 3); // L1 was not squashed even though it was 3 wide
    }
    #[test]
    fn an_exact_mdd_must_be_exact() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_deep();

        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_deep();
        let root    = mdd.config().root_node();

        mdd.relaxed(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_deep();
        let root    = mdd.config().root_node();

        mdd.relaxed(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_deep();
        let root    = mdd.config().root_node();
        mdd.restricted(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_deep();
        let root    = mdd.config().root_node();

        mdd.restricted(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }

    #[derive(Clone,Copy)]
    struct DummyInfeasibleProblem;
    impl Problem<usize> for DummyInfeasibleProblem {
        fn nb_vars(&self)       -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize   { 0 }
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
        let pb      = DummyInfeasibleProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_deep();
        let root    = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb      = DummyInfeasibleProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_deep();
        let root    = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(isize::min_value(), mdd.best_value())
    }
    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_deep();
        let root    = mdd.config().root_node();

        mdd.exact(&root, 100);
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_deep();
        let root    = mdd.config().root_node();

        mdd.relaxed(&root, 100);
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_deep();
        let root    = mdd.config().root_node();

        mdd.restricted(&root, 100);
        assert!(mdd.best_solution().is_none())
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
        let mut mdd = mdd_builder(&LocBoundsExamplePb, LocBoundExampleRelax)
            .with_nodes_selection_heuristic(CmpChar)
            .with_max_width(FixedWidth(3))
            .into_deep();

        let root = mdd.config.root_node();
        mdd.relaxed(&root, 0);

        assert_eq!(false, mdd.is_exact());
        assert_eq!(104,   mdd.best_value());

        let mut v = MetroHashMap::default();
        mdd.for_each_cutset_node(|n| {v.insert(*n.state, n.ub);});

        assert_eq!(16,  v[&'a']);
        assert_eq!(104, v[&'b']);
    }
}