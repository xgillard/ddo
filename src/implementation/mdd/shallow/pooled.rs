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

//! This module provides the implementation of a _pooled_ MDD. This is a kind of
//! bounded width MDD which offers a real guarantee wrt to the maximum amount
//! of used memory.

use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::sync::Arc;

use metrohash::MetroHashMap;

use crate::abstraction::heuristics::SelectableNode;
use crate::abstraction::mdd::{Config, MDD};
use crate::common::{Decision, FrontierNode, Solution, Variable, VarSet};
use crate::common::PartialAssignment::SingleExtension;
use crate::implementation::mdd::MDDType;
use crate::implementation::mdd::utils::NodeFlags;
use crate::implementation::mdd::shallow::utils::{Node, Edge};

// --- POOLED MDD --------------------------------------------------------------
/// This structure implements a _pooled_ MDD. This is a kind of bounded width
/// MDD which cannot offer a strong guarantees wrt to the maximum amount of used
/// memory. However, this structure is perfectly suited for problems like MISP,
/// where one decision can affect more than one variable, and expanding a node
/// is expensive but checkinig if a node is going to be impacted by a decision
/// is cheap.
///
/// # Note
/// The behavior of this MDD is heavily dependent on the configuration you
/// provide. Therefore, and although a public constructor exists for this
/// structure, it it recommended that you build this type of mdd using the
/// `mdd_builder` functionality as shown in the following examples.
///
/// ## Example
/// ```
/// # use ddo::common::{Variable, Domain, VarSet, Decision};
/// # use ddo::abstraction::dp::{Problem, Relaxation};
/// # use ddo::implementation::heuristics::FixedWidth;
/// # use ddo::implementation::mdd::config::mdd_builder;
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> isize { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         unimplemented!()
/// #     }
/// #     fn transition(&self, _: &usize, _: &VarSet, _: Decision) -> usize {
/// #         unimplemented!()
/// #     }
/// #     fn transition_cost(&self, _: &usize, _: &VarSet, _: Decision) -> isize {
/// #         unimplemented!()
/// #     }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #   fn merge_states(&self, _: &mut dyn Iterator<Item=&usize>) -> usize {
/// #       unimplemented!()
/// #   }
/// #   fn relax_edge(&self, _: &usize, _: &usize, _: &usize, _: Decision, _: isize) -> isize {
/// #       unimplemented!()
/// #   }
/// # }
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// // Following line configure and builds a pooled mdd.
/// let pooled_mdd = mdd_builder(&problem, relaxation).into_pooled();
///
/// // Naturally, you can also provide configuration parameters to customize
/// // the behavior of your MDD. For instance, you can use a custom max width
/// // heuristic as follows (below, a fixed width)
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let pooled_mdd = mdd_builder(&problem, relaxation)
///                 .with_max_width(FixedWidth(100))
///                 .into_pooled();
/// ```
#[derive(Debug, Clone)]
pub struct PooledMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// This is the configuration used to parameterize the behavior of this
    /// MDD. Even though the internal state (free variables) of the configuration
    /// is subject to change, the configuration itself is immutable over time.
    config: C,

    // -- The following fields characterize the current unrolling of the MDD. --
    /// This is the kind of unrolling that was requested. It determines if this
    /// mdd must be an `Exact`, `Restricted` or `Relaxed` MDD.
    mddtype: MDDType,
    /// This is the pool of candidate nodes that might possibly participate in
    /// the next layer.
    pool: MetroHashMap<Arc<T>, Node<T>>,
    /// This set of nodes comprises all nodes that belong to a
    /// _frontier cutset_ (FC).
    cutset: Vec<Node<T>>,

    /// A flag indicating whether this mdd is exact
    is_exact: bool,

    /// This is the best known lower bound at the time of the MDD unrolling.
    /// This field is set once before developing mdd.
    best_lb: isize,
    /// This is the maximum width allowed for a layer of the MDD. It is determined
    /// once at the beginning of the MDD derivation.
    max_width: usize,
    /// This field memoizes the best node of the MDD. That is, the node of this
    /// mdd having the longest path from root.
    best_node: Option<Node<T>>
}
/// As the name suggests, `PooledMDD` is an implementation of the `MDD` trait.
/// See the trait definiton for the documentation related to these methods.
impl <T, C> MDD<T, C> for PooledMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn config(&self) -> &C {
        &self.config
    }

    fn is_exact(&self) -> bool {
        self.is_exact
            || self.mddtype == MDDType::Relaxed && self.best_node.as_ref().map(|n| n.has_exact_best()).unwrap_or(true)
    }

    fn best_value(&self) -> isize {
        if let Some(node) = &self.best_node {
            node.value
        } else {
            isize::min_value()
        }
    }

    fn best_solution(&self) -> Option<Solution> {
        self.best_node.as_ref().map(|n| Solution::new(n.path()))
    }

    fn for_each_cutset_node<F>(&self, mut func: F) where F: FnMut(FrontierNode<T>) {
        if !self.is_exact {
            let ub = self.best_value();
            if ub > self.best_lb {
                self.cutset.iter().for_each(|n| {
                    let mut frontier_node = FrontierNode::from(n);
                    frontier_node.ub = ub.min(frontier_node.ub);
                    (func)(frontier_node);
                });
            }
        }
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Exact;
        self.max_width= usize::max_value();

        self.develop(root, free_vars, best_lb);
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Restricted;
        self.max_width= self.config.max_width(&free_vars);

        self.develop(root, free_vars, best_lb);
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Relaxed;
        self.max_width= self.config.max_width(&free_vars);

        self.develop(root, free_vars, best_lb);
    }
}
impl <T, C> PooledMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(config: C) -> Self {
        PooledMDD {
            config,
            mddtype  : MDDType::Exact,
            pool     : Default::default(),
            cutset   : vec![],
            is_exact : true,
            best_lb  : isize::min_value(),
            max_width: usize::max_value(),
            best_node: None
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    pub fn clear(&mut self) {
        self.mddtype   = MDDType::Exact;
        self.pool.clear();
        self.cutset.clear();
        self.is_exact  = true;
        self.best_node = None;
        self.best_lb   = isize::min_value();
    }
    /// Develops/Unrolls the requested type of MDD, starting from a given root
    /// node. It only considers nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`) and assigns a value to the variables of the specified
    /// VarSet (`vars`).
    fn develop(&mut self, root: &FrontierNode<T>, mut vars: VarSet, best_lb: isize) {
        let root     = Node::from(root);
        self.best_lb = best_lb;
        self.pool.insert(Arc::clone(&root.this_state), root);

        let mut current = vec![];
        while let Some(var) = self.next_var(&vars, &current) {
            self.add_layer(var, &mut current);
            vars.remove(var);

            // squash layer if needed
            match self.mddtype {
                MDDType::Exact => {},
                MDDType::Restricted =>
                    if current.len() > self.max_width {
                        self.restrict_last(&mut current);
                    },
                MDDType::Relaxed =>
                    if current.len() > self.max_width {
                        self.relax_last(&mut current);
                    }
            }

            for node in current.iter() {
                let src_state = node.this_state.as_ref();
                for val in self.config.domain_of(src_state, var) {
                    let decision = Decision { variable: var, value: val };
                    let state    = self.config.transition(src_state, &vars, decision);
                    let weight   = self.config.transition_cost(src_state, &vars, decision);

                    self.branch(node, state, decision, weight)
                }
            }
        }

        self.finalize()
    }
    /// Returns the next variable to branch on (according to the configured
    /// branching heuristic) or None if all variables have been assigned a value.
    fn next_var(&self, vars: &VarSet, current: &[Node<T>]) -> Option<Variable> {
        let mut curr_it = current.iter().map(|n| n.state());
        let mut next_it = self.pool.keys().map(|k| k.as_ref());

        self.config.select_var(vars, &mut curr_it, &mut next_it)
    }
    /// Adds one layer to the mdd and move to it.
    /// In practice, this amounts to selecting the relevant nodes from the pool,
    /// adding them to the current layer and removing them from the pool.
    fn add_layer(&mut self, var: Variable, current: &mut Vec<Node<T>>) {
        current.clear();

        // Add all selected nodes to the next layer
        for (s, n) in self.pool.iter() {
            if self.config.impacted_by(s, var) {
                current.push(n.clone());
            }
        }

        // Remove all nodes that belong to the current layer from the pool
        for node in current.iter() {
            self.pool.remove(&node.this_state);
        }
    }
    /// This method records the branching from the given `node` with the given
    /// `decision`. It creates a fresh node for the `dest` state (or reuses one
    /// if `dest` already belongs to the current layer) and draws an edge of
    /// of the given `weight` between `orig_id` and the new node.
    ///
    /// ### Note:
    /// In case where this branching would create a new longest path to an
    /// already existing node, the length and best parent of the pre-existing
    /// node are updated.
    fn branch(&mut self, node: &Node<T>, dest: T, decision: Decision, weight: isize) {
        let dst_node = Node {
            this_state: Arc::new(dest),
            path: Arc::new(SingleExtension { parent: Arc::clone(&node.path), decision }),
            value: node.value.saturating_add(weight),
            estimate: isize::max_value(),
            flags: node.flags, // if its inexact, it will be or relaxed it will be considered inexact or relaxed too
            best_edge: Some(Edge {
                parent_state: Arc::clone(&node.this_state),
                weight,
                decision
            })
        };
        self.add_node(dst_node)
    }

    /// Inserts the given node in the next layer or updates it if needed.
    fn add_node(&mut self, mut node: Node<T>) {
        match self.pool.entry(Arc::clone(&node.this_state)) {
            Entry::Vacant(re) => {
                node.estimate = self.config.estimate(node.state());
                if node.ub() > self.best_lb {
                    re.insert(node);
                }
            },
            Entry::Occupied(mut re) => {
                let old = re.get_mut();
                if old.is_exact() && !node.is_exact() {
                    self.cutset.push(old.clone());
                } else if node.is_exact() && !old.is_exact() {
                    self.cutset.push(node.clone());
                }
                old.merge(node);
            }
        }
    }

    /// This method restricts the last layer (the current layer !) to make sure
    /// it fits within the maximum "width" size.
    ///
    /// # Note
    /// The removed nodes and all their inbound edges are irremediably lost.
    ///
    /// # Warning
    /// It is your responsibility to make sure the layer is broad enough to be
    /// relaxed. Failing to do so will not panic... but in the future, it might!
    fn restrict_last(&mut self, current: &mut Vec<Node<T>>) {
        self.is_exact = false;

        current.sort_unstable_by(|a, b| self.config.compare(a, b).reverse());
        current.truncate(self.max_width);
    }
    /// This method relaxes the last layer (the current layer !) to make sure
    /// it fits within the maximum "width" size. The overdue nodes are merged
    /// according to the configured strategy.
    ///
    /// # Note
    /// The removed nodes and all their inbound edges are irremediably lost.
    ///
    /// # Warning
    /// It is your responsibility to make sure the layer is broad enough to be
    /// relaxed. Failing to do so would result in a panic!
    ///
    /// # Warning
    /// This function will panic if you request a relaxation that would leave
    /// zero nodes in the current layer.
    fn relax_last(&mut self, current: &mut Vec<Node<T>>) {
        self.is_exact = false;

        current.sort_unstable_by(|a, b| self.config.compare(a, b).reverse());

        let (_keep, squash) = current.split_at_mut(self.max_width - 1);
        for node in squash.iter() {
            if node.is_exact() {
                self.cutset.push(node.clone());
            }
        }

        let merged_state = self.config.merge_states(&mut squash.iter().map(|n| n.state()));
        let mut merged_path  = Arc::clone(&squash[0].path);
        let mut merged_value = isize::min_value();
        let mut merged_edge  = None;

        for node in squash {
            let best_edge    = node.best_edge.as_ref().unwrap();
            let parent_value = node.value.saturating_sub(best_edge.weight);
            let src          = best_edge.parent_state.as_ref();
            let dst          = node.this_state.as_ref();
            let decision     = best_edge.decision;
            let cost         = best_edge.weight;
            let relax_cost   = self.config.relax_edge(src, dst, &merged_state, decision, cost);

            if parent_value.saturating_add(relax_cost) > merged_value {
                merged_value = parent_value.saturating_add(relax_cost);
                merged_path  = Arc::clone(&node.path);
                merged_edge  = Some(Edge{
                    parent_state: Arc::clone(&best_edge.parent_state),
                    weight      : relax_cost,
                    decision
                });
            }
        }

        let rlx_node = Node {
            this_state: Arc::new(merged_state),
            path      : merged_path,
            value     : merged_value,
            estimate: isize::max_value(),
            flags     : NodeFlags::new_relaxed(),
            best_edge : merged_edge
        };

        current.truncate(self.max_width - 1);
        self.add_relaxed(rlx_node, current)
    }
    /// Finds another node in the current layer having the same state as the
    /// given `state` parameter (if there is one such node).
    fn find_same_state<'b>(state: &T, current: &'b mut[Node<T>]) -> Option<&'b mut Node<T>> {
        for n in current.iter_mut() {
            if n.state().eq(state) {
                return Some(n);
            }
        }
        None
    }
    /// Adds a relaxed node into the given current layer.
    fn add_relaxed(&mut self, mut node: Node<T>, into_current: &mut Vec<Node<T>>) {
        if let Some(old) = Self::find_same_state(node.state(), into_current) {
            if old.is_exact() {
                //trace!("squash:: there existed an equivalent");
                self.cutset.push(old.clone());
            }
            old.merge(node);
        } else {
            node.estimate = self.config.estimate(node.state());
            if node.ub() > self.best_lb {
                into_current.push(node);
            }
        }
    }
    /// Finalizes the computation of the MDD: it identifies the best terminal node.
    fn finalize(&mut self) {
        self.best_node = self.pool.values()
            .max_by_key(|n| n.value)
            .cloned();
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################


#[cfg(test)]
mod test_pooledmdd {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::mdd::{MDD, Config};
    use crate::common::{Decision, Domain, FrontierNode, PartialAssignment, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::mdd::MDDType;
    use crate::implementation::mdd::shallow::pooled::PooledMDD;
    use crate::test_utils::MockConfig;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd = PooledMDD::new(config);

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
        let mut mdd = PooledMDD::new(config);

        mdd.relaxed(&root_n, 0);
        assert_eq!(MDDType::Relaxed, mdd.mddtype);

        mdd.restricted(&root_n, 0);
        assert_eq!(MDDType::Restricted, mdd.mddtype);

        mdd.exact(&root_n, 0);
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

    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_pooled();

        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
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
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_pooled();

        let root = mdd.config().root_node();

        mdd.restricted(&root, 0);
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
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_pooled();

        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);
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
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_pooled();

        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_pooled();
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_populates_frontier_cutset() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 0);

        let mut cut = vec![];
        mdd.for_each_cutset_node(|n| cut.push(*n.state.as_ref()));

        cut.sort_unstable();
        assert_eq!(vec![0, 1, 2], cut);
    }

    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_pooled();
        let root = mdd.config().root_node();
        mdd.restricted(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.config().root_node();

        mdd.restricted(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }

    #[derive(Clone, Copy)]
    struct DummyInfeasibleProblem;

    impl Problem<usize> for DummyInfeasibleProblem {
        fn nb_vars(&self) -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize { 0 }
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
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(isize::min_value(), mdd.best_value())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root = mdd.config().root_node();

        mdd.exact(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root = mdd.config().root_node();

        mdd.restricted(&root, 100);
        assert!(mdd.best_solution().is_none())
    }
}


#[cfg(test)]
mod test_private {
    use std::cmp::Ordering;
    use std::hash::Hash;
    use std::sync::Arc;

    use mock_it::verify;

    use crate::abstraction::heuristics::SelectableNode;
    use crate::abstraction::mdd::{Config, MDD};
    use crate::common::{Decision, PartialAssignment, Variable};
    use crate::implementation::mdd::shallow::utils::Node;
    use crate::implementation::mdd::shallow::pooled::PooledMDD;
    use crate::test_utils::{MockConfig, Proxy};

    #[test]
    fn branch_inserts_a_node_with_given_state_when_none_exists() {
        let config  = MockConfig::default();
        let mut mdd = PooledMDD::new(config);
        let node    = Node {
            this_state: Arc::new(42),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 42,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        };

        let mut current = vec![];
        mdd.add_node(node);
        mdd.add_layer(Variable(0), &mut current);

        let src = current.iter().cloned().find(|n| 42 == *n.state()).unwrap();
        let dst = 1;
        let dec = Decision{variable: Variable(9), value: 6};
        let wt  = 1;

        assert_eq!(0, mdd.pool.len());
        mdd.branch(&src, dst, dec, wt);
        assert_eq!(1, mdd.pool.len());
        assert!(mdd.pool.get(&Arc::new(1)).is_some());
    }
    #[test]
    fn branch_wont_update_existing_node_to_remember_last_decision_and_path_if_it_doesnt_improve_value() {
        let config  = MockConfig::default();
        let mut mdd = PooledMDD::new(config);
        let node    = Node {
            this_state: Arc::new(42),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 42,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        };
        let mut current = vec![];
        mdd.add_node(node);
        mdd.add_layer(Variable(0), &mut current);

        mdd.add_node(Node {
            this_state: Arc::new(1),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 100,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        });

        let src = current.iter().cloned().find(|n| 42 == *n.state()).unwrap();
        let dst = 1;
        let dec = Decision{variable: Variable(9), value: 6};
        let wt  = 1;

        mdd.branch(&src, dst, dec, wt);
        assert!(mdd.pool[&Arc::new(1)].best_edge.is_none());
        assert_eq!(100, mdd.pool[&Arc::new(1)].value);
    }
    #[test]
    fn branch_updates_existing_node_to_remember_last_decision_and_path_if_it_improves_value() {
        let config  = MockConfig::default();
        let mut mdd = PooledMDD::new(config);
        let node    = Node {
            this_state: Arc::new(42),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 42,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        };
        let mut current = vec![];
        mdd.add_node(node);
        mdd.add_layer(Variable(0), &mut current);

        mdd.add_node(Node {
            this_state: Arc::new(1),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 1,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        });

        let src = current.iter().cloned().find(|n| 42 == *n.state()).unwrap();
        let dst = 1;
        let dec = Decision{variable: Variable(9), value: 6};
        let wt  = 1;

        mdd.branch(&src, dst, dec, wt);
        assert!(mdd.pool[&Arc::new(1)].best_edge.is_some());
        assert_eq!(43, mdd.pool[&Arc::new(1)].value);
    }


    macro_rules! get {
        ($curr: expr, $state: expr) => { &$curr.iter().cloned().find(|n| $state == *n.state()).unwrap()};
        (next $dd: expr, $state: expr) => { &$dd.pool[&Arc::new($state)]};
    }
    #[test]
    fn restrict_last_will_not_populate_cutset() { // because it is useless !
        let config  = MockConfig::default();
        let mut mdd = PooledMDD::new(config);
        add_root(&mut mdd, 33, 3);

        let mut current = vec![];
        mdd.add_layer(Variable(0), &mut current);
        let r_id = &current.iter().cloned().find(|n| 33 == *n.state()).unwrap();

        mdd.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert!(mdd.is_exact);

        // first time, a lel is saved
        mdd.max_width = 2;
        mdd.restrict_last(&mut current);
        assert!(!mdd.is_exact);
        assert_eq!(0, mdd.cutset.len());

        // not either after a subsequent restrict
        mdd.add_layer(Variable(1), &mut current);
        let r_id = get!(current, 34);
        mdd.branch(r_id, 37, Decision{variable: Variable(1), value: 1}, 3);
        mdd.branch(r_id, 38, Decision{variable: Variable(1), value: 2}, 2);
        mdd.branch(r_id, 39, Decision{variable: Variable(1), value: 3}, 1);
        mdd.branch(r_id, 40, Decision{variable: Variable(1), value: 1}, 3);
        mdd.branch(r_id, 41, Decision{variable: Variable(1), value: 2}, 2);
        mdd.branch(r_id, 42, Decision{variable: Variable(1), value: 3}, 1);
        mdd.restrict_last(&mut current);
        assert!(!mdd.is_exact);
        assert_eq!(0, mdd.cutset.len());
    }

    #[test]
    fn restrict_last_makes_the_graph_inexact() {
        let config  = MockConfig::default();
        let mut mdd = PooledMDD::new(config);
        add_root(&mut mdd, 33, 3);

        let mut current = vec![];
        mdd.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        mdd.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert!(mdd.is_exact());

        // first time, a lel is saved
        mdd.restrict_last(&mut current);
        assert!(!mdd.is_exact());
    }
    #[test]
    fn restrict_last_layer_enforces_the_max_width() {
        let config  = MockConfig::default();
        let mut mdd = PooledMDD::new(config);
        add_root(&mut mdd, 33, 3);

        let mut current = vec![];
        mdd.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        mdd.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, mdd.pool.len());

        mdd.max_width = 2;
        mdd.add_layer(Variable(0), &mut current);
        mdd.restrict_last(&mut current);
        assert_eq!(2, current.len());

        mdd.max_width = 1;
        mdd.restrict_last(&mut current);
        assert_eq!(1, current.len());
    }
    #[test]
    fn restrict_last_layer_forgets_the_state_of_deleted_nodes() {
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        let c = MockConfig::default();
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(c);
        g.max_width= 2;
        add_root(&mut g, 33, 3);

        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(1), &mut current);
        let mut states_before = current.iter().map(|k| *k.state()).collect::<Vec<usize>>();
        states_before.sort_unstable();
        assert_eq!(vec![34, 35, 36], states_before);

        g.restrict_last(&mut current);
        let mut states_after = current.iter().map(|k| *k.state()).collect::<Vec<usize>>();
        states_after.sort_unstable();
        assert_eq!(vec![35, 36], states_after);
    }
    #[test]
    fn restrict_last_layer_will_not_bring_about_new_nodes() {
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        let c = MockConfig::default();
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(c);
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);

        let r_id = get!(current, 33);
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.restrict_last(&mut current);
        assert_eq!(36, *get!(next g, 36).state());
        assert_eq!( 4,  get!(next g, 36).value());
        assert_eq!(35, *get!(next g, 35).state());
        assert_eq!( 5,  get!(next g, 35).value());
    }
    #[test]
    fn restrict_last_layer_uses_node_selection_heuristic_to_rank_nodes() {
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        let c = MockConfig::default();
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        // 1. check the appropriate heuristic is used
        g.add_layer(Variable(56), &mut current);
        let mut states_before = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states_before.sort_unstable();
        assert_eq!(vec![34, 35, 36], states_before);

        //
        g.restrict_last(&mut current);
        assert!(verify(c.compare.was_called_with((34, 35))));
        let mut states_after = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states_after.sort_unstable();
        assert_eq!(vec![35, 36], states_after);
    }
    #[test]
    fn relax_last_populates_the_cutset() {
        let c = MockConfig::default();
        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert!(g.is_exact);

        // first time, a lel is saved
        g.add_layer(Variable(1600), &mut current);
        g.relax_last(&mut current);
        assert!(!g.is_exact);
        assert_eq!(2, g.cutset.len()); // there are two parents to the merged node

        // and it is updated with all new nodes
        let r_id = get!(current, 35);
        g.branch(r_id, 37, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 38, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 39, Decision{variable: Variable(0), value: 3}, 1);
        g.branch(r_id, 40, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 41, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 42, Decision{variable: Variable(0), value: 3}, 1);
        g.add_layer(Variable(2), &mut current);
        g.relax_last(&mut current);
        assert!(!g.is_exact);
        assert_eq!(7, g.cutset.len());
    }
    #[test]
    fn relax_last_makes_the_graph_inexact() {
        let mut g  = PooledMDD::new(MockConfig::default());
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert!(g.is_exact());

        // first time, a lel is saved
        g.add_layer(Variable(12), &mut current);
        g.relax_last(&mut current);
        assert!(!g.is_exact());
    }

    #[test]
    fn relax_last_layer_enforces_the_given_max_width() {
        let c = MockConfig::default();
        // Merged state is 37 (does not exist prior to relaxation)
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        g.add_layer(Variable(6), &mut current);
        assert_eq!(3, current.len());

        g.relax_last(&mut current);
        let mut cur = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        cur.sort_unstable();
        assert_eq!(vec![36, 37], cur);
        assert_eq!(2, current.len());
    }
    #[test]
    fn relax_last_layer_can_produce_lesser_max_width_when_merged_state_already_exists() {
        let c = MockConfig::default();
        // Merged state is 36 (it already exists)
        c.merge_states.given(vec![35, 34]).will_return(36);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        g.add_layer(Variable(5), &mut current);
        assert_eq!(3, current.len());

        g.relax_last(&mut current);
        let cur = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        assert_eq!(vec![36], cur);
        assert_eq!(1, current.len());
    }
    #[test]
    fn test_relax_last_layer_when_width_is_one() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![34, 35, 36]).will_return(37);
        c.merge_states.given(vec![34, 36, 35]).will_return(37);
        c.merge_states.given(vec![35, 36, 34]).will_return(37);
        c.merge_states.given(vec![35, 34, 36]).will_return(37);
        c.merge_states.given(vec![36, 34, 35]).will_return(37);
        c.merge_states.given(vec![36, 35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 1;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        g.add_layer(Variable(5), &mut current);
        assert_eq!(3, current.len());

        g.relax_last(&mut current);
        let cur = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        assert_eq!(vec![37], cur);
        assert_eq!(1, current.len());
    }
    #[test]
    fn relax_last_layer_forgets_the_state_of_deleted_nodes() {
        let c = MockConfig::default();
        // Merged state is 37 (does not exist prior to relaxation)
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(5), &mut current);
        let mut states = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last(&mut current);
        let mut states = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![36, 37], states);
    }
    #[test]
    fn relax_last_remembers_the_state_of_merged_node() {
        let c = MockConfig::default();
        // Merged state is 37 (does not exist prior to relaxation)
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(5), &mut current);
        let mut states = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last(&mut current);
        let mut states = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![36, 37], states);
    }
    #[test]
    fn relax_last_remembers_the_state_of_merged_node_even_if_it_corresponds_to_that_of_one_that_was_deleted() {
        let c = MockConfig::default();
        // Merged state is 35 (selected for deletion)
        c.merge_states.given(vec![35, 34]).will_return(35);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(5), &mut current);
        let mut states = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last(&mut current);
        let mut states = current.iter().map(|n| *n.state()).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![35, 36], states);
    }
    #[test]
    fn relax_last_layer_will_bring_about_one_node() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(5), &mut current);
        g.relax_last(&mut current);
        // 36
        assert_eq!(36, *get!(current, 36).state());
        assert_eq!( 4,  get!(current, 36).value());

        // 37 (mock relaxes everything to 0)
        assert_eq!(37, *get!(current, 37).state());
        assert_eq!( 3,  get!(current, 37).value());
    }
    #[test]
    fn relax_last_layer_will_bring_about_one_node_unless_merged_state_is_already_known() {
        let c = MockConfig::default();
        // Merged state is 35 (selected for deletion)
        c.merge_states.given(vec![35, 34]).will_return(36);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(5), &mut current);
        g.relax_last(&mut current);
        // 36
        assert_eq!(36, *get!(current, 36).state());
        assert_eq!( 4,  get!(current, 36).value());
    }
    #[test]
    fn relax_last_layer_uses_node_selection_heuristic_to_rank_nodes() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(5), &mut current);
        g.relax_last(&mut current);
        assert!(verify(c.compare.was_called_with((34, 35))));
    }
    #[test]
    fn relax_last_relaxes_the_weight_of_all_redirected_best_edges() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // relax r-34
        c.relax_edge
            .given((33, 34, 37, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(1);
        // relax r-35
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(0);
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(0);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(5), &mut current);
        g.relax_last(&mut current);
        assert!(verify(c.relax_edge.was_called_with((33, 34, 37, Decision{variable: Variable(0), value: 1}, 3))));
        assert!(verify(c.relax_edge.was_called_with((33, 35, 37, Decision{variable: Variable(0), value: 4}, 4))));
    }
    #[test]
    fn relax_last_will_not_update_best_parent_and_value_when_there_is_no_improvement() {
        let c = MockConfig::default();
        // Merged state is 36
        c.merge_states.given(vec![35, 34]).will_return(36);
        // relax r-34
        c.relax_edge
            .given((33, 34, 36, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(0);
        // relax r-35
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(0);
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(0);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(4, get!(next g, 36).value);

        g.add_layer(Variable(5), &mut current);
        g.relax_last(&mut current);
        assert_eq!(4, get!(current, 36).value)
    }
    #[test]
    fn relax_last_updates_best_parent_and_value_if_merged_node_already_exists() {
        let c = MockConfig::default();
        // Merged state is 36
        c.merge_states.given(vec![35, 34]).will_return(36);
        // relax r-34
        c.relax_edge
            .given((33, 34, 36, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(3);
        // relax r-35
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(2);
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(17);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(33, *get!(next g, 36).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!( 4,  get!(next g, 36).value);

        g.add_layer(Variable(5), &mut current);
        g.relax_last(&mut current);
        assert_eq!(33, *get!(current, 36).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!(20,  get!(current, 36).value);
    }
    #[test]
    fn relax_last_updates_best_parent_and_value_relaxed_edge_improve_value() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // relax r-34
        c.relax_edge
            .given((33, 34, 37, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(6);
        // relax r-35
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(8);
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(10);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g  = PooledMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(33, *get!(next g, 36).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!( 4,  get!(next g, 36).value);

        g.add_layer(Variable(5), &mut current);
        g.relax_last(&mut current);
        assert_eq!(33, *get!(current, 37).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!(13,  get!(current, 37).value);

        assert_eq!(33, *get!(current, 36).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!( 4,  get!(current, 36).value);
    }
    #[test]
    #[should_panic]
    fn relax_last_panics_if_width_is_0() {
        let mut g  = PooledMDD::new(MockConfig::default());
        g.max_width= 0;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(666), &mut current);
        g.relax_last(&mut current);
    }
    #[test]
    #[should_panic]
    fn relax_last_panics_if_layer_is_not_broad_enough() {
        let mut g  = PooledMDD::new(MockConfig::default());
        g.max_width= 10;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(666), &mut current);
        g.relax_last(&mut current);
    }
    #[test]
    fn restrict_last_wont_panic_if_layer_is_not_broad_enough() {
        let mut g  = PooledMDD::new(MockConfig::default());
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        let mut current = vec![];
        g.add_layer(Variable(0), &mut current);
        let r_id = get!(current, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.add_layer(Variable(665), &mut current);
        g.restrict_last(&mut current);
    }

    fn add_root<T, C>(mdd : &mut PooledMDD<T, C>, s: T, v: isize)
        where T: Eq + Hash + Clone,
              C: Config<T> + Clone
    {
        mdd.add_node(Node{
            this_state: Arc::new(s),
            path: Arc::new(PartialAssignment::Empty),
            value: v,
            estimate: isize::max_value(),
            flags: Default::default(),
            best_edge: None
        })
    }
}