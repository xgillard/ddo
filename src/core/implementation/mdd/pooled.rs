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
//! bounded width MDD which cannot offer a strong guarantees wrt to the maximum
//! amount of used memory. However, this structure is perfectly suited for
//! problems like MISP, where one decision can affect more than one variable,
//! and expanding a node is expensive but checkinig if a node is going to be
//! impacted by a decision is cheap.
use std::cmp::min;
use std::hash::Hash;
use std::sync::Arc;

use metrohash::MetroHashMap;

use crate::core::abstraction::mdd::{MDD, MDDType};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};
use crate::core::common::{Bounds, Decision, Layer, Node, NodeInfo, Variable};
use crate::core::implementation::mdd::config::Config;

// --- POOLED MDD --------------------------------------------------------------
#[derive(Clone)]
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
/// # use ddo::core::implementation::mdd::builder::mdd_builder;
/// # use ddo::core::implementation::heuristics::FixedWidth;
/// use ddo::core::abstraction::dp::{Problem, Relaxation};
/// use ddo::core::common::{Variable, Domain, VarSet, Decision, Node};
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> i32   { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         unimplemented!()
/// #     }
/// #     fn transition(&self, _: &usize, _: &VarSet, _: Decision) -> usize {
/// #         unimplemented!()
/// #     }
/// #     fn transition_cost(&self, _: &usize, _: &VarSet, _: Decision) -> i32 {
/// #         unimplemented!()
/// #     }
/// # }
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #     fn merge_nodes(&self, _: &[Node<usize>]) -> Node<usize> {
/// #         unimplemented!()
/// #     }
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
pub struct PooledMDD<T, C> where T: Eq + Hash + Clone, C: Config<T> {
    /// This is the configuration used to parameterize the behavior of this
    /// MDD. Even though the internal state (free variables) of the configuration
    /// is subject to change, the configuration itself is immutable over time.
    config           : C,

    // -- The following fields characterize the current unrolling of the MDD. --
    /// This is the kind of unrolling that was requested. It determines if this
    /// mdd must be an `Exact`, `Restricted` or `Relaxed` MDD.
    mddtype          : MDDType,
    /// This is the pool of candidate nodes that might possibly participate in
    /// the next layer.
    pool             : MetroHashMap<T, NodeInfo>,
    /// This is the set of nodes constituting the current layer
    current          : Vec<Node<T>>,
    /// This set of nodes comprises all nodes that belong to a
    /// _frontier cutset_ (FC).
    cutset           : Vec<Node<T>>,

    // -- The following are transient fields -----------------------------------
    /// This flag indicates whether or not this MDD is an exact MDD (that is,
    /// it tells if no relaxation/restriction has occurred yet).
    is_exact         : bool,
    /// This field memoizes the best node of the MDD. That is, the node of this
    /// mdd having the longest path from root.
    best_node        : Option<NodeInfo>
}

/// PooledMDD implements the MDD abstract data type. Check its documentation
/// for further details.
impl <T, C> MDD<T> for PooledMDD<T, C> where T: Eq + Hash + Clone, C: Config<T> {
    fn mdd_type(&self) -> MDDType {
        self.mddtype
    }
    fn root(&self) -> Node<T> {
        self.config.root_node()
    }
    fn exact(&mut self, root: &Node<T>, best_lb: i32) {
        self.develop(Exact, root, best_lb);
    }
    fn restricted(&mut self, root: &Node<T>, best_lb: i32) {
        self.develop(Restricted, root, best_lb);
    }
    fn relaxed(&mut self, root: &Node<T>, best_lb: i32) {
        self.develop(Relaxed, root, best_lb);
    }
    fn for_each_cutset_node<F>(&mut self, mut f: F) where F: FnMut(&T, &mut NodeInfo) {
        self.cutset.iter_mut().for_each(|n| (f)(&n.state, &mut n.info))
    }
    fn consume_cutset<F>(&mut self, mut f: F) where F: FnMut(T, NodeInfo) {
        self.cutset.drain(..).for_each(|n| (f)(n.state, n.info))
    }
    fn is_exact(&self) -> bool {
        self.is_exact
    }
    fn best_value(&self) -> i32 {
        if self.best_node.is_none() {
            std::i32::MIN
        } else {
            self.best_node.as_ref().unwrap().lp_len
        }
    }
    fn best_node(&self) -> &Option<NodeInfo> {
        &self.best_node
    }
    fn longest_path(&self) -> Vec<Decision> {
        if self.best_node.is_none() {
            vec![]
        } else {
            self.best_node.as_ref().unwrap().longest_path()
        }
    }
}

/// Private functions
impl <T, C> PooledMDD<T, C> where T: Eq + Hash + Clone, C: Config<T> {
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(config: C) -> Self {
        PooledMDD {
            config,

            mddtype          : Exact,
            is_exact         : true,
            best_node        : None,
            pool             : Default::default(),
            current          : vec![],
            cutset           : vec![]
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    fn clear(&mut self) {
        self.mddtype          = Exact;
        self.is_exact         = true;
        self.best_node        = None;
        // unassigned vars holds stale data !

        self.pool             .clear();
        self.current          .clear();
        self.cutset           .clear();
    }
    /// Develops/Unrolls the requested type of MDD, starting from the given `root`
    /// and considering only nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`).
    fn develop(&mut self, kind: MDDType, root: &Node<T>, best_lb : i32) {
        self.init(kind, root);
        let w = if self.mddtype == Exact { usize::max_value() } else { self.config.max_width() };

        let bounds = Bounds {lb: best_lb, ub: root.info.ub};
        let nbvars = self.config.nb_free_vars();

        let mut i  = 0;
        while i < nbvars && !self.exhausted() {
            let var = self.config.select_var(self.it_current(), self.it_next());
            if var.is_none() {
                break;
            }

            let var = var.unwrap();
            self.pick_nodes_from_pool(var);
            self.maybe_squash(i, w);
            self.config.remove_var(var);
            self.unroll_layer(var, bounds);
            i += 1;
        }

        self.finalize()
    }
    /// Unrolls the current layer by making all possible decisions about the
    /// given variable `var` from all the nodes of the current layer. Only nodes
    /// having an estimated upper bound greater than the best known lower bound
    /// will be considered relevant and make their way to the next layer.
    ///
    /// # Note
    /// This type of MDD is a _reduced_ MDD. Here, the reduction rule is only
    /// applied top-down, and nodes are merged iff they have the exact same
    /// state.
    fn unroll_layer(&mut self, var: Variable, bounds: Bounds) {
        for node in self.current.iter() {
            let info    = Arc::new(node.info.clone());
            let domain  = self.config.domain_of(&node.state, var);
            for value in domain {
                let decision  = Decision{variable: var, value};
                let branching = self.config.branch(&node.state, Arc::clone(&info), decision);

                if let Some(old) = self.pool.get_mut(&branching.state) {
                    if old.is_exact && !branching.info.is_exact {
                        //trace!("main loop:: old was exact but new was not");
                        self.cutset.push(Node{state: branching.state, info: old.clone()});
                    } else if !old.is_exact && branching.info.is_exact {
                        //trace!("main loop:: new was exact but old was not");
                        self.cutset.push(branching.clone());
                    }
                    old.merge(branching.info)
                } else if self.is_relevant(bounds, &branching.state, &branching.info) {
                    self.pool.insert(branching.state, branching.info);
                }
            }
        }
    }
    /// Iterates over all nodes in the pool and selects all those that are
    /// possibly impacted by a decision on variable `var`. These nodes are
    /// then removed from the pool and effectively constitute the new
    /// `current` layer.
    fn pick_nodes_from_pool(&mut self, var: Variable) {
        self.current.clear();

        // Add all selected nodes to the next layer
        let mut items = vec![];
        for (s, _i) in self.pool.iter() {
            if self.config.impacted_by(s, var) {
                items.push(s.clone());
            }
        }

        // Remove all nodes that belong to the current layer from the pool
        for state in items {
            let info = self.pool.remove(&state).unwrap();
            self.current.push(Node{state, info});
        }
    }
    /// Returns true iff the problem space has been exhausted. That is, if
    /// all nodes have been removed from the pool (and hence no possible
    /// successor exists).
    fn exhausted(&self) -> bool {
        self.pool.is_empty()
    }
    /// Returns true iff the a node made of `state` and `info` would be relevant
    /// considering the given bounds. A node is considered to be relevant iff
    /// its estimated upper bound (rough upper bound) is strictly greater than
    /// the current lower bound.
    fn is_relevant(&self, bounds: Bounds, state: &T, info: &NodeInfo) -> bool {
        min(self.config.estimate_ub(state, info), bounds.ub) > bounds.lb
    }
    /// Takes all necessary actions to start the development of an MDD.
    /// Concretely, this means:
    ///   - clearing stale data,
    ///   - loading the set of free variables from the given root node,
    ///   - setting the type of mdd to develop, and
    ///   - inserting the given root node in the first layer of this mdd.
    fn init(&mut self, kind: MDDType, root: &Node<T>) {
        self.clear();
        self.config.load_vars(root);
        self.mddtype         = kind;

        self.pool.insert(root.state.clone(), root.info.clone());
    }
    /// Takes the necessary actions to finalize the processing of an MDD rooted
    /// in a given sub-problem. Concretely, it identifies the best terminal node
    /// and, if such a node exists, it sets a tight upper bound on the nodes
    /// from the cutset. Otherwise, it empties the cutset since it would make
    /// no sense to try to use that cutset.
    fn finalize(&mut self) {
        self.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.best_node {
            let lp_length = best.lp_len;

            for n in self.cutset.iter_mut() {
                n.info.ub = lp_length.min(self.config.estimate_ub(&n.state, &n.info));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            self.cutset.clear();
        }
    }
    /// Iterates over all nodes from the terminal layer and identifies the
    /// nodes having the longest path from the root.
    fn find_best_node(&mut self) {
        let mut best_value = std::i32::MIN;
        for info in self.pool.values() {
            if info.lp_len > best_value {
                best_value = info.lp_len;
                self.best_node = Some(info.clone());
            }
        }
    }
    /// Possibly takes actions (if layer > 1) to shrink the size of the current
    /// layer in case its width exceeds the limit.
    fn maybe_squash(&mut self, i : usize, w: usize) {
        match self.mddtype {
            MDDType::Exact      => /* nothing to do ! */(),
            MDDType::Restricted => self.maybe_restrict(i, w),
            MDDType::Relaxed    => self.maybe_relax(i, w),
        }
    }
    /// Performs a restriction of the current layer if its width exceeds the
    /// maximum limit. In other words, it drops the worst nodes of the current
    /// layer to make its width fit within the maximum size determined by the
    /// configuration.
    fn maybe_restrict(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let config = &self.config;
            while self.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;
                self.current.sort_unstable_by(|a, b| config.compare(a, b).reverse());
                self.current.truncate(w);
            }
        }
    }
    /// Performs a relaxation of the current layer if its width exceeds the
    /// maximum limit. In other words, it merges the worst nodes of the current
    /// layer to make its width fit within the maximum size determined by the
    /// configuration.
    ///
    /// # Note
    /// The role of this method is both to:  merge the nodes
    /// (delegated to `merge_overdue_nodes`) and to maintain the state of the
    /// cutset.
    ///
    fn maybe_relax(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            while self.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;

                // actually squash the layer
                let merged = self.merge_overdue_nodes(w);

                if let Some(old) = Self::find_same_state(&mut self.current, &merged.state) {
                    if old.info.is_exact {
                        //trace!("squash:: there existed an equivalent");
                        self.cutset.push(old.clone());
                    }
                    old.info.merge(merged.info);
                } else {
                    self.current.push(merged);
                }
            }
        }
    }
    /// This method effectively merges the n wost nodes and retunrs a new
    /// (inexact) relaxed one.
    fn merge_overdue_nodes(&mut self, w: usize) -> Node<T> {
        // 1. Sort the current layer so that the worst nodes are at the end.
        let config = &self.config;
        self.current.sort_unstable_by(|a, b| config.compare(a, b).reverse());
        let (_keep, squash) = self.current.split_at(w-1);

        // 2. merge the nodes
        let merged = self.config.merge_nodes(squash);

        // 3. make sure to keep the cutset complete
        for n in squash {
            if n.info.is_exact {
                self.cutset.push(n.clone())
            }
        }

        // 4. drop overdue nodes
        self.current.truncate(w - 1);
        merged
    }
    /// Finds another node in the current layer having the same state as the
    /// given `state` parameter (if there is one such node).
    fn find_same_state<'b>(current: &'b mut[Node<T>], state: &T) -> Option<&'b mut Node<T>> {
        for n in current.iter_mut() {
            if n.state.eq(state) {
                return Some(n);
            }
        }
        None
    }
    /// Returns a `Layer` iterator over the nodes of the current layer.
    fn it_current(&self) -> Layer<'_, T> {
        Layer::Plain(self.current.iter())
    }
    /// Returns a `Layer` iterator over all the nodes of the *pool*.
    fn it_next(&self) -> Layer<'_, T> {
        Layer::Mapped(self.pool.iter())
    }
}



#[cfg(test)]
mod test_mdd {
    use mock_it::verify;
    use crate::core::abstraction::dp::{Problem, Relaxation};
    use crate::core::abstraction::mdd::{MDD, MDDType};
    use crate::core::common::{Decision, Domain, Node, NodeInfo, Variable, VarSet};
    use crate::core::implementation::mdd::builder::mdd_builder;
    use crate::core::implementation::mdd::pooled::PooledMDD;
    use crate::test_utils::{MockConfig, Nothing, ProxyMut};
    use crate::core::implementation::heuristics::FixedWidth;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let mut config = MockConfig::default();
        let mdd        = PooledMDD::new(ProxyMut::new(&mut config));

        assert_eq!(MDDType::Exact, mdd.mdd_type());
    }
    #[test]
    fn mdd_type_changes_depending_on_the_requested_type_of_mdd() {
        let mut config  = MockConfig::default();
        let mut mdd     = PooledMDD::new(ProxyMut::new(&mut config));
        let root        = mdd.root();

        mdd.relaxed(&root, 0);
        assert_eq!(MDDType::Relaxed, mdd.mdd_type());

        mdd.restricted(&root, 0);
        assert_eq!(MDDType::Restricted, mdd.mdd_type());

        mdd.exact(&root, 0);
        assert_eq!(MDDType::Exact, mdd.mdd_type());
    }
    #[test]
    fn root_node_is_pass_through_to_config() {
        let mut config = MockConfig::default();
        let mdd        = PooledMDD::new(ProxyMut::new(&mut config));
        let _          = mdd.root();

        assert!(verify(config.root_node.was_called_with(Nothing)))
    }



    struct DummyProblem;
    impl Problem<usize> for DummyProblem {
        fn nb_vars(&self)       -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> i32   { 0 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..=2).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> i32 {
            d.value
        }
    }
    struct DummyRelax;
    impl Relaxation<usize> for DummyRelax {
        fn merge_nodes(&self, _: &[Node<usize>]) -> Node<usize> {
            Node{ state: 100, info: NodeInfo { is_exact: false, lp_len: 20, lp_arc: None, ub: 50}}
        }
        fn estimate_ub(&self, _state: &usize, _info: &NodeInfo) -> i32 {
            50
        }
    }

    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.exact(&root, 0);
        assert!(mdd.best_node().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.longest_path(),
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
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.restricted(&root, 0);
        assert!(mdd.best_node().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.longest_path(),
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
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.relaxed(&root, 0);
        assert!(mdd.best_node().is_some());
        // Value is 22 in this case, because, as opposd to the flat mdd,
        // the current layer is developed *after* the merge has occurred. Thus,
        // nodes from the current layer can generate terminal successors with
        // a transition. In this case, the best possible transition to generate
        // is 2.
        assert_eq!(mdd.best_value(), 22);
        // The rest is lost in the dummy relaxation. But [[ V2 <- 2 ]] remains
        // for the same reason as stated above.
        assert_eq!(mdd.longest_path(), vec![Decision { variable: Variable(2), value: 2 }]);
    }
    #[test]
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.relaxed(&root, 0);

        let mut cutset = vec![];
        mdd.consume_cutset(|s, i| cutset.push(Node { state: s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state
        assert!(cutset.iter().all(|n| n.info.is_exact));
    }
    #[test]
    fn foreach_cutset_node_iterates_over_cutset() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.relaxed(&root, 0);

        let mut cutset = vec![];
        mdd.for_each_cutset_node(|s, i| cutset.push(Node { state: *s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state

        cutset.clear();
        mdd.for_each_cutset_node(|s, i| cutset.push(Node { state: *s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state

        cutset.clear();
        mdd.for_each_cutset_node(|s, i| cutset.push(Node { state: *s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state

        cutset.clear();
        mdd.for_each_cutset_node(|s, i| cutset.push(Node { state: *s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state

        cutset.clear();
        mdd.for_each_cutset_node(|s, i| cutset.push(Node { state: *s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state

        cutset.clear();
        mdd.for_each_cutset_node(|s, i| cutset.push(Node { state: *s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state
    }
    #[test]
    fn consume_cutset_clears_the_cutset() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.relaxed(&root, 0);

        let mut cutset = vec![];
        mdd.consume_cutset(|s, i| cutset.push(Node { state: s, info: i.clone() }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state
        assert!(cutset.iter().all(|n| n.info.is_exact));

        cutset.clear();
        mdd.consume_cutset(|s, i| cutset.push(Node { state: s, info: i.clone() }));
        assert_eq!(cutset.len(), 0); // because both 1,1 and (0,2) yield same state
    }

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_pooled();
        let root = mdd.root();

        mdd.relaxed(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.relaxed(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_pooled();
        let root = mdd.root();

        mdd.restricted(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_pooled();
        let root = mdd.root();

        mdd.restricted(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }

    struct DummyInfeasibleProblem;
    impl Problem<usize> for DummyInfeasibleProblem {
        fn nb_vars(&self)       -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> i32   { 0 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..0).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> i32 {
            d.value
        }
    }
    #[test]
    fn when_the_problem_is_infeasible_there_is_no_best_node() {
        let pb      = DummyInfeasibleProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root    = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(None, mdd.best_node().clone())
    }
    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb      = DummyInfeasibleProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root    = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(i32::min_value(), mdd.best_value())
    }
    #[test]
    fn when_the_problem_is_infeasible_the_longest_path_is_empty() {
        let pb      = DummyInfeasibleProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root    = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(Vec::<Decision>::new(), mdd.longest_path())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root    = mdd.root();

        mdd.exact(&root, 100);
        assert!(mdd.best_node().is_none())
    }
    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root    = mdd.root();

        mdd.relaxed(&root, 100);
        assert!(mdd.best_node().is_none())
    }
    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_pooled();
        let root    = mdd.root();

        mdd.restricted(&root, 100);
        assert!(mdd.best_node().is_none())
    }
}