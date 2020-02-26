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

//! This module provides the implementation of a _flat_ MDD. This is a kind of
//! bounded width MDD which offers a real guarantee wrt to the maximum amount
//! of used memory. This should be your go-to implementation of a bounded-width
//! MDD, and it is the default type of MDD built by the `mdd_builder`.
use std::cmp::min;
use std::hash::Hash;
use std::sync::Arc;

use metrohash::MetroHashMap;

use crate::core::abstraction::mdd::{MDD, MDDType};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};
use crate::core::common::{Bounds, Decision, Layer, Node, NodeInfo, Variable};
use crate::core::implementation::mdd::config::Config;

// --- MDD Data Structure -----------------------------------------------------
#[derive(Clone)]
/// This is the structure implementing _flat MDD_. This is a kind of
/// bounded width MDD which offers a real guarantee wrt to the maximum amount
/// of used memory. This should be your go-to implementation of a bounded-width
/// MDD, and it is the default type of MDD built by the `mdd_builder`.
///
/// A flat mdd is highly efficient in the sense that it only maintains one
/// 'slice' of the current MDD unrolling. That is, at any time, it only knows
/// the current layer, and the next layer (being developed). All previous layers
/// are irremediably forgotten (but this causes absolutely no harm). Alongside
/// the current and next layers, this structure also knows of a 3rd layer which
/// materializes the last exact layer (exact cutset) of this mdd. Moving from
/// one layer to the next (after the next layer has been expanded) is extremely
/// inexpensive as it only amounts to swapping two integer indices.
///
/// # Remark
/// So far, the `FlatMDD` implementation only supports the *last exact layer*
/// (LEL) kind of exact cutset. This might change in the future, but it is your
/// only option at the time being since it keeps the code clean and simple.
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
/// // Following line configure and builds a flat mdd.
/// let flat_mdd   = mdd_builder(&problem, relaxation).build();
///
/// // ... or equivalently (where you emphasize the use of a *flat* mdd)
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let flat_mdd   = mdd_builder(&problem, relaxation).into_flat();
///
/// // Naturally, you can also provide configuration parameters to customize
/// // the behavior of your MDD. For instance, you can use a custom max width
/// // heuristic as follows (below, a fixed width)
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let flat_mdd = mdd_builder(&problem, relaxation)
///                 .with_max_width(FixedWidth(100))
///                 .build();
/// ```
pub struct FlatMDD<T, C> where T: Hash + Eq + Clone, C: Config<T> {
    /// This is the configuration used to parameterize the behavior of this
    /// MDD. Even though the internal state (free variables) of the configuration
    /// is subject to change, the configuration itself is immutable over time.
    config: C,

    // -- The following fields characterize the current unrolling of the MDD. --
    /// This is the kind of unrolling that was requested. It determines if this
    /// mdd must be an `Exact`, `Restricted` or `Relaxed` MDD.
    mddtype: MDDType,
    /// This array stores the three layers known by the mdd: the current, next
    /// and last exact layer (lel). The position of each layer in the array is
    /// determined by the `current`, `next` and `lel` fields of the structure.
    layers: [MetroHashMap<T, NodeInfo>; 3],
    /// The index of the current layer in the array of `layers`.
    current: usize,
    /// The index of the next layer in the array of `layers`.
    next: usize,
    /// The index of the last exact layer (lel) in the array of `layers`
    lel: usize,

    // -- The following are transient fields -----------------------------------
    /// This flag indicates whether or not this MDD is an exact MDD (that is,
    /// it tells if no relaxation/restriction has occurred yet).
    is_exact: bool,
    /// This field memoizes the best node of the MDD. That is, the node of this
    /// mdd having the longest path from root.
    best_node: Option<NodeInfo>
}

/// Be careful: this macro lets you borrow any single layer from a flat mdd.
/// While this is generally safe, it is way too easy to use this macro to break
/// aliasing rules.
///
/// # Example
/// ```
/// # use ddo::core::implementation::mdd::builder::mdd_builder;
/// # use ddo::test_utils::{MockProblem, MockRelax};
/// # use ddo::core::implementation::heuristics::FixedWidth;
///
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let mut mdd    = mdd_builder(&problem, relaxation).build();
///
/// // This gets you an immutable borrow to the mdd's next layer.
/// let next_l = layer![&mdd, next];
///
/// // This gets you a mutable borrow of the mdd's current layer.
/// let curr_l = layer![&mut mdd, current];
/// ```
macro_rules! layer {
    ($dd:expr, $id:ident) => {
        unsafe { &*$dd.layers.as_ptr().add($dd.$id) }
    };
    ($dd:expr, mut $id:ident) => {
        unsafe { &mut *$dd.layers.as_mut_ptr().add($dd.$id) }
    };
}

/// FlatMDD implements the MDD abstract data type. Check its documentation
/// for further details.
impl <T, C> MDD<T> for FlatMDD<T, C> where T: Hash + Eq + Clone, C: Config<T> {
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
        layer![self, mut lel].iter_mut().for_each(|(k, v)| (f)(k, v))
    }
    fn consume_cutset<F>(&mut self, mut f: F) where F: FnMut(T, NodeInfo) {
        layer![self, mut lel].drain().for_each(|(k, v)| (f)(k, v))
    }

    fn is_exact(&self) -> bool {
        self.is_exact
    }
    fn best_value(&self) -> i32 {
        if let Some(n) = &self.best_node {
            n.lp_len
        } else {
            i32::min_value()
        }
    }
    fn best_node(&self) -> &Option<NodeInfo> {
        &self.best_node
    }
    fn longest_path(&self) -> Vec<Decision> {
        if let Some(n) = &self.best_node {
            n.longest_path()
        } else {
            vec![]
        }
    }
}

/// Private functions
impl <T, C> FlatMDD<T, C> where T: Hash + Eq + Clone, C: Config<T> {
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(config: C) -> Self {
        FlatMDD {
            config,

            mddtype          : Exact,
            current          : 0,
            next             : 1,
            lel              : 2,

            is_exact         : true,
            best_node        : None,
            layers           : [Default::default(), Default::default(), Default::default()]
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    fn clear(&mut self) {
        self.mddtype       = Exact;
        self.is_exact      = true;
        self.best_node     = None;
        // unassigned vars holds stale data !

        self.current       = 0;
        self.next          = 1;
        self.lel           = 2;

        self.layers.iter_mut().for_each(|l| l.clear());
    }
    /// Swaps the indices of the current and last exact layers, effectively
    /// remembering current as the last exact layer.
    fn swap_current_lel(&mut self) {
        let tmp      = self.current;
        self.current = self.lel;
        self.lel     = tmp;
    }
    /// Swaps the indices of the current and next layers effectively moving
    /// to the next layer (next is now considered current)
    fn swap_current_next(&mut self) {
        let tmp      = self.current;
        self.current = self.next;
        self.next    = tmp;
    }
    /// Returns true iff the a node made of `state` and `info` would be relevant
    /// considering the given bounds. A node is considered to be relevant iff
    /// its estimated upper bound (rough upper bound) is strictly greater than
    /// the current lower bound.
    fn is_relevant(&self, bounds: Bounds, state: &T, info: &NodeInfo) -> bool {
        min(self.config.estimate_ub(state, info), bounds.ub) > bounds.lb
    }
    /// Develops/Unrolls the requested type of MDD, starting from the given `root`
    /// and considering only nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`).
    fn develop(&mut self, kind: MDDType, root: &Node<T>, best_lb: i32) {
        self.init(kind, root);
        let w = if self.mddtype == Exact { usize::max_value() } else { self.config.max_width() };

        let bounds = Bounds {lb: best_lb, ub: root.info.ub};
        let nbvars = self.config.nb_free_vars();

        let mut i  = 0;
        while i < nbvars && !layer![self, current].is_empty() {
            let var = self.config.select_var(self.it_current(), self.it_next());
            if var.is_none() { break; }

            let was_exact = self.is_exact;
            let var = var.unwrap();
            self.config.remove_var(var);
            self.unroll_layer(var, bounds);
            self.maybe_squash(i, w); // next
            self.move_to_next(was_exact);

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
        let curr = layer![self,  current];
        let next = layer![self, mut next];

        for (state, info) in curr.iter() {
            let info   = Arc::new(info.clone());
            let domain = self.config.domain_of(state, var);
            for value in domain {
                let decision  = Decision{variable: var, value};
                let branching = self.config.branch(&state, Arc::clone(&info), decision);

                if let Some(old) = next.get_mut(&branching.state) {
                    old.merge(branching.info);
                } else if self.is_relevant(bounds, &branching.state, &branching.info) {
                    next.insert(branching.state, branching.info);
                }
            }
        }
    }
    /// Takes all necessary actions to effectively prepare the unrolling of the
    /// next layer. That is, it saves the last exact layer if needed, it swaps
    /// the current and next layers, and clears the content of the next layer to
    /// make it ready to accept new nodes.
    fn move_to_next(&mut self, was_exact: bool) {
        if self.is_exact != was_exact {
            self.swap_current_lel();
        }
        self.swap_current_next();
        layer![self, mut next].clear();
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
        self.mddtype = kind;

        layer![self, mut current].insert(root.state.clone(), root.info.clone());
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

            for (state, info) in layer![self, mut lel].iter_mut() {
                info.ub = lp_length.min(self.config.estimate_ub(state, info));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            layer![self, mut lel].clear();
        }
    }
    /// Iterates over all nodes from the terminal layer and identifies the
    /// nodes having the longest path from the root.
    fn find_best_node(&mut self) {
        let mut best_value = i32::min_value();
        for info in layer![self, current].values() {
            if info.lp_len > best_value {
                best_value         = info.lp_len;
                self.best_node  = Some(info.clone());
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
            let next   = layer![self, mut next];

            let mut nodes = vec![];
            next.drain().for_each(|(k,v)| nodes.push(Node{state: k, info: v}));

            while nodes.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;
                nodes.sort_unstable_by(|a, b| config.compare(a, b).reverse());
                nodes.truncate(w);
            }

            nodes.drain(..).for_each(|n| {next.insert(n.state, n.info);});
        }
    }
    /// Performs a relaxation of the current layer if its width exceeds the
    /// maximum limit. In other words, it merges the worst nodes of the current
    /// layer to make its width fit within the maximum size determined by the
    /// configuration.
    fn maybe_relax(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let config = &self.config;
            let next   = layer![self, mut next];

            if next.len() > w {

                let mut nodes = vec![];
                nodes.reserve_exact(next.len());

                next.drain().for_each(|(k,v)| {
                    nodes.push(Node{state: k, info: v});
                });

                nodes.sort_unstable_by(|a, b| config.compare(a, b).reverse());

                let (keep, squash) = nodes.split_at(w-1);

                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;

                // actually squash the layer
                let merged = self.config.merge_nodes(squash);

                for n in keep.to_vec().drain(..) {
                    next.insert(n.state, n.info);
                }

                if let Some(old) = next.get_mut(&merged.state) {
                    old.merge(merged.info);
                } else {
                    next.insert(merged.state, merged.info);
                }
            }
        }
    }
    /// Returns a `Layer` iterator over the nodes of the current layer.
    fn it_current(&self) -> Layer<'_, T> {
        Layer::Mapped(layer![self, current].iter())
    }
    /// Returns a `Layer` iterator over the nodes of the next layer.
    fn it_next(&self) -> Layer<'_, T> {
        Layer::Mapped(layer![self, next].iter())
    }
}

#[cfg(test)]
mod test_mdd {
    use mock_it::verify;
    use crate::core::abstraction::dp::{Problem, Relaxation};
    use crate::core::abstraction::mdd::{MDD, MDDType};
    use crate::core::common::{Decision, Domain, Node, NodeInfo, Variable, VarSet};
    use crate::core::implementation::mdd::builder::mdd_builder;
    use crate::core::implementation::mdd::flat::FlatMDD;
    use crate::test_utils::{MockConfig, Nothing, ProxyMut};
    use crate::core::implementation::heuristics::FixedWidth;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let mut config = MockConfig::default();
        let mdd        = FlatMDD::new(ProxyMut::new(&mut config));

        assert_eq!(MDDType::Exact, mdd.mdd_type());
    }
    #[test]
    fn mdd_type_changes_depending_on_the_requested_type_of_mdd() {
        let mut config  = MockConfig::default();
        let mut mdd     = FlatMDD::new(ProxyMut::new(&mut config));
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
        let mdd        = FlatMDD::new(ProxyMut::new(&mut config));
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
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
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
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
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
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let root = mdd.root();

        mdd.relaxed(&root, 0);
        assert!(mdd.best_node().is_some());
        assert_eq!(mdd.best_value(), 20);
        // lost in my dummy relaxation !
        assert_eq!(mdd.longest_path(), vec![ ]);
    }
    #[test]
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let root = mdd.root();

        mdd.relaxed(&root, 0);

        let mut cutset = vec![];
        mdd.consume_cutset(|s, i| cutset.push(Node { state: s, info: i }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state
        assert!(cutset.iter().all(|n| n.info.is_exact));
    }
    #[test]
    fn foreach_cutset_node_iterates_over_cutset() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
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
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let root = mdd.root();

        mdd.relaxed(&root, 0);

        let mut cutset = vec![];
        mdd.consume_cutset(|s, i| cutset.push(Node { state: s, info: i }));
        assert_eq!(cutset.len(), 5); // because both 1,1 and (0,2) yield same state
        assert!(cutset.iter().all(|n| n.info.is_exact));

        cutset.clear();
        mdd.consume_cutset(|s, i| cutset.push(Node { state: s, info: i }));
        assert_eq!(cutset.len(), 0); // because both 1,1 and (0,2) yield same state
    }

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let root = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let root = mdd.root();

        mdd.relaxed(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }
    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let root = mdd.root();

        mdd.relaxed(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let root = mdd.root();

        mdd.restricted(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
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
        let mut mdd = mdd_builder(&pb, rlx).build();
        let root    = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(None, mdd.best_node().clone())
    }
    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb      = DummyInfeasibleProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).build();
        let root    = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(i32::min_value(), mdd.best_value())
    }
    #[test]
    fn when_the_problem_is_infeasible_the_longest_path_is_empty() {
        let pb      = DummyInfeasibleProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).build();
        let root    = mdd.root();

        mdd.exact(&root, 0);
        assert_eq!(Vec::<Decision>::new(), mdd.longest_path())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).build();
        let root    = mdd.root();

        mdd.exact(&root, 100);
        assert!(mdd.best_node().is_none())
    }
    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).build();
        let root    = mdd.root();

        mdd.relaxed(&root, 100);
        assert!(mdd.best_node().is_none())
    }
    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).build();
        let root    = mdd.root();

        mdd.restricted(&root, 100);
        assert!(mdd.best_node().is_none())
    }
}

#[cfg(test)]
/// This module tests the private methods of the flatmdd data structure
mod test_private {
    use crate::test_utils::{MockConfig, ProxyMut};
    use crate::core::implementation::mdd::flat::FlatMDD;
    use crate::core::abstraction::mdd::{MDDType, MDD};
    use crate::core::common::{NodeInfo, Bounds, Node, Variable, Decision};
    use mock_it::verify;
    use std::sync::Arc;
    use std::cmp::Ordering;
    use crate::core::implementation::heuristics::MinLP;
    use compare::Compare;

    #[test]
    fn clear_prepares_the_mdd_for_reuse_hence_mddtype_must_be_exact() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        let root    = mdd.root();

        mdd.restricted(&root, 0);
        assert_eq!(MDDType::Restricted, mdd.mdd_type());

        mdd.clear();
        assert_eq!(MDDType::Exact, mdd.mdd_type());
    }
    #[test]
    fn clear_prepares_the_mdd_for_reuse_hence_indices_must_revert_to_default() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        mdd.lel = 0;
        mdd.next = 0;
        mdd.current = 0;

        mdd.clear();
        assert_eq!(0, mdd.current);
        assert_eq!(1, mdd.next);
        assert_eq!(2, mdd.lel);
    }
    #[test]
    fn clear_prepares_the_mdd_for_reuse_hence_exact_flag_must_be_true() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        mdd.is_exact = false;
        // resets
        mdd.clear();
        assert_eq!(true, mdd.is_exact);
        // does not screw it if alright
        mdd.clear();
        assert_eq!(true, mdd.is_exact);
    }
    #[test]
    fn clear_prepares_the_mdd_for_reuse_hence_cutset_must_be_empty() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        layer![mdd, mut lel].insert(42, NodeInfo{lp_len: 12, lp_arc: None, ub: 32, is_exact: false});

        mdd.clear();
        assert_eq!(true, layer![mdd, mut lel].is_empty());
    }
    #[test]
    fn clear_prepares_the_mdd_for_reuse_hence_all_layers_must_be_empty() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        layer![mdd, mut current].insert(1, NodeInfo{lp_len: 1, lp_arc: None, ub: 3, is_exact: false});
        layer![mdd, mut next   ].insert(2, NodeInfo{lp_len: 1, lp_arc: None, ub: 3, is_exact: false});
        layer![mdd, mut lel    ].insert(3, NodeInfo{lp_len: 2, lp_arc: None, ub: 3, is_exact: false});

        mdd.clear();
        assert_eq!(true, layer![mdd, current].is_empty());
        assert_eq!(true, layer![mdd, next   ].is_empty());
        assert_eq!(true, layer![mdd, lel    ].is_empty());
    }
    #[test]
    fn clear_prepares_the_mdd_for_reuse_hence_best_node_must_be_none() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        mdd.best_node = Some(NodeInfo{lp_len: 1, lp_arc: None, ub: 3, is_exact: false});

        mdd.clear();
        assert_eq!(None, mdd.best_node);
    }

    #[test]
    fn is_relevant_iff_estimate_is_better_than_lower_bound() {
        let state   = 42;
        let info    = NodeInfo{lp_len: 1, lp_arc: None, ub: 3, is_exact: false};

        let config = MockConfig::default();
        config.estimate_ub
            .given((state, info.clone()))
            .will_return(5);

        let mdd = FlatMDD::new(config);
        let bounds  = Bounds { lb: -10, ub: 10};
        assert_eq!(true, mdd.is_relevant(bounds, &state, &info));

        let bounds  = Bounds { lb: 6, ub: 10};
        assert_eq!(false, mdd.is_relevant(bounds, &state, &info));

        let bounds  = Bounds { lb: 5, ub: 10};
        assert_eq!(false, mdd.is_relevant(bounds, &state, &info));
    }


    #[test]
    fn swap_current_lel_exchanges_the_indices() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        let cur = mdd.current;
        let lel = mdd.lel;
        mdd.swap_current_lel();

        assert_eq!(cur, mdd.lel);
        assert_eq!(lel, mdd.current);
    }
    #[test]
    fn swap_current_next_exchanges_the_indices() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        let cur = mdd.current;
        let nxt = mdd.next;
        mdd.swap_current_next();

        assert_eq!(cur, mdd.next);
        assert_eq!(nxt, mdd.current);
    }

    // -------------------------------------------------------------------------
    // Tests for the `develop` method are skipped: these would duplicate
    // the set of tests for exact, restricted, and relaxed.
    // -------------------------------------------------------------------------

    #[test]
    fn unroll_layer_must_apply_decision_for_all_values_in_the_domain_of_var_to_all_nodes() {
        let v = Variable(1);

        let a = Node::new(0, 0, None, true);
        let b = Node::new(1, 1, None, true);
        let c = Node::new(2, 2, None, true);

        let mut config = MockConfig::default();
        config.domain_of.given((0, v)).will_return(vec![0, 1]);
        config.domain_of.given((1, v)).will_return(vec![2, 3]);
        config.domain_of.given((2, v)).will_return(vec![4, 5]);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));

        layer![mdd, mut current].insert(a.state, a.info.clone());
        layer![mdd, mut current].insert(b.state, b.info.clone());
        layer![mdd, mut current].insert(c.state, c.info.clone());

        mdd.unroll_layer(v, Bounds{lb: -1000, ub: 1000});

        assert_eq!(false, layer![mdd, next].is_empty());

        let a_info = Arc::new(a.info);
        assert!(verify(config.branch.was_called_with((0, a_info.clone(), Decision{variable: v, value: 0}))));
        assert!(verify(config.branch.was_called_with((0, a_info        , Decision{variable: v, value: 1}))));

        let b_info = Arc::new(b.info);
        assert!(verify(config.branch.was_called_with((1, b_info.clone(), Decision{variable: v, value: 2}))));
        assert!(verify(config.branch.was_called_with((1, b_info        , Decision{variable: v, value: 3}))));

        let c_info = Arc::new(c.info);
        assert!(verify(config.branch.was_called_with((2, c_info.clone(), Decision{variable: v, value: 4}))));
        assert!(verify(config.branch.was_called_with((2, c_info        , Decision{variable: v, value: 5}))));
    }
    #[test]
    fn when_the_domain_is_empty_next_layer_is_empty_and_problem_becomes_unsat() {
        let v = Variable(1);

        let a = Node::new(0, 0, None, true);
        let b = Node::new(1, 1, None, true);
        let c = Node::new(2, 2, None, true);

        let mut config = MockConfig::default();
        config.domain_of.given((0, v)).will_return(vec![]);
        config.domain_of.given((1, v)).will_return(vec![]);
        config.domain_of.given((2, v)).will_return(vec![]);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));

        layer![mdd, mut current].insert(a.state, a.info);
        layer![mdd, mut current].insert(b.state, b.info);
        layer![mdd, mut current].insert(c.state, c.info);

        mdd.unroll_layer(v, Bounds{lb: -1000, ub: 1000});
        assert_eq!(true, layer![mdd, next].is_empty());
    }
    #[test]
    fn when_unroll_generates_two_nodes_with_the_same_state_only_one_is_kept() {
        let v = Variable(1);

        let a = Node::new(0, 0, None, true);
        let b = Node::new(1, 1, None, true);
        let c = Node::new(1, 2, None, false);

        let a_info = Arc::new(a.info.clone());

        let mut config = MockConfig::default();
        config.domain_of.given((0, v)).will_return(vec![0, 1]);
        config.branch
            .given((0, a_info.clone(), Decision{variable: v, value: 0}))
            .will_return(b);
        config.branch
            .given((0, a_info, Decision{variable: v, value: 1}))
            .will_return(c);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut current].insert(a.state, a.info);

        mdd.unroll_layer(v, Bounds{lb: -1000, ub: 1000});
        assert_eq!(1, layer![mdd, next].len());
    }
    #[test]
    fn when_unroll_generates_two_nodes_with_the_same_state_nodes_are_merged() {
        let v = Variable(1);

        let a = Node::new(0, 0, None, true);
        let b = Node::new(1, 1, None, true);
        let c = Node::new(1, 2, None, false);

        let a_info = Arc::new(a.info.clone());

        let mut config = MockConfig::default();
        config.domain_of.given((0, v)).will_return(vec![0, 1]);
        config.branch
            .given((0, a_info.clone(), Decision{variable: v, value: 0}))
            .will_return(b);
        config.branch
            .given((0, a_info, Decision{variable: v, value: 1}))
            .will_return(c.clone());

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut current].insert(a.state, a.info);

        mdd.unroll_layer(v, Bounds{lb: -1000, ub: 1000});
        assert_eq!(Some(&c.info), layer![mdd, next].get(&1));
    }
    #[test]
    fn unroll_layer_saves_all_distinct_target_nodes_in_the_next_layer() {
        let v = Variable(1);

        let a = Node::new( 0,  0, None, true);
        let b = Node::new( 1,  1, None, true);
        let c = Node::new( 2,  2, None, true);

        let d = Node::new( 0,  0, None, true);
        let e = Node::new( 2,  2, None, true);
        let f = Node::new( 4,  4, None, true);
        let g = Node::new( 8,  8, None, true);
        let h = Node::new(16, 16, None, true);
        let i = Node::new(32, 32, None, true);

        let a_info = Arc::new(a.info.clone());
        let b_info = Arc::new(b.info.clone());
        let c_info = Arc::new(c.info.clone());

        let mut config = MockConfig::default();
        config.domain_of.given((0, v)).will_return(vec![0, 1]);
        config.domain_of.given((1, v)).will_return(vec![2, 3]);
        config.domain_of.given((2, v)).will_return(vec![4, 5]);

        config.branch
            .given((0, a_info.clone(), Decision{variable: v, value: 0}))
            .will_return(d.clone());
        config.branch
            .given((0, a_info, Decision{variable: v, value: 1}))
            .will_return(e.clone());
        config.branch
            .given((1, b_info.clone(), Decision{variable: v, value: 2}))
            .will_return(f.clone());
        config.branch
            .given((1, b_info, Decision{variable: v, value: 3}))
            .will_return(g.clone());
        config.branch
            .given((2, c_info.clone(), Decision{variable: v, value: 4}))
            .will_return(h.clone());
        config.branch
            .given((2, c_info, Decision{variable: v, value: 5}))
            .will_return(i.clone());


        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));

        layer![mdd, mut current].insert(a.state, a.info);
        layer![mdd, mut current].insert(b.state, b.info);
        layer![mdd, mut current].insert(c.state, c.info);

        mdd.unroll_layer(v, Bounds{lb: -1000, ub: 1000});
        let next = layer![mdd, next];
        assert_eq!(6, next.len());
        assert_eq!(Some(&d.info), next.get(&d.state));
        assert_eq!(Some(&e.info), next.get(&e.state));
        assert_eq!(Some(&f.info), next.get(&f.state));
        assert_eq!(Some(&g.info), next.get(&g.state));
        assert_eq!(Some(&h.info), next.get(&h.state));
        assert_eq!(Some(&i.info), next.get(&i.state));
    }
    #[test]
    fn unroll_layer_saves_a_node_to_next_layer_iff_it_is_relevant() {
        let v = Variable(1);

        let a = Node::new( 0,  0, None, true);
        let b = Node::new( 1,  1, None, true);
        let c = Node::new( 2,  2, None, true);

        let d = Node::new( 0,  0, None, true);
        let e = Node::new( 2,  2, None, true);
        let f = Node::new( 4,  4, None, true);
        let g = Node::new( 8,  8, None, true);
        let h = Node::new(16, 16, None, true);
        let i = Node::new(32, 32, None, true);

        let a_info = Arc::new(a.info.clone());
        let b_info = Arc::new(b.info.clone());
        let c_info = Arc::new(c.info.clone());

        let mut config = MockConfig::default();
        config.domain_of.given((0, v)).will_return(vec![0, 1]);
        config.domain_of.given((1, v)).will_return(vec![2, 3]);
        config.domain_of.given((2, v)).will_return(vec![4, 5]);

        config.branch
            .given((0, a_info.clone(), Decision{variable: v, value: 0}))
            .will_return(d.clone());
        config.branch
            .given((0, a_info, Decision{variable: v, value: 1}))
            .will_return(e.clone());
        config.branch
            .given((1, b_info.clone(), Decision{variable: v, value: 2}))
            .will_return(f.clone());
        config.branch
            .given((1, b_info, Decision{variable: v, value: 3}))
            .will_return(g.clone());
        config.branch
            .given((2, c_info.clone(), Decision{variable: v, value: 4}))
            .will_return(h.clone());
        config.branch
            .given((2, c_info, Decision{variable: v, value: 5}))
            .will_return(i.clone());

        config.estimate_ub.given((d.state, d.info        )).will_return(-10);
        config.estimate_ub.given((e.state, e.info        )).will_return(-10);
        config.estimate_ub.given((f.state, f.info        )).will_return(-10);
        config.estimate_ub.given((g.state, g.info        )).will_return(  0);
        config.estimate_ub.given((h.state, h.info.clone())).will_return( 10);
        config.estimate_ub.given((i.state, i.info.clone())).will_return(100);


        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));

        layer![mdd, mut current].insert(a.state, a.info);
        layer![mdd, mut current].insert(b.state, b.info);
        layer![mdd, mut current].insert(c.state, c.info);

        mdd.unroll_layer(v, Bounds{lb: 0, ub: 1000});
        let next = layer![mdd, next];
        assert_eq!(2, next.len());
        assert_eq!(Some(&h.info), next.get(&h.state));
        assert_eq!(Some(&i.info), next.get(&i.state));
    }

    #[test]
    fn move_to_next_remembers_last_exact_layer_when_the_exact_flag_has_changed(){
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        // it was true, it is still true ==> no need to save lel
        let old_lel  = mdd.lel;
        mdd.is_exact = true;
        mdd.move_to_next(true);
        assert_eq!(old_lel, mdd.lel);

        mdd.clear();

        // it was false, it is still false ==> no need to save lel
        let old_lel  = mdd.lel;
        mdd.is_exact = false;
        mdd.move_to_next(false);
        assert_eq!(old_lel, mdd.lel);

        mdd.clear();

        // it was true, it is now false ==> save lel
        let old_cur  = mdd.current;
        mdd.is_exact = false;
        mdd.move_to_next(true);
        assert_eq!(old_cur, mdd.lel);
    }
    #[test]
    fn move_to_next_ensures_next_layer_becomes_current() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        mdd.is_exact = true;
        mdd.move_to_next(true);
        assert_eq!(1, mdd.current);

        mdd.clear();

        mdd.is_exact = false;
        mdd.move_to_next(false);
        assert_eq!(1, mdd.current);

        mdd.clear();

        mdd.is_exact = false;
        mdd.move_to_next(true);
        assert_eq!(1, mdd.current);
    }
    #[test]
    fn move_to_next_clears_the_new_next_layer() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        let node = Node::new(1, 2, None, true);
        layer![mdd, mut current].insert(node.state, node.info);

        mdd.move_to_next(true);
        assert!(mdd.layers[0].is_empty());
        assert!(mdd.layers[1].is_empty());
        assert!(mdd.layers[2].is_empty());
    }


    #[test]
    fn init_clears_stale_data(){
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        mdd.current = 2;
        mdd.next    = 2;
        mdd.lel     = 2;
        mdd.is_exact= false;
        mdd.best_node= Some(Node::new(0, 0, None, true).info);

        mdd.init(MDDType::Relaxed, &Node::new(2, 3, None, true));

        assert_eq!(0, mdd.current);
        assert_eq!(1, mdd.next);
        assert_eq!(2, mdd.lel);
        assert_eq!(true, mdd.is_exact);
        assert_eq!(None, mdd.best_node);
    }
    #[test]
    fn init_loads_the_set_of_free_variables_from_the_given_node() {
        let mut config = MockConfig::default();
        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));

        let root = Node::new(2, 3, None, true);

        mdd.init(MDDType::Relaxed, &root);

        assert!(verify(config.load_vars.was_called_with(root)));
    }
    #[test]
    fn init_sets_the_appropriate_mdd_type() {
        let mut config = MockConfig::default();
        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));

        let root = Node::new(2, 3, None, true);

        mdd.init(MDDType::Relaxed, &root);
        assert_eq!(MDDType::Relaxed, mdd.mdd_type());

        mdd.clear();
        mdd.init(MDDType::Restricted, &root);
        assert_eq!(MDDType::Restricted, mdd.mdd_type());

        mdd.clear();
        mdd.init(MDDType::Exact, &root);
        assert_eq!(MDDType::Exact, mdd.mdd_type());
    }
    #[test]
    fn init_inserts_the_given_root_node_in_the_first_layer_of_the_mdd() {
        let mut config = MockConfig::default();
        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));

        let root = Node::new(2, 3, None, true);

        mdd.init(MDDType::Relaxed, &root);
        assert_eq!(1, layer![mdd, current].len());
        assert_eq!(Some(&root.info), layer![mdd, current].get(&root.state));
    }

    #[test]
    fn finalize_finds_the_best_node_if_there_is_one() {
        let mut config = MockConfig::default();
        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));

        // when there is no best node, it finds nothing
        mdd.finalize();
        assert_eq!(None, mdd.best_node);

        mdd.clear();
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);
        layer![mdd, mut current].insert(w.state, w.info);
        layer![mdd, mut current].insert(x.state, x.info);
        layer![mdd, mut current].insert(y.state, y.info.clone());
        layer![mdd, mut current].insert(z.state, z.info);
        mdd.finalize();

        assert_eq!(Some(y.info), mdd.best_node);
    }
    #[test]
    fn when_there_is_a_best_node_finalize_sets_an_upper_bound_on_the_cutset_nodes() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,1000, None, false);

        let mut config = MockConfig::default();
        config.estimate_ub.given((w.state, w.info.clone())).will_return(40);
        config.estimate_ub.given((x.state, x.info.clone())).will_return(30);
        config.estimate_ub.given((y.state, y.info.clone())).will_return(20);

        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut lel].insert(w.state, w.info.clone());
        layer![mdd, mut lel].insert(x.state, x.info.clone());
        layer![mdd, mut lel].insert(y.state, y.info.clone());
        layer![mdd, mut current].insert(z.state, z.info);
        mdd.finalize();

        assert_eq!(40, layer![mdd, lel].get(&w.state).unwrap().ub);
        assert_eq!(30, layer![mdd, lel].get(&x.state).unwrap().ub);
        assert_eq!(20, layer![mdd, lel].get(&y.state).unwrap().ub);
    }
    #[test]
    fn when_there_is_a_best_node_finalize_sets_an_upper_bound_on_the_cutset_nodes_and_the_bound_is_constrained_by_lp_len_of_mdd() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,   5, None, false);

        let mut config = MockConfig::default();
        config.estimate_ub.given((w.state, w.info.clone())).will_return(40);
        config.estimate_ub.given((x.state, x.info.clone())).will_return(30);
        config.estimate_ub.given((y.state, y.info.clone())).will_return(20);

        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut lel].insert(w.state, w.info.clone());
        layer![mdd, mut lel].insert(x.state, x.info.clone());
        layer![mdd, mut lel].insert(y.state, y.info.clone());
        layer![mdd, mut current].insert(z.state, z.info);
        mdd.finalize();

        assert_eq!(5, layer![mdd, lel].get(&w.state).unwrap().ub);
        assert_eq!(5, layer![mdd, lel].get(&x.state).unwrap().ub);
        assert_eq!(5, layer![mdd, lel].get(&y.state).unwrap().ub);
    }
    #[test]
    fn when_there_is_no_best_node_finalize_clears_the_cutset() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);

        let mut config = MockConfig::default();
        config.estimate_ub.given((w.state, w.info.clone())).will_return(40);
        config.estimate_ub.given((x.state, x.info.clone())).will_return(30);
        config.estimate_ub.given((y.state, y.info.clone())).will_return(20);

        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut lel].insert(w.state, w.info);
        layer![mdd, mut lel].insert(x.state, x.info);
        layer![mdd, mut lel].insert(y.state, y.info);

        mdd.finalize();
        assert_eq!(true, layer![mdd, lel].is_empty());
    }
    #[test]
    fn find_best_node_identifies_the_terminal_node_with_maximum_longest_path() {
        let mut config = MockConfig::default();
        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));

        // when there is no best node, it finds nothing
        mdd.find_best_node();
        assert_eq!(None, mdd.best_node);

        mdd.clear();
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);
        layer![mdd, mut current].insert(w.state, w.info);
        layer![mdd, mut current].insert(x.state, x.info);
        layer![mdd, mut current].insert(y.state, y.info.clone());
        layer![mdd, mut current].insert(z.state, z.info);
        mdd.find_best_node();

        assert_eq!(Some(y.info), mdd.best_node);
    }


    #[test]
    fn no_restriction_may_occur_for_first_layer() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info);
        layer![mdd, mut next].insert(x.state, x.info);
        layer![mdd, mut next].insert(y.state, y.info);
        layer![mdd, mut next].insert(z.state, z.info);

        mdd.maybe_restrict(0, 1);
        assert_eq!(4, layer![mdd, next].len());

        mdd.maybe_restrict(1, 1);
        assert_eq!(4, layer![mdd, next].len());
        assert_eq!(true, mdd.is_exact);
    }
    #[test]
    fn no_relaxation_may_occur_for_first_layer() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info);
        layer![mdd, mut next].insert(x.state, x.info);
        layer![mdd, mut next].insert(y.state, y.info);
        layer![mdd, mut next].insert(z.state, z.info);

        mdd.maybe_relax(0, 1);
        assert_eq!(4, layer![mdd, next].len());

        mdd.maybe_relax(1, 1);
        assert_eq!(4, layer![mdd, next].len());
        assert_eq!(true, mdd.is_exact);
    }
    #[test]
    fn restriction_only_keeps_the_w_best_nodes() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info);
        layer![mdd, mut next].insert(x.state, x.info);
        layer![mdd, mut next].insert(y.state, y.info.clone());
        layer![mdd, mut next].insert(z.state, z.info.clone());

        mdd.maybe_restrict(2, 2);
        assert_eq!(2, layer![mdd, next].len());
        assert_eq!(&z.info, layer![mdd, mut next].get(&z.state).unwrap());
        assert_eq!(&y.info, layer![mdd, mut next].get(&y.state).unwrap());
    }
    #[test]
    fn restriction_makes_the_mdd_inexact() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info);
        layer![mdd, mut next].insert(x.state, x.info);
        layer![mdd, mut next].insert(y.state, y.info);
        layer![mdd, mut next].insert(z.state, z.info);

        mdd.maybe_restrict(2, 2);
        assert_eq!(false, mdd.is_exact)
    }
    #[test]
    fn relaxation_keeps_the_w_min_1_best_nodes() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        // the merge operation will yield a new artificial node
        let merged = Node::new(42, 42, None, false);
        config.merge_nodes.given(vec![x.clone(), w.clone()]).will_return(merged);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info.clone());
        layer![mdd, mut next].insert(x.state, x.info.clone());
        layer![mdd, mut next].insert(y.state, y.info.clone());
        layer![mdd, mut next].insert(z.state, z.info.clone());

        mdd.maybe_relax(2, 3); // max width of 3
        assert_eq!(3, layer![mdd, next].len());
        assert_eq!(&z.info, layer![mdd, mut next].get(&z.state).unwrap());
        assert_eq!(&y.info, layer![mdd, mut next].get(&y.state).unwrap());
        // the 3rd node is none of the others
        assert_eq!(None, layer![mdd, mut next].get(&w.state));
        assert_eq!(None, layer![mdd, mut next].get(&x.state));
    }
    #[test]
    fn relaxation_merges_the_worst_nodes() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        // the merge operation will yield a new artificial node
        let merged = Node::new(42, 42, None, false);
        config.merge_nodes.given(vec![x.clone(), w.clone()]).will_return(merged.clone());

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info.clone());
        layer![mdd, mut next].insert(x.state, x.info.clone());
        layer![mdd, mut next].insert(y.state, y.info);
        layer![mdd, mut next].insert(z.state, z.info);

        mdd.maybe_relax(2, 3); // max width of 3
        assert_eq!(Some(&merged.info), layer![mdd, mut next].get(&merged.state));
        assert!(verify(config.merge_nodes.was_called_with(vec![x, w])));
    }
    #[test]
    fn relaxation_makes_the_mdd_inexact() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, false);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        // the merge operation will yield a new artificial node
        let merged = Node::new(42, 42, None, false);
        config.merge_nodes.given(vec![x.clone(), w.clone()]).will_return(merged);

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info);
        layer![mdd, mut next].insert(x.state, x.info);
        layer![mdd, mut next].insert(y.state, y.info);
        layer![mdd, mut next].insert(z.state, z.info);

        mdd.maybe_relax(2, 3); // max width of 3
        assert_eq!(false, mdd.is_exact());
    }
    #[test]
    fn when_the_merged_node_has_the_same_state_as_one_of_the_best_the_two_are_merged() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, true);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        // the merge operation will yield a new artificial node
        let merged = Node::new(8, 8, None, false);
        config.merge_nodes.given(vec![x.clone(), w.clone()]).will_return(merged.clone());

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info);
        layer![mdd, mut next].insert(x.state, x.info);
        layer![mdd, mut next].insert(y.state, y.info.clone());
        layer![mdd, mut next].insert(z.state, z.info.clone());

        mdd.maybe_relax(2, 3); // max width of 3
        assert_eq!(2, layer![mdd, next].len());
        assert_eq!(&z.info, layer![mdd, mut next].get(&z.state).unwrap());
        // it has been merged w/ the relaxed node so it state is a blend of the two
        assert_ne!(&y.info, layer![mdd, mut next].get(&y.state).unwrap());
        assert_ne!(&merged.info, layer![mdd, mut next].get(&y.state).unwrap());
    }

    #[test]
    fn when_the_merged_node_is_distinct_from_all_others_it_is_added_to_the_set_of_candidate_nodes() {
        let w = Node::new(5, 100, None, false);
        let x = Node::new(7,  10, None, false);
        let y = Node::new(8, 110, None, true);
        let z = Node::new(9,  10, None, false);

        let mut config = MockConfig::default();
        // ordered alphabetically by identifier: w < x < y < z
        config.compare.given((w.clone(), x.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((w.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((x.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((x.clone(), y.clone())).will_return(Ordering::Less);
        config.compare.given((x.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((y.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((y.clone(), z.clone())).will_return(Ordering::Less);

        config.compare.given((z.clone(), w.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), x.clone())).will_return(Ordering::Greater);
        config.compare.given((z.clone(), y.clone())).will_return(Ordering::Greater);

        // the merge operation will yield a new artificial node
        let merged = Node::new(36, 36, None, false);
        config.merge_nodes.given(vec![x.clone(), w.clone()]).will_return(merged.clone());

        let mut mdd = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info);
        layer![mdd, mut next].insert(x.state, x.info);
        layer![mdd, mut next].insert(y.state, y.info.clone());
        layer![mdd, mut next].insert(z.state, z.info.clone());

        mdd.maybe_relax(2, 3); // max width of 3
        assert_eq!(3, layer![mdd, next].len());
        assert_eq!(&z.info,      layer![mdd, next].get(&z.state).unwrap());
        assert_eq!(&y.info,      layer![mdd, next].get(&y.state).unwrap());
        assert_eq!(&merged.info, layer![mdd, next].get(&merged.state).unwrap());
    }

    #[test]
    fn it_current_allows_iteration_on_all_nodes_from_current_layer(){
        let w = Node::new(5, 200, None, false);
        let x = Node::new(7, 150, None, false);
        let y = Node::new(8, 100, None, true);
        let z = Node::new(9,  90, None, false);

        let mut config = MockConfig::default();
        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut current].insert(w.state, w.info.clone());
        layer![mdd, mut current].insert(x.state, x.info.clone());
        layer![mdd, mut current].insert(y.state, y.info.clone());
        layer![mdd, mut current].insert(z.state, z.info.clone());

        let mut res = mdd.it_current().map(|(s, i)| Node{state: *s, info: i.clone()}).collect::<Vec<Node<usize>>>();
        res.sort_unstable_by(|a, b| MinLP.compare(a, b).reverse());
        assert_eq!(vec![w, x, y, z], res);
    }
    #[test]
    fn it_next_allows_iteration_on_all_nodes_from_next_layer() {
        let w = Node::new(5, 200, None, false);
        let x = Node::new(7, 150, None, false);
        let y = Node::new(8, 100, None, true);
        let z = Node::new(9,  90, None, false);

        let mut config = MockConfig::default();
        let mut mdd    = FlatMDD::new(ProxyMut::new(&mut config));
        layer![mdd, mut next].insert(w.state, w.info.clone());
        layer![mdd, mut next].insert(x.state, x.info.clone());
        layer![mdd, mut next].insert(y.state, y.info.clone());
        layer![mdd, mut next].insert(z.state, z.info.clone());

        let mut res = mdd.it_next().map(|(s, i)| Node{state: *s, info: i.clone()}).collect::<Vec<Node<usize>>>();
        res.sort_unstable_by(|a, b| MinLP.compare(a, b).reverse());
        assert_eq!(vec![w, x, y, z], res);
    }
}