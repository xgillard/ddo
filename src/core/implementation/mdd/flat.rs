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

    fn swap_current_lel(&mut self) {
        let tmp      = self.current;
        self.current = self.lel;
        self.lel     = tmp;
    }
    fn swap_current_next(&mut self) {
        let tmp      = self.current;
        self.current = self.next;
        self.next    = tmp;
    }
    fn is_relevant(&self, bounds: Bounds, state: &T, info: &NodeInfo) -> bool {
        min(self.config.estimate_ub(state, info), bounds.ub) > bounds.lb
    }
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

    fn move_to_next(&mut self, was_exact: bool) {
        if self.is_exact != was_exact {
            self.swap_current_lel();
        }
        self.swap_current_next();
        layer![self, mut next].clear();
    }

    fn init(&mut self, kind: MDDType, root: &Node<T>) {
        self.clear();
        self.config.load_vars(root);
        self.mddtype = kind;

        layer![self, mut current].insert(root.state.clone(), root.info.clone());
    }
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
    fn find_best_node(&mut self) {
        let mut best_value = i32::min_value();
        for info in layer![self, current].values() {
            if info.lp_len > best_value {
                best_value         = info.lp_len;
                self.best_node  = Some(info.clone());
            }
        }
    }
    fn maybe_squash(&mut self, i : usize, w: usize) {
        match self.mddtype {
            MDDType::Exact      => /* nothing to do ! */(),
            MDDType::Restricted => self.maybe_restrict(i, w),
            MDDType::Relaxed    => self.maybe_relax(i, w),
        }
    }
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

    fn it_current(&self) -> Layer<'_, T> {
        Layer::Mapped(layer![self, current].iter())
    }
    fn it_next(&self) -> Layer<'_, T> {
        Layer::Mapped(layer![self, next].iter())
    }
}

#[cfg(test)]
mod test_mdd {
    use crate::core::abstraction::mdd::{MDDType, MDD};
    use crate::core::implementation::mdd::flat::FlatMDD;
    use crate::test_utils::{MockConfig, ProxyMut};

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
    }
}