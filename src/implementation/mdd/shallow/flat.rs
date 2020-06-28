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
use crate::implementation::mdd::shallow::utils::{Edge, Node};
use crate::implementation::mdd::utils::NodeFlags;

/// This is nothing but a writing simplification to tell that in a flat mdd,
/// a layer is a hashmap of states to nodes
type Layer<T> = MetroHashMap<Arc<T>, Node<T>>;

/// This is the structure implementing _flat MDD_. This is a kind of
/// bounded width MDD which offers a real guarantee wrt to the maximum amount
/// of used memory.
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
/// // Following line configure and builds a flat mdd.
/// let flat_mdd   = mdd_builder(&problem, relaxation).into_flat();
///
/// // Naturally, you can also provide configuration parameters to customize
/// // the behavior of your MDD. For instance, you can use a custom max width
/// // heuristic as follows (below, a fixed width)
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let flat_mdd = mdd_builder(&problem, relaxation)
///                 .with_max_width(FixedWidth(100))
///                 .into_flat();
/// ```
#[derive(Debug, Clone)]
pub struct FlatMDD<T, C>
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
    /// This array stores the three layers known by the mdd: the current, next
    /// and last exact layer (lel). The position of each layer in the array is
    /// determined by the `current`, `next` and `lel` fields of the structure.
    layers: [Layer<T>; 3],
    /// The index of the current layer in the array of `layers`.
    current: usize,
    /// The index of the next layer in the array of `layers`.
    next: usize,
    /// The index of the last exact layer (lel) in the array of `layers`
    lel: usize,
    /// The index of the previous layer in the array of `layers`. It may either
    /// be equal to current or lel.
    prev: usize,
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
/// As the name suggests, `FlatMDD` is an implementation of the `MDD` trait.
/// See the trait definiton for the documentation related to these methods.
impl <T, C> MDD<T, C> for FlatMDD<T, C>
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
                self.layers[self.lel].values().for_each(|n| {
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
/// This macro wraps a tiny bit of unsafe code that lets us borrow the current
/// layer only. This macro should not be used anywhere outside the current file.
macro_rules! current_layer {
    ($dd:expr) => {
        unsafe { &*$dd.layers.as_ptr().add($dd.current) }
    };
}
impl <T, C> FlatMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(config: C) -> Self {
        FlatMDD {
            config,
            mddtype  : MDDType::Exact,
            layers   : [Default::default(), Default::default(), Default::default()],
            current  : 0,
            next     : 1,
            lel      : 2,
            prev     : 0,
            is_exact : true,
            max_width: usize::max_value(),
            best_lb  : isize::min_value(),
            best_node: None
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    pub fn clear(&mut self) {
        self.mddtype   = MDDType::Exact;
        self.current   = 0;
        self.next      = 1;
        self.lel       = 2;
        self.prev      = 0;
        self.is_exact  = true;
        self.best_node = None;
        self.best_lb   = isize::min_value();
        self.layers.iter_mut().for_each(|l|l.clear());
    }
    /// Develops/Unrolls the requested type of MDD, starting from a given root
    /// node. It only considers nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`) and assigns a value to the variables of the specified
    /// VarSet (`vars`).
    fn develop(&mut self, root: &FrontierNode<T>, mut vars: VarSet, best_lb: isize) {
        let root     = Node::from(root);
        self.best_lb = best_lb;
        self.layers[self.next].insert(Arc::clone(&root.this_state), root);

        let mut depth = 0;
        while let Some(var) = self.next_var(&vars) {
            self.add_layer();
            vars.remove(var);
            depth += 1;

            for node in current_layer!(self).values() {
                let src_state = node.this_state.as_ref();
                for val in self.config.domain_of(src_state, var) {
                    let decision = Decision { variable: var, value: val };
                    let state    = self.config.transition(src_state, &vars, decision);
                    let weight   = self.config.transition_cost(src_state, &vars, decision);

                    self.branch(node, state, decision, weight)
                }
            }

            // squash layer if needed
            match self.mddtype {
                MDDType::Exact => {},
                MDDType::Restricted =>
                    if self.layers[self.next].len() > self.max_width {
                        self.restrict_last();
                    },
                MDDType::Relaxed =>
                    if depth > 1 && self.layers[self.next].len() > self.max_width {
                        self.relax_last();
                    }
            }
        }

        self.finalize()
    }
    /// Returns the next variable to branch on (according to the configured
    /// branching heuristic) or None if all variables have been assigned a value.
    fn next_var(&self, vars: &VarSet) -> Option<Variable> {
        let mut curr_it = self.layers[self.prev].keys().map(|k| k.as_ref());
        let mut next_it = self.layers[self.next].keys().map(|k| k.as_ref());

        self.config.select_var(vars, &mut curr_it, &mut next_it)
    }
    /// Adds one layer to the mdd and move to it.
    /// In practice, this amounts to considering the 'next' layer as the current
    /// one, a clearing the next one.
    fn add_layer(&mut self) {
        self.swap_current_next();
        self.prev = self.current;
        self.layers[self.next].clear();
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
        match self.layers[self.next].entry(Arc::clone(&node.this_state)) {
            Entry::Vacant(re) => {
                node.estimate = self.config.estimate(node.state());
                if node.ub() > self.best_lb {
                    re.insert(node);
                }
            },
            Entry::Occupied(mut re) => { re.get_mut().merge(node); }
        }
    }

    /// Swaps the indices of the current and next layers effectively moving
    /// to the next layer (next is now considered current)
    fn swap_current_next(&mut self) {
        let tmp      = self.current;
        self.current = self.next;
        self.next    = tmp;
    }
    /// Swaps the indices of the current and last exact layers, effectively
    /// remembering current as the last exact layer.
    fn swap_current_lel(&mut self) {
        let tmp      = self.current;
        self.current = self.lel;
        self.lel     = tmp;
    }
    /// Records the last exact layer. It only has an effect when the mdd is
    /// considered to be still correct. In other words, it will only remember
    /// the LEL the first time either `restrict_last()` or `relax_last()` is
    /// called.
    fn remember_lel(&mut self) {
        if self.is_exact {
            self.is_exact = false;
            self.swap_current_lel();
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
    fn restrict_last(&mut self) {
        self.remember_lel();

        let mut nodes = self.layers[self.next].drain()
            .map(|(_, v)| v)
            .collect::<Vec<Node<T>>>();
        nodes.sort_unstable_by(|a, b| self.config.compare(a, b).reverse());
        nodes.truncate(self.max_width);

        for node in nodes.drain(..) {
            self.layers[self.next].insert(Arc::clone(&node.this_state), node);
        }
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
    fn relax_last(&mut self) {
        self.remember_lel();

        let mut nodes = self.layers[self.next].drain()
            .map(|(_, v)| v)
            .collect::<Vec<Node<T>>>();
        nodes.sort_unstable_by(|a, b| self.config.compare(a, b).reverse());

        let (keep, squash) = nodes.split_at_mut(self.max_width - 1);
        for node in keep {
            self.layers[self.next].insert(Arc::clone(&node.this_state), node.clone());
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
            estimate  : isize::max_value(),
            flags     : NodeFlags::new_relaxed(),
            best_edge : merged_edge
        };
        self.add_node(rlx_node)
    }
    /// Finalizes the computation of the MDD: it identifies the best terminal node.
    fn finalize(&mut self) {
        self.best_node = self.layers[self.next].values()
            .max_by_key(|n| n.value)
            .cloned();
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################


#[cfg(test)]
mod test_flatmdd {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::mdd::{MDD, Config};
    use crate::common::{Decision, Domain, FrontierNode, PartialAssignment, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::mdd::MDDType;
    use crate::implementation::mdd::shallow::flat::FlatMDD;
    use crate::test_utils::MockConfig;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd = FlatMDD::new(config);

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
        let mut mdd = FlatMDD::new(config);

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
            .into_flat();

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
            .into_flat();

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
            .into_flat();

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
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_flat();

        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);

        let mut cutset = vec![];
        mdd.for_each_cutset_node(|n| cutset.push(n));
        assert_eq!(cutset.len(), 3); // L1 was not squashed even though it was 3 wide
    }

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .into_flat();

        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_flat();
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_flat();
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).into_flat();
        let root = mdd.config().root_node();
        mdd.restricted(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).into_flat();
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
        let mut mdd = mdd_builder(&pb, rlx).into_flat();
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_flat();
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(isize::min_value(), mdd.best_value())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_flat();
        let root = mdd.config().root_node();

        mdd.exact(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_flat();
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = mdd_builder(&pb, rlx).into_flat();
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
    use crate::implementation::mdd::shallow::flat::FlatMDD;
    use crate::test_utils::{MockConfig, Proxy};

    #[test]
    fn branch_inserts_a_node_with_given_state_when_none_exists() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(Proxy::new(&config));
        let node    = Node {
            this_state: Arc::new(42),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 42,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        };
        mdd.add_node(node);
        mdd.add_layer();

        let src = &current_layer!(&mdd)[&Arc::new(42)];
        let dst = 1;
        let dec = Decision{variable: Variable(9), value: 6};
        let wt  = 1;

        assert_eq!(0, mdd.layers[mdd.next].len());
        mdd.branch(src, dst, dec, wt);
        assert_eq!(1, mdd.layers[mdd.next].len());
        assert!(mdd.layers[mdd.next].get(&Arc::new(1)).is_some());
        // TODO assert!(verify(config.estimate.was_called()))
    }
    #[test]
    fn branch_wont_update_existing_node_to_remember_last_decision_and_path_if_it_doesnt_improve_value() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        let node    = Node {
            this_state: Arc::new(42),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 42,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        };
        mdd.add_node(node);
        mdd.add_layer();

        mdd.add_node(Node {
            this_state: Arc::new(1),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 100,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        });

        let src = &current_layer!(&mdd)[&Arc::new(42)];
        let dst = 1;
        let dec = Decision{variable: Variable(9), value: 6};
        let wt  = 1;

        mdd.branch(src, dst, dec, wt);
        assert!(mdd.layers[mdd.next][&Arc::new(1)].best_edge.is_none());
        assert_eq!(100, mdd.layers[mdd.next][&Arc::new(1)].value);
    }
    #[test]
    fn branch_updates_existing_node_to_remember_last_decision_and_path_if_it_improves_value() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        let node    = Node {
            this_state: Arc::new(42),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 42,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        };
        mdd.add_node(node);
        mdd.add_layer();

        mdd.add_node(Node {
            this_state: Arc::new(1),
            path      : Arc::new(PartialAssignment::Empty),
            value     : 1,
            estimate  : isize::max_value(),
            flags     : Default::default(),
            best_edge : None
        });

        let src = &current_layer!(&mdd)[&Arc::new(42)];
        let dst = 1;
        let dec = Decision{variable: Variable(9), value: 6};
        let wt  = 1;

        mdd.branch(src, dst, dec, wt);
        assert!(mdd.layers[mdd.next][&Arc::new(1)].best_edge.is_some());
        assert_eq!(43, mdd.layers[mdd.next][&Arc::new(1)].value);
    }
    #[test]
    fn remember_lel_has_no_effect_when_lel_is_present() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        mdd.is_exact= false;

        assert_eq!(0,    mdd.current);
        assert_eq!(2,    mdd.lel);
        mdd.remember_lel();
        assert_eq!(0,    mdd.current);
        assert_eq!(2,    mdd.lel);
    }
    #[test]
    fn remember_lel_remembers_the_last_exact_layer() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);

        assert_eq!(true, mdd.is_exact());
        assert_eq!(0,    mdd.current);
        assert_eq!(2,    mdd.lel);
        mdd.remember_lel();
        assert_eq!(false, mdd.is_exact());
        assert_eq!(2,    mdd.current);
        assert_eq!(0,    mdd.lel);
    }

    macro_rules! get {
        ($dd: expr, $state: expr) => {
            &current_layer!($dd)[&Arc::new($state)]
        };
        (next $dd: expr, $state: expr) => {
            &next_layer!($dd)[&Arc::new($state)]
        };
    }
    #[test]
    fn restrict_last_remembers_the_last_exact_layer_if_needed() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        add_root(&mut mdd, 33, 3);
        mdd.add_layer();

        let r_id = get!(&mdd, 33);
        mdd.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert!(mdd.is_exact);

        // first time, a lel is saved
        mdd.max_width = 2;
        mdd.restrict_last();
        assert!(!mdd.is_exact);
        assert_eq!(1, mdd.lel);

        // but it is not updated after a subsequent restrict
        mdd.add_layer();
        let r_id = get!(&mdd, 34);
        mdd.branch(r_id, 37, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 38, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 39, Decision{variable: Variable(0), value: 3}, 1);
        mdd.branch(r_id, 40, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 41, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 42, Decision{variable: Variable(0), value: 3}, 1);
        mdd.restrict_last();
        assert!(!mdd.is_exact);
        assert_eq!(1, mdd.lel);
    }
    #[test]
    fn restrict_last_makes_the_graph_inexact() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        add_root(&mut mdd, 33, 3);
        mdd.add_layer();

        let r_id = get!(&mdd, 33);
        mdd.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert!(mdd.is_exact());

        // first time, a lel is saved
        mdd.restrict_last();
        assert!(!mdd.is_exact());
    }
    #[test]
    fn restrict_last_layer_enforces_the_max_width() {
        let config  = MockConfig::default();
        let mut mdd = FlatMDD::new(config);
        add_root(&mut mdd, 33, 3);
        mdd.add_layer();

        let r_id = get!(&mdd, 33);
        mdd.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        mdd.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        mdd.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, mdd.layers[mdd.next].len());

        mdd.max_width = 2;
        mdd.restrict_last();
        assert_eq!(2, mdd.layers[mdd.next].len());

        mdd.max_width = 1;
        mdd.restrict_last();
        assert_eq!(1, mdd.layers[mdd.next].len());
    }

    macro_rules! next_layer {
        ($dd: expr) => {$dd.layers[$dd.next]};
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

        let mut g  = FlatMDD::new(c);
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states_before = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
        states_before.sort_unstable();
        assert_eq!(vec![34, 35, 36], states_before);

        g.restrict_last();
        let mut states_after = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
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

        let mut g  = FlatMDD::new(c);
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();

        let r_id = get!(g, 33);
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.restrict_last();
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        // 1. check the appropriate heuristic is used
        let mut states_before = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
        states_before.sort_unstable();
        assert_eq!(vec![34, 35, 36], states_before);

        //
        g.restrict_last();
        assert!(verify(c.compare.was_called_with((34, 35))));
        let mut states_after = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
        states_after.sort_unstable();
        assert_eq!(vec![35, 36], states_after);
    }
    #[test]
    fn relax_last_remembers_the_last_exact_layer_if_needed() {
        let c = MockConfig::default();
        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert!(g.is_exact);

        // first time, a lel is saved
        g.relax_last();
        assert!(!g.is_exact);
        assert_eq!(1, g.lel);

        // but it is not updated after a subsequent restrict
        g.add_layer();
        let r_id = get!(g, 35);
        g.branch(r_id, 37, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 38, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 39, Decision{variable: Variable(0), value: 3}, 1);
        g.branch(r_id, 40, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 41, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 42, Decision{variable: Variable(0), value: 3}, 1);
        g.relax_last();
        assert!(!g.is_exact);
        assert_eq!(1, g.lel);
    }
    #[test]
    fn relax_last_makes_the_graph_inexact() {
        let mut g  = FlatMDD::new(MockConfig::default());
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert!(g.is_exact());

        // first time, a lel is saved
        g.relax_last();
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, next_layer!(g).len());

        g.relax_last();
        let mut cur = next_layer!(g).values().map(|n| *n.state()).collect::<Vec<usize>>();
        cur.sort_unstable();
        assert_eq!(vec![36, 37], cur);
        assert_eq!(2, next_layer!(g).len());
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, next_layer!(g).len());

        g.relax_last();
        let cur = next_layer!(g).values().map(|n| *n.state()).collect::<Vec<usize>>();
        assert_eq!(vec![36], cur);
        assert_eq!(1, next_layer!(g).len());
    }
    #[test]
    fn test_relax_last_layer_when_width_is_one() {
        let c = MockConfig::default();
        // Merged state is 37
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 1;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, next_layer!(g).len());

        g.relax_last();
        let cur = next_layer!(g).values().map(|n| *n.state()).collect::<Vec<usize>>();
        assert_eq!(vec![37], cur);
        assert_eq!(1, next_layer!(g).len());
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last();
        let mut states = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last();
        let mut states = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last();
        let mut states = next_layer!(g).keys().map(|k| **k).collect::<Vec<usize>>();
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last();
        // 36
        assert_eq!(36, *get!(next g, 36).state());
        assert_eq!( 4,  get!(next g, 36).value());

        // 37 (mock relaxes everything to 0)
        assert_eq!(37, *get!(next g, 37).state());
        assert_eq!( 3,  get!(next g, 37).value());
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last();
        // 36
        assert_eq!(36, *get!(next g, 36).state());
        assert_eq!( 4,  get!(next g, 36).value());
    }
    #[test]
    fn relax_last_layer_uses_node_selection_heuristic_to_rank_nodes_and_renames_others() {
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last();
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last();
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(4, get!(next g, 36).value);

        g.relax_last();
        assert_eq!(4, get!(next g, 36).value)
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

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

        g.relax_last();
        assert_eq!(33, *get!(next g, 36).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!(20,  get!(next g, 36).value);
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

        let mut g  = FlatMDD::new(Proxy::new(&c));
        g.max_width= 2;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

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

        g.relax_last();
        assert_eq!(33, *get!(next g, 37).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!(13,  get!(next g, 37).value);

        assert_eq!(33, *get!(next g, 36).best_edge.as_ref().unwrap().parent_state.as_ref());
        assert_eq!( 4,  get!(next g, 36).value);
    }
    #[test]
    #[should_panic]
    fn relax_last_panics_if_width_is_0() {
        let mut g  = FlatMDD::new(MockConfig::default());
        g.max_width= 0;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last();
    }
    #[test]
    #[should_panic]
    fn relax_last_panics_if_layer_is_not_broad_enough() {
        let mut g  = FlatMDD::new(MockConfig::default());
        g.max_width= 10;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last();
    }
    #[test]
    fn restrict_last_wont_panic_if_layer_is_not_broad_enough() {
        let mut g  = FlatMDD::new(MockConfig::default());
        g.max_width= 10;
        add_root(&mut g, 33, 3);
        g.add_layer();
        let r_id = get!(g, 33);

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.restrict_last();
    }

    fn add_root<T, C>(mdd : &mut FlatMDD<T, C>, s: T, v: isize)
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