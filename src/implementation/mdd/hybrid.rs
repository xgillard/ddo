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

//! This module provides the implementation of hybrid mdds. In this case,
//! hybrid is to be understood as a 'composite' mdd which behaves as one type
//! of mdd (ie flat mdd) or as an other one (ie deep mdd) depending on the type
//! op operation which is requested. This should make for a simple yet efficient
//! dd representation which can benefit from the strengths of its components
//! implementations.

use std::hash::Hash;
use crate::abstraction::mdd::{Config, MDD};
use crate::implementation::mdd::shallow::pooled::PooledMDD;
use crate::implementation::mdd::deep::mdd::DeepMDD;
use crate::common::{FrontierNode, Solution};
use crate::implementation::mdd::MDDType;
use std::marker::PhantomData;
use crate::implementation::mdd::shallow::flat::FlatMDD;

/// This is the composite mdd which provides the core of all hybrid
/// implementations.
/// The parameter types mean the following:
/// - T is the type of the states from the problem.
/// - C is the type of the mdd configuration
/// - X is the first kind of mdd. It is used to unroll exact and restricted mdds.
/// - Y is the 2nd kind of mdd. It is used to unroll the relaxed mdds.
#[derive(Clone)]
struct CompositeMDD<T, C, X, Y>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone,
          X: MDD<T, C>,
          Y: MDD<T, C>
{
    /// This is the kind of unrolling that was requested. It determines if this
    /// mdd must be an `Exact`, `Restricted` or `Relaxed` MDD.
    mddtype   : MDDType,
    /// This is the first 'component' representation. It is used to derive exact
    /// and restricted mdds.
    restricted: X,
    /// This is the 2nd 'component' representation. It is used to derive relaxed
    /// mdds from the problem.
    relaxed   : Y,
    /// This is nothing but a marker to let the compiler be aware that we
    /// effectively need to know the types T and C, even though we do not
    /// manipulate them immediately.
    _phantom  : PhantomData<(T, C)>
}

/// The composite mdd implements the MDD<T, C> trait and it does so by forwarding
/// the calls to the component representations (self.restricted and self.relaxed).
/// The exact and restricted mdds are derived by using the 'self.restricted'
/// repres. while the relaxed mdd are derived by using 'self.relaxed'.
///
/// For more details about the semantics of each of the trait methods, refer to
/// the trait documentation.
impl <T, C, X, Y> MDD<T, C> for CompositeMDD<T, C, X, Y>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone,
          X: MDD<T, C>,
          Y: MDD<T, C>
{
    fn config(&self) -> &C {
        self.relaxed.config()
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.mddtype = MDDType::Exact;
        self.restricted.exact(root, best_lb)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.mddtype = MDDType::Restricted;
        self.restricted.restricted(root, best_lb)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.mddtype = MDDType::Relaxed;
        self.relaxed.relaxed(root, best_lb)
    }

    fn is_exact(&self) -> bool {
        match self.mddtype {
            MDDType::Exact      => self.restricted.is_exact(),
            MDDType::Restricted => self.restricted.is_exact(),
            MDDType::Relaxed    => self.relaxed.is_exact()
        }
    }

    fn best_value(&self) -> isize {
        match self.mddtype {
            MDDType::Exact      => self.restricted.best_value(),
            MDDType::Restricted => self.restricted.best_value(),
            MDDType::Relaxed    => self.relaxed.best_value()
        }
    }

    fn best_solution(&self) -> Option<Solution> {
        match self.mddtype {
            MDDType::Exact      => self.restricted.best_solution(),
            MDDType::Restricted => self.restricted.best_solution(),
            MDDType::Relaxed    => self.relaxed.best_solution()
        }
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        match self.mddtype {
            MDDType::Exact      => self.restricted.for_each_cutset_node(func),
            MDDType::Restricted => self.restricted.for_each_cutset_node(func),
            MDDType::Relaxed    => self.relaxed.for_each_cutset_node(func)
        }
    }
}

// -----------------------------------------------------------------------------
// HYBRID FLAT - DEEP --------------------------------------------------------
// -----------------------------------------------------------------------------
/// This structure is an hybrid mdd that behaves like a `FlatMDD` when
/// unrolling an exact or relaxed mdd, and behaves like a `DeepMDD` when it
/// comes to deriving a relaxed dd.
///
/// This is a pure decorator over a `CompositeMDD` to which all calls are delegated.
pub struct HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// This is the composite that performs the heavy lifting.
    composite: CompositeMDD<T, C, FlatMDD<T, C>, DeepMDD<T, C>>
}
impl <T, C> HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// Creates a new hybrid mdd using the given configuration.
    pub fn new(c: C) -> Self {
        HybridFlatDeep {
            composite: CompositeMDD {
                mddtype   : MDDType::Exact,
                restricted: FlatMDD::new(c.clone()),
                relaxed   : DeepMDD::new(c),
                _phantom: PhantomData
            }
        }
    }
}
/// The hybrid mdd implements the MDD<T, C> trait and it does so by forwarding
/// the calls to the component representations (self.restricted and self.relaxed).
/// The exact and restricted mdds are derived by using the 'self.restricted'
/// repres. while the relaxed mdd are derived by using 'self.relaxed'.
///
/// For more details about the semantics of each of the trait methods, refer to
/// the trait documentation.
impl <T, C> MDD<T, C> for HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn config(&self) -> &C {
        self.composite.config()
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.exact(root, best_lb)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.restricted(root, best_lb)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.relaxed(root, best_lb)
    }

    fn is_exact(&self) -> bool {
        self.composite.is_exact()
    }

    fn best_value(&self) -> isize {
        self.composite.best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self.composite.best_solution()
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        self.composite.for_each_cutset_node(func)
    }
}
/// The hybrid mdd implements the From<C> trait, to indicate that it can be built
/// from the given configuration. This makes it suitable for use with the parallel
/// solver which requires the implementation of the "From<C>" trait.
impl <T, C> From<C> for HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn from(c: C) -> Self {
        Self::new(c)
    }
}

// -----------------------------------------------------------------------------
// HYBRID POOLED - DEEP --------------------------------------------------------
// -----------------------------------------------------------------------------
/// This structure is an hybrid mdd that behaves like a `PooledMDD` when
/// unrolling an exact or relaxed mdd, and behaves like a `DeepMDD` when it
/// comes to deriving a relaxed dd.
///
/// This is a pure decorator over a `CompositeMDD` to which all calls are delegated.
pub struct HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// This is the composite that performs the heavy lifting.
    composite: CompositeMDD<T, C, PooledMDD<T, C>, DeepMDD<T, C>>
}
impl <T, C> HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// Creates a new hybrid mdd using the given configuration.
    pub fn new(c: C) -> Self {
        HybridPooledDeep {
            composite: CompositeMDD {
                mddtype   : MDDType::Exact,
                restricted: PooledMDD::new(c.clone()),
                relaxed   : DeepMDD::new(c),
                _phantom: PhantomData
            }
        }
    }
}
/// The hybrid mdd implements the MDD<T, C> trait and it does so by forwarding
/// the calls to the component representations (self.restricted and self.relaxed).
/// The exact and restricted mdds are derived by using the 'self.restricted'
/// repres. while the relaxed mdd are derived by using 'self.relaxed'.
///
/// For more details about the semantics of each of the trait methods, refer to
/// the trait documentation.
impl <T, C> MDD<T, C> for HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn config(&self) -> &C {
        self.composite.config()
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.exact(root, best_lb)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.restricted(root, best_lb)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.relaxed(root, best_lb)
    }

    fn is_exact(&self) -> bool {
        self.composite.is_exact()
    }

    fn best_value(&self) -> isize {
        self.composite.best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self.composite.best_solution()
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        self.composite.for_each_cutset_node(func)
    }
}
/// The hybrid mdd implements the From<C> trait, to indicate that it can be built
/// from the given configuration. This makes it suitable for use with the parallel
/// solver which requires the implementation of the "From<C>" trait.
impl <T, C> From<C> for HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn from(c: C) -> Self {
        Self::new(c)
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################


#[cfg(test)]
mod test_hybrid_flat_deep {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::mdd::{MDD, Config};
    use crate::common::{Decision, Domain, FrontierNode, PartialAssignment, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::mdd::MDDType;
    use crate::test_utils::MockConfig;
    use crate::implementation::mdd::hybrid::HybridFlatDeep;

    type DD<T, C> = HybridFlatDeep<T, C>;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd = DD::new(config);

        assert_eq!(MDDType::Exact, mdd.composite.mddtype);
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
        let mut mdd = DD::new(config);

        mdd.relaxed(&root_n, 0);
        assert_eq!(MDDType::Relaxed, mdd.composite.mddtype);

        mdd.restricted(&root_n, 0);
        assert_eq!(MDDType::Restricted, mdd.composite.mddtype);

        mdd.exact(&root_n, 0);
        assert_eq!(MDDType::Exact, mdd.composite.mddtype);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        mdd.restricted(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let mut mdd = DD::from(config);
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
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(isize::min_value(), mdd.best_value())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.restricted(&root, 100);
        assert!(mdd.best_solution().is_none())
    }
}


#[cfg(test)]
mod test_hybrid_pooled_deep {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::mdd::{MDD, Config};
    use crate::common::{Decision, Domain, FrontierNode, PartialAssignment, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::mdd::MDDType;
    use crate::test_utils::MockConfig;
    use crate::implementation::mdd::hybrid::HybridPooledDeep;

    type DD<T, C> = HybridPooledDeep<T, C>;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd = DD::new(config);

        assert_eq!(MDDType::Exact, mdd.composite.mddtype);
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
        let mut mdd = DD::new(config);

        mdd.relaxed(&root_n, 0);
        assert_eq!(MDDType::Relaxed, mdd.composite.mddtype);

        mdd.restricted(&root_n, 0);
        assert_eq!(MDDType::Restricted, mdd.composite.mddtype);

        mdd.exact(&root_n, 0);
        assert_eq!(MDDType::Exact, mdd.composite.mddtype);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
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
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        mdd.relaxed(&root, 0);
        assert_eq!(false, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        mdd.restricted(&root, 0);
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occured() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let mut mdd = DD::from(config);
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
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 0);
        assert_eq!(isize::min_value(), mdd.best_value())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.exact(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.relaxed(&root, 100);
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        mdd.restricted(&root, 100);
        assert!(mdd.best_solution().is_none())
    }
}