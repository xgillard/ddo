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

//! This module provides an adapter for minimization problems.
//! It flips the signs of the cost and initial value of a problem so as to make
//! it behave as though the solver had been thougt with minimization in mind
//! instead of the assumed maximization.

use crate::core::abstraction::dp::Problem;
use crate::core::common::{Variable, VarSet, Domain, Decision};

#[derive(Debug, Clone)]
/// This structure provides a simple adapter to express a minimization intent
/// on the objective function instead of the assumed maximization.
///
/// # Warning
/// **The Minimize() api is still subject to change**.
/// Minimize implements the `Problem` trait by delgating all calls to the target
/// but it flips the signs of the transition costs and initial value. Hence,
/// the the sign of the final 'best' solution should be flipped in order to
/// get the actual solution value.
///
/// # Example Usage
/// ```
/// # use ddo::core::implementation::mdd::builder::mdd_builder;
/// # use ddo::core::implementation::heuristics::FixedWidth;
/// # use ddo::core::abstraction::dp::{Problem, Relaxation};
/// # use ddo::core::common::{Variable, Domain, VarSet, Decision, Node};
/// # use ddo::core::implementation::solver::parallel::ParallelSolver;
/// # use ddo::core::abstraction::solver::Solver;
/// use ddo::core::implementation::dp::Minimize;
/// # #[derive(Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> i32   { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         (0..1).into()
/// #     }
/// #     fn transition(&self, state: &usize, _: &VarSet, _: Decision) -> usize {
/// #         41
/// #     }
/// #     fn transition_cost(&self, state: &usize, _: &VarSet, _: Decision) -> i32 {
/// #         42
/// #     }
/// # }
/// # #[derive(Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #     fn merge_nodes(&self, n: &[Node<usize>]) -> Node<usize> {
/// #        n[0].clone()
/// #     }
/// # }
/// let problem    = Minimize(MockProblem);
/// let relaxation = MockRelax;
/// let mdd        = mdd_builder(&problem, relaxation).build();
/// let mut solver = ParallelSolver::new(mdd);
/// // val is the optimal value of the objective function. However, because
/// // the signs of all transition costs and initial values have been flipped,
/// // the sign of that optimal value should be flipped too.
/// let (val, sol) = solver.maximize();
/// // This is the 'true' value obtained by taking the best decisions in the
/// // original problem
/// let actual_val = -val;
/// ```
pub struct Minimize<P>(pub P);

impl <T, P: Problem<T>> Problem<T> for Minimize<P> {
    fn nb_vars(&self) -> usize {
        self.0.nb_vars()
    }

    fn initial_state(&self) -> T {
        self.0.initial_state()
    }

    fn initial_value(&self) -> i32 {
        -self.0.initial_value()
    }

    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a> {
        self.0.domain_of(state, var)
    }

    fn transition(&self, state: &T, vars: &VarSet, d: Decision) -> T {
        self.0.transition(state, vars, d)
    }

    fn transition_cost(&self, state: &T, vars: &VarSet, d: Decision) -> i32 {
        -self.0.transition_cost(state, vars, d)
    }

    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        self.0.impacted_by(state, variable)
    }
}

#[cfg(test)]
mod test_minimize {
    use crate::core::abstraction::dp::Problem;
    use crate::core::abstraction::solver::Solver;
    use crate::core::common::{Variable, VarSet, Domain, Decision};
    use crate::core::implementation::dp::Minimize;
    use crate::core::implementation::mdd::builder::mdd_builder;
    use crate::test_utils::MockRelax;
    use crate::core::implementation::solver::parallel::ParallelSolver;

    #[derive(Debug, Clone)]
    struct DummyPb;
    impl Problem<usize> for DummyPb {
        fn nb_vars(&self)       -> usize { 2 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> i32   { 1 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            vec![1, 2, 3].into()
        }

        fn transition(&self, s: &usize, _: &VarSet, d: Decision) -> usize {
            s + d.value as usize
        }

        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> i32 {
            d.value
        }
    }
    #[test]
    fn the_minimize_adapter_has_no_impact_except_for_the_costs() {
        let original = DummyPb;
        let tested   = Minimize(original.clone());

        assert_eq!(original.nb_vars(),       tested.nb_vars());
        assert_eq!(original.initial_state(), tested.initial_state());

        let state    = 0;
        let vars     = VarSet::all(3);
        let decision = Decision{variable: Variable(2), value: 2};
        assert_eq!(
            original.transition(&state, &vars, decision),
            tested  .transition(&state, &vars, decision));

        assert_eq!(-original.initial_value(), tested.initial_value());
        assert_eq!(
            -original.transition_cost(&state, &vars, decision),
            tested   .transition_cost(&state, &vars, decision))
    }

    #[test]
    fn solver_effectively_minimizes_the_function() {
        // If we were to maximize this problem, the optimal would be 7,
        // and we would obtain it by taking decision x_i = 3 everytime.
        let pb = DummyPb;
        let mdd= mdd_builder(&pb, MockRelax::default()).build();
        let mut slv= ParallelSolver::new(mdd);

        let (opt, sln) = slv.maximize();
        assert_eq!(7, opt);
        assert_eq!(sln.as_ref().unwrap().clone(), vec![
            Decision{variable: Variable(1), value: 3},
            Decision{variable: Variable(0), value: 3},
        ]);

        // ..but we are minimizing the problem, so the minimal value is 3
        // and we get it by deciding x_i = 1 at for every i.
        let pb = Minimize(DummyPb);
        let mdd= mdd_builder(&pb, MockRelax::default()).build();
        let mut slv= ParallelSolver::new(mdd);

        let (opt, sln) = slv.maximize();

        assert_eq!(3, -opt);
        assert_eq!(sln.as_ref().unwrap().clone(), vec![
            Decision{variable: Variable(1), value: 1},
            Decision{variable: Variable(0), value: 1},
        ]);
    }
}