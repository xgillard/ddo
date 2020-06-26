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

//! This module defines the basic abstractions one will manipulate when
//! implementing an MDD optimization-solver for some problem formulated as a
//! dynamic program.
//!
//! The most important abstractions that should be provided by a client are
//! `Problem` and `Relaxation`.

use crate::common::{Variable, Domain, VarSet, Decision};

/// This is the main abstraction that should be provided by any user of our
/// library. Indeed, it defines the problem to be solved in the form of a dynamic
/// program. Therefore, this trait closely sticks to the formal definition of a
/// dynamic program.
///
/// The type parameter `<T>` denotes the type of the states of the dynamic program.
pub trait Problem<T> {
    /// Returns the number of decision variables that play a role in the problem.
    fn nb_vars(&self) -> usize;
    /// Returns the initial state of the problem (when no decision is taken).
    fn initial_state(&self) -> T;
    /// Returns the initial value of the objective function (when no decision is taken).
    fn initial_value(&self) -> isize;

    /// Returns the domain of variable `var` in the given `state`. These are the
    /// possible values that might possibly be affected to `var` when the system
    /// has taken decisions leading to `state`.
    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a>;
    /// Returns the next state reached by the system if the decision `d` is
    /// taken when the system is in the given `state` and the given set of `vars`
    /// are still free (no value assigned).
    fn transition(&self, state: &T, vars : &VarSet, d: Decision) -> T;
    /// Returns the marginal benefit (in terms of objective function to maximize)
    /// of taking decision `d` is when the system is in the given `state` and
    /// the given set of `vars` are still free (no value assigned).
    fn transition_cost(&self, state: &T, vars : &VarSet, d: Decision) -> isize;

    /// Optional method for the case where you'd want to use a pooled mdd
    /// implementation. This method returns true iff taking a decision on
    /// `variable` might have an impact (state or longest path) on a node
    /// having the given `state`.
    #[allow(unused_variables)]
    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        true
    }

    /// Returns a var set with all the variables of this problem.
    ///
    /// This method is (trivially) auto-implemented, but re-implementing it
    /// does not make much sense.
    fn all_vars(&self) -> VarSet {
        VarSet::all(self.nb_vars())
    }
}

/// This is the second most important abstraction that a client should provide
/// when using this library. It defines the relaxation that may be applied to
/// the given problem. In particular, the `merge_states` method from this trait
/// defines how the nodes of a layer may be combined to provide an upper bound
/// approximation standing for an arbitrarily selected set of nodes.
///
/// Again, the type parameter `<T>` denotes the type of the states.
pub trait Relaxation<T> {
    /// This method merges the given set of `states` into a new _inexact_ state
    /// that is an overapproximation of all `states`. The returned value will be
    /// used as a replacement for the given `states` in the mdd.
    ///
    /// In the theoretical framework of Bergman et al, this would amount to
    /// providing an implementation for the $\oplus$ operator. You should really
    /// only focus on that aspect when developing a relaxation: all the rest
    /// is taken care of by the framework.
    fn merge_states(&self, states: &mut dyn Iterator<Item=&T>) -> T;

    /// This method relaxes the weight of the edge between the nodes `src` and
    /// `dst` because `dst` is replaced in the current layer by `relaxed`. The
    /// `decision` labels and the original weight (`cost`) of the edge
    /// `src` -- `dst` are also recalled.
    ///
    /// In the theoretical framework of Bergman et al, this would amount to
    /// providing an implementation for the $\Gamma$ operator.
    fn relax_edge(&self, src: &T, dst: &T, relaxed: &T, decision: Decision, cost: isize) -> isize;

    /// This optional method derives a _rough upper bound_ (RUB) on the maximum
    /// value of the subproblem rooted in the given `_state`. By default, the
    /// RUB returns the greatest positive integer; which is always safe but does
    /// not provide any pruning.
    fn estimate  (&self, _state  : &T) -> isize {isize::max_value()}
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
mod test_problem_defaults {
    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::common::{Variable, VarSet, Domain, Decision};

    struct MockProblem;
    impl Problem<usize> for MockProblem {
        fn nb_vars(&self)       -> usize {  5 }
        fn initial_state(&self) -> usize { 42 }
        fn initial_value(&self) -> isize { 84 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            unimplemented!()
        }
        fn transition(&self, _: &usize, _: &VarSet, _: Decision) -> usize {
            unimplemented!()
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, _: Decision) -> isize {
            unimplemented!()
        }
    }
    struct MockRelax;
    impl Relaxation<usize> for MockRelax {
        fn merge_states(&self, _: &mut dyn Iterator<Item=&usize>) -> usize {
            unimplemented!()
        }
        fn relax_edge(&self, _: &usize, _: &usize, _: &usize, _: Decision, _: isize) -> isize {
            unimplemented!()
        }
    }

    #[test]
    fn by_default_all_vars_return_all_possible_variables(){
        assert_eq!(VarSet::all(5), MockProblem.all_vars());
    }

    #[test]
    fn the_default_rough_upper_bound_is_infinity() {
        assert_eq!(isize::max_value(), MockRelax.estimate(&12));
    }

    #[test]
    fn by_default_all_vars_impact_all_states() {
        assert!(MockProblem.impacted_by(&0, Variable(0)));
        assert!(MockProblem.impacted_by(&4, Variable(10)));
        assert!(MockProblem.impacted_by(&92, Variable(53)));
    }
}