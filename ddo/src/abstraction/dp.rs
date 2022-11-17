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

use crate::{Variable, Decision};

/// This trait defines the "contract" of what defines an optimization problem
/// solvable with the branch-and-bound with DD paradigm. An implementation of
/// this trait effectively defines a DP formulation of the problem being solved.
/// That DP model is envisioned as a labeled transition system -- which makes
/// it more amenable to DD compilation.
pub trait Problem {
    /// The DP model of the problem manipulates a state which is user-defined.
    /// Any type implementing Problem must thus specify the type of its state.
    type State;
    /// Any problem bears on a number of variable $x_0, x_1, x_2, ... , x_{n-1}$
    /// This method returns the value of the number $n$
    fn nb_variables(&self) -> usize;
    /// This method returns the initial state of the problem (the state of $r$).
    fn initial_state(&self) -> Self::State;
    /// This method returns the intial value $v_r$ of the problem
    fn initial_value(&self) -> isize;
    /// This method is an implementation of the transition function mentioned
    /// in the mathematical model of a DP formulation for some problem.
    fn transition(&self, state: &Self::State, decision: Decision) -> Self::State;
    /// This method is an implementation of the transition cost function mentioned
    /// in the mathematical model of a DP formulation for some problem.
    fn transition_cost(&self, state: &Self::State, decision: Decision) -> isize;
    /// Any problem needs to be able to specify an ordering on the variables
    /// in order to decide which variable should be assigned next. This choice
    /// is an **heuristic** choice. The variable ordering does not need to be
    /// fixed either. It may depend on the nodes constitutive of the next layer.
    /// These nodes are made accessible to this method as an iterator.
    fn next_variable(&self, next_layer: &mut dyn Iterator<Item = &Self::State>)
        -> Option<Variable>;
    /// This method calls the function `f` for any value in the domain of 
    /// variable `var` when in state `state`.  The function `f` is a function
    /// (callback, closure, ..) that accepts one decision.
    fn for_each_in_domain(&self, var: Variable, state: &Self::State, f: &mut dyn DecisionCallback);
    /// This method returns false iff this node can be moved forward to the next
    /// layer without making any decision about the variable `_var`.
    /// When that is the case, a default decision is to be assumed about the 
    /// variable. Implementing this method is only ever useful if you intend to 
    /// compile a decision diagram that comprises long arcs.
    fn is_impacted_by(&self, _var: Variable, _state: &Self::State) -> bool {
        true
    }
}

/// A relaxation encapsulates the relaxation $\Gamma$ and $\oplus$ which are
/// necessary when compiling relaxed DDs. These operators respectively relax
/// the weight of an arc towards a merged node, and merges the staet of two or 
/// more nodes so as to create a new inexact node.
pub trait Relaxation {
    /// Similar to the DP model of the problem it relaxes, a relaxation operates
    /// on a set of states (the same as the problem). 
    type State;

    /// This method implements the merge operation: it combines several `states`
    /// and yields a new state which is supposed to stand for all the other
    /// merged states. In the mathematical model, this operation was denoted
    /// with the $\oplus$ operator.
    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State;
    
    /// This method relaxes the cost associated to a particular decision. It
    /// is called for any arc labeled `decision` whose weight needs to be 
    /// adjusted because it is redirected from connecting `src` with `dst` to 
    /// connecting `src` with `new`. In the mathematical model, this operation
    /// is denoted by the operator $\Gamma$.
    fn relax(
        &self,
        source: &Self::State,
        dest: &Self::State,
        new: &Self::State,
        decision: Decision,
        cost: isize,
    ) -> isize;

    /// Returns a very rough estimation (upper bound) of the optimal value that 
    /// could be reached if state were the initial state
    fn fast_upper_bound(&self, _state: &Self::State) -> isize {
        isize::MAX
    }
}

/// This trait basically defines a callback which is passed on to the problem
/// so as to let it efficiently enumerate the domain values of some given 
/// variable.
pub trait DecisionCallback {
    /// executes the callback using the given decision
    fn apply(&mut self, decision: Decision);
}
/// The simplest and most natural callback implementation is to simply use
/// a closure.
impl <X: FnMut(Decision)> DecisionCallback for X {
    fn apply(&mut self, decision: Decision) {
        self(decision)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Relaxation, DecisionCallback, Decision, Problem};
    
    #[test]
    fn by_default_fast_upperbound_yields_positive_max() {
        let rlx = DummyRelax;
        assert_eq!(isize::MAX, rlx.fast_upper_bound(&'x'));
    }
    #[test]
    fn by_default_all_states_are_impacted_by_all_vars() {
        let pb = DummyProblem;
        assert!(pb.is_impacted_by(crate::Variable(10), &'x'));
    }

    #[test]
    fn any_closure_is_a_decision_callback() {
        let mut changed = false;
        let chg = &mut changed;
        let closure: &mut dyn DecisionCallback = &mut |_: Decision| {
            *chg = true;
        };

        closure.apply(Decision{variable: crate::Variable(0), value: 4});
        
        assert!(changed);
    }

    struct DummyProblem;
    impl Problem for DummyProblem {
        type State = char;

        fn nb_variables(&self) -> usize {
            todo!()
        }
        fn initial_state(&self) -> Self::State {
            todo!()
        }
        fn initial_value(&self) -> isize {
            todo!()
        }
        fn transition(&self, _: &Self::State, _: Decision) -> Self::State {
            todo!()
        }
        fn transition_cost(&self, _: &Self::State, _: Decision) -> isize {
            todo!()
        }
        fn next_variable(&self, _: &mut dyn Iterator<Item = &Self::State>)
            -> Option<crate::Variable> {
            todo!()
        }
        fn for_each_in_domain(&self, _: crate::Variable, _: &Self::State, _: &mut dyn DecisionCallback) {
            todo!()
        }
    }
    struct DummyRelax;
    impl Relaxation for DummyRelax {
        type State = char;

        fn merge(&self, _states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
            todo!()
        }

        fn relax(
            &self,
            _source: &Self::State,
            _dest: &Self::State,
            _new: &Self::State,
            _decision: crate::Decision,
            _cost: isize,
        ) -> isize {
            todo!()
        }
    }
}