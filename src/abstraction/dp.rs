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
    fn for_each_in_domain<F>(&self, var: Variable, state: &Self::State, f: F)
    where F: FnMut(Decision);
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