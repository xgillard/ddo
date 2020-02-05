//! This module defines the basic abstractions one will manipulate when
//! implementing an MDD optimization-solver for some problem formulated as a
//! dynamic program.
//!
//! The most important abstractions that should be provided by a client are
//! `Problem` and `Relaxation`.
use std::i32;

use crate::core::abstraction::mdd::MDD;
use crate::core::common::{Decision, Variable, VarSet};

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
    fn initial_value(&self) -> i32;

    /// Returns the domain of variable `var` in the given `state`. These are the
    /// possible values that might possibly be affected to `var` when the system
    /// has taken decisions leading to `state`.
    fn domain_of(&self, state: &T, var: Variable) -> &[i32];
    /// Returns the next state reached by the system if the decision `d` is
    /// taken when the system is in the given `state` and the given set of `vars`
    /// are still free (no value assigned).
    fn transition(&self, state: &T, vars : &VarSet, d: Decision) -> T;
    /// Returns the marginal benefit (in terms of objective function to maximize)
    /// of taking decision `d` is when the system is in the given `state` and
    /// the given set of `vars` are still free (no value assigned).
    fn transition_cost(&self, state: &T, vars : &VarSet, d: Decision) -> i32;

    /// Optional method for the case where you'd want to use a pooled mdd implementation.
    /// Returns true iff taking a decision on 'variable' might have an impact (state or lp)
    /// on a node having the given 'state'
    #[allow(unused_variables)]
    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        true
    }
}

/// This is the second most important abstraction that a client should provide
/// when using this library. It defines the relaxation that may be applied to
/// the given problem.
///
/// Again, the type parameter `<T>` denotes the type of the states.
pub trait Relaxation<T> {
    /// Merges the given `states` into a relaxed one which is an over
    /// approximation of the given `states`.
    /// The return value is the over approximation state.
    fn merge_states(&self, dd: &dyn MDD<T>, states: &[&T]) -> T;
    /// This method yields the _relaxed cost_ of taking the given `decision`
    /// in state `from` to reach the relaxed state `to`.
    fn relax_cost(&self, dd: &dyn MDD<T>, original_cost: i32, from: &T, to: &T, decision: Decision) -> i32;

    /// Optionally compute a rough upper bound on the objective value reachable
    /// from the given state. This method should be *fast* to compute and return
    /// an upper bound on the length of the longest path passing through state
    /// `s` (and assuming that the length of the longest path to `s` is `lp` long).
    ///
    /// Returning `i32::max_value()` is always correct, but it will prevent any
    /// rough upper bound pruning to occur.
    #[allow(unused_variables)]
    fn rough_ub(&self, lp: i32, s: &T) -> i32 {
        i32::max_value()
    }
}

/// Any reference to a problem should be considered as a valid problem type.
impl <T, P: Problem<T>> Problem<T> for &P {
    fn nb_vars(&self) -> usize {
        (*self).nb_vars()
    }
    fn initial_state(&self) -> T {
        (*self).initial_state()
    }
    fn initial_value(&self) -> i32 {
        (*self).initial_value()
    }
    fn domain_of(&self, state: &T, var: Variable) -> &[i32] {
        (*self).domain_of(state, var)
    }
    fn transition(&self, state: &T, vars: &VarSet, d: Decision) -> T {
        (*self).transition(state, vars, d)
    }
    fn transition_cost(&self, state: &T, vars: &VarSet, d: Decision) -> i32 {
        (*self).transition_cost(state, vars, d)
    }
    fn impacted_by(&self, state: &T, var: Variable) -> bool {
        (*self).impacted_by(state, var)
    }
}
/// Any reference to a relaxation should be considered as as valid relaxation type.
impl <T, R: Relaxation<T>> Relaxation<T> for &R {
    fn merge_states(&self, dd: &dyn MDD<T>, states: &[&T]) -> T {
        (*self).merge_states(dd, states)
    }
    fn relax_cost(&self, dd: &dyn MDD<T>, cost: i32, from: &T, to: &T, d: Decision) -> i32 {
        (*self).relax_cost(dd, cost, from, to, d)
    }
    fn rough_ub(&self, lp: i32, s: &T) -> i32 {
        (*self).rough_ub(lp, s)
    }
}