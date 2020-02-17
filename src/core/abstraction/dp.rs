//! This module defines the basic abstractions one will manipulate when
//! implementing an MDD optimization-solver for some problem formulated as a
//! dynamic program.
//!
//! The most important abstractions that should be provided by a client are
//! `Problem` and `Relaxation`.
use std::i32;

use crate::core::common::{Decision, Variable, VarSet, Node, NodeInfo, Domain};

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
    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a>;
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

    fn root_node(&self) -> Node<T> {
        Node::new(self.initial_state(), self.initial_value(), None, true)
    }
    fn all_vars(&self) -> VarSet {
        VarSet::all(self.nb_vars())
    }
}

/// This is the second most important abstraction that a client should provide
/// when using this library. It defines the relaxation that may be applied to
/// the given problem.
///
/// Again, the type parameter `<T>` denotes the type of the states.
pub trait Relaxation<T> {
    fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T>;
    fn estimate_ub(&self, _state: &T, _info: &NodeInfo) -> i32 {
        i32::max_value()
    }
}