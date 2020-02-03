//! This module defines a layer of abstraction for the heuristics one will
//! use to customize the development of MDDs.
use crate::core::abstraction::mdd::{MDD, Node};
use std::hash::Hash;
use crate::core::abstraction::dp::{Variable, VarSet};

/// This trait defines an heuristic to determine the maximum allowed width of a
/// layer in a relaxed or restricted MDD.
pub trait WidthHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T> {
    /// Returns the maximum width allowed for the next layer of the given `dd`.
    fn max_width(&self, dd: &dyn MDD<T, N>) -> usize;
}

/// This trait defines an heuristic to determine the best variable to branch on
/// while developing a given MDD.
pub trait VariableHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T> {
    /// Returns the best variable to branch on from the set of free `vars`
    /// or `None` in case no branching is useful (`vars` is empty, no decision
    /// left to make, etc...).
    fn next_var(&self, dd: &dyn MDD<T, N>, vars: &VarSet) -> Option<Variable>;
}