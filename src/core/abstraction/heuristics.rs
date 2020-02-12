//! This module defines a layer of abstraction for the heuristics one will
//! use to customize the development of MDDs.
use crate::core::common::{Node, Variable, VarSet};
use crate::core::abstraction::mdd::Layer;

/// This trait defines an heuristic to determine the maximum allowed width of a
/// layer in a relaxed or restricted MDD.
pub trait WidthHeuristic<T> {
    /// Returns the maximum width allowed for a layer.
    fn max_width(&self, free_vars: &VarSet) -> usize;
}

/// This trait defines an heuristic to determine the best variable to branch on
/// while developing an MDD.
pub trait VariableHeuristic<T> where T: Clone + Eq {
    /// Returns the best variable to branch on from the set of `free_vars`
    /// or `None` in case no branching is useful (`free_vars` is empty, no decision
    /// left to make, etc...).
    fn next_var<'a>(&self, free_vars: &'a VarSet, current: Layer<'a, T>, next: Layer<'a, T>) -> Option<Variable>;
}

/// This trait defines a strategy/heuristic to retrieve the smallest set of free
/// variables from a given `node`
pub trait LoadVars<T>
    where T: Clone + Eq {
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    fn variables(&self, node: &Node<T>) -> VarSet;
}