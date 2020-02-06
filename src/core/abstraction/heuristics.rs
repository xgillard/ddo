//! This module defines a layer of abstraction for the heuristics one will
//! use to customize the development of MDDs.
use crate::core::abstraction::dp::Problem;
use crate::core::abstraction::mdd::{MDD, Node};
use crate::core::common::{Variable, VarSet};

/// This trait defines an heuristic to determine the maximum allowed width of a
/// layer in a relaxed or restricted MDD.
pub trait WidthHeuristic<T> {
    /// Returns the maximum width allowed for the next layer of the given `dd`.
    fn max_width(&self, dd: &dyn MDD<T>) -> usize;
}

/// This trait defines an heuristic to determine the best variable to branch on
/// while developing a given MDD.
pub trait VariableHeuristic<T> {
    /// Returns the best variable to branch on from the set of free `vars`
    /// or `None` in case no branching is useful (`vars` is empty, no decision
    /// left to make, etc...).
    fn next_var(&self, dd: &dyn MDD<T>, vars: &VarSet) -> Option<Variable>;
}

/// This trait defines a strategy/heuristic to retrieve the smallest set of free
/// variables from a given `node`, for some given `problem`.
pub trait LoadVars<T, P>
    where T: Clone + Eq,
          P: Problem<T> {
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    fn variables(&self, pb: &P, node: &Node<T>) -> VarSet;
}