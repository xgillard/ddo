//! This module defines a layer of abstraction for the heuristics one will
//! use to customize the development of MDDs.
use crate::core::abstraction::mdd::{MDD, Node};
use std::hash::Hash;
use crate::core::abstraction::dp::{Variable, VarSet, Problem};
use compare::Compare;

/// This trait defines an heuristic to determine the maximum allowed width of a
/// layer in a relaxed or restricted MDD.
pub trait WidthHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T> {
    /// Returns the maximum width allowed for the next layer of the given `dd`.
    fn max_width(&self, dd: &impl MDD<T, N>) -> usize;
}

/// This trait defines an heuristic to determine the best variable to branch on
/// while developing a given MDD.
pub trait VariableHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T> {
    /// Returns the best variable to branch on from the set of free `vars`
    /// or `None` in case no branching is useful (`vars` is empty, no decision
    /// left to make, etc...).
    fn next_var(&self, dd: &impl MDD<T, N>, vars: &VarSet) -> Option<Variable>;
}

/// This heuristic/strategy defines an order on the nodes. It is used to rank
/// the nodes during relaxation and restriction. In those cases, only the N best
/// nodes are kept. It is also possibly used to rank nodes in the fringe, and
/// hence define the order in which cutset nodes are explored.
pub trait NodeOrdering<T, N> : Compare<N>
    where T : Clone + Hash + Eq,
          N : Node<T> {
    //fn cmp(&self, a: &N, b: &N) -> Ordering;
}

/// This trait defines a strategy/heuristic to retrieve the smallest set of free
/// variables from a given `node`, for some given `problem`.
pub trait LoadVars<T, P, N>
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          N: Node<T> {
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    fn variables(&self, pb: &P, node: &N) -> VarSet;
}