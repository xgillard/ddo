//! This module provides the declaration of a `Config` trait which is used to
//! parameterize the behavior of various MDD implementations.
use std::cmp::Ordering;
use std::sync::Arc;

use crate::core::common::{Decision, Domain, Layer, Node, NodeInfo, Variable};

/// The config trait describes the configuration of an MDD. In other words, it
/// encapsulates the configurable behavior (problem, relaxation, heuristics,..)
/// of an MDD. Such a configuration is typically obtained from a builder.
pub trait Config<T> {
    /// Yields the root node of the (exact) MDD standing for the problem to solve.
    fn root_node(&self) -> Node<T>;
    /// This method returns true iff taking a decision on `variable` might
    /// have an impact (state or longest path) on a node having the given `state`
    fn impacted_by(&self, state: &T, v: Variable) -> bool;
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    fn load_vars (&mut self, root: &Node<T>);
    /// Returns the number of variables which may still be decided upon in this
    /// unrolling of the (approximate) MDD. Note: This number varies during the
    /// unrolling of the MDD. Whenever you call `remove_var`, this number should
    /// decrease.
    fn nb_free_vars(&self) -> usize;
    /// Returns the best variable to branch on from the set of 'free' variables
    /// (variables that may still be branched upon in this unrolling of the MDD).
    /// It returns `None` in case no branching is useful (ie when no decision is
    /// left to make, etc...).
    fn select_var(&self, current: Layer<'_, T>, next: Layer<'_, T>) -> Option<Variable>;
    /// Removes `v` from the set of free variables (which may be branched upon).
    /// That is, it alters the configuration so that `v` is considered to have
    /// been assigned with a value. Note: As a side effect, it should decrease
    /// the value of `nb_free_vars` and make `v` impossible to select (during
    /// this unrolling of the MDD) with `select_var`.
    fn remove_var(&mut self, v: Variable);
    /// Returns the domain of variable `v` in the given `state`. These are the
    /// possible values that might possibly be affected to `v` when the system
    /// has taken decisions leading to `state`.
    fn domain_of<'a>(&self, state: &'a T, v: Variable) -> Domain<'a>;
    /// Returns the maximum width allowed for the next layer in this unrolling
    /// of the MDD.
    fn max_width(&self) -> usize;
    /// Returns the node which is reached by taking decision `d` from the node
    /// (`state`, `info`).
    fn branch(&self, state: &T, info: Arc<NodeInfo>, d: Decision) -> Node<T>;
    /// Returns a _rough_ upper bound on the maximum objective function value
    /// reachable by passing through the node characterized by the given
    /// `state` and node `info`.
    fn estimate_ub(&self, state: &T, info: &NodeInfo) -> i32;
    /// Compares two nodes according to the node selection ranking. That is,
    /// it derives an ordering for `x` and `y` where the Gretest node has more
    /// chances of remaining in the layer (in case a restriction or merge needs
    /// to occur). In other words, if `x > y` according to this ordering, it
    /// means that `x` is more promising than `y`. A consequence of which,
    /// `x` has more chances of not being dropped/merged into an other node.
    fn compare(&self, x: &Node<T>, y: &Node<T>) -> Ordering;
    /// This method merges the given set of `nodes` into a new _inexact_ node.
    /// The role of this method is really to _only_ provide an inexact
    /// node to use as a replacement for the selected `nodes`. It is the MDD
    /// implementation 's responsibility to take care of maintaining a cutset.
    fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T>;
}