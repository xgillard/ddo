//! This module defines the traits for the constituents of an MDD: the `MDD`
//! itself and the `Nodes` that compose it.
use crate::core::abstraction::dp::{Decision, Variable, VarSet};
use std::hash::Hash;
use std::collections::HashMap;

/// This is a node from a given `MDD`.
///
/// # Type param
/// The type parameter `<T>` denotes the type of the state defined/manipulated
/// by the `Problem` definition.
pub trait Node<T>
    where T : Clone + Hash + Eq  {
    /// Returns true iff this node is an exact node (none of its ancestors
    /// is a merged node).
    fn is_exact(&self) -> bool;
    /// Returns the state (as defined by the `Problem` and `Relaxation`
    /// associated to this node.
    fn get_state(&self)-> &T;
    /// Returns the length of the longest path to this node.
    fn get_lp_len(&self) -> i32;
    /// Returns an upper bound on the length of the longest path (from root to
    /// terminal node) passing through the current node.
    fn get_ub(&self) -> i32;
    /// Sets an upper bound `ub` on the length of the longest path (from root to
    /// terminal node) passing through the current node.
    fn set_ub(&mut self, ub: i32);
    /// Returns the list of decisions along the longest path between the
    /// root node and this node.
    fn longest_path(&self) -> Vec<Decision>;
}

/// This enumeration characterizes the kind of MDD being generated. It can
/// either be
/// * `Exact` if it is a true account of the problem state space.
/// * `Restricted` if it is an under approximation of the problem state space.
/// * `Relaxed` if it is an over approximation of the problem state space.
#[derive(Copy, Clone)]
pub enum MDDType {
    Relaxed,
    Restricted,
    Exact
}

/// This trait describes an MDD
///
/// # Type param
/// The type parameter `<T>` denotes the type of the state defined/manipulated
/// by the `Problem` definition.
///
/// The type parameter `<N>` denotes the type of nodes in use in this MDD.
pub trait MDD<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T> {

    /// Tells whether this MDD is exact, relaxed, or restricted.
    fn mdd_type(&self) -> MDDType;
    /// Returns the nodes from the current layer.
    fn current_layer(&self) -> &[N];
    /// Returns a set of nodes constituting an exact cutset of this `MDD`.
    fn exact_cutset(&self) -> &[N];
    /// Returns a the map State -> Node of nodes already recorded to participate
    /// in the next layer. (All these nodes will have to be expanded to explore
    /// the complete state space).
    fn next_layer(&self) -> &HashMap<T, N>;

    /// Returns the latest `Variable` that acquired a value during the
    /// development of this `MDD`.
    ///
    /// # Note
    /// This development might still be ongoing. That last variable will then
    /// simply be variable associated to the decisions from the previous layer.
    fn last_assigned(&self) -> Variable;
    /// Returns the set of variables that have not been assigned any value
    /// (so far!).
    fn unassigned_vars(&self) -> &VarSet;

    /// Return true iff this `MDD` is exact. That is to say, it returns true if
    /// no nodes have been merged (because of relaxation) or suppressed (because
    /// of restriction).
    fn is_exact(&self) -> bool;
    /// Returns the length of the longest path between the root and the terminal
    /// node of this `MDD`.
    fn best_value(&self) -> i32;
    /// Returns the terminal node having the longest associated path in this `MDD`.
    fn best_node(&self) -> &Option<N>;
    /// Returns the list of decisions along the longest path between the
    /// root node and the best terminal node of this `MDD`.
    fn longest_path(&self) -> Vec<Decision>;
}