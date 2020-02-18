//! This module defines traits for implementations of an MDD.
use crate::core::common::{Decision, Node, NodeInfo};

/// This enumeration characterizes the kind of MDD being generated. It can
/// either be
/// * `Exact` if it is a true account of the problem state space.
/// * `Restricted` if it is an under approximation of the problem state space.
/// * `Relaxed` if it is an over approximation of the problem state space.
#[derive(Copy, Clone, Eq, PartialEq)]
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
pub trait MDD<T> {
    /// Tells whether this MDD is exact, relaxed, or restricted.
    fn mdd_type(&self) -> MDDType;

    /// Generates the root node of the problem
    fn root(&self) -> Node<T>;

    /// Expands this MDD into  an exact MDD
    fn exact(&mut self, root: &Node<T>, best_lb : i32);
    /// Expands this MDD into a restricted (lower bound approximation)
    /// version of the exact MDD.
    fn restricted(&mut self, root: &Node<T>, best_lb : i32);
    /// Expands this MDD into a relaxed (upper bound approximation)
    /// version of the exact MDD.
    fn relaxed(&mut self, root: &Node<T>, best_lb : i32);

    /// Iterates over the nodes from the cutset and applies the given function
    /// `f` to each pair of `(state, node_info)` present in the cutset of this
    /// MDD.
    fn for_each_cutset_node<F>(&mut self, f: F) where F: FnMut(&T, &mut NodeInfo);
    /// Consumes (removes) all nodes from the cutset of this mdd ands applies
    /// the given function `f` to each pair of `(state, node_info)` present in
    /// this mdd.
    ///
    /// # Note:
    /// Because the nodes are consumed, they are no longer available for use
    /// after a call to this method completes.
    ///
    /// All nodes from the cutset are considered to be used even though the
    /// function may decide to skip them. Hence, calling `for_each_cutset_node`
    /// after a call to this method completes will have absolutely no effect.
    fn consume_cutset<F>(&mut self, f: F) where F: FnMut(T, NodeInfo);

    /// Return true iff this `MDD` is exact. That is to say, it returns true if
    /// no nodes have been merged (because of relaxation) or suppressed (because
    /// of restriction).
    fn is_exact(&self) -> bool;
    /// Returns the length of the longest path between the root and the terminal
    /// node of this `MDD`.
    fn best_value(&self) -> i32;
    /// Returns the terminal node having the longest associated path in this `MDD`.
    fn best_node(&self) -> &Option<NodeInfo>;
    /// Returns the list of decisions along the longest path between the
    /// root node and the best terminal node of this `MDD`.
    fn longest_path(&self) -> Vec<Decision>;
}
