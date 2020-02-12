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

pub enum Layer<'a, T> where T: Eq + Clone {
    Plain (std::slice::Iter<'a, Node<T>>),
    Mapped(std::collections::hash_map::Iter<'a, T, NodeInfo<T>>),
}

impl <'a, T> Iterator for Layer<'a, T> where T: Eq + Clone {
    type Item = (&'a T, &'a NodeInfo<T>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Layer::Plain(i)  => i.next().map(|n| (&n.state, &n.info)),
            Layer::Mapped(i) => i.next()
        }
    }
}

/// This trait describes an MDD
///
/// # Type param
/// The type parameter `<T>` denotes the type of the state defined/manipulated
/// by the `Problem` definition.
pub trait MDD<T> where T : Clone + Eq {
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

    /// Returns a set of nodes constituting an exact cutset of this `MDD`.
    fn for_each_cutset_node<F>(&mut self, f: F) where F: FnMut(&T, &mut NodeInfo<T>);
    /// Consumes the cutset of this mdd.
    fn consume_cutset<F>(&mut self, f: F) where F: FnMut(T, NodeInfo<T>);

    /// Return true iff this `MDD` is exact. That is to say, it returns true if
    /// no nodes have been merged (because of relaxation) or suppressed (because
    /// of restriction).
    fn is_exact(&self) -> bool;
    /// Returns the length of the longest path between the root and the terminal
    /// node of this `MDD`.
    fn best_value(&self) -> i32;
    /// Returns the terminal node having the longest associated path in this `MDD`.
    fn best_node(&self) -> &Option<NodeInfo<T>>;
    /// Returns the list of decisions along the longest path between the
    /// root node and the best terminal node of this `MDD`.
    fn longest_path(&self) -> Vec<Decision>;
}