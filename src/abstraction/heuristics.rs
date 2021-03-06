// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! This module defines the traits used to encapsulate solver heuristics.
//!
//! Namely, it defines :
//!
//!  - the `WidthHeuristic` which is used to control the maximum width of an MDD
//!  - the `VariableHeuristic` which is used to control the order in which the
//!    variables are selected (major impact on the size of an MDD).
//!  - the `LoadVars` which encapsulates a strategy to retrieve the set of
//!    unassigned variables from a given frontier node.
//!  - the `Cutoff` which encapsulates a criterion (external to the solver)
//!    which imposes to stop searching for a better solution. Typically, this is
//!    done to grant a given time budget to the search.

use crate::common::{FrontierNode, Variable, VarSet, MDDType};
use std::cmp::Ordering;

/// This trait defines an heuristic to determine the maximum allowed width of a
/// layer in a relaxed or restricted MDD.
pub trait WidthHeuristic {
    /// Returns the maximum width allowed for a layer.
    fn max_width(&self, mdd_type: MDDType, free_vars: &VarSet) -> usize;
}

/// This trait defines an heuristic to determine the best variable to branch on
/// while developing an MDD.
///
/// This trait defines some optional methods that allow you to implement an
/// incremental variable heuristic (stateful). The implementation of these
/// methods is definitely *not* mandatory. This is why the trait provides a
/// default (empty) implementation for these methods. In the event where these
/// methods are implemented, you may assume that each thread owns a copy of
/// the heuristic.
pub trait VariableHeuristic<T> {
    /// Returns the best variable to branch on from the set of `free_vars`
    /// or `None` in case no branching is useful (`free_vars` is empty, no decision
    /// left to make, etc...).
    fn next_var(&self,
                free_vars: &VarSet,
                current_layer: &mut dyn Iterator<Item=&T>,
                next_layer:  &mut dyn Iterator<Item=&T>) -> Option<Variable>;

    /// This method provides a hook for you to react to the addition of a new
    /// layer (to the mdd) during development of an mdd. This might be useful
    /// when implementing an incremental variable heuristic.
    ///
    /// *The implementation of this method is _OPTIONAL_*
    fn upon_new_layer(&mut self,
                      _var: Variable,
                      _current_layer: &mut dyn Iterator<Item=&T>){}

    /// This method provides a hook for you to react to the addition of a new
    /// node to the next layer of the mdd during development of the mdd.
    /// This might be useful when implementing an incremental variable heuristic.
    ///
    /// *The implementation of this method is _OPTIONAL_*
    fn upon_node_insert(&mut self, _state: &T) {}

    /// When implementing an incremental variable selection heuristic, this
    /// method should reset the state of the heuristic to a "fresh" state.
    /// This method is called at the start of the development of any mdd.
    ///
    /// *The implementation of this method is _OPTIONAL_*
    fn clear(&mut self) {}
}

/// This trait defines a minimal abstraction over the MDD nodes so that they
/// can easily (and transparently) be ordered by the `NodeSelectionHeuristic`.
pub trait SelectableNode<T> {
    /// Returns a reference to the state of this node.
    fn state (&self) -> &T;
    /// Returns the value of the objective function at this node. In other words,
    /// it returns the length of the longest path from root to this node.
    fn value(&self) -> isize;
    /// Returns true iff the node is an exact node (not merged, and has no
    /// merged ancestor).
    fn is_exact(&self) -> bool;
}

/// This trait defines an heuristic to rank the nodes in order to remove
/// (or merge) the less relevant ones from an MDD that is growing too large.
pub trait NodeSelectionHeuristic<T> {
    /// Defines an order of 'relevance' over the nodes `a` and `b`. Greater means
    /// that `a` is more important (hence more likely to be kept) than `b`.
    fn compare(&self, a: &dyn SelectableNode<T>, b: &dyn SelectableNode<T>) -> Ordering;
}

/// This trait defines an heuristic to rank the nodes on the solver fringe.
/// That is, it defines the order in which the frontier nodes are going to be
/// popped off the fringe (either sequentially or in parallel).
///
/// # Note:
/// Because of the assumption which is made on the frontier that all nodes be
/// popped off the frontier in decreasing upper bound order, it is obligatory
/// that any frontier order uses the upper bound as first criterion. Failing to
/// do so will result in the solver stopping to search for the optimal solution
/// too early.
pub trait FrontierOrder<T> {
    /// Defines the order in which the nodes are going to be popped off the
    /// fringe. If node `a` is `Greater` than node `b`, it means that node
    /// `a` should be popped off the fringe sooner than node `b`.
    ///
    /// # Note:
    /// Because of the assumption which is made on the frontier that all nodes be
    /// popped off the frontier in decreasing upper bound order, it is obligatory
    /// that any frontier order uses the upper bound as first criterion. Failing to
    /// do so will result in the solver stopping to search for the optimal solution
    /// too early.
    fn compare(&self, a: &FrontierNode<T>, b: &FrontierNode<T>) -> Ordering;
}

/// This trait defines a strategy/heuristic to retrieve the smallest set of free
/// variables from a given `node`
pub trait LoadVars<T> {
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    fn variables(&self, node: &FrontierNode<T>) -> VarSet;
}

/// This trait encapsulates a criterion (external to the solver) which imposes
/// to stop searching for a better solution. Typically, this is done to grant
/// a given time budget to the search.
pub trait Cutoff {
    /// Returns true iff the criterion is met and the search must stop.
    ///
    /// - lb is supposed to be the best known lower bound
    /// - ub is supposed to be the best known upper bound
    fn must_stop(&self, lb: isize, ub: isize) -> bool;
}