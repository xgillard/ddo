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

use crate::common::{FrontierNode, Variable, VarSet};
use std::cmp::Ordering;

/// This trait defines an heuristic to determine the maximum allowed width of a
/// layer in a relaxed or restricted MDD.
pub trait WidthHeuristic {
    /// Returns the maximum width allowed for a layer.
    fn max_width(&self, free_vars: &VarSet) -> usize;
}

/// This trait defines an heuristic to determine the best variable to branch on
/// while developing an MDD.
pub trait VariableHeuristic<T> {
    /// Returns the best variable to branch on from the set of `free_vars`
    /// or `None` in case no branching is useful (`free_vars` is empty, no decision
    /// left to make, etc...).
    fn next_var(&self,
                free_vars: &VarSet,
                current_layer: &mut dyn Iterator<Item=&T>,
                next_layer:  &mut dyn Iterator<Item=&T>) -> Option<Variable>;
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