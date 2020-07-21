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

//! This module defines the `Frontier` trait. That is the abstraction of the
//! solver's fringe (aka fringe, aka queue). This is the set of sub-problems
//! that must be treated before the problem is considered solved (by exhaustion).
//!
//! # Note:
//! The solvers rely on the assumption that a fringe will pop nodes in descending
//! upper bound order. Hence, it is a requirement for any fringe implementation
//! to enforce that requirement.


use crate::common::FrontierNode;

/// The `Frontier`. That is the abstraction of the solver's frontier (aka fringe,
/// aka queue). This is the set of sub-problems that must be treated before the
/// problem is considered solved (by exhaustion).
///
/// # Note:
/// The solvers rely on the assumption that a fringe will pop nodes in descending
/// upper bound order. Hence, it is a requirement for any fringe implementation
/// to enforce that requirement.
pub trait Frontier<T> {
    /// This is how you push a node onto the frontier.
    fn push(&mut self, node: FrontierNode<T>);
    /// This method yields the most promising node from the frontier.
    /// # Note:
    /// The solvers rely on the assumption that a frontier will pop nodes in
    /// descending upper bound order. Hence, it is a requirement for any fringe
    /// implementation to enforce that requirement.
    fn pop(&mut self) -> Option<FrontierNode<T>>;
    /// This method clears the frontier: it removes all nodes from the queue.
    fn clear(&mut self);
    /// Yields the length of the queue.
    fn len(&self) -> usize;
    /// Returns true iff the finge is empty (len == 0)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}