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

//! This module defines a layer of abstraction for the heuristics one will
//! use to customize the development of MDDs.
use crate::core::common::{Layer, Node, Variable, VarSet};

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
    fn next_var<'a>(&self, free_vars: &'a VarSet, current: Layer<'a, T>, next: Layer<'a, T>) -> Option<Variable>;
}

/// This trait defines a strategy/heuristic to retrieve the smallest set of free
/// variables from a given `node`
pub trait LoadVars<T> {
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    fn variables(&self, node: &Node<T>) -> VarSet;
}