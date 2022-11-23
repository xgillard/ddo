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

use crate::SubProblem;


/// This trait abstracts away the implementation details of the solver fringe.
/// That is, a Fringe represents the global priority queue which stores all 
/// the nodes remaining to explore.
pub trait Fringe {
    type State;

    /// This is how you push a node onto the fringe.
    fn push(&mut self, node: SubProblem<Self::State>);
    /// This method yields the most promising node from the fringe.
    /// # Note:
    /// The solvers rely on the assumption that a fringe will pop nodes in
    /// descending upper bound order. Hence, it is a requirement for any fringe
    /// implementation to enforce that requirement.
    fn pop(&mut self) -> Option<SubProblem<Self::State>>;
    /// This method clears the fringe: it removes all nodes from the queue.
    fn clear(&mut self);
    /// Yields the length of the queue.
    fn len(&self) -> usize;
    /// Returns true iff the fringe is empty (len == 0)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}