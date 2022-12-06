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

use std::sync::Arc;

use crate::{Threshold, SubProblem, Problem};

/// This trait abstracts away the implementation details of the solver barrier.
/// That is, a Barrier represents the data structure that stores thresholds that
/// condition the re-exploration of nodes with a state already reached previously.
pub trait Barrier {
    type State;

    /// Returns true if the subproblem still must be explored,
    /// given the thresholds contained in the barrier.
    fn must_explore(&self, subproblem: &SubProblem<Self::State>) -> bool {
        let threshold = self.get_threshold(subproblem.state.as_ref(), subproblem.depth);
        if let Some(threshold) = threshold {
            subproblem.value > threshold.value || (subproblem.value == threshold.value && !threshold.explored)
        } else {
            true
        }
    }

    /// Prepare the barrier to be used with the given problem
    fn initialize(&mut self, problem: &dyn Problem<State = Self::State>);

    /// Returns the threshold currently associated with the given state, if any.
    fn get_threshold(&self, state: &Self::State, depth: usize) -> Option<Threshold>;

    /// Updates the threshold associated with the given state, only if it is increased.
    fn update_threshold(&self, state: Arc<Self::State>, depth: usize, value: isize, explored: bool);

    /// Removes all thresholds associated with states at the given depth.
    fn clear_layer(&self, depth: usize);

    /// Clears the data structure.
    fn clear(&self);
    
}