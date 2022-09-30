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
//!  - the `StateRanking` heuristic which is used to guess the nodes promisingess
//!  - the `Cutoff` heuristic which is used to impose a stopping criterion on the
//!    solver resolution.

use std::cmp::Ordering;

/// This trait enclapsulates the behavior of the heuristic that determines
/// the maximum permitted width of a decision diagram.
///
/// # Technical Note:
/// Just like `Problem`, `Relaxation` and `StateRanking`, the `WidthHeuristic`
/// trait is generic over `State`s. However, rather than using the same 
/// 'assciated-type' mechanism that was used for the former three types, 
/// `WidthHeuristic` uses a parameter type for this purpose (the type parameter
/// approach might feel more familiar to Java or C++ programmers than the 
/// associated-type). 
///
/// This choice was motivated by two factors: 
/// 1. The `Problem`, `Relaxation` and `StateRanking` are intrinsically tied
///    to *one type* of state. And thus, the `State` is really a part of the 
///    problem/relaxation itself. Therefore, it would not make sense to define 
///    a generic problem implementation which would be applicable to all kind 
///    of states. Instead, an implementation of the `Problem` trait is the
///    concrete implementation of a DP model (same argument holds for the other 
///    traits). 
///
/// 2. On the other hand, it does make sense to define a `WidthHeuristic` 
///    implementation which is applicable regardless of the state of the problem
///    which is currently being solved. For instance, the ddo framework offers
///    the `Fixed` and `NbUnassigned` width heuristics which are independent of
///    the problem. The `Fixed` width heuristic imposes that the maxumum layer
///    width be constant accross all compiled DDs whereas `NbUnassigned` lets
///    the maximum width vary depending on the number of problem variables 
///    which have already been decided upon.
pub trait WidthHeuristic<State> {
    /// Estimates a good maximum width for an MDD rooted in the given state
    fn max_width(&self, state: &State) -> usize;
}

/// A state ranking is an heuristic that imposes a partial order on states.
/// This order is used by the framework as a means to discriminate the most
/// promising nodes from the least promising ones when restricting or relaxing
/// a layer from some given DD.
pub trait StateRanking {
    /// As is the case for `Problem` and `Relaxation`, a `StateRanking` must 
    /// tell the kind of states it is able to operate on.
    type State;

    /// This method compares two states and determines which is the most 
    /// desirable to keep. In this ordering, greater means better and hence
    /// more likely to be kept
    fn compare(&self, a: &Self::State, b: &Self::State) -> Ordering;
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