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

//! This module and its sub modules provide the traits and structures required
//! to implement various flavors of bounded-width MDDs.

// ------------------------------------------------------------------------- //
// --- This enum does not need to be public, but it is convenient to use --- //
// --- in all implementations of an MDD. Still it looked like overkill   --- //
// --- to define an additional 'utility' module just to hold this enum   --- //
// ------------------------------------------------------------------------- //
/// This enumeration characterizes the kind of MDD being generated. It can
/// either be
/// * `Exact` if it is a true account of the problem state space.
/// * `Restricted` if it is an under approximation of the problem state space.
/// * `Relaxed` if it is an over approximation of the problem state space.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MDDType {
    Relaxed,
    Restricted,
    Exact
}
// ------------------------------------------------------------------------- //

pub mod utils;
pub mod config;
pub mod deep;
pub mod shallow;
pub mod hybrid;
pub mod aggressively_bounded;
