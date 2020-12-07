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

//! The prelude module is only present to ease your life while developing a new
//! solver from scratch. That way you don't have to care about manually
//! importing all structs and traits by yourself.
//!
//! # Example
//! ```
//! // At the beginning of any file of your solver (or in your own prelude) you
//! // will be willing to either import all types from the ddo prelude or to
//! // re-export them.
//!
//! // if your intention is to only iumport the types
//! use ddo::prelude::*;
//! // or if you you want to re-export all these types
//! pub use ddo::prelude::*;
//! ```

pub use crate::common::*;

// Abstractions
pub use crate::abstraction::dp::*;
pub use crate::abstraction::mdd::*;
pub use crate::abstraction::heuristics::*;
pub use crate::abstraction::frontier::*;
pub use crate::abstraction::solver::*;

// Implementations
pub use crate::implementation::mdd::{
      config::*,
      deep::mdd::DeepMDD,
      shallow::flat::FlatMDD,
      shallow::pooled::PooledMDD,
      hybrid::{CompositeMDD,HybridFlatDeep,HybridPooledDeep},
      aggressively_bounded::AggressivelyBoundedMDD,
};
pub use crate::implementation::heuristics::*;
pub use crate::implementation::frontier::*;
pub use crate::implementation::solver::sequential::SequentialSolver;
pub use crate::implementation::solver::parallel::ParallelSolver;

// And because that's convenient to import while writing tests too
#[cfg(test)]
pub use crate::test_utils::*;