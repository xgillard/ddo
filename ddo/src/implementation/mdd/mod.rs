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

//! This module provides an implementation of the mdd data structure. 
//! After this API revamping, I have decided to only include the VectorBased MDD
//! (aka DefaultMDD). But if there is a need, I might decide to bring back the 
//! flat and pooled MDD implementations as well in the future. 
//! 
//! # Note: 
//! PooledMDD was the version working best on the Maximum Independent Set Problem 
//! (MISP). If this is the problem you want to solve, you might want to stick with
//! a previous version of ddo (<= 0.5.0).
mod node_flags;
mod clean;
mod pooled;

pub use node_flags::*;
pub use clean::*;
pub use pooled::*;

use crate::{LAST_EXACT_LAYER, FRONTIER};

/// By default, the mdd implementation which you will want to use is the vector based
/// implementation. In most cases, it is faster than everything else I have tried.
/// So having a alias calling it the "default" DD implem seems to make sense.
pub type DefaultMDD<T> = DefaultMDDLEL<T>;

/// By default, the mdd implementation which you will want to use is the vector based
/// implementation. In most cases, it is faster than everything else I have tried.
/// So having a alias calling it the "default" DD implem seems to make sense.
/// 
/// This is the variant implementation that produces a last exact layer cutset when asked
pub type DefaultMDDLEL<T> = Mdd<T, LAST_EXACT_LAYER>;

/// By default, the mdd implementation which you will want to use is the vector based
/// implementation. In most cases, it is faster than everything else I have tried.
/// So having a alias calling it the "default" DD implem seems to make sense.
/// 
/// This is the variant implementation that produces a frontier cutset when asked
pub type DefaultMDDFC<T> = Mdd<T, FRONTIER>;