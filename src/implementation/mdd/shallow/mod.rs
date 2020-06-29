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

//! This module and its sub modules provide the structures required to implement
//! a _shallow_ MDDs (flat or pooled).
//!
//! A shallow MDD representation is one that only maintains a ”slice” of the
//! actual DD it stands for. This slice comprises only two layers from the MDD:
//! the current layer and the next layer. All the previous layers are forgotten
//! and the nodes belonging to the ”slice” only need to keep track of their
//! single best parent in order to be able to recover the solution maximizing
//! the objective function.
//!
//! # Note
//! This is a perfectly sound and memory efficient way of representing MDDs to
//! implement a branch-and-bound DDO solver. However, this memory efficiency
//! comes at a price: precious information is lost when the older layers are
//! forgotten. And this information loss is precisely what prevents a shallow
//! MDD representation from exploiting local bounds. So if you intend to use
//! local bounds -- and we advise you to do so, because the impact is sometimes
//! dramatic ! -- you should instead use the deep mdd representation which is
//! the default in this crate.

mod utils;
pub mod flat;
pub mod pooled;