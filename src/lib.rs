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

//! # DDO
//! DDO is a truly generic framework to develop MDD-based combinatorial
//! optimization solvers in Rust. Its goal is to let you describe your
//! optimization problem as a dynamic program (see `Problem`) along with a
//! `Relaxation`. When the dynamic program of the problem is considered as a
//! transition system, the relaxation serves the purpose of merging different
//! nodes of the transition system into an other node standing for them all.
//! In that setup, the sole condition to ensure the correctness of the
//! optimization algorithm is that the replacement node must be an over
//! approximation of all what is feasible from the merged nodes.
//!
//! ## Side benefit
//! As a side benefit from using `ddo`, you will be able to exploit all of your
//! hardware to solve your optimization in parallel.
pub mod core;

#[cfg(test)]
pub mod test_utils;