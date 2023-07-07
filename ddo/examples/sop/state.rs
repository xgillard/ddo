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

//! This module defines the types used to encode the state of a node in the 
//! SOP problem.

use std::hash::Hash;

use crate::BitSet;

/// This represents a state of the problem
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SopState {
    /// This is the last job in the current sequence
    pub previous: Previous,
    /// These are the jobs still to be scheduled
    pub must_schedule: BitSet,
    /// These are jobs that might still need to be scheduled
    pub maybe_schedule: Option<BitSet>,
    /// This is the 'depth' in the sequence, the number of jobs that have
    /// already been scheduled
    pub depth: usize
}

/// This represents the last job in the current sequence
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum Previous {
    /// It can either be at an actual job
    Job(usize),
    /// Or it can be one among a pool of jobs 
    /// (relaxed node == job is Schrödinger's cat)
    Virtual(BitSet),
}
