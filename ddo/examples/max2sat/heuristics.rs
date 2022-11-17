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

use std::cmp::Ordering;
use ddo::*;
use crate::model::State;


/// In addition to a problem definition and a relaxation (DP model and Relax), ddo requires
/// that we provide it with a `StateRanking`. This is an heuristic which is used to select 
/// the most and least promising nodes as a means to only delete/merge the *least* promising
/// nodes when compiling restricted and relaxed DDs.
#[derive(Debug, Clone)]
pub struct Max2SatRanking;
impl StateRanking for Max2SatRanking {
    type State = State;
    fn compare(&self, x: &State, y: &State) -> Ordering {
        let xrank = x.rank();
        let yrank = y.rank();
        xrank.cmp(&yrank)
    }
}