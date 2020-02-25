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

use ddo::core::abstraction::heuristics::VariableHeuristic;
use ddo::core::common::{Layer, Variable, VarSet};
use std::cmp::Ordering::Equal;

use crate::model::{Knapsack, KnapsackState};

#[derive(Clone)]
pub struct KnapsackOrder<'a> {
    pb: &'a Knapsack
}
impl <'a> KnapsackOrder<'a> {
    pub fn new(pb: &'a Knapsack) -> Self {
        KnapsackOrder{pb}
    }
}
impl VariableHeuristic<KnapsackState> for KnapsackOrder<'_> {
    fn next_var<'a>(&self, free_vars: &'a VarSet, _c: Layer<'a, KnapsackState>, _n: Layer<'a, KnapsackState>) -> Option<Variable> {
        free_vars.iter().max_by(|x, y| {
            let ix = &self.pb.data[x.id()];
            let kx = ix.profit as f32 / ix.weight as f32;

            let iy = &self.pb.data[y.id()];
            let ky = iy.profit as f32 / iy.weight as f32;

            kx.partial_cmp(&ky).unwrap_or(Equal)
        })
    }
}