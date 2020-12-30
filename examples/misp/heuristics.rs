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

use bitset_fixed::BitSet;

use ddo::{
    Problem,
    Variable,
    VarSet,
    VariableHeuristic,
    LoadVars,
    FrontierNode,
    FrontierOrder,
    BitSetIter,
};

use crate::model::Misp;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct MispVarHeu(usize);
impl MispVarHeu {
    pub fn new(pb: &Misp) -> Self {
        MispVarHeu(pb.nb_vars())
    }
}
impl VariableHeuristic<BitSet> for MispVarHeu {
    fn next_var(&self,
                _: &VarSet,
                _: &mut dyn Iterator<Item=&BitSet>,
                next: &mut dyn Iterator<Item=&BitSet>) -> Option<Variable>
    {
        let mut counters = vec![0; self.0];

        for s in next {
            for v in BitSetIter::new(s) {
                counters[v] += 1;
            }
        }

        counters.iter().enumerate()
            .filter(|(_, &count)| count > 0)
            .min_by_key(|(_, &count)| count)
            .map(|(i, _)| Variable(i))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VarsFromMispState;
impl LoadVars<BitSet> for VarsFromMispState {
    fn variables(&self, node: &FrontierNode<BitSet>) -> VarSet {
        VarSet(node.state.as_ref().clone())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MispFrontierOrder;
impl FrontierOrder<BitSet> for MispFrontierOrder {
    fn compare(&self, a: &FrontierNode<BitSet>, b: &FrontierNode<BitSet>) -> Ordering {
        a.ub.cmp(&b.ub)
            .then_with(|| a.state.count_ones().cmp(&b.state.count_ones()))
            .then_with(|| a.lp_len.cmp(&b.lp_len))
    }
}
