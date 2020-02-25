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

use ddo::core::abstraction::dp::Problem;
use ddo::core::abstraction::heuristics::VariableHeuristic;
use ddo::core::common::{Layer, Node, Variable, VarSet};
use ddo::core::utils::BitSetIter;

use crate::model::Misp;

#[derive(Debug, Clone)]
pub struct MispVarHeu(usize);
impl MispVarHeu {
    pub fn new(pb: &Misp) -> Self {
        MispVarHeu(pb.nb_vars())
    }
}
impl VariableHeuristic<BitSet> for MispVarHeu {
    fn next_var<'a>(&self, _: &'a VarSet, _: Layer<'a, BitSet>, next: Layer<'a, BitSet>) -> Option<Variable> {
        let mut counters = vec![0; self.0];

        for (s, _) in next {
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

pub fn vars_from_misp_state(n: &Node<BitSet>) -> VarSet {
    VarSet(n.state.clone())
}