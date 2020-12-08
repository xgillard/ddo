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

use ddo::{Problem, Relaxation, Decision, BitSetIter};

use crate::model::Misp;

#[derive(Debug, Clone)]
pub struct MispRelax<'a> {
    pb: &'a Misp
}

impl <'a> MispRelax<'a> {
    pub fn new(pb : &'a Misp) -> MispRelax<'a> {
        MispRelax{pb}
    }
}
impl <'a> Relaxation<BitSet> for MispRelax<'a> {
    fn merge_states(&self, states: &mut dyn Iterator<Item=&BitSet>) -> BitSet {
        let mut relaxed = BitSet::new(self.pb.nb_vars());

        for state in states {
            relaxed |= state;
        }

        relaxed
    }
    fn relax_edge(&self, _: &BitSet, _: &BitSet, _: &BitSet, _: Decision, cost: isize) -> isize {
        cost
    }
    fn estimate  (&self, state  : &BitSet) -> isize {
        BitSetIter::new(state).map(|i| 0.max(self.pb.graph.weights[i])).sum()
    }
}
