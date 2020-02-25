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

use ddo::core::common::{Node, NodeInfo};
use ddo::core::abstraction::dp::{Problem, Relaxation};

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
    fn merge_nodes(&self, nodes: &[Node<BitSet>]) -> Node<BitSet> {
        let mut state = BitSet::new(self.pb.nb_vars());
        let mut best  = &nodes[0].info;

        for node in nodes.iter() {
            state |= &node.state;

            if node.info.lp_len > best.lp_len {
                best = &node.info;
            }
        }

        let mut info = best.clone();
        info.is_exact = false;
        Node {state, info}
    }
    fn estimate_ub(&self, state: &BitSet, info: &NodeInfo) -> i32 {
        info.lp_len + state.count_ones() as i32
    }
}