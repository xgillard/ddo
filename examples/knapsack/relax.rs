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
use ddo::core::abstraction::dp::{Problem, Relaxation};
use ddo::core::common::{Node, NodeInfo, VarSet};
use crate::model::{Knapsack, KnapsackState};

#[derive(Debug, Clone)]
pub struct KnapsackRelax<'a> {
    pb: &'a Knapsack
}
impl <'a> KnapsackRelax<'a> {
    pub fn new(pb: &'a Knapsack) -> Self { KnapsackRelax {pb} }
}
impl Relaxation<KnapsackState> for KnapsackRelax<'_> {
    fn merge_nodes(&self, nodes: &[Node<KnapsackState>]) -> Node<KnapsackState> {
        let mut lp_info   = &nodes[0].info;
        let mut capacity  = 0;
        let mut free_vars = BitSet::new(self.pb.nb_vars());
        for n in nodes.iter() {
            free_vars |= &n.state.free_vars.0;
            capacity   = capacity.max(n.state.capacity);

            if n.info.lp_len > lp_info.lp_len {
                lp_info = &n.info;
            }
        }

        let state = KnapsackState {capacity, free_vars: VarSet(free_vars)};
        Node { state, info : lp_info.clone() }
    }
    fn estimate_ub(&self, state: &KnapsackState, info: &NodeInfo) -> i32 {
        info.lp_len + state.free_vars.iter().map(|v| {
            let item      = &self.pb.data[v.id()];
            let max_amout = state.capacity / item.weight;
            max_amout * item.profit
        }).sum::<usize>() as i32
    }
}