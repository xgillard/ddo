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

use ddo::core::abstraction::dp::{Problem, Relaxation};
use ddo::core::common::{Node, NodeInfo, Variable, VarSet};
use std::cmp::Ordering;

use crate::model::{Mcp, McpState};

#[derive(Debug, Clone)]
pub struct McpRelax<'a> {
    pb  : &'a Mcp,
    vars: VarSet
}
impl <'a> McpRelax<'a> {
    pub fn new(pb: &'a Mcp) -> Self { McpRelax{pb, vars: pb.all_vars()} }
}
impl Relaxation<McpState> for McpRelax<'_> {
    fn merge_nodes(&self, nodes: &[Node<McpState>]) -> Node<McpState> {
        let relaxed_state  = self.merge_states(nodes);
        let (lp, via_node) = self.relax_cost(nodes, &relaxed_state);

        Node {
            state: relaxed_state,
            info : NodeInfo {
                is_exact  : false,
                is_relaxed: true,
                lp_len    : lp,
                lp_arc    : via_node.info.lp_arc.clone(),
                ub        : via_node.info.ub
            }
        }
    }
}

// private methods
impl McpRelax<'_> {

    const POSITIVE: u8 = 1;
    const NEGATIVE: u8 = 2;
    const BOTH    : u8 = McpRelax::POSITIVE + McpRelax::NEGATIVE;

    fn merge_states(&self, nodes: &[Node<McpState>]) -> McpState {
        let mut data = vec![0; self.pb.nb_vars()];

        for v in self.vars.iter() {
            data[v.id()] = self.merge_substates(v, nodes);
        }

        McpState{ initial: false, benef: data }
    }
    fn relax_cost<'n>(&self, nodes: &'n [Node<McpState>], merged: &McpState) -> (i32, &'n Node<McpState>) {
        let mut relaxed_costs = nodes.iter().map(|n| n.info.lp_len).collect::<Vec<i32>>();

        for v in self.vars.iter() {
            for j in 0..nodes.len() {
                relaxed_costs[j] += self.difference_of_abs_benefit(v, &nodes[j].state, merged);
            }
        }

        let mut best    = 0;
        let mut longest = relaxed_costs[0];
        for (node_id, cost) in relaxed_costs.iter().cloned().enumerate() {
            if cost > longest {
                best   = node_id;
                longest= cost;
            }
        }

        (longest, &nodes[best])
    }

    fn merge_substates(&self, v: Variable, nodes: &[Node<McpState>]) -> i32 {
        match self.substate_signs(v, nodes) {
            McpRelax::POSITIVE =>  self.minimum_substate(v, nodes),              // min( u_l )
            McpRelax::NEGATIVE => -self.minimum_abs_value_of_substate(v, nodes), //-min(|u_l|)
            _ => 0 // otherwise
        }
    }

    fn substate_signs(&self, v: Variable, nodes: &[Node<McpState>]) -> u8 {
        let mut signs = 0_u8;
        for node in nodes.iter() {
            let substate = node.state.benef[v.id()];
            match substate.cmp(&0) {
                Ordering::Less    => signs |= McpRelax::NEGATIVE,
                Ordering::Greater => signs |= McpRelax::POSITIVE,
                Ordering::Equal   => /* do nothing */()
            }

            // short circuit
            if signs == McpRelax::BOTH { return signs; }
        }
        signs
    }

    fn minimum_substate(&self, v: Variable, nodes: &[Node<McpState>]) -> i32 {
        nodes.iter().map(|n| n.state.benef[v.id()]).min().unwrap()
    }
    fn minimum_abs_value_of_substate(&self, v: Variable, nodes: &[Node<McpState>]) -> i32 {
        nodes.iter().map(|n| n.state.benef[v.id()].abs()).min().unwrap()
    }
    fn difference_of_abs_benefit(&self, l: Variable, u: &McpState, m: &McpState) -> i32 {
        u.benef[l.id()].abs() - m.benef[l.id()].abs()
    }
}