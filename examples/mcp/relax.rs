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

use ddo::abstraction::dp::{Problem, Relaxation};
use ddo::common::{Decision, Variable, VarSet};

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
    fn merge_states(&self, states: &mut dyn Iterator<Item=&McpState>) -> McpState {
        let states = states.collect::<Vec<&McpState>>();
        self.merge_states(&states)
    }
    fn relax_edge(&self, _: &McpState, dst: &McpState, mrg: &McpState, _: Decision, c: isize) -> isize {
        let mut relaxed_cost = c;
        for v in self.vars.iter() {
            relaxed_cost += self.difference_of_abs_benefit(v, dst, mrg);
        }
        relaxed_cost
    }
}

// private methods
impl McpRelax<'_> {

    const POSITIVE: u8 = 1;
    const NEGATIVE: u8 = 2;
    const BOTH    : u8 = McpRelax::POSITIVE + McpRelax::NEGATIVE;

    fn merge_states(&self, nodes: &[&McpState]) -> McpState {
        let mut data = vec![0; self.pb.nb_vars()];

        for v in self.vars.iter() {
            data[v.id()] = self.merge_substates(v, nodes);
        }

        McpState{ initial: false, benef: data }
    }

    fn merge_substates(&self, v: Variable, nodes: &[&McpState]) -> isize {
        match self.substate_signs(v, nodes) {
            McpRelax::POSITIVE =>  self.minimum_substate(v, nodes),              // min( u_l )
            McpRelax::NEGATIVE => -self.minimum_abs_value_of_substate(v, nodes), // min(|u_l|)
            _ => 0 // otherwise
        }
    }

    fn substate_signs(&self, v: Variable, states: &[&McpState]) -> u8 {
        let mut signs = 0_u8;
        for state in states.iter() {
            let substate = state.benef[v.id()];
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

    fn minimum_substate(&self, v: Variable, states: &[&McpState]) -> isize {
        states.iter().map(|state| state.benef[v.id()]).min().unwrap()
    }
    fn minimum_abs_value_of_substate(&self, v: Variable, states: &[&McpState]) -> isize {
        states.iter().map(|state| state.benef[v.id()].abs()).min().unwrap()
    }
    fn difference_of_abs_benefit(&self, l: Variable, u: &McpState, m: &McpState) -> isize {
        u.benef[l.id()].abs() - m.benef[l.id()].abs()
    }
}