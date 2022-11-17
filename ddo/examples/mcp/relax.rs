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

use crate::model::{Mcp, McpState};

#[derive(Debug, Clone)]
pub struct McpRelax<'a> {
    pb  : &'a Mcp,

    vr: isize,

    /// `nk[i]` denotes the partial sum of the weights of edges connected to
    /// at least one vertex <= `i`
    nk: Vec<isize>,
    
    /// `estimates[i]` is the sum of the weight of all positive edges in the 
    /// induced graph comprising only vertices >= i
    estimates : Vec<isize>,
}
impl <'a> McpRelax<'a> {
    pub fn new(pb: &'a Mcp) -> Self {
        let estimates = Self::precompute_all_estimates(pb);
        let nk        = Self::precompute_all_nk(pb);
        let vr        = pb.initial_value();
        McpRelax{pb, vr, nk, estimates}
    }
    /// This estimate provides our RUB from the given state. Actually, it is
    /// quite simple. Given that all edges connected to at least one node
    /// from the 'undecided' set have not been taken into account (yet). It
    /// suffices to use the sum of the edges with positive weights connected
    /// to one of these vertices to derive a RUB.
    ///
    /// # Note:
    /// Because the variable ordering is static (natural order), the upper
    /// bounds can all be precomputed. This should speedup the RUB computation
    /// quite a bit (amorticized O(1)).
    fn precompute_all_estimates(pb: &Mcp) -> Vec<isize> {
        let v = pb.nb_variables();
        let mut res = vec![0_isize; 1+v];

        for (i, ri) in res.iter_mut().enumerate() {
            *ri = Self::precompute_estimate(pb, i);
        }

        res
    }
    fn precompute_estimate(pb: &Mcp, depth: usize) -> isize {
        let mut value = 0_isize;
        let n_vars = pb.nb_variables();
        for source in depth..n_vars {
            for destination in source + 1 .. n_vars {
                let weight = pb.graph[(source, destination)];
                if weight > 0 {
                    value += weight;
                }
            }
        }
        value
    }

    
    fn precompute_all_nk(pb: &Mcp) -> Vec<isize> {
        let v = pb.nb_variables();
        let mut res = vec![0_isize; 1+v];

        for (i, ri) in res.iter_mut().enumerate() {
            *ri = Self::precompute_nk(pb, i);
        }

        res
    }
    fn precompute_nk(pb: &Mcp, depth: usize) -> isize {
        let mut sum = 0_isize;

        for j in 0..depth {
            for i in 0..j {
               let w = pb.graph[(i, j)];
               if w < 0 {
                   sum += w;
               }
            }
        }

        sum
    }

}
impl Relaxation for McpRelax<'_> {
    type State = McpState;

    fn merge(&self, states: &mut dyn Iterator<Item=&McpState>) -> McpState {
        let states = states.collect::<Vec<&McpState>>();
        self.merge_states(&states)
    }
    fn relax(&self, _: &McpState, dst: &McpState, mrg: &McpState, _: Decision, c: isize) -> isize {
        let mut relaxed_cost = c;
        for v in 0..self.pb.nb_variables() {
            relaxed_cost += self.difference_of_abs_benefit(Variable(v), dst, mrg);
        }
        relaxed_cost
    }
    
    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let k = state.depth as usize;
        let marginal_benefit = state.benef.iter().copied()
            .skip(k)
            .map(|x| x.abs())
            .sum::<isize>();
        marginal_benefit + self.estimates[k] - self.vr + self.nk[k]
    }
    
}

// private methods
impl McpRelax<'_> {

    const POSITIVE: u8 = 1;
    const NEGATIVE: u8 = 2;
    const BOTH    : u8 = McpRelax::POSITIVE + McpRelax::NEGATIVE;

    fn merge_states(&self, nodes: &[&McpState]) -> McpState {
        let mut data = vec![0; self.pb.nb_variables()];

        for v in 0..self.pb.nb_variables() {
            data[v] = self.merge_substates(Variable(v), nodes);
        }

        McpState{ depth: nodes[0].depth, benef: data }
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
