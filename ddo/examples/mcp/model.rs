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

use std::cmp::{max, min};
use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};

use ddo::*;

use crate::graph::Graph;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct McpState {
    pub benef  : Vec<isize>,
    pub depth  : u16,
}

// Define a few constants to improve readability
const S      : isize = 1;
const T      : isize =-1;

#[derive(Debug, Clone)]
pub struct Mcp {
    pub graph : Graph
}
impl Mcp {
    pub fn new(g: Graph) -> Self { Mcp {graph: g} }
}
impl Problem for Mcp {
    type State = McpState;

    fn nb_variables(&self) -> usize {
        self.graph.nb_vertices
    }

    fn initial_state(&self) -> McpState {
        McpState {depth: 0, benef: vec![0; self.nb_variables()]}
    }

    fn initial_value(&self) -> isize {
        self.graph.sum_of_negative_edges()
    }

    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback) {
        if state.depth == 0 { 
            f.apply(Decision{variable, value: S});
        } else { 
            f.apply(Decision{variable, value: S});
            f.apply(Decision{variable, value: T});
        }
    }

    fn transition(&self, state: &McpState, d: Decision) -> McpState {
        let mut benefits = vec![0; self.nb_variables()];
        
        let x = d.variable.id();
        let n = self.nb_variables();
        for (v, item) in benefits.iter_mut().enumerate().take(n).skip(x) { // for all unassigned vars
            *item = state.benef[v] + d.value * self.graph[(d.variable, Variable(v))];
        }
        McpState {depth: 1 + state.depth, benef: benefits}
    }

    fn transition_cost(&self, state: &McpState, d: Decision) -> isize {
        match d.value {
            S => if state.depth == 0 { 0 } else { self.branch_on_s(state, d) },
            T => if state.depth == 0 { 0 } else { self.branch_on_t(state, d) },
            _ => unreachable!()
        }
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
            -> Option<Variable> {
        if depth < self.nb_variables() {
            Some(Variable(depth))
        } else {
            None
        }
    }
}
// private methods
impl Mcp {
    fn branch_on_s(&self, state: &McpState, d: Decision) -> isize {
        let n = self.nb_variables();
        let x = d.variable.id();

        // The \( (- s^k_k)^+ \) component
        let res = max(0, -state.benef[d.variable.id()]);
        // The \( \sum_{l > k, s^k_l w_{kl} \le 0} \min\left\{ |s^k_l|, |w_{kl}| \right\} \)
        let mut sum = 0;
        for v in x..n {
            let skl = state.benef[v];
            let wkl = self.graph[(d.variable, Variable(v))];

            if skl * wkl <= 0 { sum += min(skl.abs(), wkl.abs()); }
        }
        res + sum
    }
    fn branch_on_t(&self, state: &McpState, d: Decision) -> isize {
        let n = self.nb_variables();
        let x = d.variable.id();

        // The \( (s^k_k)^+ \) component
        let res = max(0, state.benef[d.variable.id()]);
        // The \( \sum_{l > k, s^k_l w_{kl} \le 0} \min\left\{ |s^k_l|, |w_{kl}| \right\} \)
        let mut sum = 0;
        for v in x..n {
            let skl = state.benef[v];
            let wkl = self.graph[(d.variable, Variable(v))];

            if skl * wkl >= 0 { sum += min(skl.abs(), wkl.abs()); }
        }
        res + sum
    }
}
impl From<Graph> for Mcp {
    fn from(g: Graph) -> Self {
        Mcp::new(g)
    }
}
impl From<File> for Mcp {
    fn from(f: File) -> Self {
        Mcp::new(f.into())
    }
}
impl <S: Read> From<BufReader<S>> for Mcp {
    fn from(buf: BufReader<S>) -> Mcp {
        Mcp::new(buf.into())
    }
}
impl <B: BufRead> From<Lines<B>> for Mcp {
    fn from(lines: Lines<B>) -> Mcp {
        Mcp::new(lines.into())
    }
}


pub struct McpRanking;
impl StateRanking for McpRanking {
    type State = McpState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        let xa = a.benef.iter().map(|v| v.abs()).sum::<isize>();
        let xb = b.benef.iter().map(|v| v.abs()).sum::<isize>();

        xa.cmp(&xb)
    }
}