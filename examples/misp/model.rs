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

use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};
use std::ops::Not;

use bitset_fixed::BitSet;

use ddo::{Problem, Variable, Domain, Decision, VarSet};

use crate::instance::Graph;

#[derive(Debug, Clone)]
pub struct Misp {
    pub graph : Graph
}

impl Misp {
    pub fn new(mut graph : Graph) -> Misp {
        graph.complement();
        Misp {graph}
    }
}

const YES_NO : [isize; 2] = [1, 0];
const NO     : [isize; 1] = [0];

impl Problem<BitSet> for Misp {
    fn nb_vars(&self) -> usize {
        self.graph.nb_vars
    }

    fn initial_state(&self) -> BitSet {
        BitSet::new(self.graph.nb_vars).not()
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn domain_of<'a>(&self, state: &'a BitSet, var: Variable) -> Domain<'a> {
        if state[var.0] { Domain::Slice(&YES_NO) } else { Domain::Slice(&NO) }
    }

    fn transition(&self, state: &BitSet, _vars: &VarSet, d: Decision) -> BitSet {
        let mut bs = state.clone();
        bs.set(d.variable.0, false);

        // drop adjacent vertices if needed
        if d.value == 1 {
            bs &= &self.graph.adj_matrix[d.variable.0];
        }

        bs
    }

    fn transition_cost(&self, _state: &BitSet, _vars: &VarSet, d: Decision) -> isize {
        if d.value == 0 {
            0
        } else {
            self.graph.weights[d.variable.0]
        }
    }

    fn impacted_by(&self, state: &BitSet, var: Variable) -> bool {
        state[var.id()]
    }
}
impl From<File> for Misp {
    fn from(file: File) -> Self {
        BufReader::new(file).into()
    }
}
impl <S: Read> From<BufReader<S>> for Misp {
    fn from(buf: BufReader<S>) -> Self {
        buf.lines().into()
    }
}
impl <B: BufRead> From<Lines<B>> for Misp {
    fn from(lines: Lines<B>) -> Self {
        Self::new(lines.into())
    }
}
