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

use std::{vec, collections::BTreeMap};

use ddo::*;

use crate::dp::LcsDp;

/// The state of the DP model
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LcsState {
    /// The current position in each string
    pub position: Vec<usize>,
}

/// This structure describes a Lcs instance
#[derive(Debug)]
pub struct Lcs {
    /// The input strings, converted to integers
    pub strings: Vec<Vec<usize>>,
    /// The number of strings
    n_strings: usize,
    /// The number of different characters
    n_chars: usize,
    /// The length of each string
    string_length: Vec<usize>,
    /// next[str][char][pos] gives the next position after
    /// position pos (included) in which character char occurs in string str
    next: Vec<Vec<Vec<usize>>>,
    /// rem[str][char][pos] gives the number of occurences of
    /// character char in string str after position pos (included)
    rem: Vec<Vec<Vec<isize>>>,
    /// Mapping from integers to the original characters
    pub chars: BTreeMap<usize, char>,
    /// DP tables for 2-strings problems
    tables: Vec<Vec<Vec<isize>>>,
}

/// constant for the decision to go to the end of the strings without taking any character
pub const GO_TO_END_OF_STRINGS: isize = -1;

impl Lcs {
    pub fn new(
        strings: Vec<Vec<usize>>,
        n_strings: usize,
        n_chars: usize,
        string_length: Vec<usize>,
        next: Vec<Vec<Vec<usize>>>,
        rem: Vec<Vec<Vec<isize>>>,
        chars: BTreeMap<usize, char>,
    ) -> Self {
        // solve the 2-sentences problem for each pair of consecutive sentences
        let mut tables = vec![];
        for i in 0..(n_strings - 1) {
            let dp = LcsDp {
                n_chars,
                a: &strings[i],
                b: &strings[i+1],
            };
            tables.push(dp.solve());
        }
        Self { 
            strings,
            n_strings,
            n_chars,
            string_length,
            next,
            rem,
            chars,
            tables,
        }
    }
}

impl Problem for Lcs {
    type State = LcsState;

    fn nb_variables(&self) -> usize {
        self.string_length[0]
    }

    fn initial_state(&self) -> Self::State {
        LcsState {
            position: vec![0; self.n_strings],
        }
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn transition(&self, state: &Self::State, decision: ddo::Decision) -> Self::State {
        let mut position = self.string_length.clone();

        if decision.value != GO_TO_END_OF_STRINGS {
            let char = decision.value as usize;
            
            for str in 0..self.n_strings {
                position[str] = self.next[str][char][state.position[str]] + 1;
            }
        }

        LcsState { position }
    }

    fn transition_cost(&self, _: &Self::State, decision: ddo::Decision) -> isize {
        match decision.value {
            GO_TO_END_OF_STRINGS => 0,
            _ => 1,
        }
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<ddo::Variable> {
        if depth < self.nb_variables() {
            Some(Variable(depth))
        } else {
            None
        }
    }

    fn for_each_in_domain(&self, variable: ddo::Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        let mut found_char = false;

        for char in 0..self.n_chars {
            let mut valid = true;
            for i in 0..self.n_strings {
                if self.rem[i][char][state.position[i]] == 0 {
                    valid = false;
                    break;
                }
            }

            if valid {
                found_char = true;
                f.apply(Decision { variable, value: char as isize });
            }
        }

        if !found_char {
            f.apply(Decision { variable, value: GO_TO_END_OF_STRINGS });
        }
    }

    fn is_impacted_by(&self, var: Variable, state: &Self::State) -> bool {
        // this model uses long arcs
        var.0 == state.position[0]
    }
}

/// This structure implements the Lcs relaxation
pub struct LcsRelax<'a>{
    pb: &'a Lcs,
}

impl <'a> LcsRelax<'a> {
    pub fn new(pb: &'a Lcs) -> Self {
        Self { pb }
    }
}

impl Relaxation for LcsRelax<'_> {
    type State = LcsState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut position = self.pb.string_length.clone();

        for s in states {
            for i in 0..self.pb.n_strings {
                position[i] = position[i].min(s.position[i]);
            }
        }
        
        LcsState { position }
    }

    fn relax(
        &self,
        _source: &Self::State,
        _dest: &Self::State,
        _new:  &Self::State,
        _decision: Decision,
        cost: isize,
    ) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let mut tot = 0;
        for char in 0..self.pb.n_chars {
            let mut in_common = isize::MAX;
            for i in 0..self.pb.n_strings {
                in_common = in_common.min(self.pb.rem[i][char][state.position[i]]);
            }
            tot += in_common;
        }

        let mut min_pairwise_lcs = isize::MAX;
        for i in 0..(self.pb.n_strings - 1) {
            min_pairwise_lcs = min_pairwise_lcs.min(
                self.pb.tables[i][state.position[i]][state.position[i + 1]]
            );
        }

        tot.min(min_pairwise_lcs)
    }
}

pub struct LcsRanking;
impl StateRanking for LcsRanking {
    type State = LcsState;

    // try to favor states that are at earlier positions
    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        let mut tot_a = 0;
        let mut tot_b = 0;
        for i in 0..a.position.len() {
            tot_a += a.position[i];
            tot_b += b.position[i];
        }
        tot_b.cmp(&tot_a)
    }
}