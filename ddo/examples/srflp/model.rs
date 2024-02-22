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

//! This module contains the definition of the dynamic programming formulation 
//! of the SRFLP. (Implementation of the `Problem` trait).

use ddo::{Problem, Variable, Decision};
use smallbitset::Set64;

use crate::{io_utils::SrflpInstance, state::SrflpState};

/// This is the structure encapsulating the Srflp problem.
#[derive(Debug, Clone)]
pub struct Srflp {
    pub instance: SrflpInstance,
    pub sorted_lengths: Vec<(isize, usize)>,
    pub sorted_flows: Vec<(isize, usize, usize)>,
    pub initial : SrflpState,
}
impl Srflp {
    pub fn new(inst: SrflpInstance) -> Self {
        let mut sorted_lengths: Vec<(isize, usize)> = inst.lengths.iter().enumerate().map(|(i,l)| (*l,i)).collect();
        sorted_lengths.sort_unstable();
        let mut sorted_flows = vec![];
        for i in 0..inst.nb_departments {
            for j in (i+1)..inst.nb_departments {
                sorted_flows.push((inst.flows[i][j], i, j));
            }
        }
        sorted_flows.sort_unstable();

        let mut must_place = Set64::empty();
        for i in 0..inst.nb_departments {
            must_place.add_inplace(i);
        }

        let state = SrflpState {
            must_place,
            maybe_place: None,
            cut: vec![0; inst.nb_departments],
            depth : 0
        };
        Self { instance: inst, sorted_lengths, sorted_flows, initial: state }
    }
}

impl Problem for Srflp {
    type State = SrflpState;

    fn nb_variables(&self) -> usize {
        self.instance.nb_departments
    }

    fn initial_state(&self) -> SrflpState {
        self.initial.clone()
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        let mut complete_arrangement = self.nb_variables() - state.depth;

        for i in state.must_place.iter() {
            complete_arrangement -= 1;
            f.apply(Decision { variable, value: i as isize })
        }

        if complete_arrangement > 0 {
            if let Some(maybe_visit) = &state.maybe_place {
                for i in maybe_visit.iter() {
                    f.apply(Decision { variable, value: i as isize })
                }
            }
        }
    }

    fn transition(&self, state: &SrflpState, d: Decision) -> SrflpState {
        let d = d.value as usize;

        // if it is a true move
        let mut remaining = state.must_place;
        remaining.remove_inplace(d);
        // if it is a possible move
        let mut maybes = state.maybe_place;
        if let Some(maybe) = maybes.as_mut() {
            maybe.remove_inplace(d);

            if maybe.is_empty() {
                maybes = None;
            }
        }

        let mut cut = state.cut.clone();
        cut[d] = 0;

        for i in remaining.iter() {
            cut[i] += self.instance.flows[d][i];
        }

        if let Some(maybe) = maybes.as_ref() {
            for i in maybe.iter() {
                cut[i] += self.instance.flows[d][i];
            }
        }

        SrflpState {
            must_place: remaining,
            maybe_place: maybes,
            cut,
            depth: state.depth + 1
        }
    }

    fn transition_cost(&self, state: &SrflpState, _: &Self::State, d: Decision) -> isize {
        let d = d.value as usize;

        let mut cut = 0;
        let mut complete_arrangement = self.instance.nb_departments - (state.depth + 1);

        for i in state.must_place.iter() {
            if i != d {
                cut += state.cut[i];
                complete_arrangement -= 1;
            }
        }

        if complete_arrangement > 0 {
            if let Some(maybe) = state.maybe_place.as_ref() {
                let mut temp = vec![];
                for i in maybe.iter() {
                    if i != d {
                        temp.push(state.cut[i]);
                    }
                }
                temp.sort_unstable();
                cut += temp.iter().take(complete_arrangement).sum::<isize>();
            }
        }

        // Srflp is a minimization problem but the solver works with a 
        // maximization perspective. So we have to negate the cost.
        - cut * self.instance.lengths[d]
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        if depth < self.nb_variables() {
            Some(Variable(depth))
        } else {
            None
        }
    }
}

impl Srflp {
    pub fn root_value(&self) -> f64 {
        let mut value = 0.0;

        for i in 0..self.instance.nb_departments {
            for j in (i+1)..self.instance.nb_departments {
                value += 0.5 * ((self.instance.lengths[i] + self.instance.lengths[j])
                             * self.instance.flows[i][j]) as f64;
            }
        }

        value
    }
}