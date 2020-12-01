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

use std::cmp::min;

use ddo::abstraction::dp::{Problem, Relaxation};
use ddo::common::{Decision, VarSet};

use crate::model::{Max2Sat, State};

#[derive(Debug, Clone)]
pub struct Max2SatRelax<'a> {
    problem : &'a Max2Sat,
    vars    : VarSet
}

impl <'a> Max2SatRelax<'a> {
    pub fn new(problem: &'a Max2Sat) -> Max2SatRelax<'a> {
        Max2SatRelax { problem, vars: problem.all_vars() }
    }
}

impl Relaxation<State> for Max2SatRelax<'_> {
    fn merge_states(&self, states: &mut dyn Iterator<Item=&State>) -> State {
        let states       = states.collect::<Vec<&State>>();
        let mut benefits = vec![0; self.problem.nb_vars()];

        // Compute the merged state and relax the best edges costs
        for v in self.vars.iter() {
            let mut sign = 0;
            let mut min_benef = isize::max_value();
            let mut same = true;

            for state in states.iter() {
                let substate = state[v];
                min_benef = min(min_benef, substate.abs());

                if sign == 0 && substate != 0 {
                    sign = substate.abs() / substate;
                } else if sign * substate < 0 {
                    same = false;
                    break;
                }
            }

            if same {
                benefits[v.0] = sign * min_benef;
            }
        }

        State{depth: states[0].depth, substates: benefits}
    }
    fn relax_edge(&self, _: &State, dst: &State, relaxed: &State, _: Decision, cost: isize) -> isize {
        let mut relaxed_cost = cost;
        for v in self.vars.iter() {
            relaxed_cost += dst[v].abs() - relaxed[v].abs();
        }
        relaxed_cost
    }
}