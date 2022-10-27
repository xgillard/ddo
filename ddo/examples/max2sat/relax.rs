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

use ddo::{Decision, Relaxation, Variable};
use std::cmp::min;
use crate::model::{State, Max2Sat};

#[derive(Debug, Clone)]
pub struct Max2SatRelax<'a>(pub &'a Max2Sat);
impl Relaxation for Max2SatRelax<'_> {
    type State = State;

    fn merge(&self, states: &mut dyn Iterator<Item = &State>) -> State {
        let states = states.collect::<Vec<&State>>();
        let mut benefits = vec![0; self.0.nb_vars];

        // Compute the merged state and relax the best edges costs
        for v in 0..self.0.nb_vars {
            let mut sign = 0;
            let mut min_benef = isize::max_value();
            let mut same = true;

            for state in states.iter() {
                let substate = state[Variable(v)];
                min_benef = min(min_benef, substate.abs());

                if sign == 0 && substate != 0 {
                    sign = substate.abs() / substate;
                } else if sign * substate < 0 {
                    same = false;
                    break;
                }
            }

            if same {
                benefits[v] = sign * min_benef;
            }
        }

        State {
            depth: states[0].depth,
            substates: benefits,
        }
    }
    fn relax(&self, _: &State, dst: &State, relaxed: &State, _: Decision, cost: isize) -> isize {
        let mut relaxed_cost = cost;
        for v in 0..self.0.nb_vars {
            relaxed_cost += dst[Variable(v)].abs() - relaxed[Variable(v)].abs();
        }
        relaxed_cost
    }


    fn fast_upper_bound(&self, state: &State) -> isize {
        self.0.fast_upper_bound(state)
    }
}