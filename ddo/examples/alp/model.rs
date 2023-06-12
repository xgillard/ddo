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

//! This example show how to implement a solver for the pigment sequencint problem 
//! using ddo. It is a fairly simple example but it features most of the aspects you will
//! want to copy when implementing your own solver.

use std::{vec, collections::HashSet};

use ddo::*;

use crate::io_utils::AlpInstance;

/// The state of the DP model
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlpState {
    /// The number of remaining aircrafts to schedule for each class
    pub rem: Vec<usize>,
    /// Info about the state of each runway
    pub info: Vec<RunwayState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub struct RunwayState {
    /// The time of the latest aircraft scheduled
    pub prev_time: isize,
    /// The aircraft class scheduled the latest
    pub prev_class: isize,
}

pub struct AlpDecision {
    pub class: usize,
    pub runway: usize,
}

/// This structure describes a ALP instance
#[derive(Debug, Clone)]
pub struct Alp {
    pub instance: AlpInstance,
    pub next: Vec<Vec<usize>>, // The next aircraft to schedule for each class and for each remaining number of aircrafts
    min_separation_to: Vec<isize>,
}

impl Alp {
    pub fn new(instance: AlpInstance) -> Self {
        let mut next = vec![vec![0]; instance.nb_classes];

        for i in (0..instance.nb_aircrafts).rev() {
            next[instance.classes[i]].push(i);
        }

        let mut min_separation_to = vec![isize::MAX; instance.nb_classes];

        for i in 0..instance.nb_classes {
            for j in 0..instance.nb_classes {
                min_separation_to[j] = min_separation_to[j].min(instance.separation[i][j]);
            }
        }

        Alp {
            instance,
            next,
            min_separation_to,
        }
    }

    pub fn get_arrival_time(&self, info: &Vec<RunwayState>, aircraft: usize, runway: usize) -> isize {
        if info[runway].prev_time == -1 {
            self.instance.target[aircraft]
        } else if info[runway].prev_class == -1 {
            self.instance.target[aircraft]
                .max(info[runway].prev_time + self.min_separation_to[self.instance.classes[aircraft]])
        } else {
            self.instance.target[aircraft]
                .max(info[runway].prev_time + self.instance.separation[info[runway].prev_class as usize][self.instance.classes[aircraft]])
        }
    }

    pub fn to_decision(&self, decision: &AlpDecision) -> isize {
        (decision.class + self.instance.nb_classes * decision.runway) as isize
    }

    pub fn from_decision(&self, value: isize) -> AlpDecision {
        AlpDecision {
            class: value as usize % self.instance.nb_classes,
            runway: value as usize / self.instance.nb_classes,
        }
    }
}

impl Problem for Alp {
    type State = AlpState;

    fn nb_variables(&self) -> usize {
        self.instance.nb_aircrafts
    }

    fn initial_state(&self) -> Self::State {
        let mut rem = vec![0; self.instance.nb_classes];

        for i in 0..self.instance.nb_aircrafts {
            rem[self.instance.classes[i]] += 1;
        }

        AlpState {
            rem,
            info: vec![RunwayState {prev_class: -1, prev_time: -1}; self.instance.nb_runways],
        }
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn transition(&self, state: &Self::State, decision: ddo::Decision) -> Self::State {
        if decision.value == -1 {
            state.clone()
        } else {
            let AlpDecision {class, runway} = self.from_decision(decision.value);
            let aircraft = self.next[class][state.rem[class]];

            let mut next = state.clone();
            next.rem[self.instance.classes[aircraft]] -= 1;
            next.info[runway].prev_class = class as isize;
            next.info[runway].prev_time = self.get_arrival_time(&state.info, aircraft, runway);

            next.info.sort_unstable();
            
            next
        }
    }

    fn transition_cost(&self, state: &Self::State, decision: ddo::Decision) -> isize {
        if decision.value == -1 {
            0
        } else {
            let AlpDecision {class, runway} = self.from_decision(decision.value);
            let aircraft = self.next[class][state.rem[class]];
            - (self.get_arrival_time(&state.info, aircraft, runway) - self.instance.target[aircraft])
        }
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<ddo::Variable> {
        if depth < self.instance.nb_aircrafts {
            Some(Variable(depth))
        } else {
            None
        }
    }

    fn for_each_in_domain(&self, variable: ddo::Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        let mut tot_rem = 0;
        let mut used = HashSet::new();
        for (class, rem) in state.rem.iter().copied().enumerate() {
            if rem > 0 {
                let aircraft = self.next[class][rem];

                used.clear();
                for runway in 0..self.instance.nb_runways {
                    if used.contains(&state.info[runway]) {
                        continue;
                    }

                    let arrival = self.get_arrival_time(&state.info, aircraft, runway);
                    if arrival <= self.instance.latest[aircraft] {
                        f.apply(Decision { variable, value: self.to_decision(&AlpDecision { class, runway }) });
                        used.insert(state.info[runway]);
                    }
                }
            }

            tot_rem += rem;
        }

        if tot_rem == 0 {
            f.apply(Decision {variable, value: -1 });
        }
    }
}

/// This structure implements the ALP relaxation
pub struct AlpRelax {
    pb: Alp,
}

impl AlpRelax {
    pub fn new(pb: Alp) -> Self {

        Self { pb }
    }
}

impl Relaxation for AlpRelax {
    type State = AlpState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut rem = vec![usize::MAX; self.pb.instance.nb_classes];
        let mut info = vec![RunwayState { prev_class: -1, prev_time: isize::MAX }; self.pb.instance.nb_runways];

        for s in states {
            rem.iter_mut().enumerate().for_each(|(k,r)| *r = (*r).min(s.rem[k]));
            info.iter_mut().enumerate().for_each(|(r,i)| i.prev_time = i.prev_time.min(s.info[r].prev_time));
        }

        AlpState {
            rem,
            info,
        }
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

    fn fast_upper_bound(&self, _: &Self::State) -> isize {
        0
    }
}


/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
pub struct AlpRanking;
impl StateRanking for AlpRanking {
    type State = AlpState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        let tot_a = a.info.iter().map(|i| i.prev_time).sum::<isize>();
        let tot_b = b.info.iter().map(|i| i.prev_time).sum::<isize>();
        
        tot_a.cmp(&tot_b)
    }
}
