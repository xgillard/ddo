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
//! of the TSP+TW. (Implementation of the `Problem` trait).

use ddo::{Problem, Variable, Decision};
use smallbitset::Set256;

use crate::{instance::TsptwInstance, state::{ElapsedTime, Position, TsptwState}};


/// This is the structure encapsulating the Tsptw problem.
#[derive(Clone)]
pub struct Tsptw {
    pub instance: TsptwInstance,
    pub initial : TsptwState,
}
impl Tsptw {
    pub fn new(inst: TsptwInstance) -> Self {
        let mut must_visit = Set256::default();
        (1..inst.nb_nodes).for_each(|i| {must_visit.add_inplace(i as usize);});
        let state = TsptwState {
            position  : Position::Node(0),
            elapsed   : ElapsedTime::FixedAmount{duration: 0},
            must_visit,
            maybe_visit: None,
            depth : 0
        };
        Self { instance: inst, initial: state }
    }
}

impl Problem for Tsptw {
    type State = TsptwState;

    fn nb_variables(&self) -> usize {
        self.instance.nb_nodes as usize
    }

    fn initial_state(&self) -> TsptwState {
        self.initial.clone()
    }

    fn initial_value(&self) -> isize {
        0
    }
    
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        // When we are at the end of the tour, the only possible destination is
        // to go back to the depot. Any state that violates this constraint is
        // de facto infeasible.
        if state.depth as usize == self.nb_variables() - 1 {
            if self.can_move_to(state, 0) {
                f.apply(Decision { variable, value: 0 })
            }
            return;
        }

        for i in state.must_visit.iter() {
            if !self.can_move_to(state, i) {
                return;
            }
        }
        for i in state.must_visit.iter() {
            f.apply(Decision { variable, value: i as isize })
        }

        // Add those that can possibly be visited
        if let Some(maybe_visit) = &state.maybe_visit {
            for i in maybe_visit.iter() {
                if self.can_move_to(state, i) {
                    f.apply(Decision { variable, value: i as isize })
                }
            }
        }
    }

    fn transition(&self, state: &TsptwState, d: Decision) -> TsptwState {
        // if it is a true move
        let mut remaining = state.must_visit.clone();
        remaining.remove_inplace(d.value as usize);
        // if it is a possible move
        let mut maybes = state.maybe_visit.clone();
        if let Some(maybe) = maybes.as_mut() {
            maybe.remove_inplace(d.value as usize);
        }

        let time = self.arrival_time(state, d.value as usize);

        TsptwState {
            position : Position::Node(d.value as u16),
            elapsed  : time,
            must_visit: remaining,
            maybe_visit: maybes,
            depth: state.depth + 1
        }
    }

    fn transition_cost(&self, state: &TsptwState, d: Decision) -> isize {
        // Tsptw is a minimization problem but the solver works with a 
        // maximization perspective. So we have to negate the min if we want to
        // yield a lower bound.
        let twj = self.instance.timewindows[d.value as usize];
        let travel_time = self.min_distance_to(state, d.value as usize);
        let waiting_time = match state.elapsed {
            ElapsedTime::FixedAmount{duration} => 
                if (duration + travel_time) < twj.earliest {
                    twj.earliest - (duration + travel_time)
                } else {
                    0
                },
            ElapsedTime::FuzzyAmount{earliest, ..} => 
                if (earliest + travel_time) < twj.earliest {
                    twj.earliest - (earliest + travel_time)
                } else {
                    0
                }
        };

        -( (travel_time + waiting_time) as isize)
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<Variable> {
        if depth == self.nb_variables() {
            None
        } else {
            Some(Variable(depth))
        }
    }
}

impl Tsptw {
    pub fn can_move_to(&self, state: &TsptwState, j: usize) -> bool {
        let twj         = self.instance.timewindows[j];
        let min_arrival = state.elapsed.add_duration(self.min_distance_to(state, j));
        match min_arrival {
            ElapsedTime::FixedAmount{duration}     => duration <= twj.latest,
            ElapsedTime::FuzzyAmount{earliest, ..} => earliest <= twj.latest,
        }
    }
    fn arrival_time(&self, state: &TsptwState, j: usize) -> ElapsedTime {
       let min_arrival = state.elapsed.add_duration(self.min_distance_to(state, j));
       let max_arrival = state.elapsed.add_duration(self.max_distance_to(state, j));

       let min_arrival = match min_arrival {
           ElapsedTime::FixedAmount{duration}     => duration,
           ElapsedTime::FuzzyAmount{earliest, ..} => earliest
       };
       let max_arrival = match max_arrival {
           ElapsedTime::FixedAmount{duration}    => duration,
           ElapsedTime::FuzzyAmount{latest, ..}  => latest
       };
       // This would be the arrival time if we never had to wait.
       let arrival_time = 
           if min_arrival.eq(&max_arrival) { 
               ElapsedTime::FixedAmount{duration: min_arrival} 
           } else {
               ElapsedTime::FuzzyAmount{earliest: min_arrival, latest: max_arrival}
           };
       // In order to account for the possible waiting time, we need to adjust
       // the earliest arrival time
       let twj = self.instance.timewindows[j];
       match arrival_time {
          ElapsedTime::FixedAmount{duration} => {
              ElapsedTime::FixedAmount{duration: duration.max(twj.earliest)}
          },
          ElapsedTime::FuzzyAmount{mut earliest, mut latest} => {
            earliest = earliest.max(twj.earliest);
            latest   = latest.min(twj.latest);

            if earliest.eq(&latest) {
                ElapsedTime::FixedAmount{duration: earliest}
            } else {
                ElapsedTime::FuzzyAmount{earliest, latest}
            }
          },
      }
    }
    fn min_distance_to(&self, state: &TsptwState, j: usize) -> usize {
        match &state.position {
            Position::Node(i) => self.instance.distances[*i as usize][j],
            Position::Virtual(candidates) => 
                candidates.iter()
                    .map(|i| self.instance.distances[i as usize][j as usize])
                    .min()
                    .unwrap()
        }
    }
    fn max_distance_to(&self, state: &TsptwState, j: usize) -> usize {
        match &state.position {
            Position::Node(i) => self.instance.distances[*i as usize][j],
            Position::Virtual(candidates) => 
                candidates.iter()
                    .map(|i| self.instance.distances[i as usize][j as usize])
                    .max()
                    .unwrap()
        }
    }
}
