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

//! This module contains the definition and implementation of the relaxation
//! for the TSP + TW problem.

use ddo::{Relaxation, Problem};
use smallbitset::Set256;

use crate::{
    model::Tsptw,
    state::{ElapsedTime, Position, TsptwState},
};
use std::cell::RefCell;

#[derive(Clone)]
pub struct TsptwRelax<'a> {
    pb: &'a Tsptw,
    cheapest_edge: Vec<usize>,
}
impl<'a> TsptwRelax<'a> {
    thread_local! {
        static HELPER: RefCell<RelaxHelper> = RefCell::new(RelaxHelper::new());
    }

    pub fn new(pb: &'a Tsptw) -> Self {
        let cheapest_edge = Self::compute_cheapest_edges(pb);
        Self {
            pb,
            cheapest_edge,
        }
    }

    fn compute_cheapest_edges(pb: &'a Tsptw) -> Vec<usize> {
        let mut cheapest = vec![];
        let n = pb.nb_variables();
        for i in 0..n {
            let mut min_i = usize::max_value();
            for j in 0..n {
                if i == j {
                    continue;
                }
                min_i = min_i.min(pb.instance.distances[j][i]);
            }
            cheapest.push(min_i);
        }
        cheapest
    }
}
#[derive(Clone)]
struct RelaxHelper {
    depth: u16,
    position: Set256,
    earliest: usize,
    latest: usize,
    all_must: Set256,
    all_agree: Set256,
    all_maybe: Set256,
}
impl RelaxHelper {
    fn new() -> Self {
        Self {
            depth: 0_u16,
            position: Set256::default(),
            earliest: usize::max_value(),
            latest: usize::min_value(),
            all_must: Set256::default(),
            all_agree: (Set256::default()).flip(),
            all_maybe: Set256::default(),
        }
    }
    fn clear(&mut self) {
        self.depth = 0_u16;
        self.earliest = usize::max_value();
        self.latest = usize::min_value();
        self.position = Set256::default();
        self.all_must = Set256::default();
        self.all_agree = (Set256::default()).flip();
        self.all_maybe = Set256::default();
    }
    fn track_depth(&mut self, depth: u16) {
        self.depth = self.depth.max(depth);
    }
    fn track_position(&mut self, pos: &Position) {
        match pos {
            Position::Node(x) => self.position.add_inplace(*x as usize),
            Position::Virtual(xs) => self.position.union_inplace(xs),
        };
    }
    fn track_elapsed(&mut self, elapsed: ElapsedTime) {
        match elapsed {
            ElapsedTime::FixedAmount { duration } => {
                self.earliest = self.earliest.min(duration);
                self.latest = self.latest.max(duration);
            }
            ElapsedTime::FuzzyAmount {
                earliest: ex,
                latest: lx,
            } => {
                self.earliest = self.earliest.min(ex);
                self.latest = self.latest.max(lx);
            }
        };
    }
    fn track_must_visit(&mut self, bs: &Set256) {
        self.all_agree.inter_inplace(bs);
        self.all_must.union_inplace(bs);
    }
    fn track_maybe(&mut self, bs: &Option<Set256>) {
        if let Some(bs) = bs.as_ref() {
            self.all_maybe.union_inplace(bs);
        }
    }

    fn get_depth(&self) -> u16 {
        self.depth
    }
    fn get_position(&self) -> Position {
        Position::Virtual(self.position.clone())
    }
    fn get_elapsed(&self) -> ElapsedTime {
        if self.earliest == self.latest {
            ElapsedTime::FixedAmount {
                duration: self.earliest,
            }
        } else {
            ElapsedTime::FuzzyAmount {
                earliest: self.earliest,
                latest: self.latest,
            }
        }
    }
    fn get_must_visit(&self) -> Set256 {
        self.all_agree.clone()
    }
    fn get_maybe_visit(&self) -> Option<Set256> {
        let mut maybe = self.all_maybe.clone(); // three lines: faster because it is in-place
        maybe.union_inplace(&self.all_must);
        maybe.diff_inplace(&self.all_agree);

        let count = maybe.len();
        if count > 0 {
            Some(maybe)
        } else {
            None
        }
    }
}

impl Relaxation for TsptwRelax<'_> {
    type State = TsptwState;

    fn merge(&self, states: &mut dyn Iterator<Item = &TsptwState>) -> TsptwState {
        TsptwRelax::HELPER.with(|helper| {
            let mut helper = helper.borrow_mut();
            helper.clear();

            for state in states {
                helper.track_depth(state.depth);
                helper.track_position(&state.position);
                helper.track_elapsed(state.elapsed);
                helper.track_must_visit(&state.must_visit);
                helper.track_maybe(&state.maybe_visit);
            }
    
            TsptwState {
                depth: helper.get_depth(),
                position: helper.get_position(),
                elapsed: helper.get_elapsed(),
                must_visit: helper.get_must_visit(),
                maybe_visit: helper.get_maybe_visit(),
            }
        })
    }

    fn relax(&self, _: &TsptwState, _: &TsptwState, _: &TsptwState, _: ddo::Decision, cost: isize) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let mut complete_tour = self.pb.nb_variables() - state.depth as usize;

        let mut tmp = vec![];
        let mut mandatory = 0;
        let mut back_to_depot = usize::max_value();

        for i in state.must_visit.iter() {
            complete_tour -= 1;
            mandatory += self.cheapest_edge[i];
            back_to_depot = back_to_depot.min(self.pb.instance.distances[i][0]);

            let latest = self.pb.instance.timewindows[i].latest;
            let earliest = state.elapsed.add_duration(self.cheapest_edge[i]).earliest();
            if earliest > latest {
                return isize::min_value();
            }
        }

        if let Some(maybes) = state.maybe_visit.as_ref() {
            let mut violations = 0;

            for i in maybes.iter() {
                tmp.push(self.cheapest_edge[i]);
                back_to_depot = back_to_depot.min(self.pb.instance.distances[i][0]);

                let latest = self.pb.instance.timewindows[i].latest;
                let earliest = state.elapsed.add_duration(self.cheapest_edge[i]).earliest();
                if earliest > latest {
                    violations += 1;
                }
            }

            if tmp.len() - violations < complete_tour {
                return isize::min_value();
            }

            tmp.sort_unstable();
            mandatory += tmp
                .iter()
                .copied()
                .take(complete_tour)
                .sum::<usize>();
        }

        // When there is no other city that MUST be visited, we must consider
        // the shortest distance between *here* (current position) and the
        // depot.
        if mandatory == 0 {
            back_to_depot = back_to_depot.min(match &state.position {
                Position::Node(x) => self.pb.instance.distances[*x as usize][0],
                Position::Virtual(bs) => bs.iter()
                    .map(|x| self.pb.instance.distances[x][0])
                    .min()
                    .unwrap(),
            });
        }

        // When it is impossible to get back to the depot in time, the current
        // state is infeasible. So we can give it an infinitely negative ub.
        let total_distance = mandatory + back_to_depot;
        let earliest_arrival = state.elapsed.add_duration(total_distance).earliest();
        let latest_deadline = self.pb.instance.timewindows[0].latest;
        if earliest_arrival > latest_deadline {
            isize::min_value()
        } else {
            -(total_distance as isize)
        }
    }
}
