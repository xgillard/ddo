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
//! for the SOP problem.

use std::cell::RefCell;

use ddo::{Relaxation, Problem};

use crate::{model::Sop, state::{Previous, SopState}, BitSet};

#[derive(Clone)]
pub struct SopRelax<'a> {
    pb: &'a Sop,
}
impl<'a> SopRelax<'a> {
    thread_local! {
        static HELPER: RefCell<RelaxHelper> = RefCell::new(RelaxHelper::new());
    }

    pub fn new(pb: &'a Sop) -> Self {
        Self { pb }
    }
}
#[derive(Clone)]
struct RelaxHelper {
    depth: usize,
    previous: BitSet,
    all_must: BitSet,
    all_agree: BitSet,
    all_maybe: BitSet,
}
impl RelaxHelper {
    fn new() -> Self {
        Self {
            depth: 0,
            previous: BitSet::empty(),
            all_must: BitSet::empty(),
            all_agree: (BitSet::default()).flip(),
            all_maybe: BitSet::empty(),
        }
    }
    fn clear(&mut self) {
        self.depth = 0;
        self.previous = BitSet::empty();
        self.all_must = BitSet::empty();
        self.all_agree = (BitSet::default()).flip();
        self.all_maybe = BitSet::empty();
    }
    fn track_depth(&mut self, depth: usize) {
        self.depth = self.depth.max(depth);
    }
    fn track_previous(&mut self, prec: &Previous) {
        match prec {
            Previous::Job(x) => self.previous.add_inplace(*x as usize),
            Previous::Virtual(xs) => self.previous.union_inplace(xs),
        };
    }
    fn track_must(&mut self, bs: &BitSet) {
        self.all_agree.inter_inplace(bs);
        self.all_must.union_inplace(bs);
    }
    fn track_maybe(&mut self, bs: &Option<BitSet>) {
        if let Some(bs) = bs.as_ref() {
            self.all_maybe.union_inplace(bs);
        }
    }

    fn get_depth(&self) -> usize {
        self.depth
    }
    fn get_previous(&self) -> Previous {
        Previous::Virtual(self.previous.clone())
    }
    fn get_must(&self) -> BitSet {
        self.all_agree.clone()
    }
    fn get_maybe(&self) -> Option<BitSet> {
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

impl Relaxation for SopRelax<'_> {
    type State = SopState;

    fn merge(&self, states: &mut dyn Iterator<Item = &SopState>) -> SopState {
        SopRelax::HELPER.with(|helper| {
            let mut helper = helper.borrow_mut();
            helper.clear();

            for state in states {
                helper.track_depth(state.depth);
                helper.track_previous(&state.previous);
                helper.track_must(&state.must_schedule);
                helper.track_maybe(&state.maybe_schedule);
            }
    
            SopState {
                depth: helper.get_depth(),
                previous: helper.get_previous(),
                must_schedule: helper.get_must(),
                maybe_schedule: helper.get_maybe(),
            }
        })
    }

    fn relax(&self, _: &SopState, _: &SopState, _: &SopState, _: ddo::Decision, cost: isize) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &SopState) -> isize {
        let complete_tour = self.pb.nb_variables() - state.depth as usize;
        let n_must_visit = state.must_schedule.len() as usize;
        let mut to_must_visit = vec![];
        let mut to_maybe_visit = vec![];

        // get the cheapest edge from position to any remaining node
        // and get cheapest edge to reach all other nodes

        let mut dist_from_position = isize::MAX;
 
        for i in state.must_schedule.iter() {
            for (cost, j) in self.pb.cheapest_edges[i].iter() {
                if state.must_schedule.contains(*j) {
                    to_must_visit.push(*cost);
                    break;
                } else if let Some(maybes) = state.maybe_schedule.as_ref() {
                    if maybes.contains(*j) {
                        to_must_visit.push(*cost);
                        break;
                    }
                }
            }
            
            dist_from_position = dist_from_position.min(self.pb.min_distance_to(state, i));
        }

        if n_must_visit < complete_tour {
            if let Some(maybes) = state.maybe_schedule.as_ref() {
                for i in maybes.iter() {
                    for (cost, j) in self.pb.cheapest_edges[i].iter() {
                        if state.must_schedule.contains(*j) || maybes.contains(*j) {
                            to_maybe_visit.push(*cost);
                            break;
                        }
                    }
            
                    dist_from_position = dist_from_position.min(self.pb.min_distance_to(state, i));
                }
            }
        }
    
        to_must_visit.sort_unstable();
        to_maybe_visit.sort_unstable();

        if n_must_visit >= complete_tour {
            - (dist_from_position + to_must_visit.iter().take(complete_tour - 1).sum::<isize>())
        } else if to_must_visit.is_empty() {
            - (dist_from_position + to_maybe_visit.iter().take(complete_tour - 1).sum::<isize>())
        } else if to_must_visit[to_must_visit.len() - 1] <= to_maybe_visit[0] {
            - (dist_from_position + to_must_visit.iter().sum::<isize>() + to_maybe_visit.iter().take(complete_tour - 1 - to_must_visit.len()).sum::<isize>())
        } else {
            - (dist_from_position + to_must_visit.iter().take(to_must_visit.len() - 1).sum::<isize>() + to_maybe_visit.iter().take(complete_tour - to_must_visit.len()).sum::<isize>())
        }
    }
}
