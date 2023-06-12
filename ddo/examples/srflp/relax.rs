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
//! for the SRFLP problem.

use std::cmp::Reverse;

use ddo::{Relaxation, Decision, Problem};
use ordered_float::OrderedFloat;
use smallbitset::Set64;

use super::{model::Srflp, state::SrflpState};

#[derive(Clone)]
pub struct SrflpRelax<'a> {
    pb : &'a Srflp,
}
impl <'a> SrflpRelax<'a> {
    pub fn new(pb: &'a Srflp) -> Self {
        Self{pb}
    }
}

impl Relaxation for SrflpRelax<'_> {
    type State = SrflpState;

    fn merge(&self, states: &mut dyn Iterator<Item = &SrflpState>) -> SrflpState {
        let mut depth = 0;
        let mut must_place = Set64::empty();
        let mut maybe_place = Set64::empty();
        let mut cut = vec![isize::MAX; self.pb.instance.nb_departments];

        for state in states {
            depth = depth.max(state.depth);
            must_place = must_place.inter(state.must_place);

            for i in state.must_place.iter() {
                cut[i] = cut[i].min(state.cut[i]);
            }

            maybe_place = maybe_place.union(state.must_place);
            if let Some(maybe) = &state.maybe_place {
                maybe_place = maybe_place.union(*maybe);

                for i in maybe.iter() {
                    cut[i] = cut[i].min(state.cut[i]);
                }
            }
        }
        
        maybe_place = maybe_place.diff(must_place);

        let maybe_place = if maybe_place.len() > 0 {
            Some(maybe_place)
        } else {
            None
        };

        SrflpState {
            depth,
            must_place,
            maybe_place,
            cut,
        }
    }

    fn relax(
        &self,
        _: &Self::State,
        _: &Self::State,
        _: &Self::State,
        _: Decision,
        cost: isize,
    ) -> isize
    {
        cost
    }

    fn fast_upper_bound(&self, state: &SrflpState) -> isize {
        let complete_arrangement = self.pb.nb_variables() - state.depth;
        let n_flows = complete_arrangement * (complete_arrangement - 1) / 2;
        let n_must_place = state.must_place.len();
        let n_from_maybe_place = complete_arrangement - n_must_place;

        let mut ratios = vec![];
        let mut flows = vec![];
        let mut lengths = vec![];
        let mut maybe_lengths = vec![];

        let mut n_lengths_from_maybe_place = n_from_maybe_place;
        for (l,i) in self.pb.sorted_lengths.iter().copied() {
            if state.must_place.contains(i) {
                lengths.push(l);
            } else if let Some(maybe) = state.maybe_place.as_ref() {
                if maybe.contains(i) && n_lengths_from_maybe_place > 0 {
                    lengths.push(l);
                    maybe_lengths.push(l);
                    n_lengths_from_maybe_place -= 1;
                }
            }
            if lengths.len() == complete_arrangement {
                break;
            }
        }

        let mut n_flows_from_must_to_maybe_place = n_must_place * n_from_maybe_place;
        let mut n_flows_in_maybe_place = n_from_maybe_place * n_from_maybe_place.saturating_sub(1) / 2;
        for (f,i,j) in self.pb.sorted_flows.iter().copied() {
            if state.must_place.contains(i) && state.must_place.contains(j) {
                flows.push(f);
            } else if let Some(maybe) = state.maybe_place.as_ref() {
                if state.must_place.contains(i) && maybe.contains(j) && n_flows_from_must_to_maybe_place > 0 {
                    flows.push(f);
                    n_flows_from_must_to_maybe_place -= 1;
                } else if maybe.contains(i) && state.must_place.contains(j) && n_flows_from_must_to_maybe_place > 0 {
                    flows.push(f);
                    n_flows_from_must_to_maybe_place -= 1;
                } else if maybe.contains(i) && maybe.contains(j) && n_flows_in_maybe_place > 0 {
                    flows.push(f);
                    n_flows_in_maybe_place -= 1;
                }
            }

            if flows.len() == n_flows {
                break;
            }
        }

        for i in state.must_place.iter() {
            ratios.push((OrderedFloat((state.cut[i] as f32) / (self.pb.instance.lengths[i] as f32)), self.pb.instance.lengths[i], state.cut[i]));
        }
        
        if let Some(maybe) = state.maybe_place.as_ref() {
            let mut maybe_cuts = vec![];

            for i in maybe.iter() {
                maybe_cuts.push(state.cut[i]);
            }

            maybe_cuts.sort_unstable();

            for i in 0..n_from_maybe_place {
                let l = maybe_lengths[i];
                let c = maybe_cuts[n_from_maybe_place - 1 - i];
                ratios.push((OrderedFloat((c as f32) / (l as f32)), l, c));
            }
        }

        ratios.sort_unstable_by_key(|r| Reverse(*r));

        let mut cut_bound = 0;
        let mut cumul_length = 0;
        for (_, l, c) in ratios.iter() {
            cut_bound += cumul_length * c;
            cumul_length += l;
        }

        let mut edge_bound = 0;
        let mut idx = 0;
        cumul_length = 0;
        for i in 0..(complete_arrangement-1) {
            for _ in 0..(complete_arrangement-(i+1)) {
                edge_bound += cumul_length * flows[n_flows - 1 - idx];
                idx += 1;
            }

            cumul_length += lengths[i];
        }

        - (cut_bound + edge_bound)
    }
}
