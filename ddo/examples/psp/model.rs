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

use std::{vec, collections::BinaryHeap};

use ddo::*;
use smallbitset::Set32;

use crate::ub_utils::all_mst;

/// The state of the DP model
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PspState {
    pub time: usize,
    /// The item that was produced at time t+1 
    /// (a value of -1 means that we don't know the item that is being produced next)
    pub next: isize,
    /// The time at which the previous demand for each item had been filled
    pub prev_demands: Vec<isize>,
}

/// A constant to tell your machine wont do anything
pub const IDLE: isize = -1;

/// This structure describes a PSP instance
#[derive(Debug)]
pub struct Psp {
    pub n_items: usize,
    pub horizon: usize,
    pub stocking: Vec<usize>,
    pub changeover: Vec<Vec<usize>>,
    pub prev_demands: Vec<Vec<isize>>,
    pub rem_demands: Vec<Vec<isize>>,
}

impl Problem for Psp {
    type State = PspState;

    fn nb_variables(&self) -> usize {
        self.horizon
    }

    fn initial_state(&self) -> Self::State {
        let mut prev_demands = vec![];
        for i in 0..self.n_items {
            prev_demands.push(self.prev_demands[i][self.horizon]);
        }

        PspState {
            time: self.horizon, 
            next: -1,
            prev_demands
        }
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn transition(&self, state: &Self::State, decision: ddo::Decision) -> Self::State {
        let mut ret = state.clone();
        ret.time -= 1;

        if decision.value != IDLE {
            let d        = decision.value as usize;
            ret.next            = decision.value;
            ret.prev_demands[d] = self.prev_demands[d][state.prev_demands[d] as usize];
        }

        ret
    }

    fn transition_cost(&self, state: &Self::State, decision: ddo::Decision, _successor: &Self::State) -> isize {
        if decision.value == IDLE {
            0
        } else {
            let d = decision.value as usize;
            let t = decision.variable.id() as isize;
            let duration = state.prev_demands[d] - t;
            let stocking = self.stocking[d] as isize * duration;
            let changeover = 
                if state.next != -1 {
                    self.changeover[d][state.next as usize]
                } else {
                    0
                };
            
            -(changeover as isize + stocking)
        }
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<ddo::Variable> {
        if depth < self.horizon {
            Some(Variable(self.horizon - depth - 1))
        } else {
            None
        }
    }

    fn for_each_in_domain(&self, variable: ddo::Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        let t = variable.id() as isize;
        let dom = (0..self.n_items).filter(|i| state.prev_demands[*i] >= t).collect::<Vec<usize>>();
        let rem_demands = (0..self.n_items).filter(|i| state.prev_demands[*i] >= 0).map(|i| self.rem_demands[i][state.prev_demands[i] as usize]).sum::<isize>();

        if rem_demands > t + 1 {
            return;
        }

        for i in dom.iter() {
            f.apply(Decision {variable, value: *i as isize});
        }

        if rem_demands < t + 1 {
            f.apply(Decision {variable, value: IDLE});
        }
    }
}

/// This structure implements the PSP relaxation
pub struct PspRelax<'a>{
    pb: &'a Psp,
    mst: Vec<usize>,
}

impl <'a> PspRelax<'a> {
    pub fn new(pb: &'a Psp) -> Self {
        let mst = all_mst(&pb.changeover);

        Self { pb, mst }
    }

    fn members(state: &PspState) -> Set32 {
        let mut mem = Set32::empty();
        for (i, d) in state.prev_demands.iter().copied().enumerate() {
            if d >= 0 {
                mem = mem.insert(i as u8);
            }
        }
        if state.next != -1 {
            mem = mem.insert(state.next as u8);
        }
        mem
    }
}

impl Relaxation for PspRelax<'_> {
    type State = PspState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut time = self.pb.horizon;
        let mut prev_demands = vec![isize::MAX; self.pb.n_items];

        for s in states {
            time = time.min(s.time);
            prev_demands.iter_mut()
                .zip(s.prev_demands.iter().copied())
                .for_each(|(x, y)| *x = y.min(*x));
        }

        PspState{time, next: -1, prev_demands}
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
        let idx: u32 = u32::from(Self::members(state));
        let co = self.mst[idx as usize] as isize;

        let mut prev_demands = state.prev_demands.clone();
        let mut ww = 0;
        let mut items = BinaryHeap::new();
        for time in (0..state.time).rev() {
            for i in 0..self.pb.n_items {
                while prev_demands[i] >= time as isize {
                    items.push((self.pb.stocking[i], prev_demands[i]));
                    prev_demands[i] = self.pb.prev_demands[i][prev_demands[i] as usize];
                }
            }

            if let Some((cost, deadline)) = items.pop() {
                ww += cost as isize * (time as isize - deadline);
            }
        }
    
        -(co + ww)
    }
}


/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
pub struct PspRanking;
impl StateRanking for PspRanking {
    type State = PspState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        let tot_a = a.prev_demands.iter().sum::<isize>();
        let tot_b = b.prev_demands.iter().sum::<isize>();
        
        tot_a.cmp(&tot_b)
    }
}
