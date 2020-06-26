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
use ddo::common::{Decision, Domain, Matrix, Variable, VarSet};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
/// The state of a pigment sequencing problem
pub struct PSPState {
    /// The counters of orders that have been placed and remain to be treated
    pub remaining: Vec<usize>,
    /// The last decision that was made
    pub last_dec : usize
}

/// This item means the machine is idle. It is necessary for our model to
/// be able to cope with relaxation. This choice can only be made when
/// there is no other option.
///
/// This idle marker must really be though of as a mean to tell the solver
/// 'I ended up in an inexact, infeasible state, but I consider to be done
///  with my approximation. I am not unsat'.
pub const IDLE_MARKER: isize = -1;

#[derive(Clone)]
/// The strcuture implementing pigment sequencing problem with changeover cost.
pub struct PSP {
    pub nb_periods: usize,
    pub nb_items  : usize,
    /// nb de demandes (pour le cas ou on produit parfois rien)
    pub nb_orders : usize,

    /// matrice carree
    pub changeover_cost: Matrix<usize>,
    /// stockage
    pub stocking_cost  : Vec<usize>,

    /// deadlines par item.
    pub deadlines: Vec<Vec<usize>>,
    /// somme du nombre de demandes pour chaque item
    pub demands_per_item: Vec<usize>,
}

impl PSP {
    fn latest_schedule_time(&self, state: &PSPState, item: usize) -> usize {
        self.deadlines[item][state.remaining[item]-1]
    }
    fn can_schedule(&self, state: &PSPState, item: usize, now: usize) -> bool {
        state.remaining[item] > 0 && self.latest_schedule_time(state, item) >= now
    }
    fn backward_changeover_cost(&self, state: &PSPState, item: usize) -> usize {
        self.changeover_cost[(item,state.last_dec)]
    }
    fn time_until_deadline(&self, state: &PSPState, item: usize, now: usize) -> usize {
        self.latest_schedule_time(state, item) - now
    }
}

impl Problem<PSPState> for PSP {
    fn nb_vars(&self) -> usize {
        self.nb_periods
    }

    fn initial_state(&self) -> PSPState {
        PSPState {
            last_dec : self.nb_items, // this is dummy !
            remaining: self.demands_per_item.clone()
        }
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn domain_of<'a>(&self, state: &'a PSPState, var: Variable) -> Domain<'a> {
        let now = var.id();
        let mut candidates = vec![];

        for item in 0..self.nb_items {
            if self.can_schedule(state, item, now) {
                candidates.push(item as isize);
            }
        }

        // should i use the idle marker ?
        if candidates.is_empty() {
            candidates.push(IDLE_MARKER);
        }

        candidates.into()
    }

    fn transition(&self, state: &PSPState, _vars: &VarSet, d: Decision) -> PSPState {
        let mut result = state.clone();

        // if the machine is idle, it does not do anything
        if d.value != IDLE_MARKER {
            let v = d.value as usize;

            result.last_dec = v;
            result.remaining[v] -= 1;
        }

        result
    }

    fn transition_cost(&self, state: &PSPState, _vars: &VarSet, d: Decision) -> isize {
        // if the machine is idle, it does not do anything (free... yay !)
        if d.value == IDLE_MARKER {
            0
        } else {
            let item = d.value as usize;
            let now = d.variable.id();
            let changeover = self.backward_changeover_cost(state, item);
            let duration = self.time_until_deadline(state, item, now);
            let stocking = self.stocking_cost[item] * duration;

            -((changeover + stocking) as isize)
        }
    }
}

#[derive(Clone)]
pub struct PSPRelax<'a> {
    pub problem: &'a PSP
}

// Warn: we must take the longest. Even though it is a minimization,
//       all costs have been negated. Hence longest path, is the min
//       solution from the original problem.
impl Relaxation<PSPState> for PSPRelax<'_> {
    fn merge_states(&self, states: &mut dyn Iterator<Item=&PSPState>) -> PSPState {
        let nb_items   = self.problem.nb_items;

        // any last decision is ok.
        let mut last_dec = None;

        let mut remainders = vec![usize::max_value(); nb_items];
        for state in states {
            if last_dec.is_none() {
                last_dec = Some(state.last_dec);
            }
            for (item, rem) in remainders.iter_mut().enumerate() {
                let at_node = state.remaining[item];
                *rem = min(*rem, at_node);
            }
        }

        PSPState {
            last_dec : last_dec.unwrap(),
            remaining: remainders
        }
    }

    fn relax_edge(&self, _: &PSPState, _: &PSPState, _: &PSPState, _: Decision, cost: isize) -> isize {
        cost
    }

    fn estimate  (&self, _state  : &PSPState) -> isize { 0 }
}