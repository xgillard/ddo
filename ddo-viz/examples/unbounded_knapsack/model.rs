// Copyright 2022 Xavier Gillard
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

//! This is an example of how to visualize an instance of the unbounded knapsack
//! with ddo viz. This module comprises all the modeling artifacts of the problem.
//! (not directly relevant to visualization).

use std::fmt::Debug;

use ddo::*;


#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct KPState {
    pub depth: usize,
    pub capa : usize,
}

impl Debug for KPState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.capa))
    }
}

pub struct UnboundedKP<'a> {
    pub capa: usize,
    pub weight: &'a[usize],
    pub profit: &'a[usize],
}
impl Problem for UnboundedKP<'_> {
    type State = KPState;

    fn nb_variables(&self) -> usize {
        self.weight.len()
    }

    fn initial_state(&self) -> Self::State {
        KPState{depth: 0, capa: self.capa}
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn transition(&self, state: &Self::State, decision: ddo::Decision) -> Self::State {
        let capa = state.capa - decision.value as usize * self.weight[decision.variable.0];
        KPState {depth: state.depth + 1, capa}
    }

    fn transition_cost(&self, _: &Self::State, decision: ddo::Decision) -> isize {
        decision.value * self.profit[decision.variable.0] as isize
    }

    fn next_variable(&self, next_layer: &mut dyn Iterator<Item = &Self::State>)
        -> Option<ddo::Variable> {
        let s = next_layer.next()
            .map(|s| s.depth)
            .unwrap_or(usize::MAX);
        
        if s < self.nb_variables() {
            Some(Variable(s))
        } else {
            None
        }
    }

    fn for_each_in_domain(&self, var: ddo::Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        let limit = state.capa / self.weight[var.0];
        for i in 0..=limit {
            f.apply(Decision {variable: var, value: i as isize})
        }
    }
}

pub struct KPRelax;
impl Relaxation for KPRelax {
    type State = KPState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut depth = usize::MAX;
        let mut capa = 0;

        for x in states {
            depth = depth.min(x.depth);
            capa  = capa.max(x.capa);
        }

        KPState {depth, capa}
    }

    fn relax(
        &self,
        _source: &Self::State,
        _dest: &Self::State,
        _new: &Self::State,
        _decision: Decision,
        cost: isize,
    ) -> isize {
        cost
    }
}

pub struct KPRank;
impl StateRanking for KPRank {
    type State = KPState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.capa.cmp(&b.capa)
            .then_with(|| a.depth.cmp(&b.depth))
            .reverse()
    }
}
