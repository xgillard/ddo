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
//! of the SOP. (Implementation of the `Problem` trait).

use ddo::{Problem, Variable, Decision, DecisionCallback};

use crate::{io_utils::SopInstance, state::{SopState, Previous}, BitSet};

/// This is the structure encapsulating the Sop problem.
#[derive(Debug, Clone)]
pub struct Sop {
    pub instance: SopInstance,
    pub initial : SopState,
    pub cheapest_edges: Vec<Vec<(isize, usize)>>,
}
impl Sop {
    pub fn new(inst: SopInstance) -> Self {
        let cheapest_edges: Vec<Vec<(isize, usize)>> = Self::compute_cheapest_edges(&inst);
        let mut must_schedule = BitSet::default();
        (1..inst.nb_jobs).for_each(|i| {must_schedule.add_inplace(i as usize);});
        let state = SopState {
            previous: Previous::Job(0),
            must_schedule,
            maybe_schedule: None,
            depth : 0
        };
        Self { instance: inst, initial: state, cheapest_edges }
    }

    fn compute_cheapest_edges(inst: &SopInstance) -> Vec<Vec<(isize, usize)>> {
        let mut cheapest = vec![];
        let n = inst.nb_jobs as usize;
        for i in 0..n {
            let mut cheapest_to_i = vec![];
            for j in 0..n {
                if i == j || inst.distances[j][i] == -1 {
                    continue;
                }
                cheapest_to_i.push((inst.distances[j][i], j));
            }
            cheapest_to_i.sort_unstable();
            cheapest.push(cheapest_to_i);
        }
        cheapest
    }
}

impl Problem for Sop {
    type State = SopState;

    fn nb_variables(&self) -> usize {
        // -1 because we always start from the first job
        (self.instance.nb_jobs - 1) as usize
    }

    fn initial_state(&self) -> SopState {
        self.initial.clone()
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback) {
        // When we are at the end of the schedule, the only possible destination is
        // to go to the last job.
        if state.depth as usize == self.nb_variables() - 1 {
            f.apply(Decision { variable, value: (self.instance.nb_jobs - 1) as isize });
        } else {
            for i in state.must_schedule.iter() {
                if self.can_schedule(state, i) {
                    f.apply(Decision { variable, value: i as isize })
                }
            }
    
            // Add those that can possibly be scheduled
            if let Some(maybe_visit) = &state.maybe_schedule {
                for i in maybe_visit.iter() {
                    if self.can_schedule(state, i) {
                        f.apply(Decision { variable, value: i as isize })
                    }
                }
            }
        }
    }

    fn transition(&self, state: &SopState, d: Decision) -> SopState {
        let job = d.value as usize;

        let mut next = *state;

        next.previous = Previous::Job(job);
        next.depth += 1;
        
        next.must_schedule.remove_inplace(job);
        if let Some(maybe) = next.maybe_schedule.as_mut() {
            maybe.remove_inplace(job);
        }
        
        next
    }

    fn transition_cost(&self, state: &SopState, d: Decision) -> isize {
        // Sop is a minimization problem but the solver works with a 
        // maximization perspective. So we have to negate the cost.

        - self.min_distance_to(state, d.value as usize)
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<Variable> {
        if depth < self.nb_variables() {
            Some(Variable(depth))
        } else {
            None
        }
    }
}

impl Sop {
    pub fn can_schedule(&self, state: &SopState, j: usize) -> bool {
        let maybe_scheduled = match &state.maybe_schedule {
            Some(maybes) => (state.must_schedule.union(*maybes)).flip(),
            None => state.must_schedule.flip(),
        };
        maybe_scheduled.contains_all(self.instance.predecessors[j])
    }
    pub fn min_distance_to(&self, state: &SopState, j: usize) -> isize {
        match &state.previous {
            Previous::Job(i) => if self.instance.distances[*i as usize][j] == -1 {
                isize::MAX
            } else {
                self.instance.distances[*i as usize][j]
            },
            Previous::Virtual(candidates) => 
                candidates.iter()
                    .map(|i| self.instance.distances[i as usize][j as usize])
                    .filter(|w| *w != -1)
                    .min()
                    .unwrap()
        }
    }
}
