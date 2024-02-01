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

//! This module provides the implementation of various cutoff heuristics that can 
//! be used to tune the behavior of a MDD solver.

use std::{sync::{Arc, atomic::AtomicBool}, time::Duration};

use crate::Cutoff;

/// _This is the default cutoff heuristic._ It imposes that the search goes
/// proves optimality before to stop.
/// 
/// # Typical Usage Example
/// The cutoff policy is typically created when instantiating a solver. The following
/// example shows how one can create a solver that never stops before it found the 
/// optimal solution to a given problem instance.
/// 
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// pub struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn nb_variables(&self) -> usize {
/// #         self.profit.len()
/// #     }
/// #     fn initial_state(&self) -> Self::State {
/// #         KnapsackState{ depth: 0, capacity: self.capacity }
/// #     }
/// #     fn initial_value(&self) -> isize {
/// #         0
/// #     }
/// #     fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
/// #         let mut ret = state.clone();
/// #         ret.depth  += 1;
/// #         if dec.value == TAKE_IT { 
/// #             ret.capacity -= self.weight[dec.variable.id()] 
/// #         }
/// #         ret
/// #     }
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
/// #         self.profit[dec.variable.id()] as isize * dec.value
/// #     }
/// #     fn next_variable(&self, depth: usize, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
/// #         let n = self.nb_variables();
/// #         if depth < n {
/// #             Some(Variable(depth))
/// #         } else {
/// #             None
/// #         }
/// #     }
/// #     fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
/// #     {
/// #         if state.capacity >= self.weight[variable.id()] {
/// #             f.apply(Decision { variable, value: TAKE_IT });
/// #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
/// #         } else {
/// #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
/// #         }
/// #     }
/// }
/// struct KPRelax<'a>{pb: &'a Knapsack}
/// impl Relaxation for KPRelax<'_> {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
/// #         states.max_by_key(|node| node.capacity).copied().unwrap()
/// #     }
/// #     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
/// #         cost
/// #     }
/// }
/// # 
/// struct KPRanking;
/// impl StateRanking for KPRanking {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
/// #         a.capacity.cmp(&b.capacity)
/// #     }
/// }
/// pub struct KPDominance;
/// impl Dominance for KPDominance {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     type Key = usize;
/// #     fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
/// #        Some(state.depth)
/// #     }
/// #     fn nb_dimensions(&self, _state: &Self::State) -> usize {
/// #         1
/// #     }
/// #     fn get_coordinate(&self, state: &Self::State, _: usize) -> isize {
/// #         state.capacity as isize
/// #     }
/// #     fn use_value(&self) -> bool {
/// #         true
/// #     }
/// }
/// 
/// let problem = Knapsack {
///       // details omitted
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let relaxation = KPRelax{pb: &problem};
/// let width = FixedWidth(100);
/// let dominance = SimpleDominanceChecker::new(KPDominance, problem.nb_variables());
/// let heuristic = KPRanking;
/// let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
/// #
/// let mut solver = DefaultSolver::new(
///       &problem, 
///       &relaxation, 
///       &heuristic, 
///       &width,
///       &dominance,
///       // this solver will only stop when optimality is proved
///       &NoCutoff, 
///       &mut fringe);
/// let outcome = solver.maximize();
/// ```
#[derive(Debug, Default, Copy, Clone)]
pub struct NoCutoff;
impl Cutoff for NoCutoff {
    fn must_stop(&self) -> bool {false}
}
/// This cutoff allows one to specify a maximum time budget to solve the problem.
/// Once the time budget is elapsed, the optimization stops and the best solution
/// that has been found (so far) is returned.
///
/// # Typical Usage Example
/// The cutoff policy is typically created when instantiating a solver. The following
/// example shows how one can create a solver that is allowed to run for no more than
/// 30 seconds (wall time).
/// 
/// ```
/// # use std::time::Duration;
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// pub struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn nb_variables(&self) -> usize {
/// #         self.profit.len()
/// #     }
/// #     fn initial_state(&self) -> Self::State {
/// #         KnapsackState{ depth: 0, capacity: self.capacity }
/// #     }
/// #     fn initial_value(&self) -> isize {
/// #         0
/// #     }
/// #     fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
/// #         let mut ret = state.clone();
/// #         ret.depth  += 1;
/// #         if dec.value == TAKE_IT { 
/// #             ret.capacity -= self.weight[dec.variable.id()] 
/// #         }
/// #         ret
/// #     }
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
/// #         self.profit[dec.variable.id()] as isize * dec.value
/// #     }
/// #     fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
/// #         let n = self.nb_variables();
/// #         if depth < n {
/// #             Some(Variable(depth))
/// #         } else {
/// #             None
/// #         }
/// #     }
/// #     fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
/// #     {
/// #         if state.capacity >= self.weight[variable.id()] {
/// #             f.apply(Decision { variable, value: TAKE_IT });
/// #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
/// #         } else {
/// #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
/// #         }
/// #     }
/// }
/// struct KPRelax<'a>{pb: &'a Knapsack}
/// impl Relaxation for KPRelax<'_> {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
/// #         states.max_by_key(|node| node.capacity).copied().unwrap()
/// #     }
/// #     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
/// #         cost
/// #     }
/// }
/// # 
/// struct KPRanking;
/// impl StateRanking for KPRanking {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
/// #         a.capacity.cmp(&b.capacity)
/// #     }
/// }
/// 
/// pub struct KPDominance;
/// impl Dominance for KPDominance {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     type Key = usize;
/// #     fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
/// #        Some(state.depth)
/// #     }
/// #     fn nb_dimensions(&self, _state: &Self::State) -> usize {
/// #         1
/// #     }
/// #     fn get_coordinate(&self, state: &Self::State, _: usize) -> isize {
/// #         state.capacity as isize
/// #     }
/// #     fn use_value(&self) -> bool {
/// #         true
/// #     }
/// }
/// 
/// let problem = Knapsack {
///       // details omitted
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let relaxation = KPRelax{pb: &problem};
/// let width = FixedWidth(100);
/// let dominance = SimpleDominanceChecker::new(KPDominance, problem.nb_variables());
/// let heuristic = KPRanking;
/// 
/// // this solver will be allowed to run for 30 seconds
/// let cutoff = TimeBudget::new(Duration::from_secs(30));
/// let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
/// #
/// let mut solver = DefaultSolver::new(
///       &problem, 
///       &relaxation, 
///       &heuristic, 
///       &width,
///       &dominance,
///       &cutoff, 
///       &mut fringe);
/// let outcome = solver.maximize();
/// ```
#[derive(Debug, Clone)]
pub struct TimeBudget {
    stop  : Arc<AtomicBool>
}
impl TimeBudget {
    pub fn new(budget: Duration) -> Self {
        let stop   = Arc::new(AtomicBool::new(false));
        let t_flag = Arc::clone(&stop);
        
        // timer
        std::thread::spawn(move || {
            std::thread::sleep(budget);
            t_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        });

        TimeBudget { stop }
    }
}
impl Cutoff for TimeBudget {
    fn must_stop(&self) -> bool {
        self.stop.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use std::{time::Duration, thread};

    use crate::*;

    #[test]
    fn no_cutoff_must_never_stops() {
        let cutoff = NoCutoff;
        assert!(!cutoff.must_stop());
    }

    #[test]
    fn time_budget_must_stop_only_when_elapsed() {
        let cutoff = TimeBudget::new(Duration::from_secs(3));
        assert!(!cutoff.must_stop());
        thread::sleep(Duration::from_secs(4));
        assert!(cutoff.must_stop());
    }
}