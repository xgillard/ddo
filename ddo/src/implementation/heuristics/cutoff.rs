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
#[derive(Debug, Default, Copy, Clone)]
pub struct NoCutoff;
impl Cutoff for NoCutoff {
    fn must_stop(&self) -> bool {false}
}
/// This cutoff allows one to specify a maximum time budget to solve the problem.
/// Once the time budget is elapsed, the optimization stops and the best solution
/// that has been found (so far) is returned.
///
/// # Example
/// ```
/// # use ddo::*;
/// use std::time::Duration;
/// #
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> isize   { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         (0..=1).into()
/// #     }
/// #     fn transition(&self, state: &usize, _: &VarSet, _: Decision) -> usize {
/// #         41
/// #     }
/// #     fn transition_cost(&self, state: &usize, _: &VarSet, _: Decision) -> isize {
/// #         42
/// #     }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #     fn merge_states(&self, n: &mut dyn Iterator<Item=&usize>) -> usize {
/// #         *n.next().unwrap()
/// #     }
/// #     fn relax_edge(&self, _src: &usize, _dst: &usize, _rlx: &usize, _d: Decision, cost: isize) -> isize {
/// #        cost
/// #     }
/// # }
/// # let problem = MockProblem;
/// # let relax   = MockRelax;
/// let mdd = mdd_builder(&problem, relax)
///         .with_cutoff(TimeBudget::new(Duration::from_secs(10)))
///         .into_deep();
/// let mut solver = ParallelSolver::new(mdd);
/// let optimum = solver.maximize(); // will run for maximum 10 seconds
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