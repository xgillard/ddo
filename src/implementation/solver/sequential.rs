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

//! This module provides the implementation of a sequential mdd solver. That is
//! a solver that will solve the problem on one single thread.
use std::hash::Hash;
use std::marker::PhantomData;

use crate::abstraction::frontier::Frontier;
use crate::abstraction::mdd::{Config, MDD};
use crate::abstraction::solver::Solver;
use crate::common::{Solution, Reason, Completion};
use crate::implementation::frontier::SimpleFrontier;

/// This is the structure implementing an single-threaded MDD solver.
///
/// # Example Usage
/// ```
/// # use ddo::*;
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
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let mdd        = mdd_builder(&problem, relaxation).into_deep();
/// // the solver is created using an mdd.
/// let mut solver = SequentialSolver::new(mdd);
/// // the outcome provides the value of the best solution that was found for
/// // the problem (if one was found) along with a flag indicating whether or
/// // not the solution was proven optimal. Hence an unsatisfiable problem
/// // would have `outcome.best_value == None` and `outcome.is_exact` true.
/// // The `is_exact` flag will only be false if you explicitly decide to stop
/// // searching with an arbitrary cutoff.
/// let outcome    = solver.maximize();
/// // The best solution (if one exist) is retrieved with
/// let solution   = solver.best_solution();
/// ```
pub struct SequentialSolver<T, C, F, DD>
    where T : Hash + Eq + Clone,
          C : Config<T> + Clone,
          F : Frontier<T>,
          DD: MDD<T, C> + Clone
{
    config: C,
    mdd: DD,
    fringe: F,
    explored : usize,
    best_ub  : isize,
    best_lb  : isize,
    best_sol : Option<Solution>,
    verbosity: u8,
    _phantom : PhantomData<T>
}
// private interface.
impl <T, C, DD> SequentialSolver<T, C, SimpleFrontier<T>, DD>
    where T : Hash + Eq + Clone,
          C : Config<T> + Clone,
          DD: MDD<T, C> + Clone
{
    pub fn new(mdd: DD) -> Self {
        Self::customized(mdd, 0)
    }
    pub fn customized(mdd: DD, verbosity: u8) -> Self {
        SequentialSolver {
            config: mdd.config().clone(),
            mdd,
            fringe: SimpleFrontier::default(),
            explored: 0,
            best_ub: std::isize::MAX,
            best_lb: std::isize::MIN,
            best_sol: None,
            verbosity,
            _phantom : PhantomData
        }
    }
}
impl <T, C, F, DD> SequentialSolver<T, C, F, DD>
    where T : Hash + Eq + Clone,
          C : Config<T> + Clone,
          F : Frontier<T>,
          DD: MDD<T, C> + Clone
{
    pub fn with_verbosity(mut self, verbosity: u8) -> Self {
        self.verbosity = verbosity;
        self
    }
    pub fn with_frontier<FF: Frontier<T> + Send + Sync>(self, ff: FF) -> SequentialSolver<T, C, FF, DD> {
        SequentialSolver {
            config   : self.config,
            mdd      : self.mdd,
            fringe   : ff,
            explored : self.explored,
            best_ub  : self.best_ub,
            best_lb  : self.best_lb,
            best_sol : self.best_sol,
            verbosity: self.verbosity,
            _phantom : PhantomData
        }
    }
    fn maybe_update_best(&mut self) {
        if self.mdd.best_value() > self.best_lb {
            self.best_lb   = self.mdd.best_value();
            self.best_sol = self.mdd.best_solution();
        }
    }

    fn try_maximize(&mut self) -> Result<(), Reason> {
        let root = self.config.root_node();
        self.fringe.push(root);

        while !self.fringe.is_empty() {
            let node = self.fringe.pop().unwrap();

            // Nodes are sorted on UB as first criterion. It can be updated
            // whenever we encounter a tighter value
            if node.ub < self.best_ub {
                self.best_ub = node.ub;
            }

            // We just proved optimality, Yay !
            if self.best_lb >= self.best_ub {
                break;
            }

            // Skip if this node cannot improve the current best solution
            if node.ub < self.best_lb {
                break;
            }

            self.explored += 1;
            if self.verbosity >= 2 && self.explored % 100 == 0 {
                println!("Explored {}, LB {}, UB {}, Fringe sz {}", self.explored, self.best_lb, node.ub, self.fringe.len());
            }

            // 1. RESTRICTION
            let restriction = self.mdd.restricted(&node, self.best_lb, self.best_ub)?;
            self.maybe_update_best();
            if restriction.is_exact {
                continue;
            }

            // 2. RELAXATION
            let relaxation = self.mdd.relaxed(&node, self.best_lb, self.best_ub)?;
            if relaxation.is_exact {
                self.maybe_update_best();
            } else {
                let best_ub= self.best_ub;
                let best_lb= self.best_lb;
                let fringe = &mut self.fringe;
                let mdd    = &mut self.mdd;
                mdd.for_each_cutset_node(|mut cutset_node| {
                    cutset_node.ub = best_ub.min(cutset_node.ub);
                    if cutset_node.ub > best_lb {
                        fringe.push(cutset_node);
                    }
                });
            }
        }

        // return
        if self.verbosity >= 1 {
            println!("Final {}, Explored {}", self.best_lb, self.explored);
        }

        Ok(())
    }
}

impl <T, C, F, DD> Solver for SequentialSolver<T, C, F, DD>
    where T : Hash + Eq + Clone,
          C : Config<T> + Clone,
          F : Frontier<T>,
          DD: MDD<T, C> + Clone
{
    /// Applies the branch and bound algorithm proposed by Bergman et al. to
    /// solve the problem to optimality.
    fn maximize(&mut self) -> Completion {
        let outcome = self.try_maximize();
        if outcome.is_ok() { // We proved optimality
            Completion {
                is_exact: true,
                best_value: self.best_value()
            }
        } else { // We did not prove the optimality of our solution
            Completion {
                is_exact  : false,
                best_value: self.best_value()
            }
        }
    }

    fn best_value(&self) -> Option<isize> {
        self.best_sol.as_ref().map(|_| self.best_lb)
    }
    fn best_solution(&self) -> Option<Solution> {
        self.best_sol.clone()
    }

    /// Sets the best known value and/or solution. This solution and value may
    /// be obtained from any other available means (LP relax for instance).
    fn set_primal(&mut self, value: isize, best_sol: Solution) {
        if value > self.best_lb {
            self.best_lb = value;
            self.best_sol= Some(best_sol);
        }
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################

/// Unlike the rest of the library, the solvers modules are not tested in depth
/// with unit tests (this is way too hard to do even for the sequential module).
/// So we basically unit test the configuration capabilities of the solvers
/// and then resort to the solving of benchmark instances (see examples) with
/// known optimum solution to validate the behavior of the maximize function.

#[cfg(test)]
mod test_solver {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::solver::Solver;
    use crate::common::{Decision, Domain, PartialAssignment, Solution, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::solver::sequential::SequentialSolver;
    use crate::abstraction::frontier::Frontier;

    /// Describe the binary knapsack problem in terms of a dynamic program.
        /// Here, the state of a node, is nothing more than an unsigned integer (usize).
        /// That unsigned integer represents the remaining capacity of our sack.
    #[derive(Debug, Clone)]
    struct Knapsack {
        capacity: usize,
        profit  : Vec<usize>,
        weight  : Vec<usize>
    }
    impl Problem<usize> for Knapsack {
        fn nb_vars(&self) -> usize {
            self.profit.len()
        }
        fn domain_of<'a>(&self, state: &'a usize, var: Variable) ->Domain<'a> {
            if *state >= self.weight[var.id()] {
                vec![0, 1].into()
            } else {
                vec![0].into()
            }
        }
        fn initial_state(&self) -> usize {
            self.capacity
        }
        fn initial_value(&self) -> isize {
            0
        }
        fn transition(&self, state: &usize, _vars: &VarSet, dec: Decision) -> usize {
            state - (self.weight[dec.variable.id()] * dec.value as usize)
        }
        fn transition_cost(&self, _state: &usize, _vars: &VarSet, dec: Decision) -> isize {
            self.profit[dec.variable.id()] as isize * dec.value
        }
    }

    /// Merge the nodes by creating a new fake node that has the maximum remaining
    /// capacity from the merged nodes.
    #[derive(Debug, Clone)]
    struct KPRelax;
    impl Relaxation<usize> for KPRelax {
        /// To merge a given selection of states (capacities) we will keep the
        /// maximum capacity. This is an obvious relaxation as it allows us to
        /// put more items in the sack.
        fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
            // the selection is guaranteed to have at least one state so using
            // unwrap after max to get rid of the wrapping 'Option' is perfectly safe.
            *states.max().unwrap()
        }
        /// When relaxing (merging) the states, we did not run into the risk of
        /// possibly decreasing the maximum objective value reachable from the
        /// components of the merged node. Hence, we dont need to do anything
        /// when relaxing the edge. Still, if we wanted to, we could chose to
        /// return an higher value.
        fn relax_edge(&self, _src: &usize, _dst: &usize, _relaxed: &usize, _d: Decision, cost: isize) -> isize {
            cost
        }
    }

    #[test]
    fn by_default_verbosity_is_zero() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = SequentialSolver::new(mdd);
        assert_eq!(0, solver.verbosity);
    }
    #[test]
    fn verbosity_can_be_customized() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = SequentialSolver::new(mdd).with_verbosity(2);
        assert_eq!(2, solver.verbosity);
    }
    #[test]
    fn no_solution_before_solving() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = SequentialSolver::new(mdd);
        assert!(solver.best_sol.is_none());
    }
    #[test]
    fn empty_fringe_before_solving() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = SequentialSolver::new(mdd);
        assert!(solver.fringe.is_empty());
    }
    #[test]
    fn default_best_lb_is_neg_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = SequentialSolver::new(mdd);
        assert_eq!(isize::min_value(), solver.best_lb);
    }
    #[test]
    fn default_best_ub_is_pos_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = SequentialSolver::new(mdd);
        assert_eq!(isize::max_value(), solver.best_ub);
    }

    #[test]
    fn maximizes_yields_the_optimum() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let  mdd = mdd_builder(&problem, KPRelax).into_deep();

        let mut solver = SequentialSolver::new(mdd);
        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap().iter().collect::<Vec<Decision>>();
        sln.sort_unstable_by_key(|d| d.variable.id());
        assert_eq!(sln, vec![
            Decision{variable: Variable(0), value: 0},
            Decision{variable: Variable(1), value: 1},
            Decision{variable: Variable(2), value: 1},
        ]);
    }

    #[test]
    fn maximizes_yields_the_optimum_2() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 210, 12, 5, 100, 120, 110],
            weight  : vec![10,  45, 20, 4,  20,  30,  50]
        };
        let mdd = mdd_builder(&problem, KPRelax)
            .with_max_width(FixedWidth(2))
            .into_deep();

        let mut solver = SequentialSolver::new(mdd);
        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap().iter().collect::<Vec<Decision>>();
        sln.sort_unstable_by_key(|d| d.variable.id());
        assert_eq!(sln, vec![
            Decision { variable: Variable(0), value: 0 },
            Decision { variable: Variable(1), value: 0 },
            Decision { variable: Variable(2), value: 0 },
            Decision { variable: Variable(3), value: 0 },
            Decision { variable: Variable(4), value: 1 },
            Decision { variable: Variable(5), value: 1 },
            Decision { variable: Variable(6), value: 0 }
        ]);
    }
    #[test]
    fn set_primal_overwrites_best_value_and_sol_if_it_improves() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd        = mdd_builder(&problem, KPRelax).into_deep();
        let mut solver = SequentialSolver::new(mdd);

        let d1  = Decision{variable: Variable(0), value: 10};
        let sol = Solution::new(Arc::new(PartialAssignment::SingleExtension {decision: d1, parent: Arc::new(PartialAssignment::Empty)}));

        solver.set_primal(10, sol.clone());
        assert!(solver.best_sol.is_some());
        assert_eq!(10, solver.best_lb);

        // in this case, it wont update because there is no improvement
        solver.set_primal(5, sol.clone());
        assert!(solver.best_sol.is_some());
        assert_eq!(10, solver.best_lb);

        // but here, it will update as it improves the best known sol
        solver.set_primal(10000, sol);
        assert!(solver.best_sol.is_some());
        assert_eq!(10000, solver.best_lb);

        // it wont do much as the primal is better than the actual feasible solution
        let maximized = solver.maximize();
        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(10000));
        assert!(solver.best_solution().is_some());
    }
}
