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

use binary_heap_plus::BinaryHeap;

use crate::abstraction::mdd::{MDD, Config};
use crate::abstraction::solver::Solver;
use crate::common::{FrontierNode, Solution};
use crate::implementation::heuristics::MaxUB;

/// This is the structure implementing an single-threaded MDD solver.
///
/// # Example Usage
/// ```
/// # use ddo::common::{Variable, Domain, VarSet, Decision};
/// # use ddo::abstraction::dp::{Problem, Relaxation};
/// # use ddo::abstraction::solver::Solver;
/// #
/// # use ddo::implementation::heuristics::FixedWidth;
/// # use ddo::implementation::mdd::config::mdd_builder;
/// # use ddo::implementation::solver::sequential::SequentialSolver;
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
/// // val is the optimal value of the objective function,
/// // sol is the sequence of decision yielding that optimal value (if sol exists, `sol != None`)
/// let (val, sol) = solver.maximize();
/// ```
pub struct SequentialSolver<T, C, DD>
    where T : Hash + Eq + Clone,
          C : Config<T> + Clone,
          DD: MDD<T, C> + Clone
{
    config: C,
    mdd: DD,
    fringe: BinaryHeap<FrontierNode<T>, MaxUB>,
    explored : usize,
    best_ub  : isize,
    best_lb  : isize,
    best_sol : Option<Solution>,
    verbosity: u8
}
// private interface.
impl <T, C, DD> SequentialSolver<T, C, DD>
    where T : Hash + Eq + Clone,
          C : Config<T> + Clone,
          DD: MDD<T, C> + Clone
{
    pub fn new(mdd: DD) -> Self {
        Self::customized(mdd, 0)
    }
    pub fn with_verbosity(mdd: DD, verbosity: u8) -> Self {
        Self::customized(mdd, verbosity)
    }
    pub fn customized(mdd: DD, verbosity: u8) -> Self {
        SequentialSolver {
            config: mdd.config().clone(),
            mdd,
            fringe: BinaryHeap::from_vec_cmp(vec![], MaxUB),
            explored: 0,
            best_ub: std::isize::MAX,
            best_lb: std::isize::MIN,
            best_sol: None,
            verbosity
        }
    }
    fn maybe_update_best(&mut self) {
        if self.mdd.best_value() > self.best_lb {
            self.best_lb   = self.mdd.best_value();
            self.best_sol = self.mdd.best_solution();
        }
    }
}

impl <T, C, DD> Solver for SequentialSolver<T, C, DD>
    where T : Hash + Eq + Clone,
          C : Config<T> + Clone,
          DD: MDD<T, C> + Clone
{
    /// Applies the branch and bound algorithm proposed by Bergman et al. to
    /// solve the problem to optimality.
    fn maximize(&mut self) -> (isize, Option<Solution>) {
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
            self.mdd.restricted(&node, self.best_lb);
            self.maybe_update_best();
            if self.mdd.is_exact() {
                continue;
            }

            // 2. RELAXATION
            self.mdd.relaxed(&node, self.best_lb);
            if self.mdd.is_exact() {
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
        (self.best_lb, self.best_sol.clone())
    }

    /// Sets the best known value and/or solution. This solution and value may
    /// be obtained from any other available means (LP relax for instance).
    fn set_primal(&mut self, value: isize, best_sol: Option<Solution>) {
        if value > self.best_lb {
            self.best_lb = value;
            self.best_sol= best_sol;
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
    use crate::common::{Decision, Domain, Variable, VarSet, Solution, PartialAssignment};
    use crate::abstraction::dp::{Problem, Relaxation};

    use crate::implementation::mdd::config::mdd_builder;
    use crate::implementation::solver::sequential::SequentialSolver;
    use crate::abstraction::solver::Solver;
    use crate::implementation::heuristics::FixedWidth;
    use std::sync::Arc;

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
        let solver = SequentialSolver::with_verbosity(mdd, 2);
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
        let (v, s) = solver.maximize();

        assert_eq!(v, 220);
        assert!(s.is_some());

        let mut sln = s.unwrap().iter().collect::<Vec<Decision>>();
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
        let (v, s) = solver.maximize();

        assert_eq!(v, 220);
        assert!(s.is_some());

        let mut sln = s.unwrap().iter().collect::<Vec<Decision>>();
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

        solver.set_primal(10, Some(sol));
        assert!(solver.best_sol.is_some());
        assert_eq!(10, solver.best_lb);

        // in this case, it wont update because there is no improvement
        solver.set_primal(5, None);
        assert!(solver.best_sol.is_some());
        assert_eq!(10, solver.best_lb);

        // but here, it will update as it improves the best known sol
        solver.set_primal(10000, None);
        assert!(solver.best_sol.is_none());
        assert_eq!(10000, solver.best_lb);

        // it wont do much as the primal is better than the actual feasible solution
        let (val, sol) = solver.maximize();
        assert_eq!(10000, val);
        assert!(sol.is_none());
    }
}
