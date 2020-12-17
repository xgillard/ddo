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

//! This module provides the implementation of a parallel mdd solver. That is
//! a solver that will solve the problem using as many threads as requested.
//! By default, it uses as many threads as the number of hardware threads
//! available on the machine.
use std::clone::Clone;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

use crate::abstraction::mdd::{Config, MDD};
use crate::abstraction::solver::Solver;
use crate::common::{FrontierNode, Solution, Reason, Completion};
use crate::abstraction::frontier::Frontier;
use crate::implementation::frontier::SimpleFrontier;

/// The shared data that may only be manipulated within critical sections
struct Critical<T : Eq + Hash, F: Frontier<T>> {
    /// This is the fringe: the set of nodes that must still be explored before
    /// the problem can be considered 'solved'.
    ///
    /// # Note:
    /// This fringe orders the nodes by upper bound (so the highest ub is going
    /// to pop first). So, it is guaranteed that the upper bound of the first
    /// node being popped is an upper bound on the value reachable by exploring
    /// any of the nodes remaining on the fringe. As a consequence, the
    /// exploration can be stopped as soon as a node with an ub <= current best
    /// lower bound is popped.
    fringe: F,
    /// This is the number of nodes that are currently being explored.
    ///
    /// # Note
    /// This information may seem innocuous/superfluous, whereas in fact it is
    /// very important. Indeed, this is the piece of information that lets us
    /// distinguish between a node-starvation and the completion of the problem
    /// resolution. The bottom line is, this counter needs to be carefully
    /// managed to guarantee the termination of all threads.
    ongoing  : usize,
    /// This is a counter that tracks the number of nodes that have effectively
    /// been explored. That is, the number of nodes that have been popped from
    /// the fringe, and for which a restricted and relaxed mdd have been developed.
    explored : usize,
    /// This is the value of the best known lower bound.
    best_lb  : isize,
    /// If set, this keeps the info about the best solution so far.
    best_sol : Option<Solution>,
    /// If we decide not to go through a complete proof of optimality, this is
    /// the reason why we took that decision.
    abort_proof: Option<Reason>,
    /// This is the value of the best upper bound.
    /// # WARNING
    /// This value is only set iff the abort proof is set.
    best_ub : Option<isize>,
    /// This vector is used to store the upper bound on the node which is
    /// currently processed by each thread.
    ///
    /// # Note
    /// When a thread is idle (or more generally when it is done with processing
    /// it node), it should place the value i32::min_value() in its corresponding
    /// cell.
    upper_bounds: Vec<isize>,
    ///
    new_best: bool,
    /// A marker to tell the compiler we use T
    _phantom: PhantomData<T>
}
/// The state which is shared among the many running threads: it provides an
/// access to the critical data (protected by a mutex) as well as a monitor
/// (condvar) to park threads in case of node-starvation.
struct Shared<T: Eq + Hash, F: Frontier<T>> {
    /// This is the shared state data which can only be accessed within critical
    /// sections. Therefore, it is protected by a mutex which prevents concurrent
    /// reads/writes.
    critical : Mutex<Critical<T, F>>,
    /// This is the monitor on which nodes must wait when facing an empty fringe.
    /// The corollary, it that whenever a node has completed the processing of
    /// a subproblem, it must wakeup all parked threads waiting on this monitor.
    monitor  : Condvar
}
/// The workload a thread can get from the shared state
enum WorkLoad<T> {
    /// There is no work left to be done: you can safely terminate
    Complete,
    /// There is nothing you can do right now. Check again when you wake up
    Starvation,
    /// The item to process
    WorkItem {
        must_log   : bool,
        explored   : usize,
        fringe_sz  : usize,
        best_lb    : isize,
        current_ub : isize,
        node       : FrontierNode<T>
    }
}

pub struct ParallelSolver<T, C, F, DD>
    where T : Send + Sync + Hash + Eq + Clone,
          C : Send + Sync + Clone + Config<T>,
          F : Send + Sync + Frontier<T>,
          DD: MDD<T, C> + From<C>
{
    /// This is the configuration which will be used to drive the behavior of
    /// the mdd-unrolling.
    config: C,
    /// This is an atomically-reference-counted smart pointer to the shared state.
    /// Again, each thread is going to take its own clone of this smart pointer.
    shared: Arc<Shared<T, F>>,
    /// This is a configuration parameter to tune the amount of information
    /// logged when solving the problem. So far, there are three levels of verbosity:
    ///
    ///   + *0* which prints nothing
    ///   + *1* which only prints the final statistics when the problem is solved
    ///   + *2* which prints progress information every 100 explored nodes.
    verbosity: u8,
    /// This is a configuration parameter that tunes the number of threads that
    /// will be spawned to solve the problem. By default, this number amounts
    /// to the number of hardware threads available on the machine.
    nb_threads: usize,
    /// This is just a marker that allows us to remember the exact type of the
    /// mdds to be instanciated.
    _phantom: PhantomData<DD>
}

impl <T, C, DD> ParallelSolver<T, C, SimpleFrontier<T>, DD>
    where T : Send + Sync + Hash + Eq + Clone,
          C : Send + Sync + Clone + Config<T>,
          DD: MDD<T, C> + From<C>{
    /// This creates a solver that will find the best solution in the problem
    /// described by the given `mdd` (mdd is not expanded yet). This solver will
    /// return the optimal solution from what would be an exact expansion of `mdd`.
    ///
    /// All the other parameters will use their default value.
    /// + `nb_threads` will be the number of hardware threads
    /// + `verbosity`  will be 0
    ///
    /// # Warning
    /// By passing the given MDD, you are effectively only passing the _type_
    /// of the mdd to use while developing the restricted and relaxed representations.
    /// Each thread will instantiate its own fresh copy using a clone of the
    /// configuration you provided. However, it will *not* clone your mdd. The
    /// latter will be dropped as soon as the constructor ends. However, clean
    /// copies are created when maximizing the objective function.
    pub fn new(mdd: DD) -> Self {
        Self::customized(mdd, 0, num_cpus::get())
    }
    /// This constructor lets you specify all the configuration parameters of
    /// the solver.
    ///
    /// # Return value
    /// This solver will return the optimal solution from what would be an exact
    /// expansion of `mdd`.
    ///
    /// # Verbosity
    /// So far, there are three levels of verbosity:
    ///
    ///   + *0* which prints nothing
    ///   + *1* which only prints the final statistics when the problem is solved
    ///   + *2* which prints progress information every 100 explored nodes.
    ///
    /// # Nb Threads
    /// The `nb_threads` argument lets you customize the number of threads to
    /// spawn in order to solve the problem. We advise you to use the number of
    /// hardware threads on your machine.
    ///
    /// # Warning
    /// By passing the given MDD, you are effectively only passing the _type_
    /// of the mdd to use while developing the restricted and relaxed representations.
    /// Each thread will instantiate its own fresh copy using a clone of the
    /// configuration you provided. However, it will *not* clone your mdd. The
    /// latter will be dropped as soon as the constructor ends. However, clean
    /// copies are created when maximizing the objective function.
    pub fn customized(mdd: DD, verbosity: u8, nb_threads: usize) -> Self {
        ParallelSolver {
            config: mdd.config().clone(),
            shared: Arc::new(Shared {
                monitor : Condvar::new(),
                critical: Mutex::new(Critical {
                    best_sol    : None,
                    best_lb     : isize::min_value(),
                    ongoing     : 0,
                    explored    : 0,
                    new_best    : false,
                    fringe      : SimpleFrontier::default(),
                    upper_bounds: vec![isize::min_value(); nb_threads],
                    abort_proof : None,
                    best_ub     : None,
                    _phantom    : PhantomData
                })
            }),
            verbosity,
            nb_threads,
            _phantom: PhantomData
        }
    }
}

/// private interface of the parallel solver
impl <T, C, F, DD> ParallelSolver<T, C, F, DD>
    where T : Send + Sync + Hash + Eq + Clone,
          C : Send + Sync + Clone + Config<T>,
          F : Send + Sync + Frontier<T>,
          DD: MDD<T, C> + From<C>
{
    /// Sets the verbosity of the solver
    pub fn with_verbosity(mut self, verbosity: u8) -> Self {
        self.verbosity = verbosity;
        self
    }
    /// Sets the number of threads used by the solver
    pub fn with_nb_threads(mut self, nb_threads: usize) -> Self {
        self.nb_threads = nb_threads;
        self
    }
    /// Sets the kind of frontier to use
    pub fn with_frontier<FF: Frontier<T> + Send + Sync>(self, ff: FF) -> ParallelSolver<T, C, FF, DD> {
        ParallelSolver {
            config: self.config,
            verbosity: self.verbosity,
            nb_threads: self.nb_threads,
            shared: Arc::new(Shared {
                monitor : Condvar::new(),
                critical: Mutex::new(Critical {
                    best_sol    : None,
                    best_lb     : isize::min_value(),
                    ongoing     : 0,
                    explored    : 0,
                    new_best    : false,
                    fringe      : ff,
                    abort_proof : None,
                    best_ub     : None,
                    upper_bounds: vec![isize::min_value(); self.nb_threads],
                    _phantom    : PhantomData
                })
            }),
            _phantom: PhantomData
        }
    }

    /// This method initializes the problem resolution. Put more simply, this
    /// method posts the root node of the mdd onto the fringe so that a thread
    /// can pick it up and the processing can be bootstrapped.
    fn initialize(&self) {
        let root = self.root_node();
        self.shared.critical.lock().fringe.push(root);
    }

    fn root_node(&self) -> FrontierNode<T> {
       self.config.root_node()
    }

    /// This method processes the given `node`. To do so, it reads the current
    /// best lower bound from the critical data. Then it expands a restricted
    /// and possibly a relaxed mdd rooted in `node`. If that is necessary,
    /// it stores cutset nodes onto the fringe for further parallel processing.
    fn process_one_node(mdd: &mut DD, shared: &Arc<Shared<T, F>>, node: FrontierNode<T>) -> Result<(), Reason> {
        // 1. RESTRICTION
        let (best_lb, best_ub) = Self::refresh_bounds(shared);
        let restriction = mdd.restricted(&node, best_lb, best_ub)?;
        Self::maybe_update_best(mdd, shared);
        if restriction.is_exact {
            return Ok(());
        }

        // 2. RELAXATION
        let (best_lb, best_ub) = Self::refresh_bounds(shared);
        let relaxation = mdd.relaxed(&node, best_lb, best_ub)?;
        if relaxation.is_exact {
            Self::maybe_update_best(mdd, shared);
        } else {
            Self::enqueue_cutset(mdd, shared, node.ub);
        }

        Ok(())
    }
    /// This private method updates the shared best known node and lower bound in
    /// case the best value of the current `mdd` expansion improves the current
    /// bounds.
    fn maybe_update_best(mdd: &DD, shared: &Arc<Shared<T, F>>) {
        let mut shared = shared.critical.lock();
        if mdd.best_value() > shared.best_lb {
            shared.best_lb   = mdd.best_value();
            shared.best_sol  = mdd.best_solution();
            shared.new_best  = true;
        }
    }
    /// If necessary, thightens the bound of nodes in the cutset of `mdd` and
    /// then add the relevant nodes to the shared fringe.
    fn enqueue_cutset(mdd: &mut DD, shared: &Arc<Shared<T, F>>, ub: isize) {
        let mut critical = shared.critical.lock();
        let best_lb      = critical.best_lb;
        let fringe       = &mut critical.fringe;
        mdd.for_each_cutset_node(|mut cutset_node| {
            cutset_node.ub = ub.min(cutset_node.ub);
            if cutset_node.ub > best_lb {
                fringe.push(cutset_node);
            }
        });
    }
    /// Acknowledges that a thread finished processing its node.
    fn notify_node_finished(shared: &Arc<Shared<T, F>>, thread_id: usize) {
        let mut critical = shared.critical.lock();
        critical.ongoing -= 1;
        critical.upper_bounds[thread_id] = isize::min_value();
        shared.monitor.notify_all();
    }
    /// Consults the shared state to fetch a workload. Depending on the current
    /// state, the workload can either be:
    ///
    ///   + Complete, when the problem is solved and all threads should stop
    ///   + Starvation, when there is no subproblem available for processing
    ///     at the time being (but some subproblem are still being processed
    ///     and thus the problem cannot be considered solved).
    ///   + WorkItem, when the thread successfully obtained a subproblem to
    ///     process.
    fn get_workload(shared: &Arc<Shared<T, F>>, thread_id: usize) -> WorkLoad<T> {
        let mut critical = shared.critical.lock();
        // Are we done ?
        if critical.ongoing == 0 && critical.fringe.is_empty() {
            return WorkLoad::Complete;
        }
        // Nothing to do yet ? => Wait for someone to post jobs
        if critical.fringe.is_empty() {
            shared.monitor.wait(&mut critical);
            return WorkLoad::Starvation;
        }
        // Nothing relevant ? =>  Wait for someone to post jobs
        let nn = critical.fringe.pop().unwrap();
        if nn.ub <= critical.best_lb {
            critical.fringe.clear();
            return WorkLoad::Starvation;
        }

        // Consume the current node and process it
        critical.ongoing += 1;
        critical.explored+= 1;
        critical.upper_bounds[thread_id] = nn.ub;

        let new_best = critical.new_best;
        critical.new_best = false;

        WorkLoad::WorkItem {
            must_log  : new_best,
            explored  : critical.explored,
            fringe_sz : critical.fringe.len(),
            best_lb   : critical.best_lb,
            current_ub: critical.upper_bounds.iter().cloned().max().unwrap(),
            node      : nn
        }
    }

    fn abort_search(shared: &Arc<Shared<T, F>>, reason: Reason, current_ub: isize) {
        let mut critical = shared.critical.lock();
        critical.abort_proof = Some(reason);
        match critical.best_ub {
            None     => critical.best_ub = Some(current_ub),
            Some(ub) => critical.best_ub = Some(ub.max(current_ub))
        };
        critical.fringe.clear();
    }

    fn refresh_bounds(shared: &Arc<Shared<T, F>>) -> (isize, isize) {
        let lock = shared.critical.lock();
        let lb   = lock.best_lb;
        let ub   = lock.upper_bounds.iter().copied().max().unwrap_or(isize::max_value());

        (lb, ub)
    }
    /// Depending on the verbosity configuration and the number of nodes that
    /// have been processed, prints a message showing the current progress of
    /// the problem resolution.
    fn maybe_log(verbosity: u8, must_log: bool, explored: usize, fringe_sz: usize, lb: isize, ub: isize) {
        if verbosity >= 2 && (must_log || explored % 100 == 0) {
            println!("Explored {}, LB {}, UB {}, Fringe sz {}", explored, lb, ub, fringe_sz);
        }
    }
}

impl <T, C, F, DD> Solver for ParallelSolver<T, C, F, DD>
    where T : Send + Sync + Hash + Eq + Clone,
          C : Send + Sync + Clone + Config<T>,
          F : Send + Sync + Frontier<T>,
          DD: MDD<T, C> + From<C>
{
    /// Applies the branch and bound algorithm proposed by Bergman et al. to
    /// solve the problem to optimality. To do so, it spawns `nb_threads` workers
    /// (long running threads); each of which will continually get a workload
    /// and process it until the problem is solved.
    fn maximize(&mut self) -> Completion {
        self.initialize();

        crossbeam::thread::scope(|s|{
            for i in 0..self.nb_threads {
                let shared    = Arc::clone(&self.shared);
                let conf      = self.config.clone();
                let verbosity = self.verbosity;

                s.spawn(move |_| {
                    let mut mdd = DD::from(conf);
                    loop {
                        match Self::get_workload(&shared, i) {
                            WorkLoad::Complete   => break,
                            WorkLoad::Starvation => continue,
                            WorkLoad::WorkItem {must_log, explored, fringe_sz, best_lb, current_ub, node} => {
                                Self::maybe_log(verbosity, must_log, explored, fringe_sz, best_lb, current_ub);
                                let outcome = Self::process_one_node(&mut mdd, &shared, node);
                                if let Err(reason) = outcome {
                                    Self::abort_search(&shared, reason, current_ub);
                                    Self::notify_node_finished(&shared, i);
                                    break;
                                } else {
                                    Self::notify_node_finished(&shared, i);
                                }
                            }
                        }
                    }
                });
            }
        }).expect("Something went wrong with the worker threads");

        let shared = self.shared.critical.lock();
        // return
        if self.verbosity >= 1 {
            println!("Final {}, Explored {}", shared.best_lb, shared.explored);
        }

        Completion {
            is_exact: shared.abort_proof.is_none(),
            best_value: shared.best_sol.as_ref().map(|_| shared.best_lb)
        }
    }


    /// Returns the best solution that has been identified for this problem.
    fn best_solution(&self) -> Option<Solution> {
        self.shared.critical.lock().best_sol.clone()
    }
    /// Returns the value of the best solution that has been identified for
    /// this problem.
    fn best_value(&self) -> Option<isize> {
        let critical = self.shared.critical.lock();
        critical.best_sol.as_ref().map(|_sol| critical.best_lb)
    }

    /// Returns the best lower bound that has been identified so far.
    /// In case where no solution has been found, it should return the minimum
    /// value that fits within an isize (-inf).
    fn best_lower_bound(&self) -> isize {
        let critical = self.shared.critical.lock();
        critical.best_lb
    }
    /// Returns the tightest upper bound that can be guaranteed so far.
    /// In case where no upper bound has been computed, it should return the
    /// maximum value that fits within an isize (+inf).
    fn best_upper_bound(&self) -> isize {
        let critical = self.shared.critical.lock();
        if let Some(ub) = critical.best_ub {
            ub
        } else {
            critical.best_sol.as_ref().map(|_sol| critical.best_lb).unwrap_or(isize::max_value())
        }
    }

    /// Sets the best known value and/or solution. This solution and value may
    /// be obtained from any other available means (LP relax for instance).
    fn set_primal(&mut self, value: isize, best_sol: Solution) {
        let mut critical = self.shared.critical.lock();
        if value > critical.best_lb {
            critical.best_lb = value;
            critical.best_sol= Some(best_sol);
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
    use crate::implementation::solver::parallel::ParallelSolver;
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
    fn by_default_best_lb_is_min_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd);
        assert_eq!(isize::min_value(), solver.best_lower_bound());
    }
    #[test]
    fn by_default_best_ub_is_plus_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd);
        assert_eq!(isize::max_value(), solver.best_upper_bound());
    }
    #[test]
    fn when_the_problem_is_solved_best_lb_is_best_value() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd        = mdd_builder(&problem, KPRelax).into_deep();
        let mut solver = ParallelSolver::new(mdd);
        let _ = solver.maximize();
        assert_eq!(220, solver.best_lower_bound());
    }
    #[test]
    fn when_the_problem_is_solved_best_ub_is_best_value() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd        = mdd_builder(&problem, KPRelax).into_deep();
        let mut solver = ParallelSolver::new(mdd);
        let _ = solver.maximize();
        assert_eq!(220, solver.best_upper_bound());
    }
    /* this, I can't test...
    #[test]
    fn when_the_solver_is_cutoff_ub_is_that_of_the_best_thread() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd        = mdd_builder(&problem, KPRelax)
            .with_cutoff(TimeBudget::new(Duration::from_secs(0_u64)))
            .into_deep();
        let mut solver = ParallelSolver::new(mdd);
        let _ = solver.maximize();
        assert_eq!(220, solver.best_upper_bound());
    }
    */

    #[test]
    fn by_default_verbosity_is_zero() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd);
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
        let solver = ParallelSolver::new(mdd).with_verbosity(2);
        assert_eq!(2, solver.verbosity);
    }

    #[test]
    fn by_default_it_uses_all_hw_threads() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd);
        assert_eq!(num_cpus::get(), solver.nb_threads);
    }
    #[test]
    fn num_threads_can_be_customized() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd).with_nb_threads(1);
        assert_eq!(1, solver.nb_threads);
    }
    #[test]
    fn num_threads_and_verbosity_can_be_customized() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::customized(mdd, 6, 9);
        assert_eq!(6, solver.verbosity);
        assert_eq!(9, solver.nb_threads);
    }

    #[test]
    fn no_solution_before_solving() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd);
        assert!(solver.shared.critical.lock().best_sol.is_none());
    }
    #[test]
    fn empty_fringe_before_solving() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd);
        assert!(solver.shared.critical.lock().fringe.is_empty());
    }
    #[test]
    fn default_best_lb_is_neg_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let mdd    = mdd_builder(&problem, KPRelax).into_deep();
        let solver = ParallelSolver::new(mdd);
        assert_eq!(isize::min_value(), solver.shared.critical.lock().best_lb);
    }
    #[test]
    fn maximizes_yields_the_optimum() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let        mdd = mdd_builder(&problem, KPRelax).into_deep();

        let mut solver = ParallelSolver::new(mdd);
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

        let mut solver = ParallelSolver::new(mdd);
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
        let mut solver = ParallelSolver::new(mdd);

        let d1  = Decision{variable: Variable(0), value: 10};
        let sol = Solution::new(Arc::new(PartialAssignment::SingleExtension {decision: d1, parent: Arc::new(PartialAssignment::Empty)}));

        solver.set_primal(10, sol.clone());
        assert!(solver.shared.critical.lock().best_sol.is_some());
        assert_eq!(10, solver.shared.critical.lock().best_lb);

        // in this case, it wont update because there is no improvement
        solver.set_primal(5, sol.clone());
        assert!(solver.shared.critical.lock().best_sol.is_some());
        assert_eq!(10, solver.shared.critical.lock().best_lb);

        // but here, it will update as it improves the best known sol
        solver.set_primal(10000, sol);
        assert!(solver.shared.critical.lock().best_sol.is_some());
        assert_eq!(10000, solver.shared.critical.lock().best_lb);

        // it wont do much as the primal is better than the actual feasible solution
        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(10000));
        assert!(solver.best_solution().is_some());
    }
}