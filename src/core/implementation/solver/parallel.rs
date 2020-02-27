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
use binary_heap_plus::BinaryHeap;

use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::common::{Decision, Node, NodeInfo};
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use crate::core::implementation::heuristics::MaxUB;

/// The shared data that may only be manipulated within critical sections
struct Critical<T> {
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
    fringe   : BinaryHeap<Node<T>, MaxUB>,
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
    best_lb  : i32,
    /// If set, this keeps the info about the best solution (the solution that
    /// yielded the `best_lb`, and from which `best_sol` derives).
    best_node: Option<NodeInfo>
}
/// The state which is shared among the many running threads: it provides an
/// access to the critical data (protected by a mutex) as well as a monitor
/// (condvar) to park threads in case of node-starvation.
struct Shared<T> {
    /// This is the shared state data which can only be accessed within critical
    /// sections. Therefore, it is protected by a mutex which prevents concurrent
    /// reads/writes.
    critical : Mutex<Critical<T>>,
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
        explored : usize,
        fringe_sz: usize,
        best_lb  : i32,
        node     : Node<T>
    }
}

/// This is the structure implementing a multi-threaded MDD solver.
///
/// # Example Usage
/// ```
/// # use ddo::core::implementation::mdd::builder::mdd_builder;
/// # use ddo::core::implementation::heuristics::FixedWidth;
/// # use ddo::core::abstraction::dp::{Problem, Relaxation};
/// # use ddo::core::common::{Variable, Domain, VarSet, Decision, Node};
/// # use ddo::core::implementation::solver::parallel::ParallelSolver;
/// # use ddo::core::abstraction::solver::Solver;
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> i32   { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         unimplemented!()
/// #     }
/// #     fn transition(&self, _: &usize, _: &VarSet, _: Decision) -> usize {
/// #         unimplemented!()
/// #     }
/// #     fn transition_cost(&self, _: &usize, _: &VarSet, _: Decision) -> i32 {
/// #         unimplemented!()
/// #     }
/// # }
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #     fn merge_nodes(&self, _: &[Node<usize>]) -> Node<usize> {
/// #         unimplemented!()
/// #     }
/// # }
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let mdd        = mdd_builder(&problem, relaxation).build();
/// // the solver is created using an mdd. By default, it uses as many threads
/// // as there are hardware threads on the machine.
/// let mut solver = ParallelSolver::new(mdd);
/// // val is the optimal value of the objective function,
/// // sol is the sequence of decision yielding that optimal value (if sol exists, `sol != None`)
/// let (val, sol) = solver.maximize();
/// ```
pub struct ParallelSolver<T, DD> where T: Send, DD: MDD<T> + Clone + Send {
    /// This is the MDD which will be used to expand the restricted and relaxed
    /// mdds. Technically, all threads are going to take their own copy (clone)
    /// of this mdd. Thus, this instance will only serve as a prototype to
    /// instantiate the threads.
    mdd: DD,
    /// This is an atomically-reference-counted smart pointer to the shared state.
    /// Again, each thread is going to take its own clone of this smart pointer.
    shared: Arc<Shared<T>>,
    /// This is the materialization of the best solution that has been identified.
    best_sol: Option<Vec<Decision>>,
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
    nb_threads: usize
}

/// private interface of the parallel solver
impl <T, DD> ParallelSolver<T, DD> where T: Send, DD: MDD<T> + Clone + Send {
    /// This creates a solver that will find the best solution in the problem
    /// described by the given `mdd` (mdd is not expanded yet). This solver will
    /// return the optimal solution from what would be an exact expansion of `mdd`.
    ///
    /// All the other parameters will use their default value.
    /// + `nb_threads` will be the number of hardware threads
    /// + `verbosity`  will be 0
    pub fn new(mdd: DD) -> Self {
        Self::customized(mdd, 0, num_cpus::get())
    }
    /// This creates a solver that will find the best solution in the problem
    /// described by the given `mdd` (mdd is not expanded yet) and configure that
    /// solver to be more or less verbose.
    ///
    /// When using this constructor, the default number of threads will amount
    /// to the number of available hardware threads of the platform.
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
    pub fn with_verbosity(mdd: DD, verbosity: u8) -> Self {
        Self::customized(mdd, verbosity, num_cpus::get())
    }
    /// This creates a solver that will find the best solution in the problem
    /// described by the given `mdd` (mdd is not expanded yet) using `nb_threads`.
    /// This solver will return the optimal solution from what would be an exact
    /// expansion of `mdd`.
    ///
    /// When using this constructor, the verbosity of the solver is set to level 0.
    pub fn with_nb_threads(mdd: DD, nb_threads: usize) -> Self {
        Self::customized(mdd, 0, nb_threads)
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
    pub fn customized(mdd: DD, verbosity: u8, nb_threads: usize) -> Self {
        ParallelSolver {
            mdd,
            shared: Arc::new(Shared {
                monitor : Condvar::new(),
                critical: Mutex::new(Critical {
                    best_node: None,
                    best_lb  : i32::min_value(),
                    ongoing  : 0,
                    explored : 0,
                    fringe   : BinaryHeap::from_vec_cmp(vec![], MaxUB)
                })
            }),
            best_sol: None,
            verbosity,
            nb_threads
        }
    }

    /// This method initializes the problem resolution. Put more simply, this
    /// method posts the root node of the mdd onto the fringe so that a thread
    /// can pick it up and the processing can be bootstrapped.
    fn initialize(&self) {
        let root = self.mdd.root();
        self.shared.critical.lock().fringe.push(root);
    }

    /// This method processes the given `node`. To do so, it reads the current
    /// best lower bound from the critical data. Then it expands a restricted
    /// and possibly a relaxed mdd rooted in `node`. If that is necessary,
    /// it stores cutset nodes onto the fringe for further parallel processing.
    fn process_one_node(mdd: &mut DD, shared: &Arc<Shared<T>>, node: Node<T>) {
        let mut best_lb = {shared.critical.lock().best_lb};

        // 1. RESTRICTION
        mdd.restricted(&node, best_lb);
        Self::maybe_update_best(mdd, shared);
        if mdd.is_exact() {
            Self::notify_node_finished(shared);
            return;
        }

        // 2. RELAXATION
        best_lb = {shared.critical.lock().best_lb};
        mdd.relaxed(&node, best_lb);
        if mdd.is_exact() {
            Self::maybe_update_best(mdd, shared);
        } else {
            Self::enqueue_cutset(mdd, shared, node.info.ub);
        }

        Self::notify_node_finished(shared);
    }
    /// This private method updates the shared best known node and lower bound in
    /// case the best value of the current `mdd` expansion improves the current
    /// bounds.
    fn maybe_update_best(mdd: &DD, shared: &Arc<Shared<T>>) {
        let mut shared = shared.critical.lock();
        if mdd.best_value() > shared.best_lb {
            shared.best_lb   = mdd.best_value();
            shared.best_node = mdd.best_node().clone();
        }
    }
    /// If necessary, thightens the bound of nodes in the cutset of `mdd` and
    /// then add the relevant nodes to the shared fringe.
    fn enqueue_cutset(mdd: &mut DD, shared: &Arc<Shared<T>>, ub: i32) {
        let mut critical = shared.critical.lock();
        let best_lb      = critical.best_lb;
        let fringe       = &mut critical.fringe;
        mdd.consume_cutset(|state, mut info| {
            info.ub = ub.min(info.ub);
            if info.ub > best_lb {
                fringe.push(Node { state, info });
            }
        });
    }
    /// Acknowledges that a thread finished processing its node.
    fn notify_node_finished(shared: &Arc<Shared<T>>) {
        shared.critical.lock().ongoing -= 1;
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
    fn get_workload(shared: &Arc<Shared<T>>) -> WorkLoad<T> {
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
        if nn.info.ub < critical.best_lb {
            critical.fringe.clear();
            return WorkLoad::Starvation;
        }

        // Consume the current node and process it
        critical.ongoing += 1;
        critical.explored+= 1;

        WorkLoad::WorkItem {
            explored : critical.explored,
            fringe_sz: critical.fringe.len(),
            best_lb  : critical.best_lb,
            node     : nn
        }
    }
    /// Depending on the verbosity configuration and the number of nodes that
    /// have been processed, prints a message showing the current progress of
    /// the problem resolution.
    fn maybe_log(verbosity: u8, explored: usize, fringe_sz: usize, lb: i32) {
        if verbosity >= 2 && explored % 100 == 0 {
            println!("Explored {}, LB {}, Fringe sz {}", explored, lb, fringe_sz);
        }
    }
}

impl <T, DD> Solver for ParallelSolver<T, DD> where T: Send, DD: MDD<T> + Clone + Send {
    /// Applies the branch and bound algorithm proposed by Bergman et al. to
    /// solve the problem to optimality. To do so, it spawns `nb_threads` workers
    /// (long running threads); each of which will continually get a workload
    /// and process it until the problem is solved.
    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        self.initialize();

        crossbeam::thread::scope(|s|{
            for _i in 0..self.nb_threads {
                let shared    = Arc::clone(&self.shared);
                let mut mdd   = self.mdd.clone();
                let verbosity = self.verbosity;

                s.spawn(move |_| {
                    loop {
                        match Self::get_workload(&shared) {
                            WorkLoad::Complete   => break,
                            WorkLoad::Starvation => continue,
                            WorkLoad::WorkItem {explored, fringe_sz, best_lb, node} => {
                                Self::maybe_log(verbosity, explored, fringe_sz, best_lb);
                                Self::process_one_node(&mut mdd, &shared, node);
                            }
                        }
                    }
                });
            }
        }).expect("Something went wrong with the worker threads");

        let shared = self.shared.critical.lock();
        if let Some(bn) = &shared.best_node {
            self.best_sol = Some(bn.longest_path());
        }

        // return
        if self.verbosity >= 1 {
            println!("Final {}, Explored {}", shared.best_lb, shared.explored);
        }
        (shared.best_lb, &self.best_sol)
    }
}