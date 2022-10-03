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
use std::{marker::PhantomData, sync::Arc, hash::Hash};

use parking_lot::{Condvar, Mutex};

use crate::{Frontier, Decision, Problem, Relaxation, StateRanking, WidthHeuristic, Cutoff, SubProblem, DecisionDiagram, DefaultMDD, CompilationInput, CompilationType, Solver, Solution, Completion, Reason};

/// The shared data that may only be manipulated within critical sections
struct Critical<'a, F: Frontier> {
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
    fringe: &'a mut F,
    /// This is the number of nodes that are currently being explored.
    ///
    /// # Note
    /// This information may seem innocuous/superfluous, whereas in fact it is
    /// very important. Indeed, this is the piece of information that lets us
    /// distinguish between a node-starvation and the completion of the problem
    /// resolution. The bottom line is, this counter needs to be carefully
    /// managed to guarantee the termination of all threads.
    ongoing: usize,
    /// This is a counter that tracks the number of nodes that have effectively
    /// been explored. That is, the number of nodes that have been popped from
    /// the fringe, and for which a restricted and relaxed mdd have been developed.
    explored: usize,
    /// This is the value of the best known lower bound.
    best_lb: isize,
    /// This is the value of the best known lower bound.
    /// *WARNING* This one only gets set when the interrupt condition is satisfied
    best_ub: isize,
    /// If set, this keeps the info about the best solution so far.
    best_sol: Option<Vec<Decision>>,
    /// This vector is used to store the upper bound on the node which is
    /// currently processed by each thread.
    ///
    /// # Note
    /// When a thread is idle (or more generally when it is done with processing
    /// it node), it should place the value i32::min_value() in its corresponding
    /// cell.
    upper_bounds: Vec<isize>,
    /// If we decide not to go through a complete proof of optimality, this is
    /// the reason why we took that decision.
    abort_proof: Option<Reason>,
}
/// The state which is shared among the many running threads: it provides an
/// access to the critical data (protected by a mutex) as well as a monitor
/// (condvar) to park threads in case of node-starvation.
struct Shared<'a, P, R, O, W, C, F: Frontier>
where
    P: Problem + Send + Sync + 'a,
    R: Relaxation<State = P::State> + Send + Sync + 'a,
    O: StateRanking<State = P::State> + Send + Sync + 'a,
    W: WidthHeuristic<P::State> + Send + Sync + 'a,
    C: Cutoff + Send + Sync + 'a,
    F: Frontier<State = P::State> + Send + Sync + 'a,
{
    /// A reference to the problem being solved with branch-and-bound MDD
    problem: &'a P,
    /// The relaxation used when a DD layer grows too large
    relaxation: &'a R,
    /// The ranking heuristic used to discriminate the most promising from
    /// the least promising states
    ranking: &'a O,
    /// The maximum width heuristic used to enforce a given maximum memory
    /// usage when compiling mdds
    width_heu: &'a W,
    /// A cutoff heuristic meant to decide when to stop the resolution of 
    /// a given problem.
    cutoff: &'a C,

    /// This is the shared state data which can only be accessed within critical
    /// sections. Therefore, it is protected by a mutex which prevents concurrent
    /// reads/writes.
    critical: Mutex<Critical<'a, F>>,
    /// This is the monitor on which nodes must wait when facing an empty fringe.
    /// The corollary, it that whenever a node has completed the processing of
    /// a subproblem, it must wakeup all parked threads waiting on this monitor.
    monitor: Condvar,
}
/// The workload a thread can get from the shared state
enum WorkLoad<T> {
    /// There is no work left to be done: you can safely terminate
    Complete,
    /// The work must stop because of an external cutoff
    Aborted,
    /// There is nothing you can do right now. Check again when you wake up
    Starvation,
    /// The item to process
    WorkItem { node: SubProblem<T> },
}

pub struct ParallelSolver<'a, P, R, O, W, C, F, D>
where
    P: Problem + Send + Sync + 'a,
    R: Relaxation<State = P::State> + Send + Sync + 'a,
    O: StateRanking<State = P::State> + Send + Sync + 'a,
    W: WidthHeuristic<P::State> + Send + Sync + 'a,
    C: Cutoff + Send + Sync + 'a,
    F: Frontier<State = P::State> + Send + Sync + 'a,
    D: DecisionDiagram<State = P::State> + Default,
{
    /// This is the shared state. Each thread is going to take a reference to it.
    shared: Shared<'a, P, R, O, W, C, F>,
    /// This is a configuration parameter that tunes the number of threads that
    /// will be spawned to solve the problem. By default, this number amounts
    /// to the number of hardware threads available on the machine.
    nb_threads: usize,
    /// This is just a marker that allows us to remember the exact type of the
    /// mdds to be instanciated.
    _phantom: PhantomData<D>,
}

// private interface.
impl <'a, P, R, O, W, C, F> ParallelSolver<'a, P, R, O, W, C, F, DefaultMDD<P::State>> 
where 
    P: Problem + Send + Sync + 'a,
    R: Relaxation<State = P::State> + Send + Sync + 'a,
    O: StateRanking<State = P::State> + Send + Sync + 'a,
    W: WidthHeuristic<P::State> + Send + Sync + 'a,
    C: Cutoff + Send + Sync + 'a,
    F: Frontier<State = P::State> + Send + Sync + 'a,
    P::State: Eq + Hash + Clone
{
    pub fn new(
        problem: &'a P,
        relaxation: &'a R,
        ranking: &'a O,
        width: &'a W,
        cutoff: &'a C, 
        fringe: &'a mut F,
    ) -> Self {
        Self::custom(problem, relaxation, ranking, width, cutoff, fringe, num_cpus::get())
    }
}


impl<'a, P, R, O, W, C, F, D> ParallelSolver<'a, P, R, O, W, C, F, D>
where
    P: Problem + Send + Sync + 'a,
    R: Relaxation<State = P::State> + Send + Sync + 'a,
    O: StateRanking<State = P::State> + Send + Sync + 'a,
    W: WidthHeuristic<P::State> + Send + Sync + 'a,
    C: Cutoff + Send + Sync + 'a,
    F: Frontier<State = P::State> + Send + Sync + 'a,
    D: DecisionDiagram<State = P::State> + Default,
{
    pub fn custom(
        problem: &'a P,
        relaxation: &'a R,
        ranking: &'a O,
        width_heu: &'a W,
        cutoff: &'a C,
        fringe: &'a mut F,
        nb_threads: usize,
    ) -> Self {
        ParallelSolver {
            shared: Shared {
                problem,
                relaxation,
                ranking,
                width_heu,
                cutoff,
                //
                monitor: Condvar::new(),
                critical: Mutex::new(Critical {
                    best_sol: None,
                    best_lb: isize::MIN,
                    best_ub: isize::MAX,
                    upper_bounds: vec![isize::MAX; nb_threads],
                    fringe,
                    ongoing: 0,
                    explored: 0,
                    abort_proof: None,
                }),
            },
            nb_threads,
            _phantom: Default::default(),
        }
    }
    /// Sets the number of threads used by the solver
    pub fn with_nb_threads(mut self, nb_threads: usize) -> Self {
        self.nb_threads = nb_threads;
        self
    }

    /// This method initializes the problem resolution. Put more simply, this
    /// method posts the root node of the mdd onto the fringe so that a thread
    /// can pick it up and the processing can be bootstrapped.
    fn initialize(&self) {
        let root = self.root_node();
        self.shared.critical.lock().fringe.push(root);
    }

    fn root_node(&self) -> SubProblem<P::State> {
        let shared = &self.shared;
        SubProblem {
            state: Arc::new(shared.problem.initial_state()),
            value: shared.problem.initial_value(),
            path: vec![],
            ub: isize::MAX,
        }
    }

    /// This method processes the given `node`. To do so, it reads the current
    /// best lower bound from the critical data. Then it expands a restricted
    /// and possibly a relaxed mdd rooted in `node`. If that is necessary,
    /// it stores cutset nodes onto the fringe for further parallel processing.
    fn process_one_node(
        mdd: &mut D,
        shared: &Shared<P, R, O, W, C, F>,
        node: SubProblem<P::State>,
    ) -> Result<(), Reason> {
        // 1. RESTRICTION
        let node_ub = node.ub;
        let best_lb = Self::best_lb(shared);

        if node_ub <= best_lb {
            return Ok(());
        }

        let width = shared.width_heu.max_width(&node);
        let mut compilation = CompilationInput {
            comp_type: CompilationType::Restricted,
            max_width: width,
            problem: shared.problem,
            relaxation: shared.relaxation,
            ranking: shared.ranking,
            cutoff: shared.cutoff,
            residual: node,
            //
            best_lb,
        };

        let Completion{is_exact, ..} = mdd.compile(&compilation)?;
        Self::maybe_update_best(mdd, shared);
        if is_exact {
            return Ok(());
        }

        // 2. RELAXATION
        let best_lb = Self::best_lb(shared);
        compilation.comp_type = CompilationType::Relaxed;
        compilation.best_lb = best_lb;

        let Completion{is_exact, ..} = mdd.compile(&compilation)?;
        if is_exact {
            Self::maybe_update_best(mdd, shared);
        } else {
            Self::enqueue_cutset(mdd, shared, node_ub);
        }

        Ok(())
    }

    fn best_lb(shared: &Shared<P, R, O, W, C, F>) -> isize {
        shared.critical.lock().best_lb
    }

    /// This private method updates the shared best known node and lower bound in
    /// case the best value of the current `mdd` expansion improves the current
    /// bounds.
    fn maybe_update_best(mdd: &D, shared: &Shared<P, R, O, W, C, F>) {
        let mut shared = shared.critical.lock();
        let dd_best_value = mdd.best_value().unwrap_or(isize::MIN);
        if dd_best_value > shared.best_lb {
            shared.best_lb = dd_best_value;
            shared.best_sol = mdd.best_solution();
        }
    }
    /// If necessary, thightens the bound of nodes in the cutset of `mdd` and
    /// then add the relevant nodes to the shared fringe.
    fn enqueue_cutset(mdd: &mut D, shared: &Shared<P, R, O, W, C, F>, ub: isize) {
        let mut critical = shared.critical.lock();
        let best_lb = critical.best_lb;
        let fringe = &mut critical.fringe;
        mdd.drain_cutset(|mut cutset_node| {
            cutset_node.ub = ub.min(cutset_node.ub);
            if cutset_node.ub > best_lb {
                fringe.push(cutset_node);
            }
        });
    }
    /// Acknowledges that a thread finished processing its node.
    fn notify_node_finished(shared: &Shared<P, R, O, W, C, F>, thread_id: usize) {
        let mut critical = shared.critical.lock();
        critical.ongoing -= 1;
        critical.upper_bounds[thread_id] = isize::MAX;
        shared.monitor.notify_all();
    }

    fn abort_search(shared: &Shared<P, R, O, W, C, F>, reason: Reason, current_ub: isize) {
        let mut critical = shared.critical.lock();
        critical.abort_proof = Some(reason);
        if critical.best_ub == isize::MAX {
            critical.best_ub = current_ub;
        } else {
            critical.best_ub = current_ub.max(critical.best_ub);
        }
        critical.fringe.clear();
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
    fn get_workload(shared: &Shared<P, R, O, W, C, F>, thread_id: usize) -> WorkLoad<P::State>
    {
        let mut critical = shared.critical.lock();

        // Are we done ?
        if critical.ongoing == 0 && critical.fringe.is_empty() {
            critical.best_ub = critical.best_lb;
            return WorkLoad::Complete;
        }

        // Do we need to stop
        if critical.abort_proof.is_some() {
            return WorkLoad::Aborted;
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
        critical.explored += 1;
        critical.upper_bounds[thread_id] = nn.ub;

        WorkLoad::WorkItem { node: nn }
    }
}

impl<'a, P, R, O, W, C, F, D> Solver for ParallelSolver<'a, P, R, O, W, C, F, D>
where
    P: Problem + Send + Sync + 'a,
    R: Relaxation<State = P::State> + Send + Sync + 'a,
    O: StateRanking<State = P::State> + Send + Sync + 'a,
    W: WidthHeuristic<P::State> + Send + Sync + 'a,
    C: Cutoff + Send + Sync + 'a,
    F: Frontier<State = P::State> + Send + Sync + 'a,
    D: DecisionDiagram<State = P::State> + Default,
{
    /// Applies the branch and bound algorithm proposed by Bergman et al. to
    /// solve the problem to optimality. To do so, it spawns `nb_threads` workers
    /// (long running threads); each of which will continually get a workload
    /// and process it until the problem is solved.
    fn maximize(&mut self) -> Completion {
        self.initialize();

        std::thread::scope(|s| {
            for i in 0..self.nb_threads {
                let shared = &self.shared;
                s.spawn(move || {
                    let mut mdd = D::default();
                    loop {
                        match Self::get_workload(shared, i) {
                            WorkLoad::Complete => break,
                            WorkLoad::Aborted => break, // this one cannot occur
                            WorkLoad::Starvation => continue,
                            WorkLoad::WorkItem { node } => {
                                let ub = node.ub;
                                let outcome = Self::process_one_node(&mut mdd, shared, node);
                                if let Err(reason) = outcome {
                                    Self::abort_search(&shared, reason, ub);
                                    Self::notify_node_finished(shared, i); 
                                    break;
                                } else {
                                    Self::notify_node_finished(shared, i);
                                }
                            }
                        }
                    }
                });
            }
        });

        let critical = self.shared.critical.lock();
        Completion { is_exact: critical.abort_proof.is_none(), best_value: critical.best_sol.as_ref().map(|_| critical.best_lb) }
    }

    /// Returns the best solution that has been identified for this problem.
    fn best_solution(&self) -> Option<Vec<Decision>> {
        self.shared.critical.lock().best_sol.clone()
    }
    /// Returns the value of the best solution that has been identified for
    /// this problem.
    fn best_value(&self) -> Option<isize> {
        let critical = self.shared.critical.lock();
        critical.best_sol.as_ref().map(|_sol| critical.best_lb)
    }
    /// Returns the value of the best lower bound that has been identified for
    /// this problem.
    fn best_lower_bound(&self) -> isize {
        self.shared.critical.lock().best_lb
    }
    /// Returns the value of the best upper bound that has been identified for
    /// this problem.
    fn best_upper_bound(&self) -> isize {
        self.shared.critical.lock().best_ub
    }
    /// Sets a primal (best known value and solution) of the problem.
    fn set_primal(&mut self, value: isize, solution: Solution) {
        let mut critical = self.shared.critical.lock();
        if value > critical.best_lb {
            critical.best_sol = Some(solution);
            critical.best_lb  = value;
        }
    }
}
