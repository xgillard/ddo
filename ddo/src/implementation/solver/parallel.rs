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

use crate::{Fringe, Decision, Problem, Relaxation, StateRanking, WidthHeuristic, Cutoff, SubProblem, DecisionDiagram, CompilationInput, CompilationType, Solver, Solution, Completion, Reason, Cache, DominanceChecker};

/// The shared data that may only be manipulated within critical sections
struct Critical<'a, State> {
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
    fringe: &'a mut (dyn Fringe<State> + Send + Sync),
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
    /// This is a counter of the number of nodes in the fringe, for each level of the model
    open_by_layer: Vec<usize>,
    /// This is a counter of the number of nodes in ongoing expansion, for each level of the model
    ongoing_by_layer: Vec<usize>,
    /// This is the index of the first level above which there are no nodes in the fringe
    first_active_layer: usize,
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
struct Shared<'a, State, C> where
    C : Cache<State> + Send + Sync + Default,
{
    /// A reference to the problem being solved with branch-and-bound MDD
    problem: &'a (dyn Problem<State = State> + Send + Sync),
    /// The relaxation used when a DD layer grows too large
    relaxation: &'a (dyn Relaxation<State = State> + Send + Sync),
    /// The ranking heuristic used to discriminate the most promising from
    /// the least promising states
    ranking: &'a (dyn StateRanking<State = State> + Send + Sync),
    /// The maximum width heuristic used to enforce a given maximum memory
    /// usage when compiling mdds
    width_heu: &'a (dyn WidthHeuristic<State> + Send + Sync),
    /// A cutoff heuristic meant to decide when to stop the resolution of 
    /// a given problem.
    cutoff: &'a (dyn Cutoff + Send + Sync),

    /// Data structure containing info about past compilations used to prune the search
    cache: C,
    dominance: &'a (dyn DominanceChecker<State> + Send + Sync),

    /// This is the shared state data which can only be accessed within critical
    /// sections. Therefore, it is protected by a mutex which prevents concurrent
    /// reads/writes.
    critical: Mutex<Critical<'a, State>>,
    /// This is the monitor on which nodes must wait when facing an empty fringe.
    /// The corollary, it that whenever a node has completed the processing of
    /// a subproblem, it must wake-up all parked threads waiting on this monitor.
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


/// This is the structure implementing a multi-threaded MDD solver.
///
/// # Example Usage
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// # #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// # pub struct KnapsackState {
/// #     depth: usize,
/// #     capacity: usize
/// # }
/// # 
/// # struct Knapsack {
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// # }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// # impl Problem for Knapsack {
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
/// #     fn transition_cost(&self, _state: &Self::State, _next: &Self::State, dec: Decision) -> isize {
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
/// # }
/// # struct KPRelax<'a>{pb: &'a Knapsack}
/// # impl Relaxation for KPRelax<'_> {
/// #     type State = KnapsackState;
/// # 
/// #     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
/// #         states.max_by_key(|node| node.capacity).copied().unwrap()
/// #     }
/// #     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
/// #         cost
/// #     }
/// # }
/// # 
/// # struct KPRanking;
/// # impl StateRanking for KPRanking {
/// #     type State = KnapsackState;
/// #     
/// #     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
/// #         a.capacity.cmp(&b.capacity)
/// #     }
/// # }
/// # pub struct KPDominance;
/// # impl Dominance for KPDominance {
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
/// # }
/// 
/// // To create a new solver, you need to be able to provide it with a problem instance, a relaxation
/// // and the various required heuristic. This example assumes the existence of the Knapsack structure
/// // and relaxation.
/// 
/// // 1. Create an instance of our knapsack problem
/// let problem = Knapsack {
///     capacity: 50,
///     profit  : vec![60, 100, 120],
///     weight  : vec![10,  20,  30]
/// };
/// 
/// // 2. Create a relaxation of the problem
/// let relaxation = KPRelax{pb: &problem};
/// 
/// // 3. Create a ranking to discriminate the promising and uninteresting states
/// let heuristic = KPRanking;
/// 
/// // 4. Define the policy you will want to use regarding the maximum width of the DD
/// let width = FixedWidth(100); // here we mean max 100 nodes per layer
/// 
/// // 5. Add a dominance relation checker
/// let dominance = SimpleDominanceChecker::new(KPDominance, problem.nb_variables());
/// 
/// // 6. Decide of a cutoff heuristic (if you don't want to let the solver run for ever)
/// let cutoff = NoCutoff; // might as well be a TimeBudget (or something else)
/// 
/// // 7. Create the solver fringe
/// let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
///  
/// // 8. Instantiate your solver
/// let mut solver = DefaultSolver::new(
///       &problem, 
///       &relaxation, 
///       &heuristic, 
///       &width, 
///       &dominance,
///       &cutoff, 
///       &mut fringe);
/// 
/// // 9. Maximize your objective function
/// // the outcome provides the value of the best solution that was found for
/// // the problem (if one was found) along with a flag indicating whether or
/// // not the solution was proven optimal. Hence an unsatisfiable problem
/// // would have `outcome.best_value == None` and `outcome.is_exact` true.
/// // The `is_exact` flag will only be false if you explicitly decide to stop
/// // searching with an arbitrary cutoff.
/// let outcome = solver.maximize();
/// // The best solution (if one exist) is retrieved with
/// let solution = solver.best_solution();
///
/// // 10. Do whatever you like with the optimal solution.
/// assert_eq!(Some(220), outcome.best_value);
/// println!("Solution");
/// for decision in solution.unwrap().iter() {
///     if decision.value == 1 {
///         println!("{}", decision.variable.id());
///     }
/// }
/// ```
pub struct ParallelSolver<'a, State, D, C> 
where D: DecisionDiagram<State = State> + Default,
      C: Cache<State> + Send + Sync + Default,
{
    /// This is the shared state. Each thread is going to take a reference to it.
    shared: Shared<'a, State, C>,
    /// This is a configuration parameter that tunes the number of threads that
    /// will be spawned to solve the problem. By default, this number amounts
    /// to the number of hardware threads available on the machine.
    nb_threads: usize,
    /// This is just a marker that allows us to remember the exact type of the
    /// mdds to be instantiated.
    _phantom: PhantomData<D>, 
}

impl<'a, State, D, C>  ParallelSolver<'a, State, D, C>
where 
    State: Eq + Hash + Clone,
    D: DecisionDiagram<State = State> + Default,
    C: Cache<State> + Send + Sync + Default,
{
    pub fn new(
        problem: &'a (dyn Problem<State = State> + Send + Sync),
        relaxation: &'a (dyn Relaxation<State = State> + Send + Sync),
        ranking: &'a (dyn StateRanking<State = State> + Send + Sync),
        width: &'a (dyn WidthHeuristic<State> + Send + Sync),
        dominance: &'a (dyn DominanceChecker<State> + Send + Sync),
        cutoff: &'a (dyn Cutoff + Send + Sync), 
        fringe: &'a mut (dyn Fringe<State> + Send + Sync),
    ) -> Self {
        Self::custom(problem, relaxation, ranking, width, dominance, cutoff, fringe, num_cpus::get())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn custom(
        problem: &'a (dyn Problem<State = State> + Send + Sync),
        relaxation: &'a (dyn Relaxation<State = State> + Send + Sync),
        ranking: &'a (dyn StateRanking<State = State> + Send + Sync),
        width_heu: &'a (dyn WidthHeuristic<State> + Send + Sync),
        dominance: &'a (dyn DominanceChecker<State> + Send + Sync),
        cutoff: &'a (dyn Cutoff + Send + Sync),
        fringe: &'a mut (dyn Fringe<State> + Send + Sync),
        nb_threads: usize,
    ) -> Self {
        ParallelSolver {
            shared: Shared {
                problem,
                relaxation,
                ranking,
                width_heu,
                cutoff,
                cache: C::default(),
                dominance,
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
                    open_by_layer: vec![0; problem.nb_variables() + 1],
                    ongoing_by_layer: vec![0; problem.nb_variables() + 1],
                    first_active_layer: 0,
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
    fn initialize(&mut self) {
        let root = self.root_node();
        self.shared.cache.initialize(self.shared.problem);
        let mut critical = self.shared.critical.lock();
        critical.fringe.push(root);
        critical.open_by_layer[0] += 1;
    }

    fn root_node(&self) -> SubProblem<State> {
        let shared = &self.shared;
        SubProblem {
            state: Arc::new(shared.problem.initial_state()),
            value: shared.problem.initial_value(),
            path: vec![],
            ub: isize::MAX,
            depth: 0,
        }
    }

    /// This method processes the given `node`. To do so, it reads the current
    /// best lower bound from the critical data. Then it expands a restricted
    /// and possibly a relaxed mdd rooted in `node`. If that is necessary,
    /// it stores cut-set nodes onto the fringe for further parallel processing.
    fn process_one_node(
        mdd: &mut D,
        shared: &Shared<'a, State, C>,
        node: SubProblem<State>,
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
            residual: &node,
            //
            best_lb,
            cache: &shared.cache,
            dominance: shared.dominance,
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
        Self::maybe_update_best(mdd, shared);
        if !is_exact {
            Self::enqueue_cutset(mdd, shared, node_ub);
        }

        Ok(())
    }

    fn best_lb(shared: &Shared<'a, State, C>) -> isize {
        shared.critical.lock().best_lb
    }

    /// This private method updates the shared best known node and lower bound in
    /// case the best value of the current `mdd` expansion improves the current
    /// bounds.
    fn maybe_update_best(mdd: &D, shared: &Shared<'a, State, C>) {
        let mut shared = shared.critical.lock();
        let dd_best_value = mdd.best_exact_value().unwrap_or(isize::MIN);
        if dd_best_value > shared.best_lb {
            shared.best_lb = dd_best_value;
            shared.best_sol = mdd.best_exact_solution();
        }
    }
    /// If necessary, tightens the bound of nodes in the cut-set of `mdd` and
    /// then add the relevant nodes to the shared fringe.
    fn enqueue_cutset(mdd: &mut D, shared: &Shared<'a, State, C>, ub: isize) {
        let mut critical = shared.critical.lock();
        let best_lb = critical.best_lb;
        mdd.drain_cutset(|mut cutset_node| {
            cutset_node.ub = ub.min(cutset_node.ub);
            if cutset_node.ub > best_lb {
                let depth = cutset_node.depth;
                let before = critical.fringe.len();
                critical.fringe.push(cutset_node);
                let after = critical.fringe.len();
                critical.open_by_layer[depth] += after - before;
            }
        });
    }
    /// Acknowledges that a thread finished processing its node.
    fn notify_node_finished(shared: &Shared<'a, State, C>, thread_id: usize, depth: usize) {
        let mut critical = shared.critical.lock();
        critical.ongoing -= 1;
        critical.upper_bounds[thread_id] = isize::MAX;
        critical.ongoing_by_layer[depth] -= 1;
        shared.monitor.notify_all();
    }

    fn abort_search(shared: &Shared<'a, State, C>, reason: Reason, current_ub: isize) {
        let mut critical = shared.critical.lock();
        critical.abort_proof = Some(reason);
        if critical.best_ub == isize::MAX {
            critical.best_ub = current_ub;
        } else {
            critical.best_ub = current_ub.max(critical.best_ub);
        }
        critical.fringe.clear();
        shared.cache.clear();
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
    fn get_workload(shared: &Shared<'a, State, C>, thread_id: usize) -> WorkLoad<State>
    {
        let mut critical = shared.critical.lock();

        // Can we clean up the cache?
        while critical.first_active_layer < shared.problem.nb_variables() &&
                critical.open_by_layer[critical.first_active_layer] + critical.ongoing_by_layer[critical.first_active_layer] == 0 {
            shared.cache.clear_layer(critical.first_active_layer);
            critical.first_active_layer += 1;
        }

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

        let mut nn = critical.fringe.pop().unwrap();
        loop {
            // Nothing relevant ? =>  Wait for someone to post jobs
            if nn.ub <= critical.best_lb {
                critical.fringe.clear();
                critical.open_by_layer.iter_mut().for_each(|o| *o = 0);
                return WorkLoad::Starvation;
            }
            
            if shared.cache.must_explore(&nn) {
                shared.cache.update_threshold(nn.state.clone(), nn.depth, nn.value, true);
                break;
            } else {
                critical.open_by_layer[nn.depth] -= 1;

                if critical.fringe.is_empty() {
                    return WorkLoad::Starvation;
                }
    
                nn = critical.fringe.pop().unwrap();
            }
        }

        // Consume the current node and process it
        critical.ongoing += 1;
        critical.explored += 1;
        critical.upper_bounds[thread_id] = nn.ub;
        critical.open_by_layer[nn.depth] -= 1;
        critical.ongoing_by_layer[nn.depth] += 1;

        WorkLoad::WorkItem { node: nn }
    }

}

impl<'a, State, D, C> Solver for ParallelSolver<'a, State, D, C>
where
    State: Eq + PartialEq + Hash + Clone,
    D: DecisionDiagram<State = State> + Default,
    C: Cache<State> + Send + Sync + Default,
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
                                let depth = node.depth;
                                let outcome = Self::process_one_node(&mut mdd, shared, node);
                                if let Err(reason) = outcome {
                                    Self::abort_search(shared, reason, ub);
                                    Self::notify_node_finished(shared, i, depth); 
                                    break;
                                } else {
                                    Self::notify_node_finished(shared, i, depth);
                                }
                            }
                        }
                    }
                });
            }
        });

        let mut critical = self.shared.critical.lock();
        if let Some(sol) = critical.best_sol.as_mut() { sol.sort_unstable_by_key(|d| d.variable.0) }
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
    /// Returns the number of nodes that have been explored so far.
    fn explored(&self) -> usize {
        self.shared.critical.lock().explored
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
    use crate::*;
    
    type DdLel<'a, T> = ParallelSolver<'a, T, DefaultMDDLEL<T>, EmptyCache<T>>;
    type DdFc <'a, T> = ParallelSolver<'a, T, DefaultMDDFC<T>, SimpleCache<T>>;

    #[test]
    fn by_default_best_lb_is_min_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        assert_eq!(isize::min_value(), solver.best_lower_bound());
    }
    #[test]
    fn by_default_best_ub_is_plus_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        assert_eq!(isize::max_value(), solver.best_upper_bound());
    }
    #[test]
    fn when_the_problem_is_solved_best_lb_is_best_value() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

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
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let _ = solver.maximize();
        assert_eq!(220, solver.best_upper_bound());
    }
    
    #[test]
    fn no_solution_before_solving() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );
        assert!(solver.best_solution().is_none());
    }
    #[test]
    fn empty_fringe_before_solving() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        assert!(solver.best_value().is_none());
        assert!(fringe.is_empty());
    }

    /* this, I can't test...
    #[test]
    fn when_the_solver_is_cutoff_ub_is_that_of_the_best_thread() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &&problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DD::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe
        );
        let _ = solver.maximize();
        assert_eq!(220, solver.best_upper_bound());
    }
    */

    #[test]
    fn default_best_lb_is_neg_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        assert_eq!(isize::min_value(), solver.best_lower_bound());
    }
    #[test]
    fn default_best_ub_is_pos_infinity() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        assert_eq!(isize::max_value(), solver.best_upper_bound());
    }

    #[test]
    fn maximizes_yields_the_optimum_1a() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap();
        sln.sort_unstable_by_key(|d| d.variable.id());
        assert_eq!(sln, vec![
            Decision{variable: Variable(0), value: 0},
            Decision{variable: Variable(1), value: 1},
            Decision{variable: Variable(2), value: 1},
        ]);
    }

    #[test]
    fn maximizes_yields_the_optimum_1b() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdFc::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap();
        sln.sort_unstable_by_key(|d| d.variable.id());
        assert_eq!(sln, vec![
            Decision{variable: Variable(0), value: 0},
            Decision{variable: Variable(1), value: 1},
            Decision{variable: Variable(2), value: 1},
        ]);
    }

    #[test]
    fn maximizes_yields_the_optimum_2a() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 210, 12, 5, 100, 120, 110],
            weight  : vec![10,  45, 20, 4,  20,  30,  50]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap();
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
    fn maximizes_yields_the_optimum_2b() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 210, 12, 5, 100, 120, 110],
            weight  : vec![10,  45, 20, 4,  20,  30,  50]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdFc::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap();
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
    fn maximizes_yields_the_optimum_2c() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 210, 12, 5, 100, 120, 110],
            weight  : vec![10,  45, 20, 4,  20,  30,  50]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap();
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
    fn maximizes_yields_the_optimum_2d() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 210, 12, 5, 100, 120, 110],
            weight  : vec![10,  45, 20, 4,  20,  30,  50]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdFc::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let maximized = solver.maximize();

        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(220));
        assert!(solver.best_solution().is_some());

        let mut sln = solver.best_solution().unwrap();
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
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let d1  = Decision{variable: Variable(0), value: 10};
        let sol = vec![d1];

        solver.set_primal(10, sol.clone());
        assert!(solver.best_solution().is_some());
        assert_eq!(10, solver.best_lower_bound());

        // in this case, it wont update because there is no improvement
        solver.set_primal(5, sol.clone());
        assert!(solver.best_solution().is_some());
        assert_eq!(10, solver.best_lower_bound());

        // but here, it will update as it improves the best known sol
        solver.set_primal(10000, sol);
        assert!(solver.best_solution().is_some());
        assert_eq!(10000, solver.best_lower_bound());

        // it wont do much as the primal is better than the actual feasible solution
        let maximized = solver.maximize();
        assert!(maximized.is_exact);
        assert_eq!(maximized.best_value, Some(10000));
        assert!(solver.best_solution().is_some());
    }

    #[test]
    fn when_no_solution_is_found_the_gap_is_one() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        assert_eq!(1.0, solver.gap());
    }
    #[test]
    fn when_optimum_solution_is_found_the_gap_is_zero() {
        let problem = Knapsack {
            capacity: 50,
            profit  : vec![60, 100, 120],
            weight  : vec![10,  20,  30]
        };
        let relax = KPRelax {pb: &problem};
        let ranking = KPRanking;
        let cutoff = NoCutoff;
        let width = NbUnassignedWidth(problem.nb_variables());
        let dominance = EmptyDominanceChecker::default();
        let mut fringe = SimpleFringe::new(MaxUB::new(&ranking));
        let mut solver = DdLel::custom(
            &problem,
            &relax,
            &ranking,
            &width,
            &dominance,
            &cutoff,
            &mut fringe,
            1,
        );

        let Completion{is_exact, best_value} = solver.maximize();
        assert!(is_exact);
        assert!(best_value.is_some());
        assert_eq!(0.0, solver.gap());
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct KnapsackState {
        depth: usize,
        capacity: usize
    }
    struct Knapsack {
        capacity: usize,
        profit: Vec<usize>,
        weight: Vec<usize>,
    }

    const TAKE_IT: isize = 1;
    const LEAVE_IT_OUT: isize = 0;

    impl Problem for Knapsack {
        type State = KnapsackState;
        fn nb_variables(&self) -> usize {
            self.profit.len()
        }
        fn initial_state(&self) -> Self::State {
            KnapsackState{ depth: 0, capacity: self.capacity }
        }
        fn initial_value(&self) -> isize {
            0
        }
        fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
            let mut ret = *state;
            ret.depth  += 1;
            if dec.value == TAKE_IT { 
                ret.capacity -= self.weight[dec.variable.id()] 
            }
            ret
        }
        fn transition_cost(&self, _state: &Self::State, _: &Self::State, dec: Decision) -> isize {
            self.profit[dec.variable.id()] as isize * dec.value
        }
        fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
            let n = self.nb_variables();
            if depth < n {
                Some(Variable(depth))
            } else {
                None
            }
        }
        fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
        {
            if state.capacity >= self.weight[variable.id()] {
                f.apply(Decision { variable, value: TAKE_IT });
                f.apply(Decision { variable, value: LEAVE_IT_OUT });
            } else {
                f.apply(Decision { variable, value: LEAVE_IT_OUT });
            }
        }
    }
    struct KPRelax<'a>{pb: &'a Knapsack}
    impl Relaxation for KPRelax<'_> {
        type State = KnapsackState;

        fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
            states.max_by_key(|node| node.capacity).copied().unwrap()
        }
        fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
            cost
        }
        fn fast_upper_bound(&self, state: &Self::State) -> isize {
            let mut tot = 0;
            for var in state.depth..self.pb.nb_variables() {
                if self.pb.weight[var] <= state.capacity {
                    tot += self.pb.profit[var];
                }
            }
            tot as isize
        }
    }
    struct KPRanking;
    impl StateRanking for KPRanking {
        type State = KnapsackState;

        fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
            a.capacity.cmp(&b.capacity)
        }
    }
}
