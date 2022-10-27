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
//! a solver that will solve the problem using one single thread of execution.
//! 
//! This is usually not the implementation you will want to use if you are 
//! after solving a hard problem efficiently. However, in those cases where
//! you would be using a constrained environment (e.g. a python interpreter)
//! where multithreading is not an option; then you might want to use this 
//! implementation instead.
use std::clone::Clone;
use std::{sync::Arc, hash::Hash};

use crate::{Frontier, Decision, Problem, Relaxation, StateRanking, WidthHeuristic, Cutoff, SubProblem, DecisionDiagram, DefaultMDD, CompilationInput, CompilationType, Solver, Solution, Completion, Reason};

/// The workload a thread can get from the shared state
enum WorkLoad<T> {
    /// There is no work left to be done: you can safely terminate
    Complete,
    /// The work must stop because of an external cutoff
    Aborted,
    /// The item to process
    WorkItem { node: SubProblem<T> },
}

pub struct SequentialSolver<'a, State, D> 
where D: DecisionDiagram<State = State> + Default,
{
    /// A reference to the problem being solved with branch-and-bound MDD
    problem: &'a (dyn Problem<State = State>),
    /// The relaxation used when a DD layer grows too large
    relaxation: &'a (dyn Relaxation<State = State>),
    /// The ranking heuristic used to discriminate the most promising from
    /// the least promising states
    ranking: &'a (dyn StateRanking<State = State>),
    /// The maximum width heuristic used to enforce a given maximum memory
    /// usage when compiling mdds
    width_heu: &'a (dyn WidthHeuristic<State>),
    /// A cutoff heuristic meant to decide when to stop the resolution of 
    /// a given problem.
    cutoff: &'a (dyn Cutoff),

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
    fringe: &'a mut (dyn Frontier<State = State>),
    /// This is a counter that tracks the number of nodes that have effectively
    /// been explored. That is, the number of nodes that have been popped from
    /// the fringe, and for which a restricted and relaxed mdd have been developed.
    explored: usize,
    /// This is the value of the best known lower bound.
    best_lb: isize,
    /// This is the value of the best known upper bound.
    best_ub: isize,
    /// If set, this keeps the info about the best solution so far.
    best_sol: Option<Vec<Decision>>,
    /// If we decide not to go through a complete proof of optimality, this is
    /// the reason why we took that decision.
    abort_proof: Option<Reason>,

    /// This is just a marker that allows us to remember the exact type of the
    /// mdds to be instanciated.
    mdd: D,
}

// private interface.
impl <'a, State> SequentialSolver<'a, State, DefaultMDD<State>>
where State: Eq + Hash + Clone
{
    pub fn new(
        problem: &'a (dyn Problem<State = State>),
        relaxation: &'a (dyn Relaxation<State = State>),
        ranking: &'a (dyn StateRanking<State = State>),
        width: &'a (dyn WidthHeuristic<State>),
        cutoff: &'a (dyn Cutoff), 
        fringe: &'a mut (dyn Frontier<State = State>),
    ) -> Self {
        Self::custom(problem, relaxation, ranking, width, cutoff, fringe)
    }
}


impl<'a, State, D>  SequentialSolver<'a, State, D>
where 
    State: Eq + Hash + Clone,
    D: DecisionDiagram<State = State> + Default,
{
    pub fn custom(
        problem: &'a (dyn Problem<State = State>),
        relaxation: &'a (dyn Relaxation<State = State>),
        ranking: &'a (dyn StateRanking<State = State>),
        width_heu: &'a (dyn WidthHeuristic<State>),
        cutoff: &'a (dyn Cutoff),
        fringe: &'a mut (dyn Frontier<State = State>),
    ) -> Self {
        SequentialSolver {
            problem,
            relaxation,
            ranking,
            width_heu,
            cutoff,
            //
            best_sol: None,
            best_lb: isize::MIN,
            best_ub: isize::MAX,
            fringe,
            explored: 0,
            abort_proof: None,
            mdd: D::default(),
        }
    }

    /// This method initializes the problem resolution. Put more simply, this
    /// method posts the root node of the mdd onto the fringe so that a thread
    /// can pick it up and the processing can be bootstrapped.
    fn initialize(&mut self) {
        let root = self.root_node();
        self.fringe.push(root);
    }

    fn root_node(&self) -> SubProblem<State> {
        SubProblem {
            state: Arc::new(self.problem.initial_state()),
            value: self.problem.initial_value(),
            path: vec![],
            ub: isize::MAX,
        }
    }

    /// This method processes the given `node`. To do so, it reads the current
    /// best lower bound from the critical data. Then it expands a restricted
    /// and possibly a relaxed mdd rooted in `node`. If that is necessary,
    /// it stores cutset nodes onto the fringe for further parallel processing.
    fn process_one_node(
        &mut self,
        node: SubProblem<State>,
    ) -> Result<(), Reason> {
        // 1. RESTRICTION
        let node_ub = node.ub;
        let best_lb = self.best_lb;

        if node_ub <= best_lb {
            return Ok(());
        }

        let width = self.width_heu.max_width(&node);
        let mut compilation = CompilationInput {
            comp_type: CompilationType::Restricted,
            max_width: width,
            problem: self.problem,
            relaxation: self.relaxation,
            ranking: self.ranking,
            cutoff: self.cutoff,
            residual: node,
            //
            best_lb,
        };

        let Completion{is_exact, ..} = self.mdd.compile(&compilation)?;
        self.maybe_update_best();
        if is_exact {
            return Ok(());
        }

        // 2. RELAXATION
        let best_lb = self.best_lb;
        compilation.comp_type = CompilationType::Relaxed;
        compilation.best_lb = best_lb;

        let Completion{is_exact, ..} = self.mdd.compile(&compilation)?;
        if is_exact {
            self.maybe_update_best();
        } else {
            self.enqueue_cutset(node_ub);
        }

        Ok(())
    }

    /// This private method updates the shared best known node and lower bound in
    /// case the best value of the current `mdd` expansion improves the current
    /// bounds.
    fn maybe_update_best(&mut self) {
        let dd_best_value = self.mdd.best_value().unwrap_or(isize::MIN);
        if dd_best_value > self.best_lb {
            self.best_lb = dd_best_value;
            self.best_sol = self.mdd.best_solution();
        }
    }
    /// If necessary, thightens the bound of nodes in the cutset of `mdd` and
    /// then add the relevant nodes to the shared fringe.
    fn enqueue_cutset(&mut self, ub: isize) {
        let best_lb = self.best_lb;
        let fringe = &mut self.fringe;
        self.mdd.drain_cutset(|mut cutset_node| {
            cutset_node.ub = ub.min(cutset_node.ub);
            if cutset_node.ub > best_lb {
                fringe.push(cutset_node);
            }
        });
    }

    fn abort_search(&mut self, reason: Reason) {
        self.abort_proof = Some(reason);
        self.fringe.clear();
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
    fn get_workload(&mut self) -> WorkLoad<State>
    {
        // Do we need to stop
        if self.abort_proof.is_some() {
            return WorkLoad::Aborted;
        }
        if self.fringe.is_empty() {
            self.best_ub = self.best_lb;
            return WorkLoad::Complete;
        }

        let nn = self.fringe.pop().unwrap();

        // Consume the current node and process it
        self.explored += 1;
        self.best_ub   = nn.ub;

        WorkLoad::WorkItem { node: nn }
    }
}

impl<'a, State, D> Solver for SequentialSolver<'a, State, D>
where
    State: Eq + PartialEq + Hash + Clone,
    D: DecisionDiagram<State = State> + Default,
{
    /// Applies the branch and bound algorithm proposed by Bergman et al. to
    /// solve the problem to optimality. To do so, it spawns `nb_threads` workers
    /// (long running threads); each of which will continually get a workload
    /// and process it until the problem is solved.
    fn maximize(&mut self) -> Completion {
        self.initialize();

        loop {
            match self.get_workload() {
                WorkLoad::Complete => break,
                WorkLoad::Aborted => break, // this one cannot occur
                WorkLoad::WorkItem { node } => {
                    let outcome = self.process_one_node(node);
                    if let Err(reason) = outcome {
                        self.abort_search(reason); 
                        break;
                    }
                }
            }
        }

        Completion { is_exact: self.abort_proof.is_none(), best_value: self.best_sol.as_ref().map(|_| self.best_lb) }
    }

    /// Returns the best solution that has been identified for this problem.
    fn best_solution(&self) -> Option<Vec<Decision>> {
        self.best_sol.clone()
    }
    /// Returns the value of the best solution that has been identified for
    /// this problem.
    fn best_value(&self) -> Option<isize> {
        self.best_sol.as_ref().map(|_sol| self.best_lb)
    }
    /// Returns the value of the best lower bound that has been identified for
    /// this problem.
    fn best_lower_bound(&self) -> isize {
        self.best_lb
    }
    /// Returns the value of the best upper bound that has been identified for
    /// this problem.
    fn best_upper_bound(&self) -> isize {
        self.best_ub
    }
    /// Sets a primal (best known value and solution) of the problem.
    fn set_primal(&mut self, value: isize, solution: Solution) {
        if value > self.best_lb {
            self.best_sol = Some(solution);
            self.best_lb  = value;
        }
    }
}
