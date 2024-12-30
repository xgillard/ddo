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

use crate::{SubProblem, Completion, Reason, Problem, Relaxation, StateRanking, Solution, Cutoff, Cache, DominanceChecker};

// FIXME: Replace that with the following enum definition when const generics allow enum types
/// What type of cut-set are we using for relaxed DDs ?
pub type CutsetType = u8;
/// enqueue the last layer with only exact nodes
pub const LAST_EXACT_LAYER: u8 = 1;
/// enqueue all exact nodes that have at least a relaxed child node
pub const FRONTIER: u8 = 2;
/*
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutsetType {
    /// enqueue the last layer with only exact nodes
    LastExactLayer,
    /// enqueue all exact nodes that have at least a relaxed child node
    Frontier,
}
*/

/// How are we to compile the decision diagram ? 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationType {
    /// If you want to use a pure DP resolution of the problem
    Exact,
    /// If you want to compile a restricted DD which yields a lower bound on the objective
    Relaxed,
    /// If you want to compile a relaxed DD which yields an upper bound on the objective
    Restricted,
}

/// The set of parameters used to tweak the compilation of a MDD
pub struct CompilationInput<'a, State> {   
    /// How is the mdd being compiled ?
    pub comp_type: CompilationType,
    /// A reference to the original problem we try to maximize
    pub problem: &'a dyn Problem<State = State>,
    /// The relaxation which we use to merge nodes in a relaxed dd
    pub relaxation: &'a dyn Relaxation<State = State>,
    /// The state ranking heuristic to chose the nodes to keep and those to discard
    pub ranking: &'a dyn StateRanking<State = State>,
    /// The cutoff used to decide when to stop trying to solve the problem
    pub cutoff: &'a dyn Cutoff,
    /// What is the maximum width of the mdd ?
    pub max_width: usize,
    /// The subproblem whose state space must be explored
    pub residual: &'a SubProblem<State>,
    /// The best known lower bound at the time when the dd is being compiled
    pub best_lb: isize,
    /// Data structure containing info about past compilations used to prune the search
    pub cache: &'a dyn Cache<State>,
    pub dominance: &'a dyn DominanceChecker<State>,
}

/// This trait describes the operations that can be expected from an abstract
/// decision diagram regardless of the way it is implemented.
pub trait DecisionDiagram<State> {
    /// This method provokes the compilation of the DD based on the given 
    /// compilation input (compilation type, and root subproblem)
    fn compile(&mut self, input: &CompilationInput<State>) 
        -> Result<Completion, Reason>;
    /// Returns true iff the DD which has been compiled is an exact DD.
    fn is_exact(&self) -> bool;
    /// Returns the optimal value of the objective function or None when no 
    /// feasible solution has been identified (no r-t path) either because
    /// the subproblem at the root of this DD is infeasible or because restriction
    /// has removed all feasible paths that could potentially have been found.
    fn best_value(&self) -> Option<isize>;
    /// Returns the best solution of this subproblem as a sequence of decision
    /// maximizing the objective value. When no feasible solution exists in the
    /// approximate DD, it returns the value None instead.
    fn best_solution(&self) -> Option<Solution>;
    /// Returns the value of the objective function for the best exact node in the DD
    /// or None when no feasible solution has been identified (no r-t path) either because
    /// the subproblem at the root of this DD is infeasible or because restriction/relaxation
    /// has removed all feasible paths that could potentially have been found.
    fn best_exact_value(&self) -> Option<isize>;
    /// Returns the best exact solution of this subproblem as a sequence of decision
    /// maximizing the objective value. When no feasible solution exists in the
    /// approximate DD, it returns the value None instead.
    fn best_exact_solution(&self) -> Option<Solution>;
    /// Iteratively applies the given function `func` to each element of the
    /// exact cut-set that was computed during DD compilation.
    ///
    /// # Important:
    /// This can only be called if the DD was compiled in relaxed mode.
    /// All implementations of the DecisionDiagram trait are allowed to assume
    /// this method will be called at most once per relaxed DD compilation.
    fn drain_cutset<F>(&mut self, func: F)
    where
        F: FnMut(SubProblem<State>);
}