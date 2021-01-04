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

//! This module defines traits for implementations of an MDD.

use crate::common::{Solution, FrontierNode, VarSet, Variable, Domain, Decision, Completion, Reason};
use std::cmp::Ordering;
use crate::abstraction::heuristics::SelectableNode;

/// This trait describes an MDD
///
/// # Type param
/// The type parameter `<T>` denotes the type of the state defined/manipulated
/// by the `Problem` definition.
pub trait MDD<T, C: Config<T>> {
    /// Returns a reference to the configuration of this MDD.
    fn config(&self) -> &C;
    fn config_mut(&mut self) -> &mut C;

    /// Expands this MDD into  an exact MDD
    fn exact(&mut self, root: &FrontierNode<T>, best_lb : isize, ub: isize) -> Result<Completion, Reason>;
    /// Expands this MDD into a restricted (lower bound approximation)
    /// version of the exact MDD.
    fn restricted(&mut self, root: &FrontierNode<T>, best_lb : isize, ub: isize) -> Result<Completion, Reason>;
    /// Expands this MDD into a relaxed (upper bound approximation)
    /// version of the exact MDD.
    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb : isize, ub: isize) -> Result<Completion, Reason>;

    /// Return true iff this `MDD` is exact. That is to say, it returns true if
    /// no nodes have been merged (because of relaxation) or suppressed (because
    /// of restriction).
    fn is_exact(&self) -> bool;
    /// Returns the length of the longest path between the root and the terminal
    /// node of this `MDD`.
    fn best_value(&self) -> isize;
    /// Returns the longest path between the root and terminal node of this MDD.
    fn best_solution(&self) -> Option<Solution>;

    /// Applies the given function `func` to all nodes from the exact cutset of
    /// this mdd.
    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>);
}

/// The config trait describes the configuration of an MDD. In other words, it
/// encapsulates the configurable behavior (problem, relaxation, heuristics,..)
/// of an MDD. Such a configuration is typically obtained from a builder.
pub trait Config<T> {
    // ------------------------------------------------------------------------
    // --- From the Problem definition ----------------------------------------
    // ------------------------------------------------------------------------
    /// Yields the root node of the (exact) MDD standing for the problem to solve.
    fn root_node(&self) -> FrontierNode<T>;
    /// Returns the domain of variable `v` in the given `state`. These are the
    /// possible values that might possibly be affected to `v` when the system
    /// has taken decisions leading to `state`.
    fn domain_of<'a>(&self, state: &'a T, v: Variable) -> Domain<'a>;
    /// Returns the next state reached by the system if the decision `d` is
    /// taken when the system is in the given `state` and the given set of `vars`
    /// are still free (no value assigned).
    fn transition(&self, state: &T, vars : &VarSet, d: Decision) -> T;
    /// Returns the marginal benefit (in terms of objective function to maximize)
    /// of taking decision `d` is when the system is in the given `state` and
    /// the given set of `vars` are still free (no value assigned).
    fn transition_cost(&self, state: &T, vars : &VarSet, d: Decision) -> isize;
    /// Optional method for the case where you'd want to use a pooled mdd
    /// implementation. This method returns true iff taking a decision on
    /// `variable` might have an impact (state or longest path) on a node
    /// having the given `state`.
    fn impacted_by(&self, state: &T, variable: Variable) -> bool;

    // ------------------------------------------------------------------------
    // --- From the Relaxation ------------------------------------------------
    // ------------------------------------------------------------------------
    /// This method merges the given set of `states` into a new _inexact_ state
    /// that is an overapproximation of all `states`. The returned value will be
    /// used as a replacement for the given `states` in the mdd.
    ///
    /// In the theoretical framework of Bergman et al, this would amount to
    /// providing an implementation for the $\oplus$ operator. You should really
    /// only focus on that aspect when developing a relaxation: all the rest
    /// is taken care of by the framework.
    fn merge_states(&self, states: &mut dyn Iterator<Item=&T>) -> T;
    /// This method relaxes the weight of the edge between the nodes `src` and
    /// `dst` because `dst` is replaced in the current layer by `relaxed`. The
    /// `decision` labels and the original weight (`cost`) of the edge
    /// `src` -- `dst` are also recalled.
    ///
    /// In the theoretical framework of Bergman et al, this would amount to
    /// providing an implementation for the $\Gamma$ operator.
    fn relax_edge(&self, src: &T, dst: &T, relaxed: &T, decision: Decision, cost: isize) -> isize;
    /// This optional method derives a _rough upper bound_ (RUB) on the maximum
    /// value of the subproblem rooted in the given `_state`. By default, the
    /// RUB returns the greatest positive integer; which is always safe but does
    /// not provide any pruning.
    fn estimate  (&self, _state  : &T) -> isize {isize::max_value()}

    // ------------------------------------------------------------------------
    // --- The Heuristics -----------------------------------------------------
    // ------------------------------------------------------------------------
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    fn load_variables(&self, node: &FrontierNode<T>) -> VarSet;

    /// Returns the best variable to branch on from the set of 'free' variables
    /// (`free_vars`, the variables that may still be branched upon in this MDD).
    /// It returns `None` in case no branching is useful (ie when no decision is
    /// left to make, etc...).
    fn select_var(
        &self,
        free_vars: &VarSet,
        current_layer: &mut dyn Iterator<Item=&T>,
        next_layer: &mut dyn Iterator<Item=&T>) -> Option<Variable>;

    /// Returns the maximum width allowed for the next layer of the MDD when
    /// a value must still be assigned to each of the variables in `free_vars`.
    fn max_width(&self, free_vars: &VarSet) -> usize;

    /// Defines an order of 'relevance' over the nodes `a` and `b`. Greater means
    /// that `a` is more important (hence more likely to be kept) than `b`.
    fn compare(&self, a: &dyn SelectableNode<T>, b: &dyn SelectableNode<T>) -> Ordering;

    /// Returns true iff the cutoff criterion is met and the search must stop.
    fn must_stop(&self, lb: isize, ub: isize) -> bool;

    // ------------------------------------------------------------------------
    // --- Hooks for stateful heuristics --------------------------------------
    // ------------------------------------------------------------------------
    /// This method provides a hook for you to react to the addition of a new
    /// layer (to the mdd) during development of an mdd. This might be useful
    /// when working with incremental (stateful) heuristics (ie variable
    /// selection heuristic).
    fn upon_new_layer(&mut self,
                      _var: Variable,
                      _current_layer: &mut dyn Iterator<Item=&T>);

    /// This method provides a hook for you to react to the addition of a new
    /// node to the next layer of the mdd during development of the mdd.
    ///
    /// This might be useful when working with incremental (stateful)
    /// heuristics (ie variable selection heuristic).
    fn upon_node_insert(&mut self, _state: &T);

    /// When implementing an incremental variable selection heuristic, this
    /// method should reset the state of the heuristic to a "fresh" state.
    /// This method is called at the start of the development of any mdd.
    ///
    /// This might be useful when working with incremental (stateful)
    /// heuristics (ie variable selection heuristic).
    fn clear(&mut self);
}
