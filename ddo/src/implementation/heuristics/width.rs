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

//! This module provides the implementation of various maximum width heuristics.

use crate::{WidthHeuristic, SubProblem};


/// This strategy specifies a fixed maximum width for all the layers of an
/// approximate MDD. This is a *static* heuristic as the width will remain fixed
/// regardless of the approximate MDD to generate.
///
/// # Example
/// Assuming a fixed width of 100, and problem with 5 variables (0..=4). The
/// heuristic will return 100 regardles of the suproblem being processed.
///
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// let heuristic = FixedWidth(100); // assume a fixed width of 100
/// 
/// // assume the exsitence of whatever subroblem you like..
/// let subproblem = SubProblem {state: Arc::new('a'), value: 42, ub: 100, depth: 0, path: vec![]};
/// // still, the heuristic always return 100.
/// assert_eq!(100, heuristic.max_width(&subproblem));
/// ```
/// 
/// # Typical usage example
/// Typically, you will only ever create a FixedWidth policy when instanciating 
/// your solver. The following example shows how you create a solver that imposes
/// a fixed maximum layer width to all layers it compiles.
/// 
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
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
/// #         self.profit[dec.variable.id()] as isize * dec.value
/// #     }
/// #     fn next_variable(&self, depth: usize, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
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
/// # let problem = Knapsack {
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// # };
/// # let relaxation = KPRelax{pb: &problem};
/// # let heuristic = KPRanking;
/// # let dominance = SimpleDominanceChecker::new(KPDominance);
/// # let cutoff = NoCutoff; // might as well be a TimeBudget (or something else) 
/// # let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
/// #
/// let mut solver = DefaultSolver::new(
///       &problem, 
///       &relaxation, 
///       &heuristic, 
///       &FixedWidth(100), // all DDs will be compiled with a maximum width of 100 nodes 
///       &dominance,
///       &cutoff, 
///       &mut fringe);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct FixedWidth(pub usize);
impl <X> WidthHeuristic<X> for FixedWidth {
    fn max_width(&self, _: &SubProblem<X>) -> usize {
        self.0
    }
}

/// This strategy specifies a variable maximum width for the layers of an
/// approximate MDD. When using this heuristic, each layer of an approximate
/// MDD is allowed to have as many nodes as there are free variables to decide
/// upon.
///
/// # Example
/// Assuming a problem with 5 variables (0..=4). If we are calling this heuristic
/// to derive the maximum allowed width for the layers of an approximate MDD
/// when variables {1, 3, 4} have been fixed, then there are only two "free"
/// variables. Namely, variables 0 and 2. In that case, this strategy will return
/// a max width of two.
///
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///      type State = KnapsackState;
///      fn nb_variables(&self) -> usize {
///          5
///      }
///       // details omitted in this example
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
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
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
/// }
/// let problem = Knapsack {
///       // details omitted in this example
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let heuristic = NbUnassignedWitdh(problem.nb_variables());
/// let subproblem= SubProblem {
/// #    state: Arc::new(KnapsackState{depth: 3, capacity: 2}), 
/// #    value: 5, 
/// #    ub: 100, 
/// #    depth: 3,
///     // three decisions have already been made. There only remain variables 0 and 2
///     path: vec![
///         Decision{variable: Variable(1), value: 1},
///         Decision{variable: Variable(3), value: 1},
///         Decision{variable: Variable(4), value: 1},
///     ]
/// };
/// assert_eq!(2, heuristic.max_width(&subproblem));
/// ```
/// 
/// # Typical usage example
/// Typically, you will only ever create a NbUnassignedWitdh policy when instantiating 
/// your solver. The following example shows how you create a solver that imposes
/// a maximum layer width of one node per unassigned variable.
/// 
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// pub struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///       // details omitted in this example
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
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
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
/// }
/// struct KPRelax<'a>{pb: &'a Knapsack}
/// impl Relaxation for KPRelax<'_> {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
/// #         states.max_by_key(|node| node.capacity).copied().unwrap()
/// #     }
/// #     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
/// #         cost
/// #     }
/// }
/// # 
/// struct KPRanking;
/// impl StateRanking for KPRanking {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
/// #         a.capacity.cmp(&b.capacity)
/// #     }
/// }
/// pub struct KPDominance;
/// impl Dominance for KPDominance {
///       // details omitted in this example
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
/// }
/// 
/// let problem = Knapsack {
///       // details omitted
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let relaxation = KPRelax{pb: &problem};
/// let heuristic = KPRanking;
/// let dominance = SimpleDominanceChecker::new(KPDominance);
/// let cutoff = NoCutoff; // might as well be a TimeBudget (or something else) 
/// let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
/// #
/// let mut solver = DefaultSolver::new(
///       &problem, 
///       &relaxation, 
///       &heuristic, 
///       &NbUnassignedWitdh(problem.nb_variables()),
///       &dominance,
///       &cutoff, 
///       &mut fringe);
/// ```
#[derive(Default, Debug, Copy, Clone)]
pub struct NbUnassignedWitdh(pub usize);
impl <X> WidthHeuristic<X> for NbUnassignedWitdh {
    fn max_width(&self, x: &SubProblem<X>) -> usize {
        self.0 - x.path.len()
    }
}

/// This strategy acts as a decorator for an other max width heuristic. It
/// multiplies the maximum width of the strategy it delegates to by a constant
/// (configured) factor. It is typically used in conjunction with NbUnassigned
/// to provide a maximum width that allows a certain number of nodes.
/// Using a constant factor of 1 means that this decorator will have absolutely
/// no impact.
/// 
/// # Note:
/// This wrapper forces a minimum width of one. So it is *never*
/// going to return 0 for a value of the max width.
///
/// # Example
/// Here is an example of how to use this strategy to allow 5 times as many nodes
/// as there are nodes unassigned variable in a layer.
/// 
/// In the following example, there are 5 variables. Three of these have already
/// been assigned. Which means there are only two unassigned variables left.
/// Using the Times(5) decorator would mean that the maximum allowed width for a
/// layer is 10 nodes. 
///
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///      type State = KnapsackState;
///      fn nb_variables(&self) -> usize {
///          5
///      }
///       // details omitted in this example
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
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
/// #         self.profit[dec.variable.id()] as isize * dec.value
/// #     }
/// #     fn next_variable(&self, depth: usize, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
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
/// }
/// let problem = Knapsack {
///       // details omitted in this example
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let heuristic = Times(5, NbUnassignedWitdh(problem.nb_variables()));
/// let subproblem= SubProblem {
/// #    state: Arc::new(KnapsackState{depth: 3, capacity: 2}), 
/// #    value: 5, 
/// #    ub: 100, 
/// #    depth: 3,
///     // three decisions have already been made. There only remain variables 0 and 2
///     path: vec![
///         Decision{variable: Variable(1), value: 1},
///         Decision{variable: Variable(3), value: 1},
///         Decision{variable: Variable(4), value: 1},
///     ]
/// };
/// assert_eq!(10, heuristic.max_width(&subproblem));
/// ```
/// 
/// # Typical usage example
/// Typically, you will only ever create a Times policy when instantiating 
/// your solver. The following example shows how you create a solver that imposes
/// a fixed maximum layer width of five node per unassigned variable.
/// 
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// pub struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///       // details omitted in this example
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
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
/// #         self.profit[dec.variable.id()] as isize * dec.value
/// #     }
/// #     fn next_variable(&self, depth: usize, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
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
/// }
/// struct KPRelax<'a>{pb: &'a Knapsack}
/// impl Relaxation for KPRelax<'_> {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
/// #         states.max_by_key(|node| node.capacity).copied().unwrap()
/// #     }
/// #     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
/// #         cost
/// #     }
/// }
/// # 
/// struct KPRanking;
/// impl StateRanking for KPRanking {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
/// #         a.capacity.cmp(&b.capacity)
/// #     }
/// }
/// pub struct KPDominance;
/// impl Dominance for KPDominance {
///       // details omitted in this example
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
/// }
/// 
/// let problem = Knapsack {
///       // details omitted
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let relaxation = KPRelax{pb: &problem};
/// let heuristic = KPRanking;
/// let dominance = SimpleDominanceChecker::new(KPDominance);
/// let cutoff = NoCutoff; // might as well be a TimeBudget (or something else) 
/// let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
/// #
/// let mut solver = DefaultSolver::new(
///       &problem, 
///       &relaxation, 
///       &heuristic, 
///       &Times(5, NbUnassignedWitdh(problem.nb_variables())),
///       &dominance,
///       &cutoff, 
///       &mut fringe);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Times<X>(pub usize, pub X);

impl <S, X: WidthHeuristic<S>> WidthHeuristic<S> for Times<X> {
    fn max_width(&self, x: &SubProblem<S>) -> usize {
        1.max(self.0 * self.1.max_width(x))
    }
}

/// This strategy acts as a decorator for an other max width heuristic. It
/// divides the maximum width of the strategy it delegates to by a constant
/// (configured) factor. It is typically used in conjunction with NbUnassigned
/// to provide a maximum width that allows a certain number of nodes.
/// Using a constant factor of 1 means that this decorator will have absolutely
/// no impact.
///
/// # Note
/// The maximum width is bounded by one at the very minimum. So it is *never*
/// going to return 0 for a value of the max width.
///
/// # Example
/// Here is an example of how to use this strategy to allow 1 nodes per two
/// unassigned variables in a layer.
///
/// In the following example, there are 5 variables. Three of these have already
/// been assigned. Which means there are only two unassigned variables left.
/// Using the DivBy(2) decorator would mean that the maximum allowed width for a
/// layer is 1 node. 
///
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///      type State = KnapsackState;
///      fn nb_variables(&self) -> usize {
///          5
///      }
///       // details omitted in this example
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
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
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
/// }
/// let problem = Knapsack {
///       // details omitted in this example
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let heuristic = DivBy(2, NbUnassignedWitdh(problem.nb_variables()));
/// let subproblem= SubProblem {
/// #    state: Arc::new(KnapsackState{depth: 3, capacity: 2}), 
/// #    value: 5, 
/// #    ub: 100, 
/// #    depth: 3,
///     // three decisions have already been made. There only remain variables 0 and 2
///     path: vec![
///         Decision{variable: Variable(1), value: 1},
///         Decision{variable: Variable(3), value: 1},
///         Decision{variable: Variable(4), value: 1},
///     ]
/// };
/// assert_eq!(1, heuristic.max_width(&subproblem));
/// ```
/// 
/// # Typical usage example
/// Typically, you will only ever create a DivBy policy when instantiating 
/// your solver. The following example shows how you create a solver that imposes
/// a maximum layer width of one node per two unassigned variables.
/// 
/// ```
/// # use ddo::*;
/// # use std::sync::Arc;
/// #
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// pub struct KnapsackState {
///       // details omitted in this example
/// #     depth: usize,
/// #     capacity: usize
/// }
/// # 
/// struct Knapsack {
///       // details omitted in this example
/// #     capacity: usize,
/// #     profit: Vec<usize>,
/// #     weight: Vec<usize>,
/// }
/// # 
/// # const TAKE_IT: isize = 1;
/// # const LEAVE_IT_OUT: isize = 0;
/// # 
/// impl Problem for Knapsack {
///       // details omitted in this example
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
/// #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
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
/// }
/// struct KPRelax<'a>{pb: &'a Knapsack}
/// impl Relaxation for KPRelax<'_> {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
/// #         states.max_by_key(|node| node.capacity).copied().unwrap()
/// #     }
/// #     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
/// #         cost
/// #     }
/// }
/// # 
/// struct KPRanking;
/// impl StateRanking for KPRanking {
///       // details omitted in this example
/// #     type State = KnapsackState;
/// #     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
/// #         a.capacity.cmp(&b.capacity)
/// #     }
/// }
/// pub struct KPDominance;
/// impl Dominance for KPDominance {
///       // details omitted in this example
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
/// }
/// let problem = Knapsack {
///       // details omitted
/// #     capacity: 50,
/// #     profit  : vec![60, 100, 120],
/// #     weight  : vec![10,  20,  30]
/// };
/// let relaxation = KPRelax{pb: &problem};
/// let heuristic = KPRanking;
/// let dominance = SimpleDominanceChecker::new(KPDominance);
/// let cutoff = NoCutoff; // might as well be a TimeBudget (or something else) 
/// let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
/// #
/// let mut solver = DefaultSolver::new(
///       &problem, 
///       &relaxation, 
///       &heuristic, 
///       &DivBy(2, NbUnassignedWitdh(problem.nb_variables())),
///       &dominance,
///       &cutoff, 
///       &mut fringe);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DivBy<X>(pub usize, pub X);

impl <S, X: WidthHeuristic<S>> WidthHeuristic<S> for DivBy<X> {
    fn max_width(&self, x: &SubProblem<S>) -> usize {
        1.max(self.1.max_width(x) / self.0)
    }
}


#[cfg(test)]
mod test_nbunassigned {
    use std::sync::Arc;

    use crate::*;

    #[test]
    fn non_empty() {
        // assume a problem with 5 variables
        let heu = NbUnassignedWitdh(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![Decision{variable: Variable(0), value: 4}],
            depth: 1,
        };
        assert_eq!(4, heu.max_width(&sub));
    }
    #[test]
    fn all() {
        let heu = NbUnassignedWitdh(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![], // no decision made, all vars are available
            depth: 0,
        };
        assert_eq!(5, heu.max_width(&sub));
    }
    #[test]
    fn empty() {
        // assume a problem with 5 variables
        let heu = NbUnassignedWitdh(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ],
            depth: 5,
        };
        assert_eq!(0, heu.max_width(&sub));
    }
}
#[cfg(test)]
mod test_fixedwidth {
    use std::sync::Arc;

    use crate::*;

    #[test]
    fn non_empty() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![
                Decision{variable: Variable(0), value: 0},
                ],
            depth: 1,
        };
        assert_eq!(5, heu.max_width(&sub));
    }
    #[test]
    fn all() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![],
            depth: 0,
        };
        assert_eq!(5, heu.max_width(&sub));
    }
    #[test]
    fn empty() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ],
            depth: 5,
        };
        assert_eq!(5, heu.max_width(&sub));
    }
}
#[cfg(test)]
mod test_adapters {
    use std::sync::Arc;

    use crate::*;

    #[test]
    fn test_times() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ],
            depth: 5,
        };
        assert_eq!(10, Times( 2, heu).max_width(&sub));
        assert_eq!(15, Times( 3, heu).max_width(&sub));
        assert_eq!( 5, Times( 1, heu).max_width(&sub));
        assert_eq!(50, Times(10, heu).max_width(&sub));
    }
    #[test]
    fn test_div_by() {
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ],
            depth: 5,
        };
        assert_eq!( 2, DivBy( 2, FixedWidth(4)).max_width(&sub));
        assert_eq!( 3, DivBy( 3, FixedWidth(9)).max_width(&sub));
        assert_eq!(10, DivBy( 1, FixedWidth(10)).max_width(&sub));
    }

    #[test]
    fn wrappers_never_return_a_zero_maxwidth() {
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ],
            depth: 5,
        };
        assert_eq!( 1, Times( 0, FixedWidth(10)).max_width(&sub));
        assert_eq!( 1, Times(10, FixedWidth( 0)).max_width(&sub));
    }

    #[test] #[should_panic]
    fn test_div_by_panics_when_div_by_zero() {
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ],
            depth: 5,
        };
        DivBy( 0, FixedWidth(0)).max_width(&sub);
    }
}
