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


//! # DDO
//! DDO is a truly generic framework to develop MDD-based combinatorial
//! optimization solvers in Rust. Its goal is to let you describe your
//! optimization problem as a dynamic program (see `Problem`) along with a
//! `Relaxation`. When the dynamic program of the problem is considered as a
//! transition system, the relaxation serves the purpose of merging different
//! nodes of the transition system into an other node standing for them all.
//! In that setup, the sole condition to ensure the correctness of the
//! optimization algorithm is that the replacement node must be an over
//! approximation of all what is feasible from the merged nodes.
//!
//! ## Side benefit
//! As a side benefit from using `ddo`, you will be able to exploit all of your
//! hardware to solve your optimization in parallel.
//!
//! ## Quick Example
//! The following presents a minimalistic use of ddo. It implements a solver for
//! the knapsack problem which uses all the available computing resources to
//! complete its task. This example is shown for illustration purpose because
//! it is pretty simple and chances are high anybody is already comfortable with
//! the problem definition.
//!
//! #### Note:
//! The `example` folder of our repository contains many other examples in
//! addition to this one. So please consider checking them out for further
//! details.
//!
//! #### Describe the problem as dynamic program
//! The first thing to do in this example is to describe the binary knapsack
//! problem in terms of a dynamic program. Here, the state of a node, is a simple
//! structure that comprises the remaining capacity of the sack (usize) and 
//! a depth to denote the number of variables that have already been assigned.
//! ```
//! /// In our DP model, we consider a state that simply consists of the remaining 
//! /// capacity in the knapsack. Additionally, we also consider the *depth* (number
//! /// of assigned variables) as part of the state since it useful when it comes to
//! /// determine the next variable to branch on.
//! #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//! struct KnapsackState {
//!     /// the number of variables that have already been decided upon in the complete
//!     /// problem.
//!     depth: usize,
//!     /// the remaining capacity in the knapsack. That is the maximum load the sack
//!     /// can bear without cracking **given what is already in the sack**.
//!     capacity: usize
//! }
//! ```
//! 
//! Additionally, we also define a Knapsack structure to store the parameters
//! of the instance being solved. Knapsack is the structure that actually 
//! implements the dynamic programming model for the problem at hand.
//! ```
//! use ddo::*;
//! #
//! # #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//! # struct KnapsackState {
//! #     depth: usize,
//! #     capacity: usize
//! # }
//! # 
//! /// This structure represents a particular instance of the knapsack problem.
//! /// This is the structure that will implement the knapsack model.
//! /// 
//! /// The problem definition is quite easy to understand: there is a knapsack having 
//! /// a maximum (weight) capacity, and a set of items to chose from. Each of these
//! /// items having a weight and a profit, the goal is to select the best subset of
//! /// the items to place them in the sack so as to maximize the profit.
//! struct Knapsack {
//!     /// The maximum capacity of the sack (when empty)
//!     capacity: usize,
//!     /// the profit of each item
//!     profit: Vec<usize>,
//!     /// the weight of each item.
//!     weight: Vec<usize>,
//! }
//! 
//! /// For each variable in the decision problem, there are two possible choices:
//! /// either we take the item in the sack, or we decide to leave it out. This
//! /// constant is used to indicate that the item is to be taken in the sack.
//! const TAKE_IT: isize = 1;
//! /// For each variable in the decision problem, there are two possible choices:
//! /// either we take the item in the sack, or we decide to leave it out. This
//! /// constant is used to indicate that the item is to be left out of the sack.
//! const LEAVE_IT_OUT: isize = 0;
//! 
//! /// This is how you implement the labeled transition system (LTS) semantics of
//! /// a simple dynamic program solving the knapsack problem. The definition of
//! /// each of the methods should be pretty clear and easy to grasp. Should you
//! /// want more details on the role of each of these methods, then you are 
//! /// encouraged to go checking the documentation of the `Problem` trait.
//! impl Problem for Knapsack {
//!     // This associated type indicates that the type which is used to represent
//!     // a state of the knapsack problem is `KnapsackState`. Hence the state-space
//!     // of the problem consists of the set of KnapsackStates that can be represented
//!     type State = KnapsackState;
//! 
//!     // This method is used to tell the number of variables in the knapsack instance
//!     // you are willing to solve. In the literature, it is often referred to as 'N'.
//!     fn nb_variables(&self) -> usize {
//!         self.profit.len()
//!     }
//!     // This method returns the initial state of your DP model. In our case, that
//!     // is nothing but an empty sack.
//!     fn initial_state(&self) -> Self::State {
//!         KnapsackState{ depth: 0, capacity: self.capacity }
//!     }
//!     // This method returns the initial value of the DP. This value accounts for the
//!     // constant factors that have an impact on the final objective. In the case of
//!     // the knapsack, when the sack is empty, the objective value is 0. Hence the
//!     // initial value is zero as well.
//!     fn initial_value(&self) -> isize {
//!         0
//!     }
//!     // This method implements a transition in the DP model. It yields a new sate
//!     // based on a decision (affectation of a value to a variable) which is made from
//!     // a given state. 
//!     fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
//!         let mut ret = state.clone();
//!         ret.depth  += 1;
//!         if dec.value == TAKE_IT { 
//!             ret.capacity -= self.weight[dec.variable.id()] 
//!         }
//!         ret
//!     }
//!     // This method is analogous to the transition function. But instead to returning
//!     // the next state when a decision is made, it returns the "cost", that is the 
//!     // impact of making that decision on the objective function.
//!     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
//!         self.profit[dec.variable.id()] as isize * dec.value
//!     }
//!     // This method is used to determine the order in which the variables will be branched
//!     // on when solving the knapsack. In this case, we implement a basic scheme telling that
//!     // the variables are selected in order (0, 1, 2, ... , N).
//!     fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
//!         let n = self.nb_variables();
//!         if depth < n {
//!             Some(Variable(depth))
//!         } else {
//!             None
//!         }
//!     }
//!     // If you followed this example until now, you might be surprised not to have seen
//!     // any mention of the domain of the variables. Search no more. This function is 
//!     // designed to perform a call to the callback `f` for each possible decision regarding
//!     // a given state and variable. In other words, it calls the callback `f` for each value
//!     // in the domain of `variable` given that the current state is `state`.
//!     fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
//!     {
//!         if state.capacity >= self.weight[variable.id()] {
//!             f.apply(Decision { variable, value: TAKE_IT });
//!             f.apply(Decision { variable, value: LEAVE_IT_OUT });
//!         } else {
//!             f.apply(Decision { variable, value: LEAVE_IT_OUT });
//!         }
//!     }
//! }
//! ```
//!
//! #### Define a Relaxation
//! The relaxation we will define is probably the simplest you can think of.
//! When one needs to define a new state to replace those exceeding the maximum
//! width of the MDD, we will simply keep the state with the maximum capacity
//! as it enables at least all the possibly behaviors feasible with lesser capacities.
//!
//! Optionally, we could also implement a rough upper bound estimator for our
//! problem in the relaxation. However, we wont do it in this minimalistic
//! example since the framework provides you with a default implementation.
//! If you were to override the default implementation you would need to
//! implement the `fast_upper_bound()` method of the `Relaxation` trait.
//!
//! ```
//! # use ddo::*;
//! #
//! # #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//! # struct KnapsackState {
//! #     depth: usize,
//! #     capacity: usize
//! # }
//! # 
//! # struct Knapsack {
//! #     capacity: usize,
//! #     profit: Vec<usize>,
//! #     weight: Vec<usize>,
//! # }
//! # 
//! # const TAKE_IT: isize = 1;
//! # const LEAVE_IT_OUT: isize = 0;
//! # 
//! # impl Problem for Knapsack {
//! #     type State = KnapsackState;
//! #     fn nb_variables(&self) -> usize {
//! #         self.profit.len()
//! #     }
//! #     fn initial_state(&self) -> Self::State {
//! #         KnapsackState{ depth: 0, capacity: self.capacity }
//! #     }
//! #     fn initial_value(&self) -> isize {
//! #         0
//! #     }
//! #     fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
//! #         let mut ret = state.clone();
//! #         ret.depth  += 1;
//! #         if dec.value == TAKE_IT { 
//! #             ret.capacity -= self.weight[dec.variable.id()] 
//! #         }
//! #         ret
//! #     }
//! #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
//! #         self.profit[dec.variable.id()] as isize * dec.value
//! #     }
//! #     fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
//! #         let n = self.nb_variables();
//! #         if depth < n {
//! #             Some(Variable(depth))
//! #         } else {
//! #             None
//! #         }
//! #     }
//! #     fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
//! #     {
//! #         if state.capacity >= self.weight[variable.id()] {
//! #             f.apply(Decision { variable, value: TAKE_IT });
//! #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
//! #         } else {
//! #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
//! #         }
//! #     }
//! # }
//! struct KPRelax<'a>{pb: &'a Knapsack}
//! impl Relaxation for KPRelax<'_> {
//!     // The type of states which this relaxation operates on is KnapsackState.
//!     // Just like the Problem definition which told us that its state spaces
//!     // consisted of all the possible KnapsackStates.
//!     type State = KnapsackState;
//!     
//!     // This method creates and returns a new KnapsackState that will stand for
//!     // all the states returned by the 'states' iterator. The newly created state
//!     // will replace all these nodes in a relaxed DD that has too many nodes.
//!     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
//!         states.max_by_key(|node| node.capacity).copied().unwrap()
//!     }
//!     // This method is used to offset a portion of the cost that would be lost in
//!     // the merge operations towards the edges entering the merged node. It is important
//!     // to know this method exists, even though most of the time, you will simply return
//!     // the cost of the relaxed edge (that is you wont offset any cost on the entering
//!     // edges as that wont be required by your relaxation. But is some -- infrequent -- cases
//!     // your model will require that you do something smart here). 
//!     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
//!         cost
//!     }
//! }
//! ```
//!
//! ### State Ranking
//! There is a third piece of information which you will need to pass on to the 
//! solver before being able to use ddo. This third bit of information is called
//! a `StateRanking` and it is an heuristic used to discriminate the most promising 
//! states from the least promising one. That way, the solver isn't blind when it
//! needs to decide which nodes to delete or merge as it compiles restricted and
//! relaxed DDs for you.
//! 
//! For instance, in the case of the knapsack, when all else is equal, you will 
//! obviously prefer that the solver leaves the states with a higher remaining
//! capacity untouched and merge or delete the others.
//! ```
//! use ddo::*;
//! #
//! # #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//! # struct KnapsackState {
//! #     depth: usize,
//! #     capacity: usize
//! # }
//! # 
//! struct KPRanking;
//! impl StateRanking for KPRanking {
//!     // This associated type has the same meaning as in the problem and 
//!     // relaxation definitions.
//!     type State = KnapsackState;
//!     
//!     // It compares two states and returns an ordering. Greater means that
//!     // state a is preferred over state b. Less means that state b should be 
//!     // preferred over state a. And Equals means you don't care.
//!     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
//!         a.capacity.cmp(&b.capacity)
//!     }
//! }
//! ```
//! 
//! # Instantiate your Solver
//! As soon as you have defined a problem and relaxation and state ranking, you are 
//! good to go. The only thing you still need to do is to write your main method and 
//! spin your solver to solve actual problems. Here is how you would do it.
//!
//! ```
//! # use ddo::*;
//! # use std::sync::Arc;
//! #
//! # #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//! # pub struct KnapsackState {
//! #     depth: usize,
//! #     capacity: usize
//! # }
//! # 
//! # struct Knapsack {
//! #     capacity: usize,
//! #     profit: Vec<usize>,
//! #     weight: Vec<usize>,
//! # }
//! # 
//! # const TAKE_IT: isize = 1;
//! # const LEAVE_IT_OUT: isize = 0;
//! # 
//! # impl Problem for Knapsack {
//! #     type State = KnapsackState;
//! #     fn nb_variables(&self) -> usize {
//! #         self.profit.len()
//! #     }
//! #     fn initial_state(&self) -> Self::State {
//! #         KnapsackState{ depth: 0, capacity: self.capacity }
//! #     }
//! #     fn initial_value(&self) -> isize {
//! #         0
//! #     }
//! #     fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
//! #         let mut ret = state.clone();
//! #         ret.depth  += 1;
//! #         if dec.value == TAKE_IT { 
//! #             ret.capacity -= self.weight[dec.variable.id()] 
//! #         }
//! #         ret
//! #     }
//! #     fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
//! #         self.profit[dec.variable.id()] as isize * dec.value
//! #     }
//! #     fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
//! #         let n = self.nb_variables();
//! #         if depth < n {
//! #             Some(Variable(depth))
//! #         } else {
//! #             None
//! #         }
//! #     }
//! #     fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
//! #     {
//! #         if state.capacity >= self.weight[variable.id()] {
//! #             f.apply(Decision { variable, value: TAKE_IT });
//! #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
//! #         } else {
//! #             f.apply(Decision { variable, value: LEAVE_IT_OUT });
//! #         }
//! #     }
//! # }
//! # struct KPRelax<'a>{pb: &'a Knapsack}
//! # impl Relaxation for KPRelax<'_> {
//! #     type State = KnapsackState;
//! # 
//! #     fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
//! #         states.max_by_key(|node| node.capacity).copied().unwrap()
//! #     }
//! #     fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
//! #         cost
//! #     }
//! # }
//! # 
//! # struct KPRanking;
//! # impl StateRanking for KPRanking {
//! #     type State = KnapsackState;
//! #     
//! #     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
//! #         a.capacity.cmp(&b.capacity)
//! #     }
//! # }
//! # pub struct KPDominance;
//! # impl Dominance for KPDominance {
//! #     type State = KnapsackState;
//! #     type Key = usize;
//! #     fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
//! #        Some(state.depth)
//! #     }
//! #     fn nb_dimensions(&self, _state: &Self::State) -> usize {
//! #         1
//! #     }
//! #     fn get_coordinate(&self, state: &Self::State, _: usize) -> isize {
//! #         state.capacity as isize
//! #     }
//! #     fn use_value(&self) -> bool {
//! #         true
//! #     }
//! # }
//! 
//! // 1. Create an instance of our knapsack problem
//! let problem = Knapsack {
//!     capacity: 50,
//!     profit  : vec![60, 100, 120],
//!     weight  : vec![10,  20,  30]
//! };
//! 
//! // 2. Create a relaxation of the problem
//! let relaxation = KPRelax{pb: &problem};
//! 
//! // 3. Create a ranking to discriminate the promising and uninteresting states
//! let heuristic = KPRanking;
//! 
//! // 4. Define the policy you will want to use regarding the maximum width of the DD
//! let width = FixedWidth(100); // here we mean max 100 nodes per layer
//! 
//! // 5. Add a dominance relation checker
//! let dominance = SimpleDominanceChecker::new(KPDominance);
//! 
//! // 6. Decide of a cutoff heuristic (if you don't want to let the solver run for ever)
//! let cutoff = NoCutoff; // might as well be a TimeBudget (or something else)
//! 
//! // 7. Create the solver fringe
//! let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
//!  
//! // 8. Instantiate your solver
//! let mut solver = DefaultSolver::new(
//!       &problem, 
//!       &relaxation, 
//!       &heuristic, 
//!       &width, 
//!       &dominance,
//!       &cutoff, 
//!       &mut fringe);
//! 
//! // 9. Maximize your objective function
//! // the outcome provides the value of the best solution that was found for
//! // the problem (if one was found) along with a flag indicating whether or
//! // not the solution was proven optimal. Hence an unsatisfiable problem
//! // would have `outcome.best_value == None` and `outcome.is_exact` true.
//! // The `is_exact` flag will only be false if you explicitly decide to stop
//! // searching with an arbitrary cutoff.
//! let outcome = solver.maximize();
//! // The best solution (if one exist) is retrieved with
//! let solution = solver.best_solution();
//!
//! // 10. Do whatever you like with the optimal solution.
//! assert_eq!(Some(220), outcome.best_value);
//! println!("Solution");
//! for decision in solution.unwrap().iter() {
//!     if decision.value == 1 {
//!         println!("{}", decision.variable.id());
//!     }
//! }
//! ```
//!
//! ## Going further / Getting a grasp on the codebase
//! The easiest way to get your way around with DDO is probably to start
//! exploring the available APIs and then to move to the exploration of the
//! examples. (Or the other way around, that's really up to you !).
//! For the exploration of the APIs, you are encouraged to start with the types
//! `ddo::Problem` and `ddo::Relaxation` which defines the core abstractions
//! you will need to implement. After that, it is also interesting to have a
//! look at the various heuristics available and the configuration options you
//! can use when customizing the behavior of your solver and mdd. That should 
//! get you covered and you should be able to get a deep understanding of how 
//! to use our library.
//!
//! ## Citing DDO
//! If you use DDO, or find it useful for your purpose (research, teaching,
//! business, ...) please cite:
//! ```plain
//! @misc{gillard:20:ddo,
//!     author       = {Xavier Gillard, Pierre Schaus, Vianney Coppé},
//!     title        = {Ddo, a generic and efficient framework for MDD-based optimization},
//!     howpublished = {IJCAI-20},
//!     year         = {2020},
//!     note         = {Available from \url{https://github.com/xgillard/ddo}},
//! }
//! ```

// I don't want to emit a lint warning because of the main method appearing
// in the crate documentation. It is specifically the purpose of that doc to
// show an example (including the main) of how to use the ddo library.
#![allow(clippy::needless_doctest_main)]

mod common;
mod abstraction;
mod implementation;

pub use common::*;
pub use abstraction::*;
pub use implementation::*;