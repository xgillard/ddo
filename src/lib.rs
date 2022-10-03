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
//! problem in terms of a dynamic program. Here, the state of a node, is nothing
//! more than an unsigned integer (usize). That unsigned integer represents the
//! remaining capacity of our sack. To do so, you define your own structure and
//! make sure it implements the `Problem<usize>` trait.
//! ```
//! # use ddo::*;
//! #
//! #[derive(Debug, Clone)]
//! struct Knapsack {
//!     capacity: usize,
//!     profit  : Vec<usize>,
//!     weight  : Vec<usize>
//! }
//! impl Problem<usize> for Knapsack {
//!     fn nb_vars(&self) -> usize {
//!         self.profit.len()
//!     }
//!     fn domain_of<'a>(&self, state: &'a usize, var: Variable) ->Domain<'a> {
//!         if *state >= self.weight[var.id()] {
//!             vec![0, 1].into()
//!         } else {
//!             vec![0].into()
//!         }
//!     }
//!     fn initial_state(&self) -> usize {
//!         self.capacity
//!     }
//!     fn initial_value(&self) -> isize {
//!         0
//!     }
//!     fn transition(&self, state: &usize, _vars: &VarSet, dec: Decision) -> usize {
//!         state - (self.weight[dec.variable.id()] * dec.value as usize)
//!     }
//!     fn transition_cost(&self, _state: &usize, _vars: &VarSet, dec: Decision) -> isize {
//!         self.profit[dec.variable.id()] as isize * dec.value
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
//! implement the `estimate()` method of the `Relaxation` trait.
//!
//! ```
//! # use ddo::*;
//! # 
//! #[derive(Debug, Clone)]
//! struct KPRelax;
//! impl Relaxation<usize> for KPRelax {
//!     /// To merge a given selection of states (capacities) we will keep the
//!     /// maximum capacity. This is an obvious relaxation as it allows us to
//!     /// put more items in the sack.
//!     fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
//!         // the selection is guaranteed to have at least one state so using
//!         // unwrap after max to get rid of the wrapping 'Option' is perfectly safe.
//!         *states.max().unwrap()
//!     }
//!     /// When relaxing (merging) the states, we did not run into the risk of
//!     /// possibly decreasing the maximum objective value reachable from the
//!     /// components of the merged node. Hence, we dont need to do anything
//!     /// when relaxing the edge. Still, if we wanted to, we could chose to
//!     /// return an higher value.
//!     fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, d: Decision, cost: isize) -> isize {
//!         cost
//!     }
//! }
//! ```
//!
//! # Instanciate your Solver
//! As soon as you have defined a problem and relaxation, you are good to go.
//! The only thing you still need to do is to write your main method and spin
//! your solver to solve actual problems. Here is how you would do it.
//!
//! ```
//! # use ddo::*;
//! #
//! # #[derive(Debug, Clone)]
//! # struct Knapsack {
//! #     capacity: usize,
//! #     profit  : Vec<usize>,
//! #     weight  : Vec<usize>
//! # }
//! # impl Problem<usize> for Knapsack {
//! #     fn nb_vars(&self) -> usize {
//! #         self.profit.len()
//! #     }
//! #     fn domain_of<'a>(&self, state: &'a usize, var: Variable) ->Domain<'a> {
//! #         if *state >= self.weight[var.id()] {
//! #             vec![0, 1].into()
//! #         } else {
//! #             vec![0].into()
//! #         }
//! #     }
//! #     fn initial_state(&self) -> usize {
//! #         self.capacity
//! #     }
//! #     fn initial_value(&self) -> isize {
//! #         0
//! #     }
//! #     fn transition(&self, state: &usize, _vars: &VarSet, dec: Decision) -> usize {
//! #         state - (self.weight[dec.variable.id()] * dec.value as usize)
//! #     }
//! #     fn transition_cost(&self, _state: &usize, _vars: &VarSet, dec: Decision) -> isize {
//! #         self.profit[dec.variable.id()] as isize * dec.value
//! #     }
//! # }
//! #
//! # #[derive(Debug, Clone)]
//! # struct KPRelax;
//! # impl Relaxation<usize> for KPRelax {
//! #     /// To merge a given selection of states (capacities) we will keep the
//! #     /// maximum capacity. This is an obvious relaxation as it allows us to
//! #     /// put more items in the sack.
//! #     fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
//! #         // the selection is guaranteed to have at least one state so using
//! #         // unwrap after max to get rid of the wrapping 'Option' is perfectly safe.
//! #         *states.max().unwrap()
//! #     }
//! #     /// When relaxing (merging) the states, we did not run into the risk of
//! #     /// possibly decreasing the maximum objective value reachable from the
//! #     /// components of the merged node. Hence, we dont need to do anything
//! #     /// when relaxing the edge. Still, if we wanted to, we could chose to
//! #     /// return an higher value.
//! #     fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, d: Decision, cost: isize) -> isize {
//! #         cost
//! #     }
//! # }
//! // 1. Create an instance of our knapsack problem
//! let problem = Knapsack {
//!     capacity: 50,
//!     profit  : vec![60, 100, 120],
//!     weight  : vec![10,  20,  30]
//! };
//! // 2. Build an MDD for the given problem and relaxation
//! let mdd = mdd_builder(&problem, KPRelax).into_deep();
//! // 3. Create a parllel solver on the basis of this MDD (this is how
//! //    you can specify the MDD implementation you wish to use to develop
//! //    the relaxed and restricted MDDs).
//! let mut solver = ParallelSolver::new(mdd);
//! // 4. Maximize your objective function
//! // the outcome provides the value of the best solution that was found for
//! // the problem (if one was found) along with a flag indicating whether or
//! // not the solution was proven optimal. Hence an unsatisfiable problem
//! // would have `outcome.best_value == None` and `outcome.is_exact` true.
//! // The `is_exact` flag will only be false if you explicitly decide to stop
//! // searching with an arbitrary cutoff.
//! let outcome    = solver.maximize();
//! // The best solution (if one exist) is retrieved with
//! let solution   = solver.best_solution();
//!
//! // 5. Do whatever you like with the optimal solution.
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
//! look at the various heuristics availble and the configuration options you
//! can use when cutomizing the behavior of your solver and mdd. That should 
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