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
//! a solver that will solve the problem on one single thread.
use std::hash::Hash;

use binary_heap_plus::BinaryHeap;

use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::common::{Decision, Node, NodeInfo};
use crate::core::implementation::heuristics::MaxUB;

/// This is the structure implementing an single-threaded MDD solver.
pub struct SequentialSolver<T, DD>
    where T: Hash + Eq + Clone, DD: MDD<T> {
    /// This is the MDD which will be used to expand the restricted and relaxed
    /// mdds.
    mdd: DD,
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
    fringe: BinaryHeap<Node<T>, MaxUB>,

    /// This is a counter that tracks the number of nodes that have effectively
    /// been explored. That is, the number of nodes that have been popped from
    /// the fringe, and for which a restricted and relaxed mdd have been developed.
    explored : usize,
    /// This is the value of the current best upper bound (= the upper bound of
    /// the last node that has been popped from the fringe).
    best_ub  : i32,
    /// This is the value of the best known lower bound.
    best_lb  : i32,
    /// If set, this keeps the info about the best solution (the solution that
    /// yielded the `best_lb`, and from which `best_sol` derives).
    best_node: Option<NodeInfo>,
    /// This is the materialization of the best solution that has been idenditied.
    best_sol : Option<Vec<Decision>>,
    /// This is just a configuration parameter to tune the amount of information
    /// logged when solving the problem.
    verbosity: u8
}
// private interface.
impl <T, DD> SequentialSolver<T, DD>
    where T: Hash + Eq + Clone, DD: MDD<T> {
    /// This creates a solver that will find the best solution in the problem
    /// described by the given `mdd` (mdd is not expanded yet). This solver will
    /// return the optimal solution from what would be an exact expansion of `mdd`.
    pub fn new(mdd: DD) -> Self {
        Self::customized(mdd, 0)
    }
    /// This creates a solver that will find the best solution in the problem
    /// described by the given `mdd` (mdd is not expanded yet) and configure that
    /// solver to be more or less verbose.
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
        Self::customized(mdd, verbosity)
    }
    /// This constructor lets you specify all the configurable elements of this
    /// solver as parameters. So far, the functionality of this constructor is
    /// exactly the same as that of `with_verbosity`. The latter is considered
    /// clearer, and thus you are encouraged to use that one.
    ///
    /// This constructor creates a solver that will find the best solution in the
    /// problem described by the given `mdd` (mdd is not expanded yet) and
    /// configure that solver to be more or less verbose.
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
    pub fn customized(mdd: DD, verbosity: u8) -> Self {
        SequentialSolver {
            mdd,
            fringe: BinaryHeap::from_vec_cmp(vec![], MaxUB),
            explored: 0,
            best_ub: std::i32::MAX,
            best_lb: std::i32::MIN,
            best_node: None,
            best_sol: None,
            verbosity
        }
    }
    /// This private method updates the solver's best node and lower bound in
    /// case the best value of the current mdd expansion improves the current
    /// bounds.
    fn maybe_update_best(&mut self) {
        if self.mdd.best_value() > self.best_lb {
            self.best_lb   = self.mdd.best_value();
            self.best_node = self.mdd.best_node().clone();
        }
    }
}

impl <T, DD> Solver for SequentialSolver<T, DD> where T: Hash + Eq + Clone, DD: MDD<T> {
    /// Applies the branch and bound algorithm proposed by Bergman et al. to
    /// solve the problem to optimality.
    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        let root = self.mdd.root();
        self.fringe.push(root);

        while !self.fringe.is_empty() {
            let node = self.fringe.pop().unwrap();

            // Nodes are sorted on UB as first criterion. It can be updated
            // whenever we encounter a tighter value
            if node.info.ub < self.best_ub {
                self.best_ub = node.info.ub;
            }

            // We just proved optimality, Yay !
            if self.best_lb >= self.best_ub {
                break;
            }

            // Skip if this node cannot improve the current best solution
            if node.info.ub < self.best_lb {
                continue;
            }

            self.explored += 1;
            if self.verbosity >= 2 && self.explored % 100 == 0 {
                println!("Explored {}, LB {}, UB {}, Fringe sz {}", self.explored, self.best_lb, node.info.ub, self.fringe.len());
            }

            // 1. RESTRICTION
            self.mdd.restricted(&node, self.best_lb);
            self.maybe_update_best();
            if self.mdd.is_exact() {
                continue;
            }

            // 2. RELAXATION
            self.mdd.relaxed(&node, self.best_lb);
            if self.mdd.is_exact() {
                self.maybe_update_best();
            } else {
                let best_ub= self.best_ub;
                let best_lb= self.best_lb;
                let fringe = &mut self.fringe;
                let mdd    = &mut self.mdd;
                mdd.consume_cutset(|state, mut info| {
                    info.ub = best_ub.min(info.ub);
                    if info.ub > best_lb {
                        fringe.push(Node{state, info});
                    }
                });
            }
        }

        if let Some(bn) = &self.best_node {
            self.best_sol = Some(bn.longest_path());
        }

        // return
        if self.verbosity >= 1 {
            println!("Final {}, Explored {}", self.best_lb, self.explored);
        }
        (self.best_lb, &self.best_sol)
    }
}