use std::hash::Hash;

use binary_heap_plus::BinaryHeap;

use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::common::{Decision, Node, NodeInfo};
use crate::core::implementation::heuristics::MaxUB;

pub struct SequentialSolver<T, DD>
    where T: Hash + Eq + Clone, DD: MDD<T> {

    mdd          : DD,
    fringe       : BinaryHeap<Node<T>, MaxUB>,

    pub explored : usize,
    pub best_ub  : i32,
    pub best_lb  : i32,
    pub best_node: Option<NodeInfo>,
    pub best_sol : Option<Vec<Decision>>,
    pub verbosity: u8
}

impl <T, DD> SequentialSolver<T, DD>
    where T: Hash + Eq + Clone, DD: MDD<T> {
    pub fn new(mdd: DD) -> Self {
        Self::customized(mdd, 0)
    }
    pub fn with_verbosity(mdd: DD, verbosity: u8) -> Self {
        Self::customized(mdd, verbosity)
    }
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

    fn maybe_update_best(&mut self) {
        if self.mdd.best_value() > self.best_lb {
            self.best_lb   = self.mdd.best_value();
            self.best_node = self.mdd.best_node().clone();
        }
    }
}

impl <T, DD> Solver for SequentialSolver<T, DD> where T: Hash + Eq + Clone, DD: MDD<T> {

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