use std::hash::Hash;

use binary_heap_plus::BinaryHeap;
use compare::Compare;

use crate::core::abstraction::heuristics::LoadVars;
use crate::core::abstraction::mdd::{MDDGenerator, Node, NodeInfo};
use crate::core::abstraction::solver::Solver;
use crate::core::common::Decision;

pub struct BBSolver<T, DDG, BO, VARS>
    where T    : Hash + Eq + Clone,
          DDG  : MDDGenerator<T>,
          BO   : Compare<Node<T>>,
          VARS : LoadVars<T> {

    ddg          : DDG,

    fringe       : BinaryHeap<Node<T>, BO>,
    load_vars    : VARS,

    pub explored : usize,
    pub best_ub  : i32,
    pub best_lb  : i32,
    pub best_node: Option<NodeInfo<T>>,
    pub best_sol : Option<Vec<Decision>>,
    pub verbosity: u8
}

impl <T, DDG, BO, VARS> BBSolver<T, DDG, BO, VARS>
    where T    : Hash + Eq + Clone,
          DDG  : MDDGenerator<T>,
          BO   : Compare<Node<T>>,
          VARS : LoadVars<T> {
    pub fn new(ddg: DDG,
               bo : BO,
               load_vars: VARS) -> BBSolver<T, DDG, BO, VARS> {
        BBSolver {
            ddg,
            fringe: BinaryHeap::from_vec_cmp(vec![], bo),
            load_vars,
            explored: 0,
            best_ub: std::i32::MAX,
            best_lb: std::i32::MIN,
            best_node: None,
            best_sol: None,
            verbosity: 0
        }
    }
}

impl <T, DDG, BO, VARS> Solver for BBSolver<T, DDG, BO, VARS>
    where T    : Hash + Eq + Clone,
          DDG  : MDDGenerator<T>,
          BO   : Compare<Node<T>>,
          VARS : LoadVars<T> {

    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        let root = self.ddg.root();
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

            let vars = self.load_vars.variables(&node);

            // 1. RESTRICTION
            self.ddg.restricted(vars.clone(),&node, self.best_lb);
            if self.ddg.mdd().best_value() > self.best_lb {
                self.best_lb   = self.ddg.mdd().best_value();
                self.best_node = self.ddg.mdd().best_node().clone();
            }
            if self.ddg.mdd().is_exact() {
                continue;
            }

            // 2. RELAXATION
            self.ddg.relaxed(vars, &node, self.best_lb);
            if self.ddg.mdd().is_exact() {
                if self.ddg.mdd().best_value() > self.best_lb {
                    self.best_lb   = self.ddg.mdd().best_value();
                    self.best_node = self.ddg.mdd().best_node().clone();
                }
            } else {
                let best_ub= self.best_ub;
                let best_lb= self.best_lb;
                let fringe = &mut self.fringe;
                let ddg    = &mut self.ddg;
                ddg.consume_cutset(|state, mut info| {
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