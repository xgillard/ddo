use std::hash::Hash;

use binary_heap_plus::BinaryHeap;
use compare::Compare;

use crate::core::abstraction::dp::{Problem, ProblemOps};
use crate::core::abstraction::heuristics::LoadVars;
use crate::core::abstraction::mdd::{MDDGenerator, Node};
use crate::core::abstraction::solver::Solver;
use crate::core::common::Decision;
use crate::core::implementation::heuristics::FromLongestPath;

pub struct BBSolver<T, PB, DDG, BO, VARS = FromLongestPath>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          DDG  : MDDGenerator<T>,
          BO   : Compare<Node<T>>,
          VARS : LoadVars<T, PB> {

    pb           : PB,
    ddg          : DDG,

    fringe       : BinaryHeap<Node<T>, BO>,
    load_vars    : VARS,

    pub explored : usize,
    pub best_ub  : i32,
    pub best_lb  : i32,
    pub best_node: Option<Node<T>>,
    pub best_sol : Option<Vec<Decision>>,
    pub verbosity: u8
}

impl <T, PB, DDG, BO, VARS> BBSolver<T, PB, DDG, BO, VARS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          DDG  : MDDGenerator<T>,
          BO   : Compare<Node<T>>,
          VARS : LoadVars<T, PB> {
    pub fn new(pb : PB,
               ddg: DDG,
               bo : BO,
               load_vars: VARS) -> BBSolver<T, PB, DDG, BO, VARS> {
        BBSolver {
            pb,
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

impl <T, PB, DDG, BO, VARS> Solver for BBSolver<T, PB, DDG, BO, VARS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          DDG  : MDDGenerator<T>,
          BO   : Compare<Node<T>>,
          VARS : LoadVars<T, PB> {

    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        let root = self.pb.root_node();
        self.fringe.push(root);
        
        while !self.fringe.is_empty() {
            let node = self.fringe.pop().unwrap();

            // Nodes are sorted on UB as first criterion. It can be updated
            // whenever we encounter a tighter value
            if node.get_ub() < self.best_ub {
                self.best_ub = node.get_ub();
            }

            // We just proved optimality, Yay !
            if self.best_lb >= self.best_ub {
                break;
            }

            // Skip if this node cannot improve the current best solution
            if node.get_ub() < self.best_lb {
                continue;
            }

            self.explored += 1;
            if self.verbosity >= 2 && self.explored % 100 == 0 {
                println!("Explored {}, LB {}, UB {}, Fringe sz {}", self.explored, self.best_lb, node.get_ub(), self.fringe.len());
            }

            let vars = self.load_vars.variables(&self.pb, &node);

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
                ddg.for_each_cutset_node(|branch| {
                    branch.ub = best_ub.min(branch.get_ub());
                    if branch.ub > best_lb {
                        fringe.push(branch.clone());
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