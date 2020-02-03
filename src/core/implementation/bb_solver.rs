use std::rc::Rc;
use crate::core::abstraction::dp::{Problem, Relaxation, Decision, VarSet};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic, LoadVars, NodeOrdering};
use crate::core::abstraction::mdd::{Node, MDD};
use crate::core::abstraction::solver::Solver;
use std::hash::Hash;
use std::marker::PhantomData;
use crate::core::implementation::pooled_mdd::{PooledMDD, PooledNode};
use binary_heap_plus::BinaryHeap;
use crate::core::implementation::heuristics::FromLongestPath;

pub struct BBSolver<T, PB, RLX, VS, WDTH, NS, BO, VARS = FromLongestPath>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T, PooledNode<T>>,
          WDTH : WidthHeuristic<T, PooledNode<T>>,
          NS   : NodeOrdering<T, PooledNode<T>>,
          BO   : NodeOrdering<T, PooledNode<T>>,
          VARS : LoadVars<T, PB, PooledNode<T>> {

    pb           : Rc<PB>,
    mdd          : PooledMDD<T, PB, RLX, VS, WDTH, NS>,

    fringe       : BinaryHeap<PooledNode<T>, BO>,
    load_vars    : VARS,

    pub explored : usize,
    pub best_ub  : i32,
    pub best_lb  : i32,
    pub best_node: Option<PooledNode<T>>,
    pub best_sol : Option<Vec<Decision>>,
    pub verbosity: u8,
    phantom      : PhantomData<T>
}

impl <T, PB, RLX, VS, WDTH, NS, BO, VARS> BBSolver<T, PB, RLX, VS, WDTH, NS, BO, VARS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T, PooledNode<T>>,
          WDTH : WidthHeuristic<T, PooledNode<T>>,
          NS   : NodeOrdering<T, PooledNode<T>>,
          BO   : NodeOrdering<T, PooledNode<T>>,
          VARS : LoadVars<T, PB, PooledNode<T>> {
    pub fn new(pb: Rc<PB>,
               relax: RLX,
               vs: VS,
               width: WDTH,
               ns: NS,
               bo: BO,
               load_vars: VARS) -> BBSolver<T, PB, RLX, VS, WDTH, NS, BO, VARS> {
        BBSolver {
            pb: Rc::clone(&pb),
            mdd: PooledMDD::new(Rc::clone(&pb), relax, vs, width, ns),
            fringe: BinaryHeap::from_vec_cmp(vec![], bo),
            load_vars,
            explored: 0,
            best_ub: std::i32::MAX,
            best_lb: std::i32::MIN,
            best_node: None,
            best_sol: None,
            verbosity: 0,
            phantom: PhantomData
        }
    }
}

impl <T, PB, RLX, VS, WDTH, NS, BO, VARS> Solver for BBSolver<T, PB, RLX, VS, WDTH, NS, BO, VARS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T, PooledNode<T>>,
          WDTH : WidthHeuristic<T, PooledNode<T>>,
          NS   : NodeOrdering<T, PooledNode<T>>,
          BO   : NodeOrdering<T, PooledNode<T>>,
          VARS : LoadVars<T, PB, PooledNode<T>> {

    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        let root = PooledNode::new(
            self.pb.initial_state(),
            self.pb.initial_value(),
            None,
            true
        );

        // 0. Initial relaxation:
        self.explored = 1;
        self.mdd.relaxed(VarSet::all(self.pb.nb_vars()), &root, self.best_lb);
        if self.mdd.is_exact() {
            if self.mdd.best_value() > self.best_lb {
                self.best_lb   = self.mdd.best_value();
                self.best_node = self.mdd.best_node().clone();
                self.best_sol  = Some(self.best_node.as_ref().unwrap().longest_path());
            }
            if self.verbosity >= 1 {
                println!("Immediate {} ", self.best_lb);
            }
            return (self.best_lb, &self.best_sol);
        } else {
            for node in self.mdd.exact_cutset() {
                self.fringe.push(node.clone());
            }
        }
        
        while !self.fringe.is_empty() {
            let node = self.fringe.pop().unwrap();

            // Nodes are sorted on UB as first criterion. It can be updated
            // whenever we encounter a tighter value
            if node.get_ub() < self.best_ub {
                self.best_ub = node.get_ub();
            }

            // We just proved optimality, Yay !
            if self.best_lb == self.best_ub {
                break;
            }

            // Skip if this node cannot improve the current best solution
            if node.get_ub() < self.best_lb {
                continue;
            }

            self.explored += 1;
            if self.verbosity >= 1 && self.explored % 100 == 0 {
                println!("Explored {}, LB {}, UB {}, Fringe sz {}", self.explored, self.best_lb, node.get_ub(), self.fringe.len());
            }

            let vars = self.load_vars.variables(self.pb.as_ref(), &node);

            // 1. RESTRICTION
            self.mdd.restricted(vars.clone(),&node, self.best_lb);
            if self.mdd.best_value() > self.best_lb {
                self.best_lb   = self.mdd.best_value();
                self.best_node = self.mdd.best_node().clone();
            }
            if self.mdd.is_exact() {
                continue;
            }

            // 2. RELAXATION
            self.mdd.relaxed(vars, &node, self.best_lb);
            if self.mdd.is_exact() {
                if self.mdd.best_value() > self.best_lb {
                    self.best_lb   = self.mdd.best_value();
                    self.best_node = self.mdd.best_node().clone();
                }
            } else {
                for branch in self.mdd.exact_cutset() {
                    let branch_ub = self.best_ub.min(branch.get_ub());

                    if branch.get_ub() > self.best_lb {
                        self.fringe.push(PooledNode {
                            state   : branch.state.clone(),
                            is_exact: true,
                            lp_len  : branch.lp_len,
                            lp_arc  : branch.lp_arc.clone(),
                            ub      : branch_ub
                        });
                    }
                }
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