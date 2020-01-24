use std::rc::Rc;
use crate::core::abstraction::dp::{Problem, Relaxation, Decision, VarSet};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{Node, MDD};
use std::hash::Hash;
use std::marker::PhantomData;
use crate::core::implem::pooled_mdd::{PooledMDD, PooledNode};
use std::cmp::Ordering;
use binary_heap_plus::{BinaryHeap, FnComparator};

pub struct Solver<T, NS, BO>
    where T : Clone + Hash + Eq,
          NS: Fn(&Rc<PooledNode<T>>, &Rc<PooledNode<T>>) -> Ordering,
          BO: Clone + Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering {

    pb           : Rc<dyn Problem<T>>,
    mdd          : PooledMDD<T, NS>,

    fringe       : BinaryHeap<PooledNode<T>, FnComparator<BO>>,

    pub explored : usize,
    pub best_ub  : i32,
    pub best_lb  : i32,
    pub best_sol : Option<Vec<Decision>>,
    phantom      : PhantomData<T>
}

impl <T, NS, BO> Solver<T, NS, BO>
    where T : Clone + Hash + Eq + Ord,
          NS: Fn(&Rc<PooledNode<T>>, &Rc<PooledNode<T>>) -> Ordering,
          BO: Clone + Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering {

    pub fn new(pb    : Rc<dyn Problem<T>>,
               relax : Rc<dyn Relaxation<T>>,
               vs    : Rc<dyn VariableHeuristic<T, PooledNode<T>>>,
               width : Rc<dyn WidthHeuristic<T, PooledNode<T>>>,
               ns    : NS,
               bo    : BO) -> Solver<T, NS, BO> {

        Solver {
            pb         : Rc::clone(&pb),
            mdd        : PooledMDD::new(Rc::clone(&pb), relax, vs, width, ns),
            fringe     : BinaryHeap::new_by(bo),
            explored   : 0,
            best_ub    : std::i32::MAX,
            best_lb    : std::i32::MIN,
            best_sol   : None,
            phantom    : PhantomData
        }

    }

    pub fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>) {
        let root = PooledNode::new(
            self.pb.initial_state(),
            self.pb.initial_value(),
            None,
            true
        );

        // 0. Initial relaxation:
        self.explored = 1;
        self.mdd.relaxed(VarSet::all(self.pb.nb_vars()), Rc::new(root), self.best_lb);
        if self.mdd.is_exact() {
            if self.mdd.best_value() > self.best_lb {
                self.best_lb = self.mdd.best_value();
                self.best_sol= Some(self.mdd.longest_path());
            }
            return (self.best_lb, &self.best_sol);
        } else {
            self.best_ub = self.mdd.best_value();
            for branch in self.mdd.exact_cutset() {
                let branch_ub = self.best_ub.min(branch.get_ub());

                if branch.get_ub() > self.best_lb {
                    self.fringe.push(PooledNode {
                        state   : branch.get_state().clone(),
                        is_exact: true,
                        lp_len  : branch.get_lp_len(),
                        lp_arc  : branch.get_lp_arc().clone(),
                        ub      : branch_ub
                    });
                }
            }
        }

        println!("After root : UB {}, Fringe {}", self.best_ub, self.fringe.len());
        while !self.fringe.is_empty() {
            let node = self.fringe.pop().unwrap();
            let node= Rc::new(node);

            // Nodes are sorted on UB as first criterion. It can be updated
            // whenever we encounter a tighter value
            if node.get_ub() < self.best_ub {
                self.best_ub = node.get_ub();
            }

            // Skip if this node cannot improve the current best solution
            if node.get_ub() < self.best_lb {
                continue;
            }

            self.explored += 1;
            if self.explored % 100 == 0 {
                println!("Explored {}, LB {}, UB {}, Fringe sz {}", self.explored, self.best_lb, node.get_ub(), self.fringe.len());
            }

            let mut vars = VarSet::all(self.pb.nb_vars());
            for d in node.longest_path() {
                vars.remove(d.variable);
            }

            // 1. RELAXATION
            self.mdd.relaxed(vars.clone(), Rc::clone(&node), self.best_lb);
            if self.mdd.is_exact() {
                if self.mdd.best_value() > self.best_lb {
                    self.best_lb = self.mdd.best_value();
                }
                continue;
            } else {
                for branch in self.mdd.exact_cutset() {
                    let branch_ub = self.best_ub.min(branch.get_ub());

                    if branch.get_ub() > self.best_lb {
                        self.fringe.push(PooledNode {
                            state   : branch.get_state().clone(),
                            is_exact: true,
                            lp_len  : branch.get_lp_len(),
                            lp_arc  : branch.get_lp_arc().clone(),
                            ub      : branch_ub
                        });
                    }
                }
            }

            // 2. RESTRICTION
            self.mdd.restricted(vars,Rc::clone(&node), self.best_lb);
            if self.mdd.best_value() > self.best_lb {
                self.best_lb = self.mdd.best_value();
            }
        }

        if self.mdd.is_exact() {
            self.best_sol = Some(self.mdd.longest_path());
        }

        // return
        (self.best_lb, &self.best_sol)
    }
}