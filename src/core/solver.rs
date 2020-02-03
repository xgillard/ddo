use std::rc::Rc;
use crate::core::abstraction::dp::{Problem, Relaxation, Decision, VarSet};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{Node, MDD};
use std::hash::Hash;
use std::marker::PhantomData;
use crate::core::implem::pooled_mdd::{PooledMDD, PooledNode};
use std::cmp::Ordering;
use binary_heap_plus::{BinaryHeap, FnComparator};

pub trait LoadVars<T>
    where T: Hash + Clone + Eq {
    fn variables(&self, pb: &dyn Problem<T>, node: &PooledNode<T>) -> VarSet;
}

pub struct FromLongestPath;

impl <T> LoadVars<T> for FromLongestPath
    where T: Hash + Clone + Eq {

    fn variables(&self, pb: &dyn Problem<T>, node: &PooledNode<T>) -> VarSet {
        let mut vars = VarSet::all(pb.nb_vars());
        for d in node.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}

pub struct FromFunction<T, F>
    where T: Hash + Clone + Eq,
          F: Fn(&dyn Problem<T>, &PooledNode<T>) -> VarSet {
    func: F,
    phantom: PhantomData<T>
}
impl <T, F> FromFunction<T, F>
    where T: Hash + Clone + Eq,
          F: Fn(&dyn Problem<T>, &PooledNode<T>) -> VarSet {

    pub fn new(func: F) -> FromFunction<T, F> {
        FromFunction{func, phantom: PhantomData}
    }
}
impl <T, F> LoadVars<T> for FromFunction<T, F>
    where T: Hash + Clone + Eq,
          F: Fn(&dyn Problem<T>, &PooledNode<T>) -> VarSet {

    fn variables(&self, pb: &dyn Problem<T>, node: &PooledNode<T>) -> VarSet {
        (self.func)(pb, node)
    }
}

pub struct Solver<T, PB, RLX, VS, WDTH, NS, BO, VARS = FromLongestPath>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T, PooledNode<T>>,
          WDTH : WidthHeuristic<T, PooledNode<T>>,
          NS   : Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering,
          BO   : Clone + Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering,
          VARS : LoadVars<T> {

    pb           : Rc<PB>,
    mdd          : PooledMDD<T, PB, RLX, VS, WDTH, NS>,

    fringe       : BinaryHeap<PooledNode<T>, FnComparator<BO>>,
    load_vars    : VARS,

    pub explored : usize,
    pub best_ub  : i32,
    pub best_lb  : i32,
    pub best_node: Option<PooledNode<T>>,
    pub best_sol : Option<Vec<Decision>>,
    pub verbosity: u8,
    phantom      : PhantomData<T>
}

impl <T, PB, RLX, VS, WDTH, NS, BO, VARS> Solver<T, PB, RLX, VS, WDTH, NS, BO, VARS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T, PooledNode<T>>,
          WDTH : WidthHeuristic<T, PooledNode<T>>,
          NS   : Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering,
          BO   : Clone + Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering,
          VARS : LoadVars<T> {

    pub fn new(pb       : Rc<PB>,
               relax    : RLX,
               vs       : VS,
               width    : WDTH,
               ns       : NS,
               bo       : BO,
               load_vars: VARS) -> Solver<T, PB, RLX, VS, WDTH, NS, BO, VARS> {

        Solver {
            pb         : Rc::clone(&pb),
            mdd        : PooledMDD::new(Rc::clone(&pb), relax, vs, width, ns),
            fringe     : BinaryHeap::new_by(bo),
            load_vars,
            explored   : 0,
            best_ub    : std::i32::MAX,
            best_lb    : std::i32::MIN,
            best_node  : None,
            best_sol   : None,
            verbosity  : 0,
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