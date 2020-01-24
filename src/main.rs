extern crate rust_mdd_solver;

use rust_mdd_solver::examples::misp::misp::Misp;
use rust_mdd_solver::core::implem::pooled_mdd::{PooledMDD, PooledNode};
use std::rc::Rc;
use rust_mdd_solver::examples::misp::relax::MispRelax;
use rust_mdd_solver::core::abstraction::heuristics::VariableHeuristic;
use bitset_fixed::BitSet;
use rust_mdd_solver::core::abstraction::dp::{Variable, Problem, VarSet};
use rust_mdd_solver::core::abstraction::mdd::{MDD, Node};
use rust_mdd_solver::core::implem::heuristics::FixedWidth;
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less, Equal};
use std::time::SystemTime;

struct ActiveLexOrder;

impl ActiveLexOrder {
    pub fn new() -> ActiveLexOrder {
        ActiveLexOrder{}
    }
}
impl VariableHeuristic<BitSet, PooledNode<BitSet>> for ActiveLexOrder {
    fn next_var(&self, _dd: &dyn MDD<BitSet, PooledNode<BitSet>>, vars: &VarSet) -> Option<Variable> {
        /*
        let mut active_vertices = BitSet::new(self.pb.nb_vars());
        for (state, _) in dd.next_layer() {
            active_vertices |= state;
        }

        for v in BitSetIter::new(active_vertices) {
            return Some(Variable(v));
        }

        None
        */

        for x in vars.iter() {
            return Some(x)
        }
        None
    }
}

fn misp_min_lp(a: &Rc<PooledNode<BitSet>>, b: &Rc<PooledNode<BitSet>>) -> Ordering {
    match a.get_lp_len().cmp(&b.get_lp_len()) {
        Ordering::Greater => Greater,
        Ordering::Less    => Less,
        Ordering::Equal   => {
            //a.get_state().cmp(b.get_state())
            let x = a.get_state();
            let y = b.get_state();
            for i in 0..x.size() {
                if x[i] != y[i] {
                    return x[i].cmp(&y[i]);
                }
            }
            Equal
        }
    }
}

fn main() {
    let fname = "/Users/user/Desktop/mdd_cpp/instances/keller4.clq";

    let misp = Rc::new(Misp::from_file(fname));
    let relax = Rc::new(MispRelax::new(Rc::clone(&misp)));
    let vs = Rc::new(ActiveLexOrder::new());
    let w = Rc::new(FixedWidth(100));

    let inst = Rc::clone(&misp);
    let mut mdd = PooledMDD::new(inst, relax, vs, w, misp_min_lp);


    let vars = VarSet::all(misp.nb_vars());
    let root = Rc::new(PooledNode::new(vars.0.clone(), misp.initial_value(), None,true));

    let start = SystemTime::now();
    //mdd.restricted(vars.clone(), Rc::clone(&root), misp.initial_value());

    let lb = 0;//mdd.best_value();
    mdd.relaxed(vars, root, lb);
    let end = SystemTime::now();
    println!("Tm {:?}", end.duration_since(start).unwrap());

    println!("LB = {}", lb);
    println!("UB = {}", mdd.best_value());
    println!("Cutset = {}", mdd.exact_cutset().len());
}