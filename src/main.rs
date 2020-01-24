extern crate rust_mdd_solver;

use rust_mdd_solver::examples::misp::misp::Misp;
use rust_mdd_solver::core::implem::pooled_mdd::{PooledMDD, PooledNode};
use std::rc::Rc;
use rust_mdd_solver::examples::misp::relax::MispRelax;
use rust_mdd_solver::core::abstraction::heuristics::VariableHeuristic;
use bitset_fixed::BitSet;
use rust_mdd_solver::core::abstraction::dp::{Variable, Problem};
use rust_mdd_solver::core::abstraction::mdd::{MDD, Node};
use rust_mdd_solver::core::implem::heuristics::FixedWidth;
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less, Equal};
use std::ops::Not;
use std::time::SystemTime;
use rust_mdd_solver::core::utils::BitSetIter;

struct ActiveLexOrder {
    pb: Rc<Misp>
}
impl ActiveLexOrder {
    pub fn new(pb: Rc<Misp>) -> ActiveLexOrder {
        ActiveLexOrder{pb}
    }
}
impl VariableHeuristic<BitSet, PooledNode<BitSet>> for ActiveLexOrder {
    fn next_var(&self, dd: &dyn MDD<BitSet, PooledNode<BitSet>>, vars: &BitSet) -> Option<Variable> {
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

        for i in 0..self.pb.nb_vars() {
            if vars[i] {
                return Some(Variable(i))
            }
        }
        None
    }
}

fn misp_min_lp(a: &Rc<PooledNode<BitSet>>, b: &Rc<PooledNode<BitSet>>) -> Ordering {
    match a.get_lp_len().cmp(&b.get_lp_len()) {
        Ordering::Greater => Greater,
        Ordering::Less    => Less,
        Ordering::Equal   => a.get_state().cmp(b.get_state())
    }
}

fn main() {
    let fname = "/Users/user/Desktop/mdd_cpp/instances/keller4.clq";

    let misp = Rc::new(Misp::from_file(fname));
    let relax = Rc::new(MispRelax::new(Rc::clone(&misp)));
    let vs = Rc::new(ActiveLexOrder::new(Rc::clone(&misp)));
    let w = Rc::new(FixedWidth(100));

    let inst = Rc::clone(&misp);
    let mut mdd = PooledMDD::new(inst, relax, vs, w, misp_min_lp);


    let vars = BitSet::new(misp.nb_vars()).not();
    let root = Rc::new(PooledNode::new(vars.clone(), misp.initial_value(), None,true));

    let start = SystemTime::now();
    mdd.restricted(vars.clone(), Rc::clone(&root), misp.initial_value());

    let lb = mdd.best_value();
    mdd.relaxed(vars, root, lb);
    let end = SystemTime::now();
    println!("Tm {:?}", end.duration_since(start).unwrap());

    println!("LB = {}", lb);
    println!("UB = {}", mdd.best_value());
    println!("Cutset = {}", mdd.exact_cutset().len());
}