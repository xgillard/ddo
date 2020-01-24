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
use std::cmp::Ordering::{Greater, Less};
use std::ops::Not;

struct ActiveLexOrder {
    pb: Rc<Misp>
}
impl ActiveLexOrder {
    pub fn new(pb: Rc<Misp>) -> ActiveLexOrder {
        ActiveLexOrder{pb}
    }
}
impl VariableHeuristic<BitSet, PooledNode<BitSet>> for ActiveLexOrder {
    fn next_var(&self, _dd: &dyn MDD<BitSet, PooledNode<BitSet>>, vars: &BitSet) -> Variable {
        /*
        let mut active_vertices = BitSet::new(self.pb.nb_vars());
        for n in dd.current_layer() {
            active_vertices |= n.get_state();
        }
        */
        for i in 0..self.pb.nb_vars() {
            if vars[i] {
                return Variable(i);
            }
        }

        panic!("No variable to branch on")
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
    let fname = "/Users/user/Documents/REPO/scala-mdd-solver/src/test/resources/instances/complement/keller4.clq";

    let misp = Rc::new(Misp::from_file(fname));
    let relax = Rc::new(MispRelax::new(Rc::clone(&misp)));
    let vs = Rc::new(ActiveLexOrder::new(Rc::clone(&misp)));
    let w = Rc::new(FixedWidth(100));

    let inst = Rc::clone(&misp);
    let mut mdd = PooledMDD::new(inst, relax, vs, w, misp_min_lp);


    let vars = BitSet::new(misp.nb_vars()).not();
    let root = Rc::new(PooledNode::new(vars.clone(), 0, None,true));

    mdd.relaxed(vars, root, 0);

    println!("UB = {}", mdd.best_value())
}
