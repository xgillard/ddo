extern crate rust_mdd_solver;

use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::rc::Rc;
use std::time::SystemTime;

use bitset_fixed::BitSet;

use rust_mdd_solver::core::abstraction::dp::{Variable, VarSet, Problem};
use rust_mdd_solver::core::abstraction::heuristics::VariableHeuristic;
use rust_mdd_solver::core::abstraction::mdd::{MDD, Node};
use rust_mdd_solver::core::implem::heuristics::FixedWidth;
use rust_mdd_solver::core::implem::pooled_mdd::PooledNode;
use rust_mdd_solver::core::solver::Solver;
use rust_mdd_solver::examples::misp::misp::Misp;
use rust_mdd_solver::examples::misp::relax::MispRelax;
use rust_mdd_solver::core::utils::{LexBitSet, BitSetIter};

struct ActiveLexOrder{
    pb: Rc<Misp>
}

impl ActiveLexOrder {
    pub fn new(pb: Rc<Misp>) -> ActiveLexOrder {
        ActiveLexOrder{pb}
    }
}
impl VariableHeuristic<BitSet, PooledNode<BitSet>> for ActiveLexOrder {
    fn next_var(&self, dd: &dyn MDD<BitSet, PooledNode<BitSet>>, vars: &VarSet) -> Option<Variable> {
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
            LexBitSet(a.get_state()).cmp(&LexBitSet(b.get_state()))
        }
    }
}

fn misp_ub_order(a : &PooledNode<BitSet>, b: &PooledNode<BitSet>) -> Ordering {
    let by_ub = a.get_ub().cmp(&b.get_ub());
    if by_ub == Equal {
        let by_sz = a.get_state().count_ones().cmp(&b.get_state().count_ones());
        if by_sz == Equal {
            let by_lp_len = a.get_lp_len().cmp(&b.get_lp_len());
            if by_lp_len == Equal {
                LexBitSet(a.get_state()).cmp(&LexBitSet(b.get_state()))
            } else { by_lp_len }
        } else { by_sz.reverse() }
    } else { by_ub }
}

fn main() {
    let fname = "/Users/user/Desktop/mdd_cpp/instances/keller4.clq";
    let start = SystemTime::now();

    let misp = Rc::new(Misp::from_file(fname));
    let relax = Rc::new(MispRelax::new(Rc::clone(&misp)));
    let vs = Rc::new(ActiveLexOrder::new(Rc::clone(&misp)));
    let w = Rc::new(FixedWidth(100));

    let mut solver = Solver::new(misp, relax, vs, w,
                                 misp_min_lp,
                                 misp_ub_order);
    let (opt, _) = solver.maximize();
    println!("Optimum  {}", opt);
    println!("Explored {}", solver.explored);

    let end = SystemTime::now();
    println!("Tm {:?}", end.duration_since(start).unwrap());

}