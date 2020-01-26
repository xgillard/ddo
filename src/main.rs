extern crate rust_mdd_solver;
extern crate thunder;
use thunder::thunderclap;

use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::rc::Rc;

use bitset_fixed::BitSet;

use rust_mdd_solver::core::abstraction::dp::{VarSet, Problem};
use rust_mdd_solver::core::abstraction::mdd::Node;
use rust_mdd_solver::core::implem::heuristics::{FixedWidth, NaturalOrder};
use rust_mdd_solver::core::implem::pooled_mdd::PooledNode;
use rust_mdd_solver::core::solver::{Solver, FromFunction};
use rust_mdd_solver::examples::misp::misp::Misp;
use rust_mdd_solver::examples::misp::relax::MispRelax;
use rust_mdd_solver::core::utils::LexBitSet;

fn vars_from_misp_state(_pb: &dyn Problem<BitSet>, n: &PooledNode<BitSet>) -> VarSet {
    VarSet(n.get_state().clone())
}

fn misp_min_lp(a: &PooledNode<BitSet>, b: &PooledNode<BitSet>) -> Ordering {
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
        } else { by_sz }
    } else { by_ub }
}


struct MispApp;
/// Solves MISP instances in DIMACS (.clq) format with an MDD branch and bound.
#[thunderclap]
impl MispApp {
    /// Solves the given misp instance with fixed width mdds
    ///
    /// # Arguments
    ///
    /// fname is the path name of some dimacs .clq file holding graph
    ///           description of the instance to solve
    /// width is the maximum allowed width of a layer.
    ///
    pub fn misp(fname: &str, width: usize) {
        let misp = Rc::new(Misp::from_file(fname));
        let relax = Rc::new(MispRelax::new(Rc::clone(&misp)));
        let vs = Rc::new(NaturalOrder::new());
        let w = Rc::new(FixedWidth(width));

        let mut solver = Solver::new(misp, relax, vs, w,
                                     misp_min_lp,
                                     misp_ub_order,
                                     FromFunction::new(vars_from_misp_state));

        solver.maximize();
    }
}

fn main() {
    MispApp::start();
}