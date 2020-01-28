extern crate rust_mdd_solver;
extern crate structopt;

use structopt::StructOpt;

use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::rc::Rc;

use bitset_fixed::BitSet;

use rust_mdd_solver::core::abstraction::dp::{VarSet, Problem};
use rust_mdd_solver::core::abstraction::mdd::Node;
use rust_mdd_solver::core::implem::heuristics::{FixedWidth, NaturalOrder, NbUnassigned};
use rust_mdd_solver::core::implem::pooled_mdd::PooledNode;
use rust_mdd_solver::core::solver::{Solver, FromFunction};
use rust_mdd_solver::examples::misp::misp::Misp;
use rust_mdd_solver::examples::misp::relax::MispRelax;
use rust_mdd_solver::core::utils::LexBitSet;
use rust_mdd_solver::core::abstraction::heuristics::WidthHeuristic;
use simplelog::{SimpleLogger, Config};
use log::{debug, warn, log_enabled, Level, LevelFilter};
use std::time::SystemTime;

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

/// Solves hard combinatorial problems with bounded width MDDs
#[derive(StructOpt)]
enum RustMddSolver {
    /// Solve maximum weighted independent set problem from DIMACS (.clq) files
    Misp {
        /// Path to the DIMACS MSIP instance
        fname: String,
        /// Log the progression
        #[structopt(short, long, parse(from_occurrences))]
        verbose: u8,
        /// If specified, the max width allowed for any layer
        width: Option<usize>
    }
}

/// Solves the given misp instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .clq file holding graph
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
fn misp<WDTH>(fname: &str, verbose: u8, width: WDTH) -> i32
    where WDTH : WidthHeuristic<BitSet, PooledNode<BitSet>> {
    let loglevel = match verbose {
        0 => LevelFilter::Off,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace
    };
    SimpleLogger::init(loglevel, Config::default()).unwrap();

    let misp = Rc::new(Misp::from_file(fname));
    let relax = MispRelax::new(Rc::clone(&misp));
    let vs = NaturalOrder::new();

    let mut solver = Solver::new(misp, relax, vs,
                                 width,
                                 misp_min_lp,
                                 misp_ub_order,
                                 FromFunction::new(vars_from_misp_state));

    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if log_enabled!(Level::Debug) {
        debug!("Optimum computed in {:?}", (end.duration_since(start).unwrap()));
    }

    if log_enabled!(Level::Debug) {
        debug!("### Solution: ################################################");
        if let Some(sln) = sln {
            let mut sln = sln.clone();
            sln.sort_by_key(|d| d.variable.0);
            let solution_txt = sln.iter()
                .filter(|&d|d.value == 1)
                .fold(String::new(), |a, i| format!("{} {}", a, i.variable.0 + 1));

            debug!("{}", solution_txt);
        }
    }
    if sln.is_none() {
        warn!("No solution !");
    }
    opt
}

fn main() {
    let args = RustMddSolver::from_args();
    match args {
        RustMddSolver::Misp {fname, verbose, width} =>
        if let Some(width) = width {
            misp(&fname, verbose, FixedWidth(width))
        } else {
            misp(&fname, verbose, NbUnassigned)
        }
    };
}