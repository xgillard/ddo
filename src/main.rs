extern crate rust_mdd_solver;
extern crate structopt;

use std::fs::File;
use std::time::SystemTime;

use bitset_fixed::BitSet;
use structopt::StructOpt;

use rust_mdd_solver::core::abstraction::heuristics::WidthHeuristic;
use rust_mdd_solver::core::abstraction::solver::Solver;
use rust_mdd_solver::core::implementation::bb_solver::BBSolver;
use rust_mdd_solver::core::implementation::heuristics::{FixedWidth, MaxUB, MinLP, NaturalOrder, NbUnassigned};
use rust_mdd_solver::core::implementation::pooled_mdd::PooledMDDGenerator;
use rust_mdd_solver::core::utils::RefFunc;
use rust_mdd_solver::examples::misp::heuristics::vars_from_misp_state;
use rust_mdd_solver::examples::misp::relax::MispRelax;

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
    where WDTH : WidthHeuristic<BitSet> {

    let misp  = File::open(fname).expect("File not found").into();
    let relax = MispRelax::new(&misp);
    let vs    = NaturalOrder;

    let ddg = PooledMDDGenerator::new(&misp, relax, vs, width, MinLP);

    let mut solver = BBSolver::new(&misp, ddg, MaxUB, RefFunc(vars_from_misp_state));
    solver.verbosity = verbose;
    
    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum computed in {:?}", (end.duration_since(start).unwrap()));
    }

    if verbose >= 2 {
        println!("### Solution: ################################################");
        if let Some(sln) = sln {
            let mut sln = sln.clone();
            sln.sort_by_key(|d| d.variable.0);
            let solution_txt = sln.iter()
                .filter(|&d|d.value == 1)
                .fold(String::new(), |a, i| format!("{} {}", a, i.variable.0 + 1));

            println!("{}", solution_txt);
        }
    }
    if verbose >= 1 && sln.is_none() {
        println!("No solution !");
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