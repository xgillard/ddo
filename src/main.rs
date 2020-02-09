extern crate rust_mdd_solver;
extern crate structopt;

use std::fs::File;
use std::time::SystemTime;

use bitset_fixed::BitSet;
use structopt::StructOpt;

use rust_mdd_solver::core::abstraction::heuristics::WidthHeuristic;
use rust_mdd_solver::core::abstraction::solver::Solver;
use rust_mdd_solver::core::implementation::bb_solver::BBSolver;
use rust_mdd_solver::core::implementation::heuristics::{FixedWidth, MaxUB, MinLP, NaturalOrder, NbUnassigned, FromLongestPath};
use rust_mdd_solver::core::implementation::pooled_mdd::PooledMDDGenerator;
use rust_mdd_solver::examples::misp::heuristics::vars_from_misp_state;
use rust_mdd_solver::examples::misp::relax::MispRelax;
use rust_mdd_solver::core::common::Decision;
use rust_mdd_solver::examples::max2sat::relax::Max2SatRelax;
use rust_mdd_solver::examples::max2sat::heuristics::Max2SatOrder;
use rust_mdd_solver::examples::max2sat::model::State;
use rust_mdd_solver::core::utils::Func;

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
    },
    /// Solve weighed max2sat from DIMACS (.wcnf) files
    Max2Sat {
        /// Path to the DIMACS MSIP instance
        fname: String,
        /// Log the progression
        #[structopt(short, long, parse(from_occurrences))]
        verbose: u8,
        /// If specified, the max width allowed for any layer
        width: Option<usize>
    }
}

fn maybe_print_misp_solution(verbose: u8, sln: &Option<Vec<Decision>>) {
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
}
fn maybe_print_max2sat_solution(verbose: u8, sln: &Option<Vec<Decision>>) {
    if verbose >= 2 {
        println!("### Solution: ################################################");
        if let Some(sln) = sln {
            let mut sln = sln.clone();
            sln.sort_by_key(|d| d.variable.0);
            let solution_txt = sln.iter()
                .fold(String::new(), |a, i| format!("{} {}", a, (i.variable.0 + 1) as i32 * i.value));

            println!("{}", solution_txt);
        }
    }
    if verbose >= 1 && sln.is_none() {
        println!("No solution !");
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

    let mut solver = BBSolver::new(ddg, MaxUB, Func(vars_from_misp_state));
    solver.verbosity = verbose;
    
    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum computed in {:?}", (end.duration_since(start).unwrap()));
    }
    maybe_print_misp_solution(verbose, sln);

    opt
}

/// Solves the given max2sat instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .wcnf file holding
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
fn max2sat<WDTH>(fname: &str, verbose: u8, width: WDTH) -> i32
    where WDTH : WidthHeuristic<State> {

    let problem = File::open(fname).expect("File not found").into();
    let relax   = Max2SatRelax::new(&problem);
    let vs      = Max2SatOrder::new(&problem);
    let bo      = MaxUB;
    let load_v  = FromLongestPath::new(&problem);

    let ddg = PooledMDDGenerator::new(&problem, relax, vs, width, MinLP);

    let mut solver = BBSolver::new(ddg, bo, load_v);
    solver.verbosity = verbose;

    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum computed in {:?}", (end.duration_since(start).unwrap()));
    }
    maybe_print_max2sat_solution(verbose, sln);

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
            },
        RustMddSolver::Max2Sat {fname, verbose, width} =>
            if let Some(width) = width {
                max2sat(&fname, verbose, FixedWidth(width))
            } else {
                max2sat(&fname, verbose, NbUnassigned)
            }
    };
}