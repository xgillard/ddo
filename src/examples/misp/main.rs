use std::fs::File;

use bitset_fixed::BitSet;

use crate::core::common::Decision;
use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::implementation::bb_solver::BBSolver;
use crate::core::implementation::heuristics::{FixedWidth, MaxUB};
use crate::core::implementation::mdd::builder::mdd_builder;
use crate::core::utils::Func;
use crate::examples::misp::heuristics::{MispVarHeu, vars_from_misp_state};
use crate::examples::misp::relax::MispRelax;
use std::time::SystemTime;

/// Solves the given misp instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .clq file holding graph
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
pub fn misp(fname: &str, verbose: u8, width:Option<usize>) {
    let misp = File::open(fname).expect("File not found").into();
    match width {
        Some(max_size) => solve(mdd_builder(&misp, MispRelax::new(&misp))
            .with_load_vars(Func(vars_from_misp_state))
            .with_max_width(FixedWidth(max_size))
            .with_branch_heuristic(MispVarHeu::new(&misp))
            .into_pooled(), verbose),

        None => solve(mdd_builder(&misp, MispRelax::new(&misp))
            .with_load_vars(Func(vars_from_misp_state))
            .with_branch_heuristic(MispVarHeu::new(&misp))
            .into_pooled(), verbose)
    }
}
fn solve<DD: MDD<BitSet> + Clone + Send >(mdd: DD, verbose: u8) {
    let mut solver   = BBSolver::new(mdd, MaxUB);
    solver.verbosity = verbose;

    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum {} computed in {:?}", opt, end.duration_since(start).unwrap());
    }
    maybe_print_misp_solution(verbose, sln)
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