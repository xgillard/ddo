use std::fs::File;

use crate::core::common::Decision;
use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::implementation::bb_solver::BBSolver;
use crate::core::implementation::heuristics::{FixedWidth, MaxUB};
use crate::core::implementation::mdd::builder::mdd_builder;
use crate::examples::max2sat::heuristics::{Max2SatOrder, MinRank};
use crate::examples::max2sat::model::State;
use crate::examples::max2sat::relax::Max2SatRelax;
use std::time::SystemTime;

/// Solves the given max2sat instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .wcnf file holding
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
pub fn max2sat(fname: &str, verbose: u8, width: Option<usize>) {
    let problem = File::open(fname).expect("File not found").into();
    match width {
        Some(max_width) => solve(mdd_builder(&problem, Max2SatRelax::new(&problem))
                          .with_branch_heuristic(Max2SatOrder::new(&problem))
                          .with_nodes_selection_heuristic(MinRank)
                          .with_max_width(FixedWidth(max_width))
                          .into_flat(), verbose),
        None => solve(mdd_builder(&problem, Max2SatRelax::new(&problem))
                          .with_branch_heuristic(Max2SatOrder::new(&problem))
                          .with_nodes_selection_heuristic(MinRank)
                          .into_flat(), verbose)
    }
}
fn solve<DD: MDD<State>>(mdd: DD, verbose: u8) {
    let mut solver   = BBSolver::new(mdd, MaxUB);
    solver.verbosity = verbose;

    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum {} computed in {:?}", opt, end.duration_since(start).unwrap());
    }
    maybe_print_solution(verbose, sln)
}
fn maybe_print_solution(verbose: u8, sln: &Option<Vec<Decision>>) {
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
