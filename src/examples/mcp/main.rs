use std::fs::File;

use crate::core::common::Decision;
use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::solver::Solver;
use crate::core::implementation::solver::parallel::BBSolver;
use crate::core::implementation::heuristics::FixedWidth;
use crate::core::implementation::mdd::builder::mdd_builder;
use crate::examples::mcp::model::McpState;
use crate::examples::mcp::relax::McpRelax;
use std::time::SystemTime;

/// Solves the given mcp instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .wcnf file holding
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
pub fn mcp(fname: &str, verbose: u8, width: Option<usize>) {
    let problem = File::open(fname).expect("File not found").into();
    match width {
        Some(max_width) => solve(mdd_builder(&problem, McpRelax::new(&problem))
                           .with_max_width(FixedWidth(max_width))
                           .into_flat(), verbose),
        None => solve(mdd_builder(&problem, McpRelax::new(&problem))
                          .into_flat(), verbose)
    }
}
fn solve<DD: MDD<McpState> + Clone + Send>(mdd: DD, verbose: u8) {
    let mut solver = BBSolver::with_verbosity(mdd, verbose);

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