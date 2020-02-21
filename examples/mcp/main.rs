extern crate structopt;
use std::fs::File;

use ddo::core::common::Decision;
use ddo::core::abstraction::mdd::MDD;
use ddo::core::abstraction::solver::Solver;
use ddo::core::implementation::solver::parallel::ParallelSolver;
use ddo::core::implementation::heuristics::FixedWidth;
use ddo::core::implementation::mdd::builder::mdd_builder;
use std::time::SystemTime;
use structopt::StructOpt;

mod graph;
mod model;
mod relax;
use model::McpState;
use relax::McpRelax;

#[derive(StructOpt)]
/// Solve max cut from a DIMACS graph files
struct Mcp {
    /// Path to the graph instance
    fname: String,
    /// Log the progression
    #[structopt(short, long, parse(from_occurrences))]
    verbose: u8,
    /// The number of threads to use (default: number of physical threads on this machine)
    #[structopt(name="threads", short, long)]
    threads: Option<usize>,
    /// If specified, the max width allowed for any layer
    #[structopt(name="width", short, long)]
    width: Option<usize>
}

fn main() {
    let args = Mcp::from_args();
    mcp(&args.fname, args.verbose, args.threads, args.width);
}

/// Solves the given mcp instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .wcnf file holding
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
pub fn mcp(fname: &str, verbose: u8, threads: Option<usize>, width: Option<usize>) {
    let problem = File::open(fname).expect("File not found").into();
    match width {
        Some(max_width) => solve(mdd_builder(&problem, McpRelax::new(&problem))
                           .with_max_width(FixedWidth(max_width))
                           .into_flat(), verbose, threads),
        None => solve(mdd_builder(&problem, McpRelax::new(&problem))
                          .into_flat(), verbose, threads)
    }
}
fn solve<DD: MDD<McpState> + Clone + Send>(mdd: DD, verbose: u8, threads: Option<usize>) {
    let threads    = threads.unwrap_or_else(num_cpus::get);
    let mut solver = ParallelSolver::customized(mdd, verbose, threads);

    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
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