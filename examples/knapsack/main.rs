extern crate structopt;

use ddo::core::abstraction::mdd::MDD;
use ddo::core::abstraction::solver::Solver;
use ddo::core::common::Decision;
use ddo::core::implementation::heuristics::FixedWidth;
use ddo::core::implementation::mdd::builder::mdd_builder;
use ddo::core::implementation::solver::parallel::ParallelSolver;
use std::fs::File;
use std::time::SystemTime;
use structopt::StructOpt;

mod heuristics;
mod model;
mod relax;

use heuristics::KnapsackOrder;
use model::KnapsackState;
use relax::KnapsackRelax;

#[derive(StructOpt)]
/// Solves hard combinatorial problems with bounded width MDDs
struct Knapsack {
    /// Path to the knapsack instance file.
    ///
    /// The format of one such instance file should be the following:
    ///  - The header line describes the sack and should look as follows: `<sack_capacity> <nb_items>`
    ///  - Then, subsequent lines each describe an item. They should look like `<id> <profit> <weight> <quantity>`
    ///
    /// # Example
    /// The file with the following content describes a knapsack problem
    /// where the sack has a capacity of 23 and may be filled with 3 different
    /// type of items.
    ///
    /// One unit of the first item (id = 1) brings 10 of profit, and costs 5
    /// of capacity. The user may only place at most two such objects in the sack.
    ///
    /// One unit of the 2nd item (id = 2) brings 7 of profit, and costs 6
    /// of capacity. The user may only place at most one such item in the sack.
    ///
    /// One unit of the 3rd item (id = 3) brings 9 of profit, and costs 1
    //  of capacity. The user may only place at most one such item in the sack.
    /// ```
    /// 23 3
    /// 1 10 5 2
    /// 2 7  6 1
    /// 3 9  1 1
    /// ```
    ///
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
    let args = Knapsack::from_args();
    knapsack(&args.fname, args.verbose, args.threads, args.width);
}

/// Solves the given knapsack instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some file holding the description of the
///       instance to solve
/// width is the maximum allowed width of a layer.
///
fn knapsack(fname: &str, verbose: u8, threads: Option<usize>, width: Option<usize>) {
    let problem = File::open(fname).expect("File not found").into();
    match width {
        Some(max_width) => solve(mdd_builder(&problem, KnapsackRelax::new(&problem))
                           .with_branch_heuristic(KnapsackOrder::new(&problem))
                           .with_max_width(FixedWidth(max_width))
                           .into_flat(), verbose, threads),
        None => solve(mdd_builder(&problem, KnapsackRelax::new(&problem))
                          .with_branch_heuristic(KnapsackOrder::new(&problem))
                          .into_flat(), verbose, threads)
    }
}
fn solve<DD: MDD<KnapsackState> + Clone + Send >(mdd: DD, verbose: u8, threads: Option<usize>) {
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

            for i in sln.iter() {
                println!("{} -> {}", (i.variable.0 + 1), i.value);
            }
        }
    }
    if verbose >= 1 && sln.is_none() {
        println!("No solution !");
    }
}
