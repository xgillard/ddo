// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

mod instance;
mod model;
mod relax;
mod heuristics;

use heuristics::{Max2SatOrder, MinRank};
use model::State;
use relax::Max2SatRelax;

#[derive(StructOpt)]
/// Solve weighed max2sat from DIMACS (.wcnf) files
struct Max2sat {
    /// Path to the DIMACS MAX2SAT instance
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
    let args = Max2sat::from_args();
    max2sat(&args.fname, args.verbose, args.threads, args.width);
}


/// Solves the given max2sat instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .wcnf file holding
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
fn max2sat(fname: &str, verbose: u8, threads: Option<usize>, width: Option<usize>) -> i32 {
    let problem = File::open(fname).expect("File not found").into();
    match width {
        Some(max_width) => solve(mdd_builder(&problem, Max2SatRelax::new(&problem))
                          .with_branch_heuristic(Max2SatOrder::new(&problem))
                          .with_nodes_selection_heuristic(MinRank)
                          .with_max_width(FixedWidth(max_width))
                          .into_flat(), verbose, threads),
        None => solve(mdd_builder(&problem, Max2SatRelax::new(&problem))
                          .with_branch_heuristic(Max2SatOrder::new(&problem))
                          .with_nodes_selection_heuristic(MinRank)
                          .into_flat(), verbose, threads)
    }
}
fn solve<DD: MDD<State> + Clone + Send>(mdd: DD, verbose: u8, threads: Option<usize>) -> i32 {
    let threads    = threads.unwrap_or_else(num_cpus::get);
    let mut solver = ParallelSolver::customized(mdd, verbose, threads);

    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
    }
    maybe_print_solution(verbose, sln);
    opt
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


#[cfg(test)]
mod test_max2sat {
    use std::path::PathBuf;
    use crate::max2sat;

    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("examples/tests/resources/max2sat/")
            .join(id)
    }

    fn solve_id(id: &str) -> i32 {
        let fname = locate(id);
        max2sat(fname.to_str().unwrap(), 0, None, Some(5))
    }

    #[test]
    fn debug() {
        assert_eq!(solve_id("debug.wcnf"), 24);
    }
    #[test]
    fn debug2() {
        assert_eq!(solve_id("debug2.wcnf"), 13);
    }

    #[test]
    fn pass() {
        assert_eq!(solve_id("pass.wcnf"), 54);
    }
    #[test]
    fn tautology() {
        assert_eq!(solve_id("tautology.wcnf"), 7);
    }
    #[test]
    fn unit() {
        assert_eq!(solve_id("unit.wcnf"), 6);
    }
    #[test]
    fn negative_wt() {
        assert_eq!(solve_id("negative_wt.wcnf"), 4258);
    }

    #[test]
    fn frb10_6_1() {
        assert_eq!(solve_id("frb10-6-1.wcnf"), 37_037);
    }
    #[test]
    fn frb10_6_2() {
        assert_eq!(solve_id("frb10-6-2.wcnf"), 38_196);
    }
    #[test]
    fn frb10_6_3() {
        assert_eq!(solve_id("frb10-6-3.wcnf"), 36_671);
    }
    #[test]
    fn frb10_6_4() {
        assert_eq!(solve_id("frb10-6-4.wcnf"), 38_928);
    }
    #[ignore] #[test]
    fn frb15_9_1() {
        assert_eq!(solve_id("frb15-9-1.wcnf"), 341_783);
    }
    #[ignore] #[test]
    fn frb15_9_2() {
        assert_eq!(solve_id("frb15-9-2.wcnf"), 341_919);
    }
    #[ignore] #[test]
    fn frb15_9_3() {
        assert_eq!(solve_id("frb15-9-3.wcnf"), 339_471);
    }
    #[ignore] #[test]
    fn frb15_9_4() {
        assert_eq!(solve_id("frb15-9-4.wcnf"), 340_559);
    }
    #[ignore] #[test]
    fn frb15_9_5() {
        assert_eq!(solve_id("frb15-9-5.wcnf"), 348_311);
    }
    #[ignore] #[test]
    fn frb20_11_1() {
        assert_eq!(solve_id("frb20-11-1.wcnf"), 1_245_134);
    }
    #[ignore] #[test]
    fn frb20_11_2() {
        assert_eq!(solve_id("frb20-11-2.wcnf"), 1_231_874);
    }
    #[ignore] #[test]
    fn frb20_11_3() {
        assert_eq!(solve_id("frb20-11-2.wcnf"), 1_240_493);
    }
    #[ignore] #[test]
    fn frb20_11_4() {
        assert_eq!(solve_id("frb20-11-4.wcnf"), 1_231_653);
    }
    #[ignore] #[test]
    fn frb20_11_5() {
        assert_eq!(solve_id("frb20-11-5.wcnf"), 1_237_841);
    }
}