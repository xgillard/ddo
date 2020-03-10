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

use bitset_fixed::BitSet;
use ddo::core::abstraction::mdd::MDD;
use ddo::core::abstraction::solver::Solver;
use ddo::core::common::Decision;
use ddo::core::implementation::heuristics::FixedWidth;
use ddo::core::implementation::mdd::builder::mdd_builder_ref;
use ddo::core::implementation::solver::parallel::ParallelSolver;
use ddo::core::utils::Func;
use std::fs::File;
use std::time::SystemTime;
use structopt::StructOpt;

mod instance;
mod model;
mod relax;
mod heuristics;

use heuristics::{MispVarHeu, vars_from_misp_state};
use relax::MispRelax;

/// Solve maximum weighted independent set problem from DIMACS (.clq) files
#[derive(StructOpt)]
struct Misp {
    /// Path to the DIMACS MSIP instance
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
    let args = Misp::from_args();
    misp(&args.fname, args.verbose, args.threads, args.width);
}

/// Solves the given misp instance with fixed width mdds
///
/// # Arguments
///
/// fname is the path name of some dimacs .clq file holding graph
///           description of the instance to solve
/// width is the maximum allowed width of a layer.
///
pub fn misp(fname: &str, verbose: u8, threads: Option<usize>, width:Option<usize>) -> i32 {
    let misp = File::open(fname).expect("File not found").into();
    match width {
        Some(max_size) => solve(mdd_builder_ref(&misp, MispRelax::new(&misp))
            .with_load_vars(Func(vars_from_misp_state))
            .with_max_width(FixedWidth(max_size))
            .with_branch_heuristic(MispVarHeu::new(&misp))
            .into_pooled(), verbose, threads),

        None => solve(mdd_builder_ref(&misp, MispRelax::new(&misp))
            .with_load_vars(Func(vars_from_misp_state))
            .with_branch_heuristic(MispVarHeu::new(&misp))
            .into_pooled(), verbose, threads)
    }
}
fn solve<DD: MDD<BitSet> + Clone + Send>(mdd: DD, verbose: u8, threads: Option<usize>) -> i32 {
    let threads    = threads.unwrap_or_else(num_cpus::get);
    let mut solver = ParallelSolver::customized(mdd, verbose, threads);

    let start = SystemTime::now();
    let (opt, sln) = solver.maximize();
    let end = SystemTime::now();

    if verbose >= 1 {
        println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
    }
    maybe_print_misp_solution(verbose, sln);
    opt
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

#[cfg(test)]
mod test_misp {
    use std::path::PathBuf;
    use crate::misp;


    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("examples/tests/resources/misp/")
            .join(id)
    }

    fn solve_id(id: &str) -> i32 {
        let fname = locate(id);
        misp(fname.to_str().unwrap(), 0, None, Some(100))
    }

    /// This test takes > 60s to solve on my machine
    #[ignore] #[test]
    fn brock200_1() {
        assert_eq!(solve_id("brock200_1.clq"), 21);
    }
    #[test]
    fn brock200_2() {
        assert_eq!(solve_id("brock200_2.clq"), 12);
    }
    #[test]
    fn brock200_3() {
        assert_eq!(solve_id("brock200_3.clq"), 15);
    }
    #[ignore] #[test]
    fn brock200_4() {
        assert_eq!(solve_id("brock200_4.clq"), 17);
    }


    #[test]
    fn c_fat200_1() {
        assert_eq!(solve_id("c-fat200-1.clq"), 12);
    }
    #[test]
    fn c_fat200_2() {
        assert_eq!(solve_id("c-fat200-2.clq"), 24);
    }
    #[test]
    fn c_fat500_1() {
        assert_eq!(solve_id("c-fat500-1.clq"), 14);
    }
    #[test]
    fn c_fat500_2() {
        assert_eq!(solve_id("c-fat500-2.clq"), 26);
    }
    #[test]
    fn c_fat200_5() {
        assert_eq!(solve_id("c-fat200-5.clq"), 58);
    }

    #[test]
    fn hamming6_2() {
        assert_eq!(solve_id("hamming6-2.clq"), 32);
    }

    #[test]
    fn hamming6_4() {
        assert_eq!(solve_id("hamming6-4.clq"), 4);
    }
    #[ignore] #[test]
    fn hamming8_2() {
        assert_eq!(solve_id("hamming8-2.clq"), 128);
    }
    #[ignore] #[test]
    fn hamming8_4() {
        assert_eq!(solve_id("hamming8-4.clq"), 16);
    }
    #[ignore] #[test]
    fn hamming10_4() {
        assert_eq!(solve_id("hamming10-4.clq"), 40);
    }

    #[test]
    fn johnson8_2_4() {
        assert_eq!(solve_id("johnson8-2-4.clq"), 4);
    }
    #[test]
    fn johnson8_4_4() {
        assert_eq!(solve_id("johnson8-4-4.clq"), 14);
    }

    #[test]
    fn keller4() {
        assert_eq!(solve_id("keller4.clq"), 11);
    }
    #[ignore] #[test]
    fn keller5() {
        assert_eq!(solve_id("keller5.clq"), 27);
    }

    #[test]
    fn mann_a9() {
        assert_eq!(solve_id("MANN_a9.clq"), 16);
    }
    #[ignore] #[test]
    fn mann_a27() {
        assert_eq!(solve_id("MANN_a27.clq"), 126);
    }
    #[ignore] #[test]
    fn mann_a45() {
        assert_eq!(solve_id("MANN_a45.clq"), 315);
    }

    #[test]
    fn p_hat300_1() {
        assert_eq!(solve_id("p_hat300-1.clq"), 8);
    }
    #[ignore] #[test]
    fn p_hat300_2() {
        assert_eq!(solve_id("p_hat300-2.clq"), 25);
    }
    #[ignore] #[test]
    fn p_hat300_3() {
        assert_eq!(solve_id("p_hat300-3.clq"), 36);
    }
    #[ignore] #[test]
    fn p_hat700_1() {
        assert_eq!(solve_id("p_hat700-1.clq"), 11);
    }
    #[ignore] #[test]
    fn p_hat700_2() {
        assert_eq!(solve_id("p_hat700-2.clq"), 44);
    }
    #[ignore] #[test]
    fn p_hat700_3() {
        assert_eq!(solve_id("p_hat700-3.clq"), 62);
    }
    #[ignore] #[test]
    fn p_hat1500_1() {
        assert_eq!(solve_id("p_hat1500-1.clq"), 12);
    }
    #[ignore] #[test]
    fn p_hat1500_2() {
        assert_eq!(solve_id("p_hat1500-2.clq"), 65);
    }
    #[ignore] #[test]
    fn p_hat1500_3() {
        assert_eq!(solve_id("p_hat1500-3.clq"), 94);
    }
}