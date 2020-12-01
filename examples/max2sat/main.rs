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

use std::fs::File;
use std::time::SystemTime;

use structopt::StructOpt;

use ddo::abstraction::solver::Solver;
use ddo::implementation::mdd::config::mdd_builder;
use ddo::implementation::solver::parallel::ParallelSolver;

use crate::heuristics::{Max2SatOrder, MinRank, LoadVarsFromMax2SatState};
use crate::relax::Max2SatRelax;
use ddo::implementation::frontier::NoDupFrontier;

mod instance;
mod model;
mod relax;
mod heuristics;

#[derive(StructOpt)]
pub struct Args {
    pub fname: String,
    #[structopt(name="threads", short, long)]
    threads: Option<usize>,
}


fn max2sat(fname: &str, threads: Option<usize>) -> isize {
    let threads   = threads.unwrap_or_else(num_cpus::get);
    let problem   = File::open(fname).expect("file not found").into();
    let relax     = Max2SatRelax::new(&problem);
    let mdd       = mdd_builder(&problem, relax)
        .with_branch_heuristic(Max2SatOrder::new(&problem))
        .with_load_vars(LoadVarsFromMax2SatState::new(&problem))
        .with_nodes_selection_heuristic(MinRank)
        .into_deep();

    let mut solver = ParallelSolver::customized(mdd, 2, threads)
        .with_frontier(NoDupFrontier::default());

    let start  = SystemTime::now();
    let opt    = solver.maximize().best_value.unwrap_or(isize::min_value());
    let end    = SystemTime::now();
    println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
    opt
}

fn main() {
    let args  = Args::from_args();
    let value = max2sat(&args.fname, args.threads);

    println!("best value = {}", value);
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

    fn solve_id(id: &str) -> isize {
        let fname = locate(id);
        max2sat(fname.to_str().unwrap(), None)
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