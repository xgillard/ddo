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
use ddo::implementation::heuristics::FixedWidth;
use ddo::implementation::mdd::config::mdd_builder;
use ddo::implementation::solver::parallel::ParallelSolver;

use crate::model::{KnapsackOrder, KnapsackRelax};

pub mod model;

#[derive(StructOpt)]
pub struct Args {
    pub fname: String,
    pub width: usize,
    #[structopt(name="threads", short, long)]
    threads: Option<usize>,
}


fn kp(fname: &str, width: usize, threads: Option<usize>) -> isize {
    let threads   = threads.unwrap_or_else(num_cpus::get);
    let problem   = File::open(fname).expect("file not found").into();
    let relax     = KnapsackRelax::new(&problem);
    let mdd       = mdd_builder(&problem, relax)
        .with_max_width(FixedWidth(width))
        .with_branch_heuristic(KnapsackOrder::new(&problem))
        .into_deep();

    let mut solver = ParallelSolver::customized(mdd, 2, threads);
    let start  = SystemTime::now();
    let opt    = solver.maximize().best_value.unwrap_or(isize::min_value());
    let end    = SystemTime::now();
    println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
    opt
}

fn main() {
    let args  = Args::from_args();
    let value = kp(&args.fname, args.width, args.threads);

    println!("best value = {}", value);
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::rc::Rc;

    use metrohash::MetroHashMap;

    use ddo::common::VarSet;

    use crate::kp;
    use crate::model::KnapsackState;

    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("examples/tests/resources/knapsack/")
            .join(id)
    }

    fn solve_id(id: &str, w: usize) -> isize {
        let fname = locate(id);
        kp(fname.to_str().unwrap(), w, None)
    }

    /*
        #[test]
        fn example2_w5() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 5);
            println!("{}", v);
        }

        #[test]
        fn example2_w10() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 10);
            println!("{}", v);
        }

        #[test]
        fn example2_w15() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 15);
            println!("{}", v);
        }

        #[test]
        fn example2_w20() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 20);
            println!("{}", v);
        }

        #[test]
        fn example2_w25() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 25);
            println!("{}", v);
        }

        #[test]
        fn example2_w30() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 30);
            println!("{}", v);
        }

        #[test]
        fn example2_w100() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 100);
            println!("{}", v);
        }

        #[test]
        fn example2_w1000() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 1000);
            println!("{}", v);
        }

        #[test]
        fn example2_w12_000() {
            let v = kp("/Users/user/Documents/REPO/rust-mdd-solver/examples/tests/resources/knapsack/example2", 12_000);
            println!("{}", v);
        }
    */
    #[test]
    fn example3_w1000() {
        assert_eq!(solve_id("example3", 1_000), 13_570);
    }
    #[test]
    fn example2_w12_500() {
        assert_eq!(solve_id("example2", 12_500), 13_570);
    }
    #[test]
    fn example2_w13_000() {
        assert_eq!(solve_id("example2", 13_000), 13_570);
    }

    #[test]
    fn example2_w100_000() {
        assert_eq!(solve_id("example2", 100_000), 13_570);
    }

    #[test]
    fn example2_w1_000_000() {
        assert_eq!(solve_id("example2", 1_000_000), 13_570);
    }

    #[test]
    fn an_hashmap_indexed_on_rc_will_do_the_trick() {
        let mut map : MetroHashMap<Rc<KnapsackState>, usize> = Default::default();

        let state1 = KnapsackState{free_vars: VarSet::all(5), capacity: 6};
        let state2 = KnapsackState{free_vars: VarSet::all(5), capacity: 6};
        let state3 = KnapsackState{free_vars: VarSet::all(5), capacity: 7};


        let state1 = Rc::new(state1);
        let state2 = Rc::new(state2);
        let state3 = Rc::new(state3);

        map.insert(state1, 1);
        map.insert(state3, 3);

        assert_eq!(1, map[&state2]); // state2 was never inserted. still, it is equal to state1
    }
}
