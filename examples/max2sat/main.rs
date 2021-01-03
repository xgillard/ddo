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
use std::time::{Duration, Instant};

use structopt::StructOpt;

use ddo::{Solver, ParallelSolver, NoDupFrontier, Solution, Completion, config_builder, FixedWidth, TimeBudget, Problem, DeepMDD};

use crate::heuristics::{Max2SatOrder, MinRank, LoadVarsFromMax2SatState};
use crate::relax::Max2SatRelax;
use std::path::Path;
use crate::model::Max2Sat;

mod instance;
mod model;
mod relax;
mod heuristics;


/// MAX2SAT is a solver based on branch-and-bound mdd which solves the maximum
/// 2-satisfiability problem to optimality.
///
/// The implementation of max2sat is based on
/// 'ddo: a generic and efficient framework for MDD-based optimization' (IJCAI20)
#[derive(StructOpt)]
enum Args {
    /// This is the action you want to take in order to actually solve an
    /// instance.
    Solve {
        /// The path to the MAX2SAT instance that needs to be solved.
        instance: String,
        /// The verbosity level of what is going to be logged on the console.
        #[structopt(name="verbosity", short, long)]
        verbosity: Option<u8>,
        /// The maximum width of an mdd layer.
        #[structopt(name="width", short, long)]
        width: Option<usize>,
        /// How many threads do you want to use to solve the problem ?
        #[structopt(name="threads", short, long)]
        threads: Option<usize>,
        /// How long do you want the solver to keep working on your problem ?
        /// (in seconds)
        #[structopt(name="duration", short, long)]
        duration: Option<u64>,
        /// Shall we print the header in addition to solving the instance ?
        #[structopt(name="header", long)]
        header: bool
    },
    /// Use this command if you only intend to print the solution header.
    PrintHeader
}

fn main() {
    let args  = Args::from_args();
    match args {
        Args::PrintHeader => {
            print_header();
        },
        Args::Solve {instance, verbosity, width, threads, duration, header} => {
            let threads    = threads.unwrap_or_else(num_cpus::get);
            let verbosity  = verbosity.unwrap_or(0);

            let problem    = Max2Sat::from(File::open(&instance).expect("File not found"));
            let relax      = Max2SatRelax::new(&problem);
            let mut solver = solver(&problem, relax, width, duration, threads, verbosity);

            let start      = Instant::now();
            let outcome    = solver.maximize();
            let finish     = Instant::now();

            let instance   = instance_name(&instance);
            let nb_vars    = problem.nb_vars();
            let lb         = objective(solver.as_ref().best_lower_bound());
            let ub         = objective(solver.as_ref().best_upper_bound());
            let solution   = solver.as_ref().best_solution();
            let duration   = finish - start;

            if header {
                print_header();
            }
            print_solution(&instance, nb_vars, outcome, &lb, &ub, duration, solution);
        },
    }
}

fn solver<'a>(pb:    &'a Max2Sat,
              rlx:       Max2SatRelax<'a>,
              width:     Option<usize>,
              cutoff:    Option<u64>,
              threads:   usize,
              verbosity: u8)
              -> Box<dyn Solver + 'a>
{
    match (width, cutoff) {
        (Some(w), Some(c)) => {
            let conf = config_builder(pb, rlx)
                .with_branch_heuristic(Max2SatOrder::new(&pb))
                .with_load_vars(LoadVarsFromMax2SatState::new(&pb))
                .with_nodes_selection_heuristic(MinRank)
                .with_max_width(FixedWidth(w))
                .with_cutoff(TimeBudget::new(Duration::from_secs(c)))
                .build();
            let mdd = DeepMDD::from(conf);
            let solver =ParallelSolver::customized(mdd, verbosity, threads)
                .with_frontier(NoDupFrontier::default());
            Box::new(solver)
        },
        (Some(w), None) => {
            let conf = config_builder(pb, rlx)
                .with_branch_heuristic(Max2SatOrder::new(&pb))
                .with_load_vars(LoadVarsFromMax2SatState::new(&pb))
                .with_nodes_selection_heuristic(MinRank)
                .with_max_width(FixedWidth(w))
                .build();
            let mdd = DeepMDD::from(conf);
            let solver =ParallelSolver::customized(mdd, verbosity, threads)
                .with_frontier(NoDupFrontier::default());
            Box::new(solver)
        },
        (None, Some(c)) => {
            let conf = config_builder(pb, rlx)
                .with_branch_heuristic(Max2SatOrder::new(&pb))
                .with_load_vars(LoadVarsFromMax2SatState::new(&pb))
                .with_nodes_selection_heuristic(MinRank)
                .with_cutoff(TimeBudget::new(Duration::from_secs(c)))
                .build();
            let mdd = DeepMDD::from(conf);
            let solver =ParallelSolver::customized(mdd, verbosity, threads)
                .with_frontier(NoDupFrontier::default());
            Box::new(solver)
        },
        (None, None) => {
            let conf = config_builder(pb, rlx)
                .with_branch_heuristic(Max2SatOrder::new(&pb))
                .with_load_vars(LoadVarsFromMax2SatState::new(&pb))
                .with_nodes_selection_heuristic(MinRank)
                .build();
            let mdd = DeepMDD::from(conf);
            let solver =ParallelSolver::customized(mdd, verbosity, threads)
                .with_frontier(NoDupFrontier::default());
            Box::new(solver)
        },
    }
}

fn print_header() {
    println!("{:40} | {:10} | {:10} | {:10} | {:10} | {:8}",
             "INSTANCE", "STATUS", "LB", "UB", "DURATION", "SOLUTION");
}
fn print_solution(name: &str, n: usize, completion: Completion, lb: &str, ub: &str, duration: Duration, solution: Option<Solution>) {
    println!("{:40} | {:10} | {:10} | {:10} | {:10.3} | {}",
             name,
             status(completion),
             lb, ub,
             duration.as_secs_f32(),
             solution_to_string(n, solution));
}
fn instance_name<P: AsRef<Path>>(fname: P) -> String {
    fname.as_ref().file_name().unwrap().to_str().unwrap().to_string()
}
fn objective(x: isize) -> String {
    match x {
        isize::MIN => "-inf".to_string(),
        isize::MAX => "+inf".to_string(),
        _ => x.to_string()
    }
}
fn status(completion: Completion) -> &'static str {
    if completion.is_exact {
        "Proved"
    } else {
        "Timeout"
    }
}
fn solution_to_string(nb_vars: usize, solution: Option<Solution>) -> String {
    match solution {
        None   => "No feasible solution found".to_string(),
        Some(s)=> {
            let mut perm = vec![0; nb_vars];
            for d in s.iter() {
                perm[d.variable.id()] = d.value;
            }
            let mut txt = String::new();
            for v in perm {
                txt = format!("{} {}", txt, v);
            }
            txt
        }
    }
}


#[cfg(test)]
mod test_max2sat {
    use std::fs::File;
    use std::path::PathBuf;
    use std::time::Instant;
    use crate::{solver, relax::Max2SatRelax};

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


    fn max2sat(fname: &str, threads: Option<usize>) -> isize {
        let threads   = threads.unwrap_or_else(num_cpus::get);
        let problem   = File::open(fname).expect("file not found").into();
        let relax     = Max2SatRelax::new(&problem);
        let mut solver = solver(&problem, relax, None, None, threads, 0);

        let start  = Instant::now();
        let opt    = solver.maximize().best_value.unwrap_or(isize::min_value());
        let end    = Instant::now();
        println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start), threads);
        opt
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
