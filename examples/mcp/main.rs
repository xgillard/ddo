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

use ddo::{Problem, Variable, VarSet, Solver, ParallelSolver, LoadVars, FrontierNode, NoDupFrontier, Solution, Completion, config_builder, FixedWidth, TimeBudget, DeepMDD};

use crate::relax::McpRelax;
use crate::model::{McpState, Mcp};
use std::path::Path;

pub mod graph;
pub mod model;
pub mod relax;

/// MCP is a solver based on branch-and-bound mdd which solves the maximum cut
/// problem to optimality.
///
/// The implementation of mcp is based on
/// 'ddo: a generic and efficient framework for MDD-based optimization' (IJCAI20)
#[derive(StructOpt)]
enum Args {
    /// This is the action you want to take in order to actually solve an
    /// instance.
    Solve {
        /// The path to the MCP instance that needs to be solved.
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

#[derive(Clone, Copy)]
struct LoadVarsFromState<'a> {
    pb: &'a Mcp
}
impl <'a> LoadVarsFromState<'a> {
    pub fn new(pb: &'a Mcp) -> Self {
        Self {pb}
    }
}

impl LoadVars<McpState> for LoadVarsFromState<'_> {
    fn variables(&self, node: &FrontierNode<McpState>) -> VarSet {
        let mut vars = self.pb.all_vars();
        let depth = node.state.as_ref().depth;

        for i in 0..depth {
            vars.remove(Variable(i as usize));
        }

        vars
    }
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

            let problem    = Mcp::from(File::open(&instance).expect("File not found"));
            let relax      = McpRelax::new(&problem);
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

fn solver<'a>(pb:    &'a Mcp,
              rlx:       McpRelax<'a>,
              width:     Option<usize>,
              cutoff:    Option<u64>,
              threads:   usize,
              verbosity: u8)
              -> Box<dyn Solver + 'a>
{
    match (width, cutoff) {
        (Some(w), Some(c)) => {
            let conf = config_builder(pb, rlx)
                .with_load_vars(LoadVarsFromState::new(&pb))
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
                .with_load_vars(LoadVarsFromState::new(&pb))
                .with_max_width(FixedWidth(w))
                .build();
            let mdd = DeepMDD::from(conf);
            let solver =ParallelSolver::customized(mdd, verbosity, threads)
                .with_frontier(NoDupFrontier::default());
            Box::new(solver)
        },
        (None, Some(c)) => {
            let conf = config_builder(pb, rlx)
                .with_load_vars(LoadVarsFromState::new(&pb))
                .with_cutoff(TimeBudget::new(Duration::from_secs(c)))
                .build();
            let mdd = DeepMDD::from(conf);
            let solver =ParallelSolver::customized(mdd, verbosity, threads)
                .with_frontier(NoDupFrontier::default());
            Box::new(solver)
        },
        (None, None) => {
            let conf = config_builder(pb, rlx)
                .with_load_vars(LoadVarsFromState::new(&pb))
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
mod tests {
    use std::fs::File;
    use std::path::PathBuf;
    use std::time::Instant;
    use crate::{solver, relax::McpRelax};

    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("examples/tests/resources/mcp/")
            .join(id)
    }

    fn solve_id(id: &str) -> isize {
        let fname = locate(id);
        //mcp(fname.to_str().unwrap(), 2, Some(1))
        //mcp(fname.to_str().unwrap(), 30_000_000, Some(1))
        //mcp(fname.to_str().unwrap(), 1_000, Some(1))
        //mcp(fname.to_str().unwrap(), 500, Some(1))
        //mcp(fname.to_str().unwrap(), 30, Some(1))
        //mcp(fname.to_str().unwrap(), 3, Some(1))
        mcp(fname.to_str().unwrap(), Some(1))
    }


    fn mcp(fname: &str, threads: Option<usize>) -> isize {
        let threads   = threads.unwrap_or_else(num_cpus::get);
        let problem   = File::open(fname).expect("file not found").into();
        let relax     = McpRelax::new(&problem);
        let mut solver= solver(&problem, relax, None, None, threads, 0);

        let start = Instant::now();
        let opt   = solver.maximize().best_value.unwrap_or(isize::min_value());
        let end   = Instant::now();
        println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start), threads);
        opt
    }

    #[test]
    fn mcp_n30_p01_000() {
        assert_eq!(solve_id("mcp_n30_p0.1_000.mcp"), 13);
    }
    #[test]
    fn mcp_n30_p01_001() {
        assert_eq!(solve_id("mcp_n30_p0.1_001.mcp"), 18);
    }
    #[test]
    fn mcp_n30_p01_002() {
        assert_eq!(solve_id("mcp_n30_p0.1_002.mcp"), 15);
    }
    #[test]
    fn mcp_n30_p01_003() {
        assert_eq!(solve_id("mcp_n30_p0.1_003.mcp"), 19);
    }
    #[test]
    fn mcp_n30_p01_004() {
        assert_eq!(solve_id("mcp_n30_p0.1_004.mcp"), 16);
    }
    #[test]
    fn mcp_n30_p01_005() {
        assert_eq!(solve_id("mcp_n30_p0.1_005.mcp"), 19);
    }
    #[test]
    fn mcp_n30_p01_006() {
        assert_eq!(solve_id("mcp_n30_p0.1_006.mcp"), 12);
    }
    #[test]
    fn mcp_n30_p01_007() {
        assert_eq!(solve_id("mcp_n30_p0.1_007.mcp"), 18);
    }
    #[test]
    fn mcp_n30_p01_008() {
        assert_eq!(solve_id("mcp_n30_p0.1_008.mcp"), 20);
    }
    #[test]
    fn mcp_n30_p01_009() {
        assert_eq!(solve_id("mcp_n30_p0.1_009.mcp"), 22);
    }
}
