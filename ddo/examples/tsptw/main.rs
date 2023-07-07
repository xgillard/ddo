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

//! This is the main entry point of the program. This is what gets compiled to
//! the tsptw binary.

use std::{fs::File, path::Path, time::{Duration, Instant}};

use clap::Parser;
use ddo::{Completion, TimeBudget, NoDupFringe, MaxUB, Solution, SimpleDominanceChecker, Problem, DefaultBarrierSolver, Solver};
use dominance::TsptwDominance;
use heuristics::{TsptwWidth, TsptwRanking};
use instance::TsptwInstance;
use model::Tsptw;
use relax::TsptwRelax;

mod instance;
mod state;
mod model;
mod relax;
mod heuristics;
mod dominance;

#[cfg(test)]
mod tests;

/// TSPTW is a solver based on branch-and-bound mdd which solves the traveling
/// salesman problem with time windows to optimality. 
///
/// The implementation of tsptw is based on 
/// 'ddo: a generic and efficient framework for MDD-based optimization' (IJCAI20) 
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the TSP+TW instance that needs to be solved.
    instance: String,
    /// The maximum width of an mdd layer. The value you provide to this 
    /// argument will serve as a multiplicator to the default. Hence, 
    /// providing an argument value `width == 5` for an instance having 20 
    /// "cities" to visit, means that the maximum layer width will be 100.
    /// By default, the number of nodes equates to the number of unassigned
    /// variables.
    #[clap(short, long)]
    width: Option<usize>,
    /// How many threads do you want to use to solve the problem ?
    #[clap(short, long)]
    threads: Option<usize>,
    /// How long do you want the solver to keep working on your problem ? 
    /// (in seconds)
    #[clap(short, long)]
    duration: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let inst = TsptwInstance::from(File::open(&args.instance).unwrap());
    let pb = Tsptw::new(inst);
    let relax    = TsptwRelax::new(&pb);
    let width = TsptwWidth::new(pb.nb_variables(), args.width.unwrap_or(1));
    let dominance = SimpleDominanceChecker::new(TsptwDominance);
    let cutoff = TimeBudget::new(Duration::from_secs(args.duration.unwrap_or(u64::MAX)));
    let mut fringe = NoDupFringe::new(MaxUB::new(&TsptwRanking));
    let mut solver = DefaultBarrierSolver::custom(
        &pb, 
        &relax,
        &TsptwRanking,
        &width,
        &dominance,
        &cutoff,
        &mut fringe,
        args.threads.unwrap_or(num_cpus::get())
    );

    let start    = Instant::now();
    let outcome  = solver.maximize();
    let finish   = Instant::now();

    let instance = instance_name(&args.instance);
    let nb_vars   = pb.nb_variables();
    let lb       = objective(solver.best_lower_bound());
    let ub       = objective(solver.best_upper_bound());
    let solution = solver.best_solution();
    let duration = finish - start;

    print_solution(&instance, nb_vars, outcome, &lb, &ub, duration, solution);
}
fn print_solution(name: &str, n: usize, completion: Completion, lb: &str, ub: &str, duration: Duration, solution: Option<Solution>) {
    println!("instance : {name}");
    println!("status   : {}", status(completion));
    println!("lower bnd: {lb}");
    println!("upper bnd: {ub}");
    println!("duration : {}", duration.as_secs_f32());
    println!("solution : {}", solution_to_string(n, solution));
}
fn instance_name<P: AsRef<Path>>(fname: P) -> String {
    let name = fname.as_ref().file_name().unwrap().to_str().unwrap();
    let bench= fname.as_ref().parent().unwrap().file_name().unwrap().to_str().unwrap();

    format!("{}/{}", bench, name)
}
fn objective(x: isize) -> String {
    match x {
        isize::MIN => "+inf".to_string(),
        isize::MAX => "-inf".to_string(),
        _ => format!("{:.2}", -(x as f32 / 10_000.0_f32))
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