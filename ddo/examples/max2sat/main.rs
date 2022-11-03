use std::time::{Duration, Instant};

use clap::Parser;
use ddo::{TimeBudget, NoCutoff, Cutoff, FixedWidth, DefaultSolver, NoDupFrontier, MaxUB, Solver, Completion, WidthHeuristic, NbUnassignedWitdh, Problem, Decision, Variable};
use model::{f, t};

use crate::{heuristics::Max2SatRanking, model::{Max2Sat, v}, relax::Max2SatRelax, data::read_instance};

mod errors;
mod heuristics;
mod data;
mod model;
mod relax;

/// Solve max2sat instance
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Params {
    /// the instance file
    #[arg(short, long)]
    file: String,
    /// maximum width in a layer
    #[arg(short, long)]
    width: Option<usize>,
    /// max time to find the solution
    #[arg(short, long)]
    timeout: Option<u64>,
}

fn main() {
    let Params{file, width, timeout} = Params::parse();
    let problem = Max2Sat::new(read_instance(&file).unwrap());
    let relax = Max2SatRelax(&problem);
    let rank = Max2SatRanking;
    let width = max_width(&problem, width);
    let cutoff = cutoff(timeout);
    let mut fringe = NoDupFrontier::new(MaxUB::new(&Max2SatRanking));

    let mut solver = DefaultSolver::new(
        &problem, 
        &relax, 
        &rank, 
        width.as_ref(), 
        cutoff.as_ref(), 
        &mut fringe);

        let start = Instant::now();
        let Completion{ is_exact, best_value } = solver.maximize();
        
        let duration = start.elapsed();
        let upper_bound = solver.best_upper_bound();
        let lower_bound = solver.best_lower_bound();
        let gap = solver.gap();
        let best_solution  = solver.best_solution().map(|mut decisions|{
            decisions.sort_unstable_by_key(|d| d.variable.id());
            decisions.iter().map(|d| v(d.variable) * d.value).collect()
        });
    
        println!("Duration:   {:.3} seconds", duration.as_secs_f32());
        println!("Objective:  {}",            best_value.unwrap_or(-1));
        println!("Upper Bnd:  {}",            upper_bound);
        println!("Lower Bnd:  {}",            lower_bound);
        println!("Gap:        {:.3}",         gap);
        println!("Aborted:    {}",            !is_exact);
        println!("Cost:       {:?}",          solution_cost(&problem, &solver.best_solution()));
        println!("Solution:   {:?}",          best_solution.unwrap_or(vec![]));
}

fn cutoff(timeout: Option<u64>) -> Box<dyn Cutoff + Send + Sync> {
    if let Some(t) = timeout {
        Box::new(TimeBudget::new(Duration::from_secs(t)))
    } else {
        Box::new(NoCutoff)
    }
}
fn max_width<P: Problem>(p: &P, w: Option<usize>) -> Box<dyn WidthHeuristic<P::State> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
    } else {
        Box::new(NbUnassignedWitdh(p.nb_variables()))
    }
}

fn solution_cost(pb: &Max2Sat, solution: &Option<Vec<Decision>>) -> isize {
    if let Some(sol) = solution {
        let n = pb.nb_vars;
        let mut model = vec![0; n];
        for d in sol.iter() {
            model[d.variable.id()] = d.value;
        }

        let mut cost = 0;
        for i in 0..n {
            for j in i..n {
                if model[i] == 1 && model[j] == 1 {
                    cost += pb.weight(f(Variable(i)), f(Variable(j)))
                }
                if model[i] ==-1 && model[j] ==-1 {
                    cost += pb.weight(t(Variable(i)), t(Variable(j)))
                }
                if model[i] == 1 && model[j] ==-1 {
                    cost += pb.weight(f(Variable(i)), t(Variable(j)))
                }
                if model[i] ==-1 && model[j] == 1 {
                    cost += pb.weight(t(Variable(i)), f(Variable(j)))
                }
            }
        }
        cost
    } else {
        0
    }
}