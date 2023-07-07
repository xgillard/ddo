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

//! This example uses ddo to solve the Single-Row Facility Layout Problem
//! Instances can be downloaded from https://miguelanjos.com/flplib and
//! https://www.philipphungerlaender.com/benchmark-libraries/layout-lib/row-layout-instances

use std::{time::{Duration, Instant}};

use clap::Parser;
use ddo::*;
use heuristics::SrflpWidth;
use state::SrflpState;

use crate::{io_utils::read_instance, model::Srflp, relax::SrflpRelax, heuristics::SrflpRanking};

mod model;
mod relax;
mod state;
mod heuristics;
mod io_utils;

#[cfg(test)]
mod tests;

/// This structure uses `clap-derive` annotations and define the arguments that can
/// be passed on to the executable solver.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the instance file
    fname: String,
    /// The number of concurrent threads
    #[clap(short, long, default_value = "8")]
    threads: usize,
    /// The maximum amount of time you would like this solver to run
    #[clap(short, long)]
    duration: Option<u64>,
    /// The maximum number of nodes per layer
    #[clap(short, long)]
    width: Option<usize>,
}

/// An utility function to return an max width heuristic that can either be a fixed width
/// policy (if w is fixed) or an adaptive policy returning the number of unassigned variables
/// in the overall problem.
fn max_width<P: Problem>(p: &P, w: Option<usize>) -> Box<dyn WidthHeuristic<SrflpState> + Send + Sync> {
    if let Some(w) = w {
        Box::new(SrflpWidth::new(p.nb_variables(), w))
    } else {
        Box::new(NbUnassignedWitdh(p.nb_variables()))
    }
}
/// An utility function to return a cutoff heuristic that can either be a time budget policy
/// (if timeout is fixed) or no cutoff policy.
fn cutoff(timeout: Option<u64>) -> Box<dyn Cutoff + Send + Sync> {
    if let Some(t) = timeout {
        Box::new(TimeBudget::new(Duration::from_secs(t)))
    } else {
        Box::new(NoCutoff)
    }
}

/// This is your executable's entry point. It is the place where all the pieces are put together
/// to create a fast an effective solver for the srflp problem.
fn main() {
    let args = Args::parse();
    let fname = &args.fname;
    let instance = read_instance(fname).unwrap();
    let problem = Srflp::new(instance);
    let relaxation = SrflpRelax::new(&problem);
    let ranking = SrflpRanking;

    let width = max_width(&problem, args.width);
    let dominance = EmptyDominanceChecker::default();
    let cutoff = cutoff(args.duration);
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    let mut solver = DefaultBarrierSolver::custom(
        &problem, 
        &relaxation, 
        &ranking, 
        width.as_ref(), 
        &dominance,
        cutoff.as_ref(), 
        &mut fringe,
        args.threads,
    );

    let start = Instant::now();
    let Completion{ is_exact, best_value } = solver.maximize();
    
    let duration = start.elapsed();
    let upper_bound = - solver.best_upper_bound();
    let lower_bound = - solver.best_lower_bound();
    let gap = solver.gap();
    let best_solution: Option<Vec<_>>  = solver.best_solution()
        .map(|mut decisions|{
            decisions.sort_unstable_by_key(|d| d.variable.id());
            decisions.iter()
                .map(|d| d.value)
                .collect()
        });
    let best_solution = best_solution.unwrap_or_default();
    let best_value = best_value.map(|v| - v as f64 + problem.root_value()).unwrap_or(-1.0);
    
    println!("Duration:   {:.3} seconds", duration.as_secs_f32());
    println!("Objective:  {}",            best_value);
    println!("Upper Bnd:  {}",            upper_bound);
    println!("Lower Bnd:  {}",            lower_bound);
    println!("Gap:        {:.3}",         gap);
    println!("Aborted:    {}",            !is_exact);
    println!("Solution:   {:?}",          best_solution);
}
