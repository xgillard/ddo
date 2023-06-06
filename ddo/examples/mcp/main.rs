use std::{time::{Duration, Instant}, fs::File};

use clap::Parser;
use ddo::*;

use crate::{graph::Graph, model::{Mcp, McpRanking}, relax::McpRelax};

mod graph;
mod model;
mod relax;
#[cfg(test)]
mod tests;

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
    let graph = Graph::from(File::open(&file).expect("could not open file"));
    let problem = Mcp::from(graph);
    let relax = McpRelax::new(&problem);
    let rank = McpRanking;
    let width = max_width(&problem, width);
    let dominance = EmptyDominanceChecker::default();
    let cutoff = cutoff(timeout);
    let mut fringe = NoDupFringe::new(MaxUB::new(&rank));

    let mut solver = DefaultSolver::new(
        &problem, 
        &relax, 
        &rank, 
        width.as_ref(), 
        &dominance,
        cutoff.as_ref(), 
        &mut fringe,
    );

        let start = Instant::now();
        let Completion{ is_exact, best_value } = solver.maximize();
        
        let duration = start.elapsed();
        let upper_bound = solver.best_upper_bound();
        let lower_bound = solver.best_lower_bound();
        let gap = solver.gap();
        let best_solution  = solver.best_solution();
    
        println!("Duration:   {:.3} seconds", duration.as_secs_f32());
        println!("Objective:  {}",            best_value.unwrap_or(-1));
        println!("Upper Bnd:  {}",            upper_bound);
        println!("Lower Bnd:  {}",            lower_bound);
        println!("Gap:        {:.3}",         gap);
        println!("Aborted:    {}",            !is_exact);
        println!("Solution:   {:?}",          best_solution.unwrap_or_default());
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