use std::{fs::File, path::Path, time::{Duration, Instant}};

use clap::Parser;
use ddo::{TimeBudget, NoCutoff, Cutoff, FixedWidth, DefaultSolver, NoDupFrontier, MaxUB, Solver, Completion, WidthHeuristic, NbUnassignedWitdh, Problem};

use crate::{heuristics::Max2SatRanking, model::Max2Sat, relax::Max2SatRelax};

mod heuristics;
mod instance;
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

fn main() -> Result<(), std::io::Error>{
    let Params{file, width, timeout} = Params::parse();

    let path: &dyn AsRef<Path> = &file;
    let name = path.as_ref().file_stem().map(|x| x.to_string_lossy()).unwrap();

    let problem = Max2Sat::from(File::open(&file)?);
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
    let Completion{is_exact, best_value} = solver.maximize();
    let duration = start.elapsed();
    let best_value = best_value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "not found".to_owned());

    let lb = solver.best_lower_bound();
    let ub = solver.best_upper_bound();
    let gap = solver.gap();

    let status = if is_exact {
        "proved"
    } else {
        "timeout"
    };

    println!(
        "{:>30} | {:>15} | {:>8.2} | {:>10} | {:>10} | {:>10} | {:>5.4}",
        name,
        status,
        duration.as_secs_f32(),
        best_value,
        lb,
        ub,
        gap
    );

    Ok(())
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