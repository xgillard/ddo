use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::time::SystemTime;

use structopt::StructOpt;

use ddo::abstraction::solver::Solver;
use ddo::implementation::mdd::config::mdd_builder;
use ddo::implementation::solver::parallel::ParallelSolver;

use crate::graph::Graph;
use crate::model::Minla;
use crate::relax::MinlaRelax;

mod model;
mod relax;
mod graph;

#[derive(StructOpt)]
struct Opt {
    /// Path to the instance (*.gra | *.dimacs)
    fname: String,
    /// Log the progression
    #[structopt(name="verbose", short, long, default_value = "1")]
    verbose: u8,
    /// The number of threads to use (default: number of physical threads on this machine)
    #[structopt(name="threads", short, long)]
    threads: Option<usize>,
    /// Use dimacs instead of gra
    #[structopt(name="dimacs", long)]
    dimacs: bool
}

fn main() {
    let opt = Opt::from_args();

    let threads = opt.threads.unwrap_or_else(num_cpus::get);
    let problem = if opt.dimacs { read_dimacs(&opt.fname) } else { read_gra(&opt.fname) }.unwrap();
    let relax   = MinlaRelax::new(&problem);
    let mdd     = mdd_builder(&problem, relax).into_deep();
    let mut solver  = ParallelSolver::customized(mdd, opt.verbose, threads);

    let start = SystemTime::now();
    let opt = solver.maximize().best_value.unwrap_or(isize::min_value());
    let end = SystemTime::now();

    println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
}

fn read_gra(fname: &str) -> Result<Minla, io::Error> {
    let file = File::open(fname).expect("File not found.");
    let buffered = BufReader::new(file);

    let mut numbers = Vec::new();
    for line in buffered.lines() {
        let line = line?;
        let iter = line.trim().split_whitespace();
        for x in iter {
            let x = x.parse::<i32>().unwrap();
            if x >= 0 {
                numbers.push(x as usize)
            }
        }
    }

    let n = numbers[0];
    let mut g = vec![vec![0; n]; n];

    let mut cumul= 2+n;
    for i in 0..n {
        for j in cumul..(cumul+numbers[2+i]) {
            g[i][numbers[j]] = 1;
            g[numbers[j]][i] = 1;
        }
        cumul += numbers[2+i];
    }

    Ok(Minla::new(g))
}

fn read_dimacs(fname: &str) -> Result<Minla, io::Error> {
    let file = File::open(fname).expect("File not found.");
    let graph = Graph::from(file);

    let n = graph.nb_vertices;
    let mut g = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            g[i][j] = graph[(i,j)];
        }
    }

    Ok(Minla::new(g))
}