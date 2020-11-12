use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::time::SystemTime;

use structopt::StructOpt;

use ddo::abstraction::solver::Solver;
use ddo::implementation::mdd::config::config_builder;
use ddo::implementation::solver::parallel::ParallelSolver;
use ddo::implementation::frontier::NoDupFrontier;
use ddo::implementation::heuristics::NbUnassignedWitdh;

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
    /// The number of threads to use (default: number of physical threads on this machine)
    #[structopt(name="threads", short, long)]
    threads: Option<usize>
}

fn main() {
    let opt = Opt::from_args();

    let threads = opt.threads.unwrap_or_else(num_cpus::get);
    let problem = read_file(&opt.fname).unwrap();
    let relax   = MinlaRelax::new(&problem);
    let mdd  = config_builder(&problem, relax)
        .with_max_width(NbUnassignedWitdh)
        .into_deep();
    let mut solver  = ParallelSolver::customized(mdd, 2, threads)
        .with_frontier(NoDupFrontier::default()); // miracle !

    let start = SystemTime::now();
    let opt = solver.maximize().best_value.unwrap_or(isize::min_value());
    let end = SystemTime::now();

    println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
}

fn read_file(fname: &str) -> Result<Minla, io::Error> {
    if fname.contains("gra") {
        read_gra(fname)
    } else if fname.contains("dimacs") {
        read_dimacs(fname)
    } else if fname.contains("mtx") {
        read_mtx(fname)
    } else {
        read_gra(fname)
    }
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

// FIXME: @vcoppe: maybe give variables n,g,x longer names to be more explicit
#[allow(clippy::many_single_char_names)]
fn read_mtx(fname: &str) -> Result<Minla, io::Error> {
    let file = File::open(fname).expect("File not found.");
    let buffered = BufReader::new(file);
    let mut n = 0;
    let mut g = vec![];

    for line in buffered.lines() {
        let line = line?;

        if line.starts_with('%') {
            continue;
        }

        let x: Vec<f32> = line.trim().split_whitespace().map(|s| s.parse::<f32>().unwrap()).collect();

        if n == 0 {
            n = x[0] as usize;
            g = vec![vec![0; n]; n];
        } else {
            let i = (x[0] as usize)-1;
            let j = (x[1] as usize)-1;
            g[i][j] = 1;
            g[j][i] = 1;
        }
    }

    Ok(Minla::new(g))
}
