mod model;
mod relax;

use model::Minla;
use relax::MinlaRelax;
use ddo::core::implementation::mdd::builder::mdd_builder_ref;
use ddo::core::abstraction::solver::Solver;
use ddo::core::implementation::heuristics::{NaturalOrder, FixedWidth};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::io;
use std::time::SystemTime;
use ddo::core::implementation::solver::parallel::ParallelSolver;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    /// Path to the instance (*.gra)
    fname: String,
    /// Log the progression
    #[structopt(name="verbose", short, long, default_value = "1")]
    verbose: u8,
    /// The number of threads to use (default: number of physical threads on this machine)
    #[structopt(name="threads", short, long)]
    threads: Option<usize>,
    /// If specified, the max width allowed for any layer
    #[structopt(name="width", short, long)]
    width: Option<usize>
}

fn main() {
    let opt = Opt::from_args();

    let minla = read_gra(&opt.fname).unwrap();
    let relax = MinlaRelax::new(&minla);
    let width = match opt.width {
        Some(x) => x,
        None => 1000
    };
    let threads = match opt.threads {
        Some(x) => x,
        None => num_cpus::get()
    };

    let mdd = mdd_builder_ref(&minla, relax)
        .with_max_width(FixedWidth(width))
        .with_branch_heuristic(NaturalOrder)
        .into_flat();

    let mut solver = ParallelSolver::customized(mdd, opt.verbose, threads);

    let start = SystemTime::now();
    let (opt, _) = solver.maximize();
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

    Ok(Minla { g })
}