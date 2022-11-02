use std::{cell::RefCell, path::Path, fs::File, io::{BufReader, BufRead}, num::ParseIntError, time::{Duration, Instant}};

use bit_set::BitSet;
use clap::Parser;
use ddo::*;
use regex::Regex;

struct Misp {
    nb_vars  : usize,
    neighbors: Vec<BitSet>,
    weight   : Vec<isize>,
}

const YES: isize = 1;
const NO : isize = 0;

impl Problem for Misp {
    type State = BitSet;

    fn nb_variables(&self) -> usize {
        self.nb_vars
    }

    fn initial_state(&self) -> Self::State {
        (0..self.nb_variables()).collect()
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn transition(&self, state: &Self::State, decision: Decision) -> Self::State {
        let mut res = state.clone();
        res.remove(decision.variable.id());
        if decision.value == YES {
            // intersect with complement of the neighbors for fast set difference
            res.intersect_with(&self.neighbors[decision.variable.id()]); 
        }
        res
    }

    fn transition_cost(&self, _: &Self::State, decision: Decision) -> isize {
        if decision.value == NO {
            0
        } else {
            self.weight[decision.variable.id()]
        }
    }

    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback) {
        if state.contains(variable.id()) {
            f.apply(Decision{variable, value: YES});
            f.apply(Decision{variable, value: NO });
        } else {
            f.apply(Decision{variable, value: NO });
        }
    }

    fn next_variable(&self, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        Misp::VAR_HEURISTIC.with(|heu| {
            let mut heu = heu.borrow_mut();
            let heu: &mut Vec<usize> = heu.as_mut();

            // initialize
            heu.reserve_exact(self.nb_variables());
            if heu.is_empty() {
                for _ in 0..self.nb_variables() { heu.push(0); }
            } else {
                heu.iter_mut().for_each(|i| *i = 0);
            }
            
            // count the occurence of each var
            for s in next_layer {
                for sit in s.iter() {
                    heu[sit] += 1;
                }
            }

            // take the one occurring the least often
            heu.iter().copied().enumerate()
                .filter(|(_, v)| *v > 0)
                .min_by_key(|(_, v)| *v)
                .map(|(x, _)| Variable(x))
        })
    }

}

impl Misp {
    thread_local! {
        static VAR_HEURISTIC: RefCell<Vec<usize>> = RefCell::new(vec![]);
    }
}

struct MispRelax<'a>{pb: &'a Misp}
impl Relaxation for MispRelax<'_> {
    type State = BitSet;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut state = BitSet::with_capacity(self.pb.nb_variables());
        for s in states {
            state.union_with(s);
        }
        state
    }

    fn relax(
        &self,
        _source: &Self::State,
        _dest: &Self::State,
        _new: &Self::State,
        _decision: Decision,
        cost: isize,
    ) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        state.iter().map(|x| self.pb.weight[x]).sum()
    }
}

struct MispRanking;
impl StateRanking for MispRanking {
    type State = BitSet;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.len().cmp(&b.len())
            .then_with(|| a.cmp(b))
    }
}

// #### ONLY USEFUL FOR THE EXAMPLE #######################################################

#[derive(Debug, thiserror::Error)]
enum Errors {
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    #[error("parse int {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("ill formed instance")]
    Format,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// An easy way to solve the maximum independent set problem efficiently with 
/// branch and bound MDD
struct Args {
    /// The path to the instance file
    fname: String,
    /// The number of concurrent threads
    #[clap(short, long, default_value = "8")]
    threads: usize,
    /// The maximum amount of time you would like this solver to run
    #[clap(short, long, default_value = "30")]
    duration: u64,
}

fn read_instance<P: AsRef<Path>>(fname: P) -> Result<Misp, Errors> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    
    let comment   = Regex::new(r"^c\s.*$").unwrap();
    let pb_decl   = Regex::new(r"^p\s+edge\s+(?P<vars>\d+)\s+(?P<edges>\d+)$").unwrap();
    let node_decl = Regex::new(r"^n\s+(?P<node>\d+)\s+(?P<weight>-?\d+)").unwrap();
    let edge_decl = Regex::new(r"^e\s+(?P<src>\d+)\s+(?P<dst>\d+)").unwrap();

    let mut g = Misp{nb_vars: 0, neighbors: vec![], weight: vec![]};
    for line in f.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if comment.is_match(&line) {
            continue;
        }

        if let Some(caps) = pb_decl.captures(&line) {
            let n = caps["vars"].to_string().parse::<usize>()?;
            let full = (0..n).collect(); 

            g.nb_vars    = n;
            g.neighbors  = vec![full; n];
            g.weight     = vec![1; n];
            continue;
        }

        if let Some(caps) = node_decl.captures(&line) {
            let n = caps["node"].to_string().parse::<usize>()?;
            let w = caps["weight"].to_string().parse::<isize>()?;

            let n = n - 1;
            g.weight[n] = w;
            continue;
        }

        if let Some(caps) = edge_decl.captures(&line) {
            let src = caps["src"].to_string().parse::<usize>()?;
            let dst = caps["dst"].to_string().parse::<usize>()?;

            let src = src-1;
            let dst = dst-1;

            g.neighbors[src].remove(dst);
            g.neighbors[dst].remove(src);

            continue;
        }

        // skip
        return Err(Errors::Format)
    }

    Ok(g)
}


fn main() {
    let args = Args::parse();
    let fname = &args.fname;
    let problem = read_instance(fname).unwrap();
    let relaxation = MispRelax {pb: &&problem};
    let ranking = MispRanking;

    let width = NbUnassignedWitdh(problem.nb_variables());
    let cutoff = TimeBudget::new(Duration::from_secs(args.duration));
    let mut fringe = NoDupFrontier::new(MaxUB::new(&ranking));

    let mut solver = DefaultSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &cutoff, 
        &mut fringe)
    .with_nb_threads(args.threads);

    let start = Instant::now();
    let Completion{ is_exact, best_value } = solver.maximize();
    
    let duration = start.elapsed();
    let upper_bound = solver.best_upper_bound();
    let lower_bound = solver.best_lower_bound();
    let gap = solver.gap();
    let best_solution  = solver.best_solution().map(|mut decisions|{
        decisions.sort_unstable_by_key(|d| d.variable.id());
        decisions.iter().map(|d| d.value).collect()
    });

    println!("Duration:   {:.3} seconds \nObjective:  {}\nUpper Bnd:  {}\nLower Bnd:  {}\nGap:        {:.3}\nAborted:    {}\nSolution:   {:?}",
            duration.as_secs_f32(), 
            best_value.unwrap_or(-1), 
            upper_bound, 
            lower_bound, 
            gap, 
            !is_exact, 
            best_solution.unwrap_or(vec![]));
}