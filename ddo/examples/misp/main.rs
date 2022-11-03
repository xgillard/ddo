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

//! This example show how to implement a solver for the maximum independent set problem 
//! using ddo. It is a fairly simple example but it features most of the aspects you will
//! want to copy when implementing your own solver.
use std::{cell::RefCell, path::Path, fs::File, io::{BufReader, BufRead}, num::ParseIntError, time::{Duration, Instant}};

use bit_set::BitSet;
use clap::Parser;
use ddo::*;
use regex::Regex;

/// This structure represents an instance of the Maximum Idependent Set Problem. 
/// It is this structure that implements a simple dynamic programming model for the
/// MISP. In that model, the state is simply a bitset where each bit represents 
/// a node that may be kept or left out of the MIS. 
struct Misp {
    /// The number of variables in the problem instance
    nb_vars: usize,
    /// For each vertex 'i' of the original graph, the field 'neighbors[i]' contains
    /// a bitmask representing the COMPLEMENT of the adjacency list of i in the 
    /// original graph. While this may seem a complicated take on the representation
    /// of this problem instance, using the complement is helpful as it allows to
    /// easily remove all the neighbors of a vertex from a state very efficiently.
    neighbors: Vec<BitSet>,
    /// For each vertex 'i', the value of 'weight[i]' denotes the weight associated
    /// to vertex i in the problem instance. The goal of MISP is to select the nodes
    /// from the underlying graph such that the resulting set is an independent set
    /// where the sum of the weights of selected vertices is maximum.
    weight: Vec<isize>,
}

/// A constant to mean take the node in the independent set.
const YES: isize = 1;
/// A constant to mean leave the node out of the independent set.
const NO: isize = 0;

/// The Misp class implements the 'Problem' trait. This means Misp is the definition
/// of the DP model. That DP model is pretty straightforward, still you might want
/// to check the implementation of the branching heuristic (next_variable method)
/// since it does interesting stuffs. 
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

    /// This method is (apparently) a bit more hairy. What it does is it simply decides to branch on
    /// the variable that occurs in the least number of states present in the next layer. The intuition
    /// here is to limit the max width as much as possible when developing the layers since all 
    /// nodes that are not impacted by the change on the selectd vertex are simply copied over to the
    /// next layer.
    fn next_variable(&self, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        // The thread local stuff is possibly one of the most surprising bits of this code. It declares
        // a static variable called VAR_HEURISTIC storing the counts of each vertex in the next layer.
        // The fact that it is static means that it will not be re-created (re allocated) upon each
        // pass. The fact that it is declared within a thread local block, means that this static var
        // will be created with a potentially mutable access for each thread.
        thread_local! {
            static VAR_HEURISTIC: RefCell<Vec<usize>> = RefCell::new(vec![]);
        }
        VAR_HEURISTIC.with(|heu| {
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

/// In addition to a dynamic programming (DP) model of the problem you want to solve, 
/// the branch and bound with MDD algorithm (and thus ddo) requires that you provide
/// an additional relaxation allowing to control the maximum amount of space used by
/// the decision diagrams that are compiled. 
/// 
/// That relaxation requires two operations: one to merge several nodes into one 
/// merged node that acts as an over approximation of the other nodes. The second
/// operation is used to possibly offset some weight that would otherwise be lost 
/// to the arcs entering the newly created merged node.
/// 
/// The role of this very simple structure is simply to provide an implementation
/// of that relaxation.
/// 
/// # Note:
/// In addition to the aforementioned two operations, the MispRelax structure implements
/// an optional `fast_upper_bound` method. Whichone provides a useful bound to 
/// prune some portions of the state-space as the decision diagrams are compiled.
/// (aka rough upper bound pruning).
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


/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
struct MispRanking;
impl StateRanking for MispRanking {
    type State = BitSet;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.len().cmp(&b.len())
            .then_with(|| a.cmp(b))
    }
}


// #########################################################################################
// # THE INFORMATION BEYOND THIS LINE IS NOT DIRECTLY RELATED TO THE IMPLEMENTATION OF     #
// # A SOLVER BASED ON DDO. INSTEAD, THAT PORTION OF THE CODE CONTAINS GENERIC FUNCTION    #
// # THAT ARE USED TO READ AN INSTANCE FROM FILE, PROCESS COMMAND LINE ARGUMENTS, AND      #
// # THE MAIN FUNCTION. THESE ARE THUS NOT REQUIRED 'PER-SE', BUT I BELIEVE IT IS USEFUL   #
// # TO SHOW HOW IT CAN BE DONE IN AN EXAMPLE.                                             #
// #########################################################################################

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

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// misp instance from file. There can be io errors (file unavailable ?), format error
/// (e.g. the file is not an instance but contains the text of your next paper), 
/// or parse int errors (which are actually a variant of the format errror since it tells 
/// you that the parser expected an integer number but got ... something else).
#[derive(Debug, thiserror::Error)]
enum Error {
    /// There was an io related error
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    /// The parser expected to read somehting that was an integer but got some garbage
    #[error("parse int {0}")]
    ParseInt(#[from] ParseIntError),
    /// The file was not properly formatted.
    #[error("ill formed instance")]
    Format,
}

/// This funciton is used to read a misp instance from file. It returns either a
/// misp instance if everything went on well or an error describing the problem.
fn read_instance<P: AsRef<Path>>(fname: P) -> Result<Misp, Error> {
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
        return Err(Error::Format)
    }

    Ok(g)
}

/// An utility function to return an max width heuristic that can either be a fixed width
/// policy (if w is fixed) or an adaptative policy returning the number of unassigned variables
/// in the overall problem.
fn max_width<P: Problem>(p: &P, w: Option<usize>) -> Box<dyn WidthHeuristic<P::State> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
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
/// to create a fast an effectuve solver for the knapsack problem.
fn main() {
    let args = Args::parse();
    let fname = &args.fname;
    let problem = read_instance(fname).unwrap();
    let relaxation = MispRelax {pb: &&problem};
    let ranking = MispRanking;

    let width = max_width(&problem, args.width);
    let cutoff = cutoff(args.duration);
    let mut fringe = NoDupFrontier::new(MaxUB::new(&ranking));

    let mut solver = DefaultSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        width.as_ref(), 
        cutoff.as_ref(), 
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

        
    println!("Duration:   {:.3} seconds", duration.as_secs_f32());
    println!("Objective:  {}",            best_value.unwrap_or(-1));
    println!("Upper Bnd:  {}",            upper_bound);
    println!("Lower Bnd:  {}",            lower_bound);
    println!("Gap:        {:.3}",         gap);
    println!("Aborted:    {}",            !is_exact);
    println!("Solution:   {:?}",          best_solution.unwrap_or(vec![]));
}
