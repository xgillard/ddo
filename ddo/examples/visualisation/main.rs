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

//! This example show how to visualize a decision diagram you have compiled. It does
//! so by using the knapsack example that has already been used over and over..
use std::{path::{Path, PathBuf}, fs::File, io::{BufReader, BufRead}, num::ParseIntError, sync::Arc};

use clap::Parser;
use ddo::*;

/// In our DP model, we consider a state that simply consists of the remaining 
/// capacity in the knapsack. Additionally, we also consider the *depth* (number
/// of assigned variables) as part of the state since it useful when it comes to
/// determine the next variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KnapsackState {
    /// the number of variables that have already been decided upon in the complete
    /// problem.
    depth: usize,
    /// the remaining capacity in the knapsack. That is the maximum load the sack
    /// can bear without cracking **given what is already in the sack**.
    capacity: usize
}

/// This structure represents a particular instance of the knapsack problem.
/// This is the structure that will implement the knapsack model.
/// 
/// The problem definition is quite easy to understand: there is a knapsack having 
/// a maximum (weight) capacity, and a set of items to chose from. Each of these
/// items having a weight and a profit, the goal is to select the best subset of
/// the items to place them in the sack so as to maximize the profit.
pub struct Knapsack {
    /// The maximum capacity of the sack (when empty)
    capacity: usize,
    /// the profit of each item
    profit: Vec<usize>,
    /// the weight of each item.
    weight: Vec<usize>,
}

/// For each variable in the decision problem, there are two possible choices:
/// either we take the item in the sack, or we decide to leave it out. This
/// constant is used to indicate that the item is to be taken in the sack.
const TAKE_IT: isize = 1;
/// For each variable in the decision problem, there are two possible choices:
/// either we take the item in the sack, or we decide to leave it out. This
/// constant is used to indicate that the item is to be left out of the sack.
const LEAVE_IT_OUT: isize = 0;

/// This is how you implement the labeled transition system (LTS) semantics of
/// a simple dynamic program solving the knapsack problem. The definition of
/// each of the methods should be pretty clear and easy to grasp. Should you
/// want more details on the role of each of these methods, then you are 
/// encouraged to go checking the documentation of the `Problem` trait.
impl Problem for Knapsack {
    type State = KnapsackState;

    fn nb_variables(&self) -> usize {
        self.profit.len()
    }
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
    {
        if state.capacity >= self.weight[variable.id()] {
            f.apply(Decision { variable, value: TAKE_IT });
            f.apply(Decision { variable, value: LEAVE_IT_OUT });
        } else {
            f.apply(Decision { variable, value: LEAVE_IT_OUT });
        }
    }
    fn initial_state(&self) -> Self::State {
        KnapsackState{ depth: 0, capacity: self.capacity }
    }
    fn initial_value(&self) -> isize {
        0
    }
    fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
        let mut ret = *state;
        ret.depth  += 1;
        if dec.value == TAKE_IT { 
            ret.capacity -= self.weight[dec.variable.id()] 
        }
        ret
    }
    fn transition_cost(&self, _state: &Self::State, _: &Self::State, dec: Decision) -> isize {
        self.profit[dec.variable.id()] as isize * dec.value
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        let n = self.nb_variables();
        if depth < n {
            Some(Variable(depth))
        } else {
            None
        }
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
/// In addition to the aforementioned two operations, the KPRelax structure implements
/// an optional `fast_upper_bound` method. Which one provides a useful bound to 
/// prune some portions of the state-space as the decision diagrams are compiled.
/// (aka rough upper bound pruning).
pub struct KPRelax<'a>{pub pb: &'a Knapsack}
impl Relaxation for KPRelax<'_> {
    type State = KnapsackState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        states.max_by_key(|node| node.capacity).copied().unwrap()
    }

    fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let mut tot = 0;
        for var in state.depth..self.pb.nb_variables() {
            if self.pb.weight[var] <= state.capacity {
                tot += self.pb.profit[var];
            }
        }
        tot as isize
    }
}

/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
pub struct KPranking;
impl StateRanking for KPranking {
    type State = KnapsackState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.capacity.cmp(&b.capacity)
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
    #[clap(short, long, default_value = "30")]
    duration: u64,
    /// The maximum width of a layer when solving an instance. By default, it will allow
    /// as many nodes in a layer as there are unassigned variables in the global problem.
    #[clap(short, long)]
    width: Option<usize>,
}

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// knapsack instance from file. There can be io errors (file unavailable ?), format error
/// (e.g. the file is not a knapsack instance but contains the text of your next paper), 
/// or parse int errors (which are actually a variant of the format error since it tells 
/// you that the parser expected an integer number but got ... something else).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// There was an io related error
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    /// The parser expected to read something that was an integer but got some garbage
    #[error("parse int {0}")]
    ParseInt(#[from] ParseIntError),
    /// The file was not properly formatted.
    #[error("ill formed instance")]
    Format,
}

/// This function is used to read a knapsack instance from file. It returns either a
/// knapsack instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<Knapsack, Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    
    let mut is_first = true;
    let mut n = 0;
    let mut count = 0;
    let mut capa = 0;
    let mut profit = vec![];
    let mut weight = vec![];

    for line in f.lines() {
        let line = line?;
        if line.starts_with('c') {
            continue;
        }
        if is_first {
            is_first = false;
            let mut ab = line.split(' ');
            n = ab.next().ok_or(Error::Format)?.parse()?;
            capa = ab.next().ok_or(Error::Format)?.parse()?;
        } else {
            if count >= n {
                break;
            }
            let mut ab = line.split(' ');
            profit.push(ab.next().ok_or(Error::Format)?.parse()?);
            weight.push(ab.next().ok_or(Error::Format)?.parse()?);
            count += 1;
        }
    }
    Ok(Knapsack { capacity: capa, profit, weight })
}

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/knapsack/")
        .join(id)
}

/// This is your executable's entry point. It is the place where all the pieces are put together
/// to create a fast an effective solver for the knapsack problem.
fn main() {
    let fname = locate("f1_l-d_kp_10_269");
       
    let problem = read_instance(fname).unwrap();
    let relaxation = KPRelax{pb: &problem};
    let ranking = KPranking;
    let mut cache = SimpleCache::default();
    cache.initialize(&problem);
    let dominance = EmptyDominanceChecker::default();

    let residual = SubProblem { 
        state: Arc::new(problem.initial_state()), 
        value: 0, 
        path: vec![], 
        ub: isize::MAX, 
        depth: 0
     };
    let input = CompilationInput {
        comp_type: CompilationType::Relaxed,
        problem: &problem,
        relaxation: &relaxation,
        ranking: &ranking,
        cutoff: &NoCutoff,
        max_width: 5,
        residual: &residual,
        best_lb: isize::MIN,
        cache: &cache,
        dominance: &dominance,
    };

    let mut clean = Mdd::<KnapsackState, {FRONTIER}>::new();
    _ = clean.compile(&input);

    let config = VizConfigBuilder::default()
        .show_deleted(true)
        .group_merged(true)
        .build()
        .unwrap();
    
    let dot = clean.as_graphviz(&config);
    println!("{dot}");
}
