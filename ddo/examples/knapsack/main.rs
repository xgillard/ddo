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

//! This example show how to implement a solver for the knapsack problem using ddo.
//! It is a fairly simple example but  features most of the aspects you will want to
//! copy when implementing your own solver.
use std::{path::Path, fs::File, io::{BufReader, BufRead}, time::{Duration, Instant}, num::ParseIntError, sync::Arc};

use clap::Parser;
use ddo::*;
use ordered_float::OrderedFloat;

#[cfg(test)]
mod tests;

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
    /// can bear withouth cracking **given what is already in the sack**.
    capacity: usize
}

/// This structure represents a particular instance of the knapsack problem.
/// This is the sctructure that will implement the knapsack model.
/// 
/// The problem definition is quite easy to understand: there is a knapsack having 
/// a maximum (weight) capacity, and a set of items to chose from. Each of these
/// items having a weight and a profit, the goal is to select the best subset of
/// the items to place them in the sack so as to maximize the profit.
pub struct Knapsack {
    /// The maximum capacity of the sack (when empty)
    capacity: usize,
    /// the profit of each item
    profit: Vec<isize>,
    /// the weight of each item.
    weight: Vec<usize>,
    /// the order in which the items are considered
    order: Vec<usize>,
}

impl Knapsack {
    pub fn new(capacity: usize, profit: Vec<isize>, weight: Vec<usize>) -> Self {
        let mut order = (0..profit.len()).collect::<Vec<usize>>();
        order.sort_unstable_by_key(|i| OrderedFloat(- profit[*i] as f64 / weight[*i] as f64));

        Knapsack { capacity, profit, weight, order }
    }
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
        }
        f.apply(Decision { variable, value: LEAVE_IT_OUT });
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
    fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
        self.profit[dec.variable.id()] * dec.value
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        let n = self.nb_variables();
        if depth < n {
            Some(Variable(self.order[depth]))
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
/// an optional `fast_upper_bound` method. Whichone provides a useful bound to 
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
        let mut depth = state.depth;
        let mut max_profit = 0;
        let mut capacity = state.capacity;

        while capacity > 0 && depth < self.pb.profit.len() {
            let item = self.pb.order[depth];

            if capacity >= self.pb.weight[item] {
                max_profit += self.pb.profit[item];
                capacity -= self.pb.weight[item];
            } else {
                let item_ratio = capacity as f64 / self.pb.weight[item] as f64;
                let item_profit = item_ratio * self.pb.profit[item] as f64;
                max_profit += item_profit.floor() as isize;
                capacity = 0;
            }

            depth += 1;
        }

        max_profit
    }
}

/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
pub struct KPRanking;
impl StateRanking for KPRanking {
    type State = KnapsackState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.capacity.cmp(&b.capacity)
    }
}

/// Optionally, define dominance relations between states obtained throughout the search.
/// In this case, s1 dominates s2 if s1.capacity >= s2.capacity and s1 has a larger value than s2.
pub struct KPDominance;
impl Dominance for KPDominance {
    type State = KnapsackState;
    type Key = usize;

    fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
        Some(state.depth)
    }

    fn nb_dimensions(&self, _state: &Self::State) -> usize {
        1
    }

    fn get_coordinate(&self, state: &Self::State, _: usize) -> isize {
        state.capacity as isize
    }

    fn use_value(&self) -> bool {
        true
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
/// or parse int errors (which are actually a variant of the format errror since it tells 
/// you that the parser expected an integer number but got ... something else).
#[derive(Debug, thiserror::Error)]
pub enum Error {
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

/// This funciton is used to read a knapsack instance from file. It returns either a
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
    Ok(Knapsack::new(capa, profit, weight))
}

/// An utility function to return an max width heuristic that can either be a fixed width
/// policy (if w is fixed) or an adaptative policy returning the number of unassigned variables
/// in the overall problem.
fn max_width<T>(nb_vars: usize, w: Option<usize>) -> Box<dyn WidthHeuristic<T> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
    } else {
        Box::new(NbUnassignedWitdh(nb_vars))
    }
}

/// This is your executable's entry point. It is the place where all the pieces are put together
/// to create a fast an effectuve solver for the knapsack problem.
fn main() {
    let args = Args::parse();
    let problem = read_instance(&args.fname).unwrap();
    let relaxation= KPRelax{pb: &problem};
    let heuristic= KPRanking;
    let width = max_width(problem.nb_variables(), args.width);
    let dominance = SimpleDominanceChecker::new(KPDominance);
    let cutoff = TimeBudget::new(Duration::from_secs(15));//NoCutoff;
    let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));

    let mut solver = DefaultBarrierSolver::new(
        &problem, 
        &relaxation, 
        &heuristic, 
        width.as_ref(), 
        &dominance,
        &cutoff, 
        &mut fringe,
    );

    let start = Instant::now();
    let Completion{ is_exact, best_value } = solver.maximize();
    
    let duration = start.elapsed();
    let upper_bound = solver.best_upper_bound();
    let lower_bound = solver.best_lower_bound();
    let gap = solver.gap();
    let best_solution  = solver.best_solution().map(|mut decisions|{
        decisions.sort_unstable_by_key(|d| d.variable.id());
        decisions.iter().map(|d| d.value).collect::<Vec<_>>()
    });

    println!("Duration:   {:.3} seconds", duration.as_secs_f32());
    println!("Objective:  {}",            best_value.unwrap_or(-1));
    println!("Upper Bnd:  {}",            upper_bound);
    println!("Lower Bnd:  {}",            lower_bound);
    println!("Gap:        {:.3}",         gap);
    println!("Aborted:    {}",            !is_exact);
    println!("Solution:   {:?}",          best_solution.unwrap_or_default());
}
