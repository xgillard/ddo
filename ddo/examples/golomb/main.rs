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

//! This example show how to implement a solver for the Golomb ruler problem using ddo.
//! For more information on this problem https://en.wikipedia.org/wiki/Golomb_ruler
//! This model was introduced by Willem-Jan van Hoeve
//! In this model, each variable/layer is the position of the next mark to be placed.
//! The domain of each variable is the set of all possible positions for the next mark.
//! One can only add a mark if the distance between the new mark and all previous marks
//! is not already present in the set of distances between marks.
//! The cost of a transition is the distance between the new mark and the previous last one.
//! The cost of a solution is thus the position of the last mark.
//!
use std::{time::{Duration, Instant}};
use bit_set::BitSet;

use clap::Parser;
use ddo::*;

#[cfg(test)]
mod tests;


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GolombState {
    marks: BitSet, // the set of marks
    distances: BitSet, // the set of pairwise distances
    number_of_marks: isize, // the number of marks
    last_mark: isize, // location of last mark
}

/// Instance of the Golomb problem.
pub struct Golomb {
    n: isize,
}

impl Golomb {
    pub fn new(n: isize) -> Self {
        Golomb {
            n: n
        }
    }
}

/// This is how you implement the labeled transition system (LTS) semantics of
/// a simple dynamic program solving the Golomb problem. The definition of
/// each of the methods should be pretty clear and easy to grasp. Should you
/// want more details on the role of each of these methods, then you are
/// encouraged to go checking the documentation of the `Problem` trait.
impl Problem for Golomb {

    type State = GolombState;

    fn nb_variables(&self) -> usize {
        self.n as usize
    }

    // create the edges (decisions) from the given state
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback) {
        for i in (state.last_mark) + 1..(self.n * self.n) {
            if state.marks.iter().any(|j| state.distances.contains(i as usize - j)) {
                continue; // this distance is already present, invalid mark at it (alldifferent)
            } else {
                f.apply(Decision { variable, value: i });
            }
        }
    }

    // create the initial state
    fn initial_state(&self) -> Self::State {
        // upper-bound on the number of marks and distances
        let n2 = (self.n * self.n) as usize;
        GolombState {
            marks: BitSet::with_capacity(n2),
            distances: BitSet::with_capacity(n2),
            number_of_marks: 0,
            last_mark: -1,
        }
    }

    fn initial_value(&self) -> isize {
        1
    }

    // compute the next state from the current state and the decision
    fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
        let mut ret = state.clone();
        let l = dec.value; // the new mark
        ret.marks.insert(l as usize); // add the new mark
        // add distances between new mark and previous marks
        for i in state.marks.iter() {
            ret.distances.insert(l as usize - i);
        }
        ret.number_of_marks += 1; // increment the number of marks
        ret.last_mark = l; // update the last mark
        ret
    }

    // compute the cost of the decision from the given state
    fn transition_cost(&self, state: &Self::State, dec: Decision) -> isize {
        // distance between the new mark and the previous one
        return -(dec.value - state.last_mark as isize) as isize; // put a minus to turn objective into maximization (ddo requirement)
    }

    // next variable to branch on
    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        let n = self.nb_variables();
        if depth < n {
            Some(Variable(depth))
        } else {
            None
        }
    }
}


pub struct GolombRelax<'a>{pub pb: &'a Golomb}
impl Relaxation for GolombRelax<'_> {
    type State = GolombState;

    // take the intersection of the marks and distances sets
    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {

        let states_vec: Vec<&Self::State> = states.collect();

        let mut marks = states_vec.iter().map(|node| node.marks.clone());
        let mut intersection_marks = marks.next().unwrap();
        marks.for_each(|bitset| intersection_marks.intersect_with(&bitset));

        let mut distances = states_vec.iter().map(|node| node.distances.clone());
        let mut insersection_distances = distances.next().unwrap();
        distances.for_each(|bitset| insersection_distances.intersect_with(&bitset));

        let number_of_marks = states_vec.iter().min_by_key(|node| node.number_of_marks).unwrap().number_of_marks;
        let last_mark = states_vec.iter().min_by_key(|node| node.last_mark).unwrap().last_mark;

        GolombState {
            marks: intersection_marks,
            distances: insersection_distances,
            number_of_marks,
            last_mark,
        }
    }

    fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        // known solution for n = 0..13
        let arr: [isize; 14] = [0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106];
        // there is n-k marks left to place, therefore we need at least golomb(n-k) to place them
        return -arr[self.pb.n as usize - state.number_of_marks as usize];
    }

}

/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
pub struct Golombranking;
impl StateRanking for Golombranking {
    type State = GolombState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.last_mark.cmp( &b.last_mark) // sort by last mark
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


/// An utility function to return an max width heuristic that can either be a fixed width
/// policy (if w is fixed) or an adaptive policy returning the number of unassigned variables
/// in the overall problem.
fn max_width<T>(nb_vars: usize, w: Option<usize>) -> Box<dyn WidthHeuristic<T> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
    } else {
        Box::new(NbUnassignedWitdh(nb_vars))
    }
}

/// This is your executable's entry point. It is the place where all the pieces are put together
/// to create a fast an effective solver for the Golomb problem.
fn main() {

    let problem = Golomb::new(5);
    let relaxation = GolombRelax{pb: &problem};
    let heuristic = Golombranking;
    let width = max_width(problem.nb_variables(), Some(100));
    let cutoff = TimeBudget::new(Duration::from_secs(100));//NoCutoff;
    let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));

    let mut solver = DefaultBarrierSolver::new(
        &problem,
        &relaxation,
        &heuristic,
        width.as_ref(),
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
