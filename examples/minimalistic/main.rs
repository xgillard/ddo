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

use ddo::*;

/// Describe the binary knapsack problem in terms of a dynamic program.
/// Here, the state of a node, is nothing more than an unsigned integer (usize).
/// That unsigned integer represents the remaining capacity of our sack.
#[derive(Debug, Clone)]
struct Knapsack {
    capacity: usize,
    profit  : Vec<usize>,
    weight  : Vec<usize>
}
impl Problem<usize> for Knapsack {
    fn nb_vars(&self) -> usize {
        self.profit.len()
    }
    fn domain_of<'a>(&self, state: &'a usize, var: Variable) ->Domain<'a> {
        if *state >= self.weight[var.id()] {
            vec![0, 1].into()
        } else {
            vec![0].into()
        }
    }
    fn initial_state(&self) -> usize {
        self.capacity
    }
    fn initial_value(&self) -> isize {
        0
    }
    fn transition(&self, state: &usize, _vars: &VarSet, dec: Decision) -> usize {
        state - (self.weight[dec.variable.id()] * dec.value as usize)
    }
    fn transition_cost(&self, _state: &usize, _vars: &VarSet, dec: Decision) -> isize {
        self.profit[dec.variable.id()] as isize * dec.value
    }
}

/// Merge the nodes by creating a new fake node that has the maximum remaining
/// capacity from the merged nodes.
#[derive(Debug, Clone)]
struct KPRelax;
impl Relaxation<usize> for KPRelax {
    /// To merge a given selection of states (capacities) we will keep the
    /// maximum capacity. This is an obvious relaxation as it allows us to
    /// put more items in the sack.
    fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
        // the selection is guaranteed to have at least one state so using
        // unwrap after max to get rid of the wrapping 'Option' is perfectly safe.
        *states.max().unwrap()
    }
    /// When relaxing (merging) the states, we did not run into the risk of
    /// possibly decreasing the maximum objective value reachable from the
    /// components of the merged node. Hence, we dont need to do anything
    /// when relaxing the edge. Still, if we wanted to, we could chose to
    /// return an higher value.
    fn relax_edge(&self, _src: &usize, _dst: &usize, _relaxed: &usize, _d: Decision, cost: isize) -> isize {
        cost
    }
    // [Optional]
    // We could also implement a rough upper bound estimator for our model.
    // but we will not do it in this minimal example. (A default
    // implementation is provided). If you were to override the default
    // implementation -- and if possible, you should; you would need to
    // implement the `estimate()` method of the `Relaxation` trait.
}

/// This is your main method: it is far from being useful in this case,
/// but it helps for you to get the idea.
fn main() {
    // 1. Create an instance of our knapsack problem
    let problem = Knapsack {
        capacity: 50,
        profit  : vec![60, 100, 120],
        weight  : vec![10,  20,  30]
    };
    // 2. Build an MDD for the given problem and relaxation
    let mdd = mdd_builder(&problem, KPRelax).into_deep();
    // 3. Create a parllel solver on the basis of this MDD (this is how
    //    you can specify the MDD implementation you wish to use to develop
    //    the relaxed and restricted MDDs).
    let mut solver = ParallelSolver::new(mdd);
    // 4. Maximize your objective function
    let outcome = solver.maximize();
    // The completion outcome tells you the optimum value + a boolean that
    // indicate if you prooved the optimum solution. This flag is only
    // going to be false if you use a cutoff to kill the search at some point.
    let optimal = outcome.best_value.unwrap_or(isize::min_value());
    let solution= solver.best_solution();

    // 5. Do whatever you like with the optimal solution.
    assert_eq!(220, optimal);
    println!("Solution");
    for decision in solution.unwrap().iter() {
        if decision.value == 1 {
            println!("{}", decision.variable.id());
        }
    }
}
