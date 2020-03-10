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

use ddo::core::abstraction::dp::{Problem, Relaxation};
use ddo::core::abstraction::solver::Solver;
use ddo::core::common::{Decision, Domain, Node, Variable, VarSet};
use ddo::core::implementation::mdd::builder::mdd_builder_ref;
use ddo::core::implementation::solver::parallel::ParallelSolver;

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
    fn initial_value(&self) -> i32 {
        0
    }
    fn transition(&self, state: &usize, _vars: &VarSet, dec: Decision) -> usize {
        state - (self.weight[dec.variable.id()] * dec.value as usize)
    }
    fn transition_cost(&self, _state: &usize, _vars: &VarSet, dec: Decision) -> i32 {
        self.profit[dec.variable.id()] as i32 * dec.value
    }
}

/// Merge the nodes by creating a new fake node that has the maximum remaining
/// capacity from the merged nodes and the maximum objective value (intuitiveley,
/// it keeps the best state, and the best value for the objective function).
#[derive(Debug, Clone)]
struct KPRelax;
impl Relaxation<usize> for KPRelax {
    fn merge_nodes(&self, nodes: &[Node<usize>]) -> Node<usize> {
        let lp_info = nodes.iter()
            .map(|n| &n.info)
            .max_by_key(|i| i.lp_len).unwrap();
        let max_capa= nodes.iter()
            .map(|n| n.state)
            .max().unwrap();
        Node::merged(max_capa, lp_info.lp_len, lp_info.lp_arc.clone())
    }
}

fn main() {
    let problem = Knapsack {
        capacity: 50,
        profit  : vec![60, 100, 120],
        weight  : vec![10,  20,  30]
    };
    let mdd = mdd_builder_ref(&problem, KPRelax).build();
    let mut solver = ParallelSolver::new(mdd);
    let (optimal, solution) = solver.maximize();

    assert_eq!(220, optimal);
    println!("Solution");
    for decision in solution.as_ref().unwrap() {
        if decision.value == 1 {
            println!("{}", decision.variable.id());
        }
    }
}