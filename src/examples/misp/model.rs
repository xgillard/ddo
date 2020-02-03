use crate::examples::misp::instance::Graph;
use crate::core::abstraction::dp::{Variable, Problem, Decision, VarSet};
use bitset_fixed::BitSet;
use std::ops::Not;

pub struct Misp {
    pub graph : Graph,
    yes_no    : Vec<i32>,
    no        : Vec<i32>
}

impl Misp {
    pub fn from_file(fname : &str) -> Misp {
        let mut g = Graph::from_file(fname);
        g.complement();
        Misp {graph: g, yes_no: vec![1, 0], no: vec![0]}
    }
}

impl Problem<BitSet> for Misp {
    fn nb_vars(&self) -> usize {
        self.graph.nb_vars
    }

    fn initial_state(&self) -> BitSet {
        BitSet::new(self.graph.nb_vars).not()
    }

    fn initial_value(&self) -> i32 {
        0
    }

    fn domain_of(&self, state: &BitSet, var: Variable) -> &[i32] {
        if state[var.0] { &self.yes_no } else { &self.no }
    }

    fn transition(&self, state: &BitSet, _vars: &VarSet, d: &Decision) -> BitSet {
        let mut bs = state.clone();
        bs.set(d.variable.0, false);

        // drop adjacent vertices if needed
        if d.value == 1 {
            bs &= &self.graph.adj_matrix[d.variable.0];
        }

        bs
    }

    fn transition_cost(&self, _state: &BitSet, _vars: &VarSet, d: &Decision) -> i32 {
        if d.value == 0 {
            0
        } else {
            self.graph.weights[d.variable.0]
        }
    }

    fn impacted_by(&self, state: &BitSet, variable: Variable) -> bool {
        state[variable.0]
    }
}