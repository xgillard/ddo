use crate::examples::mcp::graph::Graph;
use crate::core::abstraction::dp::Problem;
use crate::core::common::{Variable, VarSet, Domain, Decision};
use std::cmp::{max, min};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct McpState {
    pub benef  : Vec<i32>,
    pub initial: bool
}

// Define a few constants to improve readability
const S      : i32 = 1;
const T      : i32 =-1;
const ONLY_S : [i32; 1] = [S];
const BOTH_ST: [i32; 2] = [S, T];

pub struct Mcp {
    pub graph : Graph
}
impl Mcp {
    pub fn new(g: Graph) -> Self { Mcp {graph: g} }
}
impl Problem<McpState> for Mcp {
    fn nb_vars(&self) -> usize {
        self.graph.nb_vertices
    }

    fn initial_state(&self) -> McpState {
        McpState {initial: true, benef: vec![0; self.nb_vars()]}
    }

    fn initial_value(&self) -> i32 {
        self.graph.sum_of_negative_edges()
    }

    fn domain_of<'a>(&self, state: &'a McpState, _var: Variable) -> Domain<'a> {
        if state.initial { Domain::Slice(&ONLY_S) } else { Domain::Slice(&BOTH_ST) }
    }

    fn transition(&self, state: &McpState, vars: &VarSet, d: Decision) -> McpState {
        let mut benefits = vec![0; self.nb_vars()];
        for v in vars.iter() { // for all unassigned vars
            benefits[v.id()] = state.benef[v.id()] + d.value * self.graph[(d.variable, v)];
        }
        McpState {initial: false, benef: benefits}
    }

    fn transition_cost(&self, state: &McpState, vars: &VarSet, d: Decision) -> i32 {
        match d.value {
            S => if state.initial { 0 } else { self.branch_on_s(state, vars, d) },
            T => if state.initial { 0 } else { self.branch_on_t(state, vars, d) },
            _ => unreachable!()
        }
    }
}
// private methods
impl Mcp {
    fn branch_on_s(&self, state: &McpState, vars: &VarSet, d: Decision) -> i32 {
        // The \( (- s^k_k)^+ \) component
        let res = max(0, -state.benef[d.variable.id()]);
        // The \( \sum_{l > k, s^k_l w_{kl} \le 0} \min\left\{ |s^k_l|, |w_{kl}| \right\} \)
        let mut sum = 0;
        for v in vars.iter() {
            let skl = state.benef[v.id()];
            let wkl = self.graph[(d.variable, v)];

            if skl * wkl <= 0 { sum += min(skl.abs(), wkl.abs()); }
        }
        res + sum
    }
    fn branch_on_t(&self, state: &McpState, vars: &VarSet, d: Decision) -> i32 {
        // The \( (s^k_k)^+ \) component
        let res = max(0, state.benef[d.variable.id()]);
        // The \( \sum_{l > k, s^k_l w_{kl} \le 0} \min\left\{ |s^k_l|, |w_{kl}| \right\} \)
        let mut sum = 0;
        for v in vars.iter() {
            let skl = state.benef[v.id()];
            let wkl = self.graph[(d.variable, v)];

            if skl * wkl >= 0 { sum += min(skl.abs(), wkl.abs()); }
        }
        res + sum
    }
}