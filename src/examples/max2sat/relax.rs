use std::cmp::min;

use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::mdd::{Node, NodeInfo};
use crate::examples::max2sat::model::{Max2Sat, State};
use crate::core::common::VarSet;

#[derive(Debug)]
pub struct Max2SatRelax<'a> {
    problem : &'a Max2Sat,
    vars    : VarSet
}

impl <'a> Max2SatRelax<'a> {
    pub fn new(problem: &'a Max2Sat) -> Max2SatRelax<'a> {
        Max2SatRelax { problem, vars: problem.all_vars() }
    }
}

impl Relaxation<State> for Max2SatRelax<'_> {
    //fn estimate_ub(&self, node: &Node<State>) -> i32 {
    //    let sum = node.state.variables.iter()
    //        .map(|v| self.problem.sum_of_clause_weights[v.0])
    //        .sum::<i32>();
//
    //    node.lp_len + sum
    //}
    fn merge_nodes(&self, nodes: &[Node<State>]) -> Node<State> {
        let mut benefits      = vec![0; self.problem.nb_vars()];
        let mut relaxed_costs = nodes.iter()
                                     .map(|n| n.info.lp_len)
                                     .collect::<Vec<i32>>();

        // Compute the merged state and relax the best edges costs
        for v in self.vars.iter() {
            let mut sign      = 0;
            let mut min_benef = i32::max_value();
            let mut same      = true;

            for node in nodes.iter() {
                let substate = node.state[v];
                min_benef = min(min_benef, substate.abs());

                if sign == 0 && substate != 0 {
                    sign = substate.abs() / substate;
                } else if sign * substate < 0 {
                    same = false;
                    break;
                }
            }

            if same {
                benefits[v.0] = sign * min_benef;
            }

            for j in 0..nodes.len() {
                relaxed_costs[j] += nodes[j].state[v].abs() - benefits[v.0].abs();
            }
        }

        // Find the best info
        let mut best    = 0;
        let mut longest = i32::min_value();

        for (j, relaxed) in relaxed_costs.iter().cloned().enumerate() {
            if relaxed > longest {
                best    = j;
                longest = relaxed;
            }
        }

        let merged_state  = State {substates: benefits};
        let merged_infos  = NodeInfo {
            is_exact: false,
            lp_len  : longest,
            lp_arc  : nodes[best].info.lp_arc.clone(),
            ub      : nodes[best].info.ub
        };

        Node{state: merged_state, info: merged_infos}
    }
}