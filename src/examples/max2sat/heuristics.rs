use crate::examples::max2sat::model::{Max2Sat, State};
use crate::core::abstraction::heuristics::VariableHeuristic;
use crate::core::common::{Variable, VarSet};
use crate::core::abstraction::mdd::MDD;
use crate::core::abstraction::dp::Problem;

pub struct Max2SatOrder {
    sum_of_clause_weights: Vec<i32>
}

const fn idx(x: i32) -> usize {
    (x.abs() - 1) as usize
}

impl Max2SatOrder {
    pub fn new(problem: &Max2Sat) -> Max2SatOrder {
        let mut ret = Max2SatOrder{sum_of_clause_weights: vec![0; problem.nb_vars()]};
        for (clause, weight) in problem.inst.weights.iter() {
            ret.sum_of_clause_weights[idx(clause.a)] += *weight;
            ret.sum_of_clause_weights[idx(clause.b)] += *weight;
        }
        ret
    }
}

impl VariableHeuristic<State> for Max2SatOrder {
    fn next_var(&self, _dd: &dyn MDD<State>, vars: &VarSet) -> Option<Variable> {
        let mut var = None;
        let mut wt  = i32::min_value();

        for v in vars.iter() {
            let v_wt = self.sum_of_clause_weights[v.0];
            if v_wt > wt {
                var = Some(v);
                wt  = v_wt;
            }
        }

        var
    }
}