use crate::examples::max2sat::model::{Max2Sat, State};
use crate::core::abstraction::heuristics::VariableHeuristic;
use crate::core::common::{Variable, VarSet};
use crate::core::abstraction::mdd::MDD;

pub struct Max2SatOrder<'a> {
    problem: &'a Max2Sat
}

impl <'a> Max2SatOrder<'a> {
    pub fn new(problem: &'a Max2Sat) -> Max2SatOrder<'a> {
        Max2SatOrder{problem}
    }
}

impl VariableHeuristic<State> for Max2SatOrder<'_> {
    fn next_var(&self, _dd: &dyn MDD<State>, vars: &VarSet) -> Option<Variable> {
        let mut var = None;
        let mut wt  = i32::min_value();

        for v in vars.iter() {
            let v_wt = self.problem.sum_of_clause_weights[v.0];
            if v_wt > wt {
                var = Some(v);
                wt  = v_wt;
            }
        }

        var
    }
}