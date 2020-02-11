use crate::core::common::{Node, Variable, VarSet};
use crate::examples::max2sat::model::{Max2Sat, State};
use crate::core::abstraction::heuristics::VariableHeuristic;

use std::cmp::Ordering;
use compare::Compare;

pub struct Max2SatOrder<'a> {
    problem: &'a Max2Sat
}

impl <'a> Max2SatOrder<'a> {
    pub fn new(problem: &'a Max2Sat) -> Max2SatOrder<'a> {
        Max2SatOrder{problem}
    }
}

impl VariableHeuristic<State> for Max2SatOrder<'_> {
    fn next_var(&self, free_vars: &VarSet) -> Option<Variable> {
        let mut var = None;
        let mut wt  = i32::min_value();

        for v in free_vars.iter() {
            let v_wt = self.problem.sum_of_clause_weights[v.0];
            if v_wt > wt {
                var = Some(v);
                wt  = v_wt;
            }
        }

        var
    }
}

pub struct MinRank;
impl Compare<Node<State>> for MinRank {
    fn compare(&self, x: &Node<State>, y: &Node<State>) -> Ordering {
        let xrank = x.info.lp_len + x.state.rank();
        let yrank = y.info.lp_len + y.state.rank();
        xrank.cmp(&yrank)
        //    .then_with(||x.state.rank().cmp(&y.state.rank()))
        //    .then_with(||{
        //    let xdata = &x.state.0;
        //    let ydata = &y.state.0;
        //    for i in 0..xdata.len() {
        //        if xdata[i] != ydata[i] {
        //            return xdata[i].cmp(&ydata[i]);
        //        }
        //    }
        //    Equal
        //})
    }
}

#[cfg(test)]
mod test {
    use crate::examples::max2sat::heuristics::Max2SatOrder;
    use crate::examples::max2sat::model::Max2Sat;
    use std::path::PathBuf;
    use std::fs::File;
    use crate::core::abstraction::dp::Problem;
    use crate::core::abstraction::heuristics::VariableHeuristic;

    #[test]
    fn variable_ordering() {
        let problem = instance("frb10-6-1.wcnf");
        let order   = Max2SatOrder::new(&problem);
        let mut vars= problem.all_vars();

        let mut actual= vec![];
        for _ in 0..problem.nb_vars() {
            let v = order.next_var(&vars).unwrap();
            vars.remove(v);
            actual.push(v.0);
        }

        let expected = vec![
            26, 24, 28, 25, 27, 32, 43, 44, 45, 42, 47, 52, 19, 34, 11, 22, 46,
            49, 50,  4,  8, 16, 53,  5,  9, 18, 23, 48,  0, 20, 59,  1, 35, 17,
            31, 39, 54, 57,  2,  3, 14, 15, 30, 38, 55,  6,  7, 10, 12, 29, 33,
            37, 51, 56, 58, 13, 21, 36, 40, 41
        ];
        assert_eq!(actual, expected);
    }


    fn instance(id: &str) -> Max2Sat {
        let location = PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("tests/resources/max2sat/")
            .join(id);

        File::open(location).expect("File not found").into()
    }
}