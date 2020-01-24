use crate::examples::misp::misp::Misp;
use std::rc::Rc;
use crate::core::abstraction::dp::{Relaxation, Decision, Problem, VarSet};
use bitset_fixed::BitSet;

pub struct MispRelax {
    pb: Rc<Misp>
}

impl MispRelax {
    pub fn new(pb : Rc<Misp>) -> MispRelax {
        MispRelax{pb}
    }
}

impl Relaxation<BitSet> for MispRelax {
    fn merge_states(&self, states: &[&BitSet]) -> BitSet {
        let mut bs = BitSet::new(self.pb.nb_vars());
        for s in states {
            bs |= s;
        }
        bs
    }

    fn relax_cost(&self, _from: &BitSet, _to: &BitSet, decision: &Decision) -> i32 {
        if decision.value == 0 { 0 } else { self.pb.graph.weights[decision.variable.0] }
    }

    fn rough_ub(&self, lp: i32, s: &BitSet, vars: &VarSet) -> i32 {
        let mut res = lp;
        for x in vars.iter() {
            if s[x.0] {
                res += self.pb.graph.weights[x.0];
            }
        }
        res
    }
}