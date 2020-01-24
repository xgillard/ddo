use crate::examples::misp::misp::Misp;
use std::rc::Rc;
use crate::core::abstraction::dp::{Relaxation, Decision, Problem};
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

    fn rough_ub(&self, lp: i32, s: &BitSet, _vars: &BitSet) -> i32 {
        let mut res = lp;
        let mut i = 0;
        let cnt = s.count_ones();
        for x in 0..s.size() {
            if s[x] {
                res += self.pb.graph.weights[x];
                i+= 1;
                if i >= cnt {
                    break;
                }
            }
        }
        res
    }
}