use crate::examples::misp::model::Misp;
use std::rc::Rc;
use crate::core::abstraction::dp::{Relaxation, Decision, Problem};
use bitset_fixed::BitSet;
use crate::core::abstraction::mdd::MDD;

pub struct MispRelax {
    pb: Rc<Misp>
}

impl MispRelax {
    pub fn new(pb : Rc<Misp>) -> MispRelax {
        MispRelax{pb}
    }
}

impl Relaxation<BitSet> for MispRelax {
    fn merge_states(&self, _dd: &dyn MDD<BitSet>, states: &[&BitSet]) -> BitSet {
        let mut bs = BitSet::new(self.pb.nb_vars());
        for s in states {
            bs |= s;
        }
        bs
    }

    fn relax_cost(&self, _dd: &dyn MDD<BitSet>, original_cost: i32, _from: &BitSet, _to: &BitSet, _d: &Decision) -> i32 {
        original_cost
    }

    fn rough_ub(&self, lp: i32, s: &BitSet) -> i32 {
        lp + s.count_ones() as i32
        /*
        let mut res = lp;
        for x in BitSetIter::new(s.clone()) {
            if s[x.0] {
                res += self.pb.graph.weights[x.0];
            }
        }
        res
        */
    }
}