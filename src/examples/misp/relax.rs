use bitset_fixed::BitSet;

use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::mdd::{MDD, Node};
use crate::examples::misp::model::Misp;

pub struct MispRelax<'a> {
    pb: &'a Misp
}

impl <'a> MispRelax<'a> {
    pub fn new(pb : &'a Misp) -> MispRelax<'a> {
        MispRelax{pb}
    }
}
impl <'a> Relaxation<BitSet> for MispRelax<'a> {
    fn merge_nodes(&self, _dd: &dyn MDD<BitSet>, nodes: &[&Node<BitSet>]) -> Node<BitSet> {
        let mut bs   = BitSet::new(self.pb.nb_vars());
        let mut max_l= i32::min_value();
        let mut arc  = None;

        for n in nodes {
            bs |= &n.state;

            if n.lp_len > max_l {
                max_l = n.lp_len;
                arc   = n.lp_arc.clone();
            }
        }

        Node::new(bs, max_l, arc, false)
    }
    fn estimate_ub(&self, n: &Node<BitSet>) -> i32 {
        n.lp_len + n.state.count_ones() as i32
    }
}
/*
impl <'a> SimpleRelaxation<BitSet> for MispRelax<'a> {
    fn merge_states(&self, _dd: &dyn MDD<BitSet>, states: &[&BitSet]) -> BitSet {
        let mut bs = BitSet::new(self.pb.nb_vars());
        for s in states {
            bs |= s;
        }
        bs
    }

    fn relax_cost(&self, _dd: &dyn MDD<BitSet>, original_cost: i32, _from: &BitSet, _to: &BitSet, _d: Decision) -> i32 {
        original_cost
    }

    fn rough_ub(&self, lp: i32, s: &BitSet) -> i32 {
        lp + s.count_ones() as i32

        //let mut res = lp;
        //for x in BitSetIter::new(s.clone()) {
        //    if s[x.0] {
        //        res += self.pb.graph.weights[x.0];
        //    }
        //}
        //res

    }
}*/