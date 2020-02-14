use bitset_fixed::BitSet;

use crate::core::common::{Node, NodeInfo};
use crate::core::abstraction::dp::{Problem, Relaxation};
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
    fn merge_nodes(&self, nodes: &[Node<BitSet>]) -> Node<BitSet> {
        let mut state = BitSet::new(self.pb.nb_vars());
        let mut best  = &nodes[0].info;

        for node in nodes.iter() {
            state |= &node.state;

            if node.info.lp_len > best.lp_len {
                best = &node.info;
            }
        }

        let mut info = best.clone();
        info.is_exact = false;
        Node {state, info}
    }
    fn estimate_ub(&self, state: &BitSet, info: &NodeInfo) -> i32 {
        info.lp_len + state.count_ones() as i32
    }
}