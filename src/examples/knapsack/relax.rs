use crate::examples::knapsack::model::{Knapsack, KnapsackState};
use crate::core::abstraction::dp::{Relaxation, Problem};
use crate::core::common::{Node, NodeInfo, VarSet};
use bitset_fixed::BitSet;

#[derive(Debug)]
pub struct KnapsackRelax<'a> {
    pb: &'a Knapsack
}
impl <'a> KnapsackRelax<'a> {
    pub fn new(pb: &'a Knapsack) -> Self { KnapsackRelax {pb} }
}
impl Relaxation<KnapsackState> for KnapsackRelax<'_> {
    fn merge_nodes(&self, nodes: &[Node<KnapsackState>]) -> Node<KnapsackState> {
        let mut lp_info   = &nodes[0].info;
        let mut capacity  = 0;
        let mut free_vars = BitSet::new(self.pb.nb_vars());
        for n in nodes.iter() {
            free_vars |= &n.state.free_vars.0;
            capacity   = capacity.max(n.state.capacity);

            if n.info.lp_len > lp_info.lp_len {
                lp_info = &n.info;
            }
        }

        let state = KnapsackState {capacity, free_vars: VarSet(free_vars)};
        Node { state: state, info : lp_info.clone() }
    }
    fn estimate_ub(&self, state: &KnapsackState, info: &NodeInfo<KnapsackState>) -> i32 {
        info.lp_len + state.free_vars.iter().map(|v| {
            let item      = &self.pb.data[v.id()];
            let max_amout = state.capacity / item.weight;
            max_amout * item.profit
        }).sum::<usize>() as i32
    }
}