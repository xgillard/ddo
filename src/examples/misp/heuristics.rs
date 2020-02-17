use std::cmp::Ordering;

use bitset_fixed::BitSet;

use crate::core::common::{Node, VarSet, Variable};
use crate::core::utils::{LexBitSet, BitSetIter};
use crate::core::abstraction::mdd::Layer;
use crate::core::abstraction::heuristics::VariableHeuristic;
use crate::examples::misp::model::Misp;
use crate::core::abstraction::dp::Problem;

#[derive(Debug, Clone)]
pub struct MispVarHeu(usize);
impl MispVarHeu {
    pub fn new(pb: &Misp) -> Self {
        MispVarHeu(pb.nb_vars())
    }
}
impl VariableHeuristic<BitSet> for MispVarHeu {
    fn next_var<'a>(&self, _: &'a VarSet, _: Layer<'a, BitSet>, next: Layer<'a, BitSet>) -> Option<Variable> {
        let mut counters = vec![0; self.0];

        for (s, _) in next {
            for v in BitSetIter::new(s) {
                counters[v] += 1;
            }
        }

        counters.iter().enumerate()
            .filter(|(_, &count)| count > 0)
            .min_by_key(|(_, &count)| count)
            .map(|(i, _)| Variable(i))
    }
}

pub fn vars_from_misp_state(n: &Node<BitSet>) -> VarSet {
    VarSet(n.state.clone())
}

pub fn misp_min_lp(a: &Node<BitSet>, b: &Node<BitSet>) -> Ordering {
    a.info.lp_len.cmp(&b.info.lp_len)
        .then_with(|| LexBitSet(&a.state).cmp(&LexBitSet(&b.state)))
}