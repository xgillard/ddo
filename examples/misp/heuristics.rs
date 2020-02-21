use bitset_fixed::BitSet;

use ddo::core::abstraction::dp::Problem;
use ddo::core::abstraction::heuristics::VariableHeuristic;
use ddo::core::common::{Layer, Node, Variable, VarSet};
use ddo::core::utils::BitSetIter;

use crate::model::Misp;

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