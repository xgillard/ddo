use crate::core::abstraction::mdd::{MDD, Node};
use bitset_fixed::BitSet;
use std::hash::Hash;

pub trait WithHeuristics<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    fn max_width(&self, dd: &dyn MDD<T, N>) -> usize;
}

pub trait VariableHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    fn next_var(&self, dd: &dyn MDD<T, N>, vars: &BitSet) -> u32;
}

pub trait NodeHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    fn select_nodes(&self, dd: &dyn MDD<T, N>) -> &[u32];
}