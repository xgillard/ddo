use crate::core::abstraction::heuristics::WidthHeuristic;
use crate::core::abstraction::mdd::{MDD, Node};
use std::hash::Hash;

pub struct FixedWidth(pub usize);

impl <T, N> WidthHeuristic<T, N> for FixedWidth
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    fn max_width(&self, _dd: &dyn MDD<T, N>) -> usize {
        self.0
    }
}