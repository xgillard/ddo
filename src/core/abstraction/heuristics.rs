use std::collections::HashMap;
use crate::core::abstraction::mdd::{Node, MDD};

pub trait WithHeuristics<T> {
    fn max_width(&self, dd: &dyn MDD<T>) -> usize;
}

pub trait VariableHeuristic<T> {
    fn next_var(&self, dd: &dyn MDD<T>, layer: &HashMap<T, dyn Node<T>>) -> u32;
}

pub trait NodeHeuristic<T> {
    fn select_nodes(&self, dd: &dyn MDD<T>, layer: &HashMap<T, dyn Node<T>>) -> &[u32];
}