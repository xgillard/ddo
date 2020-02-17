use crate::core::common::{Node, Variable, Decision, NodeInfo, Domain};
use std::cmp::Ordering;
use std::sync::Arc;
use crate::core::abstraction::mdd::Layer;

pub trait Config<T> {
    fn root_node(&self) -> Node<T>;
    fn impacted_by(&self, state: &T, v: Variable) -> bool;
    fn load_vars (&mut self, root: &Node<T>);
    fn nb_free_vars(&self) -> usize;
    fn select_var(&self, current: Layer<'_, T>, next: Layer<'_, T>) -> Option<Variable>;
    fn remove_var(&mut self, v: Variable);
    fn domain_of<'a>(&self, state: &'a T, v: Variable) -> Domain<'a>;
    fn max_width(&self) -> usize;
    fn branch(&self, state: &T, info: Arc<NodeInfo>, d: Decision) -> Node<T>;
    fn estimate_ub(&self, state: &T, info: &NodeInfo) -> i32;
    fn compare(&self, x: &Node<T>, y: &Node<T>) -> Ordering;
    fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T>;
}