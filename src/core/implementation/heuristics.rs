use std::cmp::Ordering;

use compare::Compare;

use crate::core::abstraction::dp::Problem;
use crate::core::abstraction::heuristics::{LoadVars, VariableHeuristic, WidthHeuristic};
use crate::core::common::{Node, Variable, VarSet};
use crate::core::abstraction::mdd::Layer;

//~~~~~ Max Width Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Debug, Clone)]
pub struct FixedWidth(pub usize);
impl WidthHeuristic for FixedWidth {
    fn max_width(&self, _free: &VarSet) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct NbUnassigned;
impl WidthHeuristic for NbUnassigned {
    fn max_width(&self, free: &VarSet) -> usize {
        free.len()
    }
}

//~~~~~ Variable Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Default, Debug, Clone)]
pub struct NaturalOrder;
impl <T> VariableHeuristic<T> for NaturalOrder {
    fn next_var<'a>(&self, free_vars: &'a VarSet, _c: Layer<'a, T>, _n: Layer<'a, T>) -> Option<Variable> {
        free_vars.iter().next()
    }
}

//~~~~~ Node Ordering Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Debug, Default, Clone)]
pub struct MinLP;
impl <T> Compare<Node<T>> for MinLP {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.lp_len.cmp(&b.info.lp_len)
    }
}
#[derive(Debug, Default, Clone)]
pub struct MaxLP;
impl <T> Compare<Node<T>> for MaxLP {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.lp_len.cmp(&b.info.lp_len).reverse()
    }
}

#[derive(Debug, Default, Clone)]
pub struct MaxUB;
impl <T> Compare<Node<T>> for MaxUB {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.ub.cmp(&b.info.ub).then_with(|| a.info.lp_len.cmp(&b.info.lp_len))
    }
}

//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Debug, Clone)]
pub struct FromLongestPath<'a, P> {
    pb: &'a P
}
impl <'a, P> FromLongestPath<'a, P> {
    pub fn new(pb: &'a P) -> FromLongestPath<'a, P>{
        FromLongestPath {pb}
    }
}
impl <'a, T, P> LoadVars<T> for FromLongestPath<'a, P> where P: Problem<T> {
    fn variables(&self, node: &Node<T>) -> VarSet {
        let mut vars = self.pb.all_vars();
        for d in node.info.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}