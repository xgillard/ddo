use std::cmp::Ordering;
use std::hash::Hash;

use compare::Compare;

use crate::core::abstraction::dp::Problem;
use crate::core::abstraction::heuristics::{LoadVars, VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{MDD, Node};
use crate::core::common::{Variable, VarSet};
use std::marker::PhantomData;

//~~~~~ Max Width Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct FixedWidth(pub usize);
impl <T> WidthHeuristic<T> for FixedWidth
    where T : Clone + Hash + Eq {
    fn max_width(&self, _free: &VarSet) -> usize {
        self.0
    }
}

pub struct NbUnassigned;
impl <T> WidthHeuristic<T> for NbUnassigned
    where T : Clone + Hash + Eq  {
    fn max_width(&self, free: &VarSet) -> usize {
        free.len()
    }
}

//~~~~~ Variable Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Default)]
pub struct NaturalOrder;
impl <T> VariableHeuristic<T> for NaturalOrder
    where T : Clone + Hash + Eq {

    fn next_var(&self, _dd: &dyn MDD<T>, vars: &VarSet) -> Option<Variable> {
        vars.iter().next()
    }
}

//~~~~~ Node Ordering Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Debug, Default)]
pub struct MinLP;
impl <T> Compare<Node<T>> for MinLP where T: Clone + Hash + Eq {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.lp_len.cmp(&b.info.lp_len)
    }
}
pub struct MaxLP;
impl <T> Compare<Node<T>> for MaxLP where T: Clone + Hash + Eq {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.lp_len.cmp(&b.info.lp_len).reverse()
    }
}

#[derive(Debug, Default)]
pub struct MaxUB;
impl <T> Compare<Node<T>> for MaxUB where T: Clone + Hash + Eq {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.ub.cmp(&b.info.ub).then_with(|| a.info.lp_len.cmp(&b.info.lp_len))
    }
}

//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct FromLongestPath<'a, T, P>
    where T: Clone + Eq,
          P: Problem<T> {
    pb: &'a P,
    phantom: PhantomData<T>
}
impl <'a, T, P> FromLongestPath<'a, T, P>
    where T: Clone + Eq,
          P: Problem<T> {
    pub fn new(pb: &'a P) -> FromLongestPath<'a, T, P>{
        FromLongestPath {pb, phantom: PhantomData}
    }
}
impl <'a, T, P> LoadVars<T> for FromLongestPath<'a, T, P>
    where T: Clone + Eq,
          P: Problem<T> {

    fn variables(&self, node: &Node<T>) -> VarSet {
        let mut vars = self.pb.all_vars();
        for d in node.info.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}