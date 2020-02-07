use std::cmp::Ordering;
use std::hash::Hash;

use compare::Compare;

use crate::core::abstraction::dp::{Problem, ProblemOps};
use crate::core::abstraction::heuristics::{LoadVars, VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{MDD, Node};
use crate::core::common::{Variable, VarSet};

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
        a.get_lp_len().cmp(&b.get_lp_len())
    }
}
pub struct MaxLP;
impl <T> Compare<Node<T>> for MaxLP where T: Clone + Hash + Eq {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.get_lp_len().cmp(&b.get_lp_len()).reverse()
    }
}

#[derive(Debug, Default)]
pub struct MaxUB;
impl <T> Compare<Node<T>> for MaxUB where T: Clone + Hash + Eq {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.ub.cmp(&b.ub).then_with(|| a.lp_len.cmp(&b.lp_len))
    }
}

//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct FromLongestPath;
impl <T, P> LoadVars<T, P> for FromLongestPath
    where T: Clone + Eq,
          P: Problem<T> {

    fn variables(&self, pb: &P, node: &Node<T>) -> VarSet {
        let mut vars = pb.all_vars();
        for d in node.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}