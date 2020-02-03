use crate::core::abstraction::heuristics::{WidthHeuristic, VariableHeuristic, LoadVars, NodeOrdering};
use crate::core::abstraction::mdd::{MDD, Node};
use std::hash::Hash;
use crate::core::abstraction::dp::{VarSet, Variable, Problem};
use std::marker::PhantomData;
use std::cmp::Ordering;
use compare::Compare;

//~~~~~ Max Width Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct FixedWidth(pub usize);
impl <T> WidthHeuristic<T> for FixedWidth
    where T : Clone + Hash + Eq {
    fn max_width(&self, _dd: &dyn MDD<T>) -> usize {
        self.0
    }
}

pub struct NbUnassigned;
impl <T> WidthHeuristic<T> for NbUnassigned
    where T : Clone + Hash + Eq  {
    fn max_width(&self, dd: &dyn MDD<T>) -> usize {
        dd.unassigned_vars().len()
    }
}

//~~~~~ Variable Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Default)]
pub struct NaturalOrder;
impl NaturalOrder {
    pub fn new() -> NaturalOrder {
        NaturalOrder{}
    }
}
impl <T> VariableHeuristic<T> for NaturalOrder
    where T : Clone + Hash + Eq {

    fn next_var(&self, _dd: &dyn MDD<T>, vars: &VarSet) -> Option<Variable> {
        vars.iter().next()
    }
}

//~~~~~ Node Ordering Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Clone)]
pub struct FnNodeOrder<T, F>
    where T: Clone + Hash + Eq,
          F: Clone + Fn(&Node<T>, &Node<T>) -> Ordering {
    func : F,
    phantom : PhantomData<T>
}
impl <T, F> FnNodeOrder<T, F>
    where T: Clone + Hash + Eq,
          F: Clone + Fn(&Node<T>, &Node<T>) -> Ordering {
    pub fn new(func: F) -> FnNodeOrder<T, F> {
        FnNodeOrder { func, phantom: PhantomData }
    }
}
impl <T, F> NodeOrdering<T> for FnNodeOrder<T, F>
    where T: Clone + Hash + Eq,
          F: Clone + Fn(&Node<T>, &Node<T>) -> Ordering {
}
impl <T, F> Compare<Node<T>> for FnNodeOrder<T, F>
    where T: Clone + Hash + Eq,
          F: Clone + Fn(&Node<T>, &Node<T>) -> Ordering {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        (self.func)(a, b)
    }
}
//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct FromLongestPath;
impl <T, P> LoadVars<T, P> for FromLongestPath
    where T: Hash + Clone + Eq,
          P: Problem<T> {

    fn variables(&self, pb: &P, node: &Node<T>) -> VarSet {
        let mut vars = VarSet::all(pb.nb_vars());
        for d in node.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}

pub struct FnLoadVars<T, P, F>
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          F: Fn(&P, &Node<T>) -> VarSet {
    func: F,
    phantom_t: PhantomData<T>,
    phantom_p: PhantomData<P>
}
impl <T, P, F> FnLoadVars<T, P, F>
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          F: Fn(&P, &Node<T>) -> VarSet {

    pub fn new(func: F) -> FnLoadVars<T, P, F> {
        FnLoadVars{func, phantom_t: PhantomData, phantom_p: PhantomData }
    }
}
impl <T, P, F> LoadVars<T, P> for FnLoadVars<T, P, F>
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          F: Fn(&P, &Node<T>) -> VarSet {

    fn variables(&self, pb: &P, node: &Node<T>) -> VarSet {
        (self.func)(pb, node)
    }
}