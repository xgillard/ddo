use crate::core::abstraction::heuristics::{WidthHeuristic, VariableHeuristic, LoadVars, NodeOrdering};
use crate::core::abstraction::mdd::{MDD, Node};
use std::hash::Hash;
use crate::core::abstraction::dp::{VarSet, Variable, Problem};
use std::marker::PhantomData;
use std::cmp::Ordering;
use compare::Compare;

//~~~~~ Max Width Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct FixedWidth(pub usize);
impl <T, N> WidthHeuristic<T, N> for FixedWidth
    where T : Clone + Hash + Eq,
          N : Node<T> {
    fn max_width(&self, _dd: &impl MDD<T, N>) -> usize {
        self.0
    }
}

pub struct NbUnassigned;
impl <T, N> WidthHeuristic<T, N> for NbUnassigned
    where T : Clone + Hash + Eq,
          N : Node<T> {
    fn max_width(&self, dd: &impl MDD<T, N>) -> usize {
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
impl <T, N> VariableHeuristic<T, N> for NaturalOrder
    where T : Clone + Hash + Eq,
          N : Node<T> {

    fn next_var(&self, _dd: &impl MDD<T, N>, vars: &VarSet) -> Option<Variable> {
        vars.iter().next()
    }
}

//~~~~~ Node Ordering Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Clone)]
pub struct FnNodeOrder<T, N, F>
    where T: Clone + Hash + Eq,
          N: Clone + Node<T>,
          F: Clone + Fn(&N, &N) -> Ordering {
    func : F,
    phantom_t : PhantomData<T>,
    phantom_n : PhantomData<N>
}
impl <T, N, F> FnNodeOrder<T, N, F>
    where T: Clone + Hash + Eq,
          N: Clone + Node<T>,
          F: Clone + Fn(&N, &N) -> Ordering {
    pub fn new(func: F) -> FnNodeOrder<T, N, F> {
        FnNodeOrder { func, phantom_t: PhantomData, phantom_n: PhantomData }
    }
}
impl <T, N, F> NodeOrdering<T, N> for FnNodeOrder<T, N, F>
    where T: Clone + Hash + Eq,
          N: Clone + Node<T>,
          F: Clone + Fn(&N, &N) -> Ordering {
}
impl <T, N, F> Compare<N, N> for FnNodeOrder<T, N, F>
    where T: Clone + Hash + Eq,
          N: Clone + Node<T>,
          F: Clone + Fn(&N, &N) -> Ordering {
    fn compare(&self, a: &N, b: &N) -> Ordering {
        (self.func)(a, b)
    }
}
//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct FromLongestPath;
impl <T, P, N> LoadVars<T, P, N> for FromLongestPath
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          N: Node<T> {

    fn variables(&self, pb: &P, node: &N) -> VarSet {
        let mut vars = VarSet::all(pb.nb_vars());
        for d in node.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}

pub struct FnLoadVars<T, P, N, F>
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          N: Node<T>,
          F: Fn(&P, &N) -> VarSet {
    func: F,
    phantom_t: PhantomData<T>,
    phantom_p: PhantomData<P>,
    phantom_n: PhantomData<N>,
}
impl <T, P, N, F> FnLoadVars<T, P, N, F>
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          N: Node<T>,
          F: Fn(&P, &N) -> VarSet {

    pub fn new(func: F) -> FnLoadVars<T, P, N, F> {
        FnLoadVars{func, phantom_t: PhantomData, phantom_p: PhantomData, phantom_n: PhantomData}
    }
}
impl <T, P, N, F> LoadVars<T, P, N> for FnLoadVars<T, P, N, F>
    where T: Hash + Clone + Eq,
          P: Problem<T>,
          N: Node<T>,
          F: Fn(&P, &N) -> VarSet {

    fn variables(&self, pb: &P, node: &N) -> VarSet {
        (self.func)(pb, node)
    }
}