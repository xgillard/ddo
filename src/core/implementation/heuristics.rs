use crate::core::abstraction::heuristics::{WidthHeuristic, VariableHeuristic, LoadVars};
use crate::core::abstraction::mdd::{MDD, Node};
use std::hash::Hash;
use crate::core::abstraction::dp::{VarSet, Variable, Problem};
use std::marker::PhantomData;

//~~~~~ Max Width Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct FixedWidth(pub usize);
impl <T, N> WidthHeuristic<T, N> for FixedWidth
    where T : Clone + Hash + Eq,
          N : Node<T> {
    fn max_width(&self, _dd: &dyn MDD<T, N>) -> usize {
        self.0
    }
}

pub struct NbUnassigned;
impl <T, N> WidthHeuristic<T, N> for NbUnassigned
    where T : Clone + Hash + Eq,
          N : Node<T> {
    fn max_width(&self, dd: &dyn MDD<T, N>) -> usize {
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

    fn next_var(&self, _dd: &dyn MDD<T, N>, vars: &VarSet) -> Option<Variable> {
        vars.iter().next()
    }
}

//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct FromLongestPath;
impl <T> LoadVars<T> for FromLongestPath
    where T: Hash + Clone + Eq {

    fn variables(&self, pb: &dyn Problem<T>, node: &dyn Node<T>) -> VarSet {
        let mut vars = VarSet::all(pb.nb_vars());
        for d in node.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}

pub struct FromFunction<T, F>
    where T: Hash + Clone + Eq,
          F: Fn(&dyn Problem<T>, &dyn Node<T>) -> VarSet {
    func: F,
    phantom: PhantomData<T>
}
impl <T, F> FromFunction<T, F>
    where T: Hash + Clone + Eq,
          F: Fn(&dyn Problem<T>, &dyn Node<T>) -> VarSet {

    pub fn new(func: F) -> FromFunction<T, F> {
        FromFunction{func, phantom: PhantomData}
    }
}
impl <T, F> LoadVars<T> for FromFunction<T, F>
    where T: Hash + Clone + Eq,
          F: Fn(&dyn Problem<T>, &dyn Node<T>) -> VarSet {

    fn variables(&self, pb: &dyn Problem<T>, node: &dyn Node<T>) -> VarSet {
        (self.func)(pb, node)
    }
}