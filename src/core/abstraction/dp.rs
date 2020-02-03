use std::i32;
use bitset_fixed::BitSet;
use crate::core::utils::BitSetIter;
use std::ops::Not;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Variable(pub usize);

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct VarSet(pub BitSet);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Decision {
    pub variable : Variable,
    pub value    : i32
}

pub trait Problem<T> {
    fn nb_vars      (&self) -> usize;
    fn initial_state(&self) -> T;
    fn initial_value(&self) -> i32;

    fn domain_of      (&self, state: &T, var: Variable)     -> &[i32];
    fn transition     (&self, state: &T, vars : &VarSet, d: &Decision) -> T;
    fn transition_cost(&self, state: &T, vars : &VarSet, d: &Decision) -> i32;

    // Optional method for the case where you'd want to use a pooled mdd implementation
    // Returns true iff taking a decision on 'variable' might have an impact (state or lp)
    // on a node having the given 'state'
    #[allow(unused_variables)]
    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        true
    }
}

pub trait Relaxation<T> {
    fn merge_states(&self, states: &[&T])                         -> T;
    fn relax_cost  (&self, from: &T, to: &T, decision: &Decision) -> i32;

    // Optionally compute a rough upper bound on the objective value reachable
    // from the given state.
    #[allow(unused_variables)]
    fn rough_ub(&self, lp: i32, s: &T) -> i32 {
        i32::max_value()
    }
}

pub struct VarSetIter<'a>(pub BitSetIter<'a>);

impl VarSet {
    pub fn all(n: usize) -> VarSet {
        VarSet(BitSet::new(n).not())
    }
    pub fn add(&mut self, v: Variable) {
        self.0.set(v.0, true)
    }
    pub fn remove(&mut self, v: Variable) {
        self.0.set(v.0, false)
    }
    pub fn contains(&self, v: Variable) -> bool {
        self.0[v.0]
    }
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> VarSetIter {
        VarSetIter(BitSetIter::new(&self.0))
    }
}
impl Iterator for VarSetIter<'_> {
    type Item = Variable;

    fn next(&mut self) -> Option<Variable> {
        self.0.next().map(Variable)
    }
}