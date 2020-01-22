use std::i32;
use bitset_fixed::BitSet;

#[derive(Copy, Clone)]
pub struct Decision {
    pub variable : u32,
    pub value    : i32
}

pub trait Problem<T> {
    fn nb_vars      (&self) -> u32;
    fn initial_state(&self) -> T;
    fn initial_value(&self) -> i32;

    fn domain_of      (&self, state: &T, var: u32)          -> &[i32];
    fn transition     (&self, vars : &BitSet, d: &Decision) -> T;
    fn transition_cost(&self, vars : &BitSet, d: &Decision) -> i32;

    // Optional method for the case where you'd want to use a pooled mdd implementation
    // Returns true iff taking a decision on 'variable' might have an impact (state or lp)
    // on a node having the given 'state'
    fn impacted_by(&self, vars : &BitSet, state: &T) -> bool {
        true
    }
}

pub trait Relaxation<T> {
    fn merge_states(&self, states: &[T])                          -> T;
    fn relax_cost  (&self, from: &T, to: &T, decision: &Decision) -> i32;

    // Optionally compute a rough upper bound on the objective value reachable
    // from the given state.
    fn rough_ub(&self, lp: i32, s: &T, vars: &BitSet) -> i32 {
        i32::max_value()
    }
}