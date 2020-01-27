use crate::core::abstraction::dp::{Decision, Variable, VarSet};
use std::hash::Hash;
use std::collections::HashMap;

pub trait Node<T>
    where T : Clone + Hash + Eq  {
    fn is_exact(&self) -> bool;
    fn get_state(&self)-> &T;
    fn get_lp_len(&self) -> i32;
    fn get_ub(&self) -> i32;
    fn set_ub(&mut self, ub: i32);

    fn longest_path(&self) -> Vec<Decision>;
}

#[derive(Copy, Clone)]
pub enum MDDType {
    Relaxed,
    Restricted,
    Exact
}

pub trait MDD<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T> {

    fn mdd_type(&self)           -> MDDType;
    fn current_layer(&self)      -> &[N];
    fn exact_cutset(&self)       -> &[N];
    fn next_layer(&self)         -> &HashMap<T, N>;

    fn last_assigned(&self)      -> Variable;
    fn unassigned_vars(&self)    -> &VarSet;

    fn is_exact(&self)           -> bool;
    fn best_value(&self)         -> i32;
    fn longest_path(&self)       -> Vec<Decision>;
}