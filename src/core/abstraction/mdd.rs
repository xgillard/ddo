use crate::core::abstraction::dp::{Decision, Variable};
use std::rc::Rc;
use std::hash::Hash;
use std::marker::PhantomData;
use bitset_fixed::BitSet;

#[derive(Clone)]
pub struct Arc<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    pub src     : Rc<N>,
    pub decision: Decision,
    pub weight  : i32,

    pub phantom: PhantomData<T>
}

pub trait Node<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    fn is_exact(&self) -> bool;
    fn get_state(&self)-> &T;
    fn get_lp_len(&self) -> i32;
    fn get_lp_arc(&self) -> &Option<Arc<T, N>>;
    fn get_ub(&self) -> i32;
    fn set_ub(&mut self, ub: i32);

    fn add_arc (&mut self, arc: Arc<T, N>);
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
          N : Node<T, N> {

    fn mdd_type(&self)        -> MDDType;
    fn current_layer(&self)   -> &[Rc<N>];
    fn exact_cutset(&self)    -> &[N];
    fn last_assigned(&self)   -> Variable;
    fn unassigned_vars(&self) -> &BitSet;

    fn is_exact(&self)        -> bool;
    fn best_value(&self)      -> i32;
    fn longest_path(&self)    -> Vec<Decision>;
}