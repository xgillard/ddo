use crate::core::abstraction::dp::Decision;
use std::rc::Rc;
use std::hash::Hash;
use std::marker::PhantomData;

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

pub trait MDD<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {

    fn current_layer(&self) -> &[Rc<N>];
}