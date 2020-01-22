use crate::core::abstraction::dp::Decision;
use std::rc::Rc;
use std::cmp::max;
use std::fs::copy;

#[derive(Clone)]
pub struct Arc<T> {
    pub src     : Rc<dyn Node<T>>,
    pub decision: Decision,
    pub weight  : i32
}

pub trait Node<T> {
    fn is_exact(&self) -> bool;
    fn get_state(&self)-> &T;
    fn get_lp_len(&self) -> i32;
    fn get_lp_arc(&self) -> Option<Arc<T>>;
    fn get_ub(&self) -> i32;
    fn set_ub(&mut self, ub: i32);

    fn add_arc (&mut self, arc: Arc<T>);
    fn longest_path(&self) -> Vec<Decision>;
    fn clear(&mut self);
}

pub trait MDD<T>{}