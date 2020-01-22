use super::super::abstraction::mdd::*;
use crate::core::abstraction::dp::Decision;
use std::cmp::max;

struct PooledNode<T> {
    state   : T,
    is_exact: bool,
    lp_len  : i32,
    ub      : i32,

    lp_arc  : Option<Arc<T>>,
}

impl <T> PooledNode<T> {
    fn new(state: T, is_exact: bool, lp_len: i32, ub: i32) -> PooledNode<T> {
        PooledNode{state, is_exact, lp_len, ub, lp_arc: None}
    }
}

impl <T> Node<T> for PooledNode<T> {
    fn is_exact(&self) -> bool {
        self.is_exact
    }
    fn get_state(&self)-> &T {
        self.state
    }
    fn get_lp_len(&self) -> i32 {
        self.lp_len
    }
    fn get_lp_arc(&self) -> Option<Arc<T>> {
        self.lp_arc.clone()
    }
    fn get_ub(&self) -> i32 {
        self.ub
    }
    fn set_ub(&mut self, ub: i32) {
        self.ub = max(self.ub, ub);
    }

    fn clear(&mut self) {
        self.arcs.clear()
    }

    fn add_arc (&mut self, arc: Arc<T>) {
        if (self.lp_arc.is_none() || arc.source.lp_len + arc.weight > self.lp_len) {
            self.ub     = max(self.lp_len, arc.source.lp_length + arc.weight);
            self.lp_arc = Some(arc.clone());
            self.lp_len = arc.source.lp_len + arc.weight;
        }

        self.is_exact &= arc.source.is_exact;
        self.arcs.push(arc);
    }

    fn longest_path(&self) -> Vec<Decision> {
        let mut ret = vec![];
        let mut arc = &self.lp_arc;

        while (arc.is_some()) {
            ret.push(arc.decision);
            arc = &ret.src.lp_arc;
        }

        ret
    }
}