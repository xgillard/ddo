// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! This module contains the implementation of some data structures that are
//! useful when implementing a solver. In particular, it provides the NoDupPq
//! which provides the implementation of a priority queue, but avoid it to
//! hold two equivalent states.

use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::sync::Arc;

use binary_heap_plus::BinaryHeap;
use metrohash::MetroHashMap;

use crate::common::{FrontierNode, PartialAssignment};
use crate::implementation::heuristics::MaxUB;

pub struct NoDupPQ<T> where T: Eq + Hash {
    states: MetroHashMap<Arc<T>, (isize, Arc<PartialAssignment>)>,
    queue : BinaryHeap<FrontierNode<T>, MaxUB>,
}

impl <T> NoDupPQ<T>  where T: Eq + Hash {
    pub fn new() -> Self {
        NoDupPQ {
            states: MetroHashMap::default(),
            queue : BinaryHeap::from_vec_cmp(vec![], MaxUB),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    pub fn push(&mut self, node: FrontierNode<T>) {
        match self.states.entry(Arc::clone(&node.state)) {
            Entry::Vacant(e) => {
                e.insert((node.lp_len, Arc::clone(&node.path)));
                self.queue.push(node);
            },
            Entry::Occupied(mut e) => {
                let (val, path) = e.get_mut();
                if node.lp_len > *val {
                    *val = node.lp_len;
                    *path= Arc::clone(&node.path);
                }
            }
        }
    }
    pub fn pop(&mut self) -> Option<FrontierNode<T>> {
        self.queue.pop().map(|mut n| {
            let (val,path) = self.states.remove(&n.state).unwrap();
            n.lp_len = val;
            n.path   = path;
            n
        })
    }
    pub fn clear(&mut self) {
        self.states.clear();
        self.queue .clear();
    }
}
impl <T> Default for NoDupPQ<T>  where T: Eq + Hash {
    fn default() -> Self { Self::new() }
}