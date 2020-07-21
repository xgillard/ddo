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

//! This module provides the implementation of usual frontiers.

use binary_heap_plus::BinaryHeap;
use crate::common::{FrontierNode, PartialAssignment};
use crate::implementation::heuristics::MaxUB;
use crate::abstraction::frontier::Frontier;
use metrohash::MetroHashMap;
use std::sync::Arc;
use std::collections::hash_map::Entry;
use std::hash::Hash;

/// The simplest frontier implementation you can think of: is basically consists
/// of a binary heap that pushes an pops frontier nodes
pub struct SimpleFrontier<T> {
    heap: BinaryHeap<FrontierNode<T>, MaxUB>
}
impl <T> Default for SimpleFrontier<T> {
    fn default() -> Self {
        Self{ heap: BinaryHeap::from_vec_cmp(vec![], MaxUB) }
    }
}
impl <T> Frontier<T> for SimpleFrontier<T> {
    fn push(&mut self, node: FrontierNode<T>) {
        self.heap.push(node)
    }

    fn pop(&mut self) -> Option<FrontierNode<T>> {
        self.heap.pop()
    }

    fn clear(&mut self) {
        self.heap.clear()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}

/// A frontier that enforces the requirement that a given state will never be
/// present twice in the frontier.
pub struct NoDupFrontier<T> where T: Eq + Hash {
    states: MetroHashMap<Arc<T>, (isize, Arc<PartialAssignment>)>,
    queue : SimpleFrontier<T>
}
impl <T> Default for NoDupFrontier<T> where T: Eq + Hash {
    fn default() -> Self {
        Self {
            states: MetroHashMap::default(),
            queue : SimpleFrontier::default()
        }
    }
}
impl <T> Frontier<T> for NoDupFrontier<T> where T: Eq + Hash {
    fn push(&mut self, node: FrontierNode<T>) {
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

    fn pop(&mut self) -> Option<FrontierNode<T>> {
        self.queue.pop().map(|mut n| {
            let (val,path) = self.states.remove(&n.state).unwrap();
            n.lp_len = val;
            n.path   = path;
            n
        })
    }

    fn clear(&mut self) {
        self.states.clear();
        self.queue .clear();
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}