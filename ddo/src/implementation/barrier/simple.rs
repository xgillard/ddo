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

//! This module (and its submodule) provide the abstractions for the basic
//! building blocks of an MDD solvers. A client willing to use our library to
//! implement a solver for his/her particular problem should look into the `dp`
//! submodule. Indeed, `dp` is the place where the traits `Problem` and
//! `Relaxation` are defined. These are the two abstractions that one *must*
//! implement in order to be able to use our library.

use std::{sync::Arc, hash::Hash};

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::{Barrier, Threshold};

/// Simple implementation of the Barrier using one hashmap for each layer,
/// each protected with a read-write lock.
pub struct SimpleBarrier<T>
where T: Hash + Eq {
    thresholds_by_layer: Vec<RwLock<FxHashMap<Arc<T>, Threshold>>>,
}

impl<T> SimpleBarrier<T>
where T: Hash + Eq {
    pub fn new(nb_variables: usize) -> Self {
        let mut layers = vec![];
        for _ in 0..=nb_variables {
            layers.push(RwLock::new(Default::default()));
        }
        SimpleBarrier { thresholds_by_layer: layers }
    }
}

impl<T> Barrier for SimpleBarrier<T>
where T: Hash + Eq {
    type State = T;

    fn get_threshold(&self, state: Arc<T>, depth: usize) -> Option<Threshold> {
        self.thresholds_by_layer[depth].read().get(state.as_ref()).copied()
    }

    fn update_threshold(&self, state: Arc<T>, depth: usize, value: isize, explored: bool) {
        let mut guard = self.thresholds_by_layer[depth].write();
        let current = guard
            .entry(state)
            .or_insert(Threshold { value: isize::MIN, explored: false });
        let mut new = Threshold { value, explored };
        *current = *current.max(&mut new);
    }

    fn clear_layer(&self, depth: usize) {
        self.thresholds_by_layer[depth].write().clear();
    }

    fn clear(&self) {
        self.thresholds_by_layer.iter().for_each(|l| l.write().clear());
    }
}