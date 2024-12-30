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

use dashmap::DashMap;

use crate::{Cache, Threshold};

/// Simple implementation of Cache using one hashmap for each layer,
/// each protected with a read-write lock.
#[derive(Debug)]
pub struct SimpleCache<State>
where State: Hash + Eq {
    thresholds_by_layer: Vec<DashMap<Arc<State>, Threshold, fxhash::FxBuildHasher>>,
}
impl <State> Default for SimpleCache<State> 
where State: Hash + Eq {
    fn default() -> Self {
        Self { thresholds_by_layer: vec![] }
    }
}

impl<State> Cache<State> for SimpleCache<State>
where State: Hash + Eq {
    fn initialize(&mut self, problem: &dyn crate::Problem<State = State>) {
        let nb_variables = problem.nb_variables();
        for _ in 0..=nb_variables {
            self.thresholds_by_layer.push(Default::default());
        }
    }

    fn get_threshold(&self, state: &State, depth: usize) -> Option<Threshold> {
        self.thresholds_by_layer[depth].get(state).as_deref().copied()
    }

    fn update_threshold(&self, state: Arc<State>, depth: usize, value: isize, explored: bool) {
        self.thresholds_by_layer[depth].entry(state)
            .and_modify(|e| *e = Threshold { value, explored }.max(*e))
            .or_insert(Threshold { value, explored });
    }

    fn clear_layer(&self, depth: usize) {
        self.thresholds_by_layer[depth].clear();
    }

    fn clear(&self) {
        self.thresholds_by_layer.iter().for_each(|l| l.clear());
    }
}