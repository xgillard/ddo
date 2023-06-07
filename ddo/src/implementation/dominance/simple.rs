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

use std::{hash::Hash, cmp::Ordering, sync::Arc, fmt::Debug};
use dashmap::{DashMap, mapref::entry::Entry};

use crate::{Dominance, DominanceChecker};

/// Simple implementation of a dominance checker that stores a vector of non-dominated
/// states for each disctinct key.

#[derive(Debug)]
struct DominanceEntry<T> {
    state: Arc<T>,
    value: isize,
}

#[derive(Debug)]
pub struct SimpleDominanceChecker<D>
where
    D: Dominance,
    D::Key: Eq + PartialEq + Hash,
    D::State: Clone + Debug,
{
    dominance: D,
    data: DashMap<D::Key, Vec<DominanceEntry<D::State>>, fxhash::FxBuildHasher>,
}

impl<D> SimpleDominanceChecker<D> 
where
    D: Dominance,
    D::Key: Eq + PartialEq + Hash,
    D::State: Clone + Debug,
{
    pub fn new(dominance: D) -> Self {
        Self { dominance, data: Default::default() }
    }
}

impl<D> DominanceChecker for SimpleDominanceChecker<D> 
where
    D: Dominance,
    D::Key: Eq + PartialEq + Hash,
    D::State: Clone + Debug,
{
    type State = D::State;

    fn is_dominated_or_insert(&self, state: Arc<Self::State>, value: isize) -> bool {
        if let Some(key) = self.dominance.get_key(state.clone()) {
            match self.data.entry(key) {
                Entry::Occupied(mut e) => {
                    let mut dominated = false;
                    e.get_mut().retain(|other| {
                        match self.dominance.partial_cmp(state.as_ref(), value, other.state.as_ref(), other.value) {
                            Some(ord) => match ord {
                                Ordering::Less => {
                                    dominated = true;
                                    true
                                },
                                Ordering::Equal => false,
                                Ordering::Greater => false,
                            },
                            None => true,
                        }
                    });
                    if !dominated {
                        e.get_mut().push(DominanceEntry { state, value });
                    }
                    dominated
                },
                Entry::Vacant(e) => {
                    e.insert(vec![DominanceEntry { state, value }]);
                    false
                },
            }
        } else {
            false
        }
    }

    fn cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Ordering {
        self.dominance.cmp(a, val_a, b, val_b)
    }
}