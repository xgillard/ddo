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

use crate::{Dominance, DominanceChecker, DominanceCmpResult, DominanceCheckResult};

/// Simple implementation of a dominance checker that stores a vector of non-dominated
/// states for each distinct key.

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
{
    dominance: D,
    data: DashMap<D::Key, Vec<DominanceEntry<D::State>>, fxhash::FxBuildHasher>,
}

impl<D> SimpleDominanceChecker<D> 
where
    D: Dominance,
    D::Key: Eq + PartialEq + Hash,
{
    pub fn new(dominance: D) -> Self {
        Self { dominance, data: Default::default() }
    }
}

impl<D> DominanceChecker for SimpleDominanceChecker<D> 
where
    D: Dominance,
    D::Key: Eq + PartialEq + Hash,
{
    type State = D::State;

    fn is_dominated_or_insert(&self, state: Arc<Self::State>, value: isize) -> DominanceCheckResult {
        if let Some(key) = self.dominance.get_key(state.clone()) {
            match self.data.entry(key) {
                Entry::Occupied(mut e) => {
                    let mut dominated = false;
                    let mut threshold = Some(isize::MAX);
                    e.get_mut().retain(|other| {
                        match self.dominance.partial_cmp(state.as_ref(), value, other.state.as_ref(), other.value) {
                            Some(cmp) => match cmp {
                                DominanceCmpResult { ordering: Ordering::Less, only_val_diff} => {
                                    dominated = true;
                                    if self.dominance.use_value() {
                                        if only_val_diff {
                                            threshold = threshold.min(Some(other.value.saturating_sub(1)));
                                        } else {
                                            threshold = threshold.min(Some(other.value));
                                        }
                                    }
                                    true
                                },
                                DominanceCmpResult { ordering: Ordering::Equal, only_val_diff: _ } => false,
                                DominanceCmpResult { ordering: Ordering::Greater, only_val_diff: _ } => false,
                            },
                            None => true,
                        }
                    });
                    if !dominated {
                        threshold = None;
                        e.get_mut().push(DominanceEntry { state, value });
                    }
                    DominanceCheckResult { dominated, threshold }
                },
                Entry::Vacant(e) => {
                    e.insert(vec![DominanceEntry { state, value }]);
                    DominanceCheckResult { dominated: false, threshold: None }
                },
            }
        } else {
            DominanceCheckResult { dominated: false, threshold: None }
        }
    }

    fn cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Ordering {
        self.dominance.cmp(a, val_a, b, val_b)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{Dominance, SimpleDominanceChecker, DominanceChecker, DominanceCheckResult};

    #[test]
    fn not_dominated_when_keys_are_different() {
        let dominance = SimpleDominanceChecker::new(DummyDominanceWithValue);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![3, 0]), 3));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![2, 0]), 2));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![1, 0]), 1));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 0));
    }

    #[test]
    fn dominated_when_keys_are_equal() {
        let dominance = SimpleDominanceChecker::new(DummyDominance);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 3]), 0));

        let res = dominance.is_dominated_or_insert(Arc::new(vec![0, 2]), 2);
        assert!(res.dominated);

        let res = dominance.is_dominated_or_insert(Arc::new(vec![0, 1]), 1);
        assert!(res.dominated);

        let res = dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 0);
        assert!(res.dominated);

        let dominance = SimpleDominanceChecker::new(DummyDominanceWithValue);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 3));

        let res = dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 2);
        assert!(res.dominated);

        let res = dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 1);
        assert!(res.dominated);

        let res = dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 0);
        assert!(res.dominated);

        let res = dominance.is_dominated_or_insert(Arc::new(vec![0, -1]), 3);
        assert!(res.dominated);
    }

    #[test]
    fn not_dominated_when_keys_are_equal() {
        let dominance = SimpleDominanceChecker::new(DummyDominance);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0, 3]), 3));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0, 3]), 1));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 1, 1]), 5));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0, 4]), 3));

        let dominance = SimpleDominanceChecker::new(DummyDominanceWithValue);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 3));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 3]), 0));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 1]), 1));
        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 5));
    }

    #[test]
    fn pruning_threshold_when_value_is_used() {
        let dominance = SimpleDominanceChecker::new(DummyDominanceWithValue);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 3));
        assert_eq!(DominanceCheckResult{ dominated: true, threshold: Some(2) }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 2));
        assert_eq!(DominanceCheckResult{ dominated: true, threshold: Some(2) }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 1));
        assert_eq!(DominanceCheckResult{ dominated: true, threshold: Some(3) }, dominance.is_dominated_or_insert(Arc::new(vec![0, -1]), 0));
    }

    #[test]
    fn entry_is_added_only_when_dominant() {
        let dominance = SimpleDominanceChecker::new(DummyDominanceWithValue);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 3));
        assert!(dominance.data.get(&0).is_some());
        assert_eq!(1, dominance.data.get(&0).unwrap().len());

        assert_eq!(DominanceCheckResult{ dominated: true, threshold: Some(2) }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 1));
        assert_eq!(1, dominance.data.get(&0).unwrap().len());

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 1]), 1));
        assert_eq!(2, dominance.data.get(&0).unwrap().len());

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, -1]), 5));
        assert_eq!(3, dominance.data.get(&0).unwrap().len());
    }

    #[test]
    fn entry_is_removed_when_dominated() {
        let dominance = SimpleDominanceChecker::new(DummyDominanceWithValue);

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 0]), 3));
        assert!(dominance.data.get(&0).is_some());
        assert_eq!(1, dominance.data.get(&0).unwrap().len());

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 1]), 5));
        assert_eq!(1, dominance.data.get(&0).unwrap().len());

        assert_eq!(DominanceCheckResult{ dominated: false, threshold: None }, dominance.is_dominated_or_insert(Arc::new(vec![0, 2]), 7));
        assert_eq!(1, dominance.data.get(&0).unwrap().len());
    }

    struct DummyDominance;
    impl Dominance for DummyDominance {
        type State = Vec<isize>;
        type Key = isize;

        fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
            Some(state[0])
        }

        fn nb_dimensions(&self, state: &Self::State) -> usize {
            state.len()
        }

        fn get_coordinate(&self, state: &Self::State, i: usize) -> isize {
            state[i]
        }
    }

    struct DummyDominanceWithValue;
    impl Dominance for DummyDominanceWithValue {
        type State = Vec<isize>;
        type Key = isize;

        fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
            Some(state[0])
        }

        fn nb_dimensions(&self, state: &Self::State) -> usize {
            state.len()
        }

        fn get_coordinate(&self, state: &Self::State, i: usize) -> isize {
            state[i]
        }

        fn use_value(&self) -> bool {
            true
        }
    }
}