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

use std::{cmp::Ordering, sync::Arc};

/// Helper struct that encapsulates the result of a dominance comparison
#[derive(Debug, PartialEq, Eq)]
pub struct DominanceCmpResult {
    /// The ordering of two states with respect to their dominance relation
    pub ordering: Ordering,
    /// Whether the only difference between the two states is their value
    pub only_val_diff: bool,
}

/// This trait abstracts gives the possibility to model dominance relations
/// between the states of a specific problem. The dominance relation is evaluated
/// only for pairs of states that are mapped to the same key. A dominance relation
/// exists if the coordinates of a state are greater or equal than those of another state
/// for all given dimensions. The value obtained by the solver for each state can
/// optionally be used as a coordinate in the comparison.
pub trait Dominance {
    type State;
    type Key;

    /// Takes a state and returns a key that maps it to comparable states
    fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key>;

    /// Returns the number of dimensions to include in the comparison
    fn nb_dimensions(&self, state: &Self::State) -> usize;

    /// Returns the i-th coordinate associated with the given state
    /// Greater is better for the dominance check
    fn get_coordinate(&self, state: &Self::State, i: usize) -> isize;

    /// Whether to include the value as a coordinate in the dominance check
    fn use_value(&self) -> bool { false }

    /// Checks whether there is a dominance relation between the two states, given the coordinates
    /// provided by the function get_coordinate evaluated for all i in 0..self.nb_dimensions()
    /// Note: the states are assumed to have the same key, otherwise they are not comparable for dominance
    fn partial_cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Option<DominanceCmpResult> {
        let mut ordering = Ordering::Equal;
        for i in 0..self.nb_dimensions(a) {
            match (ordering, self.get_coordinate(a, i).cmp(&self.get_coordinate(b, i))) {
                (Ordering::Less, Ordering::Greater)  => return None,
                (Ordering::Greater, Ordering::Less)  => return None,
                (Ordering::Equal, Ordering::Greater) => ordering = Ordering::Greater,
                (Ordering::Equal, Ordering::Less)    => ordering = Ordering::Less,
                (_, _)                               => (),
            }
        }
        if self.use_value() {
            match (ordering, val_a.cmp(&val_b)) {
                (Ordering::Less, Ordering::Greater)  => None,
                (Ordering::Greater, Ordering::Less)  => None,
                (Ordering::Equal, Ordering::Greater) => Some(DominanceCmpResult { ordering: Ordering::Greater, only_val_diff: true }),
                (Ordering::Equal, Ordering::Less)    => Some(DominanceCmpResult { ordering: Ordering::Less, only_val_diff: true }),
                (_, _)                               => Some(DominanceCmpResult { ordering, only_val_diff: false }),
            }
        } else {
            Some(DominanceCmpResult { ordering, only_val_diff: false })
        }
    }

    /// Comparator to order states by increasing value, regardless of their key
    fn cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Ordering {
        if self.use_value() {
            match val_a.cmp(&val_b) {
                Ordering::Less    => return Ordering::Less,
                Ordering::Greater => return Ordering::Greater,
                Ordering::Equal   => (),
            }
        }
        for i in 0..self.nb_dimensions(a) {
            match self.get_coordinate(a, i).cmp(&self.get_coordinate(b, i)) {
                Ordering::Less    => return Ordering::Less,
                Ordering::Greater => return Ordering::Greater,
                Ordering::Equal   => (),
            }
        }
        Ordering::Equal
    }

}

/// Helper struct that encapsulates the result of a dominance check
#[derive(Debug, PartialEq, Eq)]
pub struct DominanceCheckResult {
    /// Whether the state is dominated by a state contained in the checker
    pub dominated: bool,
    /// When the state is dominated and the value is considered in the comparison,
    /// the pruning threshold must be returned i.e. the minimum value that would
    /// allow the same state to avoid being dominated
    pub threshold: Option<isize>,
}

pub trait DominanceChecker {
    type State;
    
    /// Returns true if the state is dominated by a stored one, and a potential
    /// pruning threshold, and inserts the (key, value) pair otherwise
    fn is_dominated_or_insert(&self, state: Arc<Self::State>, value: isize) -> DominanceCheckResult;

    /// Comparator to order states by increasing value, regardless of their key
    fn cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Ordering;
    
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, cmp::Ordering};

    use crate::{Dominance, DominanceCmpResult};

    #[test]
    fn by_default_value_is_unused() {
        let dominance = DummyDominance;
        assert!(!dominance.use_value());
    }

    #[test]
    fn partial_cmp_returns_none_when_coordinates_disagree() {
        let dominance = DummyDominance;

        assert_eq!(None, dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, -1, 1], 0));

        let dominance = DummyDominanceWithValue;

        assert_eq!(None, dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, 1], -1));
    }

    #[test]
    fn partial_cmp_returns_some_when_coordinates_agree() {
        let dominance = DummyDominance;


        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Less, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, 1], 0));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Less, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, 1], -1));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Greater, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, -1], 0));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Greater, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, -1], 1));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Equal, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, 0], 1));

        let dominance = DummyDominanceWithValue;

        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Less, only_val_diff: true}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, 0], 1));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Less, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![1, 1, 1], 1));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Greater, only_val_diff: true}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, 0], -1));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Greater, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![-1, -1, -1], -1));
        assert_eq!(Some(DominanceCmpResult{ordering: Ordering::Equal, only_val_diff: false}), dominance.partial_cmp(&vec![0, 0, 0], 0, &vec![0, 0, 0], 0));
    }

    #[test]
    fn cmp_returns_first_diff() {
        let dominance = DummyDominance;

        assert_eq!(Ordering::Less, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, 1], 0));
        assert_eq!(Ordering::Less, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 1, -1], 0));
        assert_eq!(Ordering::Less, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, 1], -1));
        assert_eq!(Ordering::Greater, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, -1], 0));
        assert_eq!(Ordering::Greater, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, -1, 1], 0));
        assert_eq!(Ordering::Greater, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, -1], 1));
        assert_eq!(Ordering::Equal, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, 0], 1));

        let dominance = DummyDominanceWithValue;

        assert_eq!(Ordering::Less, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, 1], 0));
        assert_eq!(Ordering::Less, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 1, -1], 0));
        assert_eq!(Ordering::Less, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, -1], 1));
        assert_eq!(Ordering::Greater, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, -1], 0));
        assert_eq!(Ordering::Greater, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, -1, 1], 0));
        assert_eq!(Ordering::Greater, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, 1], -1));
        assert_eq!(Ordering::Equal, dominance.cmp(&vec![0, 0, 0], 0, &vec![0, 0, 0], 0));
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