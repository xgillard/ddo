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
    fn partial_cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Option<Ordering> {
        let mut ordering = Ordering::Equal;
        if self.use_value() {
            ordering = val_a.cmp(&val_b);
        }
        for i in 0..self.nb_dimensions(a) {
            match (ordering, self.get_coordinate(a, i).cmp(&self.get_coordinate(b, i))) {
                (Ordering::Less, Ordering::Greater)  => return None,
                (Ordering::Greater, Ordering::Less)  => return None,
                (Ordering::Equal, Ordering::Greater) => ordering = Ordering::Greater,
                (Ordering::Equal, Ordering::Less)    => ordering = Ordering::Less,
                (_, _)                               => (),
            }
        }
        Some(ordering)
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

pub trait DominanceChecker {
    type State;
    
    /// Returns true if the state is dominated by a stored one
    /// And insert the (key, value) pair otherwise
    fn is_dominated_or_insert(&self, state: Arc<Self::State>, value: isize) -> bool;

    /// Comparator to order states by increasing value, regardless of their key
    fn cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Ordering;
    
}