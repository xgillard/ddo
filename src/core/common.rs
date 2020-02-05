//! This module defines the most basic data types that are used both in the
//! abstraction and implementation layers of our library. These are also the
//! types your client library is likely to work with.

use bitset_fixed::BitSet;
use crate::core::utils::{BitSetIter, LexBitSet};
use std::ops::Not;
use std::cmp::Ordering;

/// This type denotes a variable from the optimization problem at hand.
/// In this case, each variable is assumed to be identified with an integer
/// ranging from 0 until `problem.nb_vars()`
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Variable(pub usize);

/// This denotes a decision that was made during the search. It affects a given
/// `value` to the specified `variable`. Any given `Decision` should be
/// understood as ```[[ variable = value ]]````
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Decision {
    pub variable : Variable,
    pub value    : i32
}

/// This type denotes a set of variable. It encodes them compactly as a fixed
/// size bitset. A `VarSet` can be efficiently iterated upon.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct VarSet(pub BitSet);

/// This is the implementation of the core features of a `VarSet`.
impl VarSet {
    /// Returns a `VarSet` where all the possible `n` variables are presetn.
    pub fn all(n: usize) -> VarSet {
        VarSet(BitSet::new(n).not())
    }
    /// Adds the given variable `v` to the set if it is not already present.
    pub fn add(&mut self, v: Variable) {
        self.0.set(v.0, true)
    }
    /// Removes the variable `v` from the set if it was present.
    pub fn remove(&mut self, v: Variable) {
        self.0.set(v.0, false)
    }
    /// Returns true iff the set contains the variable `v`.
    pub fn contains(&self, v: Variable) -> bool {
        self.0[v.0]
    }
    /// Returns the count of variables that are present in the set.
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }
    /// Returns true iff no variables are preset in the set.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns an iterator in this set of variables.
    pub fn iter(&self) -> VarSetIter {
        VarSetIter(BitSetIter::new(&self.0))
    }
}
impl Ord for VarSet {
    fn cmp(&self, other: &Self) -> Ordering {
        LexBitSet(&self.0).cmp(&LexBitSet(&other.0))
    }
}
impl PartialOrd for VarSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// This type denotes the iterator used to iterate over the `Variable`s to a
/// given `VarSet`. It should never be manually instantiated, but always via
/// the `iter()` method from the varset.
pub struct VarSetIter<'a>(pub BitSetIter<'a>);

/// Actually implement the iterator protocol.
impl Iterator for VarSetIter<'_> {
    type Item = Variable;
    /// Returns the next variable from the set, or `None` if all variables have
    /// already been iterated upon.
    fn next(&mut self) -> Option<Variable> {
        self.0.next().map(Variable)
    }
}