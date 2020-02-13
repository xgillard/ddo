//! This module defines the most basic data types that are used both in the
//! abstraction and implementation layers of our library. These are also the
//! types your client library is likely to work with.

use bitset_fixed::BitSet;
use crate::core::utils::{BitSetIter, LexBitSet};
use std::ops::{Not, Range};
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::hash::{Hasher, Hash};
use std::rc::Rc;

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
    /// Returns a `VarSet` where all the possible `n` variables are present.
    pub fn all(n: usize) -> VarSet {
        VarSet(BitSet::new(n).not())
    }
    /// Creates an empty var set
    pub fn empty() -> VarSet {
        VarSet(BitSet::new(0))
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

/// Utility structure to represent upper and lower bounds
#[derive(Debug, Copy, Clone)]
pub struct Bounds {pub lb: i32, pub ub: i32}

pub enum Domain<'a> {
    Vector(Vec<i32>),
    Slice (&'a [i32]),
    BitSet(&'a BitSet),
    VarSet(&'a VarSet),
    Range (Range<i32>),
}
impl <'a> IntoIterator for Domain<'a> {
    type Item     = i32;
    type IntoIter = DomainIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Domain::Vector(v) => DomainIter::Vector(v.into_iter()),
            Domain::Slice (s) => DomainIter::Slice (s.into_iter()),
            Domain::BitSet(b) => DomainIter::BitSet(BitSetIter::new(b)),
            Domain::VarSet(v) => DomainIter::BitSet(BitSetIter::new(&v.0)),
            Domain::Range (r) => DomainIter::Range (r)
        }
    }
}
pub enum DomainIter<'a> {
    Vector(std::vec::IntoIter<i32>),
    Slice (std::slice::Iter<'a, i32>),
    BitSet(BitSetIter<'a>),
    Range (Range<i32>)
}
impl Iterator for DomainIter<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DomainIter::Vector(i) => i.next(),
            DomainIter::Slice (i) => i.next().map(|x| *x),
            DomainIter::BitSet(i) => i.next().map(|x| x as i32),
            DomainIter::Range (i) => i.next()
        }
    }
}
impl From<Vec<i32>> for Domain<'_> {
    fn from(v: Vec<i32>) -> Self {
        Domain::Vector(v)
    }
}
impl <'a> From<&'a [i32]> for Domain<'a> {
    fn from(s: &'a [i32]) -> Self {
        Domain::Slice(s)
    }
}
impl From<Range<i32>> for Domain<'_> {
    fn from(r: Range<i32>) -> Self {
        Domain::Range(r)
    }
}
impl <'a> From<&'a BitSet> for Domain<'a> {
    fn from(b: &'a BitSet) -> Self {
        Domain::BitSet(b)
    }
}
impl <'a> From<&'a VarSet> for Domain<'a> {
    fn from(b: &'a VarSet) -> Self {
        Domain::VarSet(b)
    }
}

// --- NODE --------------------------------------------------------------------
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Edge<T> where T: Eq + Clone  {
    pub src     : Rc<Node<T>>,
    pub decision: Decision
}

#[derive(Debug, Clone, Eq)]
pub struct NodeInfo<T> where T: Eq + Clone {
    pub is_exact : bool,
    pub lp_len   : i32,
    pub lp_arc   : Option<Edge<T>>,
    pub ub       : i32
}

impl <T> NodeInfo<T> where T: Eq + Clone {
    pub fn new (lp_len: i32, lp_arc: Option<Edge<T>>, is_exact: bool) -> NodeInfo<T> {
        NodeInfo { is_exact, lp_len, lp_arc, ub: i32::max_value() }
    }

    /// Merge other into this node. That is to say, it combines the information
    /// from two nodes that are logically equivalent (should be the same).
    /// Hence, *this has nothing to do with the user-provided `merge_*` operators !*
    pub fn merge(&mut self, other: Self) {
        if  self.lp_len < other.lp_len {
            self.lp_len = other.lp_len;
            self.lp_arc = other.lp_arc;
        }
        self.is_exact &= other.is_exact;
    }

    pub fn longest_path(&self) -> Vec<Decision> {
        let mut ret = vec![];
        let mut arc = &self.lp_arc;

        while arc.is_some() {
            let a = arc.as_ref().unwrap();
            ret.push(a.decision);
            arc = &a.src.info.lp_arc;
        }

        ret
    }
}
impl <T> PartialEq for NodeInfo<T> where T: Eq + Clone {
    fn eq(&self, other: &Self) -> bool {
        self.is_exact == other.is_exact &&
            self.lp_len == other.lp_len &&
            self.ub     == other.ub     &&
            self.lp_arc == other.lp_arc
    }
}
impl <T> Ord for NodeInfo<T> where T: Eq + Clone {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ub.cmp(&other.ub)
            .then_with(|| self.lp_len.cmp(&other.lp_len))
            .then_with(|| self.is_exact.cmp(&other.is_exact))
    }
}
impl <T> PartialOrd for NodeInfo<T> where T: Eq + Clone {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}


#[derive(Debug, Clone, Eq)]
pub struct Node<T> where T: Eq + Clone {
    pub state    : T,
    pub info     : NodeInfo<T>
}

impl <T> Node<T> where T : Eq + Clone {
    pub fn new(state: T, lp_len: i32, lp_arc: Option<Edge<T>>, is_exact: bool) -> Node<T> {
        Node{state, info: NodeInfo::new(lp_len, lp_arc, is_exact)}
    }
}

impl <T> Hash for Node<T> where T: Hash + Eq + Clone {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}
impl <T> PartialEq for Node<T> where T: Eq + Clone {
    fn eq(&self, other: &Self) -> bool {
        self.state.eq(&other.state)
    }
}
impl <T> Ord for Node<T> where T: Eq + Clone + Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl <T> PartialOrd for Node<T> where T: Eq + Clone + PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let res = self.info.cmp(&other.info);
        if res != Equal {
            Some(res)
        } else {
            self.state.partial_cmp(&other.state)
        }
    }
}