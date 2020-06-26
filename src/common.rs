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

//! This module defines the most basic data types that are used throughout all
//! the code of our library (both at the abstraction and implementation levels).
//! These are also the types your client library is likely to work with.
//!
//! In particular, this module comprises the definition of the following types:
//! - `Variable`
//! - `Domain` (+ associated `DomainIter`)
//! - `Decision`
//! - `PartialAssignment` (+ associated `PartialAssignmentIter`)
//! - `Solution`
//! - `FrontierNode`
//!
//! In addition to the above, it also provides the definition of the following
//! useful types:
//! - `VarSet` (+ associated `VarSetIter`)
//! - `BitSetIter`
//! - `LexBitSet`
//! - `Matrix`

use std::cmp::Ordering;
use std::ops::{Not, Range, RangeInclusive, IndexMut, Index};
use bitset_fixed::BitSet;
use std::cmp::Ordering::Equal;
use std::iter::Cloned;
use std::slice::Iter;
use std::sync::Arc;

// ----------------------------------------------------------------------------
// --- VARIABLE ---------------------------------------------------------------
// ----------------------------------------------------------------------------
/// This type denotes a variable from the optimization problem at hand.
/// In this case, each variable is assumed to be identified with an integer
/// ranging from 0 until `problem.nb_vars()`
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Variable(pub usize);
impl Variable {
    #[inline]
    /// This function retruns the id (numeric value) of the variable.
    ///
    /// # Examples:
    /// ```
    /// # use ddo::common::Variable;
    /// assert_eq!(0, Variable(0).id());
    /// assert_eq!(1, Variable(1).id());
    /// assert_eq!(2, Variable(2).id());
    /// assert_eq!(3, Variable(3).id());
    /// ```
    pub fn id(self) -> usize {
        self.0
    }
}

// ----------------------------------------------------------------------------
// --- DOMAIN -----------------------------------------------------------------
// ----------------------------------------------------------------------------
/// A domain is a set of values (isize) that may be assigned to some variable
/// by a decision.
///
/// # Important note
/// `Domain` implements the `From` trait for all of the usual collections.
/// Therefore it is considered much better and cleaner to use `.into()` rather
/// than manually specifying the variant of this enum.
///
/// # Example
/// ```
/// # use ddo::common::Domain;
/// // This is considered much cleaner the alternative below
/// let clean : Domain<'_> = vec![1, 2, 4].into();
/// // This is more verbose and less clean. Still the two implementations
/// // are totally equivalent.
/// let ugly  : Domain<'_> = Domain::Vector(vec![1, 2, 4]);
/// ```
///
/// # Technical note
/// This enum covers the most common collections from which a domain may be
/// built. These, however, should be considered transparent and only a way to
/// iterate over those values. The technical reason for having this enum is the
/// following: designing an api (trait) for the `Problem` that would not
/// constraint how one can model a problem required the ability to return some
/// polymorphic Domain. Ideally, this domain would have been an anonymous
/// iterable data structure (`IntoIter`). But this was not possible as traits
/// declaration forbid `impl trait` for return types. Hence, the only two
/// remaining options were to either return a _trait object_ (but that would've
/// incurred a perf-penalty because of heap allocation), or to use a polymorphic
/// algebraic data type. The second option was retained as it combines both
/// efficiency and elegance (+ it is a fairly common approach to polymorphism
/// in rust and c).
#[derive(Clone)]
pub enum Domain<'a> {
    /// When the domain consists of an owned vector (vec![])
    Vector(Vec<isize>),
    /// When the domain consists of a slice (array or ref to vector)
    Slice(&'a [isize]),
    /// When the domain is a compact bitset
    BitSet(&'a BitSet),
    /// When domain materialises a relation between variables (i.e. successor
    /// in a TSP), then the domain can be a varset. The possible values will be
    /// the ids of the variables present in the set.
    VarSet(&'a VarSet),
    /// When the domain is an exclusive range (ie. 0..10)
    Range(Range<isize>),
    /// When the domain is an inclusive range (ie. 0..=10)
    RangeInclusive (RangeInclusive<isize>)
}
/// As stated above, domains are considered only useful as a means to iterate
/// over the set of values that may be affected to some given variable. Thus,
/// `Domain` implements to `IntoIterator` trait which makes it suitable to use
/// in loops (and gives anyone the ability to derive an iterator for the values
/// of the domain).
impl <'a> IntoIterator for Domain<'a> {
    type Item     = isize;
    type IntoIter = DomainIter<'a>;

    /// Yields an iterator to go over the values of the domain.
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Domain::Vector         (v) => DomainIter::Vector(v.into_iter()),
            Domain::Slice          (s) => DomainIter::Slice (s.iter()),
            Domain::BitSet         (b) => DomainIter::BitSet(BitSetIter::new(b)),
            Domain::VarSet         (v) => DomainIter::BitSet(BitSetIter::new(&v.0)),
            Domain::Range          (r) => DomainIter::Range (r),
            Domain::RangeInclusive (r) => DomainIter::RangeInclusive(r)
        }
    }
}

/// `DomainIter` is the type of the iterator used to go over the possible values
/// of a domain. Therefore, it is isomorphic to the `Domain` type itself.
///
/// The implementation of the iterator is of very little interest in and of
/// itself. It should really only be considered as a means to iterate over the
/// values in the domain.
pub enum DomainIter<'a> {
    Vector         (std::vec::IntoIter<isize>),
    Slice          (std::slice::Iter<'a, isize>),
    BitSet         (BitSetIter<'a>),
    Range          (Range<isize>),
    RangeInclusive (RangeInclusive<isize>)
}
/// `DomainIter` is an iterator for the `Domain`. As such, it implements the
/// standard `Iterator` trait.
impl Iterator for DomainIter<'_> {
    type Item = isize;

    /// Yields the next value of the domain or None if the iteration already
    /// exhausted all the values.
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DomainIter::Vector         (i) => i.next(),
            DomainIter::Slice          (i) => i.next().copied(),
            DomainIter::BitSet         (i) => i.next().map(|x| x as isize),
            DomainIter::Range          (i) => i.next(),
            DomainIter::RangeInclusive (i) => i.next()
        }
    }
}

/// Implements the conversion from a vector of integers to a domain
impl From<Vec<isize>> for Domain<'_> {
    fn from(v: Vec<isize>) -> Self {
        Domain::Vector(v)
    }
}
/// Implements the conversion from a slice of integers to a domain
impl <'a> From<&'a [isize]> for Domain<'a> {
    fn from(s: &'a [isize]) -> Self {
        Domain::Slice(s)
    }
}
/// Implements the conversion from a range of integers to a domain
impl From<Range<isize>> for Domain<'_> {
    fn from(r: Range<isize>) -> Self {
        Domain::Range(r)
    }
}
/// Implements the conversion from an inclusive range of integers to a domain
impl From<RangeInclusive<isize>> for Domain<'_> {
    fn from(r: RangeInclusive<isize>) -> Self {
        Domain::RangeInclusive(r)
    }
}
/// Implements the conversion from a bitset of integers to a domain
impl <'a> From<&'a BitSet> for Domain<'a> {
    fn from(b: &'a BitSet) -> Self {
        Domain::BitSet(b)
    }
}
/// Implements the conversion from a varset to a domain
impl <'a> From<&'a VarSet> for Domain<'a> {
    fn from(b: &'a VarSet) -> Self {
        Domain::VarSet(b)
    }
}

// ----------------------------------------------------------------------------
// --- DECISION ---------------------------------------------------------------
// ----------------------------------------------------------------------------
/// This denotes a decision that was made during the search. It affects a given
/// `value` to the specified `variable`. Any given `Decision` should be
/// understood as ```[[ variable = value ]]````
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Decision {
    pub variable : Variable,
    pub value    : isize
}

// ----------------------------------------------------------------------------
// --- SOLUTION & PARTIAL ASSIGNMENT ------------------------------------------
// ----------------------------------------------------------------------------

/// A solution is a partial assignment that assigns a value to each of the
/// problem variables. From the MDD-perspective, it is an optimal r-t path
/// in the exact MDD.
///
/// # Example Use
/// A solution is a collection of decisions. It can be iterated upon as shown
/// below
/// ```
/// # use ddo::common::{Decision, Variable, Solution};
/// # use ddo::common::PartialAssignment::{FragmentExtension, Empty};
/// # use std::sync::Arc;
/// # let d1 = Decision {variable: Variable(1), value: 1};
/// # let d2 = Decision {variable: Variable(1), value: 1};
/// # let d3 = Decision {variable: Variable(1), value: 1};
/// # let pa = FragmentExtension {parent: Arc::new(Empty), fragment: vec![d1, d2, d3]};
/// # let solution = Solution::new(Arc::new(pa));
/// // Let us assume solution was obtained from a call to `maximize()` on
/// // some given solver...
///
/// // it can be iterated with an iterator
/// for decision in solution.iter() {
///     print!("Decision {:?}", decision);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Solution(Arc<PartialAssignment>);

impl Solution {
    /// Wraps a partial assignment into a complete solution
    pub fn new(assigment: Arc<PartialAssignment>) -> Solution {
        Solution(assigment)
    }
    /// Returns an iterator over the decisions to make
    pub fn iter(&self) -> PartialAssignmentIter<'_> {
        self.0.iter()
    }
}

/// A partial assignment is an incomplete solution that can be extended into a
/// complete one. In other words, it encapsulates a collection of decisions.
///
/// # Example
/// ```
/// # use ddo::common::{PartialAssignment, Decision, Variable};
/// # use ddo::common::PartialAssignment::{Empty, SingleExtension, FragmentExtension};
/// # use std::sync::Arc;
/// #
/// # let d1 = Decision{variable: Variable(0), value: 1};
/// # let d2 = Decision{variable: Variable(1), value: 2};
/// # let d3 = Decision{variable: Variable(2), value: 3};
/// # let d4 = Decision{variable: Variable(3), value: 4};
///
/// // An empty partial assignment has no decision
/// let pa = Empty;
/// assert_eq!(0, pa.iter().count());
/// assert_eq!(None, pa.iter().next());
///
/// // A single extension extends a given partial assignment (pa) with
/// // *one single* decision.
/// let pa = SingleExtension {parent: Arc::new(pa), decision: d1};
/// assert_eq!( 1, pa.iter().count());
/// assert_eq!(Some(d1), pa.iter().next());
///
/// // A fragment extension extends a given partial assignment (pa) with
/// // *one or more* decisions.
/// let pa = FragmentExtension {parent: Arc::new(pa), fragment: vec![d2, d3, d4]};
/// assert_eq!( 4, pa.iter().count());
/// // it begins with the fragment (in the order of the fragment)
/// assert_eq!(Some(d2), pa.iter().next());
/// // then it climbs up the chained structure
/// assert_eq!(Some(d1), pa.iter().last());
/// ```
///
/// # Technical Note
/// `PartialAssignment` was implemented as an enum (alebraic datatype) in order
/// to benefit from a clean and efficient polymorphism.
#[derive(Debug, Clone)]
pub enum PartialAssignment {
    /// An empty partial assignment (when no decision has been made)
    Empty,
    /// Extends a previous partial assignment (`parent`) with *one* additional
    /// decisions.
    SingleExtension {
        /// The new decision which is added to the pre-existing partial assignment.
        decision: Decision,
        /// The extended partial assignment
        parent  : Arc<PartialAssignment>
    },
    /// Extends a previous partial assignment (`parent`) with multiple additional
    /// decisions.
    FragmentExtension{
        /// The set of new decisions added to the pre-existing partial assignment.
        fragment: Vec<Decision>,
        /// The extended partial assignment
        parent  : Arc<PartialAssignment>
    }
}
impl PartialAssignment {
    /// This method returns an iterator over the decisions of the partial
    /// assignment. This way the decisions of a partial assignment can be
    /// iterated upon in a loop.
    ///
    /// # Example
    /// ```
    /// # use ddo::common::{PartialAssignment, Decision, Variable};
    /// # use ddo::common::PartialAssignment::{Empty, SingleExtension, FragmentExtension};
    /// # use std::sync::Arc;
    /// #
    /// # let d1 = Decision{variable: Variable(0), value: 1};
    /// # let d2 = Decision{variable: Variable(1), value: 2};
    /// # let d3 = Decision{variable: Variable(2), value: 3};
    /// # let d4 = Decision{variable: Variable(3), value: 4};
    /// let pa = FragmentExtension {parent: Arc::new(Empty), fragment: vec![d1, d2, d3, d4]};
    ///
    /// for d in pa.iter() {
    ///     print!("Decision {:?}", d);
    /// }
    /// ```
    pub fn iter(&self) -> PartialAssignmentIter {
        PartialAssignmentIter {chunk: self, cursor: None}
    }
}

/// This structure backs the iteration over the elements of a `PartialAssignment`.
/// This is what enables the use of partial assignments in for loops.
pub struct PartialAssignmentIter<'a> {
    /// The current partial assignment 'block'
    chunk: &'a PartialAssignment,
    /// The iteration cursor inside the current hunk
    cursor: Option<Cloned<Iter<'a, Decision>>>
}
impl Iterator for PartialAssignmentIter<'_> {
    type Item = Decision;

    /// Returns the next decision or None if all decisions have been iterated
    fn next(&mut self) -> Option<Decision> {
        match self.chunk {
            PartialAssignment::Empty => None,
            PartialAssignment::SingleExtension{decision, parent} => {
                let  ret   = *decision;
                self.chunk = parent.as_ref();
                Some(ret)
            },
            PartialAssignment::FragmentExtension{fragment, parent} => {
                if self.cursor.is_none() {
                    self.cursor = Some(fragment.iter().cloned());
                }
                let next = self.cursor.as_mut().unwrap().next();
                if next.is_none() {
                    self.chunk = parent.as_ref();
                    self.cursor = None;
                    self.next()
                } else {
                    next
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// --- FRONTIER NODE ----------------------------------------------------------
// ----------------------------------------------------------------------------
/// Frontier nodes describes the subproblems that have been enumerated and must
/// be explored in order to prove optimality. Hence, each single frontier node
/// denotes _one_ subproblem characterized by the same transition and transition
/// cost functions as the original problem, but having a different initial `state`
/// and value (`lp_len`). Because a frontier node is a *subproblem*, of the
/// original constraint optimization problem, it also remembers the partial
/// assignment (the `path`) which transforms the global problem into this
/// region of the state space.
///
/// The name 'frontier-node' was chosen to emphazise the idea that objects of
/// this type are the constituents of a _solver's frontier_ (aka the queue).
#[derive(Debug, Clone)]
pub struct FrontierNode<T> {
    /// This is the root state of the subproblem denoted by this frontier node
    pub state: Arc<T>,
    /// This is the path (in the exact mdd) that led to the definition of the
    /// current sub-problem.
    pub path: Arc<PartialAssignment>,
    /// This is the length of the longest (known) path from root to this node
    /// in the exact mdd.
    pub lp_len: isize,
    /// This is an overapproximation on the value of the objective function
    /// for this subproblem. Typically, it should contain the minimum of the
    /// rough upper bound (RUB) and the local bound.
    pub ub: isize,
}
impl <T> PartialEq for FrontierNode<T> where T: Eq {
    fn eq(&self, other: &Self) -> bool {
        self.state.eq(&other.state)
    }
}

// ----------------------------------------------------------------------------
// --- UTILITY TYPES ----------------------------------------------------------
// ----------------------------------------------------------------------------
/// This type denotes a set of variable. It encodes them compactly as a fixed
/// size bitset. A `VarSet` can be efficiently iterated upon.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct VarSet(pub BitSet);

/// This is the implementation of the core features of a `VarSet` (set of variables).
///
/// # Example
/// ```
/// # use ddo::common::{Variable, VarSet};
/// let vs = VarSet::all(3);
/// assert!(vs.contains(Variable(0)));
/// assert!(vs.contains(Variable(1)));
/// assert!(vs.contains(Variable(2)));
/// ```
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
/// Any two sets of variables can be compared with a lexicographical ordering
/// on the id of the variables present in the sets. Hence, `VarSet` implements
/// the standard traits `Ord` and `PartialOrd`.
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
///
/// # Example
/// ```
/// # use ddo::common::{Variable, VarSet};
/// let mut vs = VarSet::all(5);
/// vs.remove(Variable(3));
/// // iterates for variables 0, 1, 4
/// for v in vs.iter() {
///     println!("{:?}", v);
/// }
/// ```
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

/// This structure defines an iterator capable of iterating over the 1-bits of
/// a fixed bitset. It uses word representation of the items in the set, so it
/// should be more efficient to use than a crude iteration over the elements of
/// the set.
///
/// # Example
/// ```
/// # use bitset_fixed::BitSet;
/// # use ddo::common::BitSetIter;
///
/// let mut bit_set = BitSet::new(5);
/// bit_set.set(1, true);
/// bit_set.set(2, true);
/// bit_set.set(4, true);
///
/// // Successively prints 1, 2, 4
/// for x in BitSetIter::new(&bit_set) {
///     println!("{}", x);
/// }
/// ```
///
pub struct BitSetIter<'a> {
    /// An iterator over the buffer of words of the bitset
    iter: Cloned<Iter<'a, u64>>,
    /// The current word (or none if we exhausted all iterations)
    word: Option<u64>,
    /// The value of position 0 in the current word
    base: usize,
    /// An offset in the current word
    offset: usize,
}
impl BitSetIter<'_> {
    /// This method creates an iterator for the given bitset from an immutable
    /// reference to that bitset.
    pub fn new(bs: &BitSet) -> BitSetIter {
        let mut iter = bs.buffer().iter().cloned();
        let word = iter.next();
        BitSetIter {iter, word, base: 0, offset: 0}
    }
}
/// `BitSetIter` is an iterator over the one bits of the bitset. As such, it
/// implements the standard `Iterator` trait.
impl Iterator for BitSetIter<'_> {
    type Item = usize;

    /// Returns the nex element from the iteration, or None, if there are no more
    /// elements to iterate upon.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(w) = self.word {
            if w == 0 || self.offset >= 64 {
                self.word   = self.iter.next();
                self.base  += 64;
                self.offset = 0;
            } else {
                let mut mask = 1_u64 << self.offset as u64;
                while (w & mask) == 0 && self.offset < 64 {
                    mask <<= 1;
                    self.offset += 1;
                }
                if self.offset < 64 {
                    let ret = Some(self.base + self.offset);
                    self.offset += 1;
                    return ret;
                }
            }
        }
        None
    }
}

/// A totally ordered Bitset wrapper. Useful to implement tie break mechanisms.
/// This wrapper orders the bitsets according to the lexical order of their
/// underlying bits.
///
/// # Note:
/// This implementation uses the underlying _words_ representation of the
/// bitsets to perform several comparisons at once. Hence, using a `LexBitSet`
/// should be more efficient than trying to establish the total ordering
/// yourself with a loop on the 1-bits of the two sets.
///
/// # Example
/// ```
/// # use bitset_fixed::BitSet;
/// # use ddo::common::LexBitSet;
///
/// let mut a = BitSet::new(5);
/// let mut b = BitSet::new(5);
///
/// a.set(2, true);  // bits 0..2 match for a and b
/// b.set(2, true);
///
/// a.set(3, false); // a and b diverge on bit 3
/// b.set(3, true);  // and a has a 0 bit in that pos
///
/// a.set(4, true);  // anything that remains after
/// b.set(4, false); // the firs lexicographical difference is ignored
///
/// assert!(LexBitSet(&a) < LexBitSet(&b));
/// ```
///
#[derive(Debug)]
pub struct LexBitSet<'a>(pub &'a BitSet);

/// The `LexBitSet` implements a total order on bitsets. As such, it must
/// implement the standard trait `Ord`.
///
/// # Note:
/// This implementation uses the underlying _words_ representation of the
/// bitsets to perform several comparisons at once. Hence, using a `LexBitSet`
/// should be more efficient than trying to establish the total ordering
/// yourself with a loop on the 1-bits of the two sets.
impl Ord for LexBitSet<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut x = self.0.buffer().iter().cloned();
        let mut y = other.0.buffer().iter().cloned();
        let end   = x.len().max(y.len());

        for _ in 0..end {
            let xi = x.next().unwrap_or(0);
            let yi = y.next().unwrap_or(0);
            if xi != yi {
                let mut mask = 1_u64;
                for _ in 0..64 {
                    let bit_x = xi & mask;
                    let bit_y = yi & mask;
                    if bit_x != bit_y {
                        return bit_x.cmp(&bit_y);
                    }
                    mask <<= 1;
                }
            }
        }
        Equal
    }
}

/// Because it is a total order, `LexBitSet` must also be a partial order.
/// Hence, it must implement the standard trait `PartialOrd`.
impl PartialOrd for LexBitSet<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Because `LexBitSet` defines a total order, it makes sense to consider that
/// it also defines an equivalence relation. As such, it implements the standard
/// `Eq` and `PartialEq` traits.
impl Eq for LexBitSet<'_> {}

/// Having `LexBitSet` to implement `PartialEq` means that it _at least_ defines
/// a partial equivalence relation.
impl PartialEq for LexBitSet<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || self.cmp(other) == Equal
    }
}

/// This structure implements a 2D matrix of size [ n X m ].
///
///
/// # Example
/// ```
/// # use ddo::common::Matrix;
///
/// let mut adjacency = Matrix::new_default(5, 5, None);
///
/// adjacency[(2, 2)] = Some(-5);
/// assert_eq!(Some(-5), adjacency[(2, 2)]);
/// ```
#[derive(Clone)]
pub struct Matrix<T> {
    /// The number of rows
    pub n: usize,
    /// The number of columns
    pub m: usize,
    /// The items of the matrix
    pub data : Vec<T>
}
impl <T : Default + Clone> Matrix<T> {
    /// Allows the creation of a matrix initialized with the default element
    pub fn new(m: usize, n: usize) -> Self {
        Matrix { m, n, data: vec![Default::default(); m * n] }
    }
}
impl <T : Clone> Matrix<T> {
    /// Allows the creation of a matrix initialized with the default element
    pub fn new_default(m: usize, n: usize, item: T) -> Self {
        Matrix { m, n, data: vec![item; m * n] }
    }
}
impl <T> Matrix<T> {
    /// Returns the position (offset in the data) of the given index
    fn pos(&self, idx: (usize, usize)) -> usize {
        self.m * idx.0 + idx.1
    }
}
/// A matrix is typically an item you'll want to adress using 2D position
impl <T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    /// It returns a reference to some item from the matrix at the given 2D index
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        let position = self.pos(idx);
        &self.data[position]
    }
}
impl <T> IndexMut<(usize, usize)> for Matrix<T> {
    /// It returns a mutable reference to some item from the matrix at the given 2D index
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        let position = self.pos(idx);
        &mut self.data[position]
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
mod test_var {
    use crate::common::Variable;

    #[test]
    fn test_var_id() {
        assert_eq!(0, Variable(0).id());
        assert_eq!(1, Variable(1).id());
        assert_eq!(2, Variable(2).id());
        assert_eq!(3, Variable(3).id());
    }
}


#[cfg(test)]
mod test_domain {
    use bitset_fixed::BitSet;
    use crate::common::{Domain, VarSet, Variable};

    #[test]
    fn from_vector_empty() {
        let domain : Domain<'_> = vec![].into();
        assert_eq!(Vec::<isize>::new(), domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_vector_non_empty() {
        let domain : Domain<'_> = vec![1, 2, 4].into();
        assert_eq!(vec![1, 2, 4], domain.into_iter().collect::<Vec<isize>>());
    }

    #[test]
    fn from_slice_empty() {
        let data = vec![];
        let slice  : &[isize] = &data;
        let domain : Domain<'_> = slice.into();
        assert_eq!(Vec::<isize>::new(), domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_slice_non_empty() {
        let data = [1, 2, 4];
        let slice  : &[isize] = &data;
        let domain : Domain<'_> = slice.into();
        assert_eq!(vec![1, 2, 4], domain.into_iter().collect::<Vec<isize>>());
    }

    #[test]
    fn from_bitset_empty() {
        let data = BitSet::new(5);
        let domain : Domain<'_> = (&data).into();
        assert_eq!(Vec::<isize>::new(), domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_bitset_non_empty() {
        let mut data = BitSet::new(5);
        data.set(2, true);
        data.set(3, true);
        let domain : Domain<'_> = (&data).into();
        assert_eq!(vec![2, 3], domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_varset_empty() {
        let data = VarSet::empty();
        let domain : Domain<'_> = (&data).into();
        assert_eq!(Vec::<isize>::new(), domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_varset_non_empty() {
        let mut data = VarSet::all(5);
        data.remove(Variable(0));
        data.remove(Variable(1));
        data.remove(Variable(4));
        let domain : Domain<'_> = (&data).into();
        assert_eq!(vec![2, 3], domain.into_iter().collect::<Vec<isize>>());
    }

    #[test]
    fn from_range_empty_going_negative() {
        let data = 0..-1_isize;
        let domain : Domain<'_> = data.into();
        assert_eq!(Vec::<isize>::new(), domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_range_empty_positive() {
        let data = 0..0_isize;
        let domain : Domain<'_> = data.into();
        assert_eq!(Vec::<isize>::new(), domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_range_non_empty() {
        let data = 0..5_isize;
        let domain : Domain<'_> = data.into();
        assert_eq!(vec![0,1,2,3,4], domain.into_iter().collect::<Vec<isize>>());
    }

    #[test]
    fn from_range_inclusive_empty_going_negative() {
        let data = 0..=-1_isize;
        let domain : Domain<'_> = data.into();
        assert_eq!(Vec::<isize>::new(), domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_range_inclusive_single_value() {
        let data = 0..=0_isize;
        let domain : Domain<'_> = data.into();
        assert_eq!(vec![0], domain.into_iter().collect::<Vec<isize>>());
    }
    #[test]
    fn from_range_inclusive_non_empty() {
        let data = 0..=5_isize;
        let domain : Domain<'_> = data.into();
        assert_eq!(vec![0,1,2,3,4,5], domain.into_iter().collect::<Vec<isize>>());
    }
}

#[cfg(test)]
mod test_partial_assignment {
    use crate::common::PartialAssignment::{Empty, SingleExtension, FragmentExtension};
    use crate::common::{Decision, Variable};
    use std::sync::Arc;

    #[test]
    fn empty_pa_has_no_decision() {
        let pa = Empty;
        assert_eq!(0, pa.iter().count());
    }

    #[test]
    fn single_extension_adds_one_decision() {
        let d1 = Decision{variable: Variable(0), value: 1};
        let d2 = Decision{variable: Variable(1), value: 2};

        // when parent pa is empty
        let pa = Empty;
        let pa = SingleExtension {parent: Arc::new(pa), decision: d1};
        assert_eq!(1, pa.iter().count());

        // when parent pa is not empty
        let pa = SingleExtension {parent: Arc::new(pa), decision: d2};
        assert_eq!(2, pa.iter().count());
    }

    #[test]
    fn fragment_extension_adds_one_or_more_decisions() {
        let d1 = Decision{variable: Variable(0), value: 1};
        let d2 = Decision{variable: Variable(1), value: 2};
        let d3 = Decision{variable: Variable(2), value: 3};
        let d4 = Decision{variable: Variable(3), value: 4};

        // when parent pa is empty
        let pa = Empty;
        let pa = FragmentExtension {parent: Arc::new(pa), fragment: vec![d1, d2]};
        assert_eq!(2, pa.iter().count());

        // when parent pa is not empty
        let pa = FragmentExtension {parent: Arc::new(pa), fragment: vec![d3, d4]};
        assert_eq!(4, pa.iter().count());
    }

    #[test]
    fn iterating_over_an_empty_partial_assignment_yields_no_decision() {
        let pa = Empty;
        let mut it = pa.iter();
        assert_eq!(None, it.next());
    }

    #[test]
    fn iterating_a_single_extension_starts_with_the_extended_info_then_climbs_up() {
        let d1 = Decision{variable: Variable(0), value: 1};
        let d2 = Decision{variable: Variable(1), value: 2};

        // when parent pa is empty
        let pa = Empty;
        let pa = SingleExtension {parent: Arc::new(pa), decision: d1};
        let mut it = pa.iter();
        assert_eq!(Some(d1), it.next());
        assert_eq!(None,     it.next());

        // when parent pa is not empty
        let pa = SingleExtension {parent: Arc::new(pa), decision: d2};
        let mut it = pa.iter();
        assert_eq!(Some(d2), it.next());
        assert_eq!(Some(d1), it.next());
        assert_eq!(None,     it.next());
    }

    #[test]
    fn iterating_a_fragment_extension_starts_with_the_extended_info_then_climbs_up() {
        let d1 = Decision{variable: Variable(0), value: 1};
        let d2 = Decision{variable: Variable(1), value: 2};
        let d3 = Decision{variable: Variable(2), value: 3};
        let d4 = Decision{variable: Variable(3), value: 4};

        // when parent pa is empty
        let pa = Empty;
        let pa = FragmentExtension {parent: Arc::new(pa), fragment: vec![d1, d2]};
        let mut it = pa.iter();
        assert_eq!(Some(d1), it.next());
        assert_eq!(Some(d2), it.next());
        assert_eq!(None,     it.next());

        // when parent pa is not empty
        let pa = FragmentExtension {parent: Arc::new(pa), fragment: vec![d3, d4]};
        let mut it = pa.iter();
        assert_eq!(Some(d3), it.next());
        assert_eq!(Some(d4), it.next());
        assert_eq!(Some(d1), it.next());
        assert_eq!(Some(d2), it.next());
        assert_eq!(None,     it.next());
    }
}

#[cfg(test)]
mod test_solution {
    use crate::common::{Decision, Variable, Solution};
    use crate::common::PartialAssignment::{Empty, FragmentExtension};
    use std::sync::Arc;

    #[test]
    pub fn a_solution_can_iterate_over_the_decisions_of_some_partial_assignment() {
        let d1 = Decision{variable: Variable(0), value: 1};
        let d2 = Decision{variable: Variable(1), value: 2};
        let d3 = Decision{variable: Variable(2), value: 3};
        let d4 = Decision{variable: Variable(3), value: 4};

        let pa = Empty;
        let pa = FragmentExtension {parent: Arc::new(pa), fragment: vec![d1, d2]};
        let pa = FragmentExtension {parent: Arc::new(pa), fragment: vec![d3, d4]};

        let sln= Solution::new(Arc::new(pa));
        let mut it = sln.iter();
        assert_eq!(Some(d3), it.next());
        assert_eq!(Some(d4), it.next());
        assert_eq!(Some(d1), it.next());
        assert_eq!(Some(d2), it.next());
        assert_eq!(None,     it.next());
    }
}

#[cfg(test)]
mod test_varset {
    use crate::common::{Variable, VarSet};

    #[test]
    fn all_contains_all_variables() {
        let vs = VarSet::all(3);

        assert_eq!(3, vs.len());
        assert!(vs.contains(Variable(0)));
        assert!(vs.contains(Variable(1)));
        assert!(vs.contains(Variable(2)));
    }
    #[test] #[should_panic]
    fn contains_panic_when_over_the_max_number_of_vars() {
        let vs = VarSet::all(3);
        assert!(!vs.contains(Variable(20)));
    }
    #[test]
    fn empty_contains_no_variable() {
        let vs = VarSet::empty();
        assert_eq!(0, vs.len());
    }
    #[test]
    fn add_adds_the_variable() {
        let mut vs = VarSet::all(3);
        vs.remove(Variable(0));
        vs.remove(Variable(1));
        vs.remove(Variable(2));

        assert!(!vs.contains(Variable(2)));
        vs.add(Variable(2));
        assert!(vs.contains(Variable(2)));
    }
    #[test]
    fn add_has_no_effect_when_var_is_already_present() {
        let mut vs = VarSet::all(3);
        assert!(vs.contains(Variable(2)));

        vs.add(Variable(2));
        assert!(vs.contains(Variable(2)));
    }
    #[test]
    fn remove_drops_the_variable() {
        let mut vs = VarSet::all(3);
        assert!(vs.contains(Variable(0)));
        assert!(vs.contains(Variable(1)));
        assert!(vs.contains(Variable(2)));

        vs.remove(Variable(1));
        assert!(vs.contains(Variable(0)));
        assert!(!vs.contains(Variable(1)));
        assert!(vs.contains(Variable(2)));
    }
    #[test]
    fn remove_has_no_effect_when_variable_already_absent() {
        let mut vs = VarSet::all(3);
        vs.remove(Variable(0));
        vs.remove(Variable(1));
        vs.remove(Variable(2));

        assert!(!vs.contains(Variable(0)));
        assert!(!vs.contains(Variable(1)));
        assert!(!vs.contains(Variable(2)));

        vs.remove(Variable(1));
        assert!(!vs.contains(Variable(0)));
        assert!(!vs.contains(Variable(1)));
        assert!(!vs.contains(Variable(2)));
    }

    #[test]
    fn len_indicates_the_size_of_the_set() {
        let mut vs = VarSet::all(3);
        assert_eq!(3, vs.len());

        vs.remove(Variable(0));
        assert_eq!(2, vs.len());
        vs.remove(Variable(1));
        assert_eq!(1, vs.len());
        vs.remove(Variable(2));
        assert_eq!(0, vs.len());

        vs.add(Variable(0));
        assert_eq!(1, vs.len());
        vs.add(Variable(1));
        assert_eq!(2, vs.len());
        vs.add(Variable(2));
        assert_eq!(3, vs.len());
    }
    #[test]
    fn is_empty_means_len_zero() {
        let mut vs = VarSet::all(3);
        assert!(!vs.is_empty());

        vs.remove(Variable(0));
        assert!(!vs.is_empty());
        vs.remove(Variable(1));
        assert!(!vs.is_empty());
        vs.remove(Variable(2));
        assert!(vs.is_empty());

        vs.add(Variable(0));
        assert!(!vs.is_empty());
        vs.add(Variable(1));
        assert!(!vs.is_empty());
        vs.add(Variable(2));
        assert!(!vs.is_empty());
    }

    #[test]
    fn iter_lets_you_iterate_over_all_variables_in_the_set() {
        let mut vs = VarSet::all(3);
        assert_eq!(vs.iter().collect::<Vec<Variable>>(),
                   vec![Variable(0), Variable(1), Variable(2)]);

        vs.remove(Variable(1));
        assert_eq!(vs.iter().collect::<Vec<Variable>>(),
                   vec![Variable(0), Variable(2)]);

        vs.remove(Variable(2));
        assert_eq!(vs.iter().collect::<Vec<Variable>>(),
                   vec![Variable(0)]);

        vs.remove(Variable(0));
        assert_eq!(vs.iter().collect::<Vec<Variable>>(),
                   vec![]);

        vs.add(Variable(1));
        assert_eq!(vs.iter().collect::<Vec<Variable>>(),
                   vec![Variable(1)]);
    }

    #[test]
    fn varset_are_lex_ordered() {
        let mut a = VarSet::all(3);
        let mut b = VarSet::all(3);

        a.remove(Variable(2));
        assert!(a < b);

        b.remove(Variable(0));
        assert!(b < a);

        a.remove(Variable(1));
        assert!(b < a);

        a.remove(Variable(0));
        assert!(a < b);

        b.remove(Variable(1));
        assert!(a < b);

        b.remove(Variable(2));
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod test_varset_iter {
    use crate::common::{VarSet, VarSetIter, Variable, BitSetIter};

    #[test]
    fn vsiter_collect() {
        let vs    = VarSet::all(3);
        let iter  = VarSetIter(BitSetIter::new(&vs.0));
        let items = iter.collect::<Vec<Variable>>();

        assert_eq!(items, vec![Variable(0), Variable(1), Variable(2)]);
    }
    #[test]
    fn vsiter_next_normal_case() {
        let vs    = VarSet::all(3);
        let mut iter  = VarSetIter(BitSetIter::new(&vs.0));

        assert_eq!(Some(Variable(0)), iter.next());
        assert_eq!(Some(Variable(1)), iter.next());
        assert_eq!(Some(Variable(2)), iter.next());
        assert_eq!(None             , iter.next());
    }
    #[test]
    fn vsiter_no_items() {
        let vs        = VarSet::empty();
        let mut iter  = VarSetIter(BitSetIter::new(&vs.0));

        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next());
    }
    #[test]
    fn vsiter_mutiple_words() {
        let mut vs    = VarSet::all(128);
        for i in 0..128 {
            vs.remove(Variable(i));
        }
        vs.add(Variable(  1));
        vs.add(Variable( 50));
        vs.add(Variable( 66));
        vs.add(Variable(100));

        let mut iter  = VarSetIter(BitSetIter::new(&vs.0));

        assert_eq!(Some(Variable(  1)), iter.next());
        assert_eq!(Some(Variable( 50)), iter.next());
        assert_eq!(Some(Variable( 66)), iter.next());
        assert_eq!(Some(Variable(100)), iter.next());
        assert_eq!(None               , iter.next());
    }
}


#[cfg(test)]
/// These tests validate the behavior of the bitset iterator `BitSetIter`.
mod tests_bitset_iter {
    use bitset_fixed::BitSet;
    use crate::common::BitSetIter;

    #[test]
    fn bsiter_collect() {
        let mut bit_set = BitSet::new(5);
        bit_set.set(1, true);
        bit_set.set(2, true);
        bit_set.set(4, true);

        let iter  = BitSetIter::new(&bit_set);
        let items = iter.collect::<Vec<usize>>();

        assert_eq!(items, vec![1, 2, 4]);
    }
    #[test]
    fn bsiter_next_normal_case() {
        let mut bit_set = BitSet::new(5);
        bit_set.set(1, true);
        bit_set.set(2, true);
        bit_set.set(4, true);

        let mut iter = BitSetIter::new(&bit_set);
        assert_eq!(Some(1), iter.next());
        assert_eq!(Some(2), iter.next());
        assert_eq!(Some(4), iter.next());
        assert_eq!(None   , iter.next());
    }
    #[test]
    fn bsiter_no_items() {
        let bit_set = BitSet::new(5);
        let mut iter    = BitSetIter::new(&bit_set);

        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next());
    }
    #[test]
    fn bsiter_mutiple_words() {
        let mut bit_set = BitSet::new(128);
        bit_set.set(  1, true);
        bit_set.set( 50, true);
        bit_set.set( 66, true);
        bit_set.set(100, true);

        let mut iter = BitSetIter::new(&bit_set);
        assert_eq!(Some(  1), iter.next());
        assert_eq!(Some( 50), iter.next());
        assert_eq!(Some( 66), iter.next());
        assert_eq!(Some(100), iter.next());
        assert_eq!(None     , iter.next());
    }
}
#[cfg(test)]
/// These tests validate the behavior of the lexicographically ordered bitsets
/// `LexBitSet`.
mod tests_lexbitset {
    use bitset_fixed::BitSet;
    use crate::common::LexBitSet;

    #[test]
    fn same_size_less_than() {
        let mut a = BitSet::new(200);
        let mut b = BitSet::new(200);

        a.set(2, true);  // bits 0..2 match for a and b
        b.set(2, true);

        a.set(3, false); // a and b diverge on bit 3
        b.set(3, true);  // and a has a 0 bit in that pos

        a.set(4, true);  // anything that remains after
        b.set(4, false); // the firs lexicographical difference is ignored

        a.set(150, true);
        b.set(150, true);

        assert!(LexBitSet(&a) <= LexBitSet(&b));
        assert!(LexBitSet(&a) <  LexBitSet(&b));
    }
    #[test]
    fn same_size_greater_than() {
        let mut a = BitSet::new(200);
        let mut b = BitSet::new(200);

        a.set(2, true);  // bits 0..2 match for a and b
        b.set(2, true);

        a.set(3, false); // a and b diverge on bit 3
        b.set(3, true);  // and a has a 0 bit in that pos

        a.set(4, true);  // anything that remains after
        b.set(4, false); // the firs lexicographical difference is ignored

        a.set(150, true);
        b.set(150, true);

        assert!(LexBitSet(&b) >= LexBitSet(&a));
        assert!(LexBitSet(&b) >  LexBitSet(&a));
    }
    #[test]
    fn same_size_equal() {
        let mut a = BitSet::new(200);
        let mut b = BitSet::new(200);

        a.set(2, true);  // bits 0..2 match for a and b
        b.set(2, true);

        a.set(150, true);
        b.set(150, true);

        assert!(LexBitSet(&a) >= LexBitSet(&b));
        assert!(LexBitSet(&b) >= LexBitSet(&a));

        assert_eq!(LexBitSet(&a), LexBitSet(&b));
        assert_eq!(LexBitSet(&a), LexBitSet(&a));
        assert_eq!(LexBitSet(&b), LexBitSet(&b));
    }

    #[test]
    /// For different sized bitsets, it behaves as though they were padded with
    /// trailing zeroes.
    fn different_sizes_considered_padded_with_zeroes() {
        let mut a = BitSet::new(20);
        let mut b = BitSet::new(200);

        a.set(2, true);  // bits 0..2 match for a and b
        b.set(2, true);

        assert_eq!(LexBitSet(&a), LexBitSet(&b));

        b.set(150, true);
        assert!(LexBitSet(&a) <= LexBitSet(&b));
        assert!(LexBitSet(&a) <  LexBitSet(&b));
    }
}

#[cfg(test)]
mod test_matrix {
    use crate::common::Matrix;
    #[test]
    fn it_is_initialized_with_default_elem() {
        let mat : Matrix<Option<usize>> = Matrix::new(5, 5);

        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(None, mat[(i, j)]);
            }
        }
    }
    #[test]
    fn it_is_initialized_with_given_elem() {
        let mat = Matrix::new_default(5, 5, Some(0));

        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(Some(0), mat[(i, j)]);
            }
        }
    }
    #[test]
    fn it_can_be_accessed_and_mutated_with_2d_position() {
        let mut mat = Matrix::new(5, 5);
        mat[(2, 2)] = Some(-5);

        assert_eq!(Some(-5), mat[(2, 2)]);
    }
}

#[cfg(test)]
mod test_frontier_node {
    use crate::common::{FrontierNode, Decision, Variable};
    use std::sync::Arc;
    use crate::common::PartialAssignment::{SingleExtension, Empty};

    #[test]
    fn frontier_nodes_are_equal_iff_they_have_equal_state_everything_equal() {
        let a = FrontierNode {
            state: Arc::new(0),
            path: Arc::new(SingleExtension {
                decision: Decision { variable: Variable(0), value: 1 },
                parent: Arc::new(Empty)
            }),
            ub: 0,
            lp_len: 0
        };

        // everything is the same
        let b = FrontierNode {
            state: Arc::new(0),
            path: Arc::new(SingleExtension {
                decision: Decision { variable: Variable(0), value: 1 },
                parent: Arc::new(Empty)
            }),
            ub: 0,
            lp_len: 0
        };
        assert_eq!(a, b);
    }
    #[test]
    fn frontier_nodes_are_equal_iff_they_have_equal_state_different_path() {
        let a = FrontierNode{
            state: Arc::new(0),
            path : Arc::new(SingleExtension {
                decision: Decision{variable: Variable(0), value: 1},
                parent: Arc::new(Empty)
            }),
            ub   : 0,
            lp_len: 0
        };
        // everything is the same
        let b = FrontierNode {
            state: Arc::new(0),
            path: Arc::new(Empty),
            ub: 0,
            lp_len: 0
        };
        assert_eq!(a, b);
    }
    #[test]
    fn frontier_nodes_are_equal_iff_they_have_equal_state_different_value() {
        let a = FrontierNode{
            state: Arc::new(0),
            path : Arc::new(Empty),
            ub   : 0,
            lp_len: 0
        };
        // everything is the same
        let b = FrontierNode {
            state: Arc::new(0),
            path: Arc::new(Empty),
            ub: 0,
            lp_len: 10
        };
        assert_eq!(a, b);
    }
    #[test]
    fn frontier_nodes_are_equal_iff_they_have_equal_state_different_ub() {
        let a = FrontierNode{
            state: Arc::new(0),
            path : Arc::new(Empty),
            ub   : 0,
            lp_len: 10
        };
        // everything is the same
        let b = FrontierNode {
            state: Arc::new(0),
            path: Arc::new(Empty),
            ub: 50,
            lp_len: 10
        };
        assert_eq!(a, b);
    }
    #[test]
    fn frontier_nodes_are_equal_iff_they_have_equal_state_not_equal() {
        let a = FrontierNode{
            state: Arc::new(10),
            path : Arc::new(Empty),
            ub   : 0,
            lp_len: 0
        };
        // everything is the same
        let b = FrontierNode{
            state: Arc::new(20),
            path : Arc::new(Empty),
            ub   : 0,
            lp_len: 0
        };
        assert_ne!(a, b);
    }
}