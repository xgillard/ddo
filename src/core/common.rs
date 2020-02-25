//! This module defines the most basic data types that are used both in the
//! abstraction and implementation layers of our library. These are also the
//! types your client library is likely to work with.

use bitset_fixed::BitSet;
use crate::core::utils::{BitSetIter, LexBitSet};
use std::ops::{Not, Range, RangeInclusive};
use std::cmp::Ordering;
use std::hash::{Hasher, Hash};
use std::sync::Arc;

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
    /// # use ddo::core::common::Variable;
    /// assert_eq!(0, Variable(0).id());
    /// assert_eq!(1, Variable(1).id());
    /// assert_eq!(2, Variable(2).id());
    /// assert_eq!(3, Variable(3).id());
    /// ```
    pub fn id(self) -> usize {
        self.0
    }
}

/// This denotes a decision that was made during the search. It affects a given
/// `value` to the specified `variable`. Any given `Decision` should be
/// understood as ```[[ variable = value ]]````
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Decision {
    pub variable : Variable,
    pub value    : i32
}

/// This type denotes a set of variable. It encodes them compactly as a fixed
/// size bitset. A `VarSet` can be efficiently iterated upon.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct VarSet(pub BitSet);

/// This is the implementation of the core features of a `VarSet` (set of variables).
///
/// # Example
/// ```
/// # use ddo::core::common::{Variable, VarSet};
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
/// # use ddo::core::common::{Variable, VarSet};
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

/// Utility structure to represent upper and lower bounds
#[derive(Debug, Copy, Clone)]
pub struct Bounds {pub lb: i32, pub ub: i32}

/// A domain is a set of values (i32) that may be assigned to some variable
/// by a decision.
///
/// # Important note
/// `Domain` implements the `From` trait for all of the usual collections.
/// Therefore it is considered much better and cleaner to use `.into()` rather
/// than manually specifying the variant of this enum.
///
/// # Example
/// ```
/// # use ddo::core::common::Domain;
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
    Vector(Vec<i32>),
    /// When the domain consists of a slice (array or ref to vector)
    Slice(&'a [i32]),
    /// When the domain is a compact bitset
    BitSet(&'a BitSet),
    /// When domain materialises a relation between variables (i.e. successor
    /// in a TSP), then the domain can be a varset. The possible values will be
    /// the ids of the variables present in the set.
    VarSet(&'a VarSet),
    /// When the domain is an exclusive range (ie. 0..10)
    Range(Range<i32>),
    /// When the domain is an inclusive range (ie. 0..=10)
    RangeInclusive (RangeInclusive<i32>)
}
/// As stated above, domains are considered only useful as a means to iterate
/// over the set of values that may be affected to some given variable. Thus,
/// `Domain` implements to `IntoIterator` trait which makes it suitable to use
/// in loops (and gives anyone the ability to derive an iterator for the values
/// of the domain).
impl <'a> IntoIterator for Domain<'a> {
    type Item     = i32;
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
    Vector         (std::vec::IntoIter<i32>),
    Slice          (std::slice::Iter<'a, i32>),
    BitSet         (BitSetIter<'a>),
    Range          (Range<i32>),
    RangeInclusive (RangeInclusive<i32>)
}
/// `DomainIter` is an iterator for the `Domain`. As such, it implements the
/// standard `Iterator` trait.
impl Iterator for DomainIter<'_> {
    type Item = i32;
    /// Yields the next value of the domain or None if the iteration already
    /// exhausted all the values.
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DomainIter::Vector         (i) => i.next(),
            DomainIter::Slice          (i) => i.next().copied(),
            DomainIter::BitSet         (i) => i.next().map(|x| x as i32),
            DomainIter::Range          (i) => i.next(),
            DomainIter::RangeInclusive (i) => i.next()
        }
    }
}
/// Implements the conversion from a vector of integers to a domain
impl From<Vec<i32>> for Domain<'_> {
    fn from(v: Vec<i32>) -> Self {
        Domain::Vector(v)
    }
}
/// Implements the conversion from a slice of integers to a domain
impl <'a> From<&'a [i32]> for Domain<'a> {
    fn from(s: &'a [i32]) -> Self {
        Domain::Slice(s)
    }
}
/// Implements the conversion from a range of integers to a domain
impl From<Range<i32>> for Domain<'_> {
    fn from(r: Range<i32>) -> Self {
        Domain::Range(r)
    }
}
/// Implements the conversion from an inclusive range of integers to a domain
impl From<RangeInclusive<i32>> for Domain<'_> {
    fn from(r: RangeInclusive<i32>) -> Self {
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

// --- NODE --------------------------------------------------------------------
/// The partial assignments (derived from longest paths in the MDD) form a
/// chained structure. This structure is made of `NodeInfo` inter-connected by
/// `Edges`. An `Edge` materialises the labelled interconnection between two
/// nodes (`NodeInfos`) of an MDD.
///
/// Because the chained structure is reference counted, the `NodeInfo` + `Edge`
/// representation is very efficient as it allows to pune the tree-shaped structure
/// and forget about paths that are no longer relevant.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Edge  {
    /// Because an edge is owned by an endpoint (a `NodeInfo`), it only knows
    /// a pointer to its 'parent' node (`src` *is* that pointer to parent).
    /// The `src` pointer is a smart, reference counted pointer that may be used
    /// in multi-threaded contexts (uses atomic counters).
    pub src: Arc<NodeInfo>,
    /// `decision` is the field which materializes the decision taken in the
    /// parent node, which yielded the current node.
    ///
    /// In other words, taking `decision` in the node owner of `src` should
    /// bring you to the node owning this edge.
    pub decision: Decision
}

/// The partial assignments (derived from longest paths in the MDD) form a
/// chained structure. This structure is made of `NodeInfo` inter-connected by
/// `Edges`. An `NodeInfo` keeps track of the metadata associated to a node.
/// Namely, it is the `NodeInfo` which is capable of telling if the node is
/// exact, the length of the longest path to it, etc..
///
/// Because the chained structure is reference counted, the `NodeInfo` + `Edge`
/// representation is very efficient as it allows to pune the tree-shaped structure
/// and forget about paths that are no longer relevant.
#[derive(Debug, Clone, Eq)]
pub struct NodeInfo {
    /// This flag is true iff the node owning this node info is exact
    pub is_exact: bool,
    /// The _length_ of the longest path from the root node of the problem
    /// to the node owner of this `NodeInfo`.
    pub lp_len: i32,
    /// The edge to the parent of this node along the longest path from root to
    /// here. The only case where this field should be None is at the root node
    /// of the problem. All other nodes should know their "best parent" and have
    /// a way to access it through here.
    pub lp_arc: Option<Edge>,
    /// An upper bound on the best objective value attainable going through
    /// this node. (It is not the role of a nodeinfo to compute this value.
    /// The upper bound information is computed by the problem relaxation).
    pub ub: i32
}
// public methods of a `NodeInfo` which do not belong to any trait.
impl NodeInfo {
    /// Constructor.
    ///
    /// Creates a new node info from the longest path information
    /// (`lp_len` and `lp_arc`) along with a flag indicating if this denotes an
    /// exaxct node or not.  The upper bound is set to a default value
    /// (the greatest i32)
    pub fn new (lp_len: i32, lp_arc: Option<Edge>, is_exact: bool) -> NodeInfo {
        NodeInfo { is_exact, lp_len, lp_arc, ub: i32::max_value() }
    }

    /// Merge other into this node. That is to say, it combines the information
    /// from two nodes that are logically equivalent (should be the same).
    /// Concretely, it means that it possibly updates the current node to keep
    /// the best _longest path info_, track the 'exactitude' of the node and
    /// keep the tightest upper bound.
    ///
    /// # Important note
    /// *This has nothing to do with the user-provided `merge_*` operators !*
    pub fn merge(&mut self, other: Self) {
        if  self.lp_len < other.lp_len {
            self.lp_len = other.lp_len;
            self.lp_arc = other.lp_arc;
        }
        self.ub = self.ub.min(other.ub);
        self.is_exact &= other.is_exact;
    }

    /// Returns the longest path (sequence of decisions) from the root of the
    /// problem until this node.
    pub fn longest_path(&self) -> Vec<Decision> {
        let mut ret = vec![];
        let mut arc = &self.lp_arc;

        while arc.is_some() {
            let a = arc.as_ref().unwrap();
            ret.push(a.decision);
            arc = &a.src.lp_arc;
        }

        ret
    }
}
/// Two `NodeInfo` are (partially) equivalent if they are both exact, have the
/// same longest path and upper bound.
impl PartialEq for NodeInfo {
    fn eq(&self, other: &Self) -> bool {
        self.is_exact == other.is_exact &&
            self.lp_len == other.lp_len &&
            self.ub     == other.ub     &&
            self.lp_arc == other.lp_arc
    }
}
/// `NodeInfo`s can be ordered by considering their upper bound, the length of
/// their longest path and their exactitude.
impl Ord for NodeInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ub.cmp(&other.ub)
            .then_with(|| self.lp_len.cmp(&other.lp_len))
            .then_with(|| self.is_exact.cmp(&other.is_exact))
    }
}
impl PartialOrd for NodeInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

/// A node is the simplest constituent of an MDD. It basically consists of an
/// user specified state together with some meta data (stored in a `NodeInfo`
/// which does some bookkeeping about the position, longest path.. of this node
/// in the decision diagram).
#[derive(Debug, Clone, Eq)]
pub struct Node<T> {
    /// This is the (user-specified) state. It is the state reached by following
    /// the best path from the root node applying the transition relation at
    /// each decision.
    pub state: T,
    /// This is the node info which stores the metadata about the node.
    pub info: NodeInfo
}
// public methods that do not belong to any trait
impl <T> Node<T> {
    /// Constructor.
    ///
    /// Creates a new node info from the state, longest path information
    /// (`lp_len` and `lp_arc`) and a flag indicating if this denotes an
    /// exaxct node or not.  The upper bound is set to a default value
    /// (the greatest i32)
    pub fn new(state: T, lp_len: i32, lp_arc: Option<Edge>, is_exact: bool) -> Node<T> {
        Node{state, info: NodeInfo::new(lp_len, lp_arc, is_exact)}
    }
    /// Alt-Constructor
    ///
    /// This constructor explicitly creates a merged node. That is, a node
    /// which is going to be marked as 'not exact'.
    pub fn merged(state: T, lp_len: i32, lp_arc: Option<Edge>) -> Node<T> {
        Node{state, info: NodeInfo::new(lp_len, lp_arc, false)}
    }
}

impl <T> Hash for Node<T> where T: Hash {
    /// A node acts as a mere pass through for the state. Hence it is hashed
    /// on basis of the state only.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}
impl <T> PartialEq for Node<T> where T: Eq {
    /// To be coherent with `Hash`, nodes implement an equality test that bears
    /// on the node state only.
    fn eq(&self, other: &Self) -> bool {
        self.state.eq(&other.state)
    }
}

/// A `Layer` is a set of nodes that may be iterated upon.
///
/// This enum should really be considered as an opaque iterator even through
/// the `Layer` type pays for itself in various ways.
///
///   1. It facilitates the lecture of the code (it is much clearer to read that
///      you have an access to the current and next layers than knowing you
///      can iterate over two sets of nodes.
///   2. It makes it possible to use a polymorphic return type in traits
pub enum Layer<'a, T> {
    /// When a layer is simply backed by a vector (or a slice) of nodes
    Plain (std::slice::Iter<'a, Node<T>>),
    /// When a layer is implemented as a hashmap
    Mapped(std::collections::hash_map::Iter<'a, T, NodeInfo>),
}
/// Because a `Layer` is an opaque iterator, it implements the
/// standard `Iterator` trait from std lib.
impl <'a, T> Iterator for Layer<'a, T> {
    type Item = (&'a T, &'a NodeInfo);
    /// Returns the next node, or None if the iteration is exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Layer::Plain(i)  => i.next().map(|n| (&n.state, &n.info)),
            Layer::Mapped(i) => i.next()
        }
    }
}

#[cfg(test)]
mod test_var {
    use crate::core::common::Variable;

    #[test]
    fn test_var_id() {
        assert_eq!(0, Variable(0).id());
        assert_eq!(1, Variable(1).id());
        assert_eq!(2, Variable(2).id());
        assert_eq!(3, Variable(3).id());
    }
}

#[cfg(test)]
mod test_varset {
    use crate::core::common::{Variable, VarSet};

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
    use crate::core::common::{VarSet, VarSetIter, Variable};
    use crate::core::utils::BitSetIter;

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
mod test_domain {
    use bitset_fixed::BitSet;
    use crate::core::common::{Domain, VarSet, Variable};

    #[test]
    fn from_vector_empty() {
        let domain : Domain<'_> = vec![].into();
        assert_eq!(Vec::<i32>::new(), domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_vector_non_empty() {
        let domain : Domain<'_> = vec![1, 2, 4].into();
        assert_eq!(vec![1, 2, 4], domain.into_iter().collect::<Vec<i32>>());
    }

    #[test]
    fn from_slice_empty() {
        let data = vec![];
        let slice  : &[i32] = &data;
        let domain : Domain<'_> = slice.into();
        assert_eq!(Vec::<i32>::new(), domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_slice_non_empty() {
        let data = [1, 2, 4];
        let slice  : &[i32] = &data;
        let domain : Domain<'_> = slice.into();
        assert_eq!(vec![1, 2, 4], domain.into_iter().collect::<Vec<i32>>());
    }

    #[test]
    fn from_bitset_empty() {
        let data = BitSet::new(5);
        let domain : Domain<'_> = (&data).into();
        assert_eq!(Vec::<i32>::new(), domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_bitset_non_empty() {
        let mut data = BitSet::new(5);
        data.set(2, true);
        data.set(3, true);
        let domain : Domain<'_> = (&data).into();
        assert_eq!(vec![2, 3], domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_varset_empty() {
        let data = VarSet::empty();
        let domain : Domain<'_> = (&data).into();
        assert_eq!(Vec::<i32>::new(), domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_varset_non_empty() {
        let mut data = VarSet::all(5);
        data.remove(Variable(0));
        data.remove(Variable(1));
        data.remove(Variable(4));
        let domain : Domain<'_> = (&data).into();
        assert_eq!(vec![2, 3], domain.into_iter().collect::<Vec<i32>>());
    }

    #[test]
    fn from_range_empty_going_negative() {
        let data = 0..-1_i32;
        let domain : Domain<'_> = data.into();
        assert_eq!(Vec::<i32>::new(), domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_range_empty_positive() {
        let data = 0..0_i32;
        let domain : Domain<'_> = data.into();
        assert_eq!(Vec::<i32>::new(), domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_range_non_empty() {
        let data = 0..5_i32;
        let domain : Domain<'_> = data.into();
        assert_eq!(vec![0,1,2,3,4], domain.into_iter().collect::<Vec<i32>>());
    }

    #[test]
    fn from_range_inclusive_empty_going_negative() {
        let data = 0..=-1_i32;
        let domain : Domain<'_> = data.into();
        assert_eq!(Vec::<i32>::new(), domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_range_inclusive_single_value() {
        let data = 0..=0_i32;
        let domain : Domain<'_> = data.into();
        assert_eq!(vec![0], domain.into_iter().collect::<Vec<i32>>());
    }
    #[test]
    fn from_range_inclusive_non_empty() {
        let data = 0..=5_i32;
        let domain : Domain<'_> = data.into();
        assert_eq!(vec![0,1,2,3,4,5], domain.into_iter().collect::<Vec<i32>>());
    }
}

#[cfg(test)]
mod test_nodeinfo {
    use crate::core::common::{NodeInfo, Edge, Decision, Variable};
    use std::sync::Arc;

    #[test]
    fn constructor_root() {
        let x = NodeInfo::new(42, None, true); // root

        assert_eq!(42,   x.lp_len);
        assert_eq!(None, x.lp_arc);
        assert_eq!(true, x.is_exact);
    }
    #[test]
    fn constructor_default_ub_value() {
        let x = NodeInfo::new(42, None, true);

        assert_eq!(i32::max_value(), x.ub);
    }
    #[test]
    fn constructor_parent() {
        let x   = NodeInfo::new(42, None, true);
        let edge= Edge {src: Arc::new(x), decision: Decision{variable: Variable(0), value: 4}};
        let y   = NodeInfo::new(64, Some(edge), true);

        assert_eq!(64,   y.lp_len);
        assert_eq!(true, y.is_exact);
        assert!(y.lp_arc.is_some());
        assert_eq!(0,   y.lp_arc.as_ref().unwrap().decision.variable.id());
        assert_eq!(4,   y.lp_arc.as_ref().unwrap().decision.value);
        assert_eq!(42,  y.lp_arc.as_ref().unwrap().src.lp_len);
        assert_eq!(true,y.lp_arc.as_ref().unwrap().src.is_exact);
    }
    #[test]
    fn longest_path() {
        let x   = NodeInfo::new(42, None, true);
        let edge= Edge {src: Arc::new(x), decision: Decision{variable: Variable(0), value: 4}};
        let y   = NodeInfo::new(64, Some(edge), true);
        let edge= Edge {src: Arc::new(y), decision: Decision{variable: Variable(1), value: 2}};
        let z   = NodeInfo::new(66, Some(edge), true);

        assert_eq!(
            vec![Decision{variable: Variable(1), value: 2},
                 Decision{variable: Variable(0), value: 4}],
            z.longest_path()
        )
    }
    #[test]
    fn merge_keeps_longest_path_other_is_best() {
        let a     = NodeInfo::new(10, None, true);
        let edge  = Edge {src: Arc::new(a), decision: Decision{variable: Variable(0), value: 0}};
        let b     = NodeInfo::new(20, Some(edge), true);
        let edge  = Edge {src: Arc::new(b), decision: Decision{variable: Variable(1), value: 45}};
        let mut c = NodeInfo::new(30, Some(edge), true);

        let x     = NodeInfo::new(42, None, true);
        let edge  = Edge {src: Arc::new(x), decision: Decision{variable: Variable(0), value: 4}};
        let y     = NodeInfo::new(64, Some(edge), true);
        let edge  = Edge {src: Arc::new(y), decision: Decision{variable: Variable(1), value: 2}};
        let z     = NodeInfo::new(66, Some(edge), true);

        c.merge(z);
        assert_eq!(66, c.lp_len);
        assert_eq!(
            c.longest_path(),
            vec![Decision{variable: Variable(1), value: 2},
                 Decision{variable: Variable(0), value: 4}]);
    }
    #[test]
    fn merge_keeps_longest_path_i_am_best() {
        let a     = NodeInfo::new(10, None, true);
        let edge  = Edge {src: Arc::new(a), decision: Decision{variable: Variable(0), value: 0}};
        let b     = NodeInfo::new(20, Some(edge), true);
        let edge  = Edge {src: Arc::new(b), decision: Decision{variable: Variable(1), value: 45}};
        let mut c = NodeInfo::new(70, Some(edge), true);

        let x     = NodeInfo::new(42, None, true);
        let edge  = Edge {src: Arc::new(x), decision: Decision{variable: Variable(0), value: 4}};
        let y     = NodeInfo::new(64, Some(edge), true);
        let edge  = Edge {src: Arc::new(y), decision: Decision{variable: Variable(1), value: 2}};
        let z     = NodeInfo::new(66, Some(edge), true);

        c.merge(z);
        assert_eq!(70, c.lp_len);
        assert_eq!(
            c.longest_path(),
            vec![Decision{variable: Variable(1), value: 45},
                 Decision{variable: Variable(0), value: 0}]);
    }
    #[test]
    fn merge_keeps_tightest_bound_other_is_best() {
        let mut a = NodeInfo::new(10, None, true);
        let mut b = NodeInfo::new(42, None, true);

        a.ub = 10;
        b.ub = 5;

        a.merge(b);
        assert_eq!(5, a.ub);
    }
    #[test]
    fn merge_keeps_tightest_bound_i_am_best() {
        let mut a = NodeInfo::new(10, None, true);
        let mut b = NodeInfo::new(42, None, true);

        a.ub = 5;
        b.ub = 10;

        a.merge(b);
        assert_eq!(5, a.ub);
    }
    #[test]
    fn merge_tracks_exactitude_both_true() {
        let mut a = NodeInfo::new(10, None, true);
        let b     = NodeInfo::new(42, None, true);

        a.merge(b);
        assert_eq!(true, a.is_exact);
    }
    #[test]
    fn merge_tracks_exactitude_both_false() {
        let mut a = NodeInfo::new(10, None, false);
        let b     = NodeInfo::new(42, None, false);

        a.merge(b);
        assert_eq!(false, a.is_exact);
    }
    #[test]
    fn merge_tracks_exactitude_me_true_she_false() {
        let mut a = NodeInfo::new(10, None, true);
        let b     = NodeInfo::new(42, None, false);

        a.merge(b);
        assert_eq!(false, a.is_exact);
    }
    #[test]
    fn merge_tracks_exactitude_me_false_she_true() {
        let mut a = NodeInfo::new(10, None, false);
        let b     = NodeInfo::new(42, None, true);

        a.merge(b);
        assert_eq!(false, a.is_exact);
    }
    #[test]
    fn partial_equiv_not_equal_if_not_same_exactitude() {
        let a = NodeInfo::new(10, None, false);
        let b = NodeInfo::new(10, None, true);
        assert_ne!(a, b);

        let a = NodeInfo::new(10, None, true);
        let b = NodeInfo::new(10, None, false);
        assert_ne!(a, b);
    }
    #[test]
    fn partial_equiv_not_equal_if_different_lp_len() {
        let a = NodeInfo::new(10, None, true);
        let b = NodeInfo::new(12, None, true);
        assert_ne!(a, b);
    }
    #[test]
    fn partial_equiv_not_equal_if_different_ub() {
        let mut a = NodeInfo::new(12, None, true);
        let mut b = NodeInfo::new(12, None, true);
        a.ub = 15;
        b.ub = 20;
        assert_ne!(a, b);
    }
    #[test]
    fn partial_equiv_not_equal_if_different_parent() {
        let a = NodeInfo::new(10, None, true);
        let e = Edge{src: Arc::new(a), decision: Decision{variable: Variable(0), value: 0}};
        let b = NodeInfo::new(12, Some(e), true);

        let x = NodeInfo::new(12, None, true);
        let e = Edge{src: Arc::new(x), decision: Decision{variable: Variable(0), value: 0}};
        let y = NodeInfo::new(12, Some(e), true);
        assert_ne!(b, y);
    }
    #[test]
    fn partial_equiv_not_equal_if_different_decision_value() {
        let a = Arc::new(NodeInfo::new(10, None, true));
        let e = Edge{src: Arc::clone(&a), decision: Decision{variable: Variable(0), value: 1}};
        let b = NodeInfo::new(12, Some(e), true);

        let e = Edge{src: Arc::clone(&a), decision: Decision{variable: Variable(0), value: 0}};
        let y = NodeInfo::new(12, Some(e), true);
        assert_ne!(b, y);
    }
    #[test]
    fn partial_equiv_not_equal_if_different_decision_variable() {
        let a = Arc::new(NodeInfo::new(10, None, true));
        let e = Edge{src: Arc::clone(&a), decision: Decision{variable: Variable(1), value: 0}};
        let b = NodeInfo::new(12, Some(e), true);

        let e = Edge{src: Arc::clone(&a), decision: Decision{variable: Variable(0), value: 0}};
        let y = NodeInfo::new(12, Some(e), true);
        assert_ne!(b, y);
    }
    #[test]
    fn partial_equiv() {
        let a = Arc::new(NodeInfo::new(10, None, true));
        let e = Edge{src: Arc::clone(&a), decision: Decision{variable: Variable(1), value: 0}};
        let b = NodeInfo::new(12, Some(e), true);

        let e = Edge{src: Arc::clone(&a), decision: Decision{variable: Variable(1), value: 0}};
        let y = NodeInfo::new(12, Some(e), true);
        assert_eq!(b, y);
    }

    #[test]
    fn cmp_less_if_lower_bound() {
        let mut a = NodeInfo::new(12, None, true);
        let mut b = NodeInfo::new(12, None, true);
        a.ub = 15;
        b.ub = 20;
        assert!(a <  b);
        assert!(a <= b);
        assert!(b >  a);
        assert!(b >= a);
    }
    #[test]
    fn cmp_less_if_shorter_lp() {
        let a = NodeInfo::new(10, None, true);
        let b = NodeInfo::new(12, None, true);

        assert!(a <  b);
        assert!(a <= b);
        assert!(b >  a);
        assert!(b >= a);
    }
    #[test]
    fn cmp_less_if_inexact() {
        let a = NodeInfo::new(12, None, false);
        let b = NodeInfo::new(12, None, true);

        assert!(a <  b);
        assert!(a <= b);
        assert!(b >  a);
        assert!(b >= a);
    }
}

#[cfg(test)]
mod test_node {
    use crate::core::common::{Node, Edge, Decision, Variable};
    use std::sync::Arc;
    use metrohash::MetroHash64;
    use std::hash::{Hash, Hasher};
    use bitset_fixed::BitSet;

    #[test]
    fn constructor_new_root() {
        let x = Node::new(64, 42, None, true); // root

        assert_eq!(64,   x.state);
        assert_eq!(42,   x.info.lp_len);
        assert_eq!(None, x.info.lp_arc);
        assert_eq!(true, x.info.is_exact);
    }
    #[test]
    fn constructor_new_default_ub_value() {
        let x = Node::new(64, 42, None, true);

        assert_eq!(i32::max_value(), x.info.ub);
    }
    #[test]
    fn constructor_new_parent() {
        let x   = Node::new(64, 42, None, true);
        let edge= Edge {src: Arc::new(x.info), decision: Decision{variable: Variable(0), value: 4}};
        let y   = Node::new(32, 64, Some(edge), true);

        assert_eq!(32,   y.state);
        assert_eq!(64,   y.info.lp_len);
        assert_eq!(true, y.info.is_exact);
        assert!(y.info.lp_arc.is_some());
        assert_eq!(0,   y.info.lp_arc.as_ref().unwrap().decision.variable.id());
        assert_eq!(4,   y.info.lp_arc.as_ref().unwrap().decision.value);
        assert_eq!(42,  y.info.lp_arc.as_ref().unwrap().src.lp_len);
        assert_eq!(true,y.info.lp_arc.as_ref().unwrap().src.is_exact);
    }
    #[test]
    fn constructor_new_longest_path() {
        let x   = Node::new(30, 42, None, true);
        let edge= Edge {src: Arc::new(x.info), decision: Decision{variable: Variable(0), value: 4}};
        let y   = Node::new(20, 64, Some(edge), true);
        let edge= Edge {src: Arc::new(y.info), decision: Decision{variable: Variable(1), value: 2}};
        let z   = Node::new(10, 66, Some(edge), true);

        assert_eq!(
            vec![Decision{variable: Variable(1), value: 2},
                 Decision{variable: Variable(0), value: 4}],
            z.info.longest_path()
        )
    }

    #[test]
    fn constructor_merged_always_not_exact() {
        let x = Node::merged(64, 42, None);
        assert_eq!(false, x.info.is_exact);
    }
    #[test]
    fn constructor_merged_default_ub_value() {
        let x = Node::merged(64, 42, None);
        assert_eq!(i32::max_value(), x.info.ub);
    }
    #[test]
    fn constructor_merged_parent() {
        let x   = Node::new(64, 42, None, true);
        let edge= Edge {src: Arc::new(x.info), decision: Decision{variable: Variable(0), value: 4}};
        let y   = Node::merged(32, 64, Some(edge));

        assert_eq!(32,    y.state);
        assert_eq!(64,    y.info.lp_len);
        assert_eq!(false, y.info.is_exact);
        assert!(y.info.lp_arc.is_some());
        assert_eq!(0,   y.info.lp_arc.as_ref().unwrap().decision.variable.id());
        assert_eq!(4,   y.info.lp_arc.as_ref().unwrap().decision.value);
        assert_eq!(42,  y.info.lp_arc.as_ref().unwrap().src.lp_len);
        assert_eq!(true,y.info.lp_arc.as_ref().unwrap().src.is_exact);
    }

    #[test]
    fn node_hash_is_a_passthrough_for_state_integer() {
        let mut n_hasher = MetroHash64::with_seed(42);
        let mut s_hasher = MetroHash64::with_seed(42);

        let x = Node::new(42, 128, None, true);
        x.hash(&mut n_hasher);
        42.hash(&mut s_hasher);

        assert_eq!(n_hasher.finish(), s_hasher.finish());
    }
    #[test]
    fn node_hash_is_a_passthrough_for_state_string() {
        let mut n_hasher = MetroHash64::with_seed(42);
        let mut s_hasher = MetroHash64::with_seed(42);

        let x = Node::new("coucou", 128, None, true);
        x.hash(&mut n_hasher);
        "coucou".hash(&mut s_hasher);

        assert_eq!(n_hasher.finish(), s_hasher.finish());
    }
    #[test]
    fn node_hash_is_a_passthrough_for_state_bitset() {
        let mut n_hasher = MetroHash64::with_seed(42);
        let mut s_hasher = MetroHash64::with_seed(42);

        let mut state = BitSet::new(5);
        state.set(3, true);
        state.set(4, true);

        let x = Node::new(state.clone(), 128, None, true);
        x.hash(&mut n_hasher);
        state.hash(&mut s_hasher);

        assert_eq!(n_hasher.finish(), s_hasher.finish());
    }
    #[test]
    fn node_equality_depends_on_state_only_integer() {
        // integers
        let x = Node::new(42, 12, None, true);
        let y = Node::new(42, 64, None, false);
        assert_eq!(x, y);
    }
    #[test]
    fn node_equality_depends_on_state_only_string() {
        // strings
        let x = Node::new("coucou", 12, None, true);
        let y = Node::new("coucou", 64, None, false);
        assert_eq!(x, y);
    }
    #[test]
    fn node_equality_depends_on_state_only_bitset() {
        // bitsets
        let mut state = BitSet::new(5);
        state.set(3, true);
        state.set(4, true);
        let x = Node::new(state.clone(), 12, None, true);
        let y = Node::new(state        , 64, None, false);
        assert_eq!(x, y);
    }
}

#[cfg(test)]
mod test_layer {
    use crate::core::common::{Node, Layer, NodeInfo};
    use std::collections::HashMap;

    #[test]
    fn plain_empty() {
        let nodes : Vec<Node<usize>> = vec![];
        let layer = Layer::Plain(nodes.iter());

        assert_eq!(Vec::<(&usize, &NodeInfo)>::new(), layer.collect::<Vec<(&usize, &NodeInfo)>>());
    }
    #[test]
    fn plain_non_empty() {
        let x = Node::new(12, 12, None, true);
        let nodes : Vec<Node<usize>> = vec![x.clone()];
        let layer = Layer::Plain(nodes.iter());

        assert_eq!(vec![(&x.state, &x.info)], layer.collect::<Vec<(&usize, &NodeInfo)>>());
    }
    #[test]
    fn plain_next() {
        let x = Node::new(12, 12, None, true);
        let nodes : Vec<Node<usize>> = vec![x.clone()];
        let mut layer = Layer::Plain(nodes.iter());

        assert_eq!(Some((&x.state, &x.info)), layer.next());
        assert_eq!(None, layer.next());
    }
    #[test]
    fn plain_next_exhausted() {
        let x = Node::new(12, 12, None, true);
        let nodes : Vec<Node<usize>> = vec![x.clone()];
        let mut layer = Layer::Plain(nodes.iter());

        assert_eq!(Some((&x.state, &x.info)), layer.next());
        assert_eq!(None, layer.next());
        assert_eq!(None, layer.next());
        assert_eq!(None, layer.next());
        assert_eq!(None, layer.next());
    }
    #[test]
    fn mapped_empty() {
        let nodes : HashMap<usize, NodeInfo> = HashMap::new();
        let layer = Layer::Mapped(nodes.iter());

        assert_eq!(Vec::<(&usize, &NodeInfo)>::new(), layer.collect::<Vec<(&usize, &NodeInfo)>>());
    }
    #[test]
    fn mapped_non_empty() {
        let x = Node::new(12, 12, None, true);
        let mut nodes : HashMap<usize, NodeInfo> = HashMap::new();
        nodes.insert(x.state.clone(), x.info.clone());
        let layer = Layer::Mapped(nodes.iter());

        assert_eq!(vec![(&x.state, &x.info)], layer.collect::<Vec<(&usize, &NodeInfo)>>());
    }
    #[test]
    fn mapped_next() {
        let x = Node::new(12, 12, None, true);
        let mut nodes : HashMap<usize, NodeInfo> = HashMap::new();
        nodes.insert(x.state.clone(), x.info.clone());
        let mut layer = Layer::Mapped(nodes.iter());

        assert_eq!(Some((&x.state, &x.info)), layer.next());
        assert_eq!(None, layer.next());
    }
    #[test]
    fn mapped_next_exhausted() {
        let x = Node::new(12, 12, None, true);
        let mut nodes : HashMap<usize, NodeInfo> = HashMap::new();
        nodes.insert(x.state.clone(), x.info.clone());
        let mut layer = Layer::Mapped(nodes.iter());

        assert_eq!(Some((&x.state, &x.info)), layer.next());
        assert_eq!(None, layer.next());
        assert_eq!(None, layer.next());
        assert_eq!(None, layer.next());
        assert_eq!(None, layer.next());
    }
}