//! This module defines some utility types that are convenient to work with
//! but do not belong to the heart of developing an MDD-based optimization
//! solver. In particular, this module defines:
//!   * an iterator to iterate over the 1-bits of a fixed bitset (`BitSetIter`)
//!   * a wrapper type that allows the comparison of two fixed bitsets, on a
//!     lexical basis (`LexBitSet`)
//!   * a typesafe adapter for function/closure pointers, which implements the
//!     `Compare`, `LoadVars`, `VariableHeuristic` and `WidthHeuristic` traits.
//!     This adapter is really a zero-cost abstraction and may be used whenever
//!     you feel it would make sense. This is not going to hamper your perf (`Func`).
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::iter::Cloned;
use std::slice::Iter;

use bitset_fixed::BitSet;
use compare::Compare;

use crate::core::abstraction::heuristics::{LoadVars, VariableHeuristic, WidthHeuristic};
use crate::core::common::{Layer, Node, Variable, VarSet};

/// This structure defines an iterator capable of iterating over the 1-bits of
/// a fixed bitset. It uses word representation of the items in the set, so it
/// should be more efficient to use than a crude iteration over the elements of
/// the set.
///
/// # Example
/// ```
/// # use bitset_fixed::BitSet;
/// # use ddo::core::utils::BitSetIter;
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
    iter  : Cloned<Iter<'a, u64>>,
    word  : Option<u64>,
    base  : usize,
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
/// # use ddo::core::utils::LexBitSet;
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

/// A zero-cost abstraction adapter for function pointers. This is typically
/// useful to pass a pointer to function/closure as an heuristic to use in the
/// solver/mdd configuration.
///
/// Note that, it implements different heuristics trait depending on the signature
/// of the wrapped function F.
#[derive(Clone)]
pub struct Func<F>(pub F);

/// If the function is a node comparator, then `Func` will auto implement the
/// `Compare` trait for that function.
impl <T, F> Compare<Node<T>> for Func<F>
    where F: Clone + Fn(&Node<T>, &Node<T>) -> Ordering {

    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        (self.0)(a, b)
    }
}
/// If the function <F> determines a maximum width given a set of free vars,
/// then `Func` will auto implement `WidthHeuristic` for that function.
impl <F> WidthHeuristic for Func<F>
    where F: Fn(&VarSet) -> usize {

    fn max_width(&self, free_vars: &VarSet) -> usize {
        (self.0)(free_vars)
    }
}
/// Whenever the parameter function <F> of `Func` is a function/closure that
/// possibly returns a variable based on a set of free variables and two layers
/// (current and next), then `Func` will auto implement the `VariableHeuristic`
/// trait for that function.
impl <T, F> VariableHeuristic<T> for Func<F>
    where F: Fn(&VarSet, Layer<'_, T>, Layer<'_, T>) -> Option<Variable> {

    fn next_var(&self, free_vars: &VarSet, current: Layer<'_, T>, next: Layer<'_, T>) -> Option<Variable> {
        (self.0)(free_vars, current, next)
    }
}
/// If <F> is able to extract a set of variables out of a reference to some
/// node, then `Func` will auto implement the `LoadVars` trait for that function.
impl <T, F> LoadVars<T> for Func<F>
    where F: Fn(&Node<T>) -> VarSet {

    fn variables(&self, node: &Node<T>) -> VarSet {
        (self.0)(node)
    }
}

#[cfg(test)]
/// These tests validate the behavior of the bitset iterator `BitSetIter`.
mod tests_bitset_iter {
    use bitset_fixed::BitSet;
    use crate::core::utils::BitSetIter;

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
    use crate::core::utils::LexBitSet;

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