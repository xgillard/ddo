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

//! This module provides some utility structures that are used when implementing
//! an mdd.

/// This structure stores a compact set of flags relating to a given node.
/// So far, it maintains the following:
/// - Exact    which is true iff the current node is exact
/// - Relaxed  which is true iff the current node is relaxed (result of a merge)
///
/// # Remark
/// It might seem as though \( \neg exact \equiv relaxed \). But actually it is not
/// the case. The subtlety is the following:
/// \( relaxed \implies \neg exact \) and \( exact \implies \neg relaxed \)
/// but \( \neg exact \not\implies relaxed \).
///
/// In other words, a node can be inexact and not relaxed either. This happens
/// when a node is not the result of a merge operation but one of its ancestors
/// is.
///
/// # Important Note
/// For the sake of clarity, and to keep the code simple and lean, it was decided
/// that the setters would not enforce the above relationships. The only exception
/// to this rule being that a node is always considered inexact when the relaxed
/// flag is on.
///
/// # Default
/// By default, a node is considered exact, not relaxed and not feasible (because
/// this field is not set until local bounds computation).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeFlags(u8);
impl NodeFlags {
    /// The position of the exact flag
    pub const F_EXACT: u8 = 1;
    /// The position of the relaxed flag
    pub const F_RELAXED: u8 = 2;
    /// The position of the marked flag.
    pub const F_MARKED: u8 = 4;
    /// The position of the cut-set flag.
    pub const F_CUTSET: u8 = 8;
    /// The position of the deleted flag.
    pub const F_DELETED: u8 = 16;
    /// The position of the cache flag.
    pub const F_CACHE: u8 = 32;
    /// The position of the above cut-set flag.
    pub const F_ABOVE_CUTSET: u8 = 64;

    /// Creates a new set of flags, either initialized with exact on or with
    /// relaxed on.
    #[allow(dead_code)]
    #[inline]
    pub fn new(relaxed: bool) -> Self {
        if relaxed {
            Self::new_relaxed()
        } else {
            Self::new_exact()
        }
    }
    /// Creates a new set of flags having only the exact flag turned on.
    #[inline]
    pub fn new_exact() -> Self {
        NodeFlags(NodeFlags::F_EXACT)
    }
    /// Creates a new set of flags having only the relaxed flag turned on.
    #[inline]
    pub fn new_relaxed() -> Self {
        NodeFlags(NodeFlags::F_RELAXED)
    }
    /// Returns true iff the exact flag is on and the relaxed flag is off
    #[inline]
    pub fn is_exact(self) -> bool {
        self.test(NodeFlags::F_EXACT) && !self.test(NodeFlags::F_RELAXED)
    }
    /// Returns true iff the exact flag is turned on
    #[inline]
    pub fn is_relaxed(self) -> bool {
        self.test(NodeFlags::F_RELAXED)
    }
    /// Returns true iff the marked flag is turned on
    #[inline]
    pub fn is_marked(self) -> bool {
        self.test(NodeFlags::F_MARKED)
    }
    /// Returns true iff the cut-set flag is turned on
    #[inline]
    pub fn is_cutset(self) -> bool {
        self.test(NodeFlags::F_CUTSET)
    }
    /// Returns true iff the above cut-set flag is turned on
    #[inline]
    pub fn is_above_cutset(self) -> bool {
        self.test(NodeFlags::F_ABOVE_CUTSET)
    }
    /// Returns true iff the deleted flag is turned on
    #[inline]
    pub fn is_deleted(self) -> bool {
        self.test(NodeFlags::F_DELETED)
    }
    /// Returns true iff the cache flag is turned on
    #[inline]
    pub fn is_pruned_by_cache(self) -> bool {
        self.test(NodeFlags::F_CACHE)
    }
    /// Sets the exact flag to the given value
    #[inline]
    pub fn set_exact(&mut self, exact: bool) {
        self.set(NodeFlags::F_EXACT, exact)
    }
    /// Sets the relaxed flat to the given value
    #[inline]
    pub fn set_relaxed(&mut self, relaxed: bool) {
        self.set(NodeFlags::F_RELAXED, relaxed)
    }
    /// Sets the marked flag to the given value
    #[inline]
    pub fn set_marked(&mut self, marked: bool) {
        self.set(NodeFlags::F_MARKED, marked)
    }
    #[inline]
    pub fn set_cutset(&mut self, cutset: bool) {
        self.set(NodeFlags::F_CUTSET, cutset)
    }
    #[inline]
    pub fn set_above_cutset(&mut self, above: bool) {
        self.set(NodeFlags::F_ABOVE_CUTSET, above)
    }
    /// Sets the deleted flag to the given value
    #[inline]
    pub fn set_deleted(&mut self, deleted: bool) {
        self.set(NodeFlags::F_DELETED, deleted)
    }
    /// Sets the cache flag to the given value
    #[inline]
    pub fn set_pruned_by_cache(&mut self, cache: bool) {
        self.set(NodeFlags::F_CACHE, cache)
    }
    /// Checks whether all the flags encoded in the given mask are turned on.
    /// Otherwise, it returns false
    #[inline]
    pub fn test(self, mask: u8) -> bool {
        self.0 & mask == mask
    }
    /// Sets the value of a given flag to the selected polarity
    #[inline]
    pub fn set(&mut self, flag: u8, value: bool) {
        if value {
            self.add(flag)
        } else {
            self.remove(flag)
        }
    }
    /// Turns the given flag(s) on.
    #[inline]
    pub fn add(&mut self, flags: u8) {
        self.0 |= flags;
    }
    /// Turns the given flag(s) off.
    #[inline]
    pub fn remove(&mut self, flags: u8) {
        self.0 &= !flags;
    }
}
impl Default for NodeFlags {
    /// Creates a default set of flags (only the exact flag is turned on).
    fn default() -> Self {
        NodeFlags::new_exact()
    }
}

// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod test_node_flags {
    use super::NodeFlags;

    #[test]
    fn new_can_be_relaxed_or_not() {
        let tested = NodeFlags::new_exact();
        assert_eq!(NodeFlags(NodeFlags::F_EXACT), tested);
        assert_eq!(true, tested.is_exact());
        assert_eq!(false, tested.is_relaxed());

        let tested = NodeFlags::new_relaxed();
        assert_eq!(NodeFlags(NodeFlags::F_RELAXED), tested);
        assert_eq!(false, tested.is_exact());
        assert_eq!(true, tested.is_relaxed());

        assert_eq!(NodeFlags::new(false), NodeFlags::new_exact());
        assert_eq!(NodeFlags::new(true), NodeFlags::new_relaxed());
    }
    #[test]
    fn at_creation_time_marked_flag_is_never_set() {
        let tested = NodeFlags::new(false);
        assert_eq!(false, tested.is_marked());

        let tested = NodeFlags::new(true);
        assert_eq!(false, tested.is_marked());
    }
    #[test]
    fn is_relaxed_iff_created_or_marked_relaxed() {
        let mut tested = NodeFlags::new_relaxed();
        assert_eq!(true, tested.is_relaxed());

        tested.set_relaxed(false);
        assert_eq!(false, tested.is_relaxed());

        tested.set_relaxed(true);
        assert_eq!(true, tested.is_relaxed());

        let mut tested = NodeFlags::new_exact();
        assert_eq!(false, tested.is_relaxed());

        tested.set_relaxed(true);
        assert_eq!(true, tested.is_relaxed());

        tested.set_relaxed(false);
        assert_eq!(false, tested.is_relaxed());
    }
    #[test]
    fn is_exact_iff_marked_so_and_not_relaxed() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(true, tested.is_exact());

        tested.set_exact(false);
        assert_eq!(false, tested.is_exact());

        tested.set_exact(true);
        assert_eq!(true, tested.is_exact());

        tested.set_relaxed(true);
        assert_eq!(false, tested.is_exact());

        tested.set_exact(false);
        assert_eq!(false, tested.is_exact());

        tested.set_exact(true);
        assert_eq!(false, tested.is_exact());
    }
    #[test]
    fn is_marked_iff_marked_so() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(false, tested.is_marked());

        tested.set_marked(true);
        assert_eq!(true, tested.is_marked());

        tested.set_marked(false);
        assert_eq!(false, tested.is_marked());

        tested.set_marked(true);
        assert_eq!(true, tested.is_marked());
    }
    #[test]
    fn is_cutset_iff_marked_so() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(false, tested.is_cutset());

        tested.set_cutset(true);
        assert_eq!(true, tested.is_cutset());

        tested.set_cutset(false);
        assert_eq!(false, tested.is_cutset());

        tested.set_cutset(true);
        assert_eq!(true, tested.is_cutset());
    }
    #[test]
    fn is_deleted_iff_marked_so() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(false, tested.is_deleted());

        tested.set_deleted(true);
        assert_eq!(true, tested.is_deleted());

        tested.set_deleted(false);
        assert_eq!(false, tested.is_deleted());

        tested.set_deleted(true);
        assert_eq!(true, tested.is_deleted());
    }
    #[test]
    fn is_pruned_by_cache_iff_marked_so() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(false, tested.is_pruned_by_cache());

        tested.set_pruned_by_cache(true);
        assert_eq!(true, tested.is_pruned_by_cache());

        tested.set_pruned_by_cache(false);
        assert_eq!(false, tested.is_pruned_by_cache());

        tested.set_pruned_by_cache(true);
        assert_eq!(true, tested.is_pruned_by_cache());
    }
    #[test]
    fn test_yields_the_value_of_the_flag() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));

        tested.set_relaxed(true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));

        tested.set_marked(true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));

        tested.set_exact(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));

        tested.set_relaxed(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));

        tested.set_marked(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));

        tested.set_deleted(true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(true, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));

        tested.set_pruned_by_cache(true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(true, tested.test(NodeFlags::F_DELETED));
        assert_eq!(true, tested.test(NodeFlags::F_CACHE));

        tested.set_deleted(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(true, tested.test(NodeFlags::F_CACHE));

        tested.set_pruned_by_cache(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_CUTSET));
        assert_eq!(false, tested.test(NodeFlags::F_DELETED));
        assert_eq!(false, tested.test(NodeFlags::F_CACHE));
    }
    #[test]
    fn test_checks_the_value_of_more_than_one_flag() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );

        tested.set_marked(true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(true, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );

        tested.set_relaxed(true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(true, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(true, tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED));
        assert_eq!(
            true,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            true,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );

        let mut tested = NodeFlags::new_exact();
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );

        tested.set_marked(true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(true, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );

        tested.set_exact(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );

        tested.set_relaxed(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );

        tested.set_marked(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT | NodeFlags::F_MARKED));
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
        assert_eq!(
            false,
            tested.test(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED)
        );
    }

    #[test]
    fn add_turns_one_or_more_flags_on() {
        let mut tested = NodeFlags(0);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        tested.add(NodeFlags::F_EXACT);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_RELAXED);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_MARKED);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_EXACT | NodeFlags::F_MARKED);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
    }
    #[test]
    fn remove_turns_one_or_more_flags_off() {
        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        tested.remove(NodeFlags::F_EXACT);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.remove(NodeFlags::F_RELAXED);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.remove(NodeFlags::F_MARKED);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.remove(NodeFlags::F_EXACT | NodeFlags::F_MARKED);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.remove(NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.remove(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));
    }
    #[test]
    fn set_turns_one_or_more_flags_on_or_off() {
        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        tested.set(NodeFlags::F_EXACT, false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.set(NodeFlags::F_RELAXED, false);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.set(NodeFlags::F_MARKED, false);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.set(NodeFlags::F_EXACT | NodeFlags::F_MARKED, false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.set(NodeFlags::F_RELAXED | NodeFlags::F_MARKED, false);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED);
        tested.set(
            NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED,
            false,
        );
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        //
        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_EXACT, true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_RELAXED, true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_MARKED, true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_EXACT | NodeFlags::F_MARKED, true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_RELAXED | NodeFlags::F_MARKED, true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));

        let mut tested = NodeFlags(0);
        tested.set(
            NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_MARKED,
            true,
        );
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_MARKED));
    }
    #[test]
    fn by_default_only_the_exact_flag_is_on() {
        assert_eq!(true, NodeFlags::default().test(NodeFlags::F_EXACT));
        assert_eq!(false, NodeFlags::default().test(NodeFlags::F_RELAXED));
        assert_eq!(false, NodeFlags::default().test(NodeFlags::F_MARKED));
        assert_eq!(false, NodeFlags::default().test(NodeFlags::F_CUTSET));
        assert_eq!(false, NodeFlags::default().test(NodeFlags::F_DELETED));
        assert_eq!(false, NodeFlags::default().test(NodeFlags::F_CACHE));
    }
}
