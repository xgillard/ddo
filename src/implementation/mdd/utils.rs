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
/// - Feasible which is true iff there exists a path between this node and the
///            terminal.
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
    const F_EXACT   : u8 = 1;
    /// The position of the relaxed flag
    const F_RELAXED : u8 = 2;
    /// The position of the feasible flag.
    const F_FEASIBLE: u8 = 4;

    /// Creates a new set of flags, either initialized with exact on or with
    /// relaxed on.
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
    /// Returns true iff the feasible flag is turned on
    #[inline]
    pub fn is_feasible(self) -> bool {
        self.test(NodeFlags::F_FEASIBLE)
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
    /// Sets the feasible flag to the given value
    #[inline]
    pub fn set_feasible(&mut self, feasible: bool) {
        self.set(NodeFlags::F_FEASIBLE, feasible)
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
mod test_node_flags {
    use crate::implementation::mdd::utils::NodeFlags;

    #[test]
    fn new_can_be_relaxed_or_not() {
        let tested =  NodeFlags::new_exact();
        assert_eq!(NodeFlags(NodeFlags::F_EXACT),  tested);
        assert_eq!(true,  tested.is_exact());
        assert_eq!(false, tested.is_relaxed());

        let tested =  NodeFlags::new_relaxed();
        assert_eq!(NodeFlags(NodeFlags::F_RELAXED),  tested);
        assert_eq!(false, tested.is_exact());
        assert_eq!(true,  tested.is_relaxed());

        assert_eq!(NodeFlags::new(false), NodeFlags::new_exact());
        assert_eq!(NodeFlags::new(true),  NodeFlags::new_relaxed());
    }
    #[test]
    fn at_creation_time_feasible_flag_is_never_set() {
        let tested = NodeFlags::new(false);
        assert_eq!(false, tested.is_feasible());

        let tested = NodeFlags::new(true);
        assert_eq!(false, tested.is_feasible());
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
    fn is_feasible_iff_marked_so() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(false, tested.is_feasible());

        tested.set_feasible(true);
        assert_eq!(true, tested.is_feasible());

        tested.set_feasible(false);
        assert_eq!(false, tested.is_feasible());

        tested.set_feasible(true);
        assert_eq!(true, tested.is_feasible());
    }
    #[test]
    fn test_yields_the_value_of_the_flag() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        tested.set_relaxed(true);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        tested.set_feasible(true);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        tested.set_exact(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        tested.set_relaxed(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        tested.set_feasible(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));
    }
    #[test]
    fn test_checks_the_value_of_more_than_one_flag() {
        let mut tested = NodeFlags::new_exact();
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));

        tested.set_feasible(true);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));

        tested.set_relaxed(true);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags::new_exact();
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));

        tested.set_feasible(true);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));

        tested.set_exact(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));

        tested.set_relaxed(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));

        tested.set_feasible(false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE));
        assert_eq!(false, tested.test(NodeFlags::F_EXACT   | NodeFlags::F_RELAXED    | NodeFlags::F_FEASIBLE));
    }

    #[test]
    fn add_turns_one_or_more_flags_on() {
        let mut tested = NodeFlags(0);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        tested.add(NodeFlags::F_EXACT);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_RELAXED);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_FEASIBLE);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_EXACT | NodeFlags::F_FEASIBLE);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.add(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));
    }
    #[test]
    fn remove_turns_one_or_more_flags_off() {
        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_FEASIBLE));

        tested.remove(NodeFlags::F_EXACT);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.remove(NodeFlags::F_RELAXED);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.remove(NodeFlags::F_FEASIBLE);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.remove(NodeFlags::F_EXACT | NodeFlags::F_FEASIBLE);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.remove(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.remove(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false,  tested.test(NodeFlags::F_FEASIBLE));
    }
    #[test]
    fn set_turns_one_or_more_flags_on_or_off() {
        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true, tested.test(NodeFlags::F_FEASIBLE));

        tested.set(NodeFlags::F_EXACT, false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.set(NodeFlags::F_RELAXED, false);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.set(NodeFlags::F_FEASIBLE, false);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.set(NodeFlags::F_EXACT | NodeFlags::F_FEASIBLE, false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.set(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE, false);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE);
        tested.set(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE, false);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false,  tested.test(NodeFlags::F_FEASIBLE));

        //
        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_EXACT, true);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_RELAXED, true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(false, tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_FEASIBLE, true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_EXACT | NodeFlags::F_FEASIBLE, true);
        assert_eq!(true,  tested.test(NodeFlags::F_EXACT));
        assert_eq!(false, tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE, true);
        assert_eq!(false, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));

        let mut tested = NodeFlags(0);
        tested.set(NodeFlags::F_EXACT | NodeFlags::F_RELAXED | NodeFlags::F_FEASIBLE, true);
        assert_eq!(true, tested.test(NodeFlags::F_EXACT));
        assert_eq!(true,  tested.test(NodeFlags::F_RELAXED));
        assert_eq!(true,  tested.test(NodeFlags::F_FEASIBLE));
    }
    #[test]
    fn by_default_only_the_exact_flag_is_on() {
        assert_eq!(true,  NodeFlags::default().test(NodeFlags::F_EXACT));
        assert_eq!(false, NodeFlags::default().test(NodeFlags::F_RELAXED));
        assert_eq!(false, NodeFlags::default().test(NodeFlags::F_FEASIBLE));
    }
}
