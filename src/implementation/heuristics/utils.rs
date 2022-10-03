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

//! This module provide some convenient utilities to work with used defined heuristics.

use std::cmp::Ordering;

use compare::Compare;

use crate::{StateRanking, SubProblemRanking, SubProblem};


/// This is a thin wrapper to convert a StateRanking into a `Compare` object
/// as is sometimes required. 
/// 
/// This struct has no behavior of its own: it simply delegates to the 
/// underlying implementation.
#[derive(Debug, Clone, Copy)]
pub struct CompareState<X:StateRanking>(X);
impl <X:StateRanking> CompareState<X> {
    /// Creates a new instance
    pub fn new(x: X) -> Self {
        Self(x)
    }
}
impl <X:StateRanking> Compare<X::State> for CompareState<X> {
    fn compare(&self, l: &X::State, r: &X::State) -> Ordering {
        self.0.compare(l, r)
    }
}

/// This is a thin wrapper to convert a SubProblemRanking into a `Compare` 
/// object as is sometimes required (e.g. to configure the order in a binary heap)
/// 
/// This struct has no behavior of its own: it simply delegates to the 
/// underlying implementation.
#[derive(Debug, Clone, Copy)]
pub struct CompareSubProblem<X:SubProblemRanking>(X);
impl <X:SubProblemRanking> CompareSubProblem<X> {
    /// Creates a new instance
    pub fn new(x: X) -> Self {
        Self(x)
    }
}
impl <X:SubProblemRanking> Compare<SubProblem<X::State>> for CompareSubProblem<X> {
    fn compare(&self, l: &SubProblem<X::State>, r: &SubProblem<X::State>) -> Ordering {
        self.0.compare(l, r)
    }
}

#[cfg(test)]
mod test {
    use std::{cmp::Ordering, ops::Deref, sync::Arc};
    use compare::Compare;
    use crate::{StateRanking, SubProblemRanking, SubProblem, CompareState, CompareSubProblem};

    /// A dummy state comparator for use in the tests
    struct CharRanking;
    impl StateRanking for CharRanking {
        type State = char;

        fn compare(&self, a: &Self::State, b: &Self::State) -> Ordering {
            a.cmp(b)
        }
    }
    impl SubProblemRanking for CharRanking {
        type State = char;

        fn compare(&self, a: &SubProblem<char>, b: &SubProblem<char>) -> Ordering {
            <Self as StateRanking>::compare(self, a.state.deref(), b.state.deref())
        }
    }

    #[test]
    fn when_a_is_less_than_b_comparestate_returns_less() {
        let cmp = CompareState::new(CharRanking);
        assert_eq!(cmp.compare(&'a', &'b'), Ordering::Less);
    }
    #[test]
    fn when_a_is_greater_than_b_comparestate_returns_greater() {
        let cmp = CompareState::new(CharRanking);
        assert_eq!(cmp.compare(&'b', &'a'), Ordering::Greater);
    }
    #[test]
    fn when_a_is_equal_to_b_comparestate_returns_equal() {
        let cmp = CompareState::new(CharRanking);
        assert_eq!(cmp.compare(&'a', &'a'), Ordering::Equal);
    }

    #[test]
    fn when_a_is_less_than_b_comparesubproblem_returns_less() {
        let cmp = CompareSubProblem::new(CharRanking);
        assert_eq!(cmp.compare(
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![]}, 
            &SubProblem{state: Arc::new('b'), value: 0, ub: isize::MAX, path: vec![]}), 
            Ordering::Less);
    }
    #[test]
    fn when_a_is_greater_than_b_comparesubproblem_returns_greater() {
        let cmp = CompareSubProblem::new(CharRanking);
        assert_eq!(cmp.compare(
            &SubProblem{state: Arc::new('b'), value: 0, ub: isize::MAX, path: vec![]}, 
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![]}), 
            Ordering::Greater);
    }
    #[test]
    fn when_a_is_equal_to_b_comparesubproblem_returns_equal() {
        let cmp = CompareSubProblem::new(CharRanking);
        assert_eq!(cmp.compare(
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![]}, 
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![]}), 
            Ordering::Equal);
    }
}