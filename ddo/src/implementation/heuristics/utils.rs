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

use crate::{SubProblemRanking, SubProblem};


/// This is a thin wrapper to convert a SubProblemRanking into a `Compare` 
/// object as is sometimes required (e.g. to configure the order in a binary heap)
/// 
/// This struct has no behavior of its own: it simply delegates to the 
/// underlying implementation.
/// 
/// # Example
/// ```
/// # use ddo::*;
/// # use binary_heap_plus::BinaryHeap;
/// 
/// // Assuming the existence of a simple KnapsackState type
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct KnapsackState {
///     depth: usize,
///     capacity: usize
/// }
/// 
/// // One can define a simple StateRanking to compare these states
/// struct KPRanking;
/// impl StateRanking for KPRanking {
///     type State = KnapsackState;
///     
///     fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
///         a.capacity.cmp(&b.capacity)
///     }
/// }
/// 
/// // However, if you were in need to create a heap capable of ordering
/// // the sub-problems with your custom state ranking, you would need to use an
/// // object that implements the `Compare` trait. This is what `CompareSubProblem`
/// // is used for: it provides a convenient zero sized adapter for that
/// // purpose.
/// 
/// // This allows to compare two sub-problems, ordering them in best first order
/// let comparator = CompareSubProblem::new(MaxUB::new(&KPRanking));
/// 
/// // And that comparator can in turn be used to parameterize the behavior
/// // of a heap (for instance).
/// let heap = BinaryHeap::from_vec_cmp(vec![], comparator);
/// ```
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
    use crate::{StateRanking, SubProblemRanking, SubProblem, CompareSubProblem};

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
    fn when_a_is_less_than_b_comparesubproblem_returns_less() {
        let cmp = CompareSubProblem::new(CharRanking);
        assert_eq!(cmp.compare(
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![], depth: 0}, 
            &SubProblem{state: Arc::new('b'), value: 0, ub: isize::MAX, path: vec![], depth: 0}), 
            Ordering::Less);
    }
    #[test]
    fn when_a_is_greater_than_b_comparesubproblem_returns_greater() {
        let cmp = CompareSubProblem::new(CharRanking);
        assert_eq!(cmp.compare(
            &SubProblem{state: Arc::new('b'), value: 0, ub: isize::MAX, path: vec![], depth: 0}, 
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![], depth: 0}), 
            Ordering::Greater);
    }
    #[test]
    fn when_a_is_equal_to_b_comparesubproblem_returns_equal() {
        let cmp = CompareSubProblem::new(CharRanking);
        assert_eq!(cmp.compare(
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![], depth: 0}, 
            &SubProblem{state: Arc::new('a'), value: 0, ub: isize::MAX, path: vec![], depth: 0}), 
            Ordering::Equal);
    }
}