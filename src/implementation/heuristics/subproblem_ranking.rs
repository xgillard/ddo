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

//! This module provides the implementation of subproblem rankings that are used to
//! set the ordering of the solver frontier.

use std::cmp::Ordering;

use crate::{StateRanking, SubProblemRanking, SubProblem};

/// The MaxUB (maximum upper bound) strategy is one that always selects the node
/// having the highest upper bound in the frontier. In case of equalities, the
/// ties are broken using the length of the longest path and eventually a state 
/// ranking.
/// 
/// In practice, MaxUB is implemented as a shim wrapper around a regular 
/// stateranking which eases the reuse of otherwise required code. The comparison
/// is initially made on the upper bound only, then the longest path value, and 
/// it is eventually delegated to the stateranking as a means to break ties when
/// an equality is detected.
///
/// # Example
/// ```
/// # use ddo::*;
/// #
/// let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
/// let b = SubProblem {state: Arc::new('b'), value:  2, ub: 100, path: vec![]};
/// let c = SubProblem {state: Arc::new('c'), value: 24, ub: 150, path: vec![]};
/// let d = SubProblem {state: Arc::new('d'), value: 13, ub:  60, path: vec![]};
/// let e = SubProblem {state: Arc::new('e'), value: 65, ub: 700, path: vec![]};
/// let f = SubProblem {state: Arc::new('f'), value: 19, ub: 100, path: vec![]};
///
/// let mut priority_q = SimpleFrontier::new_with_order(MaxUB);
/// priority_q.push(a);
/// priority_q.push(b);
/// priority_q.push(c);
/// priority_q.push(d);
/// priority_q.push(e);
/// priority_q.push(f);
///
/// assert_eq!('e', *priority_q.pop().unwrap().state); // because 700 is the highest upper bound
/// assert_eq!('a', *priority_q.pop().unwrap().state); // because 300 is the next highest
/// assert_eq!('c', *priority_q.pop().unwrap().state); // idem, because of ub = 150
/// assert_eq!('f', *priority_q.pop().unwrap().state); // because ub = 100 but value = 19
/// assert_eq!('b', *priority_q.pop().unwrap().state); // because ub = 100 but value = 2
/// assert_eq!('d', *priority_q.pop().unwrap().state); // because ub = 13 which is the worst
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MaxUB<'a, O: StateRanking>(&'a O);
impl <'a, O: StateRanking> MaxUB<'a, O> {
    /// Creates a new instance
    pub fn new(x: &'a O) -> Self {
        Self(x)
    }
}
impl<O: StateRanking> SubProblemRanking for MaxUB<'_, O> {
    type State = O::State;

    fn compare(&self, l: &SubProblem<O::State>, r: &SubProblem<O::State>) -> Ordering {
        l.ub.cmp(&r.ub)
            .then_with(|| l.value.cmp(&r.value))
            .then_with(|| self.0.compare(&l.state, &r.state))
    }
}


#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_maxub {
    use std::cmp::Ordering;
    use std::sync::Arc;

    use binary_heap_plus::BinaryHeap;

    use crate::*;
    use crate::implementation::heuristics::subproblem_ranking::MaxUB;

    /// A dummy state comparator for use in the tests
    struct CharRanking;
    impl StateRanking for CharRanking {
        type State = char;

        fn compare(&self, a: &Self::State, b: &Self::State) -> Ordering {
            a.cmp(b)
        }
    }

    #[test]
    fn example() {
        let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
        let b = SubProblem {state: Arc::new('b'), value:  2, ub: 100, path: vec![]};
        let c = SubProblem {state: Arc::new('c'), value: 24, ub: 150, path: vec![]};
        let d = SubProblem {state: Arc::new('d'), value: 13, ub:  60, path: vec![]};
        let e = SubProblem {state: Arc::new('e'), value: 65, ub: 700, path: vec![]};
        let f = SubProblem {state: Arc::new('f'), value: 19, ub: 100, path: vec![]};

        let nodes = vec![a, b, c, d, e, f];
        let mut priority_q = BinaryHeap::from_vec_cmp(nodes, CompareSubProblem::new(MaxUB::new(&CharRanking)));

        assert_eq!('e', *priority_q.pop().unwrap().state); // because 700 is the highest upper bound
        assert_eq!('a', *priority_q.pop().unwrap().state); // because 300 is the next highest
        assert_eq!('c', *priority_q.pop().unwrap().state); // idem, because of ub = 150
        assert_eq!('f', *priority_q.pop().unwrap().state); // because ub = 100 but value = 19
        assert_eq!('b', *priority_q.pop().unwrap().state); // because ub = 100 but value = 2
        assert_eq!('d', *priority_q.pop().unwrap().state); // because ub = 13 which is the worst
    }

    #[test]
    fn gt_because_ub() {
        let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
        let b = SubProblem {state: Arc::new('b'), value: 42, ub: 100, path: vec![]};
        let cmp = MaxUB::new(&CharRanking);
        assert_eq!(Ordering::Greater, cmp.compare(&a, &b));
    }
    #[test]
    fn gt_because_lplen() {
        let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
        let b = SubProblem {state: Arc::new('b'), value:  2, ub: 300, path: vec![]};
        let cmp = MaxUB::new(&CharRanking);
        assert_eq!(Ordering::Greater, cmp.compare(&a, &b));
    }
    #[test]
    fn lt_because_ub() {
        let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
        let b = SubProblem {state: Arc::new('b'), value: 42, ub: 100, path: vec![]};
        let cmp = MaxUB::new(&CharRanking);
        assert_eq!(Ordering::Less, cmp.compare(&b, &a));
    }
    #[test]
    fn lt_because_lplen() {
        let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
        let b = SubProblem {state: Arc::new('b'), value:  2, ub: 300, path: vec![]};
        let cmp = MaxUB::new(&CharRanking);
        assert_eq!(Ordering::Less, cmp.compare(&b, &a));
    }
    #[test]
    fn lt_because_state() {
        let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
        let b = SubProblem {state: Arc::new('b'), value: 42, ub: 300, path: vec![]};
        let cmp = MaxUB::new(&CharRanking);
        assert_eq!(Ordering::Less, cmp.compare(&a, &b));
    }
    #[test]
    fn eq_self() {
        let a = SubProblem {state: Arc::new('a'), value: 42, ub: 300, path: vec![]};
        let cmp = MaxUB::new(&CharRanking);
        assert_eq!(Ordering::Equal, cmp.compare(&a, &a));
    }
}

