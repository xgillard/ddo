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

//! This module provides the implementation of a simple solver fringe (priority queue)

use binary_heap_plus::BinaryHeap;

use crate::*;


/// The simplest fringe implementation you can think of: is basically consists
/// of a binary heap that pushes and pops fringe nodes
/// 
/// # Note
/// This is the default type of fringe for both sequential and parallel 
/// solvers. Hence, you don't need to take any action in order to use the
/// `SimpleFringe`.
/// 
pub struct SimpleFringe<O: SubProblemRanking> {
    heap: BinaryHeap<SubProblem<O::State>, CompareSubProblem<O>>
}
impl <O> SimpleFringe<O> where O: SubProblemRanking {
    /// This creates a new simple fringe which uses a custom fringe order.
    pub fn new(o: O) -> Self {
        Self{ heap: BinaryHeap::from_vec_cmp(vec![], CompareSubProblem::new(o)) }
    }
}
impl <O> Fringe for SimpleFringe<O> where O: SubProblemRanking {
    type State = O::State;
    
    fn push(&mut self, node: SubProblem<Self::State>) {
        self.heap.push(node)
    }

    fn pop(&mut self) -> Option<SubProblem<Self::State>> {
        self.heap.pop()
    }

    fn clear(&mut self) {
        self.heap.clear()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}


#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_simple_fringe {
    use crate::*;
    use std::{sync::Arc, cmp::Ordering, ops::Deref};

    /// A dummy state comparator for use in the tests
    struct CharRanking;
    impl StateRanking for CharRanking {
        type State = char;

        fn compare(&self, a: &Self::State, b: &Self::State) -> Ordering {
            a.cmp(b)
        }
    }

    // by default, it is empty
    #[test]
    fn by_default_it_is_empty() {
        let order = MaxUB::new(&CharRanking); 
        let front = SimpleFringe::new(order);
        assert!(front.is_empty())
    }

    // when the size is zero, then it is empty
    #[test]
    fn when_the_size_is_zero_then_it_is_empty() {
        let order = MaxUB::new(&CharRanking); 
        let fringe = SimpleFringe::new(order);
        assert_eq!(fringe.len(), 0);
        assert!(fringe.is_empty());
    } 
    
    // when the size is greater than zero, it it not empty
    #[test]
    fn when_the_size_is_greater_than_zero_it_is_not_empty() {
        let order = MaxUB::new(&CharRanking); 
        let mut fringe = SimpleFringe::new(order);
        fringe.push(SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 10,
            path : vec![]
        });
        assert_eq!(fringe.len(), 1);
        assert!(!fringe.is_empty());
    }

    // when I push a node onto the fringe then the length increases
    #[test]
    fn when_i_push_a_node_onto_the_fringe_then_the_length_increases() {
        let order = MaxUB::new(&CharRanking); 
        let mut fringe = SimpleFringe::new(order);
        fringe.push(SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 10,
            path : vec![]
        });
        fringe.push(SubProblem {
            state: Arc::new('b'),
            value: 20,
            ub   : 20,
            path : vec![]
        });

        assert_eq!(fringe.len(), 2);
    }
    // when I pop a node off the fringe then the length decreases
    #[test]
    fn when_i_pop_a_node_off_the_fringe_then_the_length_decreases() {
        let order = MaxUB::new(&CharRanking); 
        let mut fringe = SimpleFringe::new(order);
        fringe.push(SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 10,
            path : vec![]
        });
        fringe.push(SubProblem {
            state: Arc::new('b'),
            value: 20,
            ub   : 20,
            path : vec![]
        });

        assert_eq!(fringe.len(), 2);
        fringe.pop();
        assert_eq!(fringe.len(), 1);
        fringe.pop();
        assert_eq!(fringe.len(), 0);
    }

    // when I try to pop a node off an empty fringe, I get none
    #[test]
    fn when_i_try_to_pop_a_node_off_an_empty_fringe_i_get_none() {
        let order = MaxUB::new(&CharRanking); 
        let mut fringe = SimpleFringe::new(order);
        assert!(fringe.pop().is_none());
    }

    // when I pop a node, it is always the one with the largest ub (then lp_len)
    #[test]
    fn when_i_pop_a_node_it_is_always_the_one_with_the_largest_ub_then_lp() {
        let order = MaxUB::new(&CharRanking); 
        let mut fringe = SimpleFringe::new(order);
        fringe.push(SubProblem {
            state: Arc::new('a'),
            value: 1,
            ub   : 1,
            path : vec![]
        });
        fringe.push(SubProblem {
            state: Arc::new('b'),
            value: 2,
            ub   : 2,
            path : vec![]
        });
        fringe.push(SubProblem {
            state: Arc::new('c'),
            value: 3,
            ub   : 3,
            path : vec![]
        });
        fringe.push(SubProblem {
            state: Arc::new('d'),
            value: 4,
            ub   : 4,
            path : vec![]
        });
        fringe.push(SubProblem {
            state: Arc::new('e'),
            value: 4,
            ub   : 5,
            path : vec![]
        });
        fringe.push(SubProblem {
            state: Arc::new('f'),
            value: 5,
            ub   : 5,
            path : vec![]
        });
        
        assert_eq!(fringe.pop().unwrap().state.deref(), &'f');
        assert_eq!(fringe.pop().unwrap().state.deref(), &'e');
        assert_eq!(fringe.pop().unwrap().state.deref(), &'d');
        assert_eq!(fringe.pop().unwrap().state.deref(), &'c');
        assert_eq!(fringe.pop().unwrap().state.deref(), &'b');
        assert_eq!(fringe.pop().unwrap().state.deref(), &'a');
    }
    // when I clear an empty fringe, it remains empty
    #[test]
    fn when_i_clear_an_empty_fringe_it_remains_empty() {
        let order = MaxUB::new(&CharRanking); 
        let mut fringe = SimpleFringe::new(order);
        assert!(fringe.is_empty());
        fringe.clear();
        assert!(fringe.is_empty());
    }
    // when I clear a non empty fringe it becomes empty
    #[test]
    fn when_i_clear_a_non_empty_fringe_it_becomes_empty() {
        let order = MaxUB::new(&CharRanking); 
        let mut fringe = SimpleFringe::new(order);
        fringe.push(SubProblem {
            state: Arc::new('f'),
            value: 5,
            ub   : 5,
            path : vec![]
        });

        assert!(!fringe.is_empty());
        fringe.clear();
        assert!(fringe.is_empty());
    }
}
