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

//! This module provides the implementation of usual frontiers.

use binary_heap_plus::BinaryHeap;
use crate::common::FrontierNode;
use crate::implementation::heuristics::MaxUB;
use crate::implementation::utils::NoDupHeap;
use crate::abstraction::frontier::Frontier;
use std::hash::Hash;
use metrohash::MetroHashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use crate::FrontierOrder;
use compare::Compare;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// The simplest frontier implementation you can think of: is basically consists
/// of a binary heap that pushes and pops frontier nodes
/// 
/// # Note
/// This is the default type of frontier for both sequential and parallel 
/// solvers. Hence, you don't need to take any action in order to use the
/// `SimpleFrontier`.
/// 
pub struct SimpleFrontier<T, O: FrontierOrder<T>=MaxUB> {
    heap: BinaryHeap<FrontierNode<T>, FrontierOrder2Compare<T, O>>
}
impl <T> SimpleFrontier<T> {
    /// This creates a new simple frontier which uses the default frontier order
    /// (MaxUB).
    fn new() -> Self {
        SimpleFrontier::new_with_order(MaxUB)
    }
}
impl <T, O> SimpleFrontier<T, O> where O: FrontierOrder<T> {
    /// This creates a new simple frontier which uses a custiom frontier order.
    pub fn new_with_order(o: O) -> Self {
        Self{ heap: BinaryHeap::from_vec_cmp(vec![], FrontierOrder2Compare::new(o)) }
    }
}
impl <T> Default for SimpleFrontier<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl <T, O> Frontier<T> for SimpleFrontier<T, O> where O: FrontierOrder<T> {
    fn push(&mut self, node: FrontierNode<T>) {
        self.heap.push(node)
    }

    fn pop(&mut self) -> Option<FrontierNode<T>> {
        self.heap.pop()
    }

    fn clear(&mut self) {
        self.heap.clear()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}
/// This is an adapter that converts any `FrontierOrder<T>` into a
/// `Compare<FrontierNode<T>>`. This is a very trivial implementation which
/// simply forwards to the desired frontier order. The point of this structure
/// is to serve as an adapter between the frontier order and the binary heap
/// requirements.
pub struct FrontierOrder2Compare<T, X: FrontierOrder<T>> {
    cmp     : X,
    _phantom: PhantomData<T>
}
impl <T, X: FrontierOrder<T>> FrontierOrder2Compare<T,X> {
    pub fn new(cmp: X) -> Self {
        Self{cmp, _phantom: Default::default()}
    }
}
impl <T, X: FrontierOrder<T>> Compare<FrontierNode<T>> for FrontierOrder2Compare<T,X> {
    #[inline]
    fn compare(&self, a: &FrontierNode<T>, b: &FrontierNode<T>) -> Ordering {
        self.cmp.compare(a, b)
    }
}

/// A frontier that enforces the requirement that a given state will never be
/// present twice in the frontier.
/// 
/// # Warning
/// This kind of frontier does a great job to reduce thrashing during problem
/// optimization. However, it is only safe to use when any given state belongs
/// to *at most ONE* layer from the exact MDD. (In other words, a state uniquely
/// identifies a sub-problem).
/// 
/// # Example
/// ```
/// # use ddo::*;
/// #
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> isize   { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         (0..=1).into()
/// #     }
/// #     fn transition(&self, state: &usize, _: &VarSet, _: Decision) -> usize {
/// #         41
/// #     }
/// #     fn transition_cost(&self, state: &usize, _: &VarSet, _: Decision) -> isize {
/// #         42
/// #     }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #     fn merge_states(&self, n: &mut dyn Iterator<Item=&usize>) -> usize {
/// #         *n.next().unwrap()
/// #     }
/// #     fn relax_edge(&self, _src: &usize, _dst: &usize, _rlx: &usize, _d: Decision, cost: isize) -> isize {
/// #        cost
/// #     }
/// # }
/// # let problem = MockProblem;
/// # let relax   = MockRelax;
/// let mdd = mdd_builder(&problem, relax).into_deep();
/// let mut solver = ParallelSolver::new(mdd)
///     .with_frontier(NoDupFrontier::default());
/// let optimum = solver.maximize(); // will run for maximum 10 seconds
/// ```
#[derive(Clone)]
pub struct NoDupFrontier<T, O: FrontierOrder<T>=MaxUB> where T: Eq + Hash + Clone {
    heap: NoDupHeap<T, O>
}
impl <T> NoDupFrontier<T> where T: Eq + Hash + Clone {
    /// This creates a new no dup frontier which uses the default frontier order
    /// (MaxUB).
    pub fn new() -> Self {
        NoDupFrontier::new_with_order(MaxUB)
    }
}
impl <T, O> NoDupFrontier<T, O> where T: Eq+Hash+Clone, O: FrontierOrder<T> {
    /// This creates a new no dup frontier which uses a custom frontier order.
    pub fn new_with_order(o: O) -> Self {
        Self{ heap: NoDupHeap::new(o) }
    }
}
impl <T> Default for NoDupFrontier<T> where T: Eq + Hash + Clone {
    fn default() -> Self {
        Self::new()
    }
}
impl <T, O> Frontier<T> for NoDupFrontier<T, O> where T: Eq + Hash + Clone, O: FrontierOrder<T> {
    fn push(&mut self, node: FrontierNode<T>) {
        self.heap.push(node)
    }

    fn pop(&mut self) -> Option<FrontierNode<T>> {
        self.heap.pop()
    }

    fn clear(&mut self) {
        self.heap.clear()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}

/// A frontier that enforces the requirement that a given state will never be
/// enqueued again, unless it improves the longest path length.
/// 
/// # Note: 
/// The difference with `NoDupFrontier` stems from the fact that a 
/// `NoForgetFrontier` *never* forgets the states it has encountered while the
/// `NoDupFrontier` keeps track of those nodes in the queue only.
/// 
/// # Warning
/// This kind of frontier does a great job to reduce thrashing during problem
/// optimization. However, it is only safe to use when any given state belongs
/// to *at most ONE* layer from the exact MDD. (In other words, a state uniquely
/// identifies a sub-problem).
/// 
/// # Example
/// ```
/// # use ddo::*;
/// #
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> isize   { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         (0..=1).into()
/// #     }
/// #     fn transition(&self, state: &usize, _: &VarSet, _: Decision) -> usize {
/// #         41
/// #     }
/// #     fn transition_cost(&self, state: &usize, _: &VarSet, _: Decision) -> isize {
/// #         42
/// #     }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #     fn merge_states(&self, n: &mut dyn Iterator<Item=&usize>) -> usize {
/// #         *n.next().unwrap()
/// #     }
/// #     fn relax_edge(&self, _src: &usize, _dst: &usize, _rlx: &usize, _d: Decision, cost: isize) -> isize {
/// #        cost
/// #     }
/// # }
/// # let problem = MockProblem;
/// # let relax   = MockRelax;
/// let mdd = mdd_builder(&problem, relax).into_deep();
/// let mut solver = ParallelSolver::new(mdd)
///     .with_frontier(NoForgetFrontier::default());
/// let optimum = solver.maximize(); // will run for maximum 10 seconds
/// ```
#[derive(Clone)]
pub struct NoForgetFrontier<T, O=MaxUB> where T: Eq + Hash + Clone, O: FrontierOrder<T> {
    /// The frontier itself
    heap: NoDupHeap<T, O>,
    /// The collection of enqueued states with their respective longest path length
    states: MetroHashMap<Arc<T>,isize>
}
impl <T> NoForgetFrontier<T, MaxUB> where T: Eq + Hash + Clone {
    pub fn new() -> Self {
        NoForgetFrontier::new_with_order(MaxUB)
    }
}
impl <T, O>  NoForgetFrontier<T, O> where T: Eq + Hash + Clone, O: FrontierOrder<T> {
    pub fn new_with_order(ord: O) -> Self {
        Self{
            heap   : NoDupHeap::new(ord),
            states : MetroHashMap::default()
        }
    }
}
impl <T> Default for NoForgetFrontier<T, MaxUB> where T: Eq + Hash + Clone {
    fn default() -> Self {
        Self::new()
    }
}
impl <T, O> Frontier<T> for NoForgetFrontier<T, O>
    where T: Eq + Hash + Clone, O: FrontierOrder<T> {
    fn push(&mut self, node: FrontierNode<T>) {
        match self.states.entry(Arc::clone(&node.state)) {
            Entry::Vacant(e) => {
                e.insert(node.lp_len);
                self.heap.push(node);
            },
            Entry::Occupied(mut e) => {
                let lp_len = e.get_mut();
                if node.lp_len > *lp_len {
                    *lp_len = node.lp_len;
                    self.heap.push(node);
                }
            }
        }
    }

    fn pop(&mut self) -> Option<FrontierNode<T>> {
        self.heap.pop()
    }

    fn clear(&mut self) {
        self.heap.clear();
        self.states.clear();
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_simple_frontier {
    use crate::{SimpleFrontier, Frontier, FrontierNode, PartialAssignment};
    use std::sync::Arc;

    // by default, it is empty
    #[test]
    fn by_default_it_is_empty() {
        assert!(SimpleFrontier::<usize>::default().is_empty())
    }

    // when the size is zero, then it is empty
    #[test]
    fn when_the_size_is_zero_then_it_is_empty() {
        let frontier = empty_frontier();
        assert_eq!(frontier.len(), 0);
        assert!(frontier.is_empty());
    } 
    
    // when the size is greater than zero, it it not empty
    #[test]
    fn when_the_size_is_greater_than_zero_it_is_not_empty() {
        let frontier = non_empty_frontier();
        assert_eq!(frontier.len(), 1);
        assert!(!frontier.is_empty());
    }

    // when I push a node onto the frontier then the length increases
    #[test]
    fn when_i_push_a_node_onto_the_frontier_then_the_length_increases() {
        let mut frontier = empty_frontier();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 1);
        frontier.push(FrontierNode{
            state: Arc::new(43),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 2);
    }
    // when I pop a node off the frontier then the length decreases
    #[test]
    fn when_i_pop_a_node_off_the_frontier_then_the_length_decreases() {
        let mut frontier = non_empty_frontier();
        assert_eq!(frontier.len(), 1);
        frontier.pop();
        assert_eq!(frontier.len(), 0);
    }

    // when I try to pop a node off an empty frontier, I get none
    #[test]
    fn when_i_try_to_pop_a_node_off_an_empty_frontier_i_get_none() {
        let mut frontier = empty_frontier();
        assert_eq!(frontier.pop(), None);
    }

    // when I pop a node, it is always the one with the largest ub (then lp_len)
    #[test]
    fn when_i_pop_a_node_it_is_always_the_one_with_the_largest_ub_then_lp() {
        let mut frontier = empty_frontier();
        let a = FrontierNode{
            state: Arc::new(1),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 1,
            ub: 1
        };
        let b = FrontierNode{
            state: Arc::new(2),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 2,
            ub: 2
        };
        let c = FrontierNode{
            state: Arc::new(3),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 3,
            ub: 3
        };
        let d = FrontierNode{
            state: Arc::new(4),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 4,
            ub: 4
        };
        let e = FrontierNode{
            state: Arc::new(5),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 4,
            ub: 5
        };
        let f = FrontierNode{
            state: Arc::new(5),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 5,
            ub: 5
        };

        frontier.push(a.clone());
        frontier.push(f.clone());
        frontier.push(b.clone());
        frontier.push(d.clone());
        frontier.push(c.clone());
        frontier.push(e.clone());

        assert_eq!(frontier.pop(), Some(f));
        assert_eq!(frontier.pop(), Some(e));
        assert_eq!(frontier.pop(), Some(d));
        assert_eq!(frontier.pop(), Some(c));
        assert_eq!(frontier.pop(), Some(b));
        assert_eq!(frontier.pop(), Some(a));
    }
    // when I clear an empty frontier, it remains empty
    #[test]
    fn when_i_clear_an_empty_frontier_it_remains_empty() {
        let mut frontier = empty_frontier();
        assert!(frontier.is_empty());
        frontier.clear();
        assert!(frontier.is_empty());
    }
    // when I clear a non empty frontier it becomes empty
    #[test]
    fn when_i_clear_a_non_empty_frontier_it_becomes_empty() {
        let mut frontier = non_empty_frontier();
        assert!(!frontier.is_empty());
        frontier.clear();
        assert!(frontier.is_empty());
    }

    fn empty_frontier() -> SimpleFrontier<usize> {
        SimpleFrontier::default()
    }
    fn non_empty_frontier() -> SimpleFrontier<usize> {
        let mut frontier = SimpleFrontier::default();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        frontier
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_no_dup_frontier {
    use crate::*;
    use std::sync::Arc;

    // by default, it is empty
    #[test]
    fn by_default_it_is_empty() {
        assert!(NoDupFrontier::<usize>::default().is_empty())
    }

    // when the size is zero, then it is empty
    #[test]
    fn when_the_size_is_zero_then_it_is_empty() {
        let frontier = empty_frontier();
        assert_eq!(frontier.len(), 0);
        assert!(frontier.is_empty());
    }

    // when the size is greater than zero, it it not empty
    #[test]
    fn when_the_size_is_greater_than_zero_it_is_not_empty() {
        let frontier = non_empty_frontier();
        assert_eq!(frontier.len(), 1);
        assert!(!frontier.is_empty());
    }

    // when I push a non existing node onto the frontier then the length increases
    #[test]
    fn when_i_push_a_non_existing_node_onto_the_frontier_then_the_length_increases() {
        let mut frontier = empty_frontier();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 1);
        frontier.push(FrontierNode{
            state: Arc::new(43),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 2);
    }
    // when I push an existing node onto the frontier then the length wont increases
    #[test]
    fn when_i_push_an_existing_node_onto_the_frontier_then_the_length_does_not_increases() {
        let mut frontier = empty_frontier();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 1);
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 12,
            ub: 5
        });
        assert_eq!(frontier.len(), 1);
    }
    // when I pop a node off the frontier then the length decreases
    #[test]
    fn when_i_pop_a_node_off_the_frontier_then_the_length_decreases() {
        let mut frontier = non_empty_frontier();
        assert_eq!(frontier.len(), 1);
        frontier.pop();
        assert_eq!(frontier.len(), 0);
    }

    // when I try to pop a node off an empty frontier, I get none
    #[test]
    fn when_i_try_to_pop_a_node_off_an_empty_frontier_i_get_none() {
        let mut frontier = empty_frontier();
        assert_eq!(frontier.pop(), None);
    }

    // when I pop a node, it is always the one with the largest ub (then lp_len)
    #[test]
    fn when_i_pop_a_node_it_is_always_the_one_with_the_largest_ub_then_lp() {
        let mut frontier = empty_frontier();
        let a = FrontierNode{
            state: Arc::new(1),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 1,
            ub: 1
        };
        let b = FrontierNode{
            state: Arc::new(2),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 2,
            ub: 2
        };
        let c = FrontierNode{
            state: Arc::new(3),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 3,
            ub: 3
        };
        let d = FrontierNode{
            state: Arc::new(4),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 4,
            ub: 4
        };
        let e = FrontierNode{
            state: Arc::new(5),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 4,
            ub: 5
        };
        let f = FrontierNode{
            state: Arc::new(5),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 5,
            ub: 5
        };

        frontier.push(a.clone());
        frontier.push(f.clone());
        frontier.push(b.clone());
        frontier.push(d.clone());
        frontier.push(c.clone());
        frontier.push(e);

        assert_eq!(frontier.pop(), Some(f));
        //assert_eq!(frontier.pop(), Some(e)); // node 'e' will never show up
        assert_eq!(frontier.pop(), Some(d));
        assert_eq!(frontier.pop(), Some(c));
        assert_eq!(frontier.pop(), Some(b));
        assert_eq!(frontier.pop(), Some(a));
    }

    // when I pop a node off the frontier, for which multiple copies have been added
    // I retrieve the one with the longest path
    #[test]
    fn when_i_pop_a_node_off_the_frontier_for_which_multiple_copies_have_been_added_then_i_retrieve_the_one_with_longest_path(){
        let pe = Arc::new(PartialAssignment::SingleExtension {
            decision: Decision {variable: Variable(0), value: 4},
            parent: Arc::new(PartialAssignment::Empty) });

        let pf = Arc::new(PartialAssignment::SingleExtension {
            decision: Decision {variable: Variable(1), value: 5},
            parent: Arc::new(PartialAssignment::Empty) });

        let ne = FrontierNode{
            state: Arc::new(5),
            path: Arc::clone(&pe),
            lp_len: 4,
            ub: 5
        };
        let nf = FrontierNode{
            state: Arc::new(5),
            path: Arc::clone(&pf),
            lp_len: 5,
            ub: 5
        };

        let mut frontier = empty_frontier();
        frontier.push(ne);
        frontier.push(nf.clone());

        assert_eq!(frontier.pop(), Some(nf));
    }

    // when I clear an empty frontier, it remains empty
    #[test]
    fn when_i_clear_an_empty_frontier_it_remains_empty() {
        let mut frontier = empty_frontier();
        assert!(frontier.is_empty());
        frontier.clear();
        assert!(frontier.is_empty());
    }
    // when I clear a non empty frontier it becomes empty
    #[test]
    fn when_i_clear_a_non_empty_frontier_it_becomes_empty() {
        let mut frontier = non_empty_frontier();
        assert!(!frontier.is_empty());
        frontier.clear();
        assert!(frontier.is_empty());
    }


    fn empty_frontier() -> NoDupFrontier<usize> {
        NoDupFrontier::default()
    }
    fn non_empty_frontier() -> NoDupFrontier<usize> {
        let mut frontier = NoDupFrontier::default();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        frontier
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_no_forget_frontier {
    use crate::*;
    use std::sync::Arc;

    // by default, it is empty
    #[test]
    fn by_default_it_is_empty() {
        assert!(NoForgetFrontier::<usize>::default().is_empty())
    }

    // when the size is zero, then it is empty
    #[test]
    fn when_the_size_is_zero_then_it_is_empty() {
        let frontier = empty_frontier();
        assert_eq!(frontier.len(), 0);
        assert!(frontier.is_empty());
    }

    // when the size is greater than zero, it it not empty
    #[test]
    fn when_the_size_is_greater_than_zero_it_is_not_empty() {
        let frontier = non_empty_frontier();
        assert_eq!(frontier.len(), 1);
        assert!(!frontier.is_empty());
    }

    // when I push a non existing node onto the frontier then the length increases
    #[test]
    fn when_i_push_a_non_existing_node_onto_the_frontier_then_the_length_increases() {
        let mut frontier = empty_frontier();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 1);
        frontier.push(FrontierNode{
            state: Arc::new(43),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 2);
    }
    // when I push an existing node onto the frontier then the length wont increases
    #[test]
    fn when_i_push_an_existing_node_onto_the_frontier_then_the_length_does_not_increases() {
        let mut frontier = empty_frontier();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        assert_eq!(frontier.len(), 1);
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 12,
            ub: 5
        });
        assert_eq!(frontier.len(), 1);
    }
    // when I push a node which state was already visited while improving lp_len onto the frontier then the length wont increases
    #[test]
    fn when_i_push_a_better_visited_node_onto_the_frontier_then_the_length_increases() {
        let mut frontier = empty_frontier();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 10,
            ub: 0
        });
        assert_eq!(frontier.len(), 1);
        frontier.pop();
        assert_eq!(frontier.len(), 0);
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 12,
            ub: 5
        });
        assert_eq!(frontier.len(), 1);
    }
    // when I push a node which state was already visited while not improving lp_len onto the frontier then the length wont increases
    #[test]
    fn when_i_push_a_worse_visited_node_onto_the_frontier_then_the_length_does_not_increases() {
        let mut frontier = empty_frontier();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 15,
            ub: 0
        });
        assert_eq!(frontier.len(), 1);
        frontier.pop();
        assert_eq!(frontier.len(), 0);
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 12,
            ub: 5
        });
        assert_eq!(frontier.len(), 0);
    }
    // when I pop a node off the frontier then the length decreases
    #[test]
    fn when_i_pop_a_node_off_the_frontier_then_the_length_decreases() {
        let mut frontier = non_empty_frontier();
        assert_eq!(frontier.len(), 1);
        frontier.pop();
        assert_eq!(frontier.len(), 0);
    }

    // when I try to pop a node off an empty frontier, I get none
    #[test]
    fn when_i_try_to_pop_a_node_off_an_empty_frontier_i_get_none() {
        let mut frontier = empty_frontier();
        assert_eq!(frontier.pop(), None);
    }

    // when I pop a node, it is always the one with the largest ub (then lp_len)
    #[test]
    fn when_i_pop_a_node_it_is_always_the_one_with_the_largest_ub_then_lp() {
        let mut frontier = empty_frontier();
        let a = FrontierNode{
            state: Arc::new(1),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 1,
            ub: 1
        };
        let b = FrontierNode{
            state: Arc::new(2),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 2,
            ub: 2
        };
        let c = FrontierNode{
            state: Arc::new(3),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 3,
            ub: 3
        };
        let d = FrontierNode{
            state: Arc::new(4),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 4,
            ub: 4
        };
        let e = FrontierNode{
            state: Arc::new(5),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 4,
            ub: 5
        };
        let f = FrontierNode{
            state: Arc::new(5),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 5,
            ub: 5
        };

        frontier.push(a.clone());
        frontier.push(f.clone());
        frontier.push(b.clone());
        frontier.push(d.clone());
        frontier.push(c.clone());
        frontier.push(e);

        assert_eq!(frontier.pop(), Some(f));
        //assert_eq!(frontier.pop(), Some(e)); // node 'e' will never show up
        assert_eq!(frontier.pop(), Some(d));
        assert_eq!(frontier.pop(), Some(c));
        assert_eq!(frontier.pop(), Some(b));
        assert_eq!(frontier.pop(), Some(a));
    }

    // when I pop a node off the frontier, for which multiple copies have been added
    // I retrieve the one with the longest path
    #[test]
    fn when_i_pop_a_node_off_the_frontier_for_which_multiple_copies_have_been_added_then_i_retrieve_the_one_with_longest_path(){
        let pe = Arc::new(PartialAssignment::SingleExtension {
            decision: Decision {variable: Variable(0), value: 4},
            parent: Arc::new(PartialAssignment::Empty) });

        let pf = Arc::new(PartialAssignment::SingleExtension {
            decision: Decision {variable: Variable(1), value: 5},
            parent: Arc::new(PartialAssignment::Empty) });

        let ne = FrontierNode{
            state: Arc::new(5),
            path: Arc::clone(&pe),
            lp_len: 4,
            ub: 5
        };
        let nf = FrontierNode{
            state: Arc::new(5),
            path: Arc::clone(&pf),
            lp_len: 5,
            ub: 5
        };

        let mut frontier = empty_frontier();
        frontier.push(ne);
        frontier.push(nf.clone());

        assert_eq!(frontier.pop(), Some(nf));
    }

    // when I clear an empty frontier, it remains empty
    #[test]
    fn when_i_clear_an_empty_frontier_it_remains_empty() {
        let mut frontier = empty_frontier();
        assert!(frontier.is_empty());
        frontier.clear();
        assert!(frontier.is_empty());
    }
    // when I clear a non empty frontier it becomes empty
    #[test]
    fn when_i_clear_a_non_empty_frontier_it_becomes_empty() {
        let mut frontier = non_empty_frontier();
        assert!(!frontier.is_empty());
        frontier.clear();
        assert!(frontier.is_empty());
    }


    fn empty_frontier() -> NoForgetFrontier<usize> {
        NoForgetFrontier::default()
    }
    fn non_empty_frontier() -> NoForgetFrontier<usize> {
        let mut frontier = NoForgetFrontier::default();
        frontier.push(FrontierNode{
            state: Arc::new(42),
            path: Arc::new(PartialAssignment::Empty),
            lp_len: 0,
            ub: 0
        });
        frontier
    }
}
