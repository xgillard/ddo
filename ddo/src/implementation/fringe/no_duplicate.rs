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

//! This module provides the implementation of a solver fringe that forbids the
//! co-occurence of two subproblems having the same root state.

use std::{hash::Hash, sync::Arc};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};

use compare::Compare;
use rustc_hash::FxHashMap;

use crate::*;
use self::Action::{BubbleDown, BubbleUp, DoNothing};

/// This is a type-safe identifier for some node in the queue.
/// Basically, this NodeId equates to the position of the identified
/// node in the `nodes` list from the `NoDupHeap`.
#[derive(Debug, Copy, Clone)]
struct NodeId(usize);

/// An enum to know what needs to be done with a given node id
#[derive(Debug, Copy, Clone)]
enum Action {
    DoNothing,
    BubbleUp(NodeId),
    BubbleDown(NodeId),
}

/// This is an updatable binary heap backed by a vector which ensures that
/// items remain ordered in the priority queue while guaranteeing that a
/// given state will only ever be present *ONCE* in the priority queue (the
/// node with the longest path to state is the only kept copy).
pub struct NoDupFringe<O>
where
    O: SubProblemRanking,
    O::State: Eq + Hash + Clone,
{
    /// This is the comparator used to order the nodes in the binary heap
    cmp: CompareSubProblem<O>,
    /// A mapping that associates some state to a node identifier.
    states: FxHashMap<Arc<O::State>, NodeId>,
    /// The actual payload (nodes) ordered in the list
    nodes: Vec<SubProblem<O::State>>,
    /// The position of the items in the heap
    pos: Vec<usize>,
    /// This is the actual heap which orders nodes.
    heap: Vec<NodeId>,
    /// The positions in the `nodes` vector that can be recycled.
    recycle_bin: Vec<NodeId>,
}

impl<O> Fringe for NoDupFringe<O>
where
    O: SubProblemRanking,
    O::State: Eq + Hash + Clone,
{
    type State = O::State;

    /// Pushes one node onto the heap while ensuring that only one copy of the
    /// node (identified by its state) is kept in the heap.
    ///
    /// # Note:
    /// In the event where the heap already contains a copy `x` of a node having
    /// the same state as the `node` being pushed. The priority of the node
    /// left in the heap might be affected. If `node` node is "better" (greater
    /// UB and or longer longest path), the priority of the node will be
    /// increased. As always, in the event where the newly pushed node has a
    /// longer longest path than the pre-existing node, that one will be kept.
    fn push(&mut self, mut node: SubProblem<O::State>) {
        let state = Arc::clone(&node.state);

        let action = match self.states.entry(state) {
            Occupied(e) => {
                let id = *e.get();

                // info about the pre-existing node
                let old_lp = self.nodes[id.0].value;
                let old_ub = self.nodes[id.0].ub;
                // info about the new node
                let new_lp = node.value;
                let new_ub = node.ub;
                // make sure that ub is the max of the known ubs
                node.ub = new_ub.max(old_ub);

                let action = if self.cmp.compare(&node, &self.nodes[id.0]) == Greater {
                    BubbleUp(id)
                } else {
                    DoNothing
                };

                if new_lp > old_lp {
                    self.nodes[id.0] = node;
                }
                if new_ub > old_ub {
                    self.nodes[id.0].ub = new_ub;
                }

                action
            }
            Vacant(e) => {
                let id = if self.recycle_bin.is_empty() {
                    let id = NodeId(self.nodes.len());
                    self.nodes.push(node);
                    self.pos.push(0); // dummy
                    id
                } else {
                    let id = self.recycle_bin.pop().unwrap();
                    self.nodes[id.0] = node;
                    id
                };

                self.heap.push(id);
                self.pos[id.0] = self.heap.len() - 1;
                e.insert(id);
                BubbleUp(id)
            }
        };

        // restore the invariants
        self.process_action(action);
    }

    /// Pops the best node out of the heap. Here, the best is defined as the
    /// node having the best upper bound, with the longest `value`.
    fn pop(&mut self) -> Option<SubProblem<Self::State>> {
        if self.is_empty() {
            return None;
        }

        let id = self.heap.swap_remove(0);
        let action = if self.heap.is_empty() {
            DoNothing
        } else {
            self.pos[self.heap[0].0] = 0;
            BubbleDown(self.heap[0])
        };

        self.process_action(action);
        self.recycle_bin.push(id);

        let node = self.nodes[id.0].clone();
        self.states.remove(&node.state);

        Some(node)
    }

    /// Clears the content of the heap to reset it to a state equivalent to
    /// a fresh instantiation of the heap.
    fn clear(&mut self) {
        self.states.clear();
        self.nodes.clear();
        self.pos.clear();
        self.heap.clear();
        self.recycle_bin.clear();
    }

    /// Returns the 'length' of the heap. That is, the number of items that
    /// can still be popped out of the heap.
    fn len(&self) -> usize {
        self.heap.len()
    }
}

impl<O> NoDupFringe<O>
where
    O: SubProblemRanking,
    O::State: Eq + Hash + Clone,
{
    /// Creates a new instance of the no dup heap which uses cmp as
    /// comparison criterion.
    pub fn new(ranking: O) -> Self {
        Self {
            cmp: CompareSubProblem::new(ranking),
            states: Default::default(),
            nodes: vec![],
            pos: vec![],
            heap: vec![],
            recycle_bin: vec![],
        }
    }

    /// Returns true iff the heap is empty (len() == 0)
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Internal helper method to bubble a node up or down, depending of the
    /// specified action.
    fn process_action(&mut self, action: Action) {
        match action {
            BubbleUp(id) => self.bubble_up(id),
            BubbleDown(id) => self.bubble_down(id),
            DoNothing => { /* sweet life */ }
        }
    }
    /// Internal helper method to return the position of a node in the heap.
    fn position(&self, n: NodeId) -> usize {
        self.pos[n.0]
    }
    /// Internal helper method to compare the nodes identified by the ids found
    /// at the given positions in the heap.
    fn compare_at_pos(&self, x: usize, y: usize) -> Ordering {
        let node_x = &self.nodes[self.heap[x].0];
        let node_y = &self.nodes[self.heap[y].0];
        self.cmp.compare(node_x, node_y)
    }
    /// Internal method to bubble a node up and restore the heap invariant.
    fn bubble_up(&mut self, id: NodeId) {
        let mut me = self.position(id);
        let mut parent = self.parent(me);

        while !self.is_root(me) && self.compare_at_pos(me, parent) == Greater {
            let p_id = self.heap[parent];

            self.pos[p_id.0] = me;
            self.pos[id.0] = parent;
            self.heap[me] = p_id;
            self.heap[parent] = id;

            me = parent;
            parent = self.parent(me);
        }
    }
    /// Internal method to sink a node down so as to restor the heap invariant.
    fn bubble_down(&mut self, id: NodeId) {
        let mut me = self.position(id);
        let mut kid = self.max_child_of(me);

        while kid > 0 && self.compare_at_pos(me, kid) == Less {
            let k_id = self.heap[kid];

            self.pos[k_id.0] = me;
            self.pos[id.0] = kid;
            self.heap[me] = k_id;
            self.heap[kid] = id;

            me = kid;
            kid = self.max_child_of(me);
        }
    }
    /// Internal helper method that returns the position of the node which is
    /// the parent of the node at `pos` in the heap.
    fn parent(&self, pos: usize) -> usize {
        if self.is_root(pos) {
            pos
        } else if self.is_left(pos) {
            pos / 2
        } else {
            pos / 2 - 1
        }
    }
    /// Internal helper method that returns the position of the child of the
    /// node at position `pos` which is considered to be the maximum of the
    /// children of that node.
    ///
    /// # Warning
    /// When the node at `pos` is a leaf, this method returns **0** for the
    /// position of the child. This value 0 acts as a marker to tell that no
    /// child is to be found.
    fn max_child_of(&self, pos: usize) -> usize {
        let size = self.len();
        let left = self.left_child(pos);
        let right = self.right_child(pos);

        if left >= size {
            return 0;
        }
        if right >= size {
            return left;
        }

        match self.compare_at_pos(left, right) {
            Greater => left,
            _ => right,
        }
    }
    /// Internal helper method to return the position of the left child of
    /// the node at the given `pos`.
    fn left_child(&self, pos: usize) -> usize {
        pos * 2 + 1
    }
    /// Internal helper method to return the position of the right child of
    /// the node at the given `pos`.
    fn right_child(&self, pos: usize) -> usize {
        pos * 2 + 2
    }
    /// Internal helper method which returns true iff the node at `pos` is the
    /// root of the binary heap (position is zero).
    fn is_root(&self, pos: usize) -> bool {
        pos == 0
    }
    /// Internal helper method which returns true iff the node at `pos` is the
    /// left child of its parent.
    fn is_left(&self, pos: usize) -> bool {
        pos % 2 == 1
    }
    /*
    /// Internal helper method which returns true iff the node at `pos` is the
    /// right child of its parent.
    fn is_right(&self, pos: usize) -> bool {
        pos % 2 == 0
    }
    */
}


#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_no_dup_fringe {
    use crate::*;
    use std::{sync::Arc, cmp::Ordering};

    // by default, it is empty
    #[test]
    fn by_default_it_is_empty() {
        assert!(empty_fringe().is_empty())
    }

    // when the size is zero, then it is empty
    #[test]
    fn when_the_size_is_zero_then_it_is_empty() {
        let fringe = empty_fringe();
        assert_eq!(fringe.len(), 0);
        assert!(fringe.is_empty());
    }

    // when the size is greater than zero, it it not empty
    #[test]
    fn when_the_size_is_greater_than_zero_it_is_not_empty() {
        let fringe = non_empty_fringe();
        assert_eq!(fringe.len(), 1);
        assert!(!fringe.is_empty());
    }

    // when I push a non existing node onto the fringe then the length increases
    #[test]
    fn when_i_push_a_non_existing_node_onto_the_fringe_then_the_length_increases() {
        let mut fringe = empty_fringe();
        fringe.push(SubProblem {
            state: Arc::new(42),
            value: 0,
            path : vec![],
            ub   : 0
        });
        assert_eq!(fringe.len(), 1);
        fringe.push(SubProblem{
            state: Arc::new(43),
            value: 0,
            path : vec![],
            ub: 0
        });
        assert_eq!(fringe.len(), 2);
    }
    // when I push an existing node onto the fringe then the length wont increases
    #[test]
    fn when_i_push_an_existing_node_onto_the_fringe_then_the_length_does_not_increases() {
        let mut fringe = empty_fringe();
        fringe.push(SubProblem {
            state: Arc::new(42),
            value: 0,
            path : vec![],
            ub   : 0
        });
        assert_eq!(fringe.len(), 1);
        fringe.push(SubProblem {
            state: Arc::new(42),
            value: 12,
            path : vec![],
            ub   : 5
        });
        assert_eq!(fringe.len(), 1);
    }
    // when I pop a node off the fringe then the length decreases
    #[test]
    fn when_i_pop_a_node_off_the_fringe_then_the_length_decreases() {
        let mut fringe = non_empty_fringe();
        assert_eq!(fringe.len(), 1);
        fringe.pop();
        assert_eq!(fringe.len(), 0);
    }

    // when I try to pop a node off an empty fringe, I get none
    #[test]
    fn when_i_try_to_pop_a_node_off_an_empty_fringe_i_get_none() {
        let mut fringe = empty_fringe();
        assert_eq!(fringe.pop(), None);
    }

    // when I pop a node, it is always the one with the largest ub (then value)
    #[test]
    fn when_i_pop_a_node_it_is_always_the_one_with_the_largest_ub_then_lp() {
        let mut fringe = empty_fringe();
        let a = SubProblem {
            state: Arc::new(1),
            value: 1,
            path : vec![],
            ub   : 1
        };
        let b = SubProblem {
            state: Arc::new(2),
            value: 2,
            path : vec![],
            ub   : 2
        };
        let c = SubProblem {
            state: Arc::new(3),
            value: 3,
            path : vec![],
            ub   : 3
        };
        let d = SubProblem {
            state: Arc::new(4),
            path: vec![],
            value: 4,
            ub: 4
        };
        let e = SubProblem{
            state: Arc::new(5),
            path: vec![],
            value: 4,
            ub: 5
        };
        let f = SubProblem{
            state: Arc::new(5),
            path: vec![],
            value: 5,
            ub: 5
        };

        fringe.push(a.clone());
        fringe.push(f.clone());
        fringe.push(b.clone());
        fringe.push(d.clone());
        fringe.push(c.clone());
        fringe.push(e);

        assert_eq!(fringe.pop(), Some(f));
        //assert_eq!(fringe.pop(), Some(e)); // node 'e' will never show up
        assert_eq!(fringe.pop(), Some(d));
        assert_eq!(fringe.pop(), Some(c));
        assert_eq!(fringe.pop(), Some(b));
        assert_eq!(fringe.pop(), Some(a));
    }

    // when I pop a node off the fringe, for which multiple copies have been added
    // I retrieve the one with the longest path
    #[test]
    fn when_i_pop_a_node_off_the_fringe_for_which_multiple_copies_have_been_added_then_i_retrieve_the_one_with_longest_path(){
        let pe = vec![
            Decision {variable: Variable(0), value: 4},
        ];

        let pf = vec![
            Decision {variable: Variable(1), value: 5},
        ];

        let ne = SubProblem{
            state: Arc::new(5),
            path: pe,
            value: 4,
            ub: 5
        };
        let nf = SubProblem{
            state: Arc::new(5),
            path: pf,
            value: 5,
            ub: 5
        };

        let mut fringe = empty_fringe();
        fringe.push(ne);
        fringe.push(nf.clone());

        assert_eq!(fringe.pop(), Some(nf));
    }

    // when I clear an empty fringe, it remains empty
    #[test]
    fn when_i_clear_an_empty_fringe_it_remains_empty() {
        let mut fringe = empty_fringe();
        assert!(fringe.is_empty());
        fringe.clear();
        assert!(fringe.is_empty());
    }
    // when I clear a non empty fringe it becomes empty
    #[test]
    fn when_i_clear_a_non_empty_fringe_it_becomes_empty() {
        let mut fringe = non_empty_fringe();
        assert!(!fringe.is_empty());
        fringe.clear();
        assert!(fringe.is_empty());
    }


    /// A dummy state comparator for use in the tests
    #[derive(Debug, Clone, Copy, Default)]
    struct UsizeRanking;
    impl StateRanking for UsizeRanking {
        type State = usize;

        fn compare(&self, a: &Self::State, b: &Self::State) -> Ordering {
            a.cmp(b)
        }
    }

        
    fn empty_fringe() -> NoDupFringe<MaxUB<'static, UsizeRanking>> {
        NoDupFringe::new(MaxUB::new(&UsizeRanking))
    }
    fn non_empty_fringe() -> NoDupFringe<MaxUB<'static, UsizeRanking>> {
        let mut fringe = empty_fringe();
        fringe.push(SubProblem{
            state: Arc::new(42),
            path: vec![],
            value: 0,
            ub: 0
        });
        fringe
    }

    // 

    #[test]
    fn popped_in_order() {
        let nodes = [
            fnode(1, 10, 100),
            fnode(2, 10, 101),
            fnode(3, 10, 102),
            fnode(4, 10, 103),
            fnode(5, 10, 104),
        ];

        let mut heap = empty_fringe();
        push_all(&mut heap, &nodes);
        assert_eq!(5,     heap.len());
        assert_eq!(false, heap.is_empty());

        let actual   = pop_all(&mut heap);
        let expected = vec![5, 4, 3, 2, 1];
        assert_eq!(expected,  actual);
        assert_eq!(0,    heap.len());
        assert_eq!(true, heap.is_empty());
    }
    #[test]
    fn pushing_same_node_multiple_times_does_not_alter_pop_order() {
        let nodes = [
            fnode(1, 10, 100),
            fnode(2, 10, 101),
            fnode(3, 10, 102),
            fnode(4, 10, 103),
            fnode(5, 10, 104),
        ];

        let mut heap = empty_fringe();
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        // even after pushing all nodes five times, there are only 5 nodes in the heap
        assert_eq!(5,     heap.len());
        assert_eq!(false, heap.is_empty());

        let actual   = pop_all(&mut heap);
        let expected = vec![5, 4, 3, 2, 1];
        assert_eq!(expected,  actual);
        assert_eq!(0,    heap.len());
        assert_eq!(true, heap.is_empty());
    }

    #[test]
    fn pushing_nodes_triggers_reordering_if_lplen_is_better_up() {
        let nodes_1 = [
            fnode(1, 10, 100),
            fnode(2, 10, 101),
            fnode(3, 10, 102),
            fnode(4, 10, 103),
            fnode(5, 10, 104),
        ];
        let nodes_2 = [
            fnode(1, 15, 100),
            fnode(2, 15,  99),
            fnode(3, 15,  98),
            fnode(4, 15,  97),
            fnode(5, 15,  96),
        ];
        let nodes_3 = [
            fnode(1, 20,  92),
            fnode(2, 20,  93),
            fnode(3, 20,  94),
            fnode(4, 20,  95),
            fnode(5, 20,  96),
        ];

        let mut heap = empty_fringe();
        push_all(&mut heap, &nodes_1);
        push_all(&mut heap, &nodes_2);
        push_all(&mut heap, &nodes_3);
        // even after pushing all nodes five times, there are only 5 nodes in the heap
        assert_eq!(5,     heap.len());
        assert_eq!(false, heap.is_empty());

        let actual   = pop_all(&mut heap);
        let expected = vec![5, 4, 3, 2, 1];
        assert_eq!(expected,  actual);
        assert_eq!(0,    heap.len());
        assert_eq!(true, heap.is_empty());
    }

    fn push_all<T: SubProblemRanking<State = usize>>(heap: &mut NoDupFringe<T>, nodes: &[SubProblem<usize>]) {
        for n in nodes.iter() {
            heap.push(n.clone());
        }
    }
    fn pop_all<T: SubProblemRanking<State = usize>>(heap: &mut NoDupFringe<T>) -> Vec<usize> {
        let mut popped = vec![];
        while let Some(n) = heap.pop() {
            popped.push(*n.state)
        }
        popped
    }

    fn fnode(state: usize, value: isize, ub: isize) -> SubProblem<usize> {
        SubProblem {
            state: Arc::new(state),
            path : vec![],
            value,
            ub
        }
    }
}