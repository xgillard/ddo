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

//! This module defines some utility types used to help provide concrete
//! implementations of the objects in the abstraction layer.

use crate::common::FrontierNode;
use std::sync::Arc;
use metrohash::MetroHashMap;
use std::collections::hash_map::Entry::{Vacant, Occupied};
use crate::implementation::utils::Action::{BubbleUp, DoNothing, BubbleDown};
use compare::Compare;
use std::cmp::Ordering::{Greater, Less};
use std::hash::Hash;
use std::cmp::Ordering;
use crate::implementation::heuristics::MaxUB;

/// This is a type-safe identifier for some node in the queue.
/// Basically, this NodeId equates to the position of the identified
/// node in the `nodes` list from the `NoDupHeap`.
#[derive(Debug, Copy, Clone)]
struct NodeId(usize);

/// An enum to know what needs to be done with a given node id
#[derive(Debug, Copy, Clone)]
enum Action {
    DoNothing,
    BubbleUp  (NodeId),
    BubbleDown(NodeId)
}

/// This is an updatable binary heap backed by a vector which ensures that
/// items remain ordered in the priority queue while guaranteeing that a
/// given state will only ever be present *ONCE* in the priority queue (the
/// node with the longest path to state is the only kept copy).
#[derive(Clone)]
pub struct NoDupHeap<T>
    where T: Eq + Hash + Clone
{
    /// This is the comparator used to order the nodes in the binary heap
    cmp: MaxUB,
    /// A mapping that associates some state to a node identifier.
    states: MetroHashMap<Arc<T>, NodeId>,
    /// The actual payload (nodes) ordered in the list
    nodes: Vec<FrontierNode<T>>,
    /// The position of the items in the heap
    pos: Vec<usize>,
    /// This is the actual heap which orders nodes.
    heap: Vec<NodeId>,
    /// The positions in the `nodes` vector that can be recycled.
    recycle_bin: Vec<NodeId>
}
/// Construction of a Default NoDupHeap
impl <T> Default for NoDupHeap<T>
    where T: Eq + Hash + Clone
{
    fn default() -> Self {
        Self::new()
    }
}
impl <T> NoDupHeap<T>
    where T: Eq + Hash + Clone
{
    /// Creates a new instance of the no dup heap which uses cmp as
    /// comparison criterion.
    pub fn new() -> Self {
        Self {
            cmp   : MaxUB::default(),
            states: MetroHashMap::default(),
            nodes : vec![],
            pos   : vec![],
            heap  : vec![],
            recycle_bin: vec![]
        }
    }
    /// Returns the 'length' of the heap. That is, the number of items that
    /// can still be popped out of the heap.
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    /// Returns true iff the heap is empty (len() == 0)
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    /// Clears the content of the heap to reset it to a state equivalent to
    /// a fresh instantiation of the heap.
    pub fn clear(&mut self) {
        self.states.clear();
        self.nodes.clear();
        self.pos.clear();
        self.heap.clear();
        self.recycle_bin.clear();
    }
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
    pub fn push(&mut self, node: FrontierNode<T>) {
        let state = Arc::clone(&node.state);

        let action = match self.states.entry(state) {
            Occupied(e) => {
                let id     = *e.get();
                let lp_len = self.nodes[id.0].lp_len;

                let action =
                    if self.cmp.compare(&node, &self.nodes[id.0]) == Greater {
                        BubbleUp(id)
                    } else {
                        DoNothing
                    };

                if  node.lp_len > lp_len {
                    self.nodes[id.0] = node;
                }

                action
            },
            Vacant(e) => {
                let id =
                    if self.recycle_bin.is_empty() {
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
                self.pos[id.0] = self.heap.len()-1;
                e.insert(id);
                BubbleUp(id)
            },
        };

        // restore the invariants
        self.process_action(action);
    }
    /// Pops the best node out of the heap. Here, the best is defined as the
    /// node having the best upper bound, with the longest `lp_len`.
    pub fn pop(&mut self) -> Option<FrontierNode<T>> {
        if self.is_empty() {
            return None;
        }

        let id = self.heap.swap_remove(0);
        let action =
            if self.heap.is_empty() {
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
    /// Internal helper method to bubble a node up or down, depending of the
    /// specified action.
    fn process_action(&mut self, action: Action) {
        match action {
            BubbleUp(id)   => self.bubble_up(id),
            BubbleDown(id) => self.bubble_down(id),
            DoNothing      => {/* sweet life */},
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
            let p_id         = self.heap[parent];

            self.pos[p_id.0] = me;
            self.pos[id.0]   = parent;
            self.heap[me]    = p_id;
            self.heap[parent]= id;

            me     = parent;
            parent = self.parent(me);
        }
    }
    /// Internal method to sink a node down so as to restor the heap invariant.
    fn bubble_down(&mut self, id: NodeId) {
        let mut me  = self.position(id);
        let mut kid = self.max_child_of(me);

        while kid > 0 && self.compare_at_pos(me, kid) == Less {
            let k_id         = self.heap[kid];

            self.pos[k_id.0] = me;
            self.pos[id.0]   = kid;
            self.heap[me]    = k_id;
            self.heap[kid]   = id;

            me  = kid;
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
        let right= self.right_child(pos);

        if left  >= size { return 0;    }
        if right >= size { return left; }

        match self.compare_at_pos(left, right) {
            Greater => left,
            _       => right
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
mod test_heap {
    use crate::common::{FrontierNode, PartialAssignment};
    use std::sync::Arc;
    use crate::implementation::utils::NoDupHeap;

    #[test]
    fn popped_in_order() {
        let nodes = [
            fnode('A', 10, 100),
            fnode('B', 10, 101),
            fnode('C', 10, 102),
            fnode('D', 10, 103),
            fnode('E', 10, 104),
        ];

        let mut heap = NoDupHeap::new();
        push_all(&mut heap, &nodes);
        assert_eq!(5,     heap.len());
        assert_eq!(false, heap.is_empty());

        let actual   = pop_all(&mut heap);
        let expected = vec!['E', 'D', 'C', 'B', 'A'];
        assert_eq!(expected,  actual);
        assert_eq!(0,    heap.len());
        assert_eq!(true, heap.is_empty());
    }
    #[test]
    fn pushing_same_node_multiple_times_does_not_alter_pop_order() {
        let nodes = [
            fnode('A', 10, 100),
            fnode('B', 10, 101),
            fnode('C', 10, 102),
            fnode('D', 10, 103),
            fnode('E', 10, 104),
        ];

        let mut heap = NoDupHeap::new();
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        push_all(&mut heap, &nodes);
        // even after pushing all nodes five times, there are only 5 nodes in the heap
        assert_eq!(5,     heap.len());
        assert_eq!(false, heap.is_empty());

        let actual   = pop_all(&mut heap);
        let expected = vec!['E', 'D', 'C', 'B', 'A'];
        assert_eq!(expected,  actual);
        assert_eq!(0,    heap.len());
        assert_eq!(true, heap.is_empty());
    }

    #[test]
    fn pushing_nodes_triggers_reordering_if_lplen_is_better_up() {
        let nodes_1 = [
            fnode('A', 10, 100),
            fnode('B', 10, 101),
            fnode('C', 10, 102),
            fnode('D', 10, 103),
            fnode('E', 10, 104),
        ];
        let nodes_2 = [
            fnode('A', 15, 100),
            fnode('B', 15,  99),
            fnode('C', 15,  98),
            fnode('D', 15,  97),
            fnode('E', 15,  96),
        ];
        let nodes_3 = [
            fnode('A', 20,  92),
            fnode('B', 20,  93),
            fnode('C', 20,  94),
            fnode('D', 20,  95),
            fnode('E', 20,  96),
        ];

        let mut heap = NoDupHeap::new();
        push_all(&mut heap, &nodes_1);
        push_all(&mut heap, &nodes_2);
        push_all(&mut heap, &nodes_3);
        // even after pushing all nodes five times, there are only 5 nodes in the heap
        assert_eq!(5,     heap.len());
        assert_eq!(false, heap.is_empty());

        let actual   = pop_all(&mut heap);
        let expected = vec!['E', 'D', 'C', 'B', 'A'];
        assert_eq!(expected,  actual);
        assert_eq!(0,    heap.len());
        assert_eq!(true, heap.is_empty());
    }

    fn push_all(heap: &mut NoDupHeap<char>, nodes: &[FrontierNode<char>]) {
        for n in nodes.iter() {
            heap.push(n.clone());
        }
    }
    fn pop_all(heap: &mut NoDupHeap<char>) -> Vec<char> {
        let mut popped = vec![];
        while let Some(n) = heap.pop() {
            popped.push(*n.state)
        }
        popped
    }

    fn fnode(state: char, lp_len: isize, ub: isize) -> FrontierNode<char> {
        FrontierNode {
            state: Arc::new(state),
            path : Arc::new(PartialAssignment::Empty),
            lp_len,
            ub
        }
    }
}