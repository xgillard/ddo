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

//! This module provides the implementation of auxiliary structures which come
//! in handy when implementing a shallow mdd.

use std::sync::Arc;

use crate::abstraction::heuristics::SelectableNode;
use crate::common::{Decision, FrontierNode, PartialAssignment};
use crate::implementation::mdd::utils::NodeFlags;
use std::rc::Rc;
use crate::common::PartialAssignment::FragmentExtension;

/// This tiny structure represents an internal node from the flat MDD.
/// Basically, it only remembers its own state, value, the best path from root
/// and possibly the description of the 'best' edge connecting it to one of its
/// parents.
#[derive(Debug, Clone)]
pub struct Node<T> {
    pub this_state   : Rc<T>,
    pub value        : isize,
    pub estimate     : isize,
    pub flags        : NodeFlags,
    pub best_edge    : Option<Edge<T>>
}
/// This is the description of an edge. It only knows the state from the parent
/// node, the decision that caused the transition from one state to the other
/// and the weight of taking that transition.
#[derive(Debug, Clone)]
pub struct Edge<T> {
    pub parent   : Rc<Node<T>>,
    pub weight   : isize,
    pub decision : Decision,
}
/// If a layer grows too large, the branch and bound algorithm might need to
/// squash the current layer and thus to select some Nodes for merge or removal.
impl <T> SelectableNode<T> for Node<T> {
    /// Returns a reference to the state of this node
    fn state(&self) -> &T {
        self.this_state.as_ref()
    }
    /// Returns the value of the objective function at this node.
    fn value(&self) -> isize {
        self.value
    }
    /// Returns true iff the node is an exact node.
    fn is_exact(&self) -> bool {
        self.flags.is_exact()
    }
}
/// If a layer grows too large, the branch and bound algorithm might need to
/// squash the current layer and thus to select some Nodes for merge or removal.
impl <T> SelectableNode<T> for Rc<Node<T>> {
    /// Returns a reference to the state of this node
    fn state(&self) -> &T {
        self.as_ref().state()
    }
    /// Returns the value of the objective function at this node.
    fn value(&self) -> isize {
        self.as_ref().value()
    }
    /// Returns true iff the node is an exact node.
    fn is_exact(&self) -> bool {
        self.as_ref().is_exact()
    }
}
impl <T> Node<T> {
    /// Returns true iff there exists a best path from root to this node.
    pub fn has_exact_best(&self) -> bool {
        !self.flags.is_relaxed()
    }
    /// Sets the 'exact' flag of this node to the given value.
    /// This is useful to ensure that a node becomes inexact when it used to be
    /// exact but an other path passing through a inexact node produces the
    /// same state.
    pub fn set_exact(&mut self, exact: bool) {
        self.flags.set_exact(exact)
    }
    /// Merge other into this node. That is to say, it combines the information
    /// from two nodes that are logically equivalent (should be the same).
    /// Concretely, it means that it possibly updates the current node to keep
    /// the best _longest path info_, track the 'exactitude' of the node and
    /// keep the tightest upper bound.
    ///
    /// # Important note
    /// *This has nothing to do with the user-provided `merge_*` operators !*
    pub fn merge(&mut self, other: Node<T>) {
        let exact = self.is_exact() && other.is_exact();

        if other.value > self.value {
            self.value = other.value;
            self.flags = other.flags;
            self.best_edge = other.best_edge;
        }
        self.estimate = self.estimate.min(other.estimate);
        self.set_exact(exact);
    }
    /// Returns the path to this node
    pub fn path(&self) -> Vec<Decision> {
        let mut edge = &self.best_edge;
        let mut path = vec![];
        while let Some(e) = edge {
            path.push(e.decision);
            edge = &e.parent.best_edge;
        }
        path
    }
    /// Returns the nodes upper bound
    pub fn ub(&self) -> isize {
        self.value.saturating_add(self.estimate)
    }
}
impl <T: Clone> Node<T> {
    pub fn to_frontier_node(&self, root: &Arc<PartialAssignment>) -> FrontierNode<T> {
        FrontierNode {
            state : Arc::new(self.this_state.as_ref().clone()),
            path  : Arc::new(FragmentExtension {parent: Arc::clone(root), fragment: self.path()}),
            lp_len: self.value,
            ub    : self.ub()
        }
    }
}
/// Because the solver works with `FrontierNode`s but the MDD uses `Node`s, we
/// need a way to convert from one type to the other. This conversion ensures
/// that a `Node` can be built from a `FrontierNode`.
impl <T : Clone> From<&FrontierNode<T>> for Node<T> {
    fn from(n: &FrontierNode<T>) -> Self {
        Node {
            this_state: Rc::new(n.state.as_ref().clone()),
            value     : n.lp_len,
            estimate  : n.ub - n.lp_len,
            flags     : NodeFlags::new_exact(),
            best_edge : None
        }
    }
}

// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
mod test_node {
    use std::sync::Arc;

    use crate::abstraction::heuristics::SelectableNode;
    use crate::common::{Decision, FrontierNode, PartialAssignment, Variable};
    use crate::common::PartialAssignment::FragmentExtension;
    use crate::implementation::mdd::shallow::utils::{Edge, Node};
    use crate::implementation::mdd::utils::NodeFlags;
    use std::rc::Rc;

    #[test]
    fn test_from_frontier_node() {
        let empty = Arc::new(PartialAssignment::Empty);
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};

        let front = FrontierNode {
            state: Arc::new(12),
            path: Arc::new(FragmentExtension {parent: empty, fragment: vec![d1, d2]}),
            lp_len: 56,
            ub: 57
        };

        let node = Node::from(&front);
        assert_eq!(12, *node.this_state);
        assert_eq!(56, node.value);
        assert_eq!( 1, node.estimate);
        assert!(node.best_edge.is_none());
    }
    #[test]
    fn test_into_frontier_node_no_path() {
        let empty = Arc::new(PartialAssignment::Empty);
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let pa= Arc::new(FragmentExtension {parent: empty, fragment: vec![d1, d2]});
        let node  = Node {
            this_state: Rc::new(12),
            value     : 56,
            estimate  : 1,
            flags     : Default::default(),
            best_edge : None
        };

        let front = node.to_frontier_node(&pa);
        assert_eq!(12, *front.state);
        assert_eq!(vec![d1, d2], front.path.iter().collect::<Vec<Decision>>());
        assert_eq!(56, front.lp_len);
        assert_eq!(57, front.ub);
    }
    #[test]
    fn test_into_frontier_with_path() {
        let empty = Arc::new(PartialAssignment::Empty);
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let d3= Decision{variable: Variable(3), value: 3};

        let pa= Arc::new(FragmentExtension {parent: empty, fragment: vec![d1, d2]});

        let node1  = Node {
            this_state: Rc::new(12),
            value     : 56,
            estimate  : 100,
            flags     : Default::default(),
            best_edge : None
        };
        let node   = Node {
            this_state: Rc::new(12),
            value     : 57,
            estimate  : 100,
            flags     : Default::default(),
            best_edge : Some(Edge {
                parent: Rc::new(node1),
                weight: 1,
                decision: d3
            })
        };

        let front = node.to_frontier_node(&pa);
        assert_eq!(12, *front.state);
        assert_eq!(vec![d3, d1, d2], front.path.iter().collect::<Vec<Decision>>());
        assert_eq!(57, front.lp_len);
        assert_eq!(157, front.ub);
    }
    #[test]
    fn selectable_node_yields_a_ref_to_this_state() {
        let node = Node {
            this_state: Rc::new(42),
            value: 0,
            estimate: 1000,
            flags: Default::default(),
            best_edge: None
        };

        assert_eq!(42, *node.state());
    }
    #[test]
    fn selectable_node_yields_the_nodes_value() {
        let node = Node {
            this_state: Rc::new(42),
            value: 74,
            estimate: 1000,
            flags: Default::default(),
            best_edge: None
        };

        assert_eq!(74, node.value());
    }
    #[test]
    fn selectable_node_is_exact_iff_flags_is_exact_and_not_relaxed() {
        let mut node = Node {
            this_state: Rc::new(42),
            value: 74,
            estimate: 1000,
            flags: Default::default(),
            best_edge: None
        };
        assert_eq!(true, node.is_exact());

        node.flags.set_exact(false);
        assert_eq!(false, node.is_exact());

        node.flags.set_exact(true);
        assert_eq!(true, node.is_exact());

        node.flags.set_relaxed(true);
        assert_eq!(false, node.is_exact());
    }


    #[test]
    fn node_has_exact_best_iff_relaxed_flag_is_not_set() {
        let mut node = Node {
            this_state: Rc::new(42),
            value: 74,
            estimate: 1000,
            flags: Default::default(),
            best_edge: None
        };
        assert_eq!(true, node.has_exact_best());

        node.flags.set_exact(false);
        assert_eq!(true, node.has_exact_best());

        node.flags.set_exact(true);
        assert_eq!(true, node.has_exact_best());

        node.flags.set_relaxed(true);
        assert_eq!(false, node.has_exact_best());
    }

    #[test]
    fn set_exact_changes_the_value_of_exact_flag() {
        let mut node = Node {
            this_state: Rc::new(42),
            value: 74,
            estimate: 1000,
            flags: Default::default(),
            best_edge: None
        };
        assert_eq!(true, node.flags.is_exact());

        node.set_exact(false);
        assert_eq!(false, node.flags.is_exact());

        node.set_exact(true);
        assert_eq!(true, node.flags.is_exact());
    }

    #[test]
    fn merging_two_exact_nodes_has_no_effect_when_second_has_lower_value() {
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let n0 = Node {
            this_state: Rc::new(0),
            value: 0,
            estimate: 0,
            flags: Default::default(),
            best_edge: None
        };
        let n0 = Rc::new(n0);
        let mut n1 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d1
            })
        };
        let n2 = Node {
            this_state: Rc::new(1),
            value: 0,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d2
            })
        };

        n1.merge(n2);
        assert_eq!(1, *n1.state());
        assert_eq!(1, n1.value);
        assert_eq!(true, n1.is_exact());
        assert_eq!(true, n1.best_edge.is_some());
        assert_eq!(d1, n1.best_edge.unwrap().decision)
    }

    #[test]
    fn merging_two_exact_nodes_has_no_effect_when_second_has_equal_value() {
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let n0 = Node {
            this_state: Rc::new(0),
            value: 0,
            estimate: 0,
            flags: Default::default(),
            best_edge: None
        };
        let n0 = Rc::new(n0);
        let mut n1 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d1
            })
        };
        let n2 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d2
            })
        };

        n1.merge(n2);
        assert_eq!(1, *n1.state());
        assert_eq!(1, n1.value);
        assert_eq!(true, n1.is_exact());
        assert_eq!(true, n1.best_edge.is_some());
        assert_eq!(d1, n1.best_edge.unwrap().decision)
    }

    #[test]
    fn merging_two_nodes_updates_best_edge_and_value_when_second_has_higher_value() {
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let n0 = Node {
            this_state: Rc::new(0),
            value: 0,
            estimate: 0,
            flags: Default::default(),
            best_edge: None
        };
        let n0 = Rc::new(n0);
        let mut n1 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d1
            })
        };
        let n2 = Node {
            this_state: Rc::new(1),
            value: 2,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 2,
                decision: d2
            })
        };

        n1.merge(n2);
        assert_eq!(1, *n1.state());
        assert_eq!(2, n1.value);
        assert_eq!(true, n1.is_exact());
        assert_eq!(true, n1.best_edge.is_some());
        assert_eq!(d2, n1.best_edge.unwrap().decision)
    }

    #[test]
    fn merging_exact_and_inexact_makes_the_result_inexact() {
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let n0 = Node {
            this_state: Rc::new(0),
            value: 0,
            estimate: 0,
            flags: Default::default(),
            best_edge: None
        };
        let n0 = Rc::new(n0);
        let mut n1 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d1
            })
        };
        let mut n2 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d2
            })
        };
        n2.set_exact(false);
        n1.merge(n2);
        assert_eq!(false, n1.is_exact());
    }
    #[test]
    fn merging_exact_and_relaxed_yields_relaxed_if_best_has_relaxed_flag() {
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let n0 = Node {
            this_state: Rc::new(0),
            value: 0,
            estimate: 0,
            flags: Default::default(),
            best_edge: None
        };
        let n0 = Rc::new(n0);
        let mut n1 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d1
            })
        };
        let n2 = Node {
            this_state: Rc::new(1),
            value: 2,
            estimate: 1000,
            flags: NodeFlags::new_relaxed(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 2,
                decision: d2
            })
        };
        n1.merge(n2);
        assert_eq!(true, n1.flags.is_relaxed());
    }
    #[test]
    fn merging_exact_and_relaxed_not_relaxed_if_best_not_relaxed() {
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let n0 = Node {
            this_state: Rc::new(0),
            value: 0,
            estimate: 0,
            flags: Default::default(),
            best_edge: None
        };
        let n0 = Rc::new(n0);
        let mut n1 = Node {
            this_state: Rc::new(1),
            value: 2,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 2,
                decision: d1
            })
        };
        let n2 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_relaxed(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d2
            })
        };
        n1.merge(n2);
        assert_eq!(false, n1.flags.is_relaxed());
    }

    #[test]
    fn merging_exact_and_relaxed_clears_relaxed_flag_if_best_not_relaxed() {
        let d1= Decision{variable: Variable(2), value: 1};
        let d2= Decision{variable: Variable(2), value: 2};
        let n0 = Node {
            this_state: Rc::new(0),
            value: 0,
            estimate: 0,
            flags: Default::default(),
            best_edge: None
        };
        let n0 = Rc::new(n0);
        let mut n1 = Node {
            this_state: Rc::new(1),
            value: 1,
            estimate: 1000,
            flags: NodeFlags::new_relaxed(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 1,
                decision: d1
            })
        };
        let n2 = Node {
            this_state: Rc::new(1),
            value: 2,
            estimate: 1000,
            flags: NodeFlags::new_exact(),
            best_edge: Some(Edge{
                parent: Rc::clone(&n0),
                weight: 2,
                decision: d2
            })
        };
        n1.merge(n2);
        assert_eq!(false, n1.flags.is_relaxed());
    }
}