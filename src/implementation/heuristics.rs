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

//! This module provides an implementation for the usual solver heuristics
//! (node selection, width heuristic, variable ordering, load var strategies)

use std::cmp::Ordering;

use compare::Compare;

use crate::abstraction::heuristics::{LoadVars, NodeSelectionHeuristic, SelectableNode, VariableHeuristic, WidthHeuristic};
use crate::common::{FrontierNode, Variable, VarSet};

// ----------------------------------------------------------------------------
// --- VARIABLE SELECTION -----------------------------------------------------
// ----------------------------------------------------------------------------

/// This strategy branches on the variables in their ''natural'' order. That is,
/// it will first pick `Variable(0)` then `Variable(1)`, `Variable(2)`, etc...
/// until there are no variables left to branch on.
///
/// # Example
/// ```
/// # use ddo::common::{Variable, VarSet};
/// # use ddo::abstraction::heuristics::VariableHeuristic;
/// # use ddo::implementation::heuristics::NaturalOrder;
/// # let dummy = vec![0_isize];
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// #
/// let mut variables = VarSet::all(3);
/// assert_eq!(Some(Variable(0)), NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));
///
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// variables.remove(Variable(0)); // move on to the next layer
/// assert_eq!(Some(Variable(1)), NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));
///
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// variables.remove(Variable(1)); // move on to the next layer
/// assert_eq!(Some(Variable(2)), NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));
///
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// variables.remove(Variable(2)); // move on to the last layer, no more var to branch on
/// assert_eq!(None, NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));
/// ```
///
/// # Note:
/// Even though any variable heuristic may access the current and next layers
/// of the mdd being developed, the natural ordering heuristic does not use that
/// access.
///
#[derive(Default, Debug, Clone, Copy)]
pub struct NaturalOrder;
impl <T> VariableHeuristic<T> for NaturalOrder {
    fn next_var(&self, free_vars: &VarSet, _: &mut dyn Iterator<Item=&T>, _: &mut dyn Iterator<Item=&T>) -> Option<Variable>
    {
        free_vars.iter().next()
    }
}

/// This strategy selects the variable in decreasing order. This means, it has
/// provides and order which is the opposite of the `NaturalOrder`.
///
/// # Example
/// ```
/// # use ddo::common::{Variable, VarSet};
/// # use ddo::abstraction::heuristics::VariableHeuristic;
/// # use ddo::implementation::heuristics::Decreasing;
/// # let dummy = vec![0_isize];
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// #
/// let mut variables = VarSet::all(3);
/// assert_eq!(Some(Variable(2)), Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));
///
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// variables.remove(Variable(2)); // move on to the next layer
/// assert_eq!(Some(Variable(1)), Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));
///
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// variables.remove(Variable(1)); // move on to the next layer
/// assert_eq!(Some(Variable(0)), Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));
///
/// # let mut current_layer = dummy.iter();
/// # let mut next_layer    = dummy.iter();
/// variables.remove(Variable(0)); // move on to the last layer, no more var to branch on
/// assert_eq!(None, Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));
/// ```
///
/// # Note:
/// Even though any variable heuristic may access the current and next layers
/// of the mdd being developed, the natural ordering heuristic does not use that
/// access.
///
#[derive(Default, Debug, Clone, Copy)]
pub struct Decreasing;
impl <T> VariableHeuristic<T> for Decreasing {
    fn next_var(&self, vars: &VarSet, _: &mut dyn Iterator<Item=&T>, _: &mut dyn Iterator<Item=&T>) -> Option<Variable>
    {
        let len = vars.len();
        if len > 0 { Some(Variable(len-1)) } else { None }
    }
}

// ----------------------------------------------------------------------------
// --- WIDTH HEURISTICS -------------------------------------------------------
// ----------------------------------------------------------------------------

/// This strategy specifies a fixed maximum width for all the layers of an
/// approximate MDD. This is a *static* heuristic as the width will remain fixed
/// regardless of the approximate MDD to generate.
///
/// # Example
/// Assuming a fixed width of 100, and problem with 5 variables (0..=4). The
/// heuristic will return 100 no matter how many free vars there are left to
/// assign (`free_vars`).
///
/// ```
/// # use ddo::common::{Variable, VarSet};
/// # use ddo::abstraction::heuristics::WidthHeuristic;
/// # use ddo::implementation::heuristics::FixedWidth;
/// #
/// let heuristic   = FixedWidth(100); // assume a fixed width of 100
/// let mut var_set = VarSet::all(5);  // assume a problem with 5 variables
///
/// assert_eq!(100, heuristic.max_width(&var_set));
/// var_set.remove(Variable(1));       // let's say we fixed variables {1, 3, 4}.
/// var_set.remove(Variable(3));       // hence, only variables {0, 2} remain
/// var_set.remove(Variable(4));       // in the set of `free_vars`.
///
/// // still, the heuristic always return 100.
/// assert_eq!(100, heuristic.max_width(&var_set));
/// ```
#[derive(Debug, Copy, Clone)]
pub struct FixedWidth(pub usize);
impl WidthHeuristic for FixedWidth {
    fn max_width(&self, _free: &VarSet) -> usize {
        self.0
    }
}

/// This strategy specifies a variable maximum width for the layers of an
/// approximate MDD. When using this heuristic, each layer of an approximate
/// MDD is allowed to have as many nodes as there are free variables to decide
/// upon.
///
/// # Example
/// Assuming a problem with 5 variables (0..=4). If we are calling this heuristic
/// to derive the maximum allowed width for the layers of an approximate MDD
/// when variables {1, 3, 4} have been fixed, then the set of `free_vars` will
/// contain only the variables {0, 2}. In that case, this strategy will return
/// a max width of two.
///
/// ```
/// # use ddo::common::{Variable, VarSet};
/// # use ddo::abstraction::heuristics::WidthHeuristic;
/// # use ddo::implementation::heuristics::NbUnassignedWitdh;
/// #
/// let mut var_set = VarSet::all(5); // assume a problem with 5 variables
/// var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
/// var_set.remove(Variable(3));
/// var_set.remove(Variable(4));      // only variables {0, 2} remain in the set
///
/// assert_eq!(2, NbUnassignedWitdh.max_width(&var_set));
/// ```
#[derive(Default, Debug, Copy, Clone)]
pub struct NbUnassignedWitdh;
impl WidthHeuristic for NbUnassignedWitdh {
    fn max_width(&self, free: &VarSet) -> usize {
        free.len()
    }
}

/// This strategy acts as a decorator for an other max width heuristic. It
/// multiplies the maximum width of the strategy it delegates to by a constant
/// (configured) factor. It is typically used in conjunction with NbUnassigned
/// to provide a maximum width that allows a certain number of nodes.
/// Using a constant factor of 1 means that this decorator will have absolutely
/// no impact.
///
/// # Example
/// Here is an example of how to use this strategy to allow 5 nodes per
/// unassigned variable in a layer.
///
/// ```
/// # use ddo::common::{Variable, VarSet};
/// # use ddo::abstraction::heuristics::WidthHeuristic;
/// # use ddo::implementation::heuristics::{NbUnassignedWitdh, Times};
/// #
/// # let mut var_set = VarSet::all(5); // assume a problem with 5 variables
/// # var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
/// # var_set.remove(Variable(3));
/// # var_set.remove(Variable(4));      // only variables {0, 2} remain in the set
/// let custom = Times(5, NbUnassignedWitdh);
/// assert_eq!(5 * NbUnassignedWitdh.max_width(&var_set), custom.max_width(&var_set));
/// ```
#[derive(Clone)]
pub struct Times<X: WidthHeuristic + Clone>(pub usize, pub X);

impl <X: WidthHeuristic + Clone> WidthHeuristic for Times<X> {
    fn max_width(&self, free_vars: &VarSet) -> usize {
        self.0 * self.1.max_width(free_vars)
    }
}

/// This strategy acts as a decorator for an other max width heuristic. It
/// divides the maximum width of the strategy it delegates to by a constant
/// (configured) factor. It is typically used in conjunction with NbUnassigned
/// to provide a maximum width that allows a certain number of nodes.
/// Using a constant factor of 1 means that this decorator will have absolutely
/// no impact.
///
/// # Note
/// The maximum width is bounded by one at the very minimum. So it is *never*
/// going to return 0 for a value of the max width.
///
/// # Example
/// Here is an example of how to use this strategy to allow 1 nodes per two
/// unassigned variables in a layer.
///
/// ```
/// # use ddo::common::{Variable, VarSet};
/// # use ddo::abstraction::heuristics::WidthHeuristic;
/// # use ddo::implementation::heuristics::{NbUnassignedWitdh, DivBy};
/// #
/// # let mut var_set = VarSet::all(5); // assume a problem with 5 variables
/// # var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
/// # var_set.remove(Variable(3));
/// # var_set.remove(Variable(4));      // only variables {0, 2} remain in the set
/// let custom = DivBy(2, NbUnassignedWitdh);
/// assert_eq!(NbUnassignedWitdh.max_width(&var_set) / 2, custom.max_width(&var_set));
/// ```
#[derive(Clone)]
pub struct DivBy<X: WidthHeuristic + Clone>(pub usize, pub X);

impl <X: WidthHeuristic + Clone> WidthHeuristic for DivBy<X> {
    fn max_width(&self, free_vars: &VarSet) -> usize {
        1.max(self.1.max_width(free_vars) / self.0)
    }
}

// ----------------------------------------------------------------------------
// --- LOAD VARIABLES ---------------------------------------------------------
// ----------------------------------------------------------------------------

/// This strategy retrieves the set of free variables starting from the full set
/// of variables (known at construction time) and iteratively removing the
/// variables that have been assigned in the partial assignment leading to some
/// given frontier node.
#[derive(Debug, Clone)]
pub struct LoadVarFromPartialAssignment {
    /// The set of all the variables from the problem.
    all_vars: VarSet
}
impl LoadVarFromPartialAssignment {
    /// Creates a new load var strategy using `all_vars` as a basis for the
    /// complete set of variables.
    pub fn new(all_vars: VarSet) -> Self {
        LoadVarFromPartialAssignment {all_vars}
    }
}
impl <T> LoadVars<T> for LoadVarFromPartialAssignment {
    /// Returns the set of variables having no assigned value along the longest
    /// path to `node`.
    fn variables(&self, node: &FrontierNode<T>) -> VarSet {
        let mut vars = self.all_vars.clone();
        for decision in node.path.iter() {
            vars.remove(decision.variable);
        }
        vars
    }
}

// ----------------------------------------------------------------------------
// --- NODE SELECTION ---------------------------------------------------------
// ----------------------------------------------------------------------------

/// This function provides an implementation of the `MinLP` node selection
/// heuristic. In other words, it provides an heuristic that removes the nodes
/// having the shortest longest path from root when it needs to squash the size
/// of an overly large layer.
#[derive(Default, Debug, Copy, Clone)]
pub struct MinLP;

impl <T> NodeSelectionHeuristic<T> for MinLP {
    fn compare(&self, a: &dyn SelectableNode<T>, b: &dyn SelectableNode<T>) -> Ordering {
        a.value().cmp(&b.value())
    }
}

// ----------------------------------------------------------------------------
// --- FRONTIER ORDERING ------------------------------------------------------
// ----------------------------------------------------------------------------
/// _This is the default heuristic to set the order in which nodes are popped from the frontier_
/// So far, this is not only the default heuristic, but it is also your only
/// choice (this was configurable in the past and could possibly change back in
/// the future).
///
/// This ordering optimistically selects the most promising nodes first. That is,
/// it ranks nodes based on the upper bound reachable passing through them
/// (larger is ranked higher) and then on the length of their longest path
/// (again, longer is better).
///
/// # Note
/// This ordering considers the worst nodes as being the _least_ elements
/// of the order.
///
/// # Example
/// ```
/// # use binary_heap_plus::BinaryHeap;
/// # use compare::Compare;
/// # use ddo::implementation::heuristics::MaxUB;
/// # use ddo::common::FrontierNode;
/// # use std::sync::Arc;
/// # use ddo::common::PartialAssignment::Empty;
/// #
/// let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};
/// let b = FrontierNode {state: Arc::new('b'), lp_len:  2, ub: 100, path: Arc::new(Empty)};
/// let c = FrontierNode {state: Arc::new('c'), lp_len: 24, ub: 150, path: Arc::new(Empty)};
/// let d = FrontierNode {state: Arc::new('d'), lp_len: 13, ub:  60, path: Arc::new(Empty)};
/// let e = FrontierNode {state: Arc::new('e'), lp_len: 65, ub: 700, path: Arc::new(Empty)};
/// let f = FrontierNode {state: Arc::new('f'), lp_len: 19, ub: 100, path: Arc::new(Empty)};
///
/// let nodes = vec![a, b, c, d, e, f];
/// let mut priority_q = BinaryHeap::from_vec_cmp(nodes, MaxUB);
///
/// assert_eq!('e', *priority_q.pop().unwrap().state); // because 700 is the highest upper bound
/// assert_eq!('a', *priority_q.pop().unwrap().state); // because 300 is the next highest
/// assert_eq!('c', *priority_q.pop().unwrap().state); // idem, because of ub = 150
/// assert_eq!('f', *priority_q.pop().unwrap().state); // because ub = 100 but lp_len = 19
/// assert_eq!('b', *priority_q.pop().unwrap().state); // because ub = 100 but lp_len = 2
/// assert_eq!('d', *priority_q.pop().unwrap().state); // because ub = 13 which is the worst
/// ```
#[derive(Debug, Default, Copy, Clone)]
pub struct MaxUB;
impl <T> Compare<FrontierNode<T>> for MaxUB {
    fn compare(&self, a: &FrontierNode<T>, b: &FrontierNode<T>) -> Ordering {
        a.ub.cmp(&b.ub).then_with(|| a.lp_len.cmp(&b.lp_len))
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################


#[cfg(test)]
mod test_nbunassigned {
    use crate::abstraction::heuristics::WidthHeuristic;
    use crate::common::{Variable, VarSet};
    use crate::implementation::heuristics::NbUnassignedWitdh;

    #[test]
    fn non_empty() {
        let mut var_set = VarSet::all(5); // assume a problem with 5 variables
        var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
        var_set.remove(Variable(3));
        var_set.remove(Variable(4));      // only variables {0, 2} remain in the set

        assert_eq!(2, NbUnassignedWitdh.max_width(&var_set));
    }
    #[test]
    fn all() {
        let var_set = VarSet::all(5);
        assert_eq!(5, NbUnassignedWitdh.max_width(&var_set));
    }
    #[test]
    fn empty() {
        let mut var_set = VarSet::all(5);
        var_set.remove(Variable(0));
        var_set.remove(Variable(1));
        var_set.remove(Variable(2));
        var_set.remove(Variable(3));
        var_set.remove(Variable(4));

        assert_eq!(0, NbUnassignedWitdh.max_width(&var_set));
    }
}
#[cfg(test)]
mod test_fixedwidth {
    use crate::abstraction::heuristics::WidthHeuristic;
    use crate::common::{Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;

    #[test]
    fn non_empty() {
        let mut var_set = VarSet::all(5); // assume a problem with 5 variables
        var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
        var_set.remove(Variable(3));
        var_set.remove(Variable(4));      // only variables {0, 2} remain in the set

        assert_eq!(100, FixedWidth(100).max_width(&var_set));
    }
    #[test]
    fn all() {
        let var_set = VarSet::all(5);
        assert_eq!(100, FixedWidth(100).max_width(&var_set));
    }
    #[test]
    fn empty() {
        let mut var_set = VarSet::all(5);
        var_set.remove(Variable(0));
        var_set.remove(Variable(1));
        var_set.remove(Variable(2));
        var_set.remove(Variable(3));
        var_set.remove(Variable(4));

        assert_eq!(100, FixedWidth(100).max_width(&var_set));
    }
}
#[cfg(test)]
mod test_adapters {
    use crate::abstraction::heuristics::WidthHeuristic;
    use crate::common::{Variable, VarSet};
    use crate::implementation::heuristics::{Times, FixedWidth, DivBy};

    #[test]
    fn test_times() {
        let mut var_set = VarSet::all(5); // assume a problem with 5 variables
        var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
        var_set.remove(Variable(3));
        var_set.remove(Variable(4));      // only variables {0, 2} remain in the set

        let tested = Times(2, FixedWidth(1));
        assert_eq!(2, tested.max_width(&var_set));

        let tested = Times(3, FixedWidth(1));
        assert_eq!(3, tested.max_width(&var_set));

        let tested = Times(1, FixedWidth(10));
        assert_eq!(10, tested.max_width(&var_set));

        let tested = Times(10, FixedWidth(0));
        assert_eq!(0, tested.max_width(&var_set));

        let tested = Times(0, FixedWidth(10));
        assert_eq!(0, tested.max_width(&var_set));
    }
    #[test]
    fn test_div_by() {
        let mut var_set = VarSet::all(5); // assume a problem with 5 variables
        var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
        var_set.remove(Variable(3));
        var_set.remove(Variable(4));      // only variables {0, 2} remain in the set

        let tested = DivBy(2, FixedWidth(4));
        assert_eq!(2, tested.max_width(&var_set));

        let tested = DivBy(3, FixedWidth(9));
        assert_eq!(3, tested.max_width(&var_set));

        let tested = Times(1, FixedWidth(10));
        assert_eq!(10, tested.max_width(&var_set));

        let tested = Times(10, FixedWidth(0));
        assert_eq!(0, tested.max_width(&var_set));
    }
    #[test] #[should_panic]
    fn test_div_by_panics_when_div_by_zero() {
        let mut var_set = VarSet::all(5); // assume a problem with 5 variables
        var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
        var_set.remove(Variable(3));
        var_set.remove(Variable(4));      // only variables {0, 2} remain in the set

        let tested = Times(0, FixedWidth(10));
        assert_eq!(1, tested.max_width(&var_set));
    }
}

#[cfg(test)]
mod test_naturalorder {
    use crate::abstraction::heuristics::VariableHeuristic;
    use crate::common::{Variable, VarSet};
    use crate::implementation::heuristics::NaturalOrder;

    #[test]
    fn example() {
        let dummy = vec![0];

        let mut variables = VarSet::all(3);
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(Some(Variable(0)), NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));

        variables.remove(Variable(0)); // move on to the next layer
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(Some(Variable(1)), NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));

        variables.remove(Variable(1)); // move on to the next layer
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(Some(Variable(2)), NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));

        variables.remove(Variable(2)); // move on to the last layer, no more var to branch on
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(None, NaturalOrder.next_var(&variables, &mut current_layer, &mut next_layer));
    }
}
#[cfg(test)]
mod test_decreasing {
    use crate::abstraction::heuristics::VariableHeuristic;
    use crate::common::{Variable, VarSet};
    use crate::implementation::heuristics::Decreasing;

    #[test]
    fn example() {
        let dummy = vec![0];

        let mut variables = VarSet::all(3);
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(Some(Variable(2)), Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));

        variables.remove(Variable(2)); // move on to the next layer
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(Some(Variable(1)), Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));

        variables.remove(Variable(1)); // move on to the next layer
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(Some(Variable(0)), Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));

        variables.remove(Variable(0)); // move on to the last layer, no more var to branch on
        let mut current_layer = dummy.iter();
        let mut next_layer    = dummy.iter();
        assert_eq!(None, Decreasing.next_var(&variables, &mut current_layer, &mut next_layer));
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_minlp {
    use std::cmp::Ordering;

    use crate::abstraction::heuristics::NodeSelectionHeuristic;
    use crate::implementation::heuristics::MinLP;
    use crate::implementation::mdd::utils::NodeFlags;
    use crate::implementation::mdd::deep::mddgraph::{NodeData, NodeIndex};
    use std::rc::Rc;

    #[test]
    fn example() {
        let a = NodeData {state: Rc::new('a'), lp_from_top: 42, flags: NodeFlags::new_exact(), my_id: NodeIndex(0), inbound: None, best_parent: None, lp_from_bot: -1};
        let b = NodeData {state: Rc::new('b'), lp_from_top:  2, flags: NodeFlags::new_exact(), my_id: NodeIndex(1), inbound: None, best_parent: None, lp_from_bot: -1};
        let c = NodeData {state: Rc::new('c'), lp_from_top: 24, flags: NodeFlags::new_exact(), my_id: NodeIndex(2), inbound: None, best_parent: None, lp_from_bot: -1};
        let d = NodeData {state: Rc::new('d'), lp_from_top: 13, flags: NodeFlags::new_exact(), my_id: NodeIndex(3), inbound: None, best_parent: None, lp_from_bot: -1};
        let e = NodeData {state: Rc::new('e'), lp_from_top: 65, flags: NodeFlags::new_exact(), my_id: NodeIndex(4), inbound: None, best_parent: None, lp_from_bot: -1};
        let f = NodeData {state: Rc::new('f'), lp_from_top: 19, flags: NodeFlags::new_exact(), my_id: NodeIndex(5), inbound: None, best_parent: None, lp_from_bot: -1};

        let mut nodes = vec![a, b, c, d, e, f];
        nodes.sort_by(|a, b| MinLP.compare(a, b).reverse());
        assert_eq!(vec!['e', 'a', 'c', 'f', 'd', 'b'], nodes.iter().map(|n| *n.state).collect::<Vec<char>>());
    }

    #[test]
    fn gt_because_of_lplen() {
        let a = NodeData {state: Rc::new('a'), lp_from_top: 42, flags: NodeFlags::new_exact(), my_id: NodeIndex(0), inbound: None, best_parent: None, lp_from_bot: -1};
        let b = NodeData {state: Rc::new('b'), lp_from_top:  2, flags: NodeFlags::new_exact(), my_id: NodeIndex(1), inbound: None, best_parent: None, lp_from_bot: -1};

        assert_eq!(Ordering::Greater, MinLP.compare(&a, &b));
    }
    #[test]
    fn lt_because_of_lplen() {
        let a = NodeData {state: Rc::new('a'), lp_from_top: 42, flags: NodeFlags::new_exact(), my_id: NodeIndex(0), inbound: None, best_parent: None, lp_from_bot: -1};
        let b = NodeData {state: Rc::new('b'), lp_from_top:  2, flags: NodeFlags::new_exact(), my_id: NodeIndex(1), inbound: None, best_parent: None, lp_from_bot: -1};

        assert_eq!(Ordering::Less, MinLP.compare(&b, &a));
    }

    #[test]
    fn eq_if_all_match_but_state() {
        let a = NodeData {state: Rc::new('a'), lp_from_top: 42, flags: NodeFlags::new_exact(), my_id: NodeIndex(0), inbound: None, best_parent: None, lp_from_bot: -1};
        let b = NodeData {state: Rc::new('b'), lp_from_top: 42, flags: NodeFlags::new_exact(), my_id: NodeIndex(0), inbound: None, best_parent: None, lp_from_bot: -1};

        assert_eq!(Ordering::Equal, MinLP.compare(&a, &b));
    }
    #[test]
    fn eq_self() {
        let a = NodeData {state: Rc::new('a'), lp_from_top: 42, flags: NodeFlags::new_exact(), my_id: NodeIndex(0), inbound: None, best_parent: None, lp_from_bot: -1};

        assert_eq!(Ordering::Equal, MinLP.compare(&a, &a));
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_maxub {
    use std::cmp::Ordering;
    use std::sync::Arc;

    use binary_heap_plus::BinaryHeap;
    use compare::Compare;

    use crate::common::FrontierNode;
    use crate::common::PartialAssignment::Empty;
    use crate::implementation::heuristics::MaxUB;

    #[test]
    fn example() {
        let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};
        let b = FrontierNode {state: Arc::new('b'), lp_len:  2, ub: 100, path: Arc::new(Empty)};
        let c = FrontierNode {state: Arc::new('c'), lp_len: 24, ub: 150, path: Arc::new(Empty)};
        let d = FrontierNode {state: Arc::new('d'), lp_len: 13, ub:  60, path: Arc::new(Empty)};
        let e = FrontierNode {state: Arc::new('e'), lp_len: 65, ub: 700, path: Arc::new(Empty)};
        let f = FrontierNode {state: Arc::new('f'), lp_len: 19, ub: 100, path: Arc::new(Empty)};

        let nodes = vec![a, b, c, d, e, f];
        let mut priority_q = BinaryHeap::from_vec_cmp(nodes, MaxUB);

        assert_eq!('e', *priority_q.pop().unwrap().state); // because 700 is the highest upper bound
        assert_eq!('a', *priority_q.pop().unwrap().state); // because 300 is the next highest
        assert_eq!('c', *priority_q.pop().unwrap().state); // idem, because of ub = 150
        assert_eq!('f', *priority_q.pop().unwrap().state); // because ub = 100 but lp_len = 19
        assert_eq!('b', *priority_q.pop().unwrap().state); // because ub = 100 but lp_len = 2
        assert_eq!('d', *priority_q.pop().unwrap().state); // because ub = 13 which is the worst
    }

    #[test]
    fn gt_because_ub() {
        let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};
        let b = FrontierNode {state: Arc::new('b'), lp_len: 42, ub: 100, path: Arc::new(Empty)};

        assert_eq!(Ordering::Greater, MaxUB.compare(&a, &b));
    }
    #[test]
    fn gt_because_lplen() {
        let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};
        let b = FrontierNode {state: Arc::new('b'), lp_len:  2, ub: 300, path: Arc::new(Empty)};

        assert_eq!(Ordering::Greater, MaxUB.compare(&a, &b));
    }
    #[test]
    fn lt_because_ub() {
        let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};
        let b = FrontierNode {state: Arc::new('b'), lp_len: 42, ub: 100, path: Arc::new(Empty)};

        assert_eq!(Ordering::Less, MaxUB.compare(&b, &a));
    }
    #[test]
    fn lt_because_lplen() {
        let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};
        let b = FrontierNode {state: Arc::new('b'), lp_len:  2, ub: 300, path: Arc::new(Empty)};

        assert_eq!(Ordering::Less, MaxUB.compare(&b, &a));
    }
    #[test]
    fn eq_because_all_match() {
        let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};
        let b = FrontierNode {state: Arc::new('b'), lp_len: 42, ub: 300, path: Arc::new(Empty)};

        assert_eq!(Ordering::Equal, MaxUB.compare(&a, &b));
    }
    #[test]
    fn eq_self() {
        let a = FrontierNode {state: Arc::new('a'), lp_len: 42, ub: 300, path: Arc::new(Empty)};

        assert_eq!(Ordering::Equal, MaxUB.compare(&a, &a));
    }
}


#[cfg(test)]
mod test_load_var_from_pa {
    use std::sync::Arc;

    use crate::abstraction::dp::Problem;
    use crate::abstraction::heuristics::LoadVars;
    use crate::common::{Decision, FrontierNode, Variable};
    use crate::common::PartialAssignment::{Empty, FragmentExtension};
    use crate::implementation::heuristics::LoadVarFromPartialAssignment;
    use crate::test_utils::MockProblem;

    #[test]
    fn at_root() {
        let pb  = MockProblem::default();
        let heu = LoadVarFromPartialAssignment::new(pb.all_vars());
        let root= FrontierNode {state: Arc::new(0), lp_len: 0, ub: isize::max_value(), path: Arc::new(Empty)};

        let vars= heu.variables(&root);
        assert_eq!(pb.all_vars(), vars);
    }

    #[test]
    fn some_vars_assigned() {
        let pb  = MockProblem::default();
        let heu = LoadVarFromPartialAssignment::new(pb.all_vars());
        let path= FragmentExtension {parent: Arc::new(Empty), fragment: vec![
            Decision{variable: Variable(0), value: 1},
            Decision{variable: Variable(1), value: 0},
            Decision{variable: Variable(3), value: 0}
        ]};
        let node= FrontierNode {state: Arc::new(7), lp_len: 30, ub: isize::max_value(), path: Arc::new(path)};

        let vars= heu.variables(&node);

        let mut expected= pb.all_vars();
        expected.remove(Variable(0));
        expected.remove(Variable(1));
        expected.remove(Variable(3));
        assert_eq!(expected, vars);
    }

    #[test]
    fn all_vars_assigned() {
        let pb  = MockProblem::default();
        let heu = LoadVarFromPartialAssignment::new(pb.all_vars());
        let path= FragmentExtension {parent: Arc::new(Empty), fragment: vec![
            Decision{variable: Variable(0), value: 1},
            Decision{variable: Variable(1), value: 0},
            Decision{variable: Variable(2), value: 0},
            Decision{variable: Variable(3), value: 0},
            Decision{variable: Variable(4), value: 0},
        ]};
        let node= FrontierNode {state: Arc::new(0), lp_len: 50, ub: isize::max_value(), path: Arc::new(path)};

        let vars= heu.variables(&node);

        let mut expected= pb.all_vars();
        expected.remove(Variable(0));
        expected.remove(Variable(1));
        expected.remove(Variable(2));
        expected.remove(Variable(3));
        expected.remove(Variable(4));
        assert_eq!(expected, vars);
    }
}