//! This submodule provides an implementation for the most common (default)
//! heuristics one is likely to use when solving a problem.
use std::cmp::Ordering;

use compare::Compare;

use crate::core::abstraction::dp::Problem;
use crate::core::abstraction::heuristics::{LoadVars, VariableHeuristic, WidthHeuristic};
use crate::core::common::{Layer, Node, Variable, VarSet};

//~~~~~ Max Width Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// **This is the default max width heuristic**.
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
/// # use ddo::core::common::{Variable, VarSet};
/// # use ddo::core::abstraction::heuristics::WidthHeuristic;
/// # use ddo::core::implementation::heuristics::NbUnassigned;
///
/// let mut var_set = VarSet::all(5); // assume a problem with 5 variables
/// var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
/// var_set.remove(Variable(3));
/// var_set.remove(Variable(4));      // only variables {0, 2} remain in the set
///
/// assert_eq!(2, NbUnassigned.max_width(&var_set));
/// ```
#[derive(Debug, Clone)]
pub struct NbUnassigned;
impl WidthHeuristic for NbUnassigned {
    fn max_width(&self, free: &VarSet) -> usize {
        free.len()
    }
}

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
/// # use ddo::core::common::{Variable, VarSet};
/// # use ddo::core::abstraction::heuristics::WidthHeuristic;
/// # use ddo::core::implementation::heuristics::FixedWidth;
///
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
#[derive(Debug, Clone)]
pub struct FixedWidth(pub usize);
impl WidthHeuristic for FixedWidth {
    fn max_width(&self, _free: &VarSet) -> usize {
        self.0
    }
}


//~~~~~ Variable Heuristics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// **This is the default variable branching heuristic**
/// This strategy branches on the variables in their ''natural'' order. That is,
/// it will first pick `Variable(0)` then `Variable(1)`, `Variable(2)`, etc...
/// until there are no variables left to branch on.
///
/// # Example
/// ```
/// # use ddo::core::common::{Variable, VarSet};
/// # use ddo::core::abstraction::heuristics::VariableHeuristic;
/// # use ddo::core::implementation::heuristics::NaturalOrder;
/// # use ddo::core::common::Layer;
/// # let dummy = vec![];
///
/// let mut variables = VarSet::all(3);
/// # let current_layer = Layer::<u8>::Plain(dummy.iter());
/// # let next_layer    = Layer::<u8>::Plain(dummy.iter());
/// assert_eq!(Some(Variable(0)), NaturalOrder.next_var(&variables, current_layer, next_layer));
///
/// variables.remove(Variable(0)); // move on to the next layer
/// # let current_layer = Layer::<u8>::Plain(dummy.iter());
/// # let next_layer    = Layer::<u8>::Plain(dummy.iter());
/// assert_eq!(Some(Variable(1)), NaturalOrder.next_var(&variables, current_layer, next_layer));
///
/// variables.remove(Variable(1)); // move on to the next layer
/// # let current_layer = Layer::<u8>::Plain(dummy.iter());
/// # let next_layer    = Layer::<u8>::Plain(dummy.iter());
/// assert_eq!(Some(Variable(2)), NaturalOrder.next_var(&variables, current_layer, next_layer));
///
/// variables.remove(Variable(2)); // move on to the last layer, no more var to branch on
/// # let current_layer = Layer::<u8>::Plain(dummy.iter());
/// # let next_layer    = Layer::<u8>::Plain(dummy.iter());
/// assert_eq!(None, NaturalOrder.next_var(&variables, current_layer, next_layer));
/// ```
///
/// # Note:
/// Even though any variable heuristic may access the current and next layers
/// of the mdd being developed, the natural ordering heuristic does not use that
/// access.
///
#[derive(Default, Debug, Clone)]
pub struct NaturalOrder;
impl <T> VariableHeuristic<T> for NaturalOrder {
    fn next_var<'a>(&self, free_vars: &'a VarSet, _c: Layer<'a, T>, _n: Layer<'a, T>) -> Option<Variable> {
        free_vars.iter().next()
    }
}

//~~~~~ Node Ordering Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// **This is the default node selection heuristic**
/// This heuristic orders the nodes from best to worst by considering the nodes
/// having a minimal longest path as being the wost.
///
/// `MinLP` considers the worst nodes as the ones being the _least_ according
/// to this ordering. This means that in order to sort a vector of nodes from
/// best to worst (ie. to implement restriction), the ordering should be reversed.
///
/// # Example
/// ```
/// # use compare::Compare;
/// # use ddo::core::common::{Node, NodeInfo};
/// # use ddo::core::implementation::heuristics::MinLP;
///
/// let a = Node {state: 'a', info: NodeInfo{lp_len: 42, lp_arc: None, ub: 100, is_exact: true}};
/// let b = Node {state: 'b', info: NodeInfo{lp_len:  2, lp_arc: None, ub: 100, is_exact: true}};
/// let c = Node {state: 'c', info: NodeInfo{lp_len: 24, lp_arc: None, ub: 100, is_exact: true}};
/// let d = Node {state: 'd', info: NodeInfo{lp_len: 13, lp_arc: None, ub: 100, is_exact: true}};
/// let e = Node {state: 'e', info: NodeInfo{lp_len: 65, lp_arc: None, ub: 100, is_exact: true}};
/// let f = Node {state: 'f', info: NodeInfo{lp_len: 19, lp_arc: None, ub: 100, is_exact: true}};
///
/// let mut nodes = vec![&a, &b, &c, &d, &e, &f];
/// nodes.sort_by(|x, y| MinLP.compare(x, y).reverse());
/// assert_eq!(vec![&e, &a, &c, &f, &d, &b], nodes);
/// ```
#[derive(Debug, Default, Clone)]
pub struct MinLP;
impl <T> Compare<Node<T>> for MinLP {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.lp_len.cmp(&b.info.lp_len)
    }
}

/// **This is the default heuristic to set the order in which nodes are popped from the fringe**
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
/// # use ddo::core::common::{Node, NodeInfo};
/// # use ddo::core::implementation::heuristics::MaxUB;
///
/// let a = Node {state: 'a', info: NodeInfo{lp_len: 42, lp_arc: None, ub: 300, is_exact: true}};
/// let b = Node {state: 'b', info: NodeInfo{lp_len:  2, lp_arc: None, ub: 100, is_exact: true}};
/// let c = Node {state: 'c', info: NodeInfo{lp_len: 24, lp_arc: None, ub: 150, is_exact: true}};
/// let d = Node {state: 'd', info: NodeInfo{lp_len: 13, lp_arc: None, ub:  60, is_exact: true}};
/// let e = Node {state: 'e', info: NodeInfo{lp_len: 65, lp_arc: None, ub: 700, is_exact: true}};
/// let f = Node {state: 'f', info: NodeInfo{lp_len: 19, lp_arc: None, ub: 100, is_exact: true}};
///
/// let nodes = vec![a.clone(), b.clone(), c.clone(), d.clone(), e.clone(), f.clone()];
/// let mut priority_q = BinaryHeap::from_vec_cmp(nodes, MaxUB);
///
/// assert_eq!(e, priority_q.pop().unwrap()); // because 700 is the highest upper bound
/// assert_eq!(a, priority_q.pop().unwrap()); // because 300 is the next highest
/// assert_eq!(c, priority_q.pop().unwrap()); // idem, because of ub = 150
/// assert_eq!(f, priority_q.pop().unwrap()); // because ub = 100 but lp_len = 19
/// assert_eq!(b, priority_q.pop().unwrap()); // because ub = 100 but lp_len = 2
/// assert_eq!(d, priority_q.pop().unwrap()); // because ub = 13 which is the worst
/// ```
#[derive(Debug, Default, Clone)]
pub struct MaxUB;
impl <T> Compare<Node<T>> for MaxUB {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.ub.cmp(&b.info.ub).then_with(|| a.info.lp_len.cmp(&b.info.lp_len))
    }
}

//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// **This is the default strategy to recover the set of free variables from a node**
/// This strategy retrieves the set of free variables from a node by walking the
/// set of variables assigned along the longest path to the current node.
///
/// It starts from the full set of variables (`problem.all_vars()`) and
/// iteratively removes the variables that have been assigned a value on the
/// longest path.
#[derive(Debug, Clone)]
pub struct FromLongestPath<'a, P> {
    /// This is a (private) reference to the problem being solved.
    pb: &'a P
}
impl <'a, P> FromLongestPath<'a, P> {
    /// This function creates a `FromLongestPath` heuristic from a reference
    /// to the problem being solved.
    pub fn new(pb: &'a P) -> FromLongestPath<'a, P>{
        FromLongestPath {pb}
    }
}
impl <'a, T, P> LoadVars<T> for FromLongestPath<'a, P> where P: Problem<T> {
    /// Returns the set of variables having no assigned value along the longest
    /// path to `node`.
    fn variables(&self, node: &Node<T>) -> VarSet {
        let mut vars = self.pb.all_vars();
        for d in node.info.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}