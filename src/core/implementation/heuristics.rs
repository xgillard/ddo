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
///
/// let mut variables = VarSet::all(3);
/// # let current_layer = Layer::Plain(vec![].iter());
/// # let next_layer    = Layer::Plain(vec![].iter());
/// assert_eq!(Some(Variable(0)), NaturalOrder.next_var(&variables, current_layer, next_layer));
///
/// variables.remove(Variable(0)); // move on to the next layer
/// # let current_layer = Layer::Plain(vec![].iter());
/// # let next_layer    = Layer::Plain(vec![].iter());
/// assert_eq!(Some(Variable(1)), NaturalOrder.next_var(&variables, current_layer, next_layer));
///
/// variables.remove(Variable(1)); // move on to the next layer
/// # let current_layer = Layer::Plain(vec![].iter());
/// # let next_layer    = Layer::Plain(vec![].iter());
/// assert_eq!(Some(Variable(2)), NaturalOrder.next_var(&variables, current_layer, next_layer));
///
/// variables.remove(Variable(2)); // move on to the last layer, no more var to branch on
/// # let current_layer = Layer::Plain(vec![].iter());
/// # let next_layer    = Layer::Plain(vec![].iter());
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
#[derive(Debug, Default, Clone)]
pub struct MinLP;
impl <T> Compare<Node<T>> for MinLP {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.lp_len.cmp(&b.info.lp_len)
    }
}
#[derive(Debug, Default, Clone)]
pub struct MaxLP;
impl <T> Compare<Node<T>> for MaxLP {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.lp_len.cmp(&b.info.lp_len).reverse()
    }
}

#[derive(Debug, Default, Clone)]
pub struct MaxUB;
impl <T> Compare<Node<T>> for MaxUB {
    fn compare(&self, a: &Node<T>, b: &Node<T>) -> Ordering {
        a.info.ub.cmp(&b.info.ub).then_with(|| a.info.lp_len.cmp(&b.info.lp_len))
    }
}

//~~~~~ Load Vars Strategies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Debug, Clone)]
pub struct FromLongestPath<'a, P> {
    pb: &'a P
}
impl <'a, P> FromLongestPath<'a, P> {
    pub fn new(pb: &'a P) -> FromLongestPath<'a, P>{
        FromLongestPath {pb}
    }
}
impl <'a, T, P> LoadVars<T> for FromLongestPath<'a, P> where P: Problem<T> {
    fn variables(&self, node: &Node<T>) -> VarSet {
        let mut vars = self.pb.all_vars();
        for d in node.info.longest_path() {
            vars.remove(d.variable);
        }
        vars
    }
}