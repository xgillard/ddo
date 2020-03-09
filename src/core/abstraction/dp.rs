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

//! This module defines the basic abstractions one will manipulate when
//! implementing an MDD optimization-solver for some problem formulated as a
//! dynamic program.
//!
//! The most important abstractions that should be provided by a client are
//! `Problem` and `Relaxation`.
use std::i32;

use crate::core::common::{Decision, Variable, VarSet, Node, NodeInfo, Domain};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// This is the main abstraction that should be provided by any user of our
/// library. Indeed, it defines the problem to be solved in the form of a dynamic
/// program. Therefore, this trait closely sticks to the formal definition of a
/// dynamic program.
///
/// The type parameter `<T>` denotes the type of the states of the dynamic program.
pub trait Problem<T> {
    /// Returns the number of decision variables that play a role in the problem.
    fn nb_vars(&self) -> usize;
    /// Returns the initial state of the problem (when no decision is taken).
    fn initial_state(&self) -> T;
    /// Returns the initial value of the objective function (when no decision is taken).
    fn initial_value(&self) -> i32;

    /// Returns the domain of variable `var` in the given `state`. These are the
    /// possible values that might possibly be affected to `var` when the system
    /// has taken decisions leading to `state`.
    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a>;
    /// Returns the next state reached by the system if the decision `d` is
    /// taken when the system is in the given `state` and the given set of `vars`
    /// are still free (no value assigned).
    fn transition(&self, state: &T, vars : &VarSet, d: Decision) -> T;
    /// Returns the marginal benefit (in terms of objective function to maximize)
    /// of taking decision `d` is when the system is in the given `state` and
    /// the given set of `vars` are still free (no value assigned).
    fn transition_cost(&self, state: &T, vars : &VarSet, d: Decision) -> i32;

    /// Optional method for the case where you'd want to use a pooled mdd
    /// implementation. This method returns true iff taking a decision on
    /// `variable` might have an impact (state or longest path) on a node having
    /// the given `state`
    #[allow(unused_variables)]
    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        true
    }

    /// Returns the root node of an exact MDD standing for this problem.
    ///
    /// This method is (trivially) auto-implemented, but re-implementing it
    /// does not make much sense.
    fn root_node(&self) -> Node<T> {
        Node::new(self.initial_state(), self.initial_value(), None, true, false)
    }
    /// Returns a var set with all the variables of this problem.
    ///
    /// This method is (trivially) auto-implemented, but re-implementing it
    /// does not make much sense.
    fn all_vars(&self) -> VarSet {
        VarSet::all(self.nb_vars())
    }
}

/// This is the second most important abstraction that a client should provide
/// when using this library. It defines the relaxation that may be applied to
/// the given problem. In particular, the `merge_nodes` method from this trait
/// defines how the nodes of a layer may be combined to provide an upper bound
/// approximation standing for an arbitrarily selected set of nodes.
///
/// Again, the type parameter `<T>` denotes the type of the states.
pub trait Relaxation<T> {
    /// This method merges the given set of `nodes` into a new _inexact_ node.
    /// The role of this method is really to _only_ provide an inexact
    /// node to use as a replacement for the selected `nodes`. It is the MDD that
    /// takes care of maintaining a cutset.
    fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T>;
    /// This optional method derives a _rough_ upper bound on the maximum value
    /// reachable by passing through the node characterized by the given `state`
    /// and node `info`.
    fn estimate_ub(&self, _state: &T, _info: &NodeInfo) -> i32 {
        i32::max_value()
    }
}

/// Any reference to a problem is also a problem
impl <T, P: Problem<T>> Problem<T> for &P {
    fn nb_vars(&self) -> usize {
        (*self).nb_vars()
    }

    fn initial_state(&self) -> T {
        (*self).initial_state()
    }

    fn initial_value(&self) -> i32 {
        (*self).initial_value()
    }

    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a> {
        (*self).domain_of(state, var)
    }

    fn transition(&self, state: &T, vars: &VarSet, d: Decision) -> T {
        (*self).transition(state, vars, d)
    }

    fn transition_cost(&self, state: &T, vars: &VarSet, d: Decision) -> i32 {
        (*self).transition_cost(state, vars, d)
    }

    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        (*self).impacted_by(state, variable)
    }

    fn root_node(&self) -> Node<T> {
        (*self).root_node()
    }

    fn all_vars(&self) -> VarSet {
        (*self).all_vars()
    }
}

/// Any reference to a problem is also a problem
impl <T, P: Problem<T>> Problem<T> for Rc<P> {
    fn nb_vars(&self) -> usize {
        self.as_ref().nb_vars()
    }

    fn initial_state(&self) -> T {
        self.as_ref().initial_state()
    }

    fn initial_value(&self) -> i32 {
        self.as_ref().initial_value()
    }

    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a> {
        self.as_ref().domain_of(state, var)
    }

    fn transition(&self, state: &T, vars: &VarSet, d: Decision) -> T {
        self.as_ref().transition(state, vars, d)
    }

    fn transition_cost(&self, state: &T, vars: &VarSet, d: Decision) -> i32 {
        self.as_ref().transition_cost(state, vars, d)
    }

    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        self.as_ref().impacted_by(state, variable)
    }

    fn root_node(&self) -> Node<T> {
        self.as_ref().root_node()
    }

    fn all_vars(&self) -> VarSet {
        self.as_ref().all_vars()
    }
}

/// Any reference to a problem is also a problem
impl <T, P: Problem<T>> Problem<T> for Arc<P> {
    fn nb_vars(&self) -> usize {
        self.as_ref().nb_vars()
    }

    fn initial_state(&self) -> T {
        self.as_ref().initial_state()
    }

    fn initial_value(&self) -> i32 {
        self.as_ref().initial_value()
    }

    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a> {
        self.as_ref().domain_of(state, var)
    }

    fn transition(&self, state: &T, vars: &VarSet, d: Decision) -> T {
        self.as_ref().transition(state, vars, d)
    }

    fn transition_cost(&self, state: &T, vars: &VarSet, d: Decision) -> i32 {
        self.as_ref().transition_cost(state, vars, d)
    }

    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        self.as_ref().impacted_by(state, variable)
    }

    fn root_node(&self) -> Node<T> {
        self.as_ref().root_node()
    }

    fn all_vars(&self) -> VarSet {
        self.as_ref().all_vars()
    }
}

/// Blanket implementation of Problem for any refcell to a problem
impl <T, P: Problem<T>> Problem<T> for RefCell<P> {
    fn nb_vars(&self) -> usize {
        self.borrow().nb_vars()
    }

    fn initial_state(&self) -> T {
        self.borrow().initial_state()
    }

    fn initial_value(&self) -> i32 {
        self.borrow().initial_value()
    }

    fn domain_of<'a>(&self, state: &'a T, var: Variable) -> Domain<'a> {
        self.borrow().domain_of(state, var)
    }

    fn transition(&self, state: &T, vars: &VarSet, d: Decision) -> T {
        self.borrow().transition(state, vars, d)
    }

    fn transition_cost(&self, state: &T, vars: &VarSet, d: Decision) -> i32 {
        self.borrow().transition_cost(state, vars, d)
    }

    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        self.borrow().impacted_by(state, variable)
    }

    fn root_node(&self) -> Node<T> {
        self.borrow().root_node()
    }

    fn all_vars(&self) -> VarSet {
        self.borrow().all_vars()
    }
}

#[cfg(test)]
mod test_problem_defaults {
    use crate::core::abstraction::dp::{Problem, Relaxation};
    use crate::core::common::{Variable, VarSet, Domain, Decision, Node, NodeInfo};

    struct MockProblem;
    impl Problem<usize> for MockProblem {
        fn nb_vars(&self)       -> usize {  5 }
        fn initial_state(&self) -> usize { 42 }
        fn initial_value(&self) -> i32   { 84 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            unimplemented!()
        }
        fn transition(&self, _: &usize, _: &VarSet, _: Decision) -> usize {
            unimplemented!()
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, _: Decision) -> i32 {
            unimplemented!()
        }
    }
    struct MockRelax;
    impl Relaxation<usize> for MockRelax {
        fn merge_nodes(&self, _: &[Node<usize>]) -> Node<usize> {
            unimplemented!()
        }
    }

    #[test]
    fn by_default_all_vars_impact_all_states() {
        assert!(MockProblem.impacted_by(&0, Variable(0)));
        assert!(MockProblem.impacted_by(&4, Variable(10)));
        assert!(MockProblem.impacted_by(&92, Variable(53)));
    }

    #[test]
    fn by_default_all_vars_return_all_possible_variables(){
        assert_eq!(VarSet::all(5), MockProblem.all_vars());
    }

    #[test]
    fn by_default_the_root_node_consitsts_of_the_initial_state_and_value() {
        let node = Node::new(MockProblem.initial_state(), MockProblem.initial_value(), None, true, false);
        assert_eq!(node, MockProblem.root_node());
    }

    #[test]
    fn the_default_rough_upper_bound_is_infinity() {
        let info = NodeInfo::new(12, None, true, false);
        assert_eq!(i32::max_value(), MockRelax.estimate_ub(&12, &info));
    }
}

#[cfg(test)]
mod test_blanket {
    use crate::test_utils::MockProblem;
    use crate::core::abstraction::dp::{Relaxation, Problem};
    use crate::core::common::Node;
    use std::cell::RefCell;
    use std::rc::Rc;
    use crate::core::implementation::mdd::builder::mdd_builder;
    use crate::core::abstraction::mdd::{MDD, MDDType};

    struct DummyRelax<P> {
        p: Rc<RefCell<P>>
    }
    impl <P> DummyRelax<P> {
        fn new(p: Rc<RefCell<P>>) -> Self {
            DummyRelax {p}
        }
    }
    impl <T, P> Relaxation<T> for DummyRelax<P> where P: Problem<T> {
        fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T> {
            self.p.as_ptr();
            unimplemented!()
        }
    }

    #[test]
    fn one_can_build_an_internally_mutable_problem() {
        let pb  = Rc::new(RefCell::new(MockProblem::default()));
        let rlx = DummyRelax::new(Rc::clone(&pb));
        let mut mdd = mdd_builder(pb, rlx).build();

        assert_eq!(MDDType::Exact, mdd.mdd_type());
    }
}