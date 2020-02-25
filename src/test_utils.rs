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

//! This module provides some utilities to write unit tests, the most notable
//! of which are mocks definitions for the most used types in this library.§'
#![cfg(test)]


use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::common::{Variable, VarSet, Node, Domain, Decision, NodeInfo, Layer};
use mock_it::Mock;
use crate::core::abstraction::heuristics::{LoadVars, WidthHeuristic, VariableHeuristic};
use compare::Compare;
use std::cmp::Ordering;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Nothing;

/// A mock problem definition.
pub struct MockProblem {
    pub nb_vars:          Mock<Nothing, usize>,
    pub initial_state   : Mock<Nothing, usize>,
    pub initial_value   : Mock<Nothing, i32>,
    pub domain_of       : Mock<(usize, Variable), Domain<'static>>,
    pub transition      : Mock<(usize, VarSet, Decision), usize>,
    pub transition_cost : Mock<(usize, VarSet, Decision), i32>,
    pub impacted_by     : Mock<(usize, Variable), bool>,
    pub root_node       : Mock<Nothing, Node<usize>>,
    pub all_vars        : Mock<Nothing, VarSet>
}
impl Default for MockProblem {
    fn default() -> MockProblem {
        MockProblem {
            nb_vars:         Mock::new(5),
            initial_state:   Mock::new(0),
            initial_value:   Mock::new(0),
            domain_of:       Mock::new((0..1).into()),
            transition:      Mock::new(0),
            transition_cost: Mock::new(0),
            impacted_by:     Mock::new(true),
            root_node:       Mock::new(Node::new(0, 0, None, true)),
            all_vars:        Mock::new(VarSet::all(5))
        }
    }
}
impl Problem<usize> for MockProblem {
    fn nb_vars(&self) -> usize {
        self.nb_vars.called(Nothing)
    }
    fn initial_state(&self) -> usize {
        self.initial_state.called(Nothing)
    }
    fn initial_value(&self) -> i32 {
        self.initial_value.called(Nothing)
    }
    fn domain_of<'a>(&self, state: &'a usize, var: Variable) -> Domain<'a> {
        self.domain_of.called((*state, var)) as Domain<'a>
    }
    fn transition(&self, state: &usize, vars: &VarSet, d: Decision) -> usize {
        self.transition.called((*state, vars.clone(), d))
    }
    fn transition_cost(&self, state: &usize, vars: &VarSet, d: Decision) -> i32 {
        self.transition_cost.called((*state, vars.clone(), d))
    }
    fn impacted_by(&self, state: &usize, variable: Variable) -> bool {
        self.impacted_by.called((*state, variable))
    }
    fn root_node(&self) -> Node<usize> {
        self.root_node.called(Nothing)
    }
    fn all_vars(&self) -> VarSet {
        self.all_vars.called(Nothing)
    }
}

/// A mock problem relaxation
pub struct MockRelax {
    pub merge_nodes: Mock<Vec<Node<usize>>, Node<usize>>,
    pub estimate_ub: Mock<(usize, NodeInfo), i32>
}
impl Default for MockRelax {
    fn default() -> MockRelax {
        MockRelax {
            merge_nodes: Mock::new(Node::merged(0, 0, None)),
            estimate_ub: Mock::new(i32::max_value())
        }
    }
}
impl Relaxation<usize> for MockRelax {
    fn merge_nodes(&self, nodes: &[Node<usize>]) -> Node<usize> {
        self.merge_nodes.called(nodes.iter().cloned().collect::<Vec<Node<usize>>>())
    }
    fn estimate_ub(&self, state: &usize, info: &NodeInfo) -> i32 {
        self.estimate_ub.called((*state, info.clone()))
    }
}

pub struct MockLoadVars {
    pub variables: Mock<Node<usize>, VarSet>
}
impl Default for MockLoadVars {
    fn default() -> Self {
        MockLoadVars { variables: Mock::new(VarSet::all(5)) }
    }
}
impl LoadVars<usize> for MockLoadVars {
    fn variables(&self, node: &Node<usize>) -> VarSet {
        self.variables.called(node.clone())
    }
}

pub struct MockMaxWidth {
    pub max_width: Mock<VarSet, usize>
}
impl Default for MockMaxWidth {
    fn default() -> Self {
        MockMaxWidth { max_width: Mock::new(0) }
    }
}
impl WidthHeuristic for MockMaxWidth {
    fn max_width(&self, free_vars: &VarSet) -> usize {
        self.max_width.called(free_vars.clone())
    }
}

pub struct MockVariableHeuristic {
    pub next_var: Mock<(VarSet, Vec<Node<usize>>, Vec<Node<usize>>), Option<Variable>>
}
impl Default for MockVariableHeuristic {
    fn default() -> Self {
        MockVariableHeuristic {
            next_var: Mock::new(None)
        }
    }
}
impl VariableHeuristic<usize> for MockVariableHeuristic {
    fn next_var<'a>(&self, free_vars: &'a VarSet, current: Layer<'a, usize>, next: Layer<'a, usize>) -> Option<Variable> {
        let mut cur = vec![];
        for n in current {
            cur.push(Node{ state: n.0.clone(), info: n.1.clone() });
        }

        let mut nxt= vec![];
        for n in next {
            nxt.push(Node{ state: n.0.clone(), info: n.1.clone() });
        }
        self.next_var.called((free_vars.clone(), cur, nxt))
    }
}

pub struct MockNodeSelectionHeuristic {
    pub compare: Mock<(Node<usize>, Node<usize>), Ordering>
}
impl Default for MockNodeSelectionHeuristic {
    fn default() -> Self {
        MockNodeSelectionHeuristic { compare: Mock::new(Ordering::Equal) }
    }
}
impl Compare<Node<usize>> for MockNodeSelectionHeuristic {
    fn compare(&self, l: &Node<usize>, r: &Node<usize>) -> Ordering {
        self.compare.called((l.clone(), r.clone()))
    }
}

pub struct Proxy<'a, T> {
    target: &'a T
}
impl <'a, T> Proxy<'a, T> {
    pub fn new(target: &'a T) -> Self {
        Proxy { target }
    }
}
impl <X, T: Problem<X>> Problem<X> for Proxy<'_, T> {
    fn nb_vars(&self)       -> usize { self.target.nb_vars() }
    fn initial_state(&self) -> X     { self.target.initial_state() }
    fn initial_value(&self) -> i32   { self.target.initial_value() }

    fn domain_of<'a>(&self, state: &'a X, var: Variable) -> Domain<'a> {
        self.target.domain_of(state, var)
    }

    fn transition(&self, state: &X, vars: &VarSet, d: Decision) -> X {
        self.target.transition(state, vars, d)
    }

    fn transition_cost(&self, state: &X, vars: &VarSet, d: Decision) -> i32 {
        self.target.transition_cost(state, vars, d)
    }

    fn impacted_by(&self, state: &X, variable: Variable) -> bool {
        self.target.impacted_by(state, variable)
    }

    fn root_node(&self) -> Node<X> {
        self.target.root_node()
    }

    fn all_vars(&self) -> VarSet {
        self.target.all_vars()
    }
}
impl <X, T: Relaxation<X>> Relaxation<X> for Proxy<'_, T> {
    fn merge_nodes(&self, nodes: &[Node<X>]) -> Node<X> {
        self.target.merge_nodes(nodes)
    }

    fn estimate_ub(&self, state: &X, info: &NodeInfo) -> i32 {
        self.target.estimate_ub(state, info)
    }
}
impl <X, T: LoadVars<X>> LoadVars<X> for Proxy<'_, T> {
    fn variables(&self, node: &Node<X>) -> VarSet {
        self.target.variables(node)
    }
}

impl <T: WidthHeuristic> WidthHeuristic for Proxy<'_, T> {
    fn max_width(&self, free_vars: &VarSet) -> usize {
        self.target.max_width(free_vars)
    }
}
impl <X, T: VariableHeuristic<X>> VariableHeuristic<X> for Proxy<'_, T> {
    fn next_var<'a>(&self, free_vars: &'a VarSet, current: Layer<'a, X>, next: Layer<'a, X>) -> Option<Variable> {
        self.target.next_var(free_vars, current, next)
    }
}
impl <X, T: Compare<Node<X>>> Compare<Node<X>> for Proxy<'_, T> {
    fn compare(&self, l: &Node<X>, r: &Node<X>) -> Ordering {
        self.target.compare(l, r)
    }
}