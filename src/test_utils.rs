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


use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::sync::Arc;

use mock_it::Mock;

use crate::abstraction::dp::{Problem, Relaxation};
use crate::abstraction::heuristics::{LoadVars, NodeSelectionHeuristic, SelectableNode, VariableHeuristic, WidthHeuristic};
use crate::abstraction::mdd::Config;
use crate::common::{Decision, Domain, FrontierNode, Variable, VarSet};
use crate::common::PartialAssignment::Empty;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub struct Nothing;

#[derive(Clone)]
/// A mock problem definition.
pub struct MockProblem {
    pub nb_vars:          Mock<Nothing, usize>,
    pub initial_state   : Mock<Nothing, usize>,
    pub initial_value   : Mock<Nothing, isize>,
    pub domain_of       : Mock<(usize, Variable), Domain<'static>>,
    pub transition      : Mock<(usize, VarSet, Decision), usize>,
    pub transition_cost : Mock<(usize, VarSet, Decision), isize>,
    pub impacted_by     : Mock<(usize, Variable), bool>,
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
    fn initial_value(&self) -> isize {
        self.initial_value.called(Nothing)
    }
    fn domain_of<'a>(&self, state: &'a usize, var: Variable) -> Domain<'a> {
        self.domain_of.called((*state, var)) as Domain<'a>
    }
    fn transition(&self, state: &usize, vars: &VarSet, d: Decision) -> usize {
        self.transition.called((*state, vars.clone(), d))
    }
    fn transition_cost(&self, state: &usize, vars: &VarSet, d: Decision) -> isize {
        self.transition_cost.called((*state, vars.clone(), d))
    }
    fn all_vars(&self) -> VarSet {
        self.all_vars.called(Nothing)
    }
}

#[derive(Clone)]
/// A mock problem relaxation
pub struct MockRelax {
    pub merge_states: Mock<Vec<usize>, usize>,
    pub relax_edge:  Mock<(usize, usize, usize, Decision, isize), isize>,
    pub estimate: Mock<usize, isize>
}
impl Default for MockRelax {
    fn default() -> MockRelax {
        MockRelax {
            merge_states: Mock::new(0),
            relax_edge  : Mock::new(0),
            estimate : Mock::new(isize::max_value())
        }
    }
}
impl Relaxation<usize> for MockRelax {
    fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
        self.merge_states.called(states.cloned().collect::<Vec<usize>>())
    }
    fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, decision: Decision, cost: isize) -> isize {
        self.relax_edge.called((*src, *dst, *relaxed, decision, cost))
    }
    fn estimate(&self, state: &usize) -> isize {
        self.estimate.called(*state)
    }
}

#[derive(Clone)]
pub struct MockLoadVars {
    pub variables: Mock<FrontierNode<usize>, VarSet>
}
impl Default for MockLoadVars {
    fn default() -> Self {
        MockLoadVars { variables: Mock::new(VarSet::all(5)) }
    }
}
impl LoadVars<usize> for MockLoadVars {
    fn variables(&self, node: &FrontierNode<usize>) -> VarSet {
        self.variables.called(node.clone())
    }
}

#[derive(Clone)]
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

#[derive(Clone)]
pub struct MockVariableHeuristic {
    #[allow(clippy::type_complexity)]
    pub next_var: Mock<(VarSet, Vec<usize>, Vec<usize>), Option<Variable>>
}
impl Default for MockVariableHeuristic {
    fn default() -> Self {
        MockVariableHeuristic {
            next_var: Mock::new(None)
        }
    }
}
impl VariableHeuristic<usize> for MockVariableHeuristic {
    fn next_var(&self,
                free_vars: &VarSet,
                current: &mut dyn Iterator<Item=&usize>,
                next: &mut dyn Iterator<Item=&usize>) -> Option<Variable> {

        let cur = current.cloned().collect::<Vec<usize>>();
        let nxt = next.cloned().collect::<Vec<usize>>();

        self.next_var.called((free_vars.clone(), cur, nxt))
    }
}

#[derive(Clone)]
pub struct MockNodeSelectionHeuristic {
    pub compare: Mock<(usize, usize), Ordering>
}
impl Default for MockNodeSelectionHeuristic {
    fn default() -> Self {
        MockNodeSelectionHeuristic { compare: Mock::new(Ordering::Equal) }
    }
}
impl NodeSelectionHeuristic<usize> for MockNodeSelectionHeuristic {
    fn compare(&self, l: &dyn SelectableNode<usize>, r: &dyn SelectableNode<usize>) -> Ordering {
        self.compare.called((*l.state(), *r.state()))
    }
}

#[derive(Clone)]
pub struct MockConfig {
    pub root_node       : Mock<Nothing, FrontierNode<usize>>,

    pub domain_of       : Mock<(usize, Variable), Vec<isize>>,
    pub transition      : Mock<(usize, VarSet, Decision), usize>,
    pub transition_cost : Mock<(usize, VarSet, Decision), isize>,
    pub impacted_by     : Mock<(usize, Variable), bool>,

    pub merge_states    : Mock<Vec<usize>, usize>,
    pub relax_edge      : Mock<(usize, usize, usize, Decision, isize), isize>,
    pub estimate        : Mock<usize, isize>,

    #[allow(clippy::type_complexity)]
    pub select_var      : Mock<(VarSet, Vec<usize>, Vec<usize>), Option<Variable>>,
    pub load_vars       : Mock<FrontierNode<usize>, VarSet>,
    pub max_width       : Mock<VarSet, usize>,
    pub compare         : Mock<(usize, usize), Ordering>
}
impl Default for MockConfig {
    fn default() -> Self {
        MockConfig {
            root_node:      Mock::new(FrontierNode{state: Arc::new(0), lp_len: 0, ub: isize::max_value(), path: Arc::new(Empty)}),

            domain_of:      Mock::new(vec![0, 1]),
            transition:     Mock::new(0),
            transition_cost:Mock::new(0),
            impacted_by    :Mock::new(true),

            merge_states:   Mock::new(0),
            relax_edge:     Mock::new(0),
            estimate:       Mock::new(0),

            select_var:     Mock::new(None),
            load_vars:      Mock::new(VarSet::empty()),
            max_width:      Mock::new(0),
            compare:        Mock::new(Equal)
        }
    }
}
impl Config<usize> for MockConfig {
    fn root_node(&self) -> FrontierNode<usize> {
        self.root_node.called(Nothing)
    }

    fn domain_of<'a>(&self, state: &'a usize, v: Variable) -> Domain<'a> {
        Domain::from(self.domain_of.called((*state, v)))
    }

    fn transition(&self, state: &usize, vars: &VarSet, d: Decision) -> usize {
        self.transition.called((*state, vars.clone(), d))
    }

    fn transition_cost(&self, state: &usize, vars: &VarSet, d: Decision) -> isize {
        self.transition_cost.called((*state, vars.clone(), d))
    }

    fn impacted_by(&self, state: &usize, var: Variable) -> bool {
        self.impacted_by.called((*state, var))
    }

    fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
        self.merge_states.called(states.copied().collect())
    }

    fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, decision: Decision, cost: isize) -> isize {
        self.relax_edge.called((*src, *dst, *relaxed, decision, cost))
    }

    fn estimate(&self, state: &usize) -> isize {
        self.estimate.called(*state)
    }

    fn load_variables(&self, node: &FrontierNode<usize>) -> VarSet {
        self.load_vars.called(node.clone())
    }

    fn select_var(&self, free_vars: &VarSet, current_layer: &mut dyn Iterator<Item=&usize>, next_layer: &mut dyn Iterator<Item=&usize>) -> Option<Variable> {
        self.select_var.called((free_vars.clone(), current_layer.copied().collect(), next_layer.copied().collect()))
    }

    fn max_width(&self, free_vars: &VarSet) -> usize {
        self.max_width.called(free_vars.clone())
    }

    fn compare(&self, a: &dyn SelectableNode<usize>, b: &dyn SelectableNode<usize>) -> Ordering {
        self.compare.called((*a.state(), *b.state()))
    }
}


#[derive(Clone)]
pub struct Proxy<'a, T: Clone> {
    target: &'a T
}
impl <'a, T: Clone> Proxy<'a, T> {
    pub fn new(target: &'a T) -> Self {
        Proxy { target }
    }
}
impl <X, T: Relaxation<X> + Clone> Relaxation<X> for Proxy<'_, T> {
    fn merge_states(&self, states: &mut dyn Iterator<Item=&X>) -> X {
        self.target.merge_states(states)
    }

    fn relax_edge(&self, src: &X, dst: &X, relaxed: &X, decision: Decision, cost: isize) -> isize {
        self.target.relax_edge(src, dst, relaxed, decision, cost)
    }

    fn estimate(&self, state: &X) -> isize {
        self.target.estimate(state)
    }
}
impl <X, T: LoadVars<X> + Clone> LoadVars<X> for Proxy<'_, T> {
    fn variables(&self, node: &FrontierNode<X>) -> VarSet {
        self.target.variables(node)
    }
}

impl <T: WidthHeuristic + Clone> WidthHeuristic for Proxy<'_, T> {
    fn max_width(&self, free_vars: &VarSet) -> usize {
        self.target.max_width(free_vars)
    }
}
impl <X, T: VariableHeuristic<X> + Clone> VariableHeuristic<X> for Proxy<'_, T> {
    fn next_var(&self, free_vars: &VarSet, current: &mut dyn Iterator<Item=&X>, next: &mut dyn Iterator<Item=&X>) -> Option<Variable> {
        self.target.next_var(free_vars, current, next)
    }
}
impl <X, T: NodeSelectionHeuristic<X> + Clone> NodeSelectionHeuristic<X> for Proxy<'_, T> {
    fn compare(&self, l: &dyn SelectableNode<X>, r: &dyn SelectableNode<X>) -> Ordering {
        self.target.compare(l, r)
    }
}
impl <X, T: Config<X> + Clone> Config<X> for Proxy<'_, T> {
    fn root_node(&self) -> FrontierNode<X> {
        self.target.root_node()
    }

    fn domain_of<'a>(&self, state: &'a X, v: Variable) -> Domain<'a> {
        self.target.domain_of(state, v)
    }

    fn transition(&self, state: &X, vars: &VarSet, d: Decision) -> X {
        self.target.transition(state, vars, d)
    }

    fn transition_cost(&self, state: &X, vars: &VarSet, d: Decision) -> isize {
        self.target.transition_cost(state, vars, d)
    }

    fn impacted_by(&self, state: &X, var: Variable) -> bool {
        self.target.impacted_by(state, var)
    }

    fn merge_states(&self, states: &mut dyn Iterator<Item=&X>) -> X {
        self.target.merge_states(states)
    }

    fn relax_edge(&self, src: &X, dst: &X, relaxed: &X, decision: Decision, cost: isize) -> isize {
        self.target.relax_edge(src, dst, relaxed, decision, cost)
    }

    fn estimate(&self, state: &X) -> isize {
        self.target.estimate(state)
    }

    fn load_variables(&self, node: &FrontierNode<X>) -> VarSet {
        self.target.load_variables(node)
    }

    fn select_var(&self, free_vars: &VarSet, current_layer: &mut dyn Iterator<Item=&X>, next_layer: &mut dyn Iterator<Item=&X>) -> Option<Variable> {
        self.target.select_var(free_vars, current_layer, next_layer)
    }

    fn max_width(&self, free_vars: &VarSet) -> usize {
        self.target.max_width(free_vars)
    }

    fn compare(&self, a: &dyn SelectableNode<X>, b: &dyn SelectableNode<X>) -> Ordering {
        self.target.compare(a, b)
    }
}

#[derive(Clone)]
pub struct MockSelectableNode<T> {
    pub state: T,
    pub value: isize,
    pub exact: bool
}
impl <T> SelectableNode<T> for MockSelectableNode<T> {
    fn state(&self) -> &T {
        &self.state
    }

    fn value(&self) -> isize {
        self.value
    }

    fn is_exact(&self) -> bool {
        self.exact
    }
}