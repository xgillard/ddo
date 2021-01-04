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

//! This module provides the implementation of a structure implementing the
//! `Config` trait that encapsulates the configuration of an MDD. It also
//! provides a convenient builder to instantiate these configuration objects
//! and derive MDDs from them in an intelligible way.

use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::abstraction::dp::{Problem, Relaxation};
use crate::abstraction::heuristics::{LoadVars, NodeSelectionHeuristic, SelectableNode, VariableHeuristic, WidthHeuristic, Cutoff};
use crate::abstraction::mdd::Config;
use crate::common::{Decision, Domain, FrontierNode, PartialAssignment, Variable, VarSet};
use crate::implementation::heuristics::{LoadVarFromPartialAssignment, MinLP, NaturalOrder, NbUnassignedWitdh, NoCutoff};
use crate::implementation::mdd::deep::mdd::DeepMDD;
use crate::implementation::mdd::shallow::flat::FlatMDD;
use crate::implementation::mdd::shallow::pooled::PooledMDD;

/// This is the function you should use to instantiate a new MDD builder with
/// all defaults heuristics. It should be used as in the following example where
/// one creates an MDD whaving a fixed width strategy.
///
/// # Example:
/// ```
/// # use ddo::*;
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #   fn nb_vars(&self)       -> usize { 0 }
/// #   fn initial_state(&self) -> usize { 0 }
/// #   fn initial_value(&self) -> isize { 0 }
/// #   fn domain_of<'a>(&self,state: &'a usize,var: Variable) -> Domain<'a> {
/// #       (0..1).into()
/// #   }
/// #   fn transition(&self,state: &usize,vars: &VarSet,d: Decision)      -> usize { 0 }
/// #   fn transition_cost(&self,state: &usize,vars: &VarSet,d: Decision) -> isize { 0 }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #    fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
/// #      states.cloned().max().unwrap()
/// #    }
/// #    fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, decision: Decision, cost: isize) -> isize {
/// #      cost
/// #    }
/// # }
/// let problem = MockProblem;
/// let relax   = MockRelax;
/// let mdd     = mdd_builder(&problem, relax)
///                  .with_max_width(FixedWidth(100))
///                  .into_deep();
/// ```
pub fn mdd_builder<T, PB, RLX>(pb: &PB, rlx: RLX) -> ConfigurationBuilder<T, PB, RLX>
    where T: Hash + Eq + Clone, PB: Problem<T> + Clone, RLX: Relaxation<T> + Clone {
    config_builder(pb, rlx)
}

/// This is the function you should use to instantiate a new configuration builder
/// with all defaults heuristics. It should be used as in the following example
/// where one creates an MDD whaving a fixed width strategy.
///
/// The advantage of using `config_builder` over using `mdd_builder` is simply
/// that it adds readability. `config_builder` should be used when you intend
/// to use a custom mdd type (ie an hybrid mdd) whereas `mdd builder should be
/// used when you want to create an mdd by calling the one of its `into_xxx`
/// methods.
///
/// Here is an example of how to use the config_builder in order to instanciate
/// an hybrid flat-deep mdd.
///
/// # Example:
/// ```
/// # use ddo::*;
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #   fn nb_vars(&self)       -> usize { 0 }
/// #   fn initial_state(&self) -> usize { 0 }
/// #   fn initial_value(&self) -> isize { 0 }
/// #   fn domain_of<'a>(&self,state: &'a usize,var: Variable) -> Domain<'a> {
/// #       (0..1).into()
/// #   }
/// #   fn transition(&self,state: &usize,vars: &VarSet,d: Decision)      -> usize { 0 }
/// #   fn transition_cost(&self,state: &usize,vars: &VarSet,d: Decision) -> isize { 0 }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #    fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
/// #      states.cloned().max().unwrap()
/// #    }
/// #    fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, decision: Decision, cost: isize) -> isize {
/// #      cost
/// #    }
/// # }
/// let problem = MockProblem;
/// let relax   = MockRelax;
/// let config  = config_builder(&problem, relax)
///                  .with_max_width(FixedWidth(100))
///                  .build();
/// // This is nothing but an example
/// let mdd     = HybridFlatDeep::from(config);
/// ```
pub fn config_builder<T, PB, RLX>(pb: &PB, rlx: RLX) -> ConfigurationBuilder<T, PB, RLX>
    where T: Hash + Eq + Clone, PB: Problem<T> + Clone, RLX: Relaxation<T> + Clone {
    let lv = LoadVarFromPartialAssignment::new(pb.all_vars());
    ConfigurationBuilder {
        pb, rlx, lv,
        vs : NaturalOrder,
        w  : NbUnassignedWitdh,
        ns : MinLP,
        cutoff: NoCutoff,
        _phantom: PhantomData
    }
}


/// This is the structure used to build an MDD configuration. There is very little
/// logic to it: it only uses the type system to adapt to its configurations and
/// return a config which may be used efficiently (and stack allocated).
/// Concretely, a configuration builder lets you specify all the parameters of a
/// candidate configuration to build. Namely:
///  + a problem (mandatory)
///  + a relaxation (mandatory)
///  + a load variable heuristic (Defaults to `LoadVarFromPartialAssignment`)
///  + a variable selection heuristic (select the next var to branch on. Defaults to `NaturalOrder`.)
///  + a maximum width heuristic to limit the memory usage of each layer. (Defaults to `NbUnassignedWidth`.)
///  + a node selection heuritic (Ordering to chose the nodes to drop/merge. Defaults to `MinLP`).
///
/// Such a builder should not be _manually_ created: it would be cumbersome and
/// bring no usage benefit. Instead, it should be instantiated using the
/// associated `mdd_builder` function which is much simpler and does the exact
/// same job. Here is an example of how one is supposed to use the builder to
/// create an MDD with a fixed width (all other parameters can be tuned).
///
/// # Example:
/// ```
/// # use ddo::*;
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #   fn nb_vars(&self)       -> usize { 0 }
/// #   fn initial_state(&self) -> usize { 0 }
/// #   fn initial_value(&self) -> isize { 0 }
/// #   fn domain_of<'a>(&self,state: &'a usize,var: Variable) -> Domain<'a> {
/// #       (0..1).into()
/// #   }
/// #   fn transition(&self,state: &usize,vars: &VarSet,d: Decision)      -> usize { 0 }
/// #   fn transition_cost(&self,state: &usize,vars: &VarSet,d: Decision) -> isize { 0 }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #    fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
/// #      states.cloned().max().unwrap()
/// #    }
/// #    fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, decision: Decision, cost: isize) -> isize {
/// #      cost
/// #    }
/// # }
/// let problem = MockProblem;
/// let relax   = MockRelax;
/// let mdd     = mdd_builder(&problem, relax)
///                  .with_max_width(FixedWidth(100))
///                  .into_deep();
/// ```
pub struct ConfigurationBuilder<'x, T, P, R,
    L = LoadVarFromPartialAssignment,
    V = NaturalOrder,
    W = NbUnassignedWitdh,
    S = MinLP,
    C = NoCutoff>
    where T: Hash + Eq + Clone,
          P: Problem<T> + Clone,
          R: Relaxation<T> + Clone,
          L: LoadVars<T> + Clone,
          V: VariableHeuristic<T> + Clone,
          W: WidthHeuristic + Clone,
          S: NodeSelectionHeuristic<T> + Clone,
          C: Cutoff + Clone {
    pb  : &'x P,
    rlx : R,
    lv  : L,
    vs  : V,
    w   : W,
    ns  : S,
    cutoff: C,
    _phantom : PhantomData<T>
}


/// The following methods define the behavior of an mdd builder.
impl <'x, T, PB, RLX, LV, VS, WIDTH, NS, C> ConfigurationBuilder<'x, T, PB, RLX, LV, VS, WIDTH, NS, C>
    where T    : Eq + Hash + Clone,
          PB   : Problem<T> + Clone,
          RLX  : Relaxation<T> + Clone,
          LV   : LoadVars<T> + Clone,
          VS   : VariableHeuristic<T> + Clone,
          WIDTH: WidthHeuristic + Clone,
          NS   : NodeSelectionHeuristic<T> + Clone,
          C    : Cutoff + Clone {

    /// This is how you specify the load variable heuristic to use.
    pub fn with_load_vars<H: LoadVars<T> + Clone>(self, h: H) -> ConfigurationBuilder<'x, T, PB, RLX, H, VS, WIDTH, NS, C> {
        ConfigurationBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : h,
            vs : self.vs,
            w  : self.w,
            ns : self.ns,
            cutoff: self.cutoff,
            _phantom: Default::default()
        }
    }
    /// This is how you specify the branch heuristic to use (the variable selection heuristic).
    pub fn with_branch_heuristic<H: VariableHeuristic<T> + Clone>(self, h: H) -> ConfigurationBuilder<'x, T, PB, RLX, LV, H, WIDTH, NS, C> {
        ConfigurationBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : self.lv,
            vs : h,
            w  : self.w,
            ns : self.ns,
            cutoff: self.cutoff,
            _phantom: Default::default()
        }
    }
    /// This is how you specify the maximum width heuristic to use (to constrain
    /// the max width of MDD layers).
    pub fn with_max_width<H: WidthHeuristic + Clone>(self, h: H) -> ConfigurationBuilder<'x, T, PB, RLX, LV, VS, H, NS, C> {
        ConfigurationBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : self.lv,
            vs : self.vs,
            w  : h,
            ns : self.ns,
            cutoff: self.cutoff,
            _phantom: Default::default()
        }
    }
    /// This is how you specify the nodes selection heuristic to use (to decide
    /// what nodes to merge/drop in case the layer width is too large).
    pub fn with_nodes_selection_heuristic<H: NodeSelectionHeuristic<T> + Clone>(self, h: H) -> ConfigurationBuilder<'x, T, PB, RLX, LV, VS, WIDTH, H, C> {
        ConfigurationBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : self.lv,
            vs : self.vs,
            w  : self.w,
            ns : h,
            cutoff: self.cutoff,
            _phantom: Default::default()
        }
    }
    /// This is how you specify the nodes selection heuristic to use (to decide
    /// what nodes to merge/drop in case the layer width is too large).
    pub fn with_cutoff<X: Cutoff + Clone>(self, c: X) -> ConfigurationBuilder<'x, T, PB, RLX, LV, VS, WIDTH, NS, X> {
        ConfigurationBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : self.lv,
            vs : self.vs,
            w  : self.w,
            ns : self.ns,
            cutoff: c,
            _phantom: Default::default()
        }
    }

    /// This is how you instantiate a configuration object. This is not really
    /// useful per-se, unless you decide to implement your own kind of MDD and
    /// want to be able to reuse a single configuration object.
    pub fn config(self) -> PassThroughConfig<'x, T, PB, RLX, LV, VS, WIDTH, NS, C> {
        PassThroughConfig {
            problem  : self.pb,
            relax    : self.rlx,
            load_var : self.lv,
            var_heu  : self.vs,
            width_heu: self.w,
            select   : self.ns,
            cutoff   : self.cutoff,
            _phantom : self._phantom
        }
    }
    /// This is how you instantiate an MDD (using the default MDD implementation)
    /// configured with the parameters you specified.
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn build(self) -> PassThroughConfig<'x, T, PB, RLX, LV, VS, WIDTH, NS, C> {
        self.config()
    }
    /// This is how you instantiate a _deep_ MDD from using your desired
    /// configuration.
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn into_deep(self) -> DeepMDD<T, PassThroughConfig<'x, T, PB, RLX, LV, VS, WIDTH, NS, C>> {
        DeepMDD::new(self.config())
    }
    /// This is how you instantiate a _flat_ MDD from using your desired
    /// configuration.
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn into_flat(self) -> FlatMDD<T, PassThroughConfig<'x, T, PB, RLX, LV, VS, WIDTH, NS, C>> {
        FlatMDD::new(self.config())
    }
    /// This is how you instantiate a _pooled_ MDD from using your desired
    /// configuration.
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn into_pooled(self) -> PooledMDD<T, PassThroughConfig<'x, T, PB, RLX, LV, VS, WIDTH, NS, C>> {
        PooledMDD::new(self.config())
    }
}

/// This structure provides a simple 'umbrella' over the configuration of an MDD.
/// All it does it to forward calls to the appropriate heuristic.
#[derive(Debug, Clone)]
pub struct PassThroughConfig<'x, T, P, R, L, V, W, S, C>
    where T: Hash + Eq + Clone,
          P: Problem<T> + Clone,
          R: Relaxation<T> + Clone,
          L: LoadVars<T> + Clone,
          V: VariableHeuristic<T> + Clone,
          W: WidthHeuristic + Clone,
          S: NodeSelectionHeuristic<T> + Clone,
          C: Cutoff + Clone {
    problem  : &'x P,
    relax    : R,
    load_var : L,
    var_heu  : V,
    width_heu: W,
    select   : S,
    cutoff   : C,
    _phantom : PhantomData<T>
}

impl <'x, T, P, R, L, V, W, S, C> Config<T> for PassThroughConfig<'x, T, P, R, L, V, W, S, C>
    where T: Hash + Eq + Clone,
          P: Problem<T> + Clone,
          R: Relaxation<T> + Clone,
          L: LoadVars<T> + Clone,
          V: VariableHeuristic<T> + Clone,
          W: WidthHeuristic + Clone,
          S: NodeSelectionHeuristic<T> + Clone,
          C: Cutoff + Clone {

    /// Yields the root node of the (exact) MDD standing for the problem to solve.
    #[inline]
    fn root_node(&self) -> FrontierNode<T> {
        let r_state = Arc::new(self.problem.initial_state());
        let r_est   = self.relax.estimate(r_state.as_ref());

        FrontierNode {
            state : r_state,
            lp_len: self.problem.initial_value(),
            path  : Arc::new(PartialAssignment::Empty),
            ub    : r_est
        }
    }

    /// Returns the domain of variable `v` in the given `state`. These are the
    /// possible values that might possibly be affected to `v` when the system
    /// has taken decisions leading to `state`.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// problem definition.
    #[inline]
    fn domain_of<'a>(&self, state: &'a T, v: Variable) -> Domain<'a> {
        self.problem.domain_of(state, v)
    }
    /// Returns the next state reached by the system if the decision `d` is
    /// taken when the system is in the given `state` and the given set of `vars`
    /// are still free (no value assigned).
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// problem definition.
    #[inline]
    fn transition(&self, state: &T, vars: &VarSet, d: Decision) -> T {
        self.problem.transition(state, vars, d)
    }
    /// Returns the marginal benefit (in terms of objective function to maximize)
    /// of taking decision `d` is when the system is in the given `state` and
    /// the given set of `vars` are still free (no value assigned).
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// problem definition.
    #[inline]
    fn transition_cost(&self, state: &T, vars: &VarSet, d: Decision) -> isize {
        self.problem.transition_cost(state, vars, d)
    }
    /// Optional method for the case where you'd want to use a pooled mdd
    /// implementation. This method returns true iff taking a decision on
    /// `variable` might have an impact (state or longest path) on a node
    /// having the given `state`.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// problem definition.
    #[inline]
    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        self.problem.impacted_by(state, variable)
    }

    /// This method merges the given set of `states` into a new _inexact_ state
    /// that is an overapproximation of all `states`. The returned value will be
    /// used as a replacement for the given `states` in the mdd.
    ///
    /// In the theoretical framework of Bergman et al, this would amount to
    /// providing an implementation for the $\oplus$ operator. You should really
    /// only focus on that aspect when developing a relaxation: all the rest
    /// is taken care of by the framework.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// relaxation.
    #[inline]
    fn merge_states(&self, states: &mut dyn Iterator<Item=&T>) -> T {
        self.relax.merge_states(states)
    }
    /// This method relaxes the weight of the edge between the nodes `src` and
    /// `dst` because `dst` is replaced in the current layer by `relaxed`. The
    /// `decision` labels and the original weight (`cost`) of the edge
    /// `src` -- `dst` are also recalled.
    ///
    /// In the theoretical framework of Bergman et al, this would amount to
    /// providing an implementation for the $\Gamma$ operator.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// relaxation.
    #[inline]
    fn relax_edge(&self, src: &T, dst: &T, relaxed: &T, decision: Decision, cost: isize) -> isize {
        self.relax.relax_edge(src, dst, relaxed, decision, cost)
    }
    /// This optional method derives a _rough upper bound_ (RUB) on the maximum
    /// value of the subproblem rooted in the given `_state`. By default, the
    /// RUB returns the greatest positive integer; which is always safe but does
    /// not provide any pruning.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// relaxation.
    #[inline]
    fn estimate(&self, state: &T) -> isize {
        self.relax.estimate(state)
    }

    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// load-variable strategy.
    #[inline]
    fn load_variables(&self, node: &FrontierNode<T>) -> VarSet {
        self.load_var.variables(node)
    }

    /// Returns the best variable to branch on from the set of 'free' variables
    /// (`free_vars`, the variables that may still be branched upon in this MDD).
    /// It returns `None` in case no branching is useful (ie when no decision is
    /// left to make, etc...).
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// variable selection heuristic.
    #[inline]
    fn select_var(&self,
                  free_vars: &VarSet,
                  current_layer: &mut dyn Iterator<Item=&T>,
                  next_layer: &mut dyn Iterator<Item=&T>) -> Option<Variable>
    {
        self.var_heu.next_var(free_vars, current_layer, next_layer)
    }

    /// Returns the maximum width allowed for the next layer of the MDD when
    /// a value must still be assigned to each of the variables in `free_vars`.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// maximum width heuristic.
    #[inline]
    fn max_width(&self, free_vars: &VarSet) -> usize {
        self.width_heu.max_width(free_vars)
    }

    /// Defines an order of 'relevance' over the nodes `a` and `b`. Greater means
    /// that `a` is more important (hence more likely to be kept) than `b`.
    ///
    /// # Note
    /// This method does nothing but to delegate its call to the configured
    /// node selection heuristic.
    #[inline]
    fn compare(&self, a: &dyn SelectableNode<T>, b: &dyn SelectableNode<T>) -> Ordering {
        self.select.compare(a, b)
    }

    /// Returns true iff the cutoff criterion is met and the search must stop.
    #[inline]
    fn must_stop(&self, lb: isize, ub: isize) -> bool {
        self.cutoff.must_stop(lb, ub)
    }

    // ------------------------------------------------------------------------
    // --- Hooks for stateful heuristics --------------------------------------
    // ------------------------------------------------------------------------
    /// This method provides a hook for you to react to the addition of a new
    /// layer (to the mdd) during development of an mdd. This might be useful
    /// when working with incremental (stateful) heuristics (ie variable
    /// selection heuristic).
    #[inline]
    fn upon_new_layer(&mut self,
                      var: Variable,
                      current_layer: &mut dyn Iterator<Item=&T>) {
        self.var_heu.upon_new_layer(var, current_layer);
    }

    /// This method provides a hook for you to react to the addition of a new
    /// node to the next layer of the mdd during development of the mdd.
    ///
    /// This might be useful when working with incremental (stateful)
    /// heuristics (ie variable selection heuristic).
    #[inline]
    fn upon_node_insert(&mut self, state: &T) {
        self.var_heu.upon_node_insert(state);
    }

    /// When implementing an incremental variable selection heuristic, this
    /// method should reset the state of the heuristic to a "fresh" state.
    /// This method is called at the start of the development of any mdd.
    ///
    /// This might be useful when working with incremental (stateful)
    /// heuristics (ie variable selection heuristic).
    #[inline]
    fn clear(&mut self) {
        self.var_heu.clear();
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mock_it::verify;

    use crate::abstraction::dp::Problem;
    use crate::abstraction::mdd::Config;
    use crate::common::{Decision, Variable, VarSet, FrontierNode};
    use crate::implementation::mdd::config::mdd_builder;
    use crate::test_utils::{MockLoadVars, MockMaxWidth, MockNodeSelectionHeuristic, MockProblem, MockRelax, MockVariableHeuristic, Nothing, Proxy, MockSelectableNode, MockCutoff};
    use crate::common::PartialAssignment::{Empty, FragmentExtension};
    use mock_it::Matcher::Val;

    #[test]
    fn root_node_uses_problem_initial_state_value_and_relaxation_estimate_on_root_state() {
        let prob   = MockProblem::default();
        let mut relax  = MockRelax::default();

        let config  = mdd_builder(&prob, Proxy::new(&mut relax)).config();
        let _ = config.root_node();

        assert!(verify(prob.initial_state.was_called_with(Nothing)));
        assert!(verify(prob.initial_value.was_called_with(Nothing)));
        assert!(verify(relax.estimate.was_called_with(0)));
    }

    #[test]
    fn load_vars_is_pass_through_for_heuristic() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let mut lv = MockLoadVars::default();

        let config = mdd_builder(&prob, relax).with_load_vars(Proxy::new(&mut lv)).config();

        let node = FrontierNode{state: Arc::new(0), lp_len: 0, ub: isize::max_value(), path: Arc::new(Empty)};
        let _ = config.load_variables(&node);
        assert!(verify(lv.variables.was_called_with(node)));

        let node = FrontierNode{state: Arc::new(10), lp_len: 1000, ub: isize::max_value(), path: Arc::new(FragmentExtension {parent: Arc::new(Empty), fragment: vec![Decision{variable: Variable(7), value: 12}]})};
        let _ = config.load_variables(&node);
        assert!(verify(lv.variables.was_called_with(node)));
    }

    #[test]
    fn transition_is_pass_though_for_problem() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let vars   = VarSet::all(3);
        let dec    =  Decision{variable: Variable(0), value: 6};

        prob.transition.given((0, vars.clone(), dec)).will_return(64);

        let config = mdd_builder(&prob, relax).build();
        let next_s = config.transition(&0, &vars, dec);
        assert!(verify(prob.transition.was_called_with((0, vars, dec))));
        assert_eq!(64, next_s);
    }

    #[test]
    fn transition_cost_is_pass_though_for_problem() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let vars   = VarSet::all(3);
        let dec    =  Decision{variable: Variable(0), value: 6};

        prob.transition_cost.given((0, vars.clone(), dec)).will_return(-20);

        let config = mdd_builder(&prob, relax).build();
        let cost   = config.transition_cost(&0, &vars, dec);
        assert!(verify(prob.transition_cost.was_called_with((0, vars, dec))));
        assert_eq!(-20, cost);
    }

    #[test]
    fn domain_is_pass_though_for_problem() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();

        let config = mdd_builder(&prob, relax).config();
        let _ = config.domain_of(&0, Variable(2));
        assert!(verify(prob.domain_of.was_called_with((0, Variable(2)))));

        let _ = config.domain_of(&20, Variable(34));
        assert!(verify(prob.domain_of.was_called_with((20, Variable(34)))));
    }

    #[test]
    fn max_width_is_pass_through_for_heuristic() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let mut heu= MockMaxWidth::default();
        let config = mdd_builder(&prob, relax).with_max_width(Proxy::new(&mut heu)).config();

        let mut vs = prob.all_vars();
        while ! vs.is_empty() {
            let _ = config.max_width(&vs);
            assert!(verify(heu.max_width.was_called_with(vs.clone())));

            if let Some(v) = vs.iter().next() {
                vs.remove(v);
            }
        }
    }

    #[test]
    fn estimate_ub_is_a_pass_through_for_relaxation() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let config  = mdd_builder(&prob, Proxy::new(&mut relax)).config();
        let state   = 12;

        config.estimate(&state);
        assert!(verify(relax.estimate.was_called_with(state)));
    }

    #[test]
    fn merge_states_is_a_pass_through_for_relaxation_empty() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let config  = mdd_builder(&prob, Proxy::new(&mut relax)).config();
        let states  = vec![];
        config.merge_states(&mut states.iter());
        assert!(verify(relax.merge_states.was_called_with(states)));
    }

    #[test]
    fn merge_nodes_is_a_pass_through_for_relaxation_nonempty() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let config  = mdd_builder(&prob, Proxy::new(&mut relax)).config();
        let states  = vec![129];
        config.merge_states(&mut states.iter());
        assert!(verify(relax.merge_states.was_called_with(states)));
    }

    #[test]
    fn relax_edge_is_a_pass_through_for_relaxation() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let config  = mdd_builder(&prob, Proxy::new(&mut relax)).config();

        let src = 0;
        let dst = 1;
        let rlx = 2;
        let dc  = Decision{variable: Variable(0), value: 1};
        let cost= 5;
        let _   = config.relax_edge(&src, &dst, &rlx, dc, cost);

        assert!(verify(relax.relax_edge.was_called_with((src, dst, rlx, dc, cost))));
    }

    #[test]
    fn compare_is_a_pass_through_to_node_selection_heuristic() {
        let prob      = MockProblem::default();
        let mut relax = MockRelax::default();
        let mut heu   = MockNodeSelectionHeuristic::default();
        let config    = mdd_builder(&prob, Proxy::new(&mut relax))
            .with_nodes_selection_heuristic(Proxy::new(&mut heu)).config();

        let node_a  = MockSelectableNode{state: 127, value: 0, exact: true};
        let node_b  = MockSelectableNode{state: 123, value: 0, exact: true};
        config.compare(&node_a, &node_b);

        assert!(verify(heu.compare.was_called_with((127, 123))));

        let node_a  = MockSelectableNode{state: 127, value: 0, exact: true};
        let node_b  = MockSelectableNode{state: 123, value: 0, exact: true};
        config.compare(&node_b, &node_a);
        assert!(verify(heu.compare.was_called_with((123, 127))));

        let node_b  = MockSelectableNode{state: 123, value: 0, exact: true};
        config.compare(&node_b, &node_b);
        assert!(verify(heu.compare.was_called_with((123, 123))));
    }

    #[test]
    fn select_var_is_a_pass_through_to_heuristic() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let mut heu     = MockVariableHeuristic::default();
        let config  = mdd_builder(&prob, Proxy::new(&mut relax))
            .with_branch_heuristic(Proxy::new(&mut heu)).config();

        // empty
        let vs     = prob.all_vars();
        let data   = vec![];
        let curr   = &mut data.iter();
        let next   = &mut data.iter();
        config.select_var(&vs, curr, next);
        assert!(verify(heu.next_var.was_called_with((vs.clone(), vec![], vec![]))));

        // non-empty
        let data1  = vec![129];
        let data2  = vec![123];
        let curr   = &mut data1.iter();
        let next   = &mut data2.iter();
        config.select_var(&vs, curr, next);
        assert!(verify(heu.next_var.was_called_with((vs, data1, data2))))
    }

    #[test]
    fn must_stop_is_pass_through_to_heuristic() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let mut heu     = MockCutoff::default();
        let config  = mdd_builder(&prob, Proxy::new(&mut relax))
            .with_cutoff(Proxy::new(&mut heu)).config();

        // empty
        config.must_stop(isize::min_value(), isize::max_value());
        assert!(verify(heu.must_stop.was_called_with(Val((isize::min_value(), isize::max_value())))));
    }

    #[test]
    fn upon_new_layer_is_pass_through_to_variable_selection() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let mut heu     = MockVariableHeuristic::default();
        let mut config  = mdd_builder(&prob, Proxy::new(&mut relax))
            .with_branch_heuristic(Proxy::new(&mut heu))
            .config();

        // empty
        let var = Variable(42);
        let cur = vec![1, 2, 3, 4];
        config.upon_new_layer(var, &mut cur.iter());
        assert!(verify(heu.upon_new_layer.was_called_with((var, cur))));
    }
    #[test]
    fn upon_node_insert_is_pass_through_to_variable_selection() {
        let prob    = MockProblem::default();
        let mut relax   = MockRelax::default();
        let mut heu     = MockVariableHeuristic::default();
        let mut config  = mdd_builder(&prob, Proxy::new(&mut relax))
            .with_branch_heuristic(Proxy::new(&mut heu))
            .config();

        // empty
        config.upon_node_insert(&42);
        assert!(verify(heu.upon_node_insert.was_called_with(42)));
    }
    #[test]
    fn clear_is_pass_through_to_variable_selection() {
        let prob        = MockProblem::default();
        let mut relax   = MockRelax::default();
        let mut heu     = MockVariableHeuristic::default();
        let mut config  = mdd_builder(&prob, Proxy::new(&mut relax))
            .with_branch_heuristic(Proxy::new(&mut heu))
            .config();

        // empty
        config.clear();
        assert!(verify(heu.clear.was_called_with(())));
    }


    #[test]
    fn it_can_build_a_deep_mdd() {
        let prob      = MockProblem::default();
        let mut relax = MockRelax::default();
        let _         = mdd_builder(&prob, Proxy::new(&mut relax)).into_deep();
    }
    #[test]
    fn it_can_build_a_flat_mdd() {
        let prob   = MockProblem::default();
        let mut relax  = MockRelax::default();
        let _      = mdd_builder(&prob, Proxy::new(&mut relax)).into_flat();
    }
    #[test]
    fn it_can_build_a_pooled_mdd() {
        let prob   = MockProblem::default();
        let mut relax  = MockRelax::default();
        let _      = mdd_builder(&prob, Proxy::new(&mut relax)).into_pooled();
    }
    #[test]
    fn it_can_build_a_plain_config() {
        let prob   = MockProblem::default();
        let mut relax  = MockRelax::default();
        let _      = mdd_builder(&prob, Proxy::new(&mut relax)).build();
    }
}
