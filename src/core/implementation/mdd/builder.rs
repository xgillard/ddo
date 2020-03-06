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

//! This module provides a default (generic) implementation for the configuration
//! object used to parameterize the behavior of MDDs (`MDDConfig`). Alongside
//! with it, this module provides an `MDDBuilder` and an `mdd_builder` function
//! whose purpose is to create a builder. Which one serves to configure the
//! behavior of MDDs in an intelligible way.
use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use compare::Compare;

use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::heuristics::{LoadVars, VariableHeuristic, WidthHeuristic};
use crate::core::common::{Decision, Domain, Edge, Layer, Node, NodeInfo, Variable, VarSet};
use crate::core::implementation::heuristics::{FromLongestPath, MinLP, NaturalOrder, NbUnassigned};
use crate::core::implementation::mdd::config::Config;
use crate::core::implementation::mdd::flat::FlatMDD;
use crate::core::implementation::mdd::pooled::PooledMDD;

/// This is the structure used to build an MDD configuration. There is very little
/// logic to it: it only uses the type system to adapt to its configurations and
/// return a config which may be used efficiently (and stack allocated).
/// Concretely, an MDD builder lets you specify all the parameters of a candidate
/// configuration to build. Namely:
///  + a reference to the problem (mandatory)
///  + a relaxation (mandatory)
///  + a load variable heuristic (Defaults to `FromLongestPath`)
///  + a variable selection heuristic (select the next var to branch on. Defaults to `NaturalOrder`.)
///  + a maximum width heuristic to limit the memory usage of each layer. (Defaults to `NbUnassigned`.)
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
/// # use ddo::core::implementation::mdd::builder::mdd_builder;
/// # use ddo::core::abstraction::dp::{Problem, Relaxation};
/// # use ddo::core::common::{Variable, VarSet, Domain, Decision, Node};
/// # use ddo::core::implementation::heuristics::FixedWidth;
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #   fn nb_vars(&self)       -> usize { 0 }
/// #   fn initial_state(&self) -> usize { 0 }
/// #   fn initial_value(&self) -> i32   { 0 }
/// #   fn domain_of<'a>(&self,state: &'a usize,var: Variable) -> Domain<'a> {
/// #       (0..1).into()
/// #   }
/// #   fn transition(&self,state: &usize,vars: &VarSet,d: Decision)      -> usize { 0 }
/// #   fn transition_cost(&self,state: &usize,vars: &VarSet,d: Decision) -> i32   { 0 }
/// # }
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #    fn merge_nodes(&self, nodes: &[Node<usize>]) -> Node<usize> { nodes[0].clone() }
/// # }
/// let problem = MockProblem;
/// let relax   = MockRelax;
/// let mdd     = mdd_builder(&problem, relax)
///                  .with_max_width(FixedWidth(100))
///                  .build();
/// ```
pub struct MDDBuilder<'a, T, PB, RLX,
    LV   = FromLongestPath<'a, PB>,
    VS   = NaturalOrder,
    WIDTH= NbUnassigned,
    NS   = MinLP> {

    pb : &'a PB,
    rlx: RLX,
    lv : LV,
    vs : VS,
    w  : WIDTH,
    ns : NS,
    _t : PhantomData<T>
}

/// This is the function you should use to instantiate a new MDD builder with
/// all defaults heuristics. It should be used as in the following example where
/// one creates an MDD whaving a fixed width strategy.
///
/// # Example:
/// ```
/// # use ddo::core::implementation::mdd::builder::mdd_builder;
/// # use ddo::core::abstraction::dp::{Problem, Relaxation};
/// # use ddo::core::common::{Variable, VarSet, Domain, Decision, Node};
/// # use ddo::core::implementation::heuristics::FixedWidth;
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #   fn nb_vars(&self)       -> usize { 0 }
/// #   fn initial_state(&self) -> usize { 0 }
/// #   fn initial_value(&self) -> i32   { 0 }
/// #   fn domain_of<'a>(&self,state: &'a usize,var: Variable) -> Domain<'a> {
/// #       (0..1).into()
/// #   }
/// #   fn transition(&self,state: &usize,vars: &VarSet,d: Decision)      -> usize { 0 }
/// #   fn transition_cost(&self,state: &usize,vars: &VarSet,d: Decision) -> i32   { 0 }
/// # }
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #    fn merge_nodes(&self, nodes: &[Node<usize>]) -> Node<usize> { nodes[0].clone() }
/// # }
/// let problem = MockProblem;
/// let relax   = MockRelax;
/// let mdd     = mdd_builder(&problem, relax)
///                  .with_max_width(FixedWidth(100))
///                  .build();
/// ```
pub fn mdd_builder<T, PB, RLX>(pb: &PB, rlx: RLX) -> MDDBuilder<T, PB, RLX> {
    MDDBuilder {
        pb, rlx,
        lv: FromLongestPath::new(pb),
        vs: NaturalOrder,
        w : NbUnassigned,
        ns: MinLP,
        _t: PhantomData
    }
}

/// The following methods define the behavior of an mdd builder.
impl <'a, T, PB, RLX, LV, VS, WIDTH, NS> MDDBuilder<'a, T, PB, RLX, LV, VS, WIDTH, NS>
    where T    : Eq + Hash + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WIDTH: WidthHeuristic,
          NS   : Compare<Node<T>> {

    /// This is how you specify the load variable heuristic to use.
    pub fn with_load_vars<H>(self, h: H) -> MDDBuilder<'a, T, PB, RLX, H, VS, WIDTH, NS> {
        MDDBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : h,
            vs : self.vs,
            w  : self.w,
            ns : self.ns,
            _t : PhantomData
        }
    }
    /// This is how you specify the branch heuristic to use (the variable selection heuristic).
    pub fn with_branch_heuristic<H>(self, h: H) -> MDDBuilder<'a, T, PB, RLX, LV, H, WIDTH, NS> {
        MDDBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : self.lv,
            vs : h,
            w  : self.w,
            ns : self.ns,
            _t : PhantomData
        }
    }
    /// This is how you specify the maximum width heuristic to use (to constrain
    /// the max width of MDD layers).
    pub fn with_max_width<H>(self, h: H) -> MDDBuilder<'a, T, PB, RLX, LV, VS, H, NS> {
        MDDBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : self.lv,
            vs : self.vs,
            w  : h,
            ns : self.ns,
            _t : PhantomData
        }
    }
    /// This is how you specify the nodes selection heuristic to use (to decide
    /// what nodes to merge/drop in case the layer width is too large).
    pub fn with_nodes_selection_heuristic<H>(self, h: H) -> MDDBuilder<'a, T, PB, RLX, LV, VS, WIDTH, H> {
        MDDBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : self.lv,
            vs : self.vs,
            w  : self.w,
            ns : h,
            _t : PhantomData
        }
    }
    /// This is how you instantiate a configuration object. This is not really
    /// useful per-se, unless you decide to implement your own kind of MDD and
    /// want to be able to reuse a single configuration object.
    pub fn config(self) -> MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS> {
        MDDConfig::new(self.pb, self.rlx, self.lv, self.vs, self.w, self.ns)
    }
    /// This is how you instantiate an MDD (using the default MDD implementation)
    /// configured with the parameters you specified.
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn build(self) -> FlatMDD<T, MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>> {
        self.into_flat()
    }
    /// This is how you instantiate a _pooled_ MDD from using your desired
    /// configuration. Note: Unless you have a specific reason to use a pooled
    /// MDD, you are probably better off using the default implem (flat mdd).
    /// Chances are high that it will perform better.
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn into_flat(self) -> FlatMDD<T, MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>> {
        FlatMDD::new(self.config())
    }
    /// This is how you instantiate a _pooled_ MDD from using your desired
    /// configuration.
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn into_pooled(self) -> PooledMDD<T, MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>> {
        PooledMDD::new(self.config())
    }
}


/// This is the structure implementing the configuration of an MDD. It is
/// basically a structure holding all the heuristics together.
#[derive(Clone)]
pub struct MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS> {
    pb               : &'a PB,
    relax            : RLX,
    lv               : LV,
    vs               : VS,
    width            : WIDTH,
    ns               : NS,
    vars             : VarSet,
    _t               : PhantomData<T>
}

/// This is how an `MDDConfig` object implements the the `Config` trait.
impl <'a, T, PB, RLX, LV, VS, WIDTH, NS> Config<T> for MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>
    where T    : Eq + Hash + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WIDTH: WidthHeuristic,
          NS   : Compare<Node<T>> {
    /// Yields the root node of the (exact) MDD standing for the problem to solve.
    ///
    /// # Note:
    /// This is a pass through to the problem method.
    fn root_node(&self) -> Node<T> {
        self.pb.root_node()
    }
    /// This method returns true iff taking a decision on `variable` might
    /// have an impact (state or longest path) on a node having the given `state`
    ///
    /// # Note:
    /// This is a pass through to the problem method.
    fn impacted_by(&self, state: &T, v: Variable) -> bool {
        self.pb.impacted_by(state, v)
    }
    /// Returns the minimal set of free variables for the given `problem` when
    /// starting an exploration in the given `node`.
    ///
    /// # Note:
    /// This is a pass through to the load vars heuristic.
    fn load_vars(&mut self, root: &Node<T>) {
        self.vars = self.lv.variables(root);
    }
    /// Returns the number of variables which may still be decided upon in this
    /// unrolling of the (approximate) MDD. Note: This number varies during the
    /// unrolling of the MDD. Whenever you call `remove_var`, this number should
    /// decrease.
    fn nb_free_vars(&self) -> usize {
        self.vars.len()
    }
    /// Returns the best variable to branch on from the set of 'free' variables
    /// (variables that may still be branched upon in this unrolling of the MDD).
    /// It returns `None` in case no branching is useful (ie when no decision is
    /// left to make, etc...).
    ///
    /// # Note:
    /// This is almost a pass through to the node selection heuristic.
    fn select_var(&self, current: Layer<'_, T>, next: Layer<'_, T>) -> Option<Variable> {
        self.vs.next_var(&self.vars, current, next)
    }
    /// Removes `v` from the set of free variables (which may be branched upon).
    /// That is, it alters the configuration so that `v` is considered to have
    /// been assigned with a value. Note: As a side effect, it should decrease
    /// the value of `nb_free_vars` and make `v` impossible to select (during
    /// this unrolling of the MDD) with `select_var`.
    fn remove_var(&mut self, v: Variable) {
        self.vars.remove(v)
    }
    /// Returns the domain of variable `v` in the given `state`. These are the
    /// possible values that might possibly be affected to `v` when the system
    /// has taken decisions leading to `state`.
    ///
    /// # Note:
    /// This is a pass through to the problem method.
    fn domain_of<'b>(&self, state: &'b T, v: Variable) -> Domain<'b> {
        self.pb.domain_of(state, v)
    }
    /// Returns the maximum width allowed for the next layer in this unrolling
    /// of the MDD.
    ///
    /// # Note:
    /// This is a pass through to the max width heuristic.
    fn max_width(&self) -> usize {
        self.width.max_width(&self.vars)
    }
    /// Returns the node which is reached by taking decision `d` from the node
    /// (`state`, `info`).
    fn branch(&self, state: &T, info: Arc<NodeInfo>, d: Decision) -> Node<T> {
        let next  = self.transition_state(state, d);
        let cost  = self.transition_cost (state, d);

        let path  = NodeInfo {
            is_exact  : info.is_exact,
            is_relaxed: false,
            lp_len    : info.lp_len + cost,
            ub        : info.ub,
            lp_arc    : Some(Edge{src: info, decision: d}),
        };

        Node { state: next, info: path}
    }
    /// Returns a _rough_ upper bound on the maximum objective function value
    /// reachable by passing through the node characterized by the given
    /// `state` and node `info`.
    ///
    /// # Note:
    /// This is a pass through to the relaxation method.
    fn estimate_ub(&self, state: &T, info: &NodeInfo) -> i32 {
        self.relax.estimate_ub(state, info)
    }
    /// Compares two nodes according to the node selection ranking. That is,
    /// it derives an ordering for `x` and `y` where the Gretest node has more
    /// chances of remaining in the layer (in case a restriction or merge needs
    /// to occur). In other words, if `x > y` according to this ordering, it
    /// means that `x` is more promising than `y`. A consequence of which,
    /// `x` has more chances of not being dropped/merged into an other node.
    ///
    /// # Note:
    /// This is a pass through to the node selection heuristic.
    fn compare(&self, x: &Node<T>, y: &Node<T>) -> Ordering {
        self.ns.compare(x, y)
    }
    /// This method merges the given set of `nodes` into a new _inexact_ node.
    /// The role of this method is really to _only_ provide an inexact
    /// node to use as a replacement for the selected `nodes`. It is the MDD
    /// implementation 's responsibility to take care of maintaining a cutset.
    ///
    /// # Note:
    /// This is a pass through to the relaxation method.
    fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T> {
        self.relax.merge_nodes(nodes)
    }

    /// This method is just a shortcut to get the next state by applying the
    /// transition (as described per the problem) from `state` and taking the
    /// given `decision`.
    ///
    /// # Visually
    /// Calling this method returns (new state).
    ///
    /// ```plain
    /// (state) --- [decision] ---> (new_state)
    /// ```
    fn transition_state(&self, state: &T, d: Decision) -> T {
        self.pb.transition(state, &self.vars, d)
    }
    /// This method is just a shortcut to get the cost of going to next state by
    /// applying the transition cost function (as described per the problem);
    /// starting from `state` and taking the given `decision`.
    ///
    /// # Visually
    /// Calling this method returns the cost of taking decision.
    ///
    /// ```plain
    /// (state) --- [decision] ---> (new_state)
    ///                 ^
    ///                 |
    ///       cost of taking decision
    /// ```
    fn transition_cost(&self, state: &T, d: Decision) -> i32 {
        self.pb.transition_cost(state, &self.vars, d)
    }
}

/// `MDDConfig` methods that do not belong to any trait
impl <'a, T, PB, RLX, LV, VS, WIDTH, NS> MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>
    where T    : Eq + Hash + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WIDTH: WidthHeuristic,
          NS   : Compare<Node<T>> {
    /// Constructor: uses all the given parameters to build a new configuration
    /// object. Using this method is advised against (would make your code
    /// cumbersome). This is why it was made private. Anything you can do using
    /// this constructor, you can achieve by using the much clearer `mdd_builder`.
    /// This is the recommended approach and involves *no* perfromance penalty
    /// at runtime.
    fn new(pb: &'a PB, relax: RLX, lv: LV, vs: VS, width: WIDTH, ns: NS) -> Self {
        let vars = VarSet::all(pb.nb_vars());
        MDDConfig { pb, relax, lv, vs, width, ns, vars, _t: PhantomData }
    }
}

#[cfg(test)]
mod test_config_builder {
    use std::sync::Arc;

    use mock_it::verify;

    use crate::core::abstraction::dp::Problem;
    use crate::core::abstraction::mdd::MDD;
    use crate::core::common::{Decision, Edge, Layer, Node, NodeInfo, Variable, VarSet};
    use crate::core::implementation::mdd::builder::mdd_builder;
    use crate::core::implementation::mdd::config::Config;
    use crate::test_utils::{MockLoadVars, MockMaxWidth, MockNodeSelectionHeuristic, MockProblem, MockRelax, MockVariableHeuristic, Nothing, Proxy};

    #[test]
    fn root_node_is_pass_though_for_problem() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();

        let config  = mdd_builder(&prob, relax).config();
        let _ = config.root_node();
        assert!(verify(prob.root_node.was_called_with(Nothing)));
    }
    #[test]
    fn impacted_by_is_pass_though_for_problem() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();

        let config  = mdd_builder(&prob, relax).config();
        let _ = config.impacted_by(&0, Variable(0));
        assert!(verify(prob.impacted_by.was_called_with((0, Variable(0)))));

        let _ = config.impacted_by(&100, Variable(25));
        assert!(verify(prob.impacted_by.was_called_with((100, Variable(25)))));
    }
    #[test]
    fn load_vars_is_pass_through_for_heuristic() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();

        let lv     = MockLoadVars::default();

        let mut config  = mdd_builder(&prob, relax).with_load_vars(Proxy::new(&lv)).config();
        let node = Node::new(0, 0, None, true, false);
        config.load_vars(&node);
        assert!(verify(lv.variables.was_called_with(node.clone())));

        let node = Node::new(10, 1000, Some(Edge{ src: Arc::new(node.info),  decision: Decision{variable: Variable(7), value: 12}}), true, false);
        config.load_vars(&node);
        assert!(verify(lv.variables.was_called_with(node)));
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
    fn nb_free_vars_defaults_to_all_vars() {
        let prob = MockProblem::default();
        let relax = MockRelax::default();
        let config = mdd_builder(&prob, relax).config();

        assert_eq!(5, config.nb_free_vars());
    }
    #[test]
    fn nb_free_vars_decreases_upon_remove() {
        let prob        = MockProblem::default();
        let relax       = MockRelax::default();
        let mut config  = mdd_builder(&prob, relax).config();

        config.remove_var(Variable(3));
        assert_eq!(4, config.nb_free_vars());
    }
    #[test]
    fn removed_var_cannot_be_selected_anymore() {
        let prob        = MockProblem::default();
        let relax       = MockRelax::default();
        let mut config  = mdd_builder(&prob, relax).config();

        let data = vec![];
        let mut layer1 = Layer::Plain(data.iter());
        let mut layer2 = Layer::Plain(data.iter());

        let mut avail= 5;
        while let Some(v) = config.select_var(layer1, layer2) {
            config.remove_var(v);
            avail -= 1;
            assert_eq!(avail, config.nb_free_vars());

            layer1 = Layer::Plain(data.iter());
            layer2 = Layer::Plain(data.iter());
        }

        assert_eq!(0, avail);
    }

    #[test]
    fn max_width_is_pass_through_for_heuristic() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let heu    = MockMaxWidth::default();
        let mut config = mdd_builder(&prob, relax).with_max_width(Proxy::new(&heu)).config();

        let mut vs = prob.all_vars();
        while ! vs.is_empty() {
            let _ = config.max_width();
            assert!(verify(heu.max_width.was_called_with(vs.clone())));

            if let Some(v) = vs.iter().next() {
                vs.remove(v);
                config.remove_var(v);
            }
        }
    }

    #[test]
    fn branch_applies_the_desired_transition() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();

        let config = mdd_builder(&prob, relax).config();

        let varset  = VarSet::all(5);
        let state   = 0;
        let decision= Decision {variable: Variable(4), value: 7};
        let info    = NodeInfo {is_exact: true, lp_len: 10, lp_arc: None, ub: 100, is_relaxed: false};
        let info    = Arc::new(info);

        prob.transition.given((state, varset.clone(), decision)).will_return(10);
        prob.transition_cost.given((state, varset, decision)).will_return(666);

        assert_eq!(
          config.branch(&state, info.clone(), decision),
          Node {
              state: 10,
              info: NodeInfo {
                  is_exact: true,
                  is_relaxed: false,
                  lp_len  : 676,
                  lp_arc  : Some(Edge {
                      src: info,
                      decision
                  }),
                  ub: 100
              }
          }
        );
    }
    #[test]
    fn branch_maintains_the_exact_flat() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();

        let config = mdd_builder(&prob, relax).config();

        let varset  = VarSet::all(5);
        let state   = 0;
        let decision= Decision {variable: Variable(4), value: 7};
        let info    = NodeInfo {is_exact: false, lp_len: 10, lp_arc: None, ub: 100, is_relaxed: true};
        let info    = Arc::new(info);

        prob.transition.given((state, varset.clone(), decision)).will_return(10);
        prob.transition_cost.given((state, varset, decision)).will_return(666);

        assert_eq!(
          config.branch(&state, info.clone(), decision),
          Node {
              state: 10,
              info: NodeInfo {
                  is_exact: false,
                  is_relaxed: true,
                  lp_len  : 676,
                  lp_arc  : Some(Edge {
                      src: info,
                      decision
                  }),
                  ub: 100
              }
          }
        );
    }

    #[test]
    fn estimate_ub_is_a_pass_through_for_relaxation() {
        let prob    = MockProblem::default();
        let relax   = MockRelax::default();
        let config  = mdd_builder(&prob, Proxy::new(&relax)).config();

        let state   = 12;
        let info    = NodeInfo {is_exact: false, lp_len: 28, lp_arc: None, ub: 40, is_relaxed: true};
        config.estimate_ub(&state, &info);

        assert!(verify(relax.estimate_ub.was_called_with((state, info))));
    }
    #[test]
    fn merge_nodes_is_a_pass_through_for_relaxation_empty() {
        let prob    = MockProblem::default();
        let relax   = MockRelax::default();
        let config  = mdd_builder(&prob, Proxy::new(&relax)).config();

        let nodes = vec![];
        config.merge_nodes(&nodes);

        assert!(verify(relax.merge_nodes.was_called_with(nodes)));
    }
    #[test]
    fn merge_nodes_is_a_pass_through_for_relaxation_nonempty() {
        let prob    = MockProblem::default();
        let relax   = MockRelax::default();
        let config  = mdd_builder(&prob, Proxy::new(&relax)).config();

        let nodes = vec![Node {state: 129, info: NodeInfo{is_exact: false, lp_len: 27, lp_arc: None, ub: 65, is_relaxed: true}}];
        config.merge_nodes(&nodes);

        assert!(verify(relax.merge_nodes.was_called_with(nodes)));
    }

    #[test]
    fn compare_is_a_pass_through_to_node_selection_heuristic() {
        let prob    = MockProblem::default();
        let relax   = MockRelax::default();
        let heu     = MockNodeSelectionHeuristic::default();
        let config  = mdd_builder(&prob, Proxy::new(&relax)).with_nodes_selection_heuristic(Proxy::new(&heu)).config();

        let node_a  = Node {state: 129, info: NodeInfo{is_exact: false, lp_len: 27, lp_arc: None, ub: 65, is_relaxed: false}};
        let node_b  = Node {state: 123, info: NodeInfo{is_exact: true,  lp_len: 24, lp_arc: None, ub: 99, is_relaxed: false}};
        config.compare(&node_a, &node_b);
        assert!(verify(heu.compare.was_called_with((node_a, node_b))));

        let node_a  = Node {state: 129, info: NodeInfo{is_exact: false, lp_len: 27, lp_arc: None, ub: 65, is_relaxed: false}};
        let node_b  = Node {state: 123, info: NodeInfo{is_exact: true,  lp_len: 24, lp_arc: None, ub: 99, is_relaxed: false}};
        config.compare(&node_b, &node_a);
        assert!(verify(heu.compare.was_called_with((node_b, node_a))));

        let node_b  = Node {state: 123, info: NodeInfo{is_exact: true,  lp_len: 24, lp_arc: None, ub: 99, is_relaxed: false}};
        config.compare(&node_b, &node_b);
        assert!(verify(heu.compare.was_called_with((node_b.clone(), node_b))));
    }

    #[test]
    fn select_var_is_a_pass_through_to_heuristic() {
        let prob    = MockProblem::default();
        let relax   = MockRelax::default();
        let heu     = MockVariableHeuristic::default();
        let config  = mdd_builder(&prob, Proxy::new(&relax)).with_branch_heuristic(Proxy::new(&heu)).config();

        // empty
        let data = vec![];
        let curr = Layer::Plain(data.iter());
        let next = Layer::Plain(data.iter());
        config.select_var(curr, next);
        assert!(verify(heu.next_var.was_called_with((prob.all_vars(), vec![], vec![]))));

        // non-empty
        let data1 = vec![Node {state: 129, info: NodeInfo{is_exact: false, lp_len: 27, lp_arc: None, ub: 65, is_relaxed: false}}];
        let data2 = vec![Node {state: 123, info: NodeInfo{is_exact: true,  lp_len: 24, lp_arc: None, ub: 99, is_relaxed: false}}];
        let curr = Layer::Plain(data1.iter());
        let next = Layer::Plain(data2.iter());
        config.select_var(curr, next);
        assert!(verify(heu.next_var.was_called_with((prob.all_vars(), data1, data2))))
    }


    #[test]
    fn it_can_build_an_mdd() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let mdd    = mdd_builder(&prob, Proxy::new(&relax)).build();

        assert_eq!(prob.root_node(), mdd.root())
    }
    #[test]
    fn it_can_build_a_flat_mdd() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let mdd    = mdd_builder(&prob, Proxy::new(&relax)).into_flat();

        assert_eq!(prob.root_node(), mdd.root())
    }
    #[test]
    fn it_can_build_a_pooled_mdd() {
        let prob   = MockProblem::default();
        let relax  = MockRelax::default();
        let mdd    = mdd_builder(&prob, Proxy::new(&relax)).into_pooled();

        assert_eq!(prob.root_node(), mdd.root())
    }
}