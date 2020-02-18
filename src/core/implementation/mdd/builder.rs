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

impl <'a, T, PB, RLX, LV, VS, WIDTH, NS> MDDBuilder<'a, T, PB, RLX, LV, VS, WIDTH, NS>
    where T    : Eq + Hash + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WIDTH: WidthHeuristic,
          NS   : Compare<Node<T>> {

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
    pub fn config(self) -> MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS> {
        MDDConfig::new(self.pb, self.rlx, self.lv, self.vs, self.w, self.ns)
    }
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn into_flat(self) -> FlatMDD<T, MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>> {
        FlatMDD::new(self.config())
    }
    #[allow(clippy::type_complexity)] // as long as type aliases are not supported
    pub fn into_pooled(self) -> PooledMDD<T, MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>> {
        PooledMDD::new(self.config())
    }
}

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

impl <'a, T, PB, RLX, LV, VS, WIDTH, NS> Config<T> for MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>
    where T    : Eq + Hash + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WIDTH: WidthHeuristic,
          NS   : Compare<Node<T>> {

    fn root_node(&self) -> Node<T> {
        self.pb.root_node()
    }
    fn impacted_by(&self, state: &T, v: Variable) -> bool {
        self.pb.impacted_by(state, v)
    }
    fn load_vars(&mut self, root: &Node<T>) {
        self.vars = self.lv.variables(root);
    }
    fn nb_free_vars(&self) -> usize {
        self.vars.len()
    }
    fn select_var(&self, current: Layer<'_, T>, next: Layer<'_, T>) -> Option<Variable> {
        self.vs.next_var(&self.vars, current, next)
    }
    fn remove_var(&mut self, v: Variable) {
        self.vars.remove(v)
    }
    fn domain_of<'b>(&self, state: &'b T, v: Variable) -> Domain<'b> {
        self.pb.domain_of(state, v)
    }
    fn max_width(&self) -> usize {
        self.width.max_width(&self.vars)
    }

    fn branch(&self, state: &T, info: Arc<NodeInfo>, d: Decision) -> Node<T> {
        let next  = self.transition_state(state, d);
        let cost  = self.transition_cost (state, d);

        let path  = NodeInfo {
            is_exact: info.is_exact,
            lp_len  : info.lp_len + cost,
            ub      : info.ub,
            lp_arc  : Some(Edge{src: info, decision: d}),
        };

        Node { state: next, info: path}
    }

    fn estimate_ub(&self, state: &T, info: &NodeInfo) -> i32 {
        self.relax.estimate_ub(state, info)
    }

    fn compare(&self, x: &Node<T>, y: &Node<T>) -> Ordering {
        self.ns.compare(x, y)
    }

    fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T> {
        self.relax.merge_nodes(nodes)
    }
}

// private functions
impl <'a, T, PB, RLX, LV, VS, WIDTH, NS> MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>
    where T    : Eq + Hash + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WIDTH: WidthHeuristic,
          NS   : Compare<Node<T>> {

    pub fn new(pb: &'a PB, relax: RLX, lv: LV, vs: VS, width: WIDTH, ns: NS) -> Self {
        let vars = VarSet::all(pb.nb_vars());
        MDDConfig { pb, relax, lv, vs, width, ns, vars, _t: PhantomData }
    }

    fn transition_state(&self, state: &T, d: Decision) -> T {
        self.pb.transition(state, &self.vars, d)
    }
    fn transition_cost(&self, state: &T, d: Decision) -> i32 {
        self.pb.transition_cost(state, &self.vars, d)
    }
}