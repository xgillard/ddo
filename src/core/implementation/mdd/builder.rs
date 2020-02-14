use crate::core::implementation::heuristics::{FromLongestPath, NaturalOrder, NbUnassigned, MinLP};
use std::marker::PhantomData;
use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::heuristics::{LoadVars, VariableHeuristic, WidthHeuristic};
use compare::Compare;
use crate::core::common::{Node, VarSet, Variable, Decision, Edge, NodeInfo, Domain};
use crate::core::implementation::mdd::flat::FlatMDD;
use crate::core::implementation::mdd::pooled::PooledMDD;
use crate::core::implementation::mdd::config::Config;
use crate::core::abstraction::mdd::Layer;
use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::Arc;



pub struct MDDBuilder<'a, T, PB, RLX,
    LV   = FromLongestPath<'a, T, PB>,
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

pub fn mdd_builder<T, PB, RLX>(pb: &PB, rlx: RLX) -> MDDBuilder<T, PB, RLX>
    where T: Send + Eq + Hash + Clone, PB: Send + Problem<T>, RLX: Send + Relaxation<T> {
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
    where T    : Sync + Send + Hash + Eq + Clone,
          PB   : Sync + Send + Problem<T> + Clone,
          RLX  : Sync + Send + Relaxation<T> + Clone,
          LV   : Sync + Send + LoadVars<T> + Clone,
          VS   : Sync + Send + VariableHeuristic<T> + Clone,
          WIDTH: Sync + Send + WidthHeuristic<T> + Clone,
          NS   : Sync + Send + Compare<Node<T>> + Clone {

    pub fn with_load_vars<H>(self, h: H) -> MDDBuilder<'a, T, PB, RLX, H, VS, WIDTH, NS> {
        MDDBuilder {
            pb : self.pb,
            rlx: self.rlx,
            lv : h,
            vs : self.vs,
            w  : self.w,
            ns : self.ns,
            _t : self._t
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
            _t : self._t
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
            _t : self._t
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
            _t : self._t
        }
    }
    pub fn config(self) -> MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS> {
        MDDConfig::new(self.pb, self.rlx, self.lv, self.vs, self.w, self.ns)
    }
    #[allow(clippy::type_complexity)] // as long as inherent type aliases are not supported
    pub fn into_flat(self) -> FlatMDD<T, MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>> {
        FlatMDD::new(self.config())
    }
    #[allow(clippy::type_complexity)] // as long as inherent type aliases are not supported
    pub fn into_pooled(self) -> PooledMDD<T, MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>> {
        PooledMDD::new(self.config())
    }
}

#[derive(Clone)]
pub struct MDDConfig<'a, T, PB, RLX, LV, VS, WIDTH, NS>
    where T    : Send + Eq + Clone,
          PB   : Send + Problem<T> + Clone,
          RLX  : Send + Relaxation<T> + Clone,
          LV   : Send + LoadVars<T> + Clone,
          VS   : Send + VariableHeuristic<T> + Clone,
          WIDTH: Send + WidthHeuristic<T> + Clone,
          NS   : Send + Compare<Node<T>> + Clone  {

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
    where T    : Send + Eq + Clone,
          PB   : Send + Problem<T> + Clone,
          RLX  : Send + Relaxation<T> + Clone,
          LV   : Send + LoadVars<T> + Clone,
          VS   : Send + VariableHeuristic<T> + Clone,
          WIDTH: Send + WidthHeuristic<T> + Clone,
          NS   : Send + Compare<Node<T>> + Clone  {

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
    where T    : Send + Eq + Clone,
          PB   : Send + Problem<T> + Clone,
          RLX  : Send + Relaxation<T> + Clone,
          LV   : Send + LoadVars<T> + Clone,
          VS   : Send + VariableHeuristic<T> + Clone,
          WIDTH: Send + WidthHeuristic<T> + Clone,
          NS   : Send + Compare<Node<T>> + Clone {

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