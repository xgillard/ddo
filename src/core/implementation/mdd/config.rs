use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic, LoadVars};
use compare::Compare;
use crate::core::common::{Node, Variable, Decision, Arc, VarSet, NodeInfo};
use std::rc::Rc;
use std::marker::PhantomData;
use std::cmp::Ordering;

pub trait Config<T> where T: Eq + Clone {
    fn root_node(&self) -> Node<T>;
    fn impacted_by(&self, state: &T, v: Variable) -> bool;
    fn load_vars (&mut self, root: &Node<T>);
    fn nb_free_vars(&self) -> usize;
    fn select_var(&self) -> Option<Variable>;
    fn remove_var(&mut self, v: Variable);
    fn domain_of (&self, state: &T, v: Variable) -> &[i32];
    fn max_width(&self) -> usize;
    fn branch(&self, node: Rc<Node<T>>, d: Decision) -> Node<T>;
    fn estimate_ub(&self, state: &T, info: &NodeInfo<T>) -> i32;
    fn compare(&self, x: &Node<T>, y: &Node<T>) -> Ordering;
    fn merge_nodes(&self, nodes: &[Node<T>]) -> Node<T>;
}

pub struct MDDConfig<'a, T, PB, RLX, LV, VS, WDTH, NS>
    where T    : Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>>  {

    pb               : &'a PB,
    relax            : RLX,
    lv               : LV,
    vs               : VS,
    width            : WDTH,
    ns               : NS,
    vars             : VarSet,
    _t               : PhantomData<*const T>
}

impl <'a, T, PB, RLX, LV, VS, WDTH, NS> Config<T> for MDDConfig<'a, T, PB, RLX, LV, VS, WDTH, NS>
    where T    : Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>>  {

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
    fn select_var(&self) -> Option<Variable> {
        self.vs.next_var(&self.vars)
    }
    fn remove_var(&mut self, v: Variable) {
        self.vars.remove(v)
    }
    fn domain_of(&self, state: &T, v: Variable) -> &[i32] {
        self.pb.domain_of(state, v)
    }
    fn max_width(&self) -> usize {
        self.width.max_width(&self.vars)
    }

    fn branch(&self, node: Rc<Node<T>>, d: Decision) -> Node<T> {
        let state = self.transition_state(node.as_ref(), d);
        let cost  = self.transition_cost (node.as_ref(), d);

        let len   = node.info.lp_len;
        let exct  = node.info.is_exact;
        let arc   = Arc {src: node, decision: d};

        Node::new(state, len + cost, Some(arc), exct)
    }

    fn estimate_ub(&self, state: &T, info: &NodeInfo<T>) -> i32 {
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
impl <'a, T, PB, RLX, LV, VS, WDTH, NS> MDDConfig<'a, T, PB, RLX, LV, VS, WDTH, NS>
    where T    : Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          LV   : LoadVars<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>>  {

    pub fn new(pb: &'a PB, relax: RLX, lv: LV, vs: VS, width: WDTH, ns: NS) -> Self {
        let vars = VarSet::all(pb.nb_vars());
        MDDConfig { pb, relax, lv, vs, width, ns, vars, _t: PhantomData }
    }

    fn transition_state(&self, node: &Node<T>, d: Decision) -> T {
        self.pb.transition(&node.state, &self.vars, d)
    }
    fn transition_cost(&self, node: &Node<T>, d: Decision) -> i32 {
        self.pb.transition_cost(&node.state, &self.vars, d)
    }
}