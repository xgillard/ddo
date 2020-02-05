use std::cmp::{max, min};
use std::hash::Hash;
use std::rc::Rc;

use crate::core::abstraction::dp::{Decision, Problem, Relaxation, Variable, VarSet};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic, NodeOrdering};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};

use super::super::abstraction::mdd::*;
use metrohash::MetroHashMap;

// --- POOLED MDD --------------------------------------------------------------
pub struct PooledMDD<T> where T: Hash + Eq + Clone {
    mddtype          : MDDType,
    pool             : MetroHashMap<T, Node<T>>,
    current          : Vec<Node<T>>,
    cutset           : Vec<Node<T>>,

    last_assigned    : Variable,
    unassigned_vars  : VarSet,
    is_exact         : bool,
    best_node        : Option<Node<T>>
}

impl <T> Default for PooledMDD<T> where T : Hash + Clone + Eq {
    fn default() -> PooledMDD<T> {
        PooledMDD::new()
    }
}

impl <T> MDD<T> for PooledMDD<T> where T: Hash + Eq + Clone {
    fn mdd_type(&self) -> MDDType {
        self.mddtype
    }
    fn current_layer(&self) -> &[Node<T>] {
        &self.current
    }
    fn exact_cutset(&self) -> &[Node<T>] {
        &self.cutset
    }
    fn next_layer(&self) -> &MetroHashMap<T, Node<T>> {
        &self.pool
    }
    fn last_assigned(&self) -> Variable {
        self.last_assigned
    }
    fn unassigned_vars(&self) -> &VarSet {
        &self.unassigned_vars
    }
    fn is_exact(&self) -> bool {
        self.is_exact
    }
    fn best_value(&self) -> i32 {
        if self.best_node.is_none() {
            std::i32::MIN
        } else {
            self.best_node.as_ref().unwrap().lp_len
        }
    }
    fn best_node(&self) -> &Option<Node<T>> {
        &self.best_node
    }
    fn longest_path(&self) -> Vec<Decision> {
        if self.best_node.is_none() {
            vec![]
        } else {
            self.best_node.as_ref().unwrap().longest_path()
        }
    }
}

/// Private functions
impl <T> PooledMDD<T> where T    : Hash + Eq + Clone {
    fn new() -> PooledMDD<T> {
        PooledMDD {
            mddtype          : Exact,
            last_assigned    : Variable(std::usize::MAX),
            unassigned_vars  : VarSet::all(0),
            is_exact         : true,
            best_node        : None,
            pool             : Default::default(),
            current          : vec![],
            cutset           : vec![]
        }
    }

    fn clear(&mut self) {
        self.mddtype          = Exact;
        self.last_assigned    = Variable(std::usize::MAX);
        self.is_exact         = true;
        self.best_node        = None;
        // unassigned vars holds stale data !

        self.pool             .clear();
        self.current          .clear();
        self.cutset           .clear();
    }
}

// --- GENERATOR ---------------------------------------------------------------
pub struct PooledMDDGenerator<T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : NodeOrdering<T> {

    pb               : Rc<PB>,
    relax            : RLX,
    vs               : VS,
    width            : WDTH,
    ns               : NS,
    dd               : PooledMDD<T>
}

impl <T, PB, RLX, VS, WDTH, NS> MDDGenerator<T> for PooledMDDGenerator<T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : NodeOrdering<T> {
    fn exact(&mut self, vars: VarSet, root: &Node<T>, best_lb: i32) {
        self.develop(Exact, vars, root, best_lb);
    }
    fn restricted(&mut self, vars: VarSet, root: &Node<T>, best_lb: i32) {
        self.develop(Restricted, vars, root, best_lb);
    }
    fn relaxed(&mut self, vars: VarSet, root: &Node<T>, best_lb: i32) {
        self.develop(Relaxed, vars, root, best_lb);
    }
    fn mdd(&self) -> &dyn MDD<T> {
        &self.dd
    }
}

#[derive(Debug, Copy, Clone)]
struct Bounds {lb: i32, ub: i32}

impl <T, PB, RLX, VS, WDTH, NS> PooledMDDGenerator<T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : NodeOrdering<T> {

    pub fn new(pb: Rc<PB>, relax: RLX, vs: VS, width: WDTH, ns: NS) -> PooledMDDGenerator<T, PB, RLX, VS, WDTH, NS> {
        PooledMDDGenerator{ pb, relax, vs, width, ns, dd: PooledMDD::new() }
    }
    fn develop(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>, best_lb : i32) {
        self.init(kind, vars, root);

        let bounds = Bounds {lb: best_lb, ub: root.ub};
        let mut i  = 0;
        let nbvars = self.nb_vars();

        while i < nbvars && !self.exhausted() {
            let var = self.select_var();
            if var.is_none() {
                break;
            }

            let var = var.unwrap();
            self.pick_nodes_from_pool(var);
            self.maybe_squash(i);
            self.remove_var(var);
            self.unroll_layer(var, bounds);
            self.set_last_assigned(var);
            i += 1;
        }

        self.finalize()
    }
    fn unroll_layer(&mut self, var: Variable, bounds: Bounds) {
        for node in self.dd.current.iter() {
            let domain = self.pb.domain_of(&node.state, var);
            for value in domain {
                let decision  = Decision{variable: var, value: *value};
                let branching = self.branch(node, decision);

                if let Some(old) = self.dd.pool.get_mut(&branching.state) {
                    if old.is_exact && !branching.is_exact {
                        //trace!("main loop:: old was exact but new was not");
                        self.dd.cutset.push(old.clone());
                    }
                    if !old.is_exact && branching.is_exact {
                        //trace!("main loop:: new was exact but old was not");
                        self.dd.cutset.push(branching.clone());
                    }
                    old.merge(branching)
                } else if self.is_relevant(&branching, bounds) {
                    self.dd.pool.insert(branching.state.clone(), branching);
                }
            }
        }
    }
    fn pick_nodes_from_pool(&mut self, var: Variable) {
        self.dd.current.clear();

        // Add all selected nodes to the next layer
        for n in self.dd.pool.values() {
            if self.pb.impacted_by(&n.state, var) {
                self.dd.current.push(n.clone());
            }
        }
        // Remove all nodes that belong to the current layer from the pool
        for n in self.dd.current.iter() {
            self.dd.pool.remove(&n.state);
        }
    }

    fn nb_vars(&self) -> usize {
        self.dd.unassigned_vars.len()
    }
    fn exhausted(&self) -> bool {
        self.dd.pool.is_empty()
    }
    fn select_var(&self) -> Option<Variable> {
        self.vs.next_var(&self.dd, &self.dd.unassigned_vars)
    }
    fn remove_var(&mut self, var: Variable) {
        self.dd.unassigned_vars.remove(var)
    }
    fn set_last_assigned(&mut self, var: Variable) {
        self.dd.last_assigned = var
    }
    fn transition_state(&self, node: &Node<T>, d: Decision) -> T {
        self.pb.transition(&node.state, &self.dd.unassigned_vars, d)
    }
    fn transition_cost(&self, node: &Node<T>, d: Decision) -> i32 {
        self.pb.transition_cost(&node.state, &self.dd.unassigned_vars, d)
    }
    fn branch(&self, node: &Node<T>, d: Decision) -> Node<T> {
        let state = self.transition_state(node, d);
        let cost  = self.transition_cost (node, d);
        let arc   = Arc {src: Rc::new(node.clone()), decision: d, weight: cost};

        Node::new(state, node.lp_len + cost, Some(arc), node.is_exact)
    }
    fn is_relevant(&self, n: &Node<T>, bounds: Bounds) -> bool {
        min(self.relax.rough_ub(n.lp_len, &n.state), bounds.ub) > bounds.lb
    }

    fn init(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>) {
        self.dd.clear();
        self.dd.mddtype         = kind;
        self.dd.unassigned_vars = vars;

        self.dd.pool.insert(root.state.clone(), root.clone());
    }
    fn finalize(&mut self) {
        self.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.dd.best_node {
            let lp_length = best.lp_len;

            for n in self.dd.cutset.iter_mut() {
                n.ub = lp_length.min(self.relax.rough_ub(n.lp_len, &n.state));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            self.dd.cutset.clear();
        }
    }
    fn find_best_node(&mut self) {
        let mut best_value = std::i32::MIN;
        for node in self.dd.pool.values() {
            if node.lp_len > best_value {
                best_value = node.lp_len;
                self.dd.best_node = Some(node.clone());
            }
        }
    }

    fn maybe_squash(&mut self, i : usize) {
        match self.dd.mddtype {
            MDDType::Exact      => /* nothing to do ! */(),
            MDDType::Restricted => self.maybe_restrict(i),
            MDDType::Relaxed    => self.maybe_relax(i),
        }
    }
    fn maybe_restrict(&mut self, i: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let w = max(2, self.width.max_width(&self.dd));
            let ns = &self.ns;
            while self.dd.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.dd.is_exact = false;
                self.dd.current.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
                self.dd.current.truncate(w);
            }
        }
    }
    fn maybe_relax(&mut self, i: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let w = max(2, self.width.max_width(&self.dd));
            while self.dd.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.dd.is_exact = false;

                // actually squash the layer
                let merged = self.merge_overdue_nodes(w);

                if let Some(old) = Self::find_same_state(&mut self.dd.current, &merged.state) {
                    if old.is_exact {
                        //trace!("squash:: there existed an equivalent");
                        self.dd.cutset.push(old.clone());
                    }
                    old.merge(merged);
                } else {
                    self.dd.current.push(merged);
                }
            }
        }
    }
    fn merge_overdue_nodes(&mut self, w: usize) -> Node<T> {
        // 1. Sort the current layer so that the worst nodes are at the end.
        let ns = &self.ns;
        self.dd.current.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
        let (_keep, squash) = self.dd.current.split_at(w-1);

        // 2. merge state of the worst node into that of central
        let mut central = squash[0].clone();
        let mut states = vec![];
        for n in squash.iter() {
            states.push(&n.state);
        }
        central.is_exact = false;
        central.state    = self.relax.merge_states(&self.dd, states.as_slice());

        // 3. relax edges from the parents of all merged nodes (central + squashed)
        let mut arc = central.lp_arc.as_mut().unwrap();
        for n in squash.iter() {
            let narc = n.lp_arc.clone().unwrap();
            let cost = self.relax.relax_cost(&self.dd, narc.weight, &narc.src.state, &central.state, narc.decision);

            if n.lp_len - narc.weight + cost > central.lp_len {
                central.lp_len -= arc.weight;
                arc.src         = Rc::clone(&narc.src);
                arc.decision    = narc.decision;
                arc.weight      = cost;
                central.lp_len += arc.weight;
            }

            // n was an exact node, it must to to the cutset
            if n.is_exact {
                //trace!("squash:: squashed node was exact");
                self.dd.cutset.push(n.clone())
            }
        }

        // 4. drop overdue nodes
        self.dd.current.truncate(w - 1);
        central
    }
    fn find_same_state<'a>(current: &'a mut[Node<T>], state: &T) -> Option<&'a mut Node<T>> {
        for n in current.iter_mut() {
            if n.state.eq(state) {
                return Some(n);
            }
        }
        None
    }
}