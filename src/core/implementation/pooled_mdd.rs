use std::cmp::min;
use std::hash::Hash;
use std::rc::Rc;

use compare::Compare;
use metrohash::MetroHashMap;

use crate::core::common::{Arc, Decision, Node, NodeInfo, Variable, VarSet};
use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{MDD, MDDType};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};

// --- POOLED MDD --------------------------------------------------------------
pub struct PooledMDD<'a, T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>> {

    pb               : &'a PB,
    relax            : RLX,
    vs               : VS,
    width            : WDTH,
    ns               : NS,

    mddtype          : MDDType,
    pool             : MetroHashMap<T, NodeInfo<T>>,
    current          : Vec<Node<T>>,
    cutset           : Vec<Node<T>>,

    last_assigned    : Variable,
    unassigned_vars  : VarSet,
    is_exact         : bool,
    best_node        : Option<NodeInfo<T>>
}

impl <'a, T, PB, RLX, VS, WDTH, NS> MDD<T> for PooledMDD<'a, T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>> {

    fn root(&self) -> Node<T> {
        self.pb.root_node()
    }
    fn exact(&mut self, vars: VarSet, root: &Node<T>, best_lb: i32) {
        self.develop(Exact, vars, root, best_lb, usize::max_value());
    }
    fn restricted(&mut self, vars: VarSet, root: &Node<T>, best_lb: i32) {
        let w = self.width.max_width(&vars);
        self.develop(Restricted, vars, root, best_lb, w);
    }
    fn relaxed(&mut self, vars: VarSet, root: &Node<T>, best_lb: i32) {
        let w = self.width.max_width(&vars);
        self.develop(Relaxed, vars, root, best_lb, w);
    }
    fn for_each_cutset_node<F>(&mut self, mut f: F) where F: FnMut(&T, &mut NodeInfo<T>) {
        self.cutset.iter_mut().for_each(|n| (f)(&n.state, &mut n.info))
    }
    fn consume_cutset<F>(&mut self, mut f: F) where F: FnMut(T, NodeInfo<T>) {
        self.cutset.drain(..).for_each(|n| (f)(n.state, n.info))
    }
    fn mdd_type(&self) -> MDDType {
        self.mddtype
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
    fn best_node(&self) -> &Option<NodeInfo<T>> {
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

/// Utility structure to represent upper and lower bounds
#[derive(Debug, Copy, Clone)]
struct Bounds {lb: i32, ub: i32}

/// Private functions
impl <'a, T, PB, RLX, VS, WDTH, NS> PooledMDD<'a, T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>> {

    pub fn new(pb: &'a PB, relax: RLX, vs: VS, width: WDTH, ns: NS) -> Self {
        PooledMDD {
            pb, relax, vs, width, ns,

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

    fn develop(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>, best_lb : i32, w: usize) {
        self.init(kind, vars, root);

        let bounds = Bounds {lb: best_lb, ub: root.info.ub};
        let mut i  = 0;
        let nbvars = self.nb_vars();

        while i < nbvars && !self.exhausted() {
            let var = self.select_var();
            if var.is_none() {
                break;
            }

            let var = var.unwrap();
            self.pick_nodes_from_pool(var);
            self.maybe_squash(i, w);
            self.remove_var(var);
            self.unroll_layer(var, bounds);
            self.set_last_assigned(var);
            i += 1;
        }

        self.finalize()
    }
    fn unroll_layer(&mut self, var: Variable, bounds: Bounds) {
        for node in self.current.iter() {
            let domain = self.pb.domain_of(&node.state, var);
            for value in domain {
                let decision  = Decision{variable: var, value: *value};
                let branching = self.branch(node, decision);

                if let Some(old) = self.pool.get_mut(&branching.state) {
                    if old.is_exact && !branching.info.is_exact {
                        //trace!("main loop:: old was exact but new was not");
                        self.cutset.push(Node{state: branching.state, info: old.clone()});
                    } else if !old.is_exact && branching.info.is_exact {
                        //trace!("main loop:: new was exact but old was not");
                        self.cutset.push(branching.clone());
                    }
                    old.merge(branching.info)
                } else if self.is_relevant(bounds, &branching.state, &branching.info) {
                    self.pool.insert(branching.state, branching.info);
                }
            }
        }
    }
    fn pick_nodes_from_pool(&mut self, var: Variable) {
        self.current.clear();

        // Add all selected nodes to the next layer
        let mut items = vec![];
        for (s, _i) in self.pool.iter() {
            if self.pb.impacted_by(s, var) {
                items.push(s.clone());
            }
        }

        // Remove all nodes that belong to the current layer from the pool
        for state in items {
            let info = self.pool.remove(&state).unwrap();
            self.current.push(Node{state, info});
        }
    }

    fn nb_vars(&self) -> usize {
        self.unassigned_vars.len()
    }
    fn exhausted(&self) -> bool {
        self.pool.is_empty()
    }
    fn select_var(&self) -> Option<Variable> {
        self.vs.next_var(&self.unassigned_vars)
    }
    fn remove_var(&mut self, var: Variable) {
        self.unassigned_vars.remove(var)
    }
    fn set_last_assigned(&mut self, var: Variable) {
        self.last_assigned = var
    }
    fn transition_state(&self, node: &Node<T>, d: Decision) -> T {
        self.pb.transition(&node.state, &self.unassigned_vars, d)
    }
    fn transition_cost(&self, node: &Node<T>, d: Decision) -> i32 {
        self.pb.transition_cost(&node.state, &self.unassigned_vars, d)
    }
    fn branch(&self, node: &Node<T>, d: Decision) -> Node<T> {
        let state = self.transition_state(node, d);
        let cost  = self.transition_cost (node, d);
        let arc   = Arc {src: Rc::new(node.clone()), decision: d};

        Node::new(state, node.info.lp_len + cost, Some(arc), node.info.is_exact)
    }
    fn is_relevant(&self, bounds: Bounds, state: &T, info: &NodeInfo<T>) -> bool {
        min(self.relax.estimate_ub(state, info), bounds.ub) > bounds.lb
    }

    fn init(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>) {
        self.clear();
        self.mddtype         = kind;
        self.unassigned_vars = vars;

        self.pool.insert(root.state.clone(), root.info.clone());
    }
    fn finalize(&mut self) {
        self.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.best_node {
            let lp_length = best.lp_len;

            for n in self.cutset.iter_mut() {
                n.info.ub = lp_length.min(self.relax.estimate_ub(&n.state, &n.info));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            self.cutset.clear();
        }
    }
    fn find_best_node(&mut self) {
        let mut best_value = std::i32::MIN;
        for info in self.pool.values() {
            if info.lp_len > best_value {
                best_value = info.lp_len;
                self.best_node = Some(info.clone());
            }
        }
    }

    fn maybe_squash(&mut self, i : usize, w: usize) {
        match self.mddtype {
            MDDType::Exact      => /* nothing to do ! */(),
            MDDType::Restricted => self.maybe_restrict(i, w),
            MDDType::Relaxed    => self.maybe_relax(i, w),
        }
    }
    fn maybe_restrict(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let ns = &self.ns;
            while self.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;
                self.current.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
                self.current.truncate(w);
            }
        }
    }
    fn maybe_relax(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            while self.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;

                // actually squash the layer
                let merged = self.merge_overdue_nodes(w);

                if let Some(old) = Self::find_same_state(&mut self.current, &merged.state) {
                    if old.info.is_exact {
                        //trace!("squash:: there existed an equivalent");
                        self.cutset.push(old.clone());
                    }
                    old.info.merge(merged.info);
                } else {
                    self.current.push(merged);
                }
            }
        }
    }
    fn merge_overdue_nodes(&mut self, w: usize) -> Node<T> {
        // 1. Sort the current layer so that the worst nodes are at the end.
        let ns = &self.ns;
        self.current.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
        let (_keep, squash) = self.current.split_at(w-1);

        // 2. merge the nodes
        let merged = self.relax.merge_nodes(squash);

        // 3. make sure to keep the cutset complete
        for n in squash {
            if n.info.is_exact {
                self.cutset.push(n.clone())
            }
        }

        // 4. drop overdue nodes
        self.current.truncate(w - 1);
        merged
    }
    fn find_same_state<'b>(current: &'b mut[Node<T>], state: &T) -> Option<&'b mut Node<T>> {
        for n in current.iter_mut() {
            if n.state.eq(state) {
                return Some(n);
            }
        }
        None
    }
}