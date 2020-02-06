use std::cmp::{max, min};
use std::hash::Hash;
use std::rc::Rc;

use compare::Compare;
use metrohash::MetroHashMap;

use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{Arc, MDD, MDDGenerator, MDDType, Node};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};
use crate::core::common::{Decision, Variable, VarSet};

const DUMMY : Variable = Variable(usize::max_value());

// --- MDD Data Structure -----------------------------------------------------
pub struct FlatMDD<T> where T: Eq + Clone {
    mddtype          : MDDType,
    layers           : [MetroHashMap<T, Node<T>>; 3],
    current          : usize,
    next             : usize,
    lel              : usize,

    last_assigned    : Variable,
    unassigned_vars  : VarSet,
    is_exact         : bool,
    best_node        : Option<Node<T>>
}

impl <T> Default for FlatMDD<T> where T: Hash + Clone + Eq {
    fn default() -> FlatMDD<T> {
        FlatMDD::new()
    }
}

/// Be careful: this macro lets you borrow any single layer from a flat mdd.
/// While this is generally safe, it is way too easy to use this macro to break
/// aliasing rules.
macro_rules! layer {
    ($dd:expr, $id:ident) => {
        unsafe { &*$dd.layers.as_ptr().add($dd.$id) }
    };
    ($dd:expr, mut $id:ident) => {
        unsafe { &mut *$dd.layers.as_mut_ptr().add($dd.$id) }
    };
}

impl <T> MDD<T> for FlatMDD<T> where T: Hash + Clone + Eq {
    fn mdd_type(&self) -> MDDType {
        self.mddtype
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
        if let Some(n) = &self.best_node {
            n.get_lp_len()
        } else {
            i32::min_value()
        }
    }
    fn best_node(&self) -> &Option<Node<T>> {
        &self.best_node
    }
    fn longest_path(&self) -> Vec<Decision> {
        if let Some(n) = &self.best_node {
            n.longest_path()
        } else {
            vec![]
        }
    }
}

/// Private functions
impl <T> FlatMDD<T> where T: Hash + Eq + Clone {
    fn new() -> FlatMDD<T> {
        FlatMDD {
            mddtype          : Exact,
            current          : 0,
            next             : 1,
            lel              : 2,

            last_assigned    : DUMMY,
            is_exact         : true,
            best_node        : None,
            unassigned_vars  : VarSet::all(0),
            layers           : [Default::default(), Default::default(), Default::default()]
        }
    }

    fn clear(&mut self) {
        self.mddtype       = Exact;
        self.last_assigned = DUMMY;
        self.is_exact      = true;
        self.best_node     = None;
        // unassigned vars holds stale data !

        self.current       = 0;
        self.next          = 1;
        self.lel           = 2;

        self.layers.iter_mut().for_each(|l| l.clear());
    }

    fn swap_current_lel(&mut self) {
        let tmp      = self.current;
        self.current = self.lel;
        self.lel     = tmp;
    }
    fn swap_current_next(&mut self) {
        let tmp      = self.current;
        self.current = self.next;
        self.next    = tmp;
    }
}

// --- MDD Generator -----------------------------------------------------------
pub struct FlatMDDGenerator<T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>> {

    pb               : PB,
    relax            : RLX,
    vs               : VS,
    width            : WDTH,
    ns               : NS,
    dd               : FlatMDD<T>
}

impl <T, PB, RLX, VS, WDTH, NS> MDDGenerator<T> for FlatMDDGenerator<T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>> {
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
    fn for_each_cutset_node<F>(&mut self, f: F) where F: FnMut(&mut Node<T>) {
        layer![self.dd, mut lel].values_mut().for_each(f)
    }
}

#[derive(Debug, Copy, Clone)]
struct Bounds {lb: i32, ub: i32}

impl <T, PB, RLX, VS, WDTH, NS> FlatMDDGenerator<T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>> {

    pub fn new(pb: PB, relax: RLX, vs: VS, width: WDTH, ns: NS) -> FlatMDDGenerator<T, PB, RLX, VS, WDTH, NS> {
        FlatMDDGenerator { pb, relax, vs, width, ns, dd: Default::default() }
    }
    fn develop(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>, best_lb: i32) {
        self.init(kind, vars, root);

        let bounds = Bounds {lb: best_lb, ub: root.ub};
        let mut i  = 0;
        let nbvars = self.nb_vars();

        while i < nbvars && !layer![self.dd, current].is_empty() {
            let var = self.select_var();
            if var.is_none() { break; }

            let was_exact = self.dd.is_exact;
            let var = var.unwrap();
            self.remove_var(var);
            self.unroll_layer(var, bounds);
            self.set_last_assigned(var);
            self.maybe_squash(i); // next
            self.move_to_next(was_exact);

            i += 1;
        }

        self.finalize()
    }
    fn unroll_layer(&mut self, var: Variable, bounds: Bounds) {
        let curr = layer![self.dd,  current];
        let next = layer![self.dd, mut next];

        for node in curr.values() {
            let domain = self.pb.domain_of(&node.state, var);
            for value in domain {
                let decision  = Decision{variable: var, value: *value};
                let branching = self.branch(node, decision);

                if let Some(old) = next.get_mut(&branching.state) {
                    old.merge(branching);
                } else if self.is_relevant(&branching, bounds) {
                    next.insert(branching.state.clone(), branching);
                }
            }
        }
    }

    fn move_to_next(&mut self, was_exact: bool) {
        if self.dd.is_exact != was_exact {
            self.dd.swap_current_lel();
        }
        self.dd.swap_current_next();
        layer![self.dd, mut next].clear();
    }
    fn nb_vars(&self) -> usize {
        self.dd.unassigned_vars.len()
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

        let mut vars = node.vars.clone();
        vars.remove(d.variable);
        Node::new(vars, state, node.lp_len + cost, Some(arc), node.is_exact)
    }
    fn is_relevant(&self, n: &Node<T>, bounds: Bounds) -> bool {
        min(self.relax.rough_ub(n.lp_len, &n.state), bounds.ub) > bounds.lb
    }

    fn init(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>) {
        self.dd.clear();
        self.dd.mddtype         = kind;
        self.dd.unassigned_vars = vars;

        layer![self.dd, mut current].insert(root.state.clone(), root.clone());
    }
    fn finalize(&mut self) {
        self.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.dd.best_node {
            let lp_length = best.lp_len;

            for n in layer![self.dd, mut lel].values_mut() {
                n.ub = lp_length.min(self.relax.rough_ub(n.lp_len, &n.state));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            layer![self.dd, mut lel].clear();
        }
    }
    fn find_best_node(&mut self) {
        let mut best_value = i32::min_value();
        for node in layer![self.dd, current].values() {
            if node.lp_len > best_value {
                best_value         = node.lp_len;
                self.dd.best_node  = Some(node.clone());
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
        if i >= 1 {
            let w    = max(2, self.width.max_width(&self.dd));
            let ns   = &self.ns;
            let next = layer![self.dd, mut next];

            let mut nodes = vec![];
            next.drain().for_each(|(_k,v)| nodes.push(v));

            while nodes.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.dd.is_exact = false;
                nodes.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
                nodes.truncate(w);
            }

            nodes.drain(..).for_each(|n| {next.insert(n.state.clone(), n);});
        }
    }
    fn maybe_relax(&mut self, i: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i >= 1 {
            let w    = max(2, self.width.max_width(&self.dd));
            let next = layer![self.dd, mut next];

            let mut nodes = vec![];
            next.drain().for_each(|(_k,v)| nodes.push(v));
            while nodes.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.dd.is_exact = false;

                // actually squash the layer
                let merged = self.merge_overdue_nodes(&mut nodes, w);
                if let Some(old) = Self::find_same_state(&mut nodes, &merged.state) {
                    old.merge(merged);
                } else {
                    nodes.push(merged);
                }
            }
            nodes.drain(..).for_each(|n| {next.insert(n.state.clone(), n);});
        }
    }
    fn merge_overdue_nodes(&mut self, nodes: &mut Vec<Node<T>>, w: usize) -> Node<T> {
        // 1. Sort the current layer so that the worst nodes are at the end.
        let ns   = &self.ns;

        nodes.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
        let (_keep, squash) = nodes.split_at(w-1);

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
        }

        // 4. drop overdue nodes
        nodes.truncate(w - 1);
        central
    }
    fn find_same_state<'a>(zone: &'a mut[Node<T>], state: &T) -> Option<&'a mut Node<T>> {
        for n in zone.iter_mut() {
            if n.state.eq(state) {
                return Some(n);
            }
        }
        None
    }
}