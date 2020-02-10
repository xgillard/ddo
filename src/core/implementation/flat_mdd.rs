use std::cmp::min;
use std::hash::Hash;
use std::rc::Rc;

use compare::Compare;
use metrohash::MetroHashMap;

use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{Arc, MDD, MDDGenerator, MDDType, Node, NodeInfo};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};
use crate::core::common::{Decision, Variable, VarSet};

const DUMMY : Variable = Variable(usize::max_value());
// --- MDD Data Structure -----------------------------------------------------
pub struct FlatMDD<T> where T: Eq + Clone {
    mddtype          : MDDType,
    layers           : [MetroHashMap<T, NodeInfo<T>>; 3],
    current          : usize,
    next             : usize,
    lel              : usize,

    last_assigned    : Variable,
    unassigned_vars  : VarSet,
    is_exact         : bool,
    best_node        : Option<NodeInfo<T>>
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
            n.lp_len
        } else {
            i32::min_value()
        }
    }
    fn best_node(&self) -> &Option<NodeInfo<T>> {
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
pub struct FlatMDDGenerator<'a, T, PB, RLX, VS, WDTH, NS>
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
    dd               : FlatMDD<T>
}

impl <'a, T, PB, RLX, VS, WDTH, NS> MDDGenerator<T> for FlatMDDGenerator<'a, T, PB, RLX, VS, WDTH, NS>
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
    fn mdd(&self) -> &dyn MDD<T> {
        &self.dd
    }
    fn for_each_cutset_node<F>(&mut self, mut f: F) where F: FnMut(&T, &mut NodeInfo<T>) {
        layer![self.dd, mut lel].iter_mut().for_each(|(k, v)| (f)(k, v))
    }
    fn consume_cutset<F>(&mut self, mut f: F) where F: FnMut(T, NodeInfo<T>) {
        layer![self.dd, mut lel].drain().for_each(|(k, v)| (f)(k, v))
    }
}

#[derive(Debug, Copy, Clone)]
struct Bounds {lb: i32, ub: i32}

impl <'a, T, PB, RLX, VS, WDTH, NS> FlatMDDGenerator<'a, T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : Compare<Node<T>> {

    pub fn new(pb: &'a PB, relax: RLX, vs: VS, width: WDTH, ns: NS) -> FlatMDDGenerator<T, PB, RLX, VS, WDTH, NS> {
        FlatMDDGenerator { pb, relax, vs, width, ns, dd: Default::default() }
    }
    fn develop(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>, best_lb: i32, w: usize) {
        self.init(kind, vars, root);

        let bounds = Bounds {lb: best_lb, ub: root.info.ub};
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
            self.maybe_squash(i, w); // next
            self.move_to_next(was_exact);

            i += 1;
        }

        self.finalize()
    }
    fn unroll_layer(&mut self, var: Variable, bounds: Bounds) {
        let curr = layer![self.dd,  current];
        let next = layer![self.dd, mut next];

        for (state, info) in curr.iter() {
            let domain = self.pb.domain_of(state, var);
            for value in domain {
                let decision  = Decision{variable: var, value: *value};
                let branching = self.branch(Node{state: state.clone(), info: info.clone()}, decision);

                if let Some(old) = next.get_mut(&branching.state) {
                    old.merge(branching.info);
                } else if self.is_relevant(bounds, &branching.state, &branching.info) {
                    next.insert(branching.state, branching.info);
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
    fn branch(&self, node: Node<T>, d: Decision) -> Node<T> {
        let state = self.transition_state(&node, d);
        let cost  = self.transition_cost (&node, d);

        let len   = node.info.lp_len;
        let exct  = node.info.is_exact;
        let arc   = Arc {src: Rc::new(node), decision: d};

        Node::new(state, len + cost, Some(arc), exct)
    }
    fn is_relevant(&self, bounds: Bounds, state: &T, info: &NodeInfo<T>) -> bool {
        min(self.relax.estimate_ub(state, info), bounds.ub) > bounds.lb
    }

    fn init(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>) {
        self.dd.clear();
        self.dd.mddtype         = kind;
        self.dd.unassigned_vars = vars;

        layer![self.dd, mut current].insert(root.state.clone(), root.info.clone());
    }
    fn finalize(&mut self) {
        self.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.dd.best_node {
            let lp_length = best.lp_len;

            for (state, info) in layer![self.dd, mut lel].iter_mut() {
                info.ub = lp_length.min(self.relax.estimate_ub(state, info));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            layer![self.dd, mut lel].clear();
        }
    }
    fn find_best_node(&mut self) {
        let mut best_value = i32::min_value();
        for info in layer![self.dd, current].values() {
            if info.lp_len > best_value {
                best_value         = info.lp_len;
                self.dd.best_node  = Some(info.clone());
            }
        }
    }
    fn maybe_squash(&mut self, i : usize, w: usize) {
        match self.dd.mddtype {
            MDDType::Exact      => /* nothing to do ! */(),
            MDDType::Restricted => self.maybe_restrict(i, w),
            MDDType::Relaxed    => self.maybe_relax(i, w),
        }
    }
    fn maybe_restrict(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let ns   = &self.ns;
            let next = layer![self.dd, mut next];

            let mut nodes = vec![];
            next.drain().for_each(|(k,v)| nodes.push(Node{state: k, info: v}));

            while nodes.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.dd.is_exact = false;
                nodes.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
                nodes.truncate(w);
            }

            nodes.drain(..).for_each(|n| {next.insert(n.state, n.info);});
        }
    }
    fn maybe_relax(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let ns   = &self.ns;
            let next = layer![self.dd, mut next];

            if next.len() > w {

                let mut nodes = vec![];
                nodes.reserve_exact(next.len());

                next.drain().for_each(|(k,v)| {
                    nodes.push(Node{state: k, info: v});
                });

                nodes.sort_unstable_by(|a, b| ns.compare(a, b).reverse());

                let (keep, squash) = nodes.split_at(w-1);

                // we do squash the current layer so the mdd is now inexact
                self.dd.is_exact = false;

                // actually squash the layer
                let merged = self.relax.merge_nodes(squash);

                for n in keep.to_vec().drain(..) {
                    next.insert(n.state, n.info);
                }

                if let Some(old) = next.get_mut(&merged.state) {
                    old.merge(merged.info);
                } else {
                    next.insert(merged.state, merged.info);
                }
            }
        }
    }
}