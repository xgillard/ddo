use super::super::abstraction::mdd::*;
use crate::core::abstraction::dp::{Decision, Problem, Relaxation, Variable, VarSet};
use std::cmp::{max, Ordering};
use std::rc::Rc;
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic};
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};
use crate::core::utils::Decreasing;
use std::cmp::Ordering::Equal;

// --- POOLED NODE -------------------------------------------------------------
#[derive(Clone, Eq, PartialEq)]
pub struct PooledNode<T> where T : Hash + Eq + Clone {
    pub state   : T,
    pub is_exact: bool,
    pub lp_len  : i32,
    pub lp_arc  : Option<Arc<T, PooledNode<T>>>,

    pub ub      : i32
}

impl <T> PooledNode<T> where T : Hash + Eq + Clone {
    pub fn new(state: T, lp_len: i32, lp_arc: Option<Arc<T, PooledNode<T>>>, is_exact: bool) -> PooledNode<T> {
        PooledNode{state, is_exact, lp_len, lp_arc, ub: std::i32::MAX}
    }
}

impl <T> Node<T, PooledNode<T>> for PooledNode<T> where T : Hash + Eq + Clone {
    fn is_exact(&self) -> bool {
        self.is_exact
    }
    fn get_state(&self)-> &T {
        &self.state
    }
    fn get_lp_len(&self) -> i32 {
        self.lp_len
    }
    fn get_lp_arc(&self) -> &Option<Arc<T, Self>> {
        &self.lp_arc
    }
    fn get_ub(&self) -> i32 {
        self.ub
    }
    fn set_ub(&mut self, ub: i32) {
        self.ub = max(self.ub, ub);
    }

    fn add_arc (&mut self, arc: Arc<T, Self>) {
        self.is_exact &= arc.src.is_exact();

        if self.lp_arc.is_none() || arc.src.get_lp_len() + arc.weight > self.lp_len {
            self.lp_len = arc.src.lp_len + arc.weight;
            self.lp_arc = Some(arc);
        }
    }

    fn longest_path(&self) -> Vec<Decision> {
        let mut ret = vec![];
        let mut arc = &self.lp_arc;

        while arc.is_some() {
            let a = arc.as_ref().unwrap();
            ret.push(a.decision);
            arc = &a.src.get_lp_arc();
        }

        ret
    }
}

impl <T> Ord for PooledNode<T> where T : Hash + Eq + Clone + Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl <T> PartialOrd for PooledNode<T> where T : Hash + Eq + Clone + PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let cmp_ub = self.ub.cmp(&other.ub);
        if cmp_ub != Equal {
            Some(cmp_ub)
        } else {
            let cmp_lp = self.lp_len.cmp(&other.lp_len);
            if cmp_lp != Equal {
                Some(cmp_lp)
            } else {
                self.state.partial_cmp(other.get_state())
            }
        }
    }
}

// --- POOLED MDD --------------------------------------------------------------
pub struct PooledMDD<T, NS>
    where T : Hash + Eq + Clone,
          NS: Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering {
    mddtype          : MDDType,

    pb               : Rc<dyn Problem<T>>,
    relax            : Rc<dyn Relaxation<T>>,
    vs               : Rc<dyn VariableHeuristic<T, PooledNode<T>>>,

    width            : Rc<dyn WidthHeuristic<T, PooledNode<T>>>,
    ns               : Decreasing<PooledNode<T>, NS>,

    pool             : HashMap<T, PooledNode<T>>,
    current          : Vec<PooledNode<T>>,
    cutset           : Vec<PooledNode<T>>,

    last_assigned    : Variable,
    unassigned_vars  : VarSet,
    is_exact         : bool,
    best_node        : Option<PooledNode<T>>
}

impl <T, NS> PooledMDD<T, NS>
    where T : Clone + Hash + Eq,
          NS: Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering {

    pub fn new(pb    : Rc<dyn Problem<T>>,
               relax : Rc<dyn Relaxation<T>>,
               vs    : Rc<dyn VariableHeuristic<T, PooledNode<T>>>,
               width : Rc<dyn WidthHeuristic<T, PooledNode<T>>>,
               ns    : NS)
        -> PooledMDD<T, NS> {
        PooledMDD{
            mddtype          : Exact,
            last_assigned    : Variable(std::usize::MAX),
            unassigned_vars  : VarSet::all(pb.nb_vars()),
            is_exact         : true,
            best_node        : None,
            pb               : pb,
            relax            : relax,
            vs               : vs,
            width            : width,
            ns               : Decreasing::from(ns),
            pool             : HashMap::new(),
            current          : vec![],
            cutset           : vec![]}
    }

    pub fn clear(&mut self) {
        self.mddtype          = Exact;
        self.last_assigned    = Variable(std::usize::MAX);
        self.is_exact         = true;
        self.best_node        = None;
        // unassigned vars holds stale data !

        self.pool             .clear();
        self.current          .clear();
        self.cutset           .clear();
    }
    pub fn exact(&mut self, vars: VarSet, root: &PooledNode<T>, best_lb : i32) {
        self.develop(Exact, vars, root, best_lb);
    }
    pub fn restricted(&mut self, vars: VarSet, root: &PooledNode<T>, best_lb : i32) {
        self.develop(Restricted, vars, root, best_lb);
    }
    pub fn relaxed(&mut self, vars: VarSet, root: &PooledNode<T>, best_lb : i32) {
        self.develop(Relaxed, vars, root, best_lb);
    }
    fn develop(&mut self, kind: MDDType, vars: VarSet, root: &PooledNode<T>, best_lb : i32) {
        self.clear();
        self.mddtype         = kind;
        self.unassigned_vars = vars;

        self.pool.insert(root.state.clone(), root.clone());

        let nbvars= self.unassigned_vars.len();
        let mut i = 0;

        while i < nbvars && !self.pool.is_empty() {
            let selected = self.vs.next_var(self, &self.unassigned_vars);
            if selected.is_none() {
                break;
            }

            let selected = selected.unwrap();
            self.pick_nodes_from_pool(selected);
            self.maybe_squash(i);

            self.unassigned_vars.remove(selected);

            for node in self.current.iter() {
                let domain = self.pb.domain_of(&node.state, selected);
                for value in domain {
                    let decision = Decision{variable: selected, value: *value};
                    let state = self.pb.transition(&node.state, &self.unassigned_vars,&decision);
                    let cost = self.pb.transition_cost(&node.state, &self.unassigned_vars, &decision);
                    let arc = Arc {src: Rc::new(node.clone()), decision: decision, weight: cost, phantom: PhantomData};

                    match self.pool.get_mut(&state) {
                        Some(old) => {
                            if old.is_exact && !arc.src.is_exact {
                                self.cutset.push(old.clone());
                            }
                            old.add_arc(arc);
                        },
                        None => {
                            let lp_len = node.lp_len + cost;
                            let ub = root.ub.min(self.relax.rough_ub(lp_len, &state));

                            if ub > best_lb {
                                self.pool.insert(state.clone(), PooledNode::new(state, lp_len, Some(arc), node.is_exact));
                            }
                        }
                    }
                }
            }

            self.last_assigned = selected;
            i += 1;
        }

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        for node in self.cutset.iter_mut() {
            node.ub = root.ub.min(self.relax.rough_ub(node.lp_len, &node.state));
        }

        self.find_best_node();
    }

    fn pick_nodes_from_pool(&mut self, selected: Variable) {
        self.current.clear();

        let pl = &mut self.pool;
        let lr = &mut self.current;

        for (state, node) in pl.iter() {
            if self.pb.impacted_by(state, selected) {
                lr.push(node.clone());
            }
        }

        for node in lr.iter() {
            pl.remove(&node.state);
        }

    }
    fn maybe_squash(&mut self, i : usize) {
        match self.mddtype {
            MDDType::Exact      => /* nothing to do ! */(),
            MDDType::Relaxed    => self.maybe_relax(i),
            MDDType::Restricted => self.maybe_restrict(i),
        }
    }

    fn find_best_node(&mut self) {
        let mut best_value = std::i32::MIN;
        for (_, node) in self.pool.iter() {
            if node.lp_len > best_value {
                best_value = node.lp_len;
                self.best_node = Some(node.clone());
            }
        }
    }

    fn maybe_relax(&mut self, i: usize) {
        if i <= 1 {
            return /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */;
        } else {
            let w = max(2, self.width.max_width(self));
            let ns = &self.ns;
            while self.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;

                // actually squash the layer
                self.current.sort_by(|a, b| ns.compare(a, b));
                let (_keep, squash) = self.current.split_at_mut(w-1);

                let mut central = squash[0].clone();

                // 1. merge state of the worst node into that of central
                let mut states = vec![];
                for n in squash.iter() {
                    states.push(&n.state);
                }
                central.state = self.relax.merge_states(states.as_slice());
                central.is_exact = false;

                // 2. relax edges from the parents of all merged nodes (central + squashed)
                let mut arc = central.lp_arc.as_mut().unwrap();
                for n in squash.iter_mut() {
                    let narc = n.lp_arc.clone().unwrap();

                    let cost = self.relax.relax_cost(&narc.src.state, &central.state, &narc.decision);

                    if n.lp_len - narc.weight + cost > central.lp_len {
                        central.lp_len -= arc.weight;
                        arc.src     = Rc::clone(&narc.src);
                        arc.decision= narc.decision;
                        arc.weight  = cost;
                        central.lp_len += arc.weight;
                    }

                    // n was an exact node, it must to to the cutset
                    if n.is_exact {
                        self.cutset.push(n.clone())
                    }
                }

                // 3. all nodes have been merged into central: resize the current layer and add central
                let mut must_add = true;
                self.current.truncate(w - 1);
                for n in self.current.iter_mut() {
                    if n.state.eq(&central.state) {
                        // if n is exact, it must go to the cutset
                        if n.is_exact {
                            self.cutset.push(n.clone());
                        }

                        must_add   = false;
                        n.is_exact = false;
                        if n.lp_len < central.lp_len {
                            n.lp_len = central.lp_len;
                            n.lp_arc = central.lp_arc.clone();
                            n.ub     = central.ub;
                        }

                        break;
                    }
                }
                if must_add {
                    self.current.push(central);
                }
            }
        }
    }

    fn maybe_restrict(&mut self, i: usize) {
        if i <= 1 {
            return /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */;
        } else {
            let w = max(2, self.width.max_width(self));
            let ns = &self.ns;
            while self.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;
                self.current.sort_by(|a, b| ns.compare(a, b));
                self.current.truncate(w);
            }
        }
    }
}

impl <T, NS> MDD<T, PooledNode<T>> for PooledMDD<T, NS>
    where T : Clone + Hash + Eq,
          NS: Fn(&PooledNode<T>, &PooledNode<T>) -> Ordering {

    fn mdd_type(&self) -> MDDType {
        self.mddtype
    }

    fn current_layer(&self) -> &[PooledNode<T>] {
        &self.current
    }

    fn next_layer(&self) -> &HashMap<T, PooledNode<T>> {
        &self.pool
    }

    fn exact_cutset(&self) -> &[PooledNode<T>] {
        &self.cutset
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
    fn longest_path(&self) -> Vec<Decision> {
        if self.best_node.is_none() {
            vec![]
        } else {
            self.best_node.as_ref().unwrap().longest_path()
        }
    }
}