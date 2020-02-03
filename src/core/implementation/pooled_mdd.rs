use std::cmp::max;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use crate::core::abstraction::dp::{Decision, Problem, Relaxation, Variable, VarSet};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic, NodeOrdering};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};

use super::super::abstraction::mdd::*;

// --- POOLED MDD --------------------------------------------------------------
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

pub struct PooledMDD<T> where T: Hash + Eq + Clone {
    mddtype          : MDDType,
    pool             : HashMap<T, Node<T>>,
    current          : Vec<Node<T>>,
    cutset           : Vec<Node<T>>,

    last_assigned    : Variable,
    unassigned_vars  : VarSet,
    is_exact         : bool,
    best_node        : Option<Node<T>>
}

impl <T> PooledMDD<T> where T    : Hash + Eq + Clone {
    pub fn new() -> PooledMDD<T> {
        PooledMDD {
            mddtype          : Exact,
            last_assigned    : Variable(std::usize::MAX),
            unassigned_vars  : VarSet::all(0),
            is_exact         : true,
            best_node        : None,
            pool             : HashMap::new(),
            current          : vec![],
            cutset           : vec![]
        }
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

    fn find_best_node(&mut self) {
        let mut best_value = std::i32::MIN;
        for (_, node) in self.pool.iter() {
            if node.lp_len > best_value {
                best_value = node.lp_len;
                self.best_node = Some(node.clone());
            }
        }
    }
}

impl <T, PB, RLX, VS, WDTH, NS> PooledMDDGenerator<T, PB, RLX, VS, WDTH, NS>
    where T    : Hash + Eq + Clone,
          PB   : Problem<T>,
          RLX  : Relaxation<T>,
          VS   : VariableHeuristic<T>,
          WDTH : WidthHeuristic<T>,
          NS   : NodeOrdering<T> {

    pub fn new(pb: Rc<PB>, relax: RLX, vs: VS, width: WDTH, ns: NS) -> PooledMDDGenerator<T, PB, RLX, VS, WDTH, NS> {
        PooledMDDGenerator{
            pb,
            relax,
            vs,
            width,
            ns,
            dd: PooledMDD::new()
        }
    }


    fn develop(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>, best_lb : i32) {
        self.dd.clear();
        self.dd.mddtype         = kind;
        self.dd.unassigned_vars = vars;

        self.dd.pool.insert(root.state.clone(), root.clone());

        let nbvars= self.dd.unassigned_vars.len();
        let mut i = 0;

        while i < nbvars && !self.dd.pool.is_empty() {
            let selected = self.vs.next_var(&self.dd, &self.dd.unassigned_vars);
            if selected.is_none() {
                break;
            }

            let selected = selected.unwrap();
            self.pick_nodes_from_pool(selected);

            // FIXME: Just to keep it perfectly reproductible
            //let ns = &self.ns;
            //let lr = &mut self.current;
            //lr.sort_unstable_by(|a, b| ns(a, b));
            //trace!("v {}, current {}, pool {}, cutset {}", selected.0, self.current.len(), self.pool.len(), self.cutset.len());

            self.maybe_squash(i);

            self.dd.unassigned_vars.remove(selected);

            for node in self.dd.current.iter() {
                let domain = self.pb.domain_of(&node.state, selected);
                for value in domain {
                    let decision = Decision{variable: selected, value: *value};
                    let state = self.pb.transition(&node.state, &self.dd.unassigned_vars,&decision);
                    let cost = self.pb.transition_cost(&node.state, &self.dd.unassigned_vars, &decision);
                    let arc = Arc {src: Rc::new(node.clone()), decision, weight: cost};

                    let dest = Node::new(state, node.lp_len + cost, Some(arc), node.is_exact);
                    match self.dd.pool.get_mut(&dest.state) {
                        Some(old) => {
                            if old.is_exact && !dest.is_exact {
                                //trace!("main loop:: old was exact but new was not");
                                self.dd.cutset.push(old.clone());
                            }
                            if !old.is_exact && dest.is_exact {
                                //trace!("main loop:: new was exact but old was not");
                                self.dd.cutset.push(dest.clone());
                            }
                            // FIXME: maybe call add_arc here ?
                            if old.lp_len < dest.lp_len {
                                old.lp_len = dest.lp_len;
                                old.lp_arc = dest.lp_arc;
                            }
                            old.is_exact &= dest.is_exact;
                        },
                        None => {
                            let ub = root.ub.min(self.relax.rough_ub(dest.lp_len, &dest.state));

                            if ub > best_lb {
                                self.dd.pool.insert(dest.state.clone(), dest);
                            }
                        }
                    }
                }
            }

            self.dd.last_assigned = selected;
            i += 1;
        }

        self.dd.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.dd.best_node {
            let lp_length = best.lp_len;

            for node in self.dd.cutset.iter_mut() {
                node.ub = lp_length.min(self.relax.rough_ub(node.lp_len, &node.state));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            self.dd.cutset.clear();
        }
    }

    fn pick_nodes_from_pool(&mut self, selected: Variable) {
        self.dd.current.clear();

        let pl = &mut self.dd.pool;
        let lr = &mut self.dd.current;

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
        match self.dd.mddtype {
            MDDType::Exact      => /* nothing to do ! */(),
            MDDType::Relaxed    => self.maybe_relax(i),
            MDDType::Restricted => self.maybe_restrict(i),
        }
    }

    fn maybe_relax(&mut self, i: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let w = max(2, self.width.max_width(&self.dd));
            let ns = &self.ns;
            while self.dd.current.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.dd.is_exact = false;

                // actually squash the layer
                self.dd.current.sort_unstable_by(|a, b| ns.compare(a, b).reverse());
                let (_keep, squash) = self.dd.current.split_at(w-1);

                let mut central = squash[0].clone();

                // 1. merge state of the worst node into that of central
                let mut states = vec![];
                for n in squash.iter() {
                    states.push(&n.state);
                }
                let central_state = self.relax.merge_states(&self.dd, states.as_slice());

                // 2. relax edges from the parents of all merged nodes (central + squashed)
                let mut arc = central.lp_arc.as_mut().unwrap();
                for n in squash.iter() {
                    let narc = n.lp_arc.clone().unwrap();

                    let cost = self.relax.relax_cost(&self.dd, narc.weight, &narc.src.state, &central_state, &narc.decision);

                    if n.lp_len - narc.weight + cost > central.lp_len {
                        central.lp_len -= arc.weight;
                        arc.src     = Rc::clone(&narc.src);
                        arc.decision= narc.decision;
                        arc.weight  = cost;
                        central.lp_len += arc.weight;
                    }

                    // n was an exact node, it must to to the cutset
                    if n.is_exact {
                        //trace!("squash:: squashed node was exact");
                        self.dd.cutset.push(n.clone())
                    }
                }

                central.state    = central_state;
                central.is_exact = false;

                // 3. all nodes have been merged into central: resize the current layer and add central
                let mut must_add = true;
                self.dd.current.truncate(w - 1);
                for n in self.dd.current.iter_mut() {
                    if n.state.eq(&central.state) {
                        // if n is exact, it must go to the cutset
                        if n.is_exact {
                            //trace!("squash:: there existed an equivalent");
                            self.dd.cutset.push(n.clone());
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
                    self.dd.current.push(central);
                }
            }
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
    fn mdd(&self) -> &dyn MDD<T> { &self.dd }
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

    fn next_layer(&self) -> &HashMap<T, Node<T>> {
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