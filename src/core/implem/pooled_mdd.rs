use super::super::abstraction::mdd::*;
use crate::core::abstraction::dp::{Decision, Problem, Relaxation, Variable};
use std::cmp::max;
use std::rc::Rc;
use crate::core::abstraction::heuristics::{VariableHeuristic, NodeHeuristic, WidthHeuristic};
use std::collections::HashMap;
use bitset_fixed::BitSet;
use std::hash::Hash;
use std::marker::PhantomData;
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed};

// --- POOLED NODE -------------------------------------------------------------
#[derive(Clone)]
struct PooledNode<T> where T : Hash + Eq + Clone {
    state   : T,
    is_exact: bool,
    lp_len  : i32,
    ub      : i32,

    lp_arc  : Option<Arc<T, PooledNode<T>>>
}

impl <T> PooledNode<T> where T : Hash + Eq + Clone {
    fn new(state: T, is_exact: bool) -> PooledNode<T> {
        PooledNode{state, is_exact, lp_arc: None, lp_len: std::i32::MIN, ub: std::i32::MAX}
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
            self.ub     = max(self.lp_len, arc.src.get_lp_len() + arc.weight);
            self.lp_len = arc.src.get_lp_len() + arc.weight;
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

// --- POOLED MDD --------------------------------------------------------------
struct PooledMDD<T> where T : Hash + Eq + Clone {
    mddtype          : MDDType,

    pb               : Rc<dyn Problem<T>>,
    relax            : Rc<dyn Relaxation<T>>,
    vs               : Rc<dyn VariableHeuristic<T, PooledNode<T>>>,

    width            : Rc<dyn WidthHeuristic<T, PooledNode<T>>>,
    ns               : Rc<dyn NodeHeuristic<T, PooledNode<T>>>,

    pool             : HashMap<T, Rc<PooledNode<T>>>,
    current          : Vec<Rc<PooledNode<T>>>,
    cutset           : Vec<PooledNode<T>>,

    last_assigned    : Variable,
    unassigned_vars  : BitSet,
    is_exact         : bool,
    best_node        : Option<Rc<PooledNode<T>>>
}

impl <T> PooledMDD<T> where T : Clone + Hash + Eq {

    pub fn new(pb    : Rc<dyn Problem<T>>,
               relax : Rc<dyn Relaxation<T>>,
               vs    : Rc<dyn VariableHeuristic<T, PooledNode<T>>>,
               width : Rc<dyn WidthHeuristic<T, PooledNode<T>>>,
               ns    : Rc<dyn NodeHeuristic<T, PooledNode<T>>>) -> PooledMDD<T> {
        PooledMDD{
            mddtype          : Exact,
            last_assigned    : Variable(std::usize::MAX),
            unassigned_vars  : BitSet::new(pb.nb_vars()),
            is_exact         : true,
            best_node        : None,
            pb               : pb,
            relax            : relax,
            vs               : vs,
            width            : width,
            ns               : ns,
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

    pub fn relaxed(&mut self, vars: BitSet, root_s : T, root_lp : i32, root_ub: i32, best_lb : i32) {
        self.clear();
        self.mddtype         = Relaxed;
        self.unassigned_vars = vars;

        let root = PooledNode::<T>::new(root_s.clone(), true);
        self.pool.insert(root_s.clone(), Rc::new(root));


        let nbvars= self.unassigned_vars.count_ones();
        let mut i = 0;

        while i < nbvars && !self.pool.is_empty() {
            i += 1;
            let selected = self.vs.next_var(self, &self.unassigned_vars);

            self.pick_nodes_from_pool(selected);
            self.maybe_squash(i);

            self.unassigned_vars.set(selected.0, false);

            for node in self.current.iter() {
                let domain = self.pb.domain_of(&node.state, selected);
                for value in domain {
                    let decision = Decision{variable: selected, value: *value};
                    let state = self.pb.transition(&self.unassigned_vars,&decision);
                    let cost = self.pb.transition_cost(&self.unassigned_vars, &decision);
                    let arc = Arc {src: Rc::clone(&node), decision: decision, weight: cost, phantom: PhantomData};

                    let cs = &mut self.cutset;
                    let pl = &mut self.pool;
                    match pl.get_mut(&state) {
                        Some(old) => {
                            PooledMDD::maybe_add_to_cutset(cs, old, &arc);
                            Rc::get_mut(old).unwrap().add_arc(arc);
                        },
                        None => {
                            let mut dest = PooledNode::new(state, node.is_exact);
                            dest.add_arc(arc);
                            dest.set_ub(self.relax.rough_ub(dest.lp_len, &dest.state, &self.unassigned_vars));

                            if dest.ub > best_lb {
                                pl.insert(dest.state.clone(), Rc::new(dest));
                            }
                        }
                    }
                }
            }

            self.last_assigned = selected;
        }

        self.find_best_node();
    }

    fn pick_nodes_from_pool(&mut self, selected: Variable) {
        self.current.clear();

        let pl = &mut self.pool;
        let lr = &mut self.current;

        for (state, node) in pl.iter() {
            if self.pb.impacted_by(state, selected) {
                lr.push(Rc::clone(node));
            }
        }

        for node in lr.iter() {
            pl.remove(&node.state);
        }

    }
    fn maybe_squash(&mut self, i : u32) {
        match self.mddtype {
            Exact        => /* nothing to do ! */(),
            Relaxed      => self.maybe_relax(i),
            Restricted => self.maybe_restrict(i)
        }
    }
    fn maybe_add_to_cutset(cutset: &mut Vec<PooledNode<T>>, node: &PooledNode<T>, arc: &Arc<T, PooledNode<T>>) {
        if node.is_exact && !arc.src.is_exact {
            cutset.push(node.clone());
        }
    }

    fn find_best_node(&mut self) {
        let mut best_value = std::i32::MIN;
        for node in self.current.iter() {
            if node.lp_len > best_value {
                best_value = node.lp_len;
                self.best_node = Some(Rc::clone(node));
            }
        }
    }

    fn maybe_relax(&mut self, i: u32) {
        // TODO
    }

    fn maybe_restrict(&mut self, i: u32) {
        // TODO
    }
}

impl <T> MDD<T, PooledNode<T>> for PooledMDD<T> where T : Clone + Hash + Eq {
    fn mdd_type(&self) -> MDDType {
        self.mddtype
    }

    fn last_assigned(&self) -> Variable {
        self.last_assigned
    }

    fn unassigned_vars(&self) -> &BitSet {
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
    fn exact_cutset(&self) -> &[PooledNode<T>] {
        &self.cutset
    }
    fn current_layer(&self) -> &[Rc<PooledNode<T>>] {
        &self.current
    }
}