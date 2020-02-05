use std::hash::Hash;

use compare::Compare;

use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::heuristics::{VariableHeuristic, WidthHeuristic};
use crate::core::abstraction::mdd::{MDD, MDDGenerator, MDDType, Node};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};
use crate::core::common::{Decision, Variable, VarSet};

const DUMMY : Variable = Variable(usize::max_value());

// --- MDD Data Structure -----------------------------------------------------
pub struct FlatMDD<T> where T: Eq + Clone {
    mddtype          : MDDType,
    layers           : [Vec<Node<T>>; 3],
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

impl <T> MDD<T> for FlatMDD<T> where T: Hash + Clone + Eq {
    fn mdd_type(&self) -> MDDType {
        self.mddtype
    }
    fn current_layer(&self) -> &[Node<T>] {
        &self.layers[self.current]
    }
    fn exact_cutset(&self) -> &[Node<T>] {
        &self.layers[self.lel]
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
            layers           : [Default::default(), Default::default(), Default::default()],
            current          : 0,
            next             : 1,
            lel              : 2,

            last_assigned    : DUMMY,
            unassigned_vars  : VarSet::all(0),
            is_exact         : true,
            best_node        : None
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

        while i < nbvars && !self.dd.layers[self.dd.next].is_empty() {
            let var = self.select_var();
            if var.is_none() {
                break;
            }

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
        // TODO
        unimplemented!()
    }
    fn move_to_next(&mut self, was_exact: bool) {
        if self.dd.is_exact != was_exact {
            self.dd.swap_current_lel();
        }
        self.dd.swap_current_next();
        self.dd.layers[self.dd.next].clear();
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

    fn init(&mut self, kind: MDDType, vars: VarSet, root: &Node<T>) {
        self.dd.clear();
        self.dd.mddtype         = kind;
        self.dd.unassigned_vars = vars;

        self.dd.layers[self.dd.next].push(root.clone());
    }
    fn finalize(&mut self) {
        self.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.dd.best_node {
            let lp_length = best.lp_len;

            for n in self.dd.layers[self.dd.lel].iter_mut() {
                n.ub = lp_length.min(self.relax.rough_ub(n.lp_len, &n.state));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            self.dd.layers[self.dd.lel].clear();
        }
    }
    fn find_best_node(&mut self) {
        let mut best_value = i32::min_value();
        for node in self.dd.layers[self.dd.next].iter() {
            if node.lp_len > best_value {
                best_value = node.lp_len;
                self.dd.best_node = Some(node.clone());
            }
        }
    }
    fn maybe_squash(&mut self, depth: usize) {
        // TODO
        unimplemented!()
    }
}