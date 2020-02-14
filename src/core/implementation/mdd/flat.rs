use std::cmp::min;
use std::hash::Hash;

use metrohash::MetroHashMap;

use crate::core::abstraction::mdd::{MDD, MDDType, Layer};
use crate::core::abstraction::mdd::MDDType::{Exact, Relaxed, Restricted};
use crate::core::common::{Decision, Node, NodeInfo, Variable, Bounds};
use crate::core::implementation::mdd::config::Config;
use std::sync::Arc;

// --- MDD Data Structure -----------------------------------------------------
pub struct FlatMDD<T, C> where T: Hash + Eq + Clone, C: Config<T> {
    config           : C,

    mddtype          : MDDType,
    layers           : [MetroHashMap<T, NodeInfo>; 3],
    current          : usize,
    next             : usize,
    lel              : usize,

    is_exact         : bool,
    best_node        : Option<NodeInfo>
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

impl <T, C> MDD<T> for FlatMDD<T, C> where T: Hash + Eq + Clone, C: Config<T> {
    fn mdd_type(&self) -> MDDType {
        self.mddtype
    }
    fn root(&self) -> Node<T> {
        self.config.root_node()
    }
    fn exact(&mut self, root: &Node<T>, best_lb: i32) {
        self.develop(Exact, root, best_lb);
    }
    fn restricted(&mut self, root: &Node<T>, best_lb: i32) {
        self.develop(Restricted, root, best_lb);
    }
    fn relaxed(&mut self, root: &Node<T>, best_lb: i32) {
        self.develop(Relaxed, root, best_lb);
    }

    fn for_each_cutset_node<F>(&mut self, mut f: F) where F: FnMut(&T, &mut NodeInfo) {
        layer![self, mut lel].iter_mut().for_each(|(k, v)| (f)(k, v))
    }
    fn consume_cutset<F>(&mut self, mut f: F) where F: FnMut(T, NodeInfo) {
        layer![self, mut lel].drain().for_each(|(k, v)| (f)(k, v))
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
    fn best_node(&self) -> &Option<NodeInfo> {
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
impl <T, C> FlatMDD<T, C> where T: Hash + Eq + Clone, C: Config<T> {

    pub fn new(config: C) -> Self {
        FlatMDD {
            config,

            mddtype          : Exact,
            current          : 0,
            next             : 1,
            lel              : 2,

            is_exact         : true,
            best_node        : None,
            layers           : [Default::default(), Default::default(), Default::default()]
        }
    }

    fn clear(&mut self) {
        self.mddtype       = Exact;
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
    fn is_relevant(&self, bounds: Bounds, state: &T, info: &NodeInfo) -> bool {
        min(self.config.estimate_ub(state, info), bounds.ub) > bounds.lb
    }
    fn develop(&mut self, kind: MDDType, root: &Node<T>, best_lb: i32) {
        self.init(kind, root);
        let w = if self.mddtype == Exact { usize::max_value() } else { self.config.max_width() };

        let bounds = Bounds {lb: best_lb, ub: root.info.ub};
        let nbvars = self.config.nb_free_vars();

        let mut i  = 0;
        while i < nbvars && !layer![self, current].is_empty() {
            let var = self.config.select_var(self.it_current(), self.it_next());
            if var.is_none() { break; }

            let was_exact = self.is_exact;
            let var = var.unwrap();
            self.config.remove_var(var);
            self.unroll_layer(var, bounds);
            self.maybe_squash(i, w); // next
            self.move_to_next(was_exact);

            i += 1;
        }

        self.finalize()
    }
    fn unroll_layer(&mut self, var: Variable, bounds: Bounds) {
        let curr = layer![self,  current];
        let next = layer![self, mut next];

        for (state, info) in curr.iter() {
            let info   = Arc::new(info.clone());
            let domain = self.config.domain_of(state, var);
            for value in domain {
                let decision  = Decision{variable: var, value};
                let branching = self.config.branch(&state, Arc::clone(&info), decision);

                if let Some(old) = next.get_mut(&branching.state) {
                    old.merge(branching.info);
                } else if self.is_relevant(bounds, &branching.state, &branching.info) {
                    next.insert(branching.state, branching.info);
                }
            }
        }
    }

    fn move_to_next(&mut self, was_exact: bool) {
        if self.is_exact != was_exact {
            self.swap_current_lel();
        }
        self.swap_current_next();
        layer![self, mut next].clear();
    }

    fn init(&mut self, kind: MDDType, root: &Node<T>) {
        self.clear();
        self.config.load_vars(root);
        self.mddtype = kind;

        layer![self, mut current].insert(root.state.clone(), root.info.clone());
    }
    fn finalize(&mut self) {
        self.find_best_node();

        // We are done, we should assign a rough upper bound on all nodes from the exact cutset
        if let Some(best) = &self.best_node {
            let lp_length = best.lp_len;

            for (state, info) in layer![self, mut lel].iter_mut() {
                info.ub = lp_length.min(self.config.estimate_ub(state, info));
            }
        } else {
            // If no best node is found, it means this problem is unsat.
            // Hence, there is no relevant cutset to return
            layer![self, mut lel].clear();
        }
    }
    fn find_best_node(&mut self) {
        let mut best_value = i32::min_value();
        for info in layer![self, current].values() {
            if info.lp_len > best_value {
                best_value         = info.lp_len;
                self.best_node  = Some(info.clone());
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
            let config = &self.config;
            let next   = layer![self, mut next];

            let mut nodes = vec![];
            next.drain().for_each(|(k,v)| nodes.push(Node{state: k, info: v}));

            while nodes.len() > w {
                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;
                nodes.sort_unstable_by(|a, b| config.compare(a, b).reverse());
                nodes.truncate(w);
            }

            nodes.drain(..).for_each(|n| {next.insert(n.state, n.info);});
        }
    }
    fn maybe_relax(&mut self, i: usize, w: usize) {
        /* you cannot compress the 1st layer: you wouldn't get an useful cutset  */
        if i > 1 {
            let config = &self.config;
            let next   = layer![self, mut next];

            if next.len() > w {

                let mut nodes = vec![];
                nodes.reserve_exact(next.len());

                next.drain().for_each(|(k,v)| {
                    nodes.push(Node{state: k, info: v});
                });

                nodes.sort_unstable_by(|a, b| config.compare(a, b).reverse());

                let (keep, squash) = nodes.split_at(w-1);

                // we do squash the current layer so the mdd is now inexact
                self.is_exact = false;

                // actually squash the layer
                let merged = self.config.merge_nodes(squash);

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

    fn it_current(&self) -> Layer<'_, T> {
        Layer::Mapped(layer![self, current].iter())
    }
    fn it_next(&self) -> Layer<'_, T> {
        Layer::Mapped(layer![self, next].iter())
    }
}