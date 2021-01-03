
// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! This module provides a custom pooled mdd implementation tailored for the
//! needs of a MISP solver.

use std::collections::hash_map::Entry;
use std::sync::Arc;

use std::rc::Rc;
use std::collections::HashMap;
use ddo::{Config, MDDType, PartialAssignment, MDD, Solution, FrontierNode, Completion, Reason, VarSet, Variable, Decision, NodeFlags, SelectableNode};
use ddo::PartialAssignment::FragmentExtension;
use bitset_fixed::BitSet;
use crate::heuristics::MispVarHeu2;


/// This tiny structure represents an internal node from the flat MDD.
/// Basically, it only remembers its own state, value, the best path from root
/// and possibly the description of the 'best' edge connecting it to one of its
/// parents.
#[derive(Debug, Clone)]
pub struct Node {
    pub this_state   : Rc<BitSet>,
    pub value        : isize,
    pub estimate     : isize,
    pub flags        : NodeFlags,
    pub best_edge    : Option<Edge>
}
/// This is the description of an edge. It only knows the state from the parent
/// node, the decision that caused the transition from one state to the other
/// and the weight of taking that transition.
#[derive(Debug, Clone)]
pub struct Edge {
    pub parent   : Rc<Node>,
    pub weight   : isize,
    pub decision : Decision,
}
/// If a layer grows too large, the branch and bound algorithm might need to
/// squash the current layer and thus to select some Nodes for merge or removal.
impl SelectableNode<BitSet> for Node {
    /// Returns a reference to the state of this node
    fn state(&self) -> &BitSet {
        self.this_state.as_ref()
    }
    /// Returns the value of the objective function at this node.
    fn value(&self) -> isize {
        self.value
    }
    /// Returns true iff the node is an exact node.
    fn is_exact(&self) -> bool {
        self.flags.is_exact()
    }
}

impl Node {
    /// Returns true iff there exists a best path from root to this node.
    pub fn has_exact_best(&self) -> bool {
        !self.flags.is_relaxed()
    }
    /// Sets the 'exact' flag of this node to the given value.
    /// This is useful to ensure that a node becomes inexact when it used to be
    /// exact but an other path passing through a inexact node produces the
    /// same state.
    pub fn set_exact(&mut self, exact: bool) {
        self.flags.set_exact(exact)
    }
    /// Merge other into this node. That is to say, it combines the information
    /// from two nodes that are logically equivalent (should be the same).
    /// Concretely, it means that it possibly updates the current node to keep
    /// the best _longest path info_, track the 'exactitude' of the node and
    /// keep the tightest upper bound.
    ///
    /// # Important note
    /// *This has nothing to do with the user-provided `merge_*` operators !*
    pub fn merge(&mut self, other: Node) {
        let exact = self.is_exact() && other.is_exact();

        if other.value > self.value {
            self.value = other.value;
            self.flags = other.flags;
            self.best_edge = other.best_edge;
        }
        self.estimate = self.estimate.min(other.estimate);
        self.set_exact(exact);
    }
    /// Returns the path to this node
    pub fn path(&self) -> Vec<Decision> {
        let mut edge = &self.best_edge;
        let mut path = vec![];
        while let Some(e) = edge {
            path.push(e.decision);
            edge = &e.parent.best_edge;
        }
        path
    }
    /// Returns the nodes upper bound
    pub fn ub(&self) -> isize {
        self.value.saturating_add(self.estimate)
    }
}
impl Node {
    pub fn to_frontier_node(&self, root: &Arc<PartialAssignment>) -> FrontierNode<BitSet> {
        FrontierNode {
            state : Arc::new(self.this_state.as_ref().clone()),
            path  : Arc::new(FragmentExtension {parent: Arc::clone(root), fragment: self.path()}),
            lp_len: self.value,
            ub    : self.ub()
        }
    }
}
/// Because the solver works with `FrontierNode`s but the MDD uses `Node`s, we
/// need a way to convert from one type to the other. This conversion ensures
/// that a `Node` can be built from a `FrontierNode`.
impl From<&FrontierNode<BitSet>> for Node {
    fn from(n: &FrontierNode<BitSet>) -> Self {
        Node {
            this_state: Rc::new(n.state.as_ref().clone()),
            value     : n.lp_len,
            estimate  : n.ub - n.lp_len,
            flags     : NodeFlags::new_exact(),
            best_edge : None
        }
    }
}

#[derive(Debug, Clone)]
pub struct MispMDD<C>
    where C: Config<BitSet> + Clone
{
    /// This is the configuration used to parameterize the behavior of this
    /// MDD. Even though the internal state (free variables) of the configuration
    /// is subject to change, the configuration itself is immutable over time.
    config: C,

    var_heu: MispVarHeu2,

    // -- The following fields characterize the current unrolling of the MDD. --
    /// This is the kind of unrolling that was requested. It determines if this
    /// mdd must be an `Exact`, `Restricted` or `Relaxed` MDD.
    mddtype: MDDType,
    /// This is the pool of candidate nodes that might possibly participate in
    /// the next layer.
    pool: HashMap<Rc<BitSet>, Rc<Node>>,
    /// This set of nodes comprises all nodes that belong to a
    /// _frontier cutset_ (FC).
    cutset: Vec<Rc<Node>>,

    /// A flag indicating whether this mdd is exact
    is_exact: bool,

    /// This is the path in the exact mdd (partial assignment) until the root of
    /// this mdd.
    root_pa: Arc<PartialAssignment>,
    /// This is the best known lower bound at the time of the MDD unrolling.
    /// This field is set once before developing mdd.
    best_lb: isize,
    /// This is the maximum width allowed for a layer of the MDD. It is determined
    /// once at the beginning of the MDD derivation.
    max_width: usize,
    /// This field memoizes the best node of the MDD. That is, the node of this
    /// mdd having the longest path from root.
    best_node: Option<Rc<Node>>
}
/// As the name suggests, `MispMDD` is an implementation of the `MDD` trait.
/// See the trait definiton for the documentation related to these methods.
impl <C> MDD<BitSet, C> for MispMDD<C>
    where C: Config<BitSet> + Clone
{
    fn config(&self) -> &C {
        &self.config
    }
    fn config_mut(&mut self) -> &mut C {
        &mut self.config
    }

    fn is_exact(&self) -> bool {
        self.is_exact
            || self.mddtype == MDDType::Relaxed && self.best_node.as_ref().map(|n| n.has_exact_best()).unwrap_or(true)
    }

    fn best_value(&self) -> isize {
        if let Some(node) = &self.best_node {
            node.value
        } else {
            isize::min_value()
        }
    }

    fn best_solution(&self) -> Option<Solution> {
        self.best_node.as_ref().map(|n| Solution::new(self.partial_assignement(n)))
    }

    fn for_each_cutset_node<F>(&self, mut func: F) where F: FnMut(FrontierNode<BitSet>) {
        if !self.is_exact {
            let ub = self.best_value();
            if ub > self.best_lb {
                self.cutset.iter().for_each(|n| {
                    let mut frontier_node = n.to_frontier_node(&self.root_pa);
                    frontier_node.ub = ub.min(frontier_node.ub);
                    (func)(frontier_node);
                });
            }
        }
    }

    fn exact(&mut self, root: &FrontierNode<BitSet>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Exact;
        self.max_width= usize::max_value();

        self.develop(root, free_vars, best_lb, ub)
    }

    fn restricted(&mut self, root: &FrontierNode<BitSet>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Restricted;
        self.max_width= self.config.max_width(&free_vars);

        self.develop(root, free_vars, best_lb, ub)
    }

    fn relaxed(&mut self, root: &FrontierNode<BitSet>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Relaxed;
        self.max_width= self.config.max_width(&free_vars);

        self.develop(root, free_vars, best_lb, ub)
    }
}
impl <C> MispMDD<C>
    where C: Config<BitSet> + Clone
{
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(config: C, n: usize) -> Self {
        MispMDD {
            config,
            var_heu  : MispVarHeu2::new(n),
            mddtype  : MDDType::Exact,
            pool     : Default::default(),
            cutset   : vec![],
            is_exact : true,
            root_pa  : Arc::new(PartialAssignment::Empty),
            best_lb  : isize::min_value(),
            max_width: usize::max_value(),
            best_node: None
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    pub fn clear(&mut self) {
        self.mddtype   = MDDType::Exact;
        self.var_heu.clear();
        self.pool.clear();
        self.cutset.clear();
        self.is_exact  = true;
        //self.root_pa   = Arc::new(PartialAssignment::Empty);
        self.best_node = None;
        self.best_lb   = isize::min_value();
    }

    /// Develops/Unrolls the requested type of MDD, starting from a given root
    /// node. It only considers nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`) and assigns a value to the variables of the specified
    /// VarSet (`vars`).
    fn develop(&mut self, root: &FrontierNode<BitSet>, mut vars: VarSet, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.root_pa = Arc::clone(&root.path);
        let root     = Node::from(root);
        self.best_lb = best_lb;

        self.var_heu.upon_insert(&root.this_state);
        self.pool.insert(Rc::clone(&root.this_state), Rc::new(root));


        let mut current = vec![];
        while let Some(var) = self.next_var() {
            self.var_heu.upon_branch(var);

            // Did the cutoff kick in ?
            if self.config.must_stop(best_lb, ub) {
                return Err(Reason::CutoffOccurred);
            }

            self.add_layer(var, &mut current);
            vars.remove(var);

            // squash layer if needed
            match self.mddtype {
                MDDType::Exact => {},
                MDDType::Restricted =>
                    if current.len() > self.max_width {
                        self.restrict_last(&mut current);
                    },
                MDDType::Relaxed =>
                    if current.len() > self.max_width {
                        self.relax_last(&mut current);
                    }
            }

            for node in current.iter() {
                let src_state = node.this_state.as_ref();
                for val in self.config.domain_of(src_state, var) {
                    let decision = Decision { variable: var, value: val };
                    let state    = self.config.transition(src_state, &vars, decision);
                    let weight   = self.config.transition_cost(src_state, &vars, decision);

                    self.branch(node, state, decision, weight)
                }
            }
        }

        Ok(self.finalize())
    }
    /// Returns the next variable to branch on (according to the configured
    /// branching heuristic) or None if all variables have been assigned a value.
    fn next_var(&self) -> Option<Variable> {
        self.var_heu.next_var()
    }
    /// Adds one layer to the mdd and move to it.
    /// In practice, this amounts to selecting the relevant nodes from the pool,
    /// adding them to the current layer and removing them from the pool.
    fn add_layer(&mut self, var: Variable, current: &mut Vec<Rc<Node>>) {
        current.clear();

        // Add all selected nodes to the next layer
        for (s, n) in self.pool.iter() {
            if self.config.impacted_by(s, var) {
                self.var_heu.upon_select(s);
                current.push(Rc::clone(n));
            }
        }

        // Remove all nodes that belong to the current layer from the pool
        for node in current.iter() {
            self.pool.remove(&node.this_state);
        }
    }
    /// This method records the branching from the given `node` with the given
    /// `decision`. It creates a fresh node for the `dest` state (or reuses one
    /// if `dest` already belongs to the current layer) and draws an edge of
    /// of the given `weight` between `orig_id` and the new node.
    ///
    /// ### Note:
    /// In case where this branching would create a new longest path to an
    /// already existing node, the length and best parent of the pre-existing
    /// node are updated.
    fn branch(&mut self, node: &Rc<Node>, dest: BitSet, decision: Decision, weight: isize) {
        let dst_node = Node {
            this_state: Rc::new(dest),
            value: node.value + weight,
            estimate: isize::max_value(),
            flags: node.flags, // if its inexact, it will be or relaxed it will be considered inexact or relaxed too
            best_edge: Some(Edge {
                parent: Rc::clone(node),
                weight,
                decision
            })
        };
        self.add_node(dst_node)
    }

    /// Inserts the given node in the next layer or updates it if needed.
    fn add_node(&mut self, mut node: Node) {
        match self.pool.entry(Rc::clone(&node.this_state)) {
            Entry::Vacant(re) => {
                node.estimate = self.config.estimate(node.state());
                if node.ub() > self.best_lb {
                    self.var_heu.upon_insert(&node.this_state);
                    re.insert(Rc::new(node));
                }
            },
            Entry::Occupied(mut re) => {
                let old = re.get_mut();
                if old.is_exact() && !node.is_exact() {
                    self.cutset.push(Rc::clone(old));
                } else if node.is_exact() && !old.is_exact() {
                    self.cutset.push(Rc::new(node.clone()));
                }
                Self::merge(old, node);
            }
        }
    }

    /// This method restricts the last layer (the current layer !) to make sure
    /// it fits within the maximum "width" size.
    ///
    /// # Note
    /// The removed nodes and all their inbound edges are irremediably lost.
    ///
    /// # Warning
    /// It is your responsibility to make sure the layer is broad enough to be
    /// relaxed. Failing to do so will not panic... but in the future, it might!
    fn restrict_last(&mut self, current: &mut Vec<Rc<Node>>) {
        self.is_exact = false;

        current.sort_unstable_by(|a, b| self.config.compare(a.as_ref(), b.as_ref()).reverse());
        current.truncate(self.max_width);
    }
    /// This method relaxes the last layer (the current layer !) to make sure
    /// it fits within the maximum "width" size. The overdue nodes are merged
    /// according to the configured strategy.
    ///
    /// # Note
    /// The removed nodes and all their inbound edges are irremediably lost.
    ///
    /// # Warning
    /// It is your responsibility to make sure the layer is broad enough to be
    /// relaxed. Failing to do so would result in a panic!
    ///
    /// # Warning
    /// This function will panic if you request a relaxation that would leave
    /// zero nodes in the current layer.
    fn relax_last(&mut self, current: &mut Vec<Rc<Node>>) {
        self.is_exact = false;

        current.sort_unstable_by(|a, b| self.config.compare(a.as_ref(), b.as_ref()).reverse());

        let (_keep, squash) = current.split_at_mut(self.max_width - 1);
        for node in squash.iter() {
            if node.is_exact() {
                self.cutset.push(Rc::clone(node));
            }
        }

        let merged_state = self.config.merge_states(&mut squash.iter().map(|n| n.state()));
        let mut merged_value = isize::min_value();
        let mut merged_edge  = None;

        for node in squash {
            let best_edge    = node.best_edge.as_ref().unwrap();
            let parent_value = node.value - best_edge.weight;
            let src          = best_edge.parent.this_state.as_ref();
            let dst          = node.this_state.as_ref();
            let decision     = best_edge.decision;
            let cost         = best_edge.weight;
            let relax_cost   = self.config.relax_edge(src, dst, &merged_state, decision, cost);

            if parent_value + relax_cost > merged_value {
                merged_value = parent_value + relax_cost;
                merged_edge  = Some(Edge{
                    parent  : Rc::clone(&best_edge.parent),
                    weight  : relax_cost,
                    decision
                });
            }
        }

        let rlx_node = Node {
            this_state: Rc::new(merged_state),
            value     : merged_value,
            estimate: isize::max_value(),
            flags     : NodeFlags::new_relaxed(),
            best_edge : merged_edge
        };

        current.truncate(self.max_width - 1);
        self.add_relaxed(rlx_node, current)
    }
    /// Finds another node in the current layer having the same state as the
    /// given `state` parameter (if there is one such node).
    fn find_same_state<'b>(state: &BitSet, current: &'b mut[Rc<Node>]) -> Option<&'b mut Rc<Node>> {
        for n in current.iter_mut() {
            if n.state().eq(state) {
                return Some(n);
            }
        }
        None
    }
    /// Adds a relaxed node into the given current layer.
    fn add_relaxed(&mut self, mut node: Node, into_current: &mut Vec<Rc<Node>>) {
        if let Some(old) = Self::find_same_state(node.state(), into_current) {
            if old.is_exact() {
                //trace!("squash:: there existed an equivalent");
                self.cutset.push(Rc::clone(old));
            }
            Self::merge(old, node);
        } else {
            node.estimate = self.config.estimate(node.state());
            if node.ub() > self.best_lb {
                into_current.push(Rc::new(node));
            }
        }
    }
    /// Finalizes the computation of the MDD: it identifies the best terminal node.
    fn finalize(&mut self) -> Completion {
        self.best_node = self.pool.values()
            .max_by_key(|n| n.value)
            .cloned();

        Completion {
            is_exact   : self.is_exact(),
            best_value : self.best_node.as_ref().map(|n| n.value)
        }
    }

    /// This method yields a partial assignment for the given node
    fn partial_assignement(&self, n: &Node) -> Arc<PartialAssignment> {
        let root = &self.root_pa;
        let path = n.path();
        Arc::new(PartialAssignment::FragmentExtension {parent: Arc::clone(root), fragment: path})
    }

    /// This method ensures that a node be effectively merged with the 2nd one
    /// even though it is shielded behind a shared ref.
    fn merge(old: &mut Rc<Node>, new: Node) {
        if let Some(e) = Rc::get_mut(old) {
            e.merge(new)
        } else {
            let mut cpy = old.as_ref().clone();
            cpy.merge(new);
            *old = Rc::new(cpy)
        }
    }
}

impl <C> From<C> for MispMDD<C> where C: Config<BitSet> + Clone {
    fn from(c: C) -> Self {
        let root   = c.root_node();
        let nbvars = c.load_variables(&root).len();
        Self::new(c, nbvars)
    }
}