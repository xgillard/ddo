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

//! This module provides the implementation of MDDs with aggressively bounded
//! maximum width. These are bounded MDDs which only develops up to max-width
//! nodes in a layer before moving on to the next one. This is different from
//! the usual approach used when developing a restricted or relaxed MDD.
//! Indeed, bounded MDDs are usually developed by expanding a layer completely
//! before to either restrict/relax the overdue nodes. The MDDs in this module
//! are different as they will not generate these overdue nodes at all.
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::Arc;

use crate::implementation::mdd::shallow::utils::{Node, Edge};
use crate::common::{PartialAssignment, Solution, FrontierNode, MDDType, Completion, Reason, VarSet, Decision, Variable};
use crate::abstraction::mdd::{MDD, Config};
use crate::abstraction::heuristics::SelectableNode;
use crate::implementation::mdd::deep::mdd::DeepMDD;
use crate::implementation::mdd::hybrid::CompositeMDD;
use std::collections::HashMap;


/// This structure implements an MDD which is aggressively bounded when 
/// developing a restricted MDD and DeepMDD 
///
/// # Example Usage
/// ```
/// # use ddo::*;
/// #
/// # #[derive(Copy, Clone)]
/// # struct MockProblem;
/// # impl Problem<usize> for MockProblem {
/// #     fn nb_vars(&self)       -> usize {  5 }
/// #     fn initial_state(&self) -> usize { 42 }
/// #     fn initial_value(&self) -> isize   { 84 }
/// #     fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
/// #         (0..=1).into()
/// #     }
/// #     fn transition(&self, state: &usize, _: &VarSet, _: Decision) -> usize {
/// #         41
/// #     }
/// #     fn transition_cost(&self, state: &usize, _: &VarSet, _: Decision) -> isize {
/// #         42
/// #     }
/// # }
/// # #[derive(Copy, Clone)]
/// # struct MockRelax;
/// # impl Relaxation<usize> for MockRelax {
/// #     fn merge_states(&self, n: &mut dyn Iterator<Item=&usize>) -> usize {
/// #         *n.next().unwrap()
/// #     }
/// #     fn relax_edge(&self, _src: &usize, _dst: &usize, _rlx: &usize, _d: Decision, cost: isize) -> isize {
/// #        cost
/// #     }
/// # }
/// let verbosity  = 2;
/// let nb_threads = 4;
/// let problem    = MockProblem;
/// let relaxation = MockRelax;
/// let config     = config_builder(&problem, relaxation).build();
/// let mdd        = AggressivelyBoundedMDD::from(config);
/// // the solver is created using an mdd.
/// let mut solver = ParallelSolver::customized(mdd, verbosity, nb_threads);
/// solver.maximize();
/// ```
#[derive(Clone)]
pub struct AggressivelyBoundedMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone {
    composite: CompositeMDD<T, C, RestrictedOnly<T,C>, DeepMDD<T, C>>
}

impl <T, C> AggressivelyBoundedMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone {
    pub fn new(c: C) -> Self {
        Self { 
        composite: CompositeMDD::new(RestrictedOnly::new(c.clone()), DeepMDD::new(c))
        }
    }
}

impl <T, C> MDD<T, C> for AggressivelyBoundedMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone {

    fn config(&self) -> &C {
        self.composite.config()
    }

    fn config_mut(&mut self) -> &mut C {
        self.composite.config_mut()
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.composite.exact(root, best_lb, ub)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.composite.restricted(root, best_lb, ub)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.composite.relaxed(root, best_lb, ub)
    }

    fn is_exact(&self) -> bool {
        self.composite.is_exact()
    }

    fn best_value(&self) -> isize {
        self.composite.best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self.composite.best_solution()
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        self.composite.for_each_cutset_node(func)
    }
}

impl <T, C> From<C> for AggressivelyBoundedMDD<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn from(c: C) -> Self {
        Self::new(c)
    }
}

/// This is nothing but a writing simplification to tell that in a flat mdd,
/// a layer is a hashmap of states to nodes
type Layer<T> = HashMap<Rc<T>, Rc<Node<T>>>;

/// This structure implements an aggressively bounded maximum width. These are
/// bounded MDDs which only develops up to max-width nodes in a layer before
/// moving on to the next one. This is different from the usual approach used
/// when developing a restricted or relaxed MDD. Indeed, bounded MDDs are
/// usually developed by expanding a layer completely before to either
/// restrict/relax the overdue nodes. The MDDs in this module are different as
/// they will not generate these overdue nodes at all.
///
/// # Warning
/// At the time being, this structure can **only** be used to derive RESTRICTED
/// MDDs. The code to develop RELAXED MDDs is almost ready... except for the
/// relaxation part
#[derive(Debug, Clone)]
struct RestrictedOnly<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone {
    /// This is the configuration used to parameterize the behavior of this
    /// MDD. Even though the internal state (free variables) of the configuration
    /// is subject to change, the configuration itself is immutable over time.
    config: C,

    // -- The following fields characterize the current unrolling of the MDD. --
    /// This is the kind of unrolling that was requested. It determines if this
    /// mdd must be an `Exact`, `Restricted` or `Relaxed` MDD.
    mddtype: MDDType,
    /// This array stores the three layers known by the mdd: the current, next
    /// and last exact layer (lel). The position of each layer in the array is
    /// determined by the `current`, `next` and `lel` fields of the structure.
    layers: [Layer<T>; 3],
    /// The index of the current layer in the array of `layers`.
    current: usize,
    /// The index of the next layer in the array of `layers`.
    next: usize,
    /// The index of the last exact layer (lel) in the array of `layers`
    lel: usize,
    /// The index of the previous layer in the array of `layers`. It may either
    /// be equal to current or lel.
    prev: usize,
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
    best_node: Option<Rc<Node<T>>>,

    /// Transient field to store references to the nodes of the current layer
    /// and be able to efficiently develop the nodes from the parent layer
    /// starting with the most relevant
    buffer: Vec<Rc<Node<T>>>
}

/// As the name suggests, `AggressivelyBoundedFlatMDD` is an implementation of
/// the `MDD` trait.
/// See the trait definition for the documentation related to these methods.
impl <T, C> MDD<T, C> for RestrictedOnly<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
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
        self.best_node.as_ref().map(|n| Solution::new(self.partial_assignment(n)))
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        if !self.is_exact {
            match self.mddtype {
                MDDType::Exact      => {/* nothing to do */},
                MDDType::Relaxed    => self.for_each_relaxed_cutset_node(func),
                MDDType::Restricted => self.for_each_restricted_cutset_node(func),
            }
        }
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Exact;
        self.max_width= usize::max_value();

        self.develop(root, free_vars, best_lb, ub)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Restricted;
        self.max_width= self.config.max_width(MDDType::Restricted, &free_vars);

        self.develop(root, free_vars, best_lb, ub)
    }

    /// Do **NOT** use for now...
    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.clear();

        let free_vars = self.config.load_variables(root);
        self.mddtype  = MDDType::Relaxed;
        self.max_width= self.config.max_width(MDDType::Relaxed, &free_vars);

        self.develop(root, free_vars, best_lb, ub)
    }
}
/// This macro wraps a tiny bit of unsafe code that lets us borrow the current
/// layer only. This macro should not be used anywhere outside the current file.
macro_rules! current_layer {
    ($dd:expr) => {
        unsafe { &*$dd.layers.as_ptr().add($dd.current) }
    };
}
impl <T, C> RestrictedOnly<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    /// Constructor, uses the given config to parameterize the mdd's behavior
    pub fn new(config: C) -> Self {
        Self {
            config,
            mddtype  : MDDType::Exact,
            layers   : [Default::default(), Default::default(), Default::default()],
            current  : 0,
            next     : 1,
            lel      : 2,
            prev     : 0,
            root_pa  : Arc::new(PartialAssignment::Empty),
            is_exact : true,
            max_width: usize::max_value(),
            best_lb  : isize::min_value(),
            best_node: None,
            buffer   : vec![]
        }
    }
    /// Resets the state of the mdd to make it reusable and ready to explore an
    /// other subproblem-space.
    pub fn clear(&mut self) {
        self.mddtype   = MDDType::Exact;
        self.current   = 0;
        self.next      = 1;
        self.lel       = 2;
        self.prev      = 0;
        self.is_exact  = true;
        //self.root_pa   = Arc::new(PartialAssignment::Empty);
        self.best_node = None;
        self.best_lb   = isize::min_value();
        self.layers.iter_mut().for_each(|l|l.clear());
        //
        self.buffer.clear();
        self.config.clear();
    }
    /// Develops/Unrolls the requested type of MDD, starting from a given root
    /// node. It only considers nodes that are relevant wrt. the given best lower
    /// bound (`best_lb`) and assigns a value to the variables of the specified
    /// VarSet (`vars`).
    fn develop(&mut self, root: &FrontierNode<T>, mut vars: VarSet, best_lb: isize, ub: isize) -> Result<Completion, Reason> {
        self.root_pa = Arc::clone(&root.path);
        let root     = Node::from(root);
        self.best_lb = best_lb;

        self.config.upon_node_insert(root.this_state.as_ref());
        self.layers[self.next].insert(Rc::clone(&root.this_state), Rc::new(root));

        let mut depth = 0;
        while let Some(var) = self.next_var(&vars) {
            // Did the cutoff kick in ?
            if self.config.must_stop(best_lb, ub) {
                return Err(Reason::CutoffOccurred);
            }

            self.add_layer();
            vars.remove(var);
            self.config.upon_new_layer(var, &mut self.layers[self.current].keys().map(|k| k.as_ref()));
            depth += 1;

            let mut next_layer_squashed = false;

            // This unsafe block really does not hurt: buffer is only a member
            // of self to avoid repeated reallocation. Otherwise, it has
            // absolutely no role to play
            let buffer = unsafe{ &mut *(&mut self.buffer as *mut Vec<Rc<Node<T>>>) };
            buffer.clear();
            current_layer!(self).values().for_each(|n| buffer.push(Rc::clone(&n)));
            buffer.sort_unstable_by(|a, b| self.config.compare(a, b).reverse());

            for node in buffer.iter() {
                // Do I need to squash the next layer (aggressively bound
                // the max-width of the MDD) ?
                if next_layer_squashed || self.must_squash(depth) {
                    next_layer_squashed = true;
                    break;
                }

                let src_state = node.this_state.as_ref();

                for val in self.config.domain_of(src_state, var) {
                    // Do I need to squash the next layer (aggressively bound
                    // the max-width of the MDD) ?
                    if next_layer_squashed || self.must_squash(depth) {
                        next_layer_squashed = true;
                        break;
                    }

                    let decision = Decision { variable: var, value: val };
                    let state    = self.config.transition(src_state, &vars, decision);
                    let weight   = self.config.transition_cost(src_state, &vars, decision);

                    self.branch(node, state, decision, weight)
                }
            }

            // force the consistency of the next layer if needed
            if next_layer_squashed {
                match self.mddtype {
                    MDDType::Exact => {},
                    MDDType::Restricted => self.remember_lel(),
                    MDDType::Relaxed    => {
                        self.remember_lel();
                        // FIXME:
                        unimplemented!("One should add an extra node standing for all the not-explored yet nodes")
                    }
                }
            }
        }

        Ok(self.finalize())
    }

    /// Returns true iff the next layer must be squashed (incomplete)
    #[inline]
    fn must_squash(&self, depth: usize) -> bool {
        match self.mddtype {
            MDDType::Exact      =>
                false,
            MDDType::Restricted =>
                depth > 1 && self.layers[self.next].len() >= self.max_width,
            MDDType::Relaxed    =>
                depth > 1 && self.layers[self.next].len() >= self.max_width - 1
        }
    }

    /// Returns the next variable to branch on (according to the configured
    /// branching heuristic) or None if all variables have been assigned a value.
    fn next_var(&self, vars: &VarSet) -> Option<Variable> {
        let mut curr_it = self.layers[self.prev].keys().map(|k| k.as_ref());
        let mut next_it = self.layers[self.next].keys().map(|k| k.as_ref());

        self.config.select_var(vars, &mut curr_it, &mut next_it)
    }
    /// Adds one layer to the mdd and move to it.
    /// In practice, this amounts to considering the 'next' layer as the current
    /// one, a clearing the next one.
    fn add_layer(&mut self) {
        self.swap_current_next();
        self.prev = self.current;
        self.layers[self.next].clear();
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
    fn branch(&mut self, node: &Rc<Node<T>>, dest: T, decision: Decision, weight: isize) {
        let dst_node = Node {
            this_state: Rc::new(dest),
            value: node.value + weight,
            estimate: isize::max_value(),
            flags: node.flags, // if its inexact, it will be or relaxed it will be considered inexact or relaxed too
            best_edge: Some(Edge {
                parent: Rc::clone(&node),
                weight,
                decision
            })
        };
        self.add_node(dst_node)
    }

    /// Inserts the given node in the next layer or updates it if needed.
    fn add_node(&mut self, mut node: Node<T>) {
        match self.layers[self.next].entry(Rc::clone(&node.this_state)) {
            Entry::Vacant(re) => {
                node.estimate = self.config.estimate(node.state());
                if node.ub() > self.best_lb {
                    self.config.upon_node_insert(node.this_state.as_ref());
                    re.insert(Rc::new(node));
                }
            },
            Entry::Occupied(mut re) => {
                Self::merge(re.get_mut(), node);
            }
        }
    }

    /// Swaps the indices of the current and next layers effectively moving
    /// to the next layer (next is now considered current)
    fn swap_current_next(&mut self) {
        let tmp      = self.current;
        self.current = self.next;
        self.next    = tmp;
    }
    /// Swaps the indices of the current and last exact layers, effectively
    /// remembering current as the last exact layer.
    fn swap_current_lel(&mut self) {
        let tmp      = self.current;
        self.current = self.lel;
        self.lel     = tmp;
    }
    /// Records the last exact layer. It only has an effect when the mdd is
    /// considered to be still correct. In other words, it will only remember
    /// the LEL the first time either `restrict_last()` or `relax_last()` is
    /// called.
    fn remember_lel(&mut self) {
        if self.is_exact {
            self.is_exact = false;
            self.swap_current_lel();
        }
    }
    /// Finalizes the computation of the MDD: it identifies the best terminal node.
    fn finalize(&mut self) -> Completion {
        self.best_node = self.layers[self.next].values()
            .max_by_key(|n| n.value)
            .cloned();

        Completion{
            is_exact     : self.is_exact(),
            best_value   : self.best_node.as_ref().map(|node| node.value),
        }
    }
    /// This method yields a partial assignment for the given node
    fn partial_assignment(&self, n: &Node<T>) -> Arc<PartialAssignment> {
        let root = &self.root_pa;
        let path = n.path();
        Arc::new(PartialAssignment::FragmentExtension {parent: Arc::clone(root), fragment: path})
    }

    /// This method ensures that a node be effectively merged with the 2nd one
    /// even though it is shielded behind a shared ref.
    fn merge(old: &mut Rc<Node<T>>, new: Node<T>) {
        if let Some(e) = Rc::get_mut(old) {
            e.merge(new)
        } else {
            let mut cpy = old.as_ref().clone();
            cpy.merge(new);
            *old = Rc::new(cpy)
        }
    }


    fn for_each_restricted_cutset_node<F>(&self, mut func: F) where F: FnMut(FrontierNode<T>) {
        self.layers[self.lel].values().for_each(|n| {
            let frontier_node = n.to_frontier_node(&self.root_pa);
            (func)(frontier_node);
        });
    }
    fn for_each_relaxed_cutset_node<F>(&self, mut func: F) where F: FnMut(FrontierNode<T>) {
        let ub = self.best_value();
        if ub > self.best_lb {
            self.layers[self.lel].values().for_each(|n| {
                let mut frontier_node = n.to_frontier_node(&self.root_pa);
                frontier_node.ub = ub.min(frontier_node.ub);
                (func)(frontier_node);
            });
        }
    }
}

impl <T, C> From<C> for RestrictedOnly<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn from(c: C) -> Self {
        Self::new(c)
    }
}

// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
mod test_restricted_only {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::mdd::{MDD, Config};
    use crate::common::{Decision, Domain, FrontierNode, MDDType, PartialAssignment, Reason, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::config_builder;
    use crate::implementation::mdd::aggressively_bounded::RestrictedOnly;
    use crate::test_utils::{MockConfig, MockCutoff, Proxy};
    use mock_it::Matcher;


    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd = RestrictedOnly::new(config);

        assert_eq!(MDDType::Exact, mdd.mddtype);
    }

    #[test]
    fn mdd_type_changes_depending_on_the_requested_type_of_mdd() {
        let root_n = FrontierNode {
            state: Arc::new(0),
            lp_len: 0,
            ub: 24,
            path: Arc::new(PartialAssignment::Empty)
        };

        let config = MockConfig::default();
        let mut mdd = RestrictedOnly::new(config);

        //assert!(mdd.relaxed(&root_n, 0).is_ok());
        //assert_eq!(MDDType::Relaxed, mdd.mddtype);

        assert!(mdd.restricted(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Restricted, mdd.mddtype);

        assert!(mdd.exact(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Exact, mdd.mddtype);
    }

    #[derive(Copy, Clone)]
    struct DummyProblem;

    impl Problem<usize> for DummyProblem {
        fn nb_vars(&self) -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize { 0 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..=2).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> isize {
            d.value
        }
    }

    #[derive(Copy, Clone)]
    struct DummyRelax;

    impl Relaxation<usize> for DummyRelax {
        fn merge_states(&self, _: &mut dyn Iterator<Item=&usize>) -> usize {
            100
        }
        fn relax_edge(&self, _: &usize, _: &usize, _: &usize, _: Decision, _: isize) -> isize {
            20
        }
        fn estimate(&self, _state: &usize) -> isize {
            50
        }
    }

    #[test]
    fn exact_no_cutoff_completion_must_be_coherent_with_outcome() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let cfg= config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd = RestrictedOnly::from(cfg);

        let root   = mdd.config().root_node();
        let result = mdd.exact(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    #[test]
    fn restricted_no_cutoff_completion_must_be_coherent_with_outcome_() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let cfg= config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd = RestrictedOnly::from(cfg);

        let root   = mdd.config().root_node();
        let result = mdd.restricted(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    /*  !!! Restricted only !!!
    #[test]
    fn relaxed_no_cutoff_completion_must_be_coherent_with_outcome() {
        let pb = DummyProblem;
        let rlx= DummyRelax;
        let mut mdd = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();

        let root   = mdd.config().root_node();
        let result = mdd.relaxed(&root, 0);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    */

    #[test]
    fn exact_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let cfg     = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build();
        let mut mdd = RestrictedOnly::from(cfg);

        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.exact(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    #[test]
    fn restricted_fails_with_cutoff_when_cutoff_occurs() {
        let pb         = DummyProblem;
        let rlx         = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let cfg         = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build();
        let mut mdd = RestrictedOnly::from(cfg);

        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.restricted(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }

    /* !!! Restricted only !!!
    #[test]
    fn relaxed_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let mut mdd = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build();

        cutoff.must_stop.given(()).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.relaxed(&root, 0);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
        assert!(verify(cutoff.must_stop.was_called_with(())));
    }
    */


    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd = RestrictedOnly::from(cfg);

        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 2 },
                       Decision { variable: Variable(1), value: 2 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn restricted_drops_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd = RestrictedOnly::from(cfg);

        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 2);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 0 },
                       Decision { variable: Variable(1), value: 0 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    /* !!! restricted only !!!
    #[test]
    fn relaxed_merges_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();

        let root = mdd.config().root_node();
        assert!(mdd.relaxed(&root, 0).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 42);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 2 },
                       Decision { variable: Variable(1), value: 2 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();

        let root = mdd.config().root_node();
        assert!(mdd.relaxed(&root, 0).is_ok());

        let mut cutset = vec![];
        mdd.for_each_cutset_node(|n| cutset.push(n));
        assert_eq!(cutset.len(), 3); // L1 was not squashed even though it was 3 wide
    }
    */

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd = RestrictedOnly::from(cfg);

        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    /* !!! restricted only !!!
    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = config_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 0).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occurred() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = config_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 0).is_ok());
        assert_eq!(false, mdd.is_exact())
    }
    */

    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let mut mdd = RestrictedOnly::from(cfg);
        let root = mdd.config().root_node();
        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occurred() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let mut mdd = RestrictedOnly::from(cfg);

        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert_eq!(false, mdd.is_exact())
    }

    #[derive(Clone, Copy)]
    struct DummyInfeasibleProblem;

    impl Problem<usize> for DummyInfeasibleProblem {
        fn nb_vars(&self) -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize { 0 }
        #[allow(clippy::reversed_empty_ranges)]
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..0).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> isize {
            d.value
        }
    }

    #[test]
    fn when_the_problem_is_infeasible_there_is_no_solution() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx).build();
        let mut mdd = RestrictedOnly::from(cfg);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx).build();
        let mut mdd = RestrictedOnly::from(cfg);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert_eq!(isize::min_value(), mdd.best_value())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx).build();
        let mut mdd = RestrictedOnly::from(cfg);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    /* !!! restricted only !!! 
    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut mdd = config_builder(&pb, rlx).build();
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 100).is_ok());
        assert!(mdd.best_solution().is_none())
    }
    */

    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let cfg = config_builder(&pb, rlx).build();
        let mut mdd = RestrictedOnly::from(cfg);

        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }
}


#[cfg(test)]
mod test_aggressively_bounded_width {
    use std::sync::Arc;

    use crate::abstraction::dp::{Problem, Relaxation};
    use crate::abstraction::mdd::{MDD, Config};
    use crate::common::{Decision, Domain, FrontierNode, MDDType, PartialAssignment, Reason, Variable, VarSet};
    use crate::implementation::heuristics::FixedWidth;
    use crate::implementation::mdd::config::mdd_builder;
    use crate::test_utils::{MockConfig, MockCutoff, Proxy};
    use crate::implementation::mdd::aggressively_bounded::AggressivelyBoundedMDD;
    use mock_it::Matcher;
    use crate::{VariableHeuristic, NaturalOrder};

    type DD<T, C> = AggressivelyBoundedMDD<T, C>;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let config = MockConfig::default();
        let mdd = DD::new(config);

        assert_eq!(MDDType::Exact, mdd.composite.mddtype);
    }

    #[test]
    fn mdd_type_changes_depending_on_the_requested_type_of_mdd() {
        let root_n = FrontierNode {
            state: Arc::new(0),
            lp_len: 0,
            ub: 24,
            path: Arc::new(PartialAssignment::Empty)
        };

        let config = MockConfig::default();
        let mut mdd = DD::new(config);

        assert!(mdd.relaxed(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Relaxed, mdd.composite.mddtype);

        assert!(mdd.restricted(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Restricted, mdd.composite.mddtype);

        assert!(mdd.exact(&root_n, 0, 1000).is_ok());
        assert_eq!(MDDType::Exact, mdd.composite.mddtype);
    }

    #[derive(Copy, Clone)]
    struct DummyProblem;

    impl Problem<usize> for DummyProblem {
        fn nb_vars(&self) -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize { 0 }
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..=2).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> isize {
            d.value
        }
    }

    #[derive(Copy, Clone)]
    struct DummyRelax;

    impl Relaxation<usize> for DummyRelax {
        fn merge_states(&self, _: &mut dyn Iterator<Item=&usize>) -> usize {
            100
        }
        fn relax_edge(&self, _: &usize, _: &usize, _: &usize, _: Decision, _: isize) -> isize {
            20
        }
        fn estimate(&self, _state: &usize) -> isize {
            50
        }
    }

    #[derive(Copy, Clone, Default)]
    struct DummyIncrementalVarHeu {
        cleared : usize,
        inserted: usize,
        layers  : usize,
    }
    impl VariableHeuristic<usize> for DummyIncrementalVarHeu {
        fn next_var(&self, free_vars: &VarSet, current_layer: &mut dyn Iterator<Item=&usize>, next_layer: &mut dyn Iterator<Item=&usize>) -> Option<Variable> {
            NaturalOrder.next_var(free_vars, current_layer, next_layer)
        }

        fn upon_new_layer(&mut self, _var: Variable, _current_layer: &mut dyn Iterator<Item=&usize>) {
            self.layers += 1;
        }

        fn upon_node_insert(&mut self, _state: &usize) {
            self.inserted += 1;
        }

        fn clear(&mut self) {
            self.cleared += 1;
        }
    }

    #[test]
    fn exact_no_cutoff_completion_must_be_coherent_with_outcome() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();

        let mut mdd  = DD::from(config);
        let root     = mdd.config().root_node();
        let result   = mdd.exact(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    #[test]
    fn restricted_no_cutoff_completion_must_be_coherent_with_outcome() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();

        let mut mdd  = DD::from(config);
        let root     = mdd.config().root_node();
        let result   = mdd.restricted(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    #[test]
    fn relaxed_no_cutoff_completion_must_be_coherent_with_outcome() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();

        let mut mdd  = DD::from(config);
        let root     = mdd.config().root_node();
        let result   = mdd.relaxed(&root, 0, 1000);
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact  , mdd.is_exact());
        assert_eq!(completion.best_value, Some(mdd.best_value()));
    }
    #[test]
    fn exact_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let config  = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build();
        let mut mdd = DD::from(config);

        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.exact(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    #[test]
    fn restricted_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let config  = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build();
        let mut mdd = DD::from(config);


        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.restricted(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    #[test]
    fn relaxed_fails_with_cutoff_when_cutoff_occurs() {
        let pb      = DummyProblem;
        let rlx     = DummyRelax;
        let mut cutoff  = MockCutoff::default();
        let config  = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .with_cutoff(Proxy::new(&mut cutoff))
            .build();
        let mut mdd = DD::from(config);

        cutoff.must_stop.given(Matcher::Any).will_return(true);

        let root   = mdd.config().root_node();
        let result = mdd.relaxed(&root, 0, 1000);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }

    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 6);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 2 },
                       Decision { variable: Variable(1), value: 2 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn restricted_drops_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 2);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 0 },
                       Decision { variable: Variable(1), value: 0 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn relaxed_merges_the_less_interesting_nodes() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        assert!(mdd.relaxed(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), 42);
        assert_eq!(mdd.best_solution().unwrap().iter().collect::<Vec<Decision>>(),
                   vec![
                       Decision { variable: Variable(2), value: 2 },
                       Decision { variable: Variable(1), value: 2 },
                       Decision { variable: Variable(0), value: 2 },
                   ]
        );
    }

    #[test]
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        assert!(mdd.relaxed(&root, 0, 1000).is_ok());

        let mut cutset = vec![];
        mdd.for_each_cutset_node(|n| cutset.push(n));
        assert_eq!(cutset.len(), 3); // L1 was not squashed even though it was 3 wide
    }

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx)
            .with_max_width(FixedWidth(1))
            .build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        assert!(mdd.relaxed(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occurred() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        assert!(mdd.relaxed(&root, 0, 1000).is_ok());
        assert_eq!(false, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(10)).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();
        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert_eq!(true, mdd.is_exact())
    }

    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occurred() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).with_max_width(FixedWidth(1)).build();
        let mut mdd = DD::from(config);
        let root = mdd.config().root_node();
        assert!(mdd.restricted(&root, 0, 1000).is_ok());
        assert_eq!(false, mdd.is_exact())
    }

    #[derive(Clone, Copy)]
    struct DummyInfeasibleProblem;

    impl Problem<usize> for DummyInfeasibleProblem {
        fn nb_vars(&self) -> usize { 3 }
        fn initial_state(&self) -> usize { 0 }
        fn initial_value(&self) -> isize { 0 }
        #[allow(clippy::reversed_empty_ranges)]
        fn domain_of<'a>(&self, _: &'a usize, _: Variable) -> Domain<'a> {
            (0..0).into()
        }
        fn transition(&self, state: &usize, _: &VarSet, d: Decision) -> usize {
            *state + d.value as usize
        }
        fn transition_cost(&self, _: &usize, _: &VarSet, d: Decision) -> isize {
            d.value
        }
    }

    #[test]
    fn when_the_problem_is_infeasible_there_is_no_solution() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn when_the_problem_is_infeasible_the_best_value_is_min_infinity() {
        let pb = DummyInfeasibleProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 0, 1000).is_ok());
        assert_eq!(isize::min_value(), mdd.best_value())
    }

    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.exact(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.relaxed(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let config = mdd_builder(&pb, rlx).build();
        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 100, 1000).is_ok());
        assert!(mdd.best_solution().is_none())
    }



    #[test]
    fn config_is_cleared_before_developing_any_mddtype() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut heu = DummyIncrementalVarHeu::default();

        let config = mdd_builder(&pb, rlx)
            .with_branch_heuristic(Proxy::new(&mut heu))
            .build();

        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 100, 1000).is_ok());
        assert_eq!(1, heu.cleared);

        assert!(mdd.relaxed(&root, 100, 1000).is_ok());
        assert_eq!(2, heu.cleared);

        assert!(mdd.exact(&root, 100, 1000).is_ok());
        assert_eq!(3, heu.cleared);
    }

    #[test]
    fn upon_layer_is_called_whenever_a_new_layer_is_created() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut heu = DummyIncrementalVarHeu::default();

        let config = mdd_builder(&pb, rlx)
            .with_branch_heuristic(Proxy::new(&mut heu))
            .build();

        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        assert!(mdd.restricted(&root, 100, 1000).is_ok());
        assert_eq!(3, heu.layers);

        assert!(mdd.relaxed(&root, 100, 1000).is_ok());
        assert_eq!(6, heu.layers);

        assert!(mdd.exact(&root, 100, 1000).is_ok());
        assert_eq!(9, heu.layers);
    }

    #[test]
    fn upon_insert_is_called_whenever_a_non_existing_node_is_added_to_next_layer() {
        let pb = DummyProblem;
        let rlx = DummyRelax;
        let mut heu = DummyIncrementalVarHeu::default();

        let config = mdd_builder(&pb, rlx)
            .with_branch_heuristic(Proxy::new(&mut heu))
            .build();

        let mut mdd  = DD::from(config);
        let root = mdd.config().root_node();

        // Exact mdd comprises 16 nodes (includes the root)
        assert!(mdd.exact(&root, -100, 1000).is_ok());
        assert_eq!(16, heu.inserted);

        // because of the max witdh, restricted comprises 10 nodes (incl root)
        // after the restriction. Because this mdd is aggressively bounded,
        // only these 10 nodes are added.
        assert!(mdd.restricted(&root, -100, 1000).is_ok());
        assert_eq!(26, heu.inserted);

        // in spite of the max width, the relaxed performs 16 insertions (incl
        // root) just like the exact did. This is because nodes are inserted
        // then merged.
        assert!(mdd.relaxed(&root, -100, 1000).is_ok());
        assert_eq!(42, heu.inserted);
    }
}


