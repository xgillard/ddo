//! This module defines the basic abstractions one will manipulate when
//! implementing an MDD optimization-solver for some problem formulated as a
//! dynamic program.
//!
//! The most important abstractions that should be provided by a client are
//! `Problem` and `Relaxation`.
use std::i32;

use crate::core::abstraction::mdd::{MDD, Node, Arc};
use crate::core::common::{Decision, Variable, VarSet};
use std::rc::Rc;

/// This is the main abstraction that should be provided by any user of our
/// library. Indeed, it defines the problem to be solved in the form of a dynamic
/// program. Therefore, this trait closely sticks to the formal definition of a
/// dynamic program.
///
/// The type parameter `<T>` denotes the type of the states of the dynamic program.
pub trait Problem<T> where T: Eq + Clone {
    /// Returns the number of decision variables that play a role in the problem.
    fn nb_vars(&self) -> usize;
    /// Returns the initial state of the problem (when no decision is taken).
    fn initial_state(&self) -> T;
    /// Returns the initial value of the objective function (when no decision is taken).
    fn initial_value(&self) -> i32;

    /// Returns the domain of variable `var` in the given `state`. These are the
    /// possible values that might possibly be affected to `var` when the system
    /// has taken decisions leading to `state`.
    fn domain_of(&self, state: &T, var: Variable) -> &[i32];
    /// Returns the next state reached by the system if the decision `d` is
    /// taken when the system is in the given `state` and the given set of `vars`
    /// are still free (no value assigned).
    fn transition(&self, state: &T, vars : &VarSet, d: Decision) -> T;
    /// Returns the marginal benefit (in terms of objective function to maximize)
    /// of taking decision `d` is when the system is in the given `state` and
    /// the given set of `vars` are still free (no value assigned).
    fn transition_cost(&self, state: &T, vars : &VarSet, d: Decision) -> i32;

    /// Optional method for the case where you'd want to use a pooled mdd implementation.
    /// Returns true iff taking a decision on 'variable' might have an impact (state or lp)
    /// on a node having the given 'state'
    #[allow(unused_variables)]
    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        true
    }

    fn root_node(&self) -> Node<T> {
        Node::new(self.initial_state(), self.initial_value(), None, true)
    }
    fn all_vars(&self) -> VarSet {
        VarSet::all(self.nb_vars())
    }
}

/// This is the second most important abstraction that a client should provide
/// when using this library. It defines the relaxation that may be applied to
/// the given problem.
///
/// Again, the type parameter `<T>` denotes the type of the states.
pub trait Relaxation<T> where T: Eq + Clone {
    fn merge_nodes(&self, dd: &dyn MDD<T>, nodes: &[&Node<T>]) -> Node<T>;
    fn estimate_ub(&self, _node: &Node<T>) -> i32 {
        i32::max_value()
    }
}

/// This is a simplified version of the basic relaxation abstraction.
/// It is simpler in the sense that you don't need to deal with the node itself,
/// only with the states you decided to create.
///
/// *Note:*
/// All objects implementing `SimpleAbstraction` automatically implement the
/// `Relaxation` trait.
///
/// Again, the type parameter `<T>` denotes the type of the states.
pub trait SimpleRelaxation<T> {
    /// Merges the given `states` into a relaxed one which is an over
    /// approximation of the given `states`.
    /// The return value is the over approximation state.
    fn merge_states(&self, dd: &dyn MDD<T>, states: &[&T]) -> T;
    /// This method yields the _relaxed cost_ of taking the given `decision`
    /// in state `from` to reach the relaxed state `to`.
    fn relax_cost(&self, dd: &dyn MDD<T>, original_cost: i32, from: &T, to: &T, decision: Decision) -> i32;

    /// Optionally compute a rough upper bound on the objective value reachable
    /// from the given state. This method should be *fast* to compute and return
    /// an upper bound on the length of the longest path passing through state
    /// `s` (and assuming that the length of the longest path to `s` is `lp` long).
    ///
    /// Returning `i32::max_value()` is always correct, but it will prevent any
    /// rough upper bound pruning to occur.
    #[allow(unused_variables)]
    fn rough_ub(&self, lp: i32, s: &T) -> i32 {
        i32::max_value()
    }
}

impl <T: Eq + Clone, X: SimpleRelaxation<T>> Relaxation<T> for X {
    fn merge_nodes(&self, dd: &dyn MDD<T>, nodes: &[&Node<T>]) -> Node<T> {
        let mut maxlen = i32::min_value();
        let mut arc    = None;
        let mut states = vec![];
        states.reserve_exact(nodes.len());
        nodes.iter().for_each(|n| states.push(&n.state));
        let merged     = self.merge_states(dd, &states);

        for n in nodes.iter() {
            let narc = n.lp_arc.clone().unwrap();
            let cost = self.relax_cost(dd, narc.weight, &narc.src.state, &merged, narc.decision);

            if n.lp_len - narc.weight + cost > maxlen {
                maxlen      = n.lp_len - narc.weight + cost;
                arc         = Some(Arc{
                    src     : Rc::clone(&narc.src),
                    decision: narc.decision,
                    weight  : cost
                });
            }
        }
        Node::new(merged, maxlen, arc, false)
    }

    fn estimate_ub(&self, n: &Node<T>) -> i32 {
        self.rough_ub(n.lp_len, &n.state)
    }
}