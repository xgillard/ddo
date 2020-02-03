//! This module defines the basic abstractions one will manipulate when
//! implementing an MDD optimization-solver for some problem formulated as a
//! dynamic program.
//!
//! The most important abstractions that should be provided by a client are
//! `Problem` and `Relaxation`.
use std::i32;
use bitset_fixed::BitSet;
use crate::core::utils::BitSetIter;
use std::ops::Not;
use crate::core::abstraction::mdd::MDD;

/// This type denotes a variable from the optimization problem at hand.
/// In this case, each variable is assumed to be identified with an integer
/// ranging from 0 until `problem.nb_vars()`
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Variable(pub usize);

/// This type denotes a set of variable. It encodes them compactly as a fixed
/// size bitset. A `VarSet` can be efficiently iterated upon.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct VarSet(pub BitSet);

/// This denotes a decision that was made during the search. It affects a given
/// `value` to the specified `variable`. Any given `Decision` should be
/// understood as ```[[ variable = value ]]````
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Decision {
    pub variable : Variable,
    pub value    : i32
}

/// This is the main abstraction that should be provided by any user of our
/// library. Indeed, it defines the problem to be solved in the form of a dynamic
/// program. Therefore, this trait closely sticks to the formal definition of a
/// dynamic program.
///
/// The type parameter `<T>` denotes the type of the states of the dynamic program.
pub trait Problem<T> {
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
    fn transition(&self, state: &T, vars : &VarSet, d: &Decision) -> T;
    /// Returns the marginal benefit (in terms of objective function to maximize)
    /// of taking decision `d` is when the system is in the given `state` and
    /// the given set of `vars` are still free (no value assigned).
    fn transition_cost(&self, state: &T, vars : &VarSet, d: &Decision) -> i32;

    /// Optional method for the case where you'd want to use a pooled mdd implementation.
    /// Returns true iff taking a decision on 'variable' might have an impact (state or lp)
    /// on a node having the given 'state'
    #[allow(unused_variables)]
    fn impacted_by(&self, state: &T, variable: Variable) -> bool {
        true
    }
}

/// This is the second most important abstraction that a client should provide
/// when using this library. It defines the relaxation that may be applied to
/// the given problem.
///
/// Again, the type parameter `<T>` denotes the type of the states.
pub trait Relaxation<T> {
    /// Merges the given `states` into a relaxed one which is an over
    /// approximation of the given `states`.
    /// The return value is the over approximation state.
    fn merge_states(&self, states: &[&T]) -> T;
    /// This method yields the _relaxed cost_ of taking the given `decision`
    /// in state `from` to reach the relaxed state `to`.
    fn relax_cost(&self, from: &T, to: &T, decision: &Decision) -> i32;

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

/// This type denotes the iterator used to iterate over the `Variable`s to a
/// given `VarSet`. It should never be manually instantiated, but always via
/// the `iter()` method from the varset.
pub struct VarSetIter<'a>(pub BitSetIter<'a>);

/// This is the implementation of the core features of a `VarSet`.
impl VarSet {
    /// Returns a `VarSet` where all the possible `n` variables are presetn.
    pub fn all(n: usize) -> VarSet {
        VarSet(BitSet::new(n).not())
    }
    /// Adds the given variable `v` to the set if it is not already present.
    pub fn add(&mut self, v: Variable) {
        self.0.set(v.0, true)
    }
    /// Removes the variable `v` from the set if it was present.
    pub fn remove(&mut self, v: Variable) {
        self.0.set(v.0, false)
    }
    /// Returns true iff the set contains the variable `v`.
    pub fn contains(&self, v: Variable) -> bool {
        self.0[v.0]
    }
    /// Returns the count of variables that are present in the set.
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }
    /// Returns true iff no variables are preset in the set.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns an iterator in this set of variables.
    pub fn iter(&self) -> VarSetIter {
        VarSetIter(BitSetIter::new(&self.0))
    }
}
/// Actually implement the iterator protocol.
impl Iterator for VarSetIter<'_> {
    type Item = Variable;
    /// Returns the next variable from the set, or `None` if all variables have
    /// already been iterated upon.
    fn next(&mut self) -> Option<Variable> {
        self.0.next().map(Variable)
    }
}