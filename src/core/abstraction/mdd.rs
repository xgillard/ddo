//! This module defines the traits for the constituents of an MDD: the `MDD`
//! itself and the `Nodes` that compose it.
use crate::core::abstraction::dp::{Decision, Variable, VarSet};
use std::hash::Hash;
use std::collections::HashMap;
use std::rc::Rc;
use std::cmp::{max, Ordering};
use std::cmp::Ordering::Equal;


/// This enumeration characterizes the kind of MDD being generated. It can
/// either be
/// * `Exact` if it is a true account of the problem state space.
/// * `Restricted` if it is an under approximation of the problem state space.
/// * `Relaxed` if it is an over approximation of the problem state space.
#[derive(Copy, Clone)]
pub enum MDDType {
    Relaxed,
    Restricted,
    Exact
}

/// This structure is in charge of unrolling an MDD according to the requested
/// level of exactitude
pub trait MDDGenerator<T> where T : Clone + Hash + Eq {
    /// Expands this MDD into  an exact MDD
    fn exact(&mut self, vars: VarSet, root: &Node<T>, best_lb : i32);
    /// Expands this MDD into a restricted (lower bound approximation)
    /// version of the exact MDD.
    fn restricted(&mut self, vars: VarSet, root: &Node<T>, best_lb : i32);
    /// Expands this MDD into a relaxed (upper bound approximation)
    /// version of the exact MDD.
    fn relaxed(&mut self, vars: VarSet, root: &Node<T>, best_lb : i32);
    /// Returns a reference to the expanded MDD.
    fn mdd(&self) -> &dyn MDD<T>;
}

/// This trait describes an MDD
///
/// # Type param
/// The type parameter `<T>` denotes the type of the state defined/manipulated
/// by the `Problem` definition.
pub trait MDD<T>
    where T : Clone + Hash + Eq {

    /// Tells whether this MDD is exact, relaxed, or restricted.
    fn mdd_type(&self) -> MDDType;
    /// Returns the nodes from the current layer.
    fn current_layer(&self) -> &[Node<T>];
    /// Returns a set of nodes constituting an exact cutset of this `MDD`.
    fn exact_cutset(&self) -> &[Node<T>];
    /// Returns a the map State -> Node of nodes already recorded to participate
    /// in the next layer. (All these nodes will have to be expanded to explore
    /// the complete state space).
    fn next_layer(&self) -> &HashMap<T, Node<T>>;

    /// Returns the latest `Variable` that acquired a value during the
    /// development of this `MDD`.
    ///
    /// # Note
    /// This development might still be ongoing. That last variable will then
    /// simply be variable associated to the decisions from the previous layer.
    fn last_assigned(&self) -> Variable;
    /// Returns the set of variables that have not been assigned any value
    /// (so far!).
    fn unassigned_vars(&self) -> &VarSet;

    /// Return true iff this `MDD` is exact. That is to say, it returns true if
    /// no nodes have been merged (because of relaxation) or suppressed (because
    /// of restriction).
    fn is_exact(&self) -> bool;
    /// Returns the length of the longest path between the root and the terminal
    /// node of this `MDD`.
    fn best_value(&self) -> i32;
    /// Returns the terminal node having the longest associated path in this `MDD`.
    fn best_node(&self) -> &Option<Node<T>>;
    /// Returns the list of decisions along the longest path between the
    /// root node and the best terminal node of this `MDD`.
    fn longest_path(&self) -> Vec<Decision>;
}

// --- NODE --------------------------------------------------------------------
#[derive(Clone, Eq, PartialEq)]
pub struct Arc<T>
    where T : Clone + Hash + Eq {
    pub src     : Rc<Node<T>>,
    pub decision: Decision,
    pub weight  : i32
}

// --- NODE --------------------------------------------------------------------
#[derive(Clone, Eq, PartialEq)]
pub struct Node<T> where T : Hash + Eq + Clone {
    pub state   : T,
    pub is_exact: bool,
    pub lp_len  : i32,
    pub lp_arc  : Option<Arc<T>>,

    pub ub      : i32
}

impl <T> Node<T> where T : Hash + Eq + Clone {
    pub fn new(state: T, lp_len: i32, lp_arc: Option<Arc<T>>, is_exact: bool) -> Node<T> {
        Node{state, is_exact, lp_len, lp_arc, ub: std::i32::MAX}
    }

    pub fn is_exact(&self) -> bool {
        self.is_exact
    }
    pub fn get_state(&self)-> &T {
        &self.state
    }
    pub fn get_lp_len(&self) -> i32 {
        self.lp_len
    }
    pub fn get_ub(&self) -> i32 {
        self.ub
    }
    pub fn set_ub(&mut self, ub: i32) {
        self.ub = max(self.ub, ub);
    }

    /// Merge other into this node. That is to say, it combines the information
    /// from two nodes that are logically equivalent (should be the same).
    /// Hence, *this has nothing to do with the user-provided `merge_*` operators !*
    pub fn merge(&mut self, other: Self) {
        if  self.lp_len < other.lp_len {
            self.lp_len = other.lp_len;
            self.lp_arc = other.lp_arc;
        }
        self.is_exact &= other.is_exact;
    }

    pub fn longest_path(&self) -> Vec<Decision> {
        let mut ret = vec![];
        let mut arc = &self.lp_arc;

        while arc.is_some() {
            let a = arc.as_ref().unwrap();
            ret.push(a.decision);
            arc = &a.src.lp_arc;
        }

        ret
    }
}

impl <T> Ord for Node<T> where T : Hash + Eq + Clone + Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl <T> PartialOrd for Node<T> where T : Hash + Eq + Clone + PartialOrd {
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
