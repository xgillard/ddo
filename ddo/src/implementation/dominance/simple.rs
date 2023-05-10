use std::{hash::Hash, cmp::Ordering};

use dashmap::{DashMap, mapref::entry::Entry};

use crate::{Dominance, DominanceChecker};

#[derive(Debug)]
pub struct SimpleDominanceChecker<D>
where
    D: Dominance,
    D::Key: Eq + PartialEq + Hash,
    D::State: Clone,
{
    dominance: D,
    data: DashMap<D::Key, Vec<D::State>, fxhash::FxBuildHasher>,
}

impl<D> Default for SimpleDominanceChecker<D> 
where
    D: Dominance + Default,
    D::Key: Eq + PartialEq + Hash,
    D::State: Clone,
{
    fn default() -> Self {
        Self { dominance: Default::default(), data: Default::default() }
    }
}

impl<D> DominanceChecker for SimpleDominanceChecker<D> 
where
    D: Dominance,
    D::Key: Eq + PartialEq + Hash,
    D::State: Clone,
{
    type State = D::State;

    fn is_dominated_or_insert(&self, state: &Self::State) -> bool {
        if let Some(key) = self.dominance.get_key(state) {
            match self.data.entry(key) {
                Entry::Occupied(mut e) => {
                    let mut dominated = false;
                    e.get_mut().retain(|other| {
                        match self.dominance.partial_cmp(state, other) {
                            Some(ord) => match ord {
                                Ordering::Less => {
                                    dominated = true;
                                    true
                                },
                                Ordering::Equal => false,
                                Ordering::Greater => false,
                            },
                            None => true,
                        }
                    });
                    if !dominated {
                        e.get_mut().push(state.clone());
                    }
                    dominated
                },
                Entry::Vacant(e) => {
                    e.insert(vec![state.clone()]);
                    false
                },
            }
        } else {
            false
        }
    }

    fn cmp(&self, a: &Self::State, b: &Self::State) -> Ordering {
        self.dominance.cmp(a, b)
    }
}