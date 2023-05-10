use std::{marker::PhantomData, cmp::Ordering};

use crate::DominanceChecker;


pub struct EmptyDominanceChecker<T>
{
    _phantom: PhantomData<T>,
}

impl<T> Default for EmptyDominanceChecker<T> {
    fn default() -> Self {
        Self { _phantom: Default::default() }
    }
}

impl<T> DominanceChecker for EmptyDominanceChecker<T> {
    type State = T;

    fn is_dominated_or_insert(&self, _: &Self::State) -> bool {
        false
    }

    fn cmp(&self, _: &Self::State, _: &Self::State) -> Ordering {
        Ordering::Equal
    }
}