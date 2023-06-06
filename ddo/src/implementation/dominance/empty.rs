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

use std::{marker::PhantomData, cmp::Ordering, sync::Arc};
use crate::DominanceChecker;

/// Implementation of a dominance checker that never detects any dominance relation
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

    fn is_dominated_or_insert(&self, _: Arc<Self::State>, _: isize) -> bool {
        false
    }

    fn cmp(&self, _: &Self::State, _: &Self::State) -> Ordering {
        Ordering::Equal
    }
}