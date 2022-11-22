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

//! This module (and its submodule) provide the abstractions for the basic
//! building blocks of an MDD solvers. A client willing to use our library to
//! implement a solver for his/her particular problem should look into the `dp`
//! submodule. Indeed, `dp` is the place where the traits `Problem` and
//! `Relaxation` are defined. These are the two abstractions that one *must*
//! implement in order to be able to use our library.

use std::{marker::PhantomData, sync::Arc};

use crate::*;

/// Dummy implementation of Barrier with no information stored at all.
pub struct EmptyBarrier<T> {
    phantom: PhantomData<T>,
}

impl<T> EmptyBarrier<T> {
    pub fn new() -> Self {
        EmptyBarrier { phantom: PhantomData::default() }
    }
}

impl<T> Barrier for EmptyBarrier<T> {
    type State = T;

    fn get_threshold(&self, _: Arc<T>, _: usize) -> Option<Threshold> {
        None
    }

    fn update_threshold(&self, _: Arc<T>, _: usize, _: isize, _: bool) {}

    fn clear_layer(&self, _: usize) {}

    fn clear(&self) {}

}