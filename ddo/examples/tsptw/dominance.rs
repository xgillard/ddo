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

use std::{sync::Arc, hash::Hash};

use ddo::Dominance;

use crate::state::TsptwState;

pub struct TsptwKey(Arc<TsptwState>);
impl Hash for TsptwKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.position.hash(state);
        self.0.must_visit.hash(state);
    }
}
impl PartialEq for TsptwKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.position == other.0.position &&
            self.0.must_visit == other.0.must_visit
    }
}
impl Eq for TsptwKey {}

pub struct TsptwDominance;
impl Dominance for TsptwDominance {
    type State = TsptwState;
    type Key = TsptwKey;

    fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
        Some(TsptwKey(state))
    }

    fn nb_dimensions(&self, _: &Self::State) -> usize {
        0
    }

    fn get_coordinate(&self, _: &Self::State, _: usize) -> isize {
        0
    }

    fn use_value(&self) -> bool {
        true
    }
}