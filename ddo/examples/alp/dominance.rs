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

use super::model::AlpState;

pub struct AlpKey(Arc<AlpState>);
impl Hash for AlpKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.rem.hash(state);
        self.0.info.iter().for_each(|i| i.prev_class.hash(state));
    }
}

impl PartialEq for AlpKey {
    fn eq(&self, other: &Self) -> bool {
        if self.0.rem != other.0.rem {
            return false;
        }
        self.0.info.iter()
            .zip(other.0.info.iter())
            .all(|(i1, i2)| i1.prev_class == i2.prev_class)
    }
}

impl Eq for AlpKey {}

pub struct AlpDominance;
impl Dominance for AlpDominance {
    type State = AlpState;
    type Key = AlpKey;

    fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
        Some(AlpKey(state))
    }

    fn nb_dimensions(&self, state: &Self::State) -> usize {
        state.info.len()
    }

    fn get_coordinate(&self, state: &Self::State, i: usize) -> isize {
        - state.info[i].prev_time
    }

    fn use_value(&self) -> bool { true }
}