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

use ddo::Dominance;

use crate::model::LcsState;

pub struct LcsDominance;
impl Dominance for LcsDominance {
    type State = LcsState;
    type Key = usize;

    fn get_key(&self, state: std::sync::Arc<Self::State>) -> Option<Self::Key> {
        Some(state.position[0])
    }

    fn nb_dimensions(&self, state: &Self::State) -> usize {
        state.position.len()
    }

    fn get_coordinate(&self, state: &Self::State, i: usize) -> isize {
        - (state.position[i] as isize)
    }

    fn use_value(&self) -> bool {
        true
    }
}