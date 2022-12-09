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

//! This module provide the solver implementation.
mod parallel;
mod sequential;
pub use parallel::*;
pub use sequential::*;

use crate::{DefaultMDDLEL, EmptyBarrier, SimpleBarrier, DefaultMDDFC};

/// A type alias to emphasize that this is the solver that should be used by default.
pub type DefaultSolver<'a, State>        = ParNoBarrierSolverLel<'a, State>;
pub type DefaultBarrierSolver<'a, State> = ParBarrierSolverFc<'a, State>;

pub type ParNoBarrierSolverLel<'a, State>= ParallelSolver<'a, State, DefaultMDDLEL<State>, EmptyBarrier<State>>;
pub type ParNoBarrierSolverFc<'a, State> = ParallelSolver<'a, State, DefaultMDDFC<State>,  EmptyBarrier<State>>;

pub type ParBarrierSolverLel<'a, State>  = ParallelSolver<'a, State, DefaultMDDLEL<State>, SimpleBarrier<State>>;
pub type ParBarrierSolverFc<'a, State>   = ParallelSolver<'a, State, DefaultMDDFC<State>,  SimpleBarrier<State>>;


pub type SeqNoBarrierSolverLel<'a, State>= SequentialSolver<'a, State, DefaultMDDLEL<State>, EmptyBarrier<State>>;
pub type SeqNoBarrierSolverFc<'a, State> = SequentialSolver<'a, State, DefaultMDDFC<State>,  EmptyBarrier<State>>;

pub type SeqBarrierSolverLel<'a, State>  = SequentialSolver<'a, State, DefaultMDDLEL<State>, SimpleBarrier<State>>;
pub type SeqBarrierSolverFc<'a, State>   = SequentialSolver<'a, State, DefaultMDDFC<State>,  SimpleBarrier<State>>;