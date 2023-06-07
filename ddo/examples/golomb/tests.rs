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

//! This module is meant to tests the correctness of our golomb example
use ddo::*;

use crate::{Golomb, Golombranking, GolombRelax};

pub fn solve_golomb(n: usize) -> isize {

    let problem = Golomb::new(n);
    let relaxation = GolombRelax{pb: &problem};
    let heuristic = Golombranking;
    let width = NbUnassignedWitdh(problem.nb_variables());
    let dominance = EmptyDominanceChecker::default();
    let cutoff = NoCutoff;
    let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));

    let mut solver = DefaultBarrierSolver::new(
        &problem,
        &relaxation,
        &heuristic,
        &width,
        &dominance,
        &cutoff,
        &mut fringe,
    );

    let Completion{ is_exact: _, best_value } = solver.maximize();
    best_value.map(|x| -x).unwrap_or(-1)
}

#[test]
fn golomb2() {
    assert_eq!(solve_golomb(2), 1);
}

#[test]
fn golomb3() {
    assert_eq!(solve_golomb(3), 3);
}

#[test]
fn golomb4() {
    assert_eq!(solve_golomb(4), 6);
}

#[test]
fn golomb5() {
    assert_eq!(solve_golomb(5), 11);
}

#[test]
fn golomb6() {
    assert_eq!(solve_golomb(6), 17);
}

#[test]
fn golomb7() {
    assert_eq!(solve_golomb(7), 25);
}

#[test]
fn golomb8() {
    assert_eq!(solve_golomb(8), 34);
}

#[test]
fn golomb9() {
    assert_eq!(solve_golomb(9), 44);
}