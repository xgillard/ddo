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

//! This module is meant to tests the correctness of our sop example

use std::path::PathBuf;

use ddo::*;

use crate::{model::Sop, relax::SopRelax, heuristics::{SopRanking, SopWidth}, io_utils::read_instance};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/sop/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let instance = read_instance(fname).unwrap();
    let problem = Sop::new(instance);
    let relaxation = SopRelax::new(&problem);
    let ranking = SopRanking;

    let width = SopWidth::new(problem.nb_variables(), 1);
    let dominance = EmptyDominanceChecker::default();
    let cutoff = NoCutoff;
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    let mut solver = DefaultBarrierSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &dominance,
        &cutoff, 
        &mut fringe,
    );

    let Completion { best_value , ..} = solver.maximize();
    best_value.map(|x| -x).unwrap_or(-1)
}

#[test]
fn esc07() {
    assert_eq!(solve_id("ESC07.sop"), 2125);
}

#[test]
fn esc11() {
    assert_eq!(solve_id("ESC11.sop"), 2075);
}

#[test]
fn esc12() {
    assert_eq!(solve_id("ESC12.sop"), 1675);
}

#[test]
fn esc25() {
    assert_eq!(solve_id("ESC25.sop"), 1681);
}

#[ignore] #[test]
fn esc47() {
    assert_eq!(solve_id("ESC47.sop"), 1288);
}

#[test]
fn br17_10() {
    assert_eq!(solve_id("br17.10.sop"), 55);
}

#[test]
fn br17_12() {
    assert_eq!(solve_id("br17.12.sop"), 55);
}

#[ignore] #[test]
fn prob_42() {
    assert_eq!(solve_id("prob.42.sop"), 243);
}

#[ignore] #[test]
fn p43_1() {
    assert_eq!(solve_id("p43.1.sop"), 28140);
}

#[test]
fn p43_4() {
    assert_eq!(solve_id("p43.4.sop"), 83005);
}

#[ignore] #[test]
fn ry48p_1() {
    assert_eq!(solve_id("ry48p.1.sop"), 15805);
}

#[test]
fn ry48p_4() {
    assert_eq!(solve_id("ry48p.4.sop"), 31446);
}

#[test]
fn ft53_4() {
    assert_eq!(solve_id("ft53.4.sop"), 14425);
}