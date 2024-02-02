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

use std::path::PathBuf;

use ddo::*;

use crate::{relax::SrflpRelax, heuristics::SrflpRanking, io_utils::read_instance, model::Srflp};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/srflp/")
        .join(id)
}

pub fn solve_id(id: &str) -> f64 {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let instance = read_instance(fname).unwrap();
    let problem = Srflp::new(instance);
    let relaxation = SrflpRelax::new(&problem);
    let ranking = SrflpRanking;

    let width = FixedWidth(1000);
    let dominance = EmptyDominanceChecker::default();
    let cutoff = NoCutoff;
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    let mut solver = DefaultCachingSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &dominance,
        &cutoff, 
        &mut fringe
    );

    let Completion { best_value , ..} = solver.maximize();
    best_value.map(|v| - v as f64 + problem.root_value()).unwrap_or(-1.0)
}

#[test]
fn cl5() {
    assert_eq!(solve_id("Cl5"), 1100.0);
}

#[test]
fn cl6() {
    assert_eq!(solve_id("Cl6"), 1990.0);
}

#[test]
fn cl7() {
    assert_eq!(solve_id("Cl7"), 4730.0);
}

#[test]
fn cl8() {
    assert_eq!(solve_id("Cl8"), 6295.0);
}

#[test]
fn cl12() {
    assert_eq!(solve_id("Cl12"), 23365.0);
}

#[test]
fn cl15() {
    assert_eq!(solve_id("Cl15"), 44600.0);
}

#[test]
fn cl20() {
    assert_eq!(solve_id("Cl20"), 119710.0);
}

#[test]
fn s8() {
    assert_eq!(solve_id("S8"), 801.0);
}

#[test]
fn s8h() {
    assert_eq!(solve_id("S8H"), 2324.5);
}

#[test]
fn s9() {
    assert_eq!(solve_id("S9"), 2469.5);
}

#[test]
fn s9h() {
    assert_eq!(solve_id("S9H"), 4695.5);
}

#[test]
fn s10() {
    assert_eq!(solve_id("S10"), 2781.5);
}

#[test]
fn s11() {
    assert_eq!(solve_id("S11"), 6933.5);
}

#[test]
fn p15() {
    assert_eq!(solve_id("P15"), 6305.0);
}

#[test]
fn p17() {
    assert_eq!(solve_id("P17"), 9254.0);
}

#[test]
fn p18() {
    assert_eq!(solve_id("P18"), 10650.5);
}

#[test]
fn h20() {
    assert_eq!(solve_id("H20"), 15549.0);
}