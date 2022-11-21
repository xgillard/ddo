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

//! This module is meant to tests the correctness of our max2sat example

use std::path::PathBuf;

use ddo::*;

use crate::{relax::Max2SatRelax, heuristics::Max2SatRanking, data::read_instance, model::Max2Sat};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/max2sat/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let problem = Max2Sat::new(read_instance(fname).unwrap());
    let relaxation = Max2SatRelax(&problem);
    let ranking = Max2SatRanking;

    let width = NbUnassignedWitdh(problem.nb_variables());
    let cutoff = NoCutoff;
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));
    let mut barrier = EmptyBarrier{};

    // This solver compile DD that allow the definition of long arcs spanning over several layers.
    let mut solver = DefaultSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &cutoff, 
        &mut fringe,
        &mut barrier,
    );

    let Completion { best_value , ..} = solver.maximize();
    best_value.map(|x| x).unwrap_or(-1)
}


#[test]
fn debug() {
    assert_eq!(solve_id("debug.wcnf"), 24);
}
#[test]
fn debug2() {
    assert_eq!(solve_id("debug2.wcnf"), 13);
}

#[test]
fn pass() {
    assert_eq!(solve_id("pass.wcnf"), 54);
}
#[test]
fn tautology() {
    assert_eq!(solve_id("tautology.wcnf"), 7);
}
#[test]
fn unit() {
    assert_eq!(solve_id("unit.wcnf"), 6);
}
#[test]
fn negative_wt() {
    assert_eq!(solve_id("negative_wt.wcnf"), 4258);
}

#[test]
fn frb10_6_1() {
    assert_eq!(solve_id("frb10-6-1.wcnf"), 37_037);
}
#[test]
fn frb10_6_2() {
    assert_eq!(solve_id("frb10-6-2.wcnf"), 38_196);
}
#[test]
fn frb10_6_3() {
    assert_eq!(solve_id("frb10-6-3.wcnf"), 36_671);
}
#[test]
fn frb10_6_4() {
    assert_eq!(solve_id("frb10-6-4.wcnf"), 38_928);
}
#[ignore] #[test]
fn frb15_9_1() {
    assert_eq!(solve_id("frb15-9-1.wcnf"), 341_783);
}
#[ignore] #[test]
fn frb15_9_2() {
    assert_eq!(solve_id("frb15-9-2.wcnf"), 341_919);
}
#[ignore] #[test]
fn frb15_9_3() {
    assert_eq!(solve_id("frb15-9-3.wcnf"), 339_471);
}
#[ignore] #[test]
fn frb15_9_4() {
    assert_eq!(solve_id("frb15-9-4.wcnf"), 340_559);
}
#[ignore] #[test]
fn frb15_9_5() {
    assert_eq!(solve_id("frb15-9-5.wcnf"), 348_311);
}
#[ignore] #[test]
fn frb20_11_1() {
    assert_eq!(solve_id("frb20-11-1.wcnf"), 1_245_134);
}
#[ignore] #[test]
fn frb20_11_2() {
    assert_eq!(solve_id("frb20-11-2.wcnf"), 1_231_874);
}
#[ignore] #[test]
fn frb20_11_3() {
    assert_eq!(solve_id("frb20-11-2.wcnf"), 1_240_493);
}
#[ignore] #[test]
fn frb20_11_4() {
    assert_eq!(solve_id("frb20-11-4.wcnf"), 1_231_653);
}
#[ignore] #[test]
fn frb20_11_5() {
    assert_eq!(solve_id("frb20-11-5.wcnf"), 1_237_841);
}