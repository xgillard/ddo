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

//! This module is meant to tests the correctness of our lcs example

use std::path::PathBuf;

use ddo::*;

use crate::{dominance::LcsDominance, io_utils::read_instance, model::{LcsRelax, LcsRanking}};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/lcs/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let problem = read_instance(fname).unwrap();
    let relaxation = LcsRelax::new(&problem);
    let ranking = LcsRanking;

    let width = FixedWidth(100);
    let dominance = SimpleDominanceChecker::new(LcsDominance);
    let cutoff = NoCutoff;
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    // This solver compile DD that allow the definition of long arcs spanning over several layers.
    let mut solver = ParBarrierSolverPooled::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &dominance,
        &cutoff, 
        &mut fringe,
    );

    let Completion { best_value , ..} = solver.maximize();
    best_value.map(|x| x).unwrap_or(-1)
}


#[ignore] #[test]
fn bacteria_elusimicrobia() {
    assert_eq!(solve_id("Bacteria;Elusimicrobia.bacchg"), 1141);
}

#[test]
fn neg_12_3_1() {
    assert_eq!(solve_id("neg_12_3_1.txt"), 203);
}

#[test]
fn neg_12_3_3() {
    assert_eq!(solve_id("neg_12_3_3.txt"), 358);
}

#[ignore] #[test]
fn neg_12_3_6() {
    assert_eq!(solve_id("neg_12_3_6.txt"), 245);
}

#[ignore] #[test]
fn neg_12_3_8() {
    assert_eq!(solve_id("neg_12_3_8.txt"), 268);
}

#[test]
fn poz_12_3_1() {
    assert_eq!(solve_id("poz_12_3_1.txt"), 222);
}
