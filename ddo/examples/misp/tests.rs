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

//! This module is meant to tests the correctness of our misp example

use std::{path::PathBuf};

use ddo::*;

use crate::{read_instance, MispRelax, MispRanking};


fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/misp/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let problem = read_instance(fname).unwrap();
    let relaxation = MispRelax {pb: &&problem};
    let ranking = MispRanking;

    let width = NbUnassignedWitdh(problem.nb_variables());
    let cutoff = NoCutoff;
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    // This solver compile DD that allow the definition of long arcs spanning over several layers.
    let mut solver = DefaultSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &cutoff, 
        &mut fringe,
    );

    let Completion { best_value , ..} = solver.maximize();
    best_value.map(|x| x).unwrap_or(-1)
}


/// This test takes > 60s to solve on my machine
#[ignore] #[test]
fn brock200_1() {
    assert_eq!(solve_id("brock200_1.clq"), 21);
}
#[test]
fn brock200_2() {
    assert_eq!(solve_id("brock200_2.clq"), 12);
}
#[test]
fn brock200_3() {
    assert_eq!(solve_id("brock200_3.clq"), 15);
}
#[test]
fn brock200_4() {
    assert_eq!(solve_id("brock200_4.clq"), 17);
}


#[test]
fn c_fat200_1() {
    assert_eq!(solve_id("c-fat200-1.clq"), 12);
}
#[test]
fn c_fat200_2() {
    assert_eq!(solve_id("c-fat200-2.clq"), 24);
}
#[test]
fn c_fat500_1() {
    assert_eq!(solve_id("c-fat500-1.clq"), 14);
}
#[test]
fn c_fat500_2() {
    assert_eq!(solve_id("c-fat500-2.clq"), 26);
}
#[test]
fn c_fat200_5() {
    assert_eq!(solve_id("c-fat200-5.clq"), 58);
}

#[test]
fn hamming6_2() {
    assert_eq!(solve_id("hamming6-2.clq"), 32);
}

#[test]
fn hamming6_4() {
    assert_eq!(solve_id("hamming6-4.clq"), 4);
}
#[test]
fn hamming8_2() {
    assert_eq!(solve_id("hamming8-2.clq"), 128);
}
#[test]
fn hamming8_4() {
    assert_eq!(solve_id("hamming8-4.clq"), 16);
}
#[ignore] #[test]
fn hamming10_4() {
    assert_eq!(solve_id("hamming10-4.clq"), 40);
}

#[test]
fn johnson8_2_4() {
    assert_eq!(solve_id("johnson8-2-4.clq"), 4);
}
#[test]
fn johnson8_4_4() {
    assert_eq!(solve_id("johnson8-4-4.clq"), 14);
}

#[test]
fn keller4() {
    assert_eq!(solve_id("keller4.clq"), 11);
}
#[ignore] #[test]
fn keller5() {
    assert_eq!(solve_id("keller5.clq"), 27);
}

#[test]
fn mann_a9() {
    assert_eq!(solve_id("MANN_a9.clq"), 16);
}
#[ignore] #[test]
fn mann_a27() {
    assert_eq!(solve_id("MANN_a27.clq"), 126);
}
#[ignore] #[test]
fn mann_a45() {
    assert_eq!(solve_id("MANN_a45.clq"), 315);
}

#[test]
fn p_hat300_1() {
    assert_eq!(solve_id("p_hat300-1.clq"), 8);
}
#[ignore] #[test]
fn p_hat300_2() {
    assert_eq!(solve_id("p_hat300-2.clq"), 25);
}
#[ignore] #[test]
fn p_hat300_3() {
    assert_eq!(solve_id("p_hat300-3.clq"), 36);
}
#[ignore] #[test]
fn p_hat700_1() {
    assert_eq!(solve_id("p_hat700-1.clq"), 11);
}
#[ignore] #[test]
fn p_hat700_2() {
    assert_eq!(solve_id("p_hat700-2.clq"), 44);
}
#[ignore] #[test]
fn p_hat700_3() {
    assert_eq!(solve_id("p_hat700-3.clq"), 62);
}
#[ignore] #[test]
fn p_hat1500_1() {
    assert_eq!(solve_id("p_hat1500-1.clq"), 12);
}
#[ignore] #[test]
fn p_hat1500_2() {
    assert_eq!(solve_id("p_hat1500-2.clq"), 65);
}
#[ignore] #[test]
fn p_hat1500_3() {
    assert_eq!(solve_id("p_hat1500-3.clq"), 94);
}