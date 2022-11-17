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

//! This module is meant to tests the correctness of our knapsack example

use std::path::PathBuf;

use ddo::*;

use crate::{KPRelax, KPranking, read_instance};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/knapsack/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let problem = read_instance(fname).unwrap();
    let relaxation = KPRelax{pb: &problem};
    let ranking = KPranking;

    let width = NbUnassignedWitdh(problem.nb_variables());
    let cutoff = NoCutoff;
    let mut fringe = NoDupFrontier::new(MaxUB::new(&ranking));

    // This solver compile DD that allow the definition of long arcs spanning over several layers.
    let mut solver = DefaultSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &cutoff, 
        &mut fringe);

    let Completion { best_value , ..} = solver.maximize();
    best_value.map(|x| x).unwrap_or(-1)
}


#[test]
fn f9_l_d_kp_5_80() {
    assert_eq!(solve_id("f9_l-d_kp_5_80"), 130);
}
#[test]
fn f7_l_d_kp_7_50() {
    assert_eq!(solve_id("f7_l-d_kp_7_50"), 107);
}
#[test]
fn f3_l_d_kp_4_20() {
    assert_eq!(solve_id("f3_l-d_kp_4_20"), 35);
}
#[test]
fn f4_l_d_kp_4_11() {
    assert_eq!(solve_id("f4_l-d_kp_4_11"), 23);
}
#[test]
fn f10_l_d_kp_20_879() {
    assert_eq!(solve_id("f10_l-d_kp_20_879"), 1025);
}
#[test]
fn f1_l_d_kp_10_269() {
    assert_eq!(solve_id("f1_l-d_kp_10_269"), 295);
}
#[test]
fn f6_l_d_kp_10_60() {
    assert_eq!(solve_id("f6_l-d_kp_10_60"), 52);
}
#[test]
fn f8_l_d_kp_23_10000() {
    assert_eq!(solve_id("f8_l-d_kp_23_10000"), 9767);
}
#[test]
fn f2_l_d_kp_20_878() {
    assert_eq!(solve_id("f2_l-d_kp_20_878"), 1024);
}

// =================================================================
// test a few (easy) large scale instances.
// =================================================================
#[test]
fn knappi_1_100_1000_1() {
    assert_eq!(solve_id("knapPI_1_100_1000_1"), 9147);
}
#[test]
fn knappi_1_200_1000_1() {
    assert_eq!(solve_id("knapPI_1_200_1000_1"), 11238);
}
#[test]
fn knappi_2_100_1000_1() {
    assert_eq!(solve_id("knapPI_2_100_1000_1"), 1514);
}

// =================================================================
// uncomment these lines for more (large scale) integration tests.
// these may take a while..
// =================================================================
// #[test]
// fn knappi_1_5000_1000_1() {
//     assert_eq!(solve_id("knapPI_1_5000_1000_1"), 276457);
// }
// 
// #[test]
// fn knappi_2_2000_1000_1() {
//     assert_eq!(solve_id("knapPI_2_2000_1000_1"), 18051);
// }
// 
// #[test]
// fn knappi_3_200_1000_1() {
//     assert_eq!(solve_id("knapPI_3_200_1000_1"), 2697);
// }
// 
// #[test]
// fn knappi_1_500_1000_1() {
//     assert_eq!(solve_id("knapPI_1_500_1000_1"), 28857);
// }
// 
// #[test]
// fn knappi_1_10000_1000_1() {
//     assert_eq!(solve_id("knapPI_1_10000_1000_1"), 563647);
// }
// 
// #[test]
// fn knappi_1_2000_1000_1() {
//     assert_eq!(solve_id("knapPI_1_2000_1000_1"), 110625);
// }
// 
// #[test]
// fn knappi_3_1000_1000_1() {
//     assert_eq!(solve_id("knapPI_3_1000_1000_1"), 14390);
// }
// 
// #[test]
// fn knappi_2_5000_1000_1() {
//     assert_eq!(solve_id("knapPI_2_5000_1000_1"), 44356);
// }
// 
// #[test]
// fn knappi_3_500_1000_1() {
//     assert_eq!(solve_id("knapPI_3_500_1000_1"), 7117);
// }
// 
// #[test]
// fn knappi_2_10000_1000_1() {
//     assert_eq!(solve_id("knapPI_2_10000_1000_1"), 90204);
// }
// 
// #[test]
// fn knappi_3_2000_1000_1() {
//     assert_eq!(solve_id("knapPI_3_2000_1000_1"), 28919);
// }
// 
// #[test]
// fn knappi_3_100_1000_1() {
//     assert_eq!(solve_id("knapPI_3_100_1000_1"), 2397);
// }
// 
// #[test]
// fn knappi_1_1000_1000_1() {
//     assert_eq!(solve_id("knapPI_1_1000_1000_1"), 54503);
// }
// 
// #[test]
// fn knappi_3_10000_1000_1() {
//     assert_eq!(solve_id("knapPI_3_10000_1000_1"), 146919);
// }
// 
// #[test]
// fn knappi_2_200_1000_1() {
//     assert_eq!(solve_id("knapPI_2_200_1000_1"), 1634);
// }
// 
// #[test]
// fn knappi_3_5000_1000_1() {
//     assert_eq!(solve_id("knapPI_3_5000_1000_1"), 72505);
// }
// 
// #[test]
// fn knappi_2_1000_1000_1() {
//     assert_eq!(solve_id("knapPI_2_1000_1000_1"), 9052);
// }
// 
// #[test]
// fn knappi_2_500_1000_1() {
//     assert_eq!(solve_id("knapPI_2_500_1000_1"), 4566);
// }
// 