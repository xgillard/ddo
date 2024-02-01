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

use crate::{AlpRelax, AlpRanking, read_instance, AlpDominance, model::Alp};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/alp/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let instance = read_instance(fname).unwrap();
    let problem = Alp::new(instance);
    let relaxation = AlpRelax::new(problem.clone());
    let ranking = AlpRanking;

    let width = NbUnassignedWidth(problem.nb_variables());
    let dominance = SimpleDominanceChecker::new(AlpDominance, problem.nb_variables());
    let cutoff = NoCutoff;
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    let mut solver = DefaultBarrierSolver::new(
        &problem, 
        &relaxation, 
        &ranking, 
        &width, 
        &dominance,
        &cutoff, 
        &mut fringe
    );

    let Completion { best_value , ..} = solver.maximize();
    best_value.map(|x| -x).unwrap_or(-1)
}

#[test]
fn alp_n25_r1_c2_std10_s0() {
	assert_eq!(solve_id("alp_n25_r1_c2_std10_s0"), 684);
}

#[test]
fn alp_n25_r1_c2_std10_s1() {
	assert_eq!(solve_id("alp_n25_r1_c2_std10_s1"), 123);
}

#[test]
fn alp_n25_r1_c2_std10_s2() {
	assert_eq!(solve_id("alp_n25_r1_c2_std10_s2"), 547);
}

#[test]
fn alp_n25_r1_c2_std10_s3() {
	assert_eq!(solve_id("alp_n25_r1_c2_std10_s3"), 1463);
}

#[test]
fn alp_n25_r1_c2_std10_s4() {
	assert_eq!(solve_id("alp_n25_r1_c2_std10_s4"), 650);
}

/// SOMETIMES FAILING WITH MANY THREADS!!!
#[test]
fn alp_n25_r1_c2_std20_s0() {
	assert_eq!(solve_id("alp_n25_r1_c2_std20_s0"), 718);
}

#[test]
fn alp_n25_r1_c2_std20_s1() {
	assert_eq!(solve_id("alp_n25_r1_c2_std20_s1"), 131);
}

#[test]
fn alp_n25_r1_c2_std20_s2() {
	assert_eq!(solve_id("alp_n25_r1_c2_std20_s2"), 1195);
}

#[test]
fn alp_n25_r1_c2_std20_s3() {
	assert_eq!(solve_id("alp_n25_r1_c2_std20_s3"), 1118);
}

#[test]
fn alp_n25_r1_c2_std20_s4() {
	assert_eq!(solve_id("alp_n25_r1_c2_std20_s4"), 370);
}

#[test]
fn alp_n25_r1_c2_std30_s0() {
	assert_eq!(solve_id("alp_n25_r1_c2_std30_s0"), 809);
}

#[test]
fn alp_n25_r1_c2_std30_s1() {
	assert_eq!(solve_id("alp_n25_r1_c2_std30_s1"), 383);
}

#[test]
fn alp_n25_r1_c2_std30_s2() {
	assert_eq!(solve_id("alp_n25_r1_c2_std30_s2"), 2085);
}

#[test]
fn alp_n25_r1_c2_std30_s3() {
	assert_eq!(solve_id("alp_n25_r1_c2_std30_s3"), 1075);
}

#[test]
fn alp_n25_r1_c2_std30_s4() {
	assert_eq!(solve_id("alp_n25_r1_c2_std30_s4"), 292);
}

#[test]
fn alp_n25_r1_c3_std10_s0() {
	assert_eq!(solve_id("alp_n25_r1_c3_std10_s0"), 1508);
}

#[test]
fn alp_n25_r1_c3_std10_s1() {
	assert_eq!(solve_id("alp_n25_r1_c3_std10_s1"), 334);
}

#[test]
fn alp_n25_r1_c3_std10_s2() {
	assert_eq!(solve_id("alp_n25_r1_c3_std10_s2"), 1507);
}

#[test]
fn alp_n25_r1_c3_std10_s3() {
	assert_eq!(solve_id("alp_n25_r1_c3_std10_s3"), 1347);
}

#[test]
fn alp_n25_r1_c3_std20_s0() {
	assert_eq!(solve_id("alp_n25_r1_c3_std20_s0"), 1417);
}

#[test]
fn alp_n25_r1_c3_std20_s1() {
	assert_eq!(solve_id("alp_n25_r1_c3_std20_s1"), 472);
}

#[test]
fn alp_n25_r1_c3_std20_s2() {
	assert_eq!(solve_id("alp_n25_r1_c3_std20_s2"), 2215);
}

#[test]
fn alp_n25_r1_c3_std20_s3() {
	assert_eq!(solve_id("alp_n25_r1_c3_std20_s3"), 1587);
}

#[test]
fn alp_n25_r1_c3_std30_s0() {
	assert_eq!(solve_id("alp_n25_r1_c3_std30_s0"), 1343);
}

#[test]
fn alp_n25_r1_c3_std30_s1() {
	assert_eq!(solve_id("alp_n25_r1_c3_std30_s1"), 705);
}

#[test]
fn alp_n25_r1_c3_std30_s2() {
	assert_eq!(solve_id("alp_n25_r1_c3_std30_s2"), 3683);
}

#[test]
fn alp_n25_r1_c3_std30_s3() {
	assert_eq!(solve_id("alp_n25_r1_c3_std30_s3"), 1578);
}

#[test]
fn alp_n25_r1_c4_std10_s0() {
	assert_eq!(solve_id("alp_n25_r1_c4_std10_s0"), 1107);
}

#[test]
fn alp_n25_r1_c4_std10_s1() {
	assert_eq!(solve_id("alp_n25_r1_c4_std10_s1"), 647);
}

#[test]
fn alp_n25_r1_c4_std10_s2() {
	assert_eq!(solve_id("alp_n25_r1_c4_std10_s2"), 1167);
}

#[test]
fn alp_n25_r1_c4_std10_s3() {
	assert_eq!(solve_id("alp_n25_r1_c4_std10_s3"), 1175);
}

#[test]
fn alp_n25_r1_c4_std10_s4() {
	assert_eq!(solve_id("alp_n25_r1_c4_std10_s4"), 1401);
}

#[test]
fn alp_n25_r1_c4_std20_s0() {
	assert_eq!(solve_id("alp_n25_r1_c4_std20_s0"), 1331);
}

#[test]
fn alp_n25_r1_c4_std20_s1() {
	assert_eq!(solve_id("alp_n25_r1_c4_std20_s1"), 923);
}

#[test]
fn alp_n25_r1_c4_std20_s2() {
	assert_eq!(solve_id("alp_n25_r1_c4_std20_s2"), 1342);
}

#[test]
fn alp_n25_r1_c4_std20_s3() {
	assert_eq!(solve_id("alp_n25_r1_c4_std20_s3"), 1856);
}

#[test]
fn alp_n25_r1_c4_std20_s4() {
	assert_eq!(solve_id("alp_n25_r1_c4_std20_s4"), 1344);
}

#[test]
fn alp_n25_r1_c4_std30_s0() {
	assert_eq!(solve_id("alp_n25_r1_c4_std30_s0"), 1605);
}

#[test]
fn alp_n25_r1_c4_std30_s1() {
	assert_eq!(solve_id("alp_n25_r1_c4_std30_s1"), 1343);
}

#[test]
fn alp_n25_r1_c4_std30_s2() {
	assert_eq!(solve_id("alp_n25_r1_c4_std30_s2"), 1838);
}

#[test]
fn alp_n25_r1_c4_std30_s3() {
	assert_eq!(solve_id("alp_n25_r1_c4_std30_s3"), 3238);
}

#[test]
fn alp_n25_r1_c4_std30_s4() {
	assert_eq!(solve_id("alp_n25_r1_c4_std30_s4"), 1173);
}

#[test]
fn alp_n25_r2_c2_std10_s0() {
	assert_eq!(solve_id("alp_n25_r2_c2_std10_s0"), 9);
}

#[test]
fn alp_n25_r2_c2_std10_s1() {
	assert_eq!(solve_id("alp_n25_r2_c2_std10_s1"), 15);
}

#[test]
fn alp_n25_r2_c2_std10_s2() {
	assert_eq!(solve_id("alp_n25_r2_c2_std10_s2"), 99);
}

#[test]
fn alp_n25_r2_c2_std10_s3() {
	assert_eq!(solve_id("alp_n25_r2_c2_std10_s3"), 17);
}

#[test]
fn alp_n25_r2_c2_std10_s4() {
	assert_eq!(solve_id("alp_n25_r2_c2_std10_s4"), 8);
}

#[test]
fn alp_n25_r2_c2_std20_s0() {
	assert_eq!(solve_id("alp_n25_r2_c2_std20_s0"), 29);
}

#[test]
fn alp_n25_r2_c2_std20_s1() {
	assert_eq!(solve_id("alp_n25_r2_c2_std20_s1"), 22);
}

#[test]
fn alp_n25_r2_c2_std20_s2() {
	assert_eq!(solve_id("alp_n25_r2_c2_std20_s2"), 274);
}

#[test]
fn alp_n25_r2_c2_std20_s3() {
	assert_eq!(solve_id("alp_n25_r2_c2_std20_s3"), 26);
}

#[test]
fn alp_n25_r2_c2_std20_s4() {
	assert_eq!(solve_id("alp_n25_r2_c2_std20_s4"), 18);
}

#[test]
fn alp_n25_r2_c2_std30_s0() {
	assert_eq!(solve_id("alp_n25_r2_c2_std30_s0"), 58);
}

#[test]
fn alp_n25_r2_c2_std30_s1() {
	assert_eq!(solve_id("alp_n25_r2_c2_std30_s1"), 98);
}

#[test]
fn alp_n25_r2_c2_std30_s2() {
	assert_eq!(solve_id("alp_n25_r2_c2_std30_s2"), 602);
}

#[test]
fn alp_n25_r2_c2_std30_s3() {
	assert_eq!(solve_id("alp_n25_r2_c2_std30_s3"), 77);
}

#[test]
fn alp_n25_r2_c2_std30_s4() {
	assert_eq!(solve_id("alp_n25_r2_c2_std30_s4"), 20);
}

#[test]
fn alp_n25_r2_c3_std10_s0() {
	assert_eq!(solve_id("alp_n25_r2_c3_std10_s0"), 116);
}

#[test]
fn alp_n25_r2_c3_std10_s1() {
	assert_eq!(solve_id("alp_n25_r2_c3_std10_s1"), 11);
}

#[test]
fn alp_n25_r2_c3_std10_s2() {
	assert_eq!(solve_id("alp_n25_r2_c3_std10_s2"), 377);
}

#[test]
fn alp_n25_r2_c3_std10_s3() {
	assert_eq!(solve_id("alp_n25_r2_c3_std10_s3"), 144);
}

#[test]
fn alp_n25_r2_c3_std10_s4() {
	assert_eq!(solve_id("alp_n25_r2_c3_std10_s4"), 476);
}

#[test]
fn alp_n25_r2_c3_std20_s0() {
	assert_eq!(solve_id("alp_n25_r2_c3_std20_s0"), 124);
}

#[test]
fn alp_n25_r2_c3_std20_s1() {
	assert_eq!(solve_id("alp_n25_r2_c3_std20_s1"), 52);
}

#[test]
fn alp_n25_r2_c3_std20_s2() {
	assert_eq!(solve_id("alp_n25_r2_c3_std20_s2"), 816);
}

#[test]
fn alp_n25_r2_c3_std20_s3() {
	assert_eq!(solve_id("alp_n25_r2_c3_std20_s3"), 182);
}

#[test]
fn alp_n25_r2_c3_std20_s4() {
	assert_eq!(solve_id("alp_n25_r2_c3_std20_s4"), 563);
}

#[test]
fn alp_n25_r2_c3_std30_s0() {
	assert_eq!(solve_id("alp_n25_r2_c3_std30_s0"), 160);
}

#[test]
fn alp_n25_r2_c3_std30_s1() {
	assert_eq!(solve_id("alp_n25_r2_c3_std30_s1"), 124);
}

#[test]
fn alp_n25_r2_c3_std30_s2() {
	assert_eq!(solve_id("alp_n25_r2_c3_std30_s2"), 1591);
}

#[test]
fn alp_n25_r2_c3_std30_s3() {
	assert_eq!(solve_id("alp_n25_r2_c3_std30_s3"), 357);
}

#[test]
fn alp_n25_r2_c4_std10_s0() {
	assert_eq!(solve_id("alp_n25_r2_c4_std10_s0"), 259);
}

#[test]
fn alp_n25_r2_c4_std10_s1() {
	assert_eq!(solve_id("alp_n25_r2_c4_std10_s1"), 168);
}

#[test]
fn alp_n25_r2_c4_std10_s2() {
	assert_eq!(solve_id("alp_n25_r2_c4_std10_s2"), 305);
}

#[test]
fn alp_n25_r2_c4_std10_s3() {
	assert_eq!(solve_id("alp_n25_r2_c4_std10_s3"), 343);
}

#[test]
fn alp_n25_r2_c4_std10_s4() {
	assert_eq!(solve_id("alp_n25_r2_c4_std10_s4"), 326);
}

#[test]
fn alp_n25_r2_c4_std20_s0() {
	assert_eq!(solve_id("alp_n25_r2_c4_std20_s0"), 310);
}

#[test]
fn alp_n25_r2_c4_std20_s1() {
	assert_eq!(solve_id("alp_n25_r2_c4_std20_s1"), 398);
}

#[test]
fn alp_n25_r2_c4_std20_s2() {
	assert_eq!(solve_id("alp_n25_r2_c4_std20_s2"), 390);
}

#[test]
fn alp_n25_r2_c4_std20_s4() {
	assert_eq!(solve_id("alp_n25_r2_c4_std20_s4"), 354);
}

#[test]
fn alp_n25_r2_c4_std30_s0() {
	assert_eq!(solve_id("alp_n25_r2_c4_std30_s0"), 395);
}

#[test]
fn alp_n25_r2_c4_std30_s1() {
	assert_eq!(solve_id("alp_n25_r2_c4_std30_s1"), 633);
}

#[test]
fn alp_n25_r2_c4_std30_s4() {
	assert_eq!(solve_id("alp_n25_r2_c4_std30_s4"), 303);
}

#[test]
fn alp_n25_r3_c2_std10_s0() {
	assert_eq!(solve_id("alp_n25_r3_c2_std10_s0"), 0);
}

#[test]
fn alp_n25_r3_c2_std10_s1() {
	assert_eq!(solve_id("alp_n25_r3_c2_std10_s1"), 5);
}

#[test]
fn alp_n25_r3_c2_std10_s2() {
	assert_eq!(solve_id("alp_n25_r3_c2_std10_s2"), 12);
}

#[test]
fn alp_n25_r3_c2_std10_s3() {
	assert_eq!(solve_id("alp_n25_r3_c2_std10_s3"), 1);
}

#[test]
fn alp_n25_r3_c2_std10_s4() {
	assert_eq!(solve_id("alp_n25_r3_c2_std10_s4"), 1);
}

#[test]
fn alp_n25_r3_c2_std20_s0() {
	assert_eq!(solve_id("alp_n25_r3_c2_std20_s0"), 11);
}

#[test]
fn alp_n25_r3_c2_std20_s1() {
	assert_eq!(solve_id("alp_n25_r3_c2_std20_s1"), 4);
}

#[test]
fn alp_n25_r3_c2_std20_s2() {
	assert_eq!(solve_id("alp_n25_r3_c2_std20_s2"), 110);
}

#[test]
fn alp_n25_r3_c2_std20_s3() {
	assert_eq!(solve_id("alp_n25_r3_c2_std20_s3"), 0);
}

#[test]
fn alp_n25_r3_c2_std20_s4() {
	assert_eq!(solve_id("alp_n25_r3_c2_std20_s4"), 3);
}

#[test]
fn alp_n25_r3_c2_std30_s0() {
	assert_eq!(solve_id("alp_n25_r3_c2_std30_s0"), 18);
}

#[test]
fn alp_n25_r3_c2_std30_s1() {
	assert_eq!(solve_id("alp_n25_r3_c2_std30_s1"), 40);
}

#[test]
fn alp_n25_r3_c2_std30_s3() {
	assert_eq!(solve_id("alp_n25_r3_c2_std30_s3"), 33);
}

#[test]
fn alp_n25_r3_c2_std30_s4() {
	assert_eq!(solve_id("alp_n25_r3_c2_std30_s4"), 8);
}

#[test]
fn alp_n25_r3_c3_std10_s0() {
	assert_eq!(solve_id("alp_n25_r3_c3_std10_s0"), 3);
}

#[test]
fn alp_n25_r3_c3_std10_s1() {
	assert_eq!(solve_id("alp_n25_r3_c3_std10_s1"), 0);
}

#[test]
fn alp_n25_r3_c3_std10_s2() {
	assert_eq!(solve_id("alp_n25_r3_c3_std10_s2"), 74);
}

#[test]
fn alp_n25_r3_c3_std10_s3() {
	assert_eq!(solve_id("alp_n25_r3_c3_std10_s3"), 0);
}

#[test]
fn alp_n25_r3_c3_std10_s4() {
	assert_eq!(solve_id("alp_n25_r3_c3_std10_s4"), 6);
}

#[test]
fn alp_n25_r3_c3_std20_s0() {
	assert_eq!(solve_id("alp_n25_r3_c3_std20_s0"), 10);
}

#[test]
fn alp_n25_r3_c3_std20_s1() {
	assert_eq!(solve_id("alp_n25_r3_c3_std20_s1"), 10);
}

#[test]
fn alp_n25_r3_c3_std20_s3() {
	assert_eq!(solve_id("alp_n25_r3_c3_std20_s3"), 14);
}

#[test]
fn alp_n25_r3_c3_std20_s4() {
	assert_eq!(solve_id("alp_n25_r3_c3_std20_s4"), 38);
}

#[test]
fn alp_n25_r3_c3_std30_s0() {
	assert_eq!(solve_id("alp_n25_r3_c3_std30_s0"), 22);
}

#[test]
fn alp_n25_r3_c3_std30_s1() {
	assert_eq!(solve_id("alp_n25_r3_c3_std30_s1"), 48);
}

#[test]
fn alp_n25_r3_c3_std30_s3() {
	assert_eq!(solve_id("alp_n25_r3_c3_std30_s3"), 52);
}

#[test]
fn alp_n25_r3_c3_std30_s4() {
	assert_eq!(solve_id("alp_n25_r3_c3_std30_s4"), 94);
}

#[test]
fn alp_n25_r3_c4_std10_s0() {
	assert_eq!(solve_id("alp_n25_r3_c4_std10_s0"), 39);
}

#[test]
fn alp_n25_r3_c4_std10_s1() {
	assert_eq!(solve_id("alp_n25_r3_c4_std10_s1"), 53);
}

#[test]
fn alp_n25_r3_c4_std10_s2() {
	assert_eq!(solve_id("alp_n25_r3_c4_std10_s2"), 55);
}

#[test]
fn alp_n25_r3_c4_std10_s3() {
	assert_eq!(solve_id("alp_n25_r3_c4_std10_s3"), 86);
}

#[test]
fn alp_n25_r3_c4_std10_s4() {
	assert_eq!(solve_id("alp_n25_r3_c4_std10_s4"), 19);
}

#[test]
fn alp_n25_r3_c4_std20_s0() {
	assert_eq!(solve_id("alp_n25_r3_c4_std20_s0"), 76);
}

#[test]
fn alp_n25_r3_c4_std20_s1() {
	assert_eq!(solve_id("alp_n25_r3_c4_std20_s1"), 186);
}

#[test]
fn alp_n25_r3_c4_std20_s4() {
	assert_eq!(solve_id("alp_n25_r3_c4_std20_s4"), 86);
}

#[test]
fn alp_n25_r3_c4_std30_s0() {
	assert_eq!(solve_id("alp_n25_r3_c4_std30_s0"), 189);
}

#[test]
fn alp_n25_r3_c4_std30_s4() {
	assert_eq!(solve_id("alp_n25_r3_c4_std30_s4"), 145);
}

#[test]
fn alp_n25_r4_c2_std10_s0() {
	assert_eq!(solve_id("alp_n25_r4_c2_std10_s0"), 0);
}

#[test]
fn alp_n25_r4_c2_std10_s1() {
	assert_eq!(solve_id("alp_n25_r4_c2_std10_s1"), 0);
}

#[test]
fn alp_n25_r4_c2_std10_s2() {
	assert_eq!(solve_id("alp_n25_r4_c2_std10_s2"), 8);
}

#[test]
fn alp_n25_r4_c2_std10_s3() {
	assert_eq!(solve_id("alp_n25_r4_c2_std10_s3"), 0);
}

#[test]
fn alp_n25_r4_c2_std10_s4() {
	assert_eq!(solve_id("alp_n25_r4_c2_std10_s4"), 0);
}

#[test]
fn alp_n25_r4_c2_std20_s0() {
	assert_eq!(solve_id("alp_n25_r4_c2_std20_s0"), 0);
}

#[test]
fn alp_n25_r4_c2_std20_s1() {
	assert_eq!(solve_id("alp_n25_r4_c2_std20_s1"), 0);
}

#[test]
fn alp_n25_r4_c2_std20_s2() {
	assert_eq!(solve_id("alp_n25_r4_c2_std20_s2"), 67);
}

#[test]
fn alp_n25_r4_c2_std20_s3() {
	assert_eq!(solve_id("alp_n25_r4_c2_std20_s3"), 0);
}

#[test]
fn alp_n25_r4_c2_std20_s4() {
	assert_eq!(solve_id("alp_n25_r4_c2_std20_s4"), 0);
}

#[test]
fn alp_n25_r4_c2_std30_s0() {
	assert_eq!(solve_id("alp_n25_r4_c2_std30_s0"), 0);
}

#[test]
fn alp_n25_r4_c2_std30_s1() {
	assert_eq!(solve_id("alp_n25_r4_c2_std30_s1"), 13);
}

#[test]
fn alp_n25_r4_c2_std30_s3() {
	assert_eq!(solve_id("alp_n25_r4_c2_std30_s3"), 8);
}

#[test]
fn alp_n25_r4_c2_std30_s4() {
	assert_eq!(solve_id("alp_n25_r4_c2_std30_s4"), 0);
}

#[test]
fn alp_n25_r4_c3_std10_s0() {
	assert_eq!(solve_id("alp_n25_r4_c3_std10_s0"), 0);
}

#[test]
fn alp_n25_r4_c3_std10_s1() {
	assert_eq!(solve_id("alp_n25_r4_c3_std10_s1"), 0);
}

#[test]
fn alp_n25_r4_c3_std10_s2() {
	assert_eq!(solve_id("alp_n25_r4_c3_std10_s2"), 12);
}

#[test]
fn alp_n25_r4_c3_std10_s3() {
	assert_eq!(solve_id("alp_n25_r4_c3_std10_s3"), 0);
}

#[test]
fn alp_n25_r4_c3_std10_s4() {
	assert_eq!(solve_id("alp_n25_r4_c3_std10_s4"), 0);
}

#[test]
fn alp_n25_r4_c3_std20_s0() {
	assert_eq!(solve_id("alp_n25_r4_c3_std20_s0"), 0);
}

#[test]
fn alp_n25_r4_c3_std20_s1() {
	assert_eq!(solve_id("alp_n25_r4_c3_std20_s1"), 2);
}

#[test]
fn alp_n25_r4_c3_std20_s3() {
	assert_eq!(solve_id("alp_n25_r4_c3_std20_s3"), 5);
}

#[test]
fn alp_n25_r4_c3_std20_s4() {
	assert_eq!(solve_id("alp_n25_r4_c3_std20_s4"), 0);
}

#[test]
fn alp_n25_r4_c3_std30_s0() {
	assert_eq!(solve_id("alp_n25_r4_c3_std30_s0"), 0);
}

#[test]
fn alp_n25_r4_c3_std30_s1() {
	assert_eq!(solve_id("alp_n25_r4_c3_std30_s1"), 12);
}

#[test]
fn alp_n25_r4_c3_std30_s3() {
	assert_eq!(solve_id("alp_n25_r4_c3_std30_s3"), 31);
}

#[test]
fn alp_n25_r4_c3_std30_s4() {
	assert_eq!(solve_id("alp_n25_r4_c3_std30_s4"), 20);
}

#[test]
fn alp_n25_r4_c4_std10_s0() {
	assert_eq!(solve_id("alp_n25_r4_c4_std10_s0"), 0);
}

#[test]
fn alp_n25_r4_c4_std10_s1() {
	assert_eq!(solve_id("alp_n25_r4_c4_std10_s1"), 20);
}

#[test]
fn alp_n25_r4_c4_std10_s3() {
	assert_eq!(solve_id("alp_n25_r4_c4_std10_s3"), 32);
}

#[test]
fn alp_n25_r4_c4_std10_s4() {
	assert_eq!(solve_id("alp_n25_r4_c4_std10_s4"), 8);
}

#[test]
fn alp_n25_r4_c4_std20_s0() {
	assert_eq!(solve_id("alp_n25_r4_c4_std20_s0"), 10);
}

#[test]
fn alp_n25_r4_c4_std20_s1() {
	assert_eq!(solve_id("alp_n25_r4_c4_std20_s1"), 77);
}

#[test]
fn alp_n25_r4_c4_std20_s4() {
	assert_eq!(solve_id("alp_n25_r4_c4_std20_s4"), 34);
}

#[test]
fn alp_n25_r4_c4_std30_s0() {
	assert_eq!(solve_id("alp_n25_r4_c4_std30_s0"), 26);
}

#[test]
fn alp_n25_r4_c4_std30_s4() {
	assert_eq!(solve_id("alp_n25_r4_c4_std30_s4"), 44);
}

#[test]
fn alp_n50_r1_c2_std10_s0() {
	assert_eq!(solve_id("alp_n50_r1_c2_std10_s0"), 1156);
}

#[test]
fn alp_n50_r1_c2_std10_s2() {
	assert_eq!(solve_id("alp_n50_r1_c2_std10_s2"), 83);
}

#[test]
fn alp_n50_r1_c2_std10_s3() {
	assert_eq!(solve_id("alp_n50_r1_c2_std10_s3"), 615);
}

#[test]
fn alp_n50_r1_c2_std10_s4() {
	assert_eq!(solve_id("alp_n50_r1_c2_std10_s4"), 245);
}

#[test]
fn alp_n50_r1_c2_std20_s0() {
	assert_eq!(solve_id("alp_n50_r1_c2_std20_s0"), 1245);
}

#[test]
fn alp_n50_r1_c2_std20_s2() {
	assert_eq!(solve_id("alp_n50_r1_c2_std20_s2"), 329);
}

#[test]
fn alp_n50_r1_c2_std20_s3() {
	assert_eq!(solve_id("alp_n50_r1_c2_std20_s3"), 1125);
}

#[test]
fn alp_n50_r1_c2_std20_s4() {
	assert_eq!(solve_id("alp_n50_r1_c2_std20_s4"), 916);
}

#[test]
fn alp_n50_r1_c2_std30_s0() {
	assert_eq!(solve_id("alp_n50_r1_c2_std30_s0"), 1506);
}

#[test]
fn alp_n50_r1_c2_std30_s2() {
	assert_eq!(solve_id("alp_n50_r1_c2_std30_s2"), 813);
}

#[test]
fn alp_n50_r1_c2_std30_s4() {
	assert_eq!(solve_id("alp_n50_r1_c2_std30_s4"), 2471);
}

#[test]
fn alp_n50_r1_c3_std10_s0() {
	assert_eq!(solve_id("alp_n50_r1_c3_std10_s0"), 662);
}

#[test]
fn alp_n50_r1_c3_std10_s1() {
	assert_eq!(solve_id("alp_n50_r1_c3_std10_s1"), 1166);
}

#[test]
fn alp_n50_r1_c3_std10_s2() {
	assert_eq!(solve_id("alp_n50_r1_c3_std10_s2"), 871);
}

#[test]
fn alp_n50_r1_c3_std10_s3() {
	assert_eq!(solve_id("alp_n50_r1_c3_std10_s3"), 713);
}

#[test]
fn alp_n50_r1_c3_std10_s4() {
	assert_eq!(solve_id("alp_n50_r1_c3_std10_s4"), 2649);
}

#[test]
fn alp_n50_r1_c3_std20_s0() {
	assert_eq!(solve_id("alp_n50_r1_c3_std20_s0"), 923);
}

#[test]
fn alp_n50_r1_c3_std20_s2() {
	assert_eq!(solve_id("alp_n50_r1_c3_std20_s2"), 1291);
}

#[test]
fn alp_n50_r1_c3_std20_s3() {
	assert_eq!(solve_id("alp_n50_r1_c3_std20_s3"), 1212);
}

#[test]
fn alp_n50_r1_c3_std20_s4() {
	assert_eq!(solve_id("alp_n50_r1_c3_std20_s4"), 3106);
}

#[test]
fn alp_n50_r1_c3_std30_s0() {
	assert_eq!(solve_id("alp_n50_r1_c3_std30_s0"), 1458);
}

#[test]
fn alp_n50_r1_c3_std30_s2() {
	assert_eq!(solve_id("alp_n50_r1_c3_std30_s2"), 1903);
}

#[test]
fn alp_n50_r1_c3_std30_s4() {
	assert_eq!(solve_id("alp_n50_r1_c3_std30_s4"), 4326);
}

#[test]
fn alp_n50_r1_c4_std10_s0() {
	assert_eq!(solve_id("alp_n50_r1_c4_std10_s0"), 1876);
}

#[test]
fn alp_n50_r1_c4_std10_s2() {
	assert_eq!(solve_id("alp_n50_r1_c4_std10_s2"), 1385);
}

#[test]
fn alp_n50_r1_c4_std10_s4() {
	assert_eq!(solve_id("alp_n50_r1_c4_std10_s4"), 1791);
}

#[test]
fn alp_n50_r1_c4_std20_s0() {
	assert_eq!(solve_id("alp_n50_r1_c4_std20_s0"), 2210);
}

#[test]
fn alp_n50_r1_c4_std20_s2() {
	assert_eq!(solve_id("alp_n50_r1_c4_std20_s2"), 1264);
}

#[test]
fn alp_n50_r1_c4_std30_s0() {
	assert_eq!(solve_id("alp_n50_r1_c4_std30_s0"), 2606);
}

#[test]
fn alp_n50_r1_c4_std30_s2() {
	assert_eq!(solve_id("alp_n50_r1_c4_std30_s2"), 1517);
}

#[test]
fn alp_n50_r2_c2_std10_s0() {
	assert_eq!(solve_id("alp_n50_r2_c2_std10_s0"), 28);
}

#[test]
fn alp_n50_r2_c2_std10_s2() {
	assert_eq!(solve_id("alp_n50_r2_c2_std10_s2"), 24);
}

#[test]
fn alp_n50_r2_c2_std10_s4() {
	assert_eq!(solve_id("alp_n50_r2_c2_std10_s4"), 14);
}

#[test]
fn alp_n50_r2_c3_std10_s0() {
	assert_eq!(solve_id("alp_n50_r2_c3_std10_s0"), 80);
}

#[test]
fn alp_n50_r2_c3_std10_s4() {
	assert_eq!(solve_id("alp_n50_r2_c3_std10_s4"), 98);
}

#[test]
fn alp_n50_r2_c3_std20_s4() {
	assert_eq!(solve_id("alp_n50_r2_c3_std20_s4"), 208);
}

#[test]
fn alp_n50_r3_c3_std10_s4() {
	assert_eq!(solve_id("alp_n50_r3_c3_std10_s4"), 5);
}

#[test]
fn alp_n75_r1_c2_std10_s2() {
	assert_eq!(solve_id("alp_n75_r1_c2_std10_s2"), 176);
}

#[test]
fn alp_n75_r1_c2_std20_s2() {
	assert_eq!(solve_id("alp_n75_r1_c2_std20_s2"), 684);
}

#[test]
fn alp_n75_r1_c2_std30_s2() {
	assert_eq!(solve_id("alp_n75_r1_c2_std30_s2"), 1625);
}