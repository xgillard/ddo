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
    best_value.map(|x| -x).unwrap_or(-1)
}

#[test]
fn alp_n100_r1_c2_std10_s0() {
    assert_eq!(solve_id("alp_n100_r1_c2_std10_s0"), 17018);
}

#[test]
fn alp_n100_r1_c2_std10_s1() {
    assert_eq!(solve_id("alp_n100_r1_c2_std10_s1"), 3720);
}

#[test]
fn alp_n100_r1_c2_std10_s2() {
    assert_eq!(solve_id("alp_n100_r1_c2_std10_s2"), 14184);
}

#[test]
fn alp_n100_r1_c2_std10_s3() {
    assert_eq!(solve_id("alp_n100_r1_c2_std10_s3"), 1017);
}

#[test]
fn alp_n100_r1_c2_std10_s4() {
    assert_eq!(solve_id("alp_n100_r1_c2_std10_s4"), 12280);
}

#[test]
fn alp_n100_r1_c2_std20_s1() {
    assert_eq!(solve_id("alp_n100_r1_c2_std20_s1"), 7127);
}

#[test]
fn alp_n100_r1_c2_std20_s3() {
    assert_eq!(solve_id("alp_n100_r1_c2_std20_s3"), 7382);
}

#[test]
fn alp_n100_r1_c2_std30_s1() {
    assert_eq!(solve_id("alp_n100_r1_c2_std30_s1"), 24730);
}

#[test]
fn alp_n100_r1_c3_std10_s1() {
    assert_eq!(solve_id("alp_n100_r1_c3_std10_s1"), 6411);
}

#[test]
fn alp_n100_r1_c3_std10_s3() {
    assert_eq!(solve_id("alp_n100_r1_c3_std10_s3"), 12777);
}

#[test]
fn alp_n100_r1_c3_std10_s4() {
    assert_eq!(solve_id("alp_n100_r1_c3_std10_s4"), 24762);
}

#[test]
fn alp_n100_r1_c3_std20_s1() {
    assert_eq!(solve_id("alp_n100_r1_c3_std20_s1"), 11198);
}

#[test]
fn alp_n100_r1_c4_std10_s1() {
    assert_eq!(solve_id("alp_n100_r1_c4_std10_s1"), 2492);
}

#[test]
fn alp_n100_r1_c4_std10_s3() {
    assert_eq!(solve_id("alp_n100_r1_c4_std10_s3"), 8840);
}

#[test]
fn alp_n100_r1_c4_std20_s1() {
    assert_eq!(solve_id("alp_n100_r1_c4_std20_s1"), 3148);
}

#[test]
fn alp_n100_r1_c4_std20_s3() {
    assert_eq!(solve_id("alp_n100_r1_c4_std20_s3"), 11139);
}

#[test]
fn alp_n100_r1_c4_std30_s1() {
    assert_eq!(solve_id("alp_n100_r1_c4_std30_s1"), 5196);
}

#[test]
fn alp_n100_r1_c4_std30_s3() {
    assert_eq!(solve_id("alp_n100_r1_c4_std30_s3"), 14220);
}

#[test]
fn alp_n25_r1_c2_std10_s0() {
    assert_eq!(solve_id("alp_n25_r1_c2_std10_s0"), 755);
}

#[test]
fn alp_n25_r1_c2_std10_s1() {
    assert_eq!(solve_id("alp_n25_r1_c2_std10_s1"), 554);
}

#[test]
fn alp_n25_r1_c2_std10_s2() {
    assert_eq!(solve_id("alp_n25_r1_c2_std10_s2"), 637);
}

#[test]
fn alp_n25_r1_c2_std10_s3() {
    assert_eq!(solve_id("alp_n25_r1_c2_std10_s3"), 1503);
}

#[test]
fn alp_n25_r1_c2_std10_s4() {
    assert_eq!(solve_id("alp_n25_r1_c2_std10_s4"), 2985);
}

#[test]
fn alp_n25_r1_c2_std20_s0() {
    assert_eq!(solve_id("alp_n25_r1_c2_std20_s0"), 1213);
}

#[test]
fn alp_n25_r1_c2_std20_s1() {
    assert_eq!(solve_id("alp_n25_r1_c2_std20_s1"), 641);
}

#[test]
fn alp_n25_r1_c2_std20_s2() {
    assert_eq!(solve_id("alp_n25_r1_c2_std20_s2"), 3368);
}

#[test]
fn alp_n25_r1_c2_std20_s3() {
    assert_eq!(solve_id("alp_n25_r1_c2_std20_s3"), 1964);
}

#[test]
fn alp_n25_r1_c2_std30_s0() {
    assert_eq!(solve_id("alp_n25_r1_c2_std30_s0"), 2442);
}

#[test]
fn alp_n25_r1_c2_std30_s1() {
    assert_eq!(solve_id("alp_n25_r1_c2_std30_s1"), 839);
}

#[test]
fn alp_n25_r1_c2_std30_s2() {
    assert_eq!(solve_id("alp_n25_r1_c2_std30_s2"), 9440);
}

#[test]
fn alp_n25_r1_c2_std30_s3() {
    assert_eq!(solve_id("alp_n25_r1_c2_std30_s3"), 3065);
}

#[test]
fn alp_n25_r1_c3_std10_s0() {
    assert_eq!(solve_id("alp_n25_r1_c3_std10_s0"), 574);
}

#[test]
fn alp_n25_r1_c3_std10_s1() {
    assert_eq!(solve_id("alp_n25_r1_c3_std10_s1"), 182);
}

#[test]
fn alp_n25_r1_c3_std10_s2() {
    assert_eq!(solve_id("alp_n25_r1_c3_std10_s2"), 1951);
}

#[test]
fn alp_n25_r1_c3_std10_s3() {
    assert_eq!(solve_id("alp_n25_r1_c3_std10_s3"), 4508);
}

#[test]
fn alp_n25_r1_c3_std10_s4() {
    assert_eq!(solve_id("alp_n25_r1_c3_std10_s4"), 1732);
}

#[test]
fn alp_n25_r1_c3_std20_s0() {
    assert_eq!(solve_id("alp_n25_r1_c3_std20_s0"), 732);
}

#[test]
fn alp_n25_r1_c3_std20_s1() {
    assert_eq!(solve_id("alp_n25_r1_c3_std20_s1"), 323);
}

#[test]
fn alp_n25_r1_c3_std20_s2() {
    assert_eq!(solve_id("alp_n25_r1_c3_std20_s2"), 4905);
}

#[test]
fn alp_n25_r1_c3_std20_s3() {
    assert_eq!(solve_id("alp_n25_r1_c3_std20_s3"), 4426);
}

#[test]
fn alp_n25_r1_c3_std20_s4() {
    assert_eq!(solve_id("alp_n25_r1_c3_std20_s4"), 3745);
}

#[test]
fn alp_n25_r1_c3_std30_s0() {
    assert_eq!(solve_id("alp_n25_r1_c3_std30_s0"), 937);
}

#[test]
fn alp_n25_r1_c3_std30_s1() {
    assert_eq!(solve_id("alp_n25_r1_c3_std30_s1"), 446);
}

#[test]
fn alp_n25_r1_c3_std30_s2() {
    assert_eq!(solve_id("alp_n25_r1_c3_std30_s2"), 5767);
}

#[test]
fn alp_n25_r1_c3_std30_s3() {
    assert_eq!(solve_id("alp_n25_r1_c3_std30_s3"), 6723);
}

#[test]
fn alp_n25_r1_c3_std30_s4() {
    assert_eq!(solve_id("alp_n25_r1_c3_std30_s4"), 6817);
}

#[test]
fn alp_n25_r1_c4_std10_s0() {
    assert_eq!(solve_id("alp_n25_r1_c4_std10_s0"), 349);
}

#[test]
fn alp_n25_r1_c4_std10_s1() {
    assert_eq!(solve_id("alp_n25_r1_c4_std10_s1"), 826);
}

#[test]
fn alp_n25_r1_c4_std10_s2() {
    assert_eq!(solve_id("alp_n25_r1_c4_std10_s2"), 4131);
}

#[test]
fn alp_n25_r1_c4_std10_s3() {
    assert_eq!(solve_id("alp_n25_r1_c4_std10_s3"), 4799);
}

#[test]
fn alp_n25_r1_c4_std10_s4() {
    assert_eq!(solve_id("alp_n25_r1_c4_std10_s4"), 8850);
}

#[test]
fn alp_n25_r1_c4_std20_s0() {
    assert_eq!(solve_id("alp_n25_r1_c4_std20_s0"), 419);
}

#[test]
fn alp_n25_r1_c4_std20_s1() {
    assert_eq!(solve_id("alp_n25_r1_c4_std20_s1"), 1350);
}

#[test]
fn alp_n25_r1_c4_std20_s2() {
    assert_eq!(solve_id("alp_n25_r1_c4_std20_s2"), 4740);
}

#[test]
fn alp_n25_r1_c4_std20_s3() {
    assert_eq!(solve_id("alp_n25_r1_c4_std20_s3"), 4715);
}

#[test]
fn alp_n25_r1_c4_std30_s0() {
    assert_eq!(solve_id("alp_n25_r1_c4_std30_s0"), 646);
}

#[test]
fn alp_n25_r1_c4_std30_s1() {
    assert_eq!(solve_id("alp_n25_r1_c4_std30_s1"), 2209);
}

#[test]
fn alp_n25_r1_c4_std30_s2() {
    assert_eq!(solve_id("alp_n25_r1_c4_std30_s2"), 4675);
}

#[test]
fn alp_n25_r1_c4_std30_s3() {
    assert_eq!(solve_id("alp_n25_r1_c4_std30_s3"), 5912);
}

#[test]
fn alp_n25_r2_c2_std10_s0() {
    assert_eq!(solve_id("alp_n25_r2_c2_std10_s0"), 7);
}

#[test]
fn alp_n25_r2_c2_std10_s1() {
    assert_eq!(solve_id("alp_n25_r2_c2_std10_s1"), 98);
}

#[test]
fn alp_n25_r2_c2_std10_s2() {
    assert_eq!(solve_id("alp_n25_r2_c2_std10_s2"), 167);
}

#[test]
fn alp_n25_r2_c2_std10_s3() {
    assert_eq!(solve_id("alp_n25_r2_c2_std10_s3"), 53);
}

#[test]
fn alp_n25_r2_c2_std10_s4() {
    assert_eq!(solve_id("alp_n25_r2_c2_std10_s4"), 595);
}

#[test]
fn alp_n25_r2_c2_std20_s0() {
    assert_eq!(solve_id("alp_n25_r2_c2_std20_s0"), 56);
}

#[test]
fn alp_n25_r2_c2_std20_s1() {
    assert_eq!(solve_id("alp_n25_r2_c2_std20_s1"), 109);
}

#[test]
fn alp_n25_r2_c2_std20_s2() {
    assert_eq!(solve_id("alp_n25_r2_c2_std20_s2"), 1281);
}

#[test]
fn alp_n25_r2_c2_std20_s3() {
    assert_eq!(solve_id("alp_n25_r2_c2_std20_s3"), 338);
}

#[test]
fn alp_n25_r2_c2_std20_s4() {
    assert_eq!(solve_id("alp_n25_r2_c2_std20_s4"), 3043);
}

#[test]
fn alp_n25_r2_c2_std30_s0() {
    assert_eq!(solve_id("alp_n25_r2_c2_std30_s0"), 203);
}

#[test]
fn alp_n25_r2_c2_std30_s1() {
    assert_eq!(solve_id("alp_n25_r2_c2_std30_s1"), 165);
}

#[test]
fn alp_n25_r2_c2_std30_s2() {
    assert_eq!(solve_id("alp_n25_r2_c2_std30_s2"), 4169);
}

#[test]
fn alp_n25_r2_c2_std30_s3() {
    assert_eq!(solve_id("alp_n25_r2_c2_std30_s3"), 1161);
}

#[test]
fn alp_n25_r2_c3_std10_s0() {
    assert_eq!(solve_id("alp_n25_r2_c3_std10_s0"), 24);
}

#[test]
fn alp_n25_r2_c3_std10_s1() {
    assert_eq!(solve_id("alp_n25_r2_c3_std10_s1"), 27);
}

#[test]
fn alp_n25_r2_c3_std10_s2() {
    assert_eq!(solve_id("alp_n25_r2_c3_std10_s2"), 745);
}

#[test]
fn alp_n25_r2_c3_std10_s3() {
    assert_eq!(solve_id("alp_n25_r2_c3_std10_s3"), 1376);
}

#[test]
fn alp_n25_r2_c3_std10_s4() {
    assert_eq!(solve_id("alp_n25_r2_c3_std10_s4"), 75);
}

#[test]
fn alp_n25_r2_c3_std20_s0() {
    assert_eq!(solve_id("alp_n25_r2_c3_std20_s0"), 19);
}

#[test]
fn alp_n25_r2_c3_std20_s1() {
    assert_eq!(solve_id("alp_n25_r2_c3_std20_s1"), 34);
}

#[test]
fn alp_n25_r2_c3_std20_s2() {
    assert_eq!(solve_id("alp_n25_r2_c3_std20_s2"), 1777);
}

#[test]
fn alp_n25_r2_c3_std20_s3() {
    assert_eq!(solve_id("alp_n25_r2_c3_std20_s3"), 1718);
}

#[test]
fn alp_n25_r2_c3_std20_s4() {
    assert_eq!(solve_id("alp_n25_r2_c3_std20_s4"), 227);
}

#[test]
fn alp_n25_r2_c3_std30_s0() {
    assert_eq!(solve_id("alp_n25_r2_c3_std30_s0"), 49);
}

#[test]
fn alp_n25_r2_c3_std30_s1() {
    assert_eq!(solve_id("alp_n25_r2_c3_std30_s1"), 64);
}

#[test]
fn alp_n25_r2_c3_std30_s2() {
    assert_eq!(solve_id("alp_n25_r2_c3_std30_s2"), 2197);
}

#[test]
fn alp_n25_r2_c3_std30_s3() {
    assert_eq!(solve_id("alp_n25_r2_c3_std30_s3"), 2652);
}

#[test]
fn alp_n25_r2_c3_std30_s4() {
    assert_eq!(solve_id("alp_n25_r2_c3_std30_s4"), 477);
}

#[test]
fn alp_n25_r2_c4_std10_s0() {
    assert_eq!(solve_id("alp_n25_r2_c4_std10_s0"), 48);
}

#[test]
fn alp_n25_r2_c4_std10_s1() {
    assert_eq!(solve_id("alp_n25_r2_c4_std10_s1"), 320);
}

#[test]
fn alp_n25_r2_c4_std10_s2() {
    assert_eq!(solve_id("alp_n25_r2_c4_std10_s2"), 1790);
}

#[test]
fn alp_n25_r2_c4_std10_s4() {
    assert_eq!(solve_id("alp_n25_r2_c4_std10_s4"), 2547);
}

#[test]
fn alp_n25_r2_c4_std20_s0() {
    assert_eq!(solve_id("alp_n25_r2_c4_std20_s0"), 104);
}

#[test]
fn alp_n25_r2_c4_std20_s1() {
    assert_eq!(solve_id("alp_n25_r2_c4_std20_s1"), 339);
}

#[test]
fn alp_n25_r2_c4_std20_s2() {
    assert_eq!(solve_id("alp_n25_r2_c4_std20_s2"), 2053);
}

#[test]
fn alp_n25_r2_c4_std20_s3() {
    assert_eq!(solve_id("alp_n25_r2_c4_std20_s3"), 1482);
}

#[test]
fn alp_n25_r2_c4_std20_s4() {
    assert_eq!(solve_id("alp_n25_r2_c4_std20_s4"), 3281);
}

#[test]
fn alp_n25_r2_c4_std30_s0() {
    assert_eq!(solve_id("alp_n25_r2_c4_std30_s0"), 143);
}

#[test]
fn alp_n25_r2_c4_std30_s1() {
    assert_eq!(solve_id("alp_n25_r2_c4_std30_s1"), 638);
}

#[test]
fn alp_n25_r2_c4_std30_s2() {
    assert_eq!(solve_id("alp_n25_r2_c4_std30_s2"), 1859);
}

#[test]
fn alp_n25_r2_c4_std30_s3() {
    assert_eq!(solve_id("alp_n25_r2_c4_std30_s3"), 1873);
}

#[test]
fn alp_n25_r3_c2_std10_s0() {
    assert_eq!(solve_id("alp_n25_r3_c2_std10_s0"), 3);
}

#[test]
fn alp_n25_r3_c2_std10_s1() {
    assert_eq!(solve_id("alp_n25_r3_c2_std10_s1"), 19);
}

#[test]
fn alp_n25_r3_c2_std10_s3() {
    assert_eq!(solve_id("alp_n25_r3_c2_std10_s3"), 11);
}

#[test]
fn alp_n25_r3_c2_std10_s4() {
    assert_eq!(solve_id("alp_n25_r3_c2_std10_s4"), 93);
}

#[test]
fn alp_n25_r3_c2_std20_s0() {
    assert_eq!(solve_id("alp_n25_r3_c2_std20_s0"), 33);
}

#[test]
fn alp_n25_r3_c2_std20_s1() {
    assert_eq!(solve_id("alp_n25_r3_c2_std20_s1"), 34);
}

#[test]
fn alp_n25_r3_c2_std20_s3() {
    assert_eq!(solve_id("alp_n25_r3_c2_std20_s3"), 119);
}

#[test]
fn alp_n25_r3_c2_std20_s4() {
    assert_eq!(solve_id("alp_n25_r3_c2_std20_s4"), 912);
}

#[test]
fn alp_n25_r3_c2_std30_s0() {
    assert_eq!(solve_id("alp_n25_r3_c2_std30_s0"), 124);
}

#[test]
fn alp_n25_r3_c2_std30_s1() {
    assert_eq!(solve_id("alp_n25_r3_c2_std30_s1"), 36);
}

#[test]
fn alp_n25_r3_c3_std10_s0() {
    assert_eq!(solve_id("alp_n25_r3_c3_std10_s0"), 0);
}

#[test]
fn alp_n25_r3_c3_std10_s1() {
    assert_eq!(solve_id("alp_n25_r3_c3_std10_s1"), 3);
}

#[test]
fn alp_n25_r3_c3_std10_s4() {
    assert_eq!(solve_id("alp_n25_r3_c3_std10_s4"), 23);
}

#[test]
fn alp_n25_r3_c3_std20_s0() {
    assert_eq!(solve_id("alp_n25_r3_c3_std20_s0"), 4);
}

#[test]
fn alp_n25_r3_c3_std20_s4() {
    assert_eq!(solve_id("alp_n25_r3_c3_std20_s4"), 168);
}

#[test]
fn alp_n25_r3_c3_std30_s0() {
    assert_eq!(solve_id("alp_n25_r3_c3_std30_s0"), 16);
}

#[test]
fn alp_n25_r3_c3_std30_s4() {
    assert_eq!(solve_id("alp_n25_r3_c3_std30_s4"), 365);
}

#[test]
fn alp_n25_r3_c4_std10_s0() {
    assert_eq!(solve_id("alp_n25_r3_c4_std10_s0"), 19);
}

#[test]
fn alp_n25_r3_c4_std20_s0() {
    assert_eq!(solve_id("alp_n25_r3_c4_std20_s0"), 45);
}

#[test]
fn alp_n25_r3_c4_std20_s4() {
    assert_eq!(solve_id("alp_n25_r3_c4_std20_s4"), 1644);
}

#[test]
fn alp_n25_r3_c4_std30_s0() {
    assert_eq!(solve_id("alp_n25_r3_c4_std30_s0"), 69);
}

#[test]
fn alp_n25_r4_c2_std10_s0() {
    assert_eq!(solve_id("alp_n25_r4_c2_std10_s0"), 0);
}

#[test]
fn alp_n25_r4_c2_std10_s3() {
    assert_eq!(solve_id("alp_n25_r4_c2_std10_s3"), 3);
}

#[test]
fn alp_n25_r4_c2_std10_s4() {
    assert_eq!(solve_id("alp_n25_r4_c2_std10_s4"), 33);
}

#[test]
fn alp_n25_r4_c2_std20_s0() {
    assert_eq!(solve_id("alp_n25_r4_c2_std20_s0"), 0);
}

#[test]
fn alp_n25_r4_c2_std20_s1() {
    assert_eq!(solve_id("alp_n25_r4_c2_std20_s1"), 5);
}

#[test]
fn alp_n25_r4_c2_std30_s0() {
    assert_eq!(solve_id("alp_n25_r4_c2_std30_s0"), 45);
}

#[test]
fn alp_n25_r4_c3_std10_s0() {
    assert_eq!(solve_id("alp_n25_r4_c3_std10_s0"), 0);
}

#[test]
fn alp_n25_r4_c3_std10_s4() {
    assert_eq!(solve_id("alp_n25_r4_c3_std10_s4"), 19);
}

#[test]
fn alp_n25_r4_c3_std20_s0() {
    assert_eq!(solve_id("alp_n25_r4_c3_std20_s0"), 0);
}

#[test]
fn alp_n25_r4_c3_std30_s0() {
    assert_eq!(solve_id("alp_n25_r4_c3_std30_s0"), 0);
}

#[test]
fn alp_n50_r1_c2_std10_s0() {
    assert_eq!(solve_id("alp_n50_r1_c2_std10_s0"), 2054);
}

#[test]
fn alp_n50_r1_c2_std10_s1() {
    assert_eq!(solve_id("alp_n50_r1_c2_std10_s1"), 3180);
}

#[test]
fn alp_n50_r1_c2_std10_s2() {
    assert_eq!(solve_id("alp_n50_r1_c2_std10_s2"), 908);
}

#[test]
fn alp_n50_r1_c2_std10_s3() {
    assert_eq!(solve_id("alp_n50_r1_c2_std10_s3"), 305);
}

#[test]
fn alp_n50_r1_c2_std10_s4() {
    assert_eq!(solve_id("alp_n50_r1_c2_std10_s4"), 6235);
}

#[test]
fn alp_n50_r1_c2_std20_s0() {
    assert_eq!(solve_id("alp_n50_r1_c2_std20_s0"), 3396);
}

#[test]
fn alp_n50_r1_c2_std20_s1() {
    assert_eq!(solve_id("alp_n50_r1_c2_std20_s1"), 8579);
}

#[test]
fn alp_n50_r1_c2_std20_s2() {
    assert_eq!(solve_id("alp_n50_r1_c2_std20_s2"), 4173);
}

#[test]
fn alp_n50_r1_c2_std20_s3() {
    assert_eq!(solve_id("alp_n50_r1_c2_std20_s3"), 1313);
}

#[test]
fn alp_n50_r1_c2_std20_s4() {
    assert_eq!(solve_id("alp_n50_r1_c2_std20_s4"), 12646);
}

#[test]
fn alp_n50_r1_c2_std30_s0() {
    assert_eq!(solve_id("alp_n50_r1_c2_std30_s0"), 5351);
}

#[test]
fn alp_n50_r1_c2_std30_s2() {
    assert_eq!(solve_id("alp_n50_r1_c2_std30_s2"), 14233);
}

#[test]
fn alp_n50_r1_c2_std30_s3() {
    assert_eq!(solve_id("alp_n50_r1_c2_std30_s3"), 3228);
}

#[test]
fn alp_n50_r1_c3_std10_s0() {
    assert_eq!(solve_id("alp_n50_r1_c3_std10_s0"), 3038);
}

#[test]
fn alp_n50_r1_c3_std10_s2() {
    assert_eq!(solve_id("alp_n50_r1_c3_std10_s2"), 988);
}

#[test]
fn alp_n50_r1_c3_std10_s3() {
    assert_eq!(solve_id("alp_n50_r1_c3_std10_s3"), 1097);
}

#[test]
fn alp_n50_r1_c3_std20_s0() {
    assert_eq!(solve_id("alp_n50_r1_c3_std20_s0"), 13192);
}

#[test]
fn alp_n50_r1_c3_std20_s2() {
    assert_eq!(solve_id("alp_n50_r1_c3_std20_s2"), 2168);
}

#[test]
fn alp_n50_r1_c3_std20_s3() {
    assert_eq!(solve_id("alp_n50_r1_c3_std20_s3"), 4073);
}

#[test]
fn alp_n50_r1_c3_std30_s2() {
    assert_eq!(solve_id("alp_n50_r1_c3_std30_s2"), 7343);
}

#[test]
fn alp_n50_r1_c4_std10_s0() {
    assert_eq!(solve_id("alp_n50_r1_c4_std10_s0"), 8966);
}

#[test]
fn alp_n50_r1_c4_std10_s1() {
    assert_eq!(solve_id("alp_n50_r1_c4_std10_s1"), 10806);
}

#[test]
fn alp_n50_r1_c4_std10_s2() {
    assert_eq!(solve_id("alp_n50_r1_c4_std10_s2"), 2590);
}

#[test]
fn alp_n50_r1_c4_std10_s4() {
    assert_eq!(solve_id("alp_n50_r1_c4_std10_s4"), 1540);
}

#[test]
fn alp_n50_r1_c4_std20_s1() {
    assert_eq!(solve_id("alp_n50_r1_c4_std20_s1"), 7740);
}

#[test]
fn alp_n50_r1_c4_std20_s2() {
    assert_eq!(solve_id("alp_n50_r1_c4_std20_s2"), 3143);
}

#[test]
fn alp_n50_r1_c4_std20_s4() {
    assert_eq!(solve_id("alp_n50_r1_c4_std20_s4"), 2818);
}

#[test]
fn alp_n50_r1_c4_std30_s4() {
    assert_eq!(solve_id("alp_n50_r1_c4_std30_s4"), 6446);
}

#[test]
fn alp_n50_r2_c2_std10_s4() {
    assert_eq!(solve_id("alp_n50_r2_c2_std10_s4"), 369);
}

#[test]
fn alp_n75_r1_c2_std10_s0() {
    assert_eq!(solve_id("alp_n75_r1_c2_std10_s0"), 2204);
}

#[test]
fn alp_n75_r1_c2_std10_s2() {
    assert_eq!(solve_id("alp_n75_r1_c2_std10_s2"), 1101);
}

#[test]
fn alp_n75_r1_c2_std10_s3() {
    assert_eq!(solve_id("alp_n75_r1_c2_std10_s3"), 2273);
}

#[test]
fn alp_n75_r1_c2_std10_s4() {
    assert_eq!(solve_id("alp_n75_r1_c2_std10_s4"), 2475);
}

#[test]
fn alp_n75_r1_c2_std20_s0() {
    assert_eq!(solve_id("alp_n75_r1_c2_std20_s0"), 6353);
}

#[test]
fn alp_n75_r1_c2_std20_s2() {
    assert_eq!(solve_id("alp_n75_r1_c2_std20_s2"), 4661);
}

#[test]
fn alp_n75_r1_c2_std20_s3() {
    assert_eq!(solve_id("alp_n75_r1_c2_std20_s3"), 5629);
}

#[test]
fn alp_n75_r1_c2_std20_s4() {
    assert_eq!(solve_id("alp_n75_r1_c2_std20_s4"), 4520);
}

#[test]
fn alp_n75_r1_c2_std30_s0() {
    assert_eq!(solve_id("alp_n75_r1_c2_std30_s0"), 11642);
}

#[test]
fn alp_n75_r1_c2_std30_s2() {
    assert_eq!(solve_id("alp_n75_r1_c2_std30_s2"), 11097);
}

#[test]
fn alp_n75_r1_c2_std30_s4() {
    assert_eq!(solve_id("alp_n75_r1_c2_std30_s4"), 8366);
}

#[test]
fn alp_n75_r1_c3_std10_s0() {
    assert_eq!(solve_id("alp_n75_r1_c3_std10_s0"), 7303);
}

#[test]
fn alp_n75_r1_c3_std10_s2() {
    assert_eq!(solve_id("alp_n75_r1_c3_std10_s2"), 7356);
}

#[test]
fn alp_n75_r1_c3_std10_s3() {
    assert_eq!(solve_id("alp_n75_r1_c3_std10_s3"), 1928);
}

#[test]
fn alp_n75_r1_c3_std20_s3() {
    assert_eq!(solve_id("alp_n75_r1_c3_std20_s3"), 2225);
}

#[test]
fn alp_n75_r1_c3_std30_s3() {
    assert_eq!(solve_id("alp_n75_r1_c3_std30_s3"), 4257);
}

#[test]
fn alp_n75_r1_c4_std10_s0() {
    assert_eq!(solve_id("alp_n75_r1_c4_std10_s0"), 10309);
}

#[test]
fn alp_n75_r1_c4_std10_s1() {
    assert_eq!(solve_id("alp_n75_r1_c4_std10_s1"), 5632);
}

#[test]
fn alp_n75_r1_c4_std10_s2() {
    assert_eq!(solve_id("alp_n75_r1_c4_std10_s2"), 5412);
}

#[test]
fn alp_n75_r1_c4_std10_s3() {
    assert_eq!(solve_id("alp_n75_r1_c4_std10_s3"), 1955);
}

#[test]
fn alp_n75_r1_c4_std20_s0() {
    assert_eq!(solve_id("alp_n75_r1_c4_std20_s0"), 13091);
}

#[test]
fn alp_n75_r1_c4_std20_s1() {
    assert_eq!(solve_id("alp_n75_r1_c4_std20_s1"), 26667);
}

#[test]
fn alp_n75_r1_c4_std20_s2() {
    assert_eq!(solve_id("alp_n75_r1_c4_std20_s2"), 7817);
}

#[test]
fn alp_n75_r1_c4_std20_s3() {
    assert_eq!(solve_id("alp_n75_r1_c4_std20_s3"), 5268);
}

#[test]
fn alp_n75_r1_c4_std30_s2() {
    assert_eq!(solve_id("alp_n75_r1_c4_std30_s2"), 16169);
}

#[test]
fn alp_n75_r1_c4_std30_s3() {
    assert_eq!(solve_id("alp_n75_r1_c4_std30_s3"), 25379);
}