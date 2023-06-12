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

//! This module is meant to tests the correctness of our talentsched example

use std::path::PathBuf;

use ddo::*;

use crate::{model::{TalentSchedRelax, TalentSched, TalentSchedRanking}, io_utils::read_instance};

fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/talentsched/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let instance = read_instance(fname).unwrap();
    let problem = TalentSched::new(instance);
    let relaxation = TalentSchedRelax::new(problem.clone());
    let ranking = TalentSchedRanking;

    let width = FixedWidth(100);
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
fn tiny() {
    assert_eq!(solve_id("tiny"), 29);
}

#[test]
fn tiny2() {
    assert_eq!(solve_id("tiny2"), 9);
}

#[test]
fn small() {
    assert_eq!(solve_id("small"), 54);
}

#[test]
fn small2() {
    assert_eq!(solve_id("small2"), 56);
}

#[test]
fn concert() {
    assert_eq!(solve_id("concert"), 111);
}

#[test]
fn film10() {
    assert_eq!(solve_id("film-10"), 352);
}

#[test]
fn film12() {
    assert_eq!(solve_id("film-12"), 401);
}

#[test]
fn film103() {
    assert_eq!(solve_id("film103.dat"), 1031);
}

#[test]
fn film105() {
    assert_eq!(solve_id("film105.dat"), 849);
}

#[test]
fn film114() {
    assert_eq!(solve_id("film114.dat"), 867);
}

#[test]
fn film116() {
    assert_eq!(solve_id("film116.dat"), 541);
}

#[test]
fn film118() {
    assert_eq!(solve_id("film118.dat"), 853);
}

#[test]
fn film119() {
    assert_eq!(solve_id("film119.dat"), 790);
}

#[test]
fn warwick1201() {
    assert_eq!(solve_id("Warwick1201"), 222);
}

#[test]
fn shaw2020() {
    assert_eq!(solve_id("Shaw2020"), 877);
}

#[test]
fn mobstory() {
    assert_eq!(solve_id("MobStory"), 871);
}