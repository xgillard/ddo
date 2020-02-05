#![cfg(test)]
extern crate rust_mdd_solver;

use rust_mdd_solver::examples::max2sat::model::Max2Sat;
use std::path::PathBuf;
use std::fs::File;
use rust_mdd_solver::examples::max2sat::relax::Max2SatRelax;
use rust_mdd_solver::core::implementation::heuristics::{FixedWidth, MinLP, MaxUB, FromLongestPath};
use rust_mdd_solver::examples::max2sat::heuristics::Max2SatOrder;
use rust_mdd_solver::core::implementation::pooled_mdd::PooledMDDGenerator;
use rust_mdd_solver::core::implementation::bb_solver::BBSolver;
use rust_mdd_solver::core::abstraction::solver::Solver;

/// This method simply loads a resource into a problem instance to solve
fn instance(id: &str) -> Max2Sat {
    let location = PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("tests/resources/max2sat/")
        .join(id);

    File::open(location.to_owned()).expect("File not found").into()
}

/// This is the function we will use to actually solve an instance to completion
/// and check that the optimal value it identifies corresponds to the expected
/// value.
fn solve(id: &str) -> i32 {
    let problem    = instance(id);
    let relax      = Max2SatRelax::new(&problem);
    let width      = FixedWidth(3);
    let vs         = Max2SatOrder::new(&problem);
    let ns         = MinLP;

    let ddg        = PooledMDDGenerator::new(&problem, relax, vs, width, ns);
    let mut solver = BBSolver::new(&problem, ddg, MaxUB, FromLongestPath);
    solver.verbosity = 3;
    let (val, sln) = solver.maximize();
    println!("{:?}", sln);
    val
}

#[test]
fn debug() {
    assert_eq!(solve("debug.wcnf"), 24);
}
#[test]
fn debug2() {
    assert_eq!(solve("debug2.wcnf"), 13);
}

#[test]
fn pass() {
    assert_eq!(solve("pass.wcnf"), 54);
}
#[test]
fn tautology() {
    assert_eq!(solve("tautology.wcnf"), 7);
}
#[test]
fn unit() {
    assert_eq!(solve("unit.wcnf"), 6);
}
#[test]
fn negative_wt() {
    assert_eq!(solve("negative_wt.wcnf"), 4258);
}


/*
#[test]
fn frb10_6_1() {
    assert_eq!(solve("frb10-6-1.wcnf"), 37037);
}
#[test]
fn frb10_6_2() {
    assert_eq!(solve("frb10-6-2.wcnf"), 38196);
}
#[test]
fn frb10_6_3() {
    assert_eq!(solve("frb10-6-3.wcnf"), 36671);
}
#[test]
fn frb10_6_4() {
    assert_eq!(solve("frb10-6-4.wcnf"), 38298);
}
*/