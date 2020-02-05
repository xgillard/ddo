#![cfg(test)]
extern crate rust_mdd_solver;

use std::path::PathBuf;
use std::fs::File;
use rust_mdd_solver::core::implementation::bb_solver::BBSolver;
use rust_mdd_solver::examples::misp::model::Misp;
use std::rc::Rc;
use rust_mdd_solver::examples::misp::relax::MispRelax;
use rust_mdd_solver::core::implementation::heuristics::{FixedWidth, NaturalOrder, MinLP, MaxUB};
use rust_mdd_solver::core::implementation::pooled_mdd::PooledMDDGenerator;
use rust_mdd_solver::examples::misp::heuristics::vars_from_misp_state;
use rust_mdd_solver::core::utils::Func;
use rust_mdd_solver::core::abstraction::solver::Solver;

fn instance(id: &str) -> Misp {
    let location = PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("tests/resources/misp/")
        .join(id);

    File::open(location.to_owned()).expect("File not found").into()
}

fn solve(id: &str) -> i32 {
    let misp       = Rc::new(instance(id));
    let relax      = MispRelax::new(Rc::clone(&misp));
    let width      = FixedWidth(100);
    let vs         = NaturalOrder;
    let ns         = MinLP;

    let ddg        = PooledMDDGenerator::new(Rc::clone(&misp), relax, vs, width, ns);
    let mut solver = BBSolver::new(Rc::clone(&misp), ddg, MaxUB, Func(vars_from_misp_state));
    let (val,_sln) = solver.maximize();

    val
}

#[test]
fn keller4() {
    assert_eq!(solve("keller4.clq"), 11);
}

#[test]
fn c_fat200_1() {
    assert_eq!(solve("c-fat200-1.clq"), 12);
}

#[test]
fn c_fat200_2() {
    assert_eq!(solve("c-fat200-2.clq"), 24);
}

#[test]
fn c_fat200_5() {
    assert_eq!(solve("c-fat200-5.clq"), 58);
}

#[test]
fn hamming6_2() {
    assert_eq!(solve("hamming6-2.clq"), 32);
}

#[test]
fn hamming6_4() {
    assert_eq!(solve("hamming6-4.clq"), 4);
}

#[test]
fn johnson8_2_4() {
    assert_eq!(solve("johnson8-2-4.clq"), 4);
}

#[test]
fn johnson8_4_4() {
    assert_eq!(solve("johnson8-4-4.clq"), 14);
}

#[test]
fn mann_a9() {
    assert_eq!(solve("MANN_a9.clq"), 16);
}

/// This test takes > 60s to solve on my machine
#[ignore] #[test]
fn brock200_1() {
    assert_eq!(solve("brock200_1.clq"), 21);
}
#[test]
fn brock200_2() {
    assert_eq!(solve("brock200_2.clq"), 12);
}
#[test]
fn brock200_3() {
    assert_eq!(solve("brock200_3.clq"), 15);
}
#[test]
fn brock200_4() {
    assert_eq!(solve("brock200_4.clq"), 17);
}
#[test]
fn c_fat500_1() {
    assert_eq!(solve("c-fat500-1.clq"), 14);
}
#[test]
fn c_fat500_2() {
    assert_eq!(solve("c-fat500-2.clq"), 26);
}
#[test]
fn hamming8_2() {
    assert_eq!(solve("hamming8-2.clq"), 128);
}
#[test]
fn hamming8_4() {
    assert_eq!(solve("hamming8-4.clq"), 16);
}