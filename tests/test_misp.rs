#![cfg(test)]
extern crate rust_mdd_solver;

use std::fs::File;
use std::path::PathBuf;

use rust_mdd_solver::core::abstraction::solver::Solver;
use rust_mdd_solver::core::implementation::bb_solver::BBSolver;
use rust_mdd_solver::core::implementation::heuristics::{FixedWidth, MinLP, NaturalOrder};
use rust_mdd_solver::core::implementation::pooled_mdd::PooledMDDGenerator;
use rust_mdd_solver::core::utils::{RefFunc, Func};
use rust_mdd_solver::examples::misp::heuristics::{misp_ub_order, vars_from_misp_state};
use rust_mdd_solver::examples::misp::model::Misp;
use rust_mdd_solver::examples::misp::relax::MispRelax;

/// This method simply loads a resource into a problem instance to solve
fn instance(id: &str) -> Misp {
    let location = PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("tests/resources/misp/")
        .join(id);

    File::open(location).expect("File not found").into()
}

/// This is the function we will use to actually solve an instance to completion
/// and check that the optimal value it identifies corresponds to the expected
/// value.
fn solve(id: &str) -> i32 {
    let misp       = instance(id);
    let relax      = MispRelax::new(&misp);
    let width      = FixedWidth(100);
    let vs         = NaturalOrder;
    let ns         = MinLP;

    let ddg        = PooledMDDGenerator::new(&misp, relax, vs, width, ns);
    let mut solver = BBSolver::new(&misp, ddg, Func(misp_ub_order), RefFunc(vars_from_misp_state));
    let (val,_sln) = solver.maximize();

    val
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
#[ignore] #[test]
fn brock200_4() {
    assert_eq!(solve("brock200_4.clq"), 17);
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
fn c_fat500_1() {
    assert_eq!(solve("c-fat500-1.clq"), 14);
}
#[test]
fn c_fat500_2() {
    assert_eq!(solve("c-fat500-2.clq"), 26);
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
#[ignore] #[test]
fn hamming8_2() {
    assert_eq!(solve("hamming8-2.clq"), 128);
}
#[ignore] #[test]
fn hamming8_4() {
    assert_eq!(solve("hamming8-4.clq"), 16);
}
#[ignore] #[test]
fn hamming10_4() {
    assert_eq!(solve("hamming10-4.clq"), 40);
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
fn keller4() {
    assert_eq!(solve("keller4.clq"), 11);
}
#[ignore] #[test]
fn keller5() {
    assert_eq!(solve("keller5.clq"), 27);
}

#[test]
fn mann_a9() {
    assert_eq!(solve("MANN_a9.clq"), 16);
}
#[ignore] #[test]
fn mann_a27() {
    assert_eq!(solve("MANN_a27.clq"), 126);
}
#[ignore] #[test]
fn mann_a45() {
    assert_eq!(solve("MANN_a45.clq"), 315);
}

#[test]
fn p_hat300_1() {
    assert_eq!(solve("p_hat300-1.clq"), 8);
}
#[ignore] #[test]
fn p_hat300_2() {
    assert_eq!(solve("p_hat300-2.clq"), 25);
}
#[ignore] #[test]
fn p_hat300_3() {
    assert_eq!(solve("p_hat300-3.clq"), 36);
}
#[ignore] #[test]
fn p_hat700_1() {
    assert_eq!(solve("p_hat700-1.clq"), 11);
}
#[ignore] #[test]
fn p_hat700_2() {
    assert_eq!(solve("p_hat700-2.clq"), 44);
}
#[ignore] #[test]
fn p_hat700_3() {
    assert_eq!(solve("p_hat700-3.clq"), 62);
}
#[ignore] #[test]
fn p_hat1500_1() {
    assert_eq!(solve("p_hat1500-1.clq"), 12);
}
#[ignore] #[test]
fn p_hat1500_2() {
    assert_eq!(solve("p_hat1500-2.clq"), 65);
}
#[ignore] #[test]
fn p_hat1500_3() {
    assert_eq!(solve("p_hat1500-3.clq"), 94);
}
