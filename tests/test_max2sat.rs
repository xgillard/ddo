#![cfg(test)]
extern crate rust_mdd_solver;

use std::fs::File;
use std::path::PathBuf;

use rust_mdd_solver::core::abstraction::solver::Solver;
use rust_mdd_solver::core::implementation::bb_solver::BBSolver;
use rust_mdd_solver::core::implementation::flat_mdd::FlatMDDGenerator;
use rust_mdd_solver::core::implementation::heuristics::{NbUnassigned, FromLongestPath, MinLP, FixedWidth};
use rust_mdd_solver::examples::max2sat::heuristics::{Max2SatOrder, MinRank};
use rust_mdd_solver::examples::max2sat::model::{Max2Sat, State};
use rust_mdd_solver::examples::max2sat::relax::Max2SatRelax;
use rust_mdd_solver::core::utils::Func;
use rust_mdd_solver::core::abstraction::mdd::{MDDGenerator, Node};
use rust_mdd_solver::core::abstraction::dp::ProblemOps;
use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Less, Greater};
use rust_mdd_solver::core::implementation::pooled_mdd::PooledMDDGenerator;

/// This method simply loads a resource into a problem instance to solve
fn instance(id: &str) -> Max2Sat {
    let location = PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("tests/resources/max2sat/")
        .join(id);

    File::open(location).expect("File not found").into()
}

/// This is the function we will use to actually solve an instance to completion
/// and check that the optimal value it identifies corresponds to the expected
/// value.
fn solve(id: &str) -> i32 {
    let problem      = instance(id);
    let relax        = Max2SatRelax::new(&problem);
    let width        = NbUnassigned;
    let vs           = Max2SatOrder::new(&problem);
    let ns           = MinRank;
    let bo           = MinLP;//MaxUB;//Func(max2sat_ub_order);
    let vars         = FromLongestPath;

    let ddg          = PooledMDDGenerator::new(&problem, relax, vs, width, ns);
    let mut solver   = BBSolver::new(&problem, ddg, bo, vars);
    solver.verbosity = 3;
    let (val,_sln)   = solver.maximize();
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


fn xxx(x: &Node<State>, y: &Node<State>) -> Ordering {
    x.cmp(y).then_with(|| {
        let xstate = &x.state;
        let ystate = &y.state;
        for i in 0..xstate.0.len() {
            let xbi = xstate.0[i];
            let ybi = ystate.0[i];
            let disc= xbi - ybi;
            if disc != 0 {
                return if disc < 0 {Less} else {Greater};
            }
        }
        Equal
    })
}
#[test]
fn debug_ordering() {
    let id           = "frb10-6-1.wcnf";
    let problem      = instance(id);
    let relax        = Max2SatRelax::new(&problem);
    let width        = FixedWidth(100);//NbUnassigned;
    let vs           = Max2SatOrder::new(&problem);
    let ns           = Func(xxx);//MinRank;
    //let bo           = Func(max2sat_ub_order);
    //let vars         = FromLongestPath;

    let vars         = problem.all_vars();
    let root         = problem.root_node();

    let mut ddg      = FlatMDDGenerator::new(&problem, relax, vs, width, ns);

    ddg.relaxed(vars, &root, 0);
}

#[test]
fn frb10_6_1() {
    assert_eq!(solve("frb10-6-1.wcnf"), 37037);
}
#[ignore] #[test]
fn frb10_6_2() {
    assert_eq!(solve("frb10-6-2.wcnf"), 38196);
}
#[ignore] #[test]
fn frb10_6_3() {
    assert_eq!(solve("frb10-6-3.wcnf"), 36671);
}
#[ignore] #[test]
fn frb10_6_4() {
    assert_eq!(solve("frb10-6-4.wcnf"), 38298);
}
