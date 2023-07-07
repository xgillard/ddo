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

//! This module is meant to tests the correctness of our maxcut problem example

use std::{path::PathBuf, fs::File};

use ddo::*;

use crate::{graph::Graph, model::{Mcp, McpRanking}, relax::McpRelax};


fn locate(id: &str) -> PathBuf {
    PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("../resources/mcp/")
        .join(id)
}

pub fn solve_id(id: &str) -> isize {
    let fname = locate(id);
    let fname = fname.to_str();
    let fname = fname.unwrap();
    
    let graph = Graph::from(File::open(fname).expect("could not open file"));
    let problem = Mcp::from(graph);
    let relaxation = McpRelax::new(&problem);
    let ranking = McpRanking;

    let width = NbUnassignedWidth(problem.nb_variables());
    let dominance = EmptyDominanceChecker::default();
    let cutoff = NoCutoff;
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    // This solver compile DD that allow the definition of long arcs spanning over several layers.
    let mut solver = DefaultSolver::new(
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


#[test]
fn mcp_n30_p01_000() {
    assert_eq!(solve_id("mcp_n30_p0.1_000.mcp"), 13);
}
#[test]
fn mcp_n30_p01_001() {
    assert_eq!(solve_id("mcp_n30_p0.1_001.mcp"), 18);
}
#[test]
fn mcp_n30_p01_002() {
    assert_eq!(solve_id("mcp_n30_p0.1_002.mcp"), 15);
}
#[test]
fn mcp_n30_p01_003() {
    assert_eq!(solve_id("mcp_n30_p0.1_003.mcp"), 19);
}
#[test]
fn mcp_n30_p01_004() {
    assert_eq!(solve_id("mcp_n30_p0.1_004.mcp"), 16);
}
#[test]
fn mcp_n30_p01_005() {
    assert_eq!(solve_id("mcp_n30_p0.1_005.mcp"), 19);
}
#[test]
fn mcp_n30_p01_006() {
    assert_eq!(solve_id("mcp_n30_p0.1_006.mcp"), 12);
}
#[test]
fn mcp_n30_p01_007() {
    assert_eq!(solve_id("mcp_n30_p0.1_007.mcp"), 18);
}
#[test]
fn mcp_n30_p01_008() {
    assert_eq!(solve_id("mcp_n30_p0.1_008.mcp"), 20);
}
#[test]
fn mcp_n30_p01_009() {
    assert_eq!(solve_id("mcp_n30_p0.1_009.mcp"), 22);
}