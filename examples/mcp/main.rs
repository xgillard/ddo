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

use std::fs::File;
use std::time::SystemTime;

use structopt::StructOpt;

use ddo::abstraction::solver::Solver;
use ddo::implementation::mdd::config::mdd_builder;
use ddo::implementation::solver::parallel::ParallelSolver;

use crate::relax::McpRelax;

pub mod graph;
pub mod model;
pub mod relax;

#[derive(StructOpt)]
pub struct Args {
    pub fname: String,
    #[structopt(name="threads", short, long)]
    threads: Option<usize>,
}


fn mcp(fname: &str, threads: Option<usize>) -> isize {
    let threads   = threads.unwrap_or_else(num_cpus::get);
    let problem   = File::open(fname).expect("file not found").into();
    let relax     = McpRelax::new(&problem);
    let mdd       = mdd_builder(&problem, relax).into_deep();
    let mut solver= ParallelSolver::customized(mdd, 2, threads);

    let start = SystemTime::now();
    let opt   = solver.maximize().0;
    let end   = SystemTime::now();
    println!("Optimum {} computed in {:?} with {} threads", opt, end.duration_since(start).unwrap(), threads);
    opt
}

fn main() {
    let args  = Args::from_args();
    let value = mcp(&args.fname, args.threads);

    println!("best value = {}", value);
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::mcp;

    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("examples/tests/resources/mcp/")
            .join(id)
    }

    fn solve_id(id: &str) -> isize {
        let fname = locate(id);
        //mcp(fname.to_str().unwrap(), 2, Some(1))
        //mcp(fname.to_str().unwrap(), 30_000_000, Some(1))
        //mcp(fname.to_str().unwrap(), 1_000, Some(1))
        //mcp(fname.to_str().unwrap(), 500, Some(1))
        //mcp(fname.to_str().unwrap(), 30, Some(1))
        //mcp(fname.to_str().unwrap(), 3, Some(1))
        mcp(fname.to_str().unwrap(), Some(1))
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
}