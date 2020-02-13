extern crate rust_mdd_solver;
extern crate structopt;

use structopt::StructOpt;

use rust_mdd_solver::examples::max2sat::main::max2sat;
use rust_mdd_solver::examples::mcp::main::mcp;
use rust_mdd_solver::examples::misp::main::misp;

fn main() {
    let args = RustMddSolver::from_args();
    match args {
        RustMddSolver::Misp    {fname, verbose, width} => misp(&fname, verbose, width),
        RustMddSolver::Max2sat {fname, verbose, width} => max2sat(&fname, verbose, width),
        RustMddSolver::Mcp     {fname, verbose, width} => mcp(&fname, verbose, width),
    };
}

/// Solves hard combinatorial problems with bounded width MDDs
#[derive(StructOpt)]
enum RustMddSolver {
    /// Solve maximum weighted independent set problem from DIMACS (.clq) files
    Misp {
        /// Path to the DIMACS MSIP instance
        fname: String,
        /// Log the progression
        #[structopt(short, long, parse(from_occurrences))]
        verbose: u8,
        /// If specified, the max width allowed for any layer
        width: Option<usize>
    },
    /// Solve weighed max2sat from DIMACS (.wcnf) files
    Max2sat {
        /// Path to the DIMACS MAX2SAT instance
        fname: String,
        /// Log the progression
        #[structopt(short, long, parse(from_occurrences))]
        verbose: u8,
        /// If specified, the max width allowed for any layer
        width: Option<usize>
    },
    /// Solve max cut from a DIMACS graph files
    Mcp {
        /// Path to the graph instance
        fname: String,
        /// Log the progression
        #[structopt(short, long, parse(from_occurrences))]
        verbose: u8,
        /// If specified, the max width allowed for any layer
        width: Option<usize>
    }
}