extern crate ddo;
extern crate structopt;

use structopt::StructOpt;

use ddo::examples::max2sat::main::max2sat;
use ddo::examples::mcp::main::mcp;
use ddo::examples::misp::main::misp;
use ddo::examples::knapsack::main::knapsack;

fn main() {
    let args = RustMddSolver::from_args();
    match args {
        RustMddSolver::Max2sat {fname, verbose, width} => max2sat(&fname, verbose, width),
        RustMddSolver::Mcp     {fname, verbose, width} => mcp(&fname, verbose, width),
        RustMddSolver::Misp    {fname, verbose, width} => misp(&fname, verbose, width),
        RustMddSolver::Knapsack{fname, verbose, width} => knapsack(&fname, verbose, width),
    };
}

/// Solves hard combinatorial problems with bounded width MDDs
#[derive(StructOpt)]
enum RustMddSolver {
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
    },
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
    /// Solve a knapsack problem.
    Knapsack {
        /// Path to the knapsack instance file.
        ///
        /// The format of one such instance file should be the following:
        ///  - The header line describes the sack and should look as follows: `<sack_capacity> <nb_items>`
        ///  - Then, subsequent lines each describe an item. They should look like `<id> <profit> <weight> <quantity>`
        ///
        /// # Example
        /// The file with the following content describes a knapsack problem
        /// where the sack has a capacity of 23 and may be filled with 3 different
        /// type of items.
        ///
        /// One unit of the first item (id = 1) brings 10 of profit, and costs 5
        /// of capacity. The user may only place at most two such objects in the sack.
        ///
        /// One unit of the 2nd item (id = 2) brings 7 of profit, and costs 6
        /// of capacity. The user may only place at most one such item in the sack.
        ///
        /// One unit of the 3rd item (id = 3) brings 9 of profit, and costs 1
        //  of capacity. The user may only place at most one such item in the sack.
        /// ```
        /// 23 3
        /// 1 10 5 2
        /// 2 7  6 1
        /// 3 9  1 1
        /// ```
        ///
        fname: String,
        /// Log the progression
        #[structopt(short, long, parse(from_occurrences))]
        verbose: u8,
        /// If specified, the max width allowed for any layer
        width: Option<usize>
    }
}