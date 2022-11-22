use std::{sync::Arc, fs::File, io::{BufWriter, Write}};
use anyhow::{Result, Ok};
use structopt::StructOpt;
use ddo::{Problem, CompilationInput, NoCutoff, SubProblem, DecisionDiagram, CompilationType};

use ddo_viz::{viz_mdd::VizMdd, VizConfigBuilder, Visualisable};

mod model;
use model::*;

#[derive(StructOpt)]
struct Args {
    #[structopt(short, long, default_value = "3")]
    max_width: usize,
    #[structopt(short, long, default_value = "exact")]
    comp_type: String,
    #[structopt(short="V", long)]
    show_value: bool,
    #[structopt(short="L", long)]
    show_locb: bool,
    #[structopt(short="R", long)]
    show_rub: bool,
    #[structopt(short="M", long)]
    show_relaxed: bool,
    #[structopt(short="X", long)]
    show_restricted: bool,
    #[structopt(short, long, default_value = "out.dot")]
    output_file: String,
}

impl Args {
    fn compilation_type(&self) -> CompilationType {
        match self.comp_type.as_str() {
            "exact"       => CompilationType::Exact,
            "relaxed"     => CompilationType::Relaxed,
            "restricted"  => CompilationType::Restricted,
            _             => CompilationType::Relaxed,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::from_args();

    // Define the example instance we want to solve
    let inst = UnboundedKP {
        capa  : 100,
        weight: &[20, 20, 25, 30],
        profit: &[15, 15, 40, 35],
    };
    let relax = KPRelax;
    let ranking = KPRank;

    // Create the configuration for compiling the mdd we wish to 
    // visualize (this is the exact same as what is done when compiling
    // a DD to actually solve the problem)
    let input = CompilationInput {
        comp_type: args.compilation_type(),
        problem: &inst,
        relaxation: &relax,
        ranking: &ranking,
        cutoff: &NoCutoff,
        max_width: args.max_width,
        best_lb: isize::MIN,
        residual: SubProblem {
            state: Arc::new(inst.initial_state()),
            value: 0,
            path: vec![],
            ub: isize::MAX,
        },
    };

    // Effectively compile the decision diagram
    let mut dd = VizMdd::default();
    _ = dd.compile(&input);

    // And then visualize it
    let x = File::create(&args.output_file)?;
    let mut x = BufWriter::new(x);

    let cfg = VizConfigBuilder::default()
        .show_value(args.show_value)    
        .show_locb(args.show_locb)
        .show_rub(args.show_rub)
        .show_merged(args.show_relaxed)
        .show_restricted(args.show_restricted)
        .build()?;
    x.write_all( dd.visualisation().as_graphviz(&cfg).as_bytes())?;

    Ok(())
}
