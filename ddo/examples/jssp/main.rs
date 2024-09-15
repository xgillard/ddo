use std::{fs::{self, File}, io::{BufRead, BufReader}, num::ParseIntError, time::{Duration, Instant}};
use pyo3::{prelude::*, types::PyTuple};

use clap::Parser;
use ddo::*;
use model::Jssp;
use relax::JsspRelax;
use state::JsspRanking;

mod state;
mod model;
mod relax;

use crate::model::JsspInstance;

use serde_json::json;


/// This structure uses `clap-derive` annotations and define the arguments that can
/// be passed on to the executable solver.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the instance file
    fname: String,
    /// The number of concurrent threads
    #[clap(short, long, default_value = "8")]
    threads: usize,
    /// The maximum amount of time you would like this solver to run
    #[clap(short, long)]
    duration: Option<u64>,
    /// The maximum number of nodes per layer
    #[clap(short, long)]
    width: Option<usize>,
    /// Option to use ML model for restriction builidng
    /// Path to pb file for model
    #[clap(short, long, default_value = "")]
    model: String,
    /// Whether or not to write output to json file
    #[clap(short, long, action)]
    json_output: bool,
    /// Path to write output file to
    #[clap(short='x', long, default_value = "")]
    outfolder: String,
}

/// An utility function to return a cutoff heuristic that can either be a time budget policy
/// (if timeout is fixed) or no cutoff policy.
fn cutoff(timeout: Option<u64>) -> Box<dyn Cutoff + Send + Sync> {
    if let Some(t) = timeout {
        Box::new(TimeBudget::new(Duration::from_secs(t)))
    } else {
        Box::new(NoCutoff)
    }
}

/// A utility function to return an max width heuristic that can either be a fixed width
/// policy (if w is fixed) or an adaptive policy returning the number of unassigned variables
/// in the overall problem.
fn max_width<T>(nb_vars: usize, w: Option<usize>) -> Box<dyn WidthHeuristic<T> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
        // Box::new(CustomJsspWidth(w))
    } else {
        Box::new(NbUnassignedWidth(nb_vars))
    }
}

// Implement own width heuristic type
pub struct CustomJsspWidth(pub usize);
impl <JsspState> WidthHeuristic<JsspState> for CustomJsspWidth {
    fn max_width(&self, _x: &SubProblem<JsspState>) -> usize {
        self.0
    }
    fn restriction_width(&self, _x: &SubProblem<JsspState>) -> usize {
        1
    }
    fn relaxation_width(&self, x: &SubProblem<JsspState>) -> usize {
        self.max_width(x)
    }
}

fn main() {
    let args = Args::parse();
    let instance = read_instance(&args).unwrap();
    let problem = Jssp::new(instance);
    let relaxation = JsspRelax::new(&problem);
    let ranking = JsspRanking;

    let width = max_width(problem.nb_variables(), args.width);
    let dominance = EmptyDominanceChecker::default();
    let cutoff = cutoff(args.duration);
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

    // let mut solver = DefaultCachingSolver::custom(
    //     &problem, 
    //     &relaxation, 
    //     &ranking, 
    //     width.as_ref(),
    //     &dominance,
    //     cutoff.as_ref(), 
    //     &mut fringe,
    //     args.threads
    // );

    let mut solver = SeqCachingSolverLel::custom(
        &problem, 
        &relaxation, 
        &ranking, 
        width.as_ref(),
        &dominance,
        cutoff.as_ref(), 
        &mut fringe
    );

    let start = Instant::now();
    let Completion{ is_exact, best_value } = solver.maximize();
    
    let duration = start.elapsed();
    let upper_bound = -solver.best_upper_bound();
    let lower_bound = -solver.best_lower_bound();
    let gap = solver.gap();
    let best_solution = solver.best_solution().unwrap_or_default()
        .iter().map(|d| (d.value/problem.instance.nmchs as isize, d.value%problem.instance.nmchs as isize,)).collect::<Vec<(isize,isize)>>();
    
    // println!("Duration:   {:.3} seconds", duration.as_secs_f32());
    // println!("Objective:  {}",            best_value.map(|x| -x).unwrap_or(-1));
    // println!("Upper Bnd:  {}",            upper_bound);
    // println!("Lower Bnd:  {}",            lower_bound);
    // println!("Gap:        {:.3}",         gap);
    // println!("Aborted:    {}",            !is_exact);
    // println!("Solution:   {:?}",          best_solution);

    let result = json!({
        "Duration": format!("{:.3}", duration.as_secs_f32()),
        "Objective":  format!("{}", -best_value.unwrap_or(-1)),
        "Upper Bnd":  format!("{}", upper_bound),
        "Lower Bnd":  format!("{}", lower_bound),
        "Gap":        format!("{:.3}", gap),
        "Solver":     format!("{}", if problem.instance.ml_model.is_some() {"DD-RL"} else {"DD"}),
        "Aborted":    format!("{}", !is_exact),
        "Solution":   format!("{:?}", best_solution)
    });

    println!("{}", result.to_string());
    if args.json_output{
        let mut outfile = args.outfolder.to_owned();
        let instance_name = if let Some(x) = &args.fname.split("/").collect::<Vec<_>>().last() {x} else {"_"};
        outfile.push_str(&instance_name);
        outfile.push_str(".json");
        fs::write(outfile,result.to_string()).expect("unable to write json");
    }
}

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// jssp instance from file. There can be io errors (file unavailable ?), format error
/// (e.g. the file is not an instance but contains the text of your next paper), 
/// or parse int errors (which are actually a variant of the format error since it tells 
/// you that the parser expected an integer number but got ... something else).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// There was an io related error
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    /// The parser expected to read something that was an integer but got some garbage
    #[error("parse int {0}")]
    ParseInt(#[from] ParseIntError),
}

/// This function is used to read a jssp instance from file. It returns either a
/// sop instance if everything went on well or an error describing the problem.
fn read_instance(args:&Args) -> Result<JsspInstance, Error> {
    let f = File::open(&args.fname)?;
    let f = BufReader::new(f);
    
    let lines = f.lines();

    let mut lc= 0;

    let mut njobs = 0;
    let mut nmchs= 0;
    let mut machine_order = vec![vec![]];
    let mut processing = vec![vec![]];

    for line in lines {
        let line = line.unwrap();
        let line = line.trim();

        // First line is the number of jobs and machines
        if lc == 0 { 
            let params = line.split_whitespace().map(str::to_string).collect::<Vec<_>>();
            njobs  = params[0].parse::<usize>().unwrap();
            nmchs = params[1].parse::<usize>().unwrap();
            // reserve size for machine and processing
            machine_order = vec!(vec!(0;nmchs);njobs);
            processing = vec!(vec!(0;nmchs);njobs);
        }
        // The next 'nb_nodes' lines represent the distances matrix
        else if (1..=njobs).contains(&lc) {
            let jobid = (lc - 1) as usize;
            let mch_proc_pairs = line.split_whitespace().map(str::to_string).collect::<Vec<_>>();
            for (mch_index, str_index) in (0..2*nmchs).step_by(2).enumerate(){
                machine_order[jobid][mch_index] = mch_proc_pairs[str_index].parse::<usize>().unwrap();
                processing[jobid][mch_index] = mch_proc_pairs[str_index+1].parse::<usize>().unwrap();
            } 
        }        
        lc += 1;
    }

    //load model
    let model = if !args.model.is_empty(){
                            pyo3::prepare_freethreaded_python();
                            Python::with_gil(|py| {
                                let inference_engine = PyModule::import(py,"jssp")?;
                                let load_call = PyTuple::new(py,&[&args.model,&njobs.to_string(),&nmchs.to_string()]);
                                let model: PyObject = inference_engine.getattr("load_model")?.call1(load_call,)?.extract()?;
                                Ok(model)
                                })
                            } else{
                                Err(PyErr::new::<PyAny, _>("Could not load model"))
                            };
    // println!("{:?}",model);
    let ml_model:Option<PyObject> = model.ok();           
    Ok(JsspInstance{njobs: njobs, nmchs: nmchs, processing: processing, machine_order: machine_order, ml_model: ml_model})
}
