
use std::time::Duration;

use ddo::{Problem, Variable, Decision, DecisionCallback};
use pyo3::{prelude::*, types:: PyTuple};
use crate::state::JsspState;


#[derive(Debug, Clone)]
pub struct Jssp{
    pub instance: JsspInstance,
    pub initial: JsspState,
    pub inference_duration: Duration
}

#[derive(Debug, Clone)]
pub struct JsspInstance{
    pub processing: Vec<Vec<usize>>,
    pub machine_order: Vec<Vec<usize>>,
    pub njobs: usize,
    pub nmchs: usize,
    pub ml_model: Option<PyObject>,
}

impl Jssp{
    pub fn new(instance:JsspInstance)-> Self{
        let initial_state = JsspState::new(&instance);
        Self{instance, initial:initial_state, inference_duration:Duration::new(0,0)}
    }

    pub fn state_to_model_input(&self,state:&JsspState)-> (Py<PyAny>,Py<PyAny>) {

        pyo3::prepare_freethreaded_python();
            Python::with_gil(|py|  {
                (
                    [("job_times",self.instance.processing.clone()),
                    ("precedence",self.instance.machine_order.clone()),
                    ].into_py(py),

                    [("machine_utilization", state.machine_utilization.clone()),
                    ("job_early_start_time",state.job_early_start_time.clone()),
                    ("job_state",state.job_state_def.clone()),
                    ].into_py(py)
                )
                
            })
            
    }

    pub fn infer_from_model(&self, problem_input:&Py<pyo3::PyAny>, state_input:&Py<pyo3::PyAny>) -> Result<i32,pyo3::PyErr> {
            pyo3::prepare_freethreaded_python();
            Python::with_gil(|py| {
                let inference_engine = PyModule::import(py,"jssp")?;
                // fetch the model
                let model: &PyObject = match &self.instance.ml_model{
                    Some(x) => x,
                    None => panic!("No ml model exists to be called")
                };
                let model_call = PyTuple::new(py,&[model,&self.instance.njobs.into_py(py),&self.instance.nmchs.into_py(py),&problem_input,&state_input]);
                let inference_result: i32 = inference_engine.getattr("infer")?.call1(model_call,)?.extract()?;
                Ok(inference_result)
            })
    }
}

impl Problem for Jssp{
    /// The DP model of the problem manipulates a state which is user-defined.
    /// Any type implementing Problem must thus specify the type of its state.
    type State = JsspState;
    /// Any problem bears on a number of variable $x_0, x_1, x_2, ... , x_{n-1}$
    /// This method returns the value of the number $n$
    fn nb_variables(&self) -> usize{
        self.instance.njobs * self.instance.nmchs
    }
    /// This method returns the initial state of the problem (the state of $r$).
    fn initial_state(&self) -> Self::State{
        self.initial.clone()
    }
    /// This method returns the initial value $v_r$ of the problem
    fn initial_value(&self) -> isize{
        0
    }
    /// This method is an implementation of the transition function mentioned
    /// in the mathematical model of a DP formulation for some problem.
    fn transition(&self, state: &Self::State, decision: Decision) -> Self::State{
        // get the job id from the decision
        let (jobid,opid) = ((decision.value/self.instance.nmchs as isize) as usize, 
            (decision.value%self.instance.nmchs as isize) as usize);
        
        let responsible_mch = self.instance.machine_order[jobid][opid];
        let job_finish = state.machine_utilization[responsible_mch].max(state.job_early_start_time[jobid]) + 
                                            self.instance.processing[jobid][opid];
        

        let mut new_state = state.clone();
        new_state.machine_utilization[responsible_mch] = job_finish;
        new_state.job_early_start_time[jobid] = job_finish;
        // maybe should always be ahead of def
        if opid == new_state.job_state_def[jobid]{
            new_state.job_state_def[jobid] += 1;
        }
        new_state.job_state_maybe[jobid] = (new_state.job_state_maybe[jobid]+1).min(self.instance.nmchs).max(opid); 

        if new_state.job_state_def[jobid] >= self.instance.nmchs{
            //set job done bit
            new_state.job_done.add_inplace(jobid);
            //set one less job to process
            new_state.jobs_to_process -= 1;
        }
        new_state.partial_makespan =  new_state.partial_makespan.max(job_finish);
        
        new_state

    }
    /// This method is an implementation of the transition cost function mentioned
    /// in the mathematical model of a DP formulation for some problem.
    fn transition_cost(&self, source: &Self::State, dest: &Self::State, _decision: Decision) -> isize{
        // negate cost because solver maximises
        source.partial_makespan as isize - dest.partial_makespan as isize
    }
    /// Any problem needs to be able to specify an ordering on the variables
    /// in order to decide which variable should be assigned next. This choice
    /// is an **heuristic** choice. The variable ordering does not need to be
    /// fixed either. It may simply depend on the depth of the next layer or
    /// on the nodes that constitute it. These nodes are made accessible to this
    /// method as an iterator.
    fn next_variable(&self, depth: usize, _next_layer: &mut dyn Iterator<Item = &Self::State>)
        -> Option<Variable>{
            // variables processed in fixed order, they represent positions in the schedule
            // no more variables = terminal
            if depth < self.nb_variables() {
                Some(Variable(depth))
            } else {
                None
            }
        }
    /// This method calls the function `f` for any value in the domain of 
    /// variable `var` when in state `state`.  The function `f` is a function
    /// (callback, closure, ..) that accepts one decision.
    fn for_each_in_domain(&self, var: Variable, state: &Self::State, f: &mut dyn DecisionCallback){
        for jobid in 0..self.instance.njobs{
            if state.job_state_def[jobid] < self.instance.nmchs {
                // apply definitely next
                let def_next_op = (jobid*self.instance.nmchs) + state.job_state_def[jobid];
                f.apply(Decision{variable: var, value: def_next_op as isize});
                // job done updated based on def so maybe can keep growing -- only apply till max number of machines
                for op in state.job_state_def[jobid]+1..state.job_state_maybe[jobid]{
                    let next_op = (jobid*self.instance.nmchs) + op;
                    f.apply(Decision{variable: var, value: next_op as isize});
                }
            }
        }
    }
    /// This method returns false iff this node can be moved forward to the next
    /// layer without making any decision about the variable `_var`.
    /// When that is the case, a default decision is to be assumed about the 
    /// variable. Implementing this method is only ever useful if you intend to 
    /// compile a decision diagram that comprises long arcs.
    fn is_impacted_by(&self, _var: Variable, _state: &Self::State) -> bool {
        true
    }
    // This function can be implemented by a problem that provides some kind of learned oracle to provide decision support
    fn perform_ml_decision_inference(&self, var: Variable, state:&Self::State) -> Option<Decision>{
        
        let mut decision: Option<Decision> = None;
        
        if self.instance.ml_model.is_some(){
            let (problem_input,state_input) = self.state_to_model_input(state);
            let output = self.infer_from_model(&problem_input,&state_input);
            // println!("output is {:?}", output);
            let next_job_id = output.ok();
            
            match next_job_id{
                Some(id) => {
                    decision = Some(Decision{variable: var, value: (id as isize * self.instance.nmchs as isize) + state.job_state_def[id as usize] as isize})
                },
                None => {}
            }         
        }

        decision
    }

}