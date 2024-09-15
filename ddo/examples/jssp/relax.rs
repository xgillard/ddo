use std::cmp;

use crate::{model::Jssp, state::JsspState};
use ddo::{Decision, Relaxation};

#[derive(Clone)]
pub struct JsspRelax<'a> {
    pb: &'a Jssp,
}

impl<'a> JsspRelax<'a>{
    pub fn new(pb: &'a Jssp) -> Self{
        Self{pb}
    }
}
// TODO think of a stronger relaxation here, upper bound is often very weak
impl Relaxation for JsspRelax<'_> {
    /// Similar to the DP model of the problem it relaxes, a relaxation operates
    /// on a set of states (the same as the problem). 
    type State = JsspState;

    /// This method implements the merge operation: it combines several `states`
    /// and yields a new state which is supposed to stand for all the other
    /// merged states. In the mathematical model, this operation was denoted
    /// with the $\oplus$ operator.
    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State{

        let mut merged_state = JsspState::new(&self.pb.instance);

        for state in states{
            merged_state.machine_utilization = merged_state.machine_utilization.iter().zip(state.machine_utilization.iter()).map(|(&b, &v)| cmp::min(b,v)).collect::<Vec<_>>();
            merged_state.job_early_start_time = merged_state.job_early_start_time.iter().zip(state.job_early_start_time.iter()).map(|(&b, &v)| cmp::min(b,v)).collect::<Vec<_>>();
            merged_state.job_state_def = merged_state.job_state_def.iter().zip(state.job_state_def.iter()).map(|(&b, &v)| cmp::min(b,v)).collect::<Vec<_>>();
            merged_state.job_state_maybe = merged_state.job_state_maybe.iter().zip(state.job_state_maybe.iter()).map(|(&b, &v)| cmp::max(b,v)).collect::<Vec<_>>();
            merged_state.jobs_to_process = merged_state.jobs_to_process.min(state.jobs_to_process);
            merged_state.partial_makespan = merged_state.partial_makespan.min(state.partial_makespan);
            merged_state.job_done = merged_state.job_done;
        }
    
        merged_state
    }
    
    /// This method relaxes the cost associated to a particular decision. It
    /// is called for any arc labeled `decision` whose weight needs to be 
    /// adjusted because it is redirected from connecting `src` with `dst` to 
    /// connecting `src` with `new`. In the mathematical model, this operation
    /// is denoted by the operator $\Gamma$.
    fn relax(
        &self, source: &Self::State, _dest: &Self::State, new: &Self::State,
        _decision: Decision, _cost: isize,
    ) -> isize{
        (source.partial_makespan - new.partial_makespan) as isize
    }

    /// Returns a very rough estimation (upper bound) of the optimal value that 
    /// could be reached if state were the initial state
    fn fast_upper_bound(&self, _state: &Self::State) -> isize {
        // TODO implement actual rough upper bound 
        isize::MAX
    }
}