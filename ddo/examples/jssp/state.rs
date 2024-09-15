use ddo::StateRanking;
use smallbitset::Set256;

use crate::model::JsspInstance;

type BitSet = Set256;


#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct JsspState{
    // earliest availability of machine
    pub machine_utilization: Vec<usize>,
    // earliest possible start time of job
    pub job_early_start_time: Vec<usize>,
    // last scheduled operation index for each job
    pub job_state_def: Vec<usize>,
    // possible last scheduled operation index for each job
    pub job_state_maybe: Vec<usize>,
    // how many jobs still have pending operations - can figure this out from job_done bitset actually
    pub jobs_to_process: usize,
    // partial makespan a this state
    pub partial_makespan: usize,
    // if the job is done or not
    pub job_done: BitSet

}

impl JsspState {
    pub fn new(instance:&JsspInstance) -> Self {
        JsspState{
            machine_utilization: vec![0;instance.nmchs],
            job_early_start_time: vec![0;instance.njobs],
            job_state_def: vec![0;instance.njobs],
            job_state_maybe: vec![0;instance.njobs],
            jobs_to_process: instance.njobs,
            partial_makespan: 0,
            job_done: BitSet::empty()
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct JsspRanking;

impl StateRanking for JsspRanking {
    type State = JsspState;

    // flip the comparison because the solver maximizes
    // so negate to make the smaller makespan better
    fn compare(&self, sa: &Self::State, sb: &Self::State) -> std::cmp::Ordering {
        let a = -1 * sa.partial_makespan as isize;
        let b = -1 * sb.partial_makespan as isize;
        a.cmp(&b)
    }
}