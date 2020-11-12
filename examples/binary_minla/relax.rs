use bitset_fixed::BitSet;

use ddo::abstraction::dp::{Problem, Relaxation};
use ddo::common::Decision;

use crate::model::{Minla, State};

use std::cmp::Reverse;

#[derive(Debug, Clone)]
pub struct MinlaRelax<'a> {
    pb: &'a Minla
}

impl <'a> MinlaRelax<'a> {
    pub fn new(pb : &'a Minla) -> MinlaRelax<'a> {
        MinlaRelax{pb}
    }

    fn edge_lb(&self, n : isize, m : isize) -> isize {
        let mut edge_lb = 0;
        let mut edges = m;

        for k in 1..n {
            if edges <= 0 {
                break;
            } else {
                edge_lb += edges;
                edges -= n - k;
            }
        }

        edge_lb
    }

    /* --- This method is never used ---------------------------------------
    fn degree_lb(&self, vertices : &BitSet, state : &State) -> isize {
        let mut deg_lb = 0;

        for k in BitSetIter::new(&vertices) {
            let d = self.pb.deg[k] - state.cut[k];
            deg_lb += (d * d + 2 * d + d % 1) / 4; // FIXME: Any number mod 1 is 0
        }

        deg_lb / 2
    }
    ----------------------------------------------------------------------- */

    fn cut_lb(&self, state : &State) -> isize {
        let mut cuts = state.cut.clone();

        // sort decreasingly
        cuts.sort_by_key(|&b| Reverse(b));

        // cut weights in optimistic order
        let mut cut_lb = 0;
        for (dist, cut) in cuts.into_iter().enumerate() {
            if cut == 0 {
                break;
            }
            cut_lb += dist as isize * cut;
        }

        cut_lb
    }

    fn ub(&self, vertices : &BitSet, state : &State) -> isize {
        let n = vertices.count_ones() as isize;
        if n == 0 {
            0
        } else {
            - self.cut_lb(state) - self.edge_lb(n, state.m)
        }
    }
}
impl <'a> Relaxation<State> for MinlaRelax<'a> {
    fn merge_states(&self, _states: &mut dyn Iterator<Item=&State>) -> State {
        State {
            free: BitSet::new(0),
            cut: vec![0; self.pb.nb_vars()],
            m: 0
        }
    }

    fn relax_edge(&self, _src: &State, dst: &State, _relaxed: &State, _decision: Decision, cost: isize) -> isize {
        cost + self.ub(&dst.free, &dst)
    }

    fn estimate(&self, state: &State) -> isize {
        self.ub(&state.free, &state)
    }
}