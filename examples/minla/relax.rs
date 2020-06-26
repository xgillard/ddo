use bitset_fixed::BitSet;

use ddo::abstraction::dp::{Problem, Relaxation};
use ddo::common::{BitSetIter, Decision};

use crate::model::{Minla, State};

#[derive(Debug, Clone)]
pub struct MinlaRelax<'a> {
    pb: &'a Minla
}

impl <'a> MinlaRelax<'a> {
    pub fn new(pb : &'a Minla) -> MinlaRelax<'a> {
        MinlaRelax{pb}
    }

    fn edge_ub(&self, free : &BitSet) -> isize {
        let n_free = free.count_ones();

        // edge weights multiplied by optimistic distance
        let mut lb = 0;
        let mut index = 0;
        for k in 1..n_free {
            for _l in 0..k {
                while !free[self.pb.edges[index].1] || !free[self.pb.edges[index].2] {
                    index += 1;
                }
                lb += (n_free - k) as isize * self.pb.edges[index].0;
                index += 1;
            }
        }

        lb
    }

    fn cut_ub(&self, vertices : &BitSet, state : &State) -> isize {
        let mut cuts = vec![];

        // gather cut weights to free vertices
        for i in BitSetIter::new(&vertices) {
            cuts.push(state.cut[i]);
        }

        // sort decreasingly
        cuts.sort_by(|a, b| b.cmp(a));

        // cut weights in optimistic order
        let mut cut_lb = 0;
        for (dist, cut) in cuts.into_iter().enumerate() {
            cut_lb += dist as isize * cut;
        }

        cut_lb
    }

    fn ub(&self, vertices : &BitSet, state : &State) -> isize {
        - self.edge_ub(&vertices) - self.cut_ub(&vertices, &state)
    }
}
impl <'a> Relaxation<State> for MinlaRelax<'a> {
    fn merge_states(&self, _states: &mut dyn Iterator<Item=&State>) -> State {
        State {
            free: BitSet::new(0),
            cut: vec![0; self.pb.nb_vars()]
        }
    }

    fn relax_edge(&self, _src: &State, dst: &State, _relaxed: &State, _decision: Decision, cost: isize) -> isize {
        cost + self.ub(&dst.free, &dst)
    }

    fn estimate(&self, state: &State) -> isize {
        self.ub(&state.free, &state)
    }
}