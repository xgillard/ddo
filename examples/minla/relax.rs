use bitset_fixed::BitSet;
use std::cmp::min;
use std::ops::Not;

use ddo::core::common::{Node, NodeInfo};
use ddo::core::abstraction::dp::{Problem, Relaxation};

use crate::model::{Minla, State};
use ddo::core::utils::BitSetIter;

#[derive(Debug, Clone)]
pub struct MinlaRelax<'a> {
    pb: &'a Minla
}

impl <'a> MinlaRelax<'a> {
    pub fn new(pb : &'a Minla) -> MinlaRelax<'a> {
        MinlaRelax{pb}
    }
}
impl <'a> Relaxation<State> for MinlaRelax<'a> {
    fn merge_nodes(&self, nodes: &[Node<State>]) -> Node<State> {
        let mut state = State {
            free: BitSet::new(self.pb.nb_vars()).not(),
            cut: vec![std::i32::MAX; self.pb.nb_vars()]
        };
        let mut best  = &nodes[0].info;

        for node in nodes.iter() {
            state.free &= &node.state.free;
            for i in BitSetIter::new(&state.free) {
                state.cut[i] = min(state.cut[i],
                                   node.state.cut[i])
            }

            if node.info.lp_len > best.lp_len {
                best = &node.info;
            }
        }

        let mut info = best.clone();
        info.is_exact = false;
        Node {state, info}
    }

    fn estimate_ub(&self, state: &State, info: &NodeInfo) -> i32 {
        let n_free = state.free.count_ones() as usize;

        if n_free <= 1 {
            return info.lp_len
        }

        let mut cuts = Vec::new();
        let mut weights = Vec::new();

        // gather cut weights to free vertices
        // and weights of edges between 2 free vertices
        for i in BitSetIter::new(&state.free) {
            for j in BitSetIter::new(&state.free) {
                if i < j {
                    weights.push(self.pb.g[i][j] as usize);
                }
            }
            cuts.push(state.cut[i]);
        }

        // sort decreasingly
        weights.sort_unstable_by(|a, b| b.cmp(a));
        cuts.sort_unstable_by(|a, b| b.cmp(a));

        // edge weights multiplied by optimistic distance
        let mut free_lb = 0;
        let mut number = n_free - 1;
        let mut index = 0;
        while index < weights.len() {
            let end = index + number;
            while index < end {
                free_lb += (n_free - number) * weights[index];
                index += 1
            }
            number -= 1
        }

        // cut weights in optimistic order
        let mut cut_lb = 0;
        let mut dist = 0;
        for cut in cuts {
            cut_lb += dist * cut;
            dist += 1;
        }

        info.lp_len - free_lb as i32 - cut_lb
    }
}