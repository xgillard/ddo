use bitset_fixed::BitSet;
use std::cmp::min;
use std::ops::{Not};

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

    fn edge_ub(&self, n_free : usize, weights: &mut Vec<usize>) -> i32 {
        // sort decreasingly
        weights.sort_unstable_by(|a, b| b.cmp(a));

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

        free_lb as i32
    }

    fn cut_ub(&self, cuts : &mut Vec<i32>) -> i32 {
        // sort decreasingly
        cuts.sort_unstable_by(|a, b| b.cmp(a));

        // cut weights in optimistic order
        let mut cut_lb = 0;
        for (dist, cut) in cuts.into_iter().enumerate() {
            cut_lb += dist as i32 * (*cut);
        }

        cut_lb
    }
}
impl <'a> Relaxation<State> for MinlaRelax<'a> {
    fn merge_nodes(&self, nodes: &[Node<State>]) -> Node<State> {
        let mut state = State {
            free: BitSet::new(self.pb.nb_vars()).not(),
            cut: vec![std::i32::MAX; self.pb.nb_vars()]
        };
        let mut best  = &nodes[0].info;
        let mut best_adjusted_lp_len = std::i32::MIN;

        for node in nodes.iter() {
            state.free &= &node.state.free;
        }

        for node in nodes.iter() {
            for i in BitSetIter::new(&state.free) {
                state.cut[i] = min(state.cut[i],
                                   node.state.cut[i])
            }

            // compute set of vertices being deleted from this state
            let mut delete = node.state.free.clone();
            delete ^= &state.free;

            let mut cuts = Vec::new();
            let mut weights = Vec::new();

            // gather cut weights to deleted vertices
            // and weights of edges between 2 deleted vertices
            for i in BitSetIter::new(&delete) {
                for j in BitSetIter::new(&delete) {
                    if i < j {
                        weights.push(self.pb.g[i][j] as usize);
                    }
                }
                cuts.push(node.state.cut[i]);
            }

            let adjusted_lp_len = node.info.lp_len
                - self.edge_ub(delete.count_ones() as usize, weights.as_mut())
                - self.cut_ub(cuts.as_mut());
            if adjusted_lp_len > best_adjusted_lp_len {
                best = &node.info;
                best_adjusted_lp_len = adjusted_lp_len;
            }
        }

        let mut info = best.clone();
        info.is_exact = false;
        info.lp_len = best_adjusted_lp_len;
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

        info.lp_len - self.edge_ub(n_free, weights.as_mut()) - self.cut_ub(cuts.as_mut())
    }
}