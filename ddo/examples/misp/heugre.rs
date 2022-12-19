//! This is the module where I implement the greedy heuristic discussed with
//! Mohsen NAFAR from uni Bielefeld.

use std::sync::Arc;

use bit_set::BitSet;
use dashmap::DashMap;
use ddo::*;
use fxhash::{FxBuildHasher};
use crate::Misp;

pub type HeugreCache = DashMap<BitSet, isize, FxBuildHasher>;

#[derive(Clone)]
pub struct HeuGre<'a> {
    pb: &'a Misp,
    sorted_nodes: Vec<usize>,
    cache: Arc<HeugreCache>,
}
impl StateRanking for HeuGre<'_> {
    type State = BitSet;
    fn compare(&self, va: isize, a: &Self::State, vb: isize, b: &Self::State) -> std::cmp::Ordering {
        let ha = va + self.heugre(a);
        let hb = vb + self.heugre(b);

        ha.cmp(&hb).reverse()
    }
}
impl <'a> HeuGre<'a> {
    pub fn new(pb: &'a Misp, cache: Arc<HeugreCache>) -> Self {
        let mut sorted_nodes = (0..pb.nb_variables()).collect::<Vec<_>>();
        sorted_nodes.sort_unstable_by_key(|k| -pb.weight[*k]);

        Self { pb, sorted_nodes, cache }
    }
    fn heugre(&self, bs: &BitSet) -> isize {
        if let Some(heugre) = self.cache.get(bs) {
            *heugre
        } else {
            let heugre = self.compute(bs);
            self.cache.insert(bs.clone(), heugre);
            heugre
        }
    }
    fn compute(&self, bs: &BitSet) -> isize {
        let mut temp = BitSet::with_capacity(self.pb.nb_variables());
        let mut value = 0;

        for i in self.sorted_nodes.iter().copied() {
            if !bs.contains(i) { continue; }

            let mut can_add = true;
            for j in 0..i {
                if !can_add {break;   }
                if !temp.contains(j) {continue;}
                // the graph in the instance is the complement of the original graph
                if !self.pb.neighbors[j].contains(i) {
                    can_add = false;
                }
            }

            if can_add {
                temp.insert(i);
                value  += self.pb.weight[i];
            }
        }
        value
    }
}
