//! This is the module where I implement the greedy heuristic discussed with
//! Mohsen NAFAR from uni Bielefeld.

use std::sync::Arc;

use dashmap::DashMap;
use ddo::*;
use fxhash::{FxBuildHasher};
use crate::{Knapsack, KnapsackState};

pub type HeugreCache = DashMap<KnapsackState, isize, FxBuildHasher>;

#[derive(Clone)]
pub struct HeuGre<'a> {
    pb: &'a Knapsack,
    sorted_items: Vec<usize>,
    cache: Arc<HeugreCache>,
}
impl StateRanking for HeuGre<'_> {
    type State = KnapsackState;
    fn compare(&self, va: isize, a: &Self::State, vb: isize, b: &Self::State) -> std::cmp::Ordering {
        let ha = va + self.heugre(a);
        let hb = vb + self.heugre(b);

        ha.cmp(&hb).reverse()
    }
}
impl <'a> HeuGre<'a> {
    pub fn new(pb: &'a Knapsack, cache: Arc<HeugreCache>) -> Self {
        let mut sorted_items = (0..pb.nb_variables()).collect::<Vec<_>>();
        sorted_items.sort_unstable_by(|a, b| {
            let ra =  Self::ratio(pb, *a);
            let rb =  Self::ratio(pb, *b);
            ra.partial_cmp(&rb).unwrap()
        });

        Self { pb, sorted_items, cache }
    }
    fn ratio(pb: &Knapsack, i: usize) -> f32 {
        pb.profit[i] as f32 / pb.weight[i] as f32
    }
    fn heugre(&self, bs: &KnapsackState) -> isize {
        if let Some(heugre) = self.cache.get(bs) {
            *heugre
        } else {
            let heugre = self.compute(bs);
            self.cache.insert(bs.clone(), heugre);
            heugre
        }
    }
    fn compute(&self, bs: &KnapsackState) -> isize {
        let mut value = 0;
        let mut rem = bs.capacity;

        for i in self.sorted_items.iter().copied() {
            if i < bs.depth { continue; }

            if self.pb.weight[i] <= rem {
                rem -= self.pb.weight[i];
                value  += self.pb.profit[i];
            }
        }
        value as isize
    }
}
