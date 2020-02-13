use crate::core::abstraction::heuristics::VariableHeuristic;
use crate::examples::knapsack::model::{KnapsackState, Knapsack};
use crate::core::common::{Variable, VarSet};
use crate::core::abstraction::mdd::Layer;
use std::cmp::Ordering::Equal;

pub struct KnapsackOrder<'a> {
    pb: &'a Knapsack
}
impl <'a> KnapsackOrder<'a> {
    pub fn new(pb: &'a Knapsack) -> Self {
        KnapsackOrder{pb}
    }
}
impl VariableHeuristic<KnapsackState> for KnapsackOrder<'_> {
    fn next_var<'a>(&self, free_vars: &'a VarSet, _c: Layer<'a, KnapsackState>, _n: Layer<'a, KnapsackState>) -> Option<Variable> {
        free_vars.iter().max_by(|x, y| {
            let ix = &self.pb.data[x.id()];
            let kx = ix.profit as f32 / ix.weight as f32;

            let iy = &self.pb.data[y.id()];
            let ky = iy.profit as f32 / iy.weight as f32;

            kx.partial_cmp(&ky).unwrap_or(Equal)
        })
    }
}