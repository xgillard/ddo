// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::cmp::min;
use std::cmp::Ordering::Equal;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};

use regex::Regex;
use bitset_fixed::BitSet;

use ddo::{Decision, Domain, Variable, VarSet, Problem, Relaxation, VariableHeuristic};

#[derive(Debug, Clone, Copy)]
pub struct ItemData {
    pub id      : usize,
    pub profit  : usize,
    pub weight  : usize,
    pub quantity: usize
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct KnapsackState {
    pub capacity : usize,
    pub free_vars: VarSet
}

#[derive(Debug, Clone)]
pub struct Knapsack {
    pub capacity: usize,
    pub data    : Vec<ItemData>
}
impl Problem<KnapsackState> for Knapsack {
    fn nb_vars(&self) -> usize {
        self.data.len()
    }

    fn initial_state(&self) -> KnapsackState {
        KnapsackState{capacity: self.capacity, free_vars: self.all_vars()}
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn domain_of<'a>(&self, state: &'a KnapsackState, var: Variable) -> Domain<'a> {
        let item       = &self.data[var.id()];
        let amount_max = min(item.quantity, state.capacity / item.weight) as isize;
        (0..=amount_max).into()
    }

    fn transition(&self, state: &KnapsackState, _vars: &VarSet, d: Decision) -> KnapsackState {
        let item = &self.data[d.variable.id()];

        let mut next = state.clone();
        next.capacity -= item.weight * d.value as usize;
        next.free_vars.remove(d.variable);

        next
    }

    fn transition_cost(&self, _state: &KnapsackState, _vars: &VarSet, d: Decision) -> isize {
        let item = &self.data[d.variable.id()];

        item.profit as isize * d.value
    }
}
impl <B: BufRead> From<Lines<B>> for Knapsack {
    fn from(lines: Lines<B>) -> Knapsack {
        let sack = Regex::new(r"^(?P<capa>\d+)\s+(?P<nb_items>\d+)$").unwrap();
        let item = Regex::new(r"^(?P<item>\d+)\s+(?P<profit>\d+)\s+(?P<weight>\d+)\s+(?P<qty>\d+)$").unwrap();

        let mut knapsack = Knapsack {capacity: 0, data: vec![]};

        for line in lines {
            let line = line.unwrap();
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if line.starts_with('c') {
                // This is just a comment, it can be ignored
                continue;
            }

            if let Some(caps) = sack.captures(&line) {
                knapsack.capacity = caps["capa"].to_string().parse::<usize>().unwrap();

                let nb_items = caps["nb_items"].to_string().parse::<usize>().unwrap();
                knapsack.data.reserve_exact(nb_items);
                continue;
            }

            if let Some(caps) = item.captures(&line) {
                let id     = caps["item"].to_string().parse::<usize>().unwrap();
                let profit = caps["profit"].to_string().parse::<usize>().unwrap();
                let weight = caps["weight"].to_string().parse::<usize>().unwrap();
                let qty    = caps["qty"].to_string().parse::<usize>().unwrap();

                let data = ItemData {id, profit, weight, quantity: qty};
                knapsack.data.push(data);
            }
        }
        knapsack.data.sort_unstable_by_key(|i| i.id);
        knapsack
    }
}
impl <S: Read> From<BufReader<S>> for Knapsack {
    fn from(r: BufReader<S>) -> Knapsack {
        r.lines().into()
    }
}
impl From<File> for Knapsack {
    fn from(f: File) -> Knapsack {
        BufReader::new(f).into()
    }
}



#[derive(Clone)]
pub struct KnapsackOrder<'a> {
    pb: &'a Knapsack
}
impl <'a> KnapsackOrder<'a> {
    pub fn new(pb: &'a Knapsack) -> Self {
        KnapsackOrder{pb}
    }
}
impl VariableHeuristic<KnapsackState> for KnapsackOrder<'_> {
    fn next_var(&self,
                free_vars: &VarSet,
                _: &mut dyn Iterator<Item=&KnapsackState>,
                _: &mut dyn Iterator<Item=&KnapsackState>) -> Option<Variable>
    {
        free_vars.iter().max_by(|x, y| {
            let ix = &self.pb.data[x.id()];
            let kx = ix.profit as f32 / ix.weight as f32;

            let iy = &self.pb.data[y.id()];
            let ky = iy.profit as f32 / iy.weight as f32;

            kx.partial_cmp(&ky).unwrap_or(Equal)
        })
    }
}


#[derive(Debug, Clone)]
pub struct KnapsackRelax<'a> {
    pb: &'a Knapsack
}
impl <'a> KnapsackRelax<'a> {
    pub fn new(pb: &'a Knapsack) -> Self { KnapsackRelax {pb} }
}
impl Relaxation<KnapsackState> for KnapsackRelax<'_> {
    fn merge_states(&self, states: &mut dyn Iterator<Item=&KnapsackState>) -> KnapsackState {
        let mut capacity  = 0;
        let mut free_vars = BitSet::new(self.pb.nb_vars());

        for state in states {
            free_vars |= &state.free_vars.0;
            capacity   = capacity.max(state.capacity);
        }

        KnapsackState {capacity, free_vars: VarSet(free_vars)}
    }
    fn relax_edge(&self, _: &KnapsackState, _: &KnapsackState, _: &KnapsackState, _: Decision, cost: isize) -> isize {
        cost
    }
    fn estimate(&self, state: &KnapsackState) -> isize {
        state.free_vars.iter().map(|v| {
            let item      = &self.pb.data[v.id()];
            let max_amout = state.capacity / item.weight;
            max_amout * item.profit
        }).sum::<usize>() as isize
    }
}
