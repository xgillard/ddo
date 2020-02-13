use crate::core::abstraction::dp::Problem;
use crate::core::common::{Variable, VarSet, Domain, Decision};
use std::cmp::min;
use std::io::{BufRead, Lines, BufReader, Read};
use regex::Regex;
use std::fs::File;

#[derive(Debug)]
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

#[derive(Debug)]
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

    fn initial_value(&self) -> i32 {
        0
    }

    fn domain_of<'a>(&self, state: &'a KnapsackState, var: Variable) -> Domain<'a> {
        let item       = &self.data[var.id()];
        let amount_max = min(item.quantity, state.capacity / item.weight) as i32;
        Domain::Range(0..amount_max + 1)
    }

    fn transition(&self, state: &KnapsackState, _vars: &VarSet, d: Decision) -> KnapsackState {
        let item = &self.data[d.variable.id()];

        let mut next = state.clone();
        next.capacity -= item.weight * d.value as usize;
        next.free_vars.remove(d.variable);

        next
    }

    fn transition_cost(&self, _state: &KnapsackState, _vars: &VarSet, d: Decision) -> i32 {
        let item = &self.data[d.variable.id()];

        item.profit as i32 * d.value
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

            if line.starts_with("c") {
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