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


//! Ce module implémente mon idée 2: on fait un truc simple ou on garde le nombre
//! d'items, mais on fait essentiellement la meme chose que dans la toute première
//! relaxation: on fait du padding avec un élément virtuel.

use std::cmp::min;

use bitset_fixed::BitSet;

use ddo::{
    Problem, 
    Relaxation,
    Variable,
    Domain, 
    Decision,
    VarSet,
    Matrix,
    BitSetIter,
};

use crate::model::PSP;

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct VirtualItem {
    pub remaining_qty: usize,
    pub merged_into  : BitSet,
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct State2 {
    pub remaining: Vec<usize>,
    pub last_dec : usize,
    pub time_step: usize,
    pub v_item   : Option<VirtualItem>
}

impl State2 {
    fn v_data(&self) -> &VirtualItem {
        self.v_item.as_ref().expect("BUG: accessing non existing virtual item")
    }
    fn total_quantity(&self) -> usize {
        let hard = self.remaining.iter().cloned().sum::<usize>();
        let virt = self.v_item.as_ref().map_or(0, |v| v.remaining_qty);
        hard + virt
    }
}

#[derive(Clone)]
pub struct Pb2 {
    pub v_item           : usize,
    pub nb_periods       : usize,
    pub nb_items         : usize,
    pub nb_orders        : usize,
    pub changeover_cost  : Matrix<usize>,
    pub stocking_cost    : Vec<usize>,
    pub deadlines        : Vec<Vec<usize>>,
    pub demands_per_item : Vec<usize>,
}

impl Pb2 {
    pub fn new(psp: PSP) -> Pb2 {
        Pb2 {
            v_item           : psp.nb_items,
            nb_periods       : psp.nb_periods,
            nb_items         : psp.nb_items,
            nb_orders        : psp.nb_orders,
            changeover_cost  : psp.changeover_cost,
            stocking_cost    : psp.stocking_cost,
            deadlines        : psp.deadlines,
            demands_per_item : psp.demands_per_item
        }
    }

    fn component_deadline(&self, component: usize, qty: usize) -> usize {
        let deadlines = &self.deadlines[component];
        let offset    = qty.min(deadlines.len());
        deadlines[offset - 1]
    }

    fn latest_schedule_time(&self, state: &State2, item: usize) -> usize {
        if item == self.v_item {
            let v_item = state.v_data();

            BitSetIter::new(&v_item.merged_into)
                .map(|comp| self.component_deadline(comp, v_item.remaining_qty))
                .max()
                .expect("BUG: (schedule time) no merged member in the virtual item info")

        } else {
            self.deadlines[item][state.remaining[item] - 1]
        }
    }

    fn earliest_delivery_time(&self, state: &State2, item: usize, qty: usize, now: usize) -> usize {
        if item == self.v_item {
            let v_item = state.v_data();

            BitSetIter::new(&v_item.merged_into)
                .map(|comp| self.component_deadline(comp, qty))
                .filter(|time| *time >= now)
                .min()
                .unwrap_or(now)

        } else {
            self.deadlines[item][qty - 1]
        }
    }

    fn can_schedule(&self, state: &State2, item: usize, now: usize) -> bool {
        let remaining =
            if item == self.v_item {
                state.v_data().remaining_qty
            } else {
                state.remaining[item]
            };

        remaining > 0 && self.latest_schedule_time(state, item) >= now
    }
    fn backward_changeover_cost(&self, state: &State2, item: usize) -> usize {
        let v_item = self.v_item;
        if state.last_dec == item {
            0
        } else if state.last_dec == self.v_item {
            // Here, we may not use the state's virtual item (even if one is
            // present). Indeed, the previous 'virtual item' might have a different
            // meaning than the current one (different set of 'merged_into'
            // components).
            let nb_items    = self.nb_items;
            let mut min_cost= usize::max_value();
            for _ in 0..nb_items {
                min_cost = min(min_cost, self.changeover_cost[(item, state.last_dec)]);
            }
            min_cost
        } else if item == v_item {
            let last_d = state.last_dec;
            let v_data = state.v_data();
            BitSetIter::new(&v_data.merged_into)
                .map(|comp| self.changeover_cost[(comp, last_d)])
                .min()
                .expect("BUG: (changeover cost 2) no merged member in the virtual item info")
        } else {
            self.changeover_cost[(item, state.last_dec)]
        }
    }
    fn time_until_deadline(&self, state: &State2, item: usize, qty: usize, now: usize) -> usize {
        let edt = self.earliest_delivery_time(state, item, qty, now);
        if edt > now {
            edt - now
        } else {
            0
        }
    }
    fn stocking_cost_of(&self, state: &State2, item: usize) -> usize {
        if item == self.v_item {
            BitSetIter::new(&state.v_data().merged_into)
                .map(|comp| self.stocking_cost[comp])
                .min()
                .expect("BUG: (stocking cost) no merged member in the virtual item info")
        } else {
            self.stocking_cost[item]
        }
    }
    fn stocking_cost_until_deadline(&self, state: &State2, item: usize, qty: usize, now: usize) -> usize {
        let duration = self.time_until_deadline(state, item, qty, now);
        self.stocking_cost_of(state, item) * duration
    }
}
impl Problem<State2> for Pb2 {
    fn nb_vars(&self) -> usize {
        self.nb_periods
    }

    fn initial_state(&self) -> State2 {
        State2 {
            last_dec : self.nb_items, // this is dummy !
            time_step: self.nb_periods,
            remaining: self.demands_per_item.clone(),
            v_item   : None
        }
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn domain_of<'a>(&self, state: &'a State2, var: Variable) -> Domain<'a> {
        let now = var.id();
        let mut candidates = vec![];

        for item in 0..self.nb_items {
            if self.can_schedule(state, item, now) {
                candidates.push(item as isize);
            }
        }

        if state.v_item.is_some() && self.can_schedule(state, self.v_item, now) {
            candidates.push(self.v_item as isize);
        }

        candidates.into()
    }

    fn transition(&self, state: &State2, _vars: &VarSet, d: Decision) -> State2 {
        let mut result = state.clone();

        // remember the last decision
        let v = d.value as usize;
        result.last_dec = v;
        result.time_step-= 1;

        if v == self.v_item {
            let data = result.v_item.as_mut().expect("BUG: (transition) branching on non existing virtual item");
            data.remaining_qty -= 1;
        } else {
            result.remaining[v] -= 1;
        }

        result
    }

    fn transition_cost(&self, state: &State2, _vars: &VarSet, d: Decision) -> isize {
        let item = d.value as usize;
        let now = d.variable.id();
        let qty = if item == self.v_item {
            state.v_data().remaining_qty
        } else {
            state.remaining[item]
        };
        let changeover = self.backward_changeover_cost(state, item);
        let stocking   = self.stocking_cost_until_deadline(state, item, qty, now);
        -((changeover + stocking) as isize)
    }
}


#[derive(Clone)]
pub struct Rlx2<'a> {
    pub pb : &'a Pb2
}
/*
impl <'a> Rlx2<'a> {
    /// This function computes a greedy estimate that schedules all items
    /// in ascending stocking cost order
    fn greedy_estimate(&self, state: &State2) -> isize {
        let now = state.time_step;
        if now == 0 {
            0
        } else {
            let mut rem = vec![0; self.pb.nb_items + 1];

            for (i, qt) in state.remaining.iter().enumerate() {
                rem[i] = *qt;
            }
            if let Some(v) = &state.v_item {
                rem[self.pb.v_item] = v.remaining_qty;
            }

            let mut est = 0;

            for i in 0..now {
                let mut best_item = 0;
                let mut found = false;
                let mut best_cost = usize::max_value();
                for (j, qty) in rem.iter().enumerate() {
                    if *qty > 0 {
                        let cost = self.pb.stocking_cost_until_deadline(state, j, *qty, i);

                        if cost < best_cost {
                            found = true;
                            best_item = j;
                            best_cost = cost;
                        }
                    }
                }

                if found {
                    est += best_cost;
                    rem[best_item] -= 1;
                } else {
                    break;
                }
            }

            est as isize
        }
    }
}
*/

impl Relaxation<State2> for Rlx2<'_> {
    fn merge_states(&self, states: &mut dyn Iterator<Item=&State2>) -> State2 {
        let states     = states.collect::<Vec<&State2>>();
        let nb_items   = self.pb.nb_items;

        let mut virtual_itm= VirtualItem {
            remaining_qty: 0,
            merged_into  : BitSet::new(nb_items)
        };

        // Keep the minimum of each, and remember the items having a disagreement
        let mut was_set    = BitSet::new(nb_items);
        let mut remainders = vec![usize::max_value(); nb_items];
        for state in states.iter() {
            for (item, rem) in remainders.iter_mut().enumerate() {
                let at_node = state.remaining[item];

                if was_set[item] && at_node != *rem {
                    virtual_itm.merged_into.set(item, true);
                }

                *rem = min(*rem, at_node);
                was_set.set(item, true);
            }
        }

        // If some node among the merged ones held information wrt virtual item
        // that info must be merged with the current VirtualItem under construction
        for state in states.iter() {
            if let Some(virt) = &state.v_item {
                virtual_itm.merged_into |= &virt.merged_into;
            }
        }

        // The total quantity of items to produce must be kept.
        let now            = states[0].time_step;
        let total_quantity = states[0].total_quantity();
        let current_qty    = remainders.iter().cloned().sum::<usize>();
        virtual_itm.remaining_qty = total_quantity - current_qty;


        let virtual_itm = if virtual_itm.remaining_qty > 0 {
            Some(virtual_itm)
        } else {
            None
        };

        // Because the signs have been flipped, the shortest path, is the one
        // with the maximum lp_len
        State2 {
            last_dec : states[0].last_dec,
            time_step: now,
            remaining: remainders,
            v_item   : virtual_itm
        }
    }

    fn relax_edge(&self, _: &State2, _: &State2, _: &State2, _: Decision, cost: isize) -> isize {
        cost
    }

    fn estimate  (&self, _state  : &State2) -> isize {
        // alt: self.greedy_estimate(state)
        0
    }
}
