use bitset_fixed::BitSet;
use std::ops::Not;

use ddo::core::abstraction::dp::Problem;
use ddo::core::common::{Variable, Domain, VarSet, Decision};
use std::hash::{Hasher, Hash};
use ddo::core::utils::BitSetIter;

#[derive(Debug, Clone)]
pub struct State {
    pub free : BitSet,
    pub cut  : Vec<i32>
}
impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.free == other.free
    }
}
impl Eq for State {}
impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.free.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct Minla {
    pub g : Vec<Vec<i32>>
}
impl Minla {
    fn no_vertex(&self) -> usize {
        self.nb_vars()
    }
}
impl Problem<State> for Minla {
    fn nb_vars(&self) -> usize {
        self.g.len()
    }

    fn initial_state(&self) -> State {
        State {
            free: BitSet::new(self.nb_vars()).not(),
            cut: vec![0; self.nb_vars()]
        }
    }

    fn initial_value(&self) -> i32 {
        0
    }

    fn domain_of<'a>(&self, state: &'a State, var: Variable) -> Domain<'a> {
        if var.0 == 0 {
            Domain::from(0..((self.nb_vars()-1) as i32))
        } else if state.free.count_ones() == 0 { // relaxed node with empty free vertices intersection
            Domain::from(vec![self.no_vertex() as i32])
        } else {
            Domain::from(&state.free)
        }
    }

    fn transition(&self, state: &State, _vars: &VarSet, d: Decision) -> State {
        let i = d.value as usize;

        let mut free = state.free.clone();
        let mut cut = state.cut.clone();

        if i != self.no_vertex() {
            free.set(i, false);

            for j in BitSetIter::new(&free) {
                cut[j] += self.g[i][j];
            }
        }

        State { free, cut }
    }

    fn transition_cost(&self, state: &State, _vars: &VarSet, d: Decision) -> i32 {
        let i = d.value as usize;

        let mut cost = 0;
        if i != self.no_vertex() {
            for j in BitSetIter::new(&state.free) {
                if i != j {
                    cost += state.cut[j] + self.g[i][j]
                }
            }
        }

        - cost
    }
}