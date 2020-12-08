use std::hash::{Hash, Hasher};
use std::ops::Not;

use bitset_fixed::BitSet;

use ddo::{BitSetIter, Decision, Domain, Variable, VarSet, Problem};

#[derive(Debug, Clone)]
pub struct State {
    pub free : BitSet,
    pub cut  : Vec<isize>,
    pub m    : isize
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
    pub g : Vec<Vec<isize>>,
    pub deg : Vec<isize>,
    pub m : isize
}
impl Minla {
    pub fn new(g : Vec<Vec<isize>>) -> Minla {
        let n = g.len();
        let mut deg = vec![0; n];
        let mut m = 0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    m += g[i][j];
                    deg[i] += g[i][j];
                }
            }
        }
        m /= 2;
        Minla { g, deg, m }
    }

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
            cut: vec![0; self.nb_vars()],
            m: self.m
        }
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn domain_of<'a>(&self, state: &'a State, var: Variable) -> Domain<'a> {
        if var.0 == 0 {
            Domain::from(0..((self.nb_vars()-1) as isize))
        } else if state.free.count_ones() == 0 { // relaxed node with empty free vertices intersection
            Domain::from(vec![self.no_vertex() as isize])
        } else {
            Domain::from(&state.free)
        }
    }

    fn transition(&self, state: &State, _vars: &VarSet, d: Decision) -> State {
        let i = d.value as usize;

        let mut result = state.clone();

        if i != self.no_vertex() {
            result.free.set(i as usize, false);
            result.cut[i] = 0;

            for j in BitSetIter::new(&result.free) {
                result.cut[j] += self.g[i][j];
                result.m -= self.g[i][j];
            }
        }

        result
    }

    fn transition_cost(&self, state: &State, _vars: &VarSet, d: Decision) -> isize {
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
