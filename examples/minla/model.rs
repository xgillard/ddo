use std::hash::{Hash, Hasher};
use std::ops::Not;

use bitset_fixed::BitSet;

use ddo::common::{BitSetIter, Decision, Domain, Variable, VarSet};
use ddo::abstraction::dp::Problem;

#[derive(Debug, Clone)]
pub struct State {
    pub free : BitSet,
    pub cut  : Vec<isize>
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
    pub edges : Vec<(isize,usize,usize)>
}
impl Minla {
    pub fn new(g : Vec<Vec<isize>>) -> Minla {
        let mut edges = vec![];
        for (i, is) in g.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                if i < j {
                    edges.push((*js, i, j));
                }
            }
        }
        edges.sort_unstable();
        Minla {
            g,
            edges
        }
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
            cut: vec![0; self.nb_vars()]
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

            for j in BitSetIter::new(&result.free) {
                result.cut[j] += self.g[i][j];
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