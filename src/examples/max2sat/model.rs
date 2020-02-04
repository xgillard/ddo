use crate::examples::max2sat::instance::Weighed2Sat;
use crate::core::abstraction::dp::{Problem, Variable, VarSet, Decision};
use std::ops::{Index, IndexMut};
use std::cmp::{min, max};

const T  : i32      = 1;
const F  : i32      =-1;
const TF : [i32; 2] = [T, F];

const fn v (x: Variable) -> i32 { 1 + x.0 as i32}
const fn t (x: Variable) -> i32 { v(x) }
const fn f (x: Variable) -> i32 {-v(x) }
fn pos(x: i32) -> i32 { max(0, x) }


#[derive(Debug, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct State(pub Vec<i32>);

impl Index<Variable> for State {
    type Output = i32;

    fn index(&self, index: Variable) -> &i32 {
        self.0.get(index.0).unwrap()
    }
}
impl IndexMut<Variable> for State {
    fn index_mut(&mut self, index: Variable) -> &mut i32 {
        self.0.get_mut(index.0).unwrap()
    }
}

pub struct Max2Sat {
    pub inst: Weighed2Sat
}

impl Problem<State> for Max2Sat {

    fn nb_vars(&self) -> usize {
        self.inst.nb_vars
    }

    fn initial_state(&self) -> State {
        State(vec![0; self.nb_vars()])
    }

    fn initial_value(&self) -> i32 {
        // sum of all tautologies
        self.inst.weights.iter()
            .filter_map(|(clause, cost)| if clause.is_tautology() { Some(cost) } else {None} )
            .sum()
    }

    fn domain_of(&self, _state: &State, _var: Variable) -> &[i32] {
        &TF
    }

    fn transition(&self, state: &State, vars: &VarSet, d: Decision) -> State {
        let k = d.variable;
        let mut ret  = state.clone();

        ret[k] = 0;
        if d.value == F {
            for l in vars.iter() {
                ret[l] += self.inst.weight(t(k), t(l)) - self.inst.weight(t(k), f(l));
            }
        } else {
            for l in vars.iter() {
                ret[l] += self.inst.weight(f(k), t(l)) - self.inst.weight(f(k), f(l));
            }
        }
        ret
    }

    fn transition_cost(&self, state: &State, vars: &VarSet, d: Decision) -> i32 {
        let k = d.variable;
        if d.value == F {
            let res = pos(-state[k]);
            let mut sum = self.inst.weight(f(k), f(k)); // Weight if unit clause
            for l in vars.iter() {
                // Those that are satisfied by [[ k = F ]]
                let wff = self.inst.weight(f(k), f(l));
                let wft = self.inst.weight(f(k), t(l));
                // Those that actually depend on the truth value of `l`.
                let wtt = self.inst.weight(t(k), t(l));
                let wtf = self.inst.weight(t(k), f(l));

                sum += (wff + wft) + min(pos( state[l]) + wtt,
                                         pos(-state[l]) + wtf);
            }

            return res + sum;
        }
        if d.value == T {
            let res = pos(state[k]);
            let mut sum = self.inst.weight(t(k), t(k)); // Weight if unit clause
            for l in vars.iter() {
                // Those that are satisfied by [[ k = T ]]
                let wtt = self.inst.weight(t(k), t(l));
                let wtf = self.inst.weight(t(k), f(l));
                // Those that actually depend on the truth value of `l`.
                let wff = self.inst.weight(f(k), f(l));
                let wft = self.inst.weight(f(k), t(l));

                sum += (wtf + wtt) + min(pos( state[l]) + wft,
                                         pos(-state[l]) + wff);
            }

            return res + sum;
        }
        panic!("The decision value was neither T nor F");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_state() {
        let state = State(vec![1, 2, 3, 4]);

        assert_eq!(state[Variable(0)], 1);
        assert_eq!(state[Variable(1)], 2);
        assert_eq!(state[Variable(2)], 3);
        assert_eq!(state[Variable(3)], 4);
    }
    #[test]
    fn test_index_mut_state() {
        let mut state = State(vec![1, 2, 3, 4]);

        state[Variable(0)] = 42;
        state[Variable(1)] = 64;
        state[Variable(2)] = 16;
        state[Variable(3)] =  9;

        assert_eq!(state[Variable(0)], 42);
        assert_eq!(state[Variable(1)], 64);
        assert_eq!(state[Variable(2)], 16);
        assert_eq!(state[Variable(3)],  9);
    }
}