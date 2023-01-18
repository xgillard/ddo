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

//! This is the module where you will find the definition of the state and DP model
//! for the MAX2SAT problem. 
use std::cmp::{max, min, Ordering};
use std::ops::{Index, IndexMut};

use ddo::*;

use crate::data::Weighed2Sat;

/// A constant to represent the value TRUE being assigned to a given literal
const T: isize = 1;
/// A constant to represent the value FALSE being assigned to a given literal
const F: isize = -1;

/// An utility function to retrieve the canonical representation of some variable
pub const fn v(x: Variable) -> isize {
    1 + x.0 as isize
}
/// An utility function to retrieve the canonical representation of the positive
/// literal for some given variable
pub const fn t(x: Variable) -> isize {
    v(x)
}
/// An utility function to retrieve the canonical representation of the negative
/// literal for some given variable
pub const fn f(x: Variable) -> isize {
    -v(x)
}
/// An utility function to crop the value of x if it is less than 0.
/// Represents the notation $(x)^+$ in the paper
fn pos(x: isize) -> isize {
    max(0, x)
}

/// In our DP model, we consider a state that consists of the marginal benefit of 
/// assigning True to each variable. Additionally, we also consider the *depth* 
/// (number of assigned variables) as part of the state since it useful when it 
/// comes to determine the next variable to branch on.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct State {
    pub depth: usize,
    pub substates: Vec<isize>,
}
/// For the sake of convenience, a state can be indexd as if it were a plain vector
impl Index<Variable> for State {
    type Output = isize;

    fn index(&self, index: Variable) -> &isize {
        self.substates.get(index.0).unwrap()
    }
}
/// For the sake of convenience, a state can be mutated as if it were a plain vector
impl IndexMut<Variable> for State {
    fn index_mut(&mut self, index: Variable) -> &mut isize {
        self.substates.get_mut(index.0).unwrap()
    }
}
impl State {
    pub fn rank(&self) -> isize {
        self.substates.iter().map(|x| x.abs()).sum()
    }
}
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// This structure encapsulates the DP definition of the MAX2SAT problem.
/// This DP model is not trivial to understand, but it performs well and 
/// it is an exact translation of the model described in 
/// ``Discrete optimization with decision diagrams'' 
///   by Bergman, Cire, and Van Hoeve
///   in INFORMS Journal (2016)

#[derive(Debug, Clone)]
pub struct Max2Sat {
    pub nb_vars: usize,
    pub initial: isize,
    pub weights: Vec<isize>,
    pub sum_of_clause_weights: Vec<isize>,
    pub vars_by_sum_of_clause_weights: Vec<Variable>,
    // only used by the relaxation, but somehow it feels cleaner to store it here
    nk: Vec<isize>,
    estimates: Vec<isize>,
}

const fn idx(x: isize) -> usize {
    (x.abs() - 1) as usize
}
fn mk_lit(x: isize) -> usize {
    let sign = usize::from(x > 0);
    let abs = (x.abs() - 1) as usize;

    abs + abs + sign
}
impl Max2Sat {
    pub fn new(inst: Weighed2Sat) -> Max2Sat {
        let n = inst.nb_vars;

        let mut ret = Max2Sat {
            nb_vars: n,
            initial: 0,
            weights: vec![0; (2 * n) * (2 * n)],
            sum_of_clause_weights: vec![0; n],
            vars_by_sum_of_clause_weights: vec![Variable(0); n],
            //
            nk: vec![],
            estimates: vec![],
        };
        // compute the sum of clause weights
        for (clause, weight) in inst.weights.iter() {
            let of = ret.offset(clause.a, clause.b);
            ret.weights[of] = *weight;

            ret.sum_of_clause_weights[idx(clause.a)] += *weight;

            if !clause.is_unit() {
                ret.sum_of_clause_weights[idx(clause.b)] += *weight;
            }
            if clause.is_tautology() {
                ret.initial += *weight;
            }
        }
        // compute the variable ordering from worse to best
        let mut order = (0..n).map(Variable).collect::<Vec<Variable>>();
        order.sort_unstable_by_key(|v| ret.sum_of_clause_weights[v.0]);
        ret.vars_by_sum_of_clause_weights = order;

        let estimates = ret.precompute_estimates();
        let nk = ret.precompute_nks();
        ret.estimates = estimates;
        ret.nk = nk;

        ret
    }

    pub fn weight(&self, x: isize, y: isize) -> isize {
        self.weights[self.offset(x, y)]
    }

    fn offset(&self, x: isize, y: isize) -> usize {
        let a = x.min(y);
        let b = x.max(y);

        (mk_lit(a) * 2 * self.nb_vars) + mk_lit(b)
    }

    #[inline]
    fn varset(&self, state: &State) -> impl Iterator<Item = Variable> + '_ {
        let n = self.nb_vars;
        let depth = state.depth + 1;

        self.vars_by_sum_of_clause_weights[0..(n - depth)]
            .iter()
            .copied()
    }

    fn precompute_nks(&self) -> Vec<isize> {
        let mut estimates = vec![0_isize; self.nb_vars];
        for (i, est) in estimates.iter_mut().enumerate() {
            *est = self.precompute_nk(i);
        }
        estimates
    }
    fn precompute_nk(&self, k: usize) -> isize {
        let mut sum = 0_isize;
        for i in 0..k {
            let vi = self.vars_by_sum_of_clause_weights[i];
            sum += self.weight(t(vi), f(vi));
        }
        sum
    }
    fn precompute_estimates(&self) -> Vec<isize> {
        let mut estimates = vec![0_isize; self.nb_vars];
        for (i, est) in estimates.iter_mut().enumerate() {
            *est = self.precompute_estimate(i);
        }
        estimates
    }
    fn precompute_estimate(&self, k: usize) -> isize {
        let n = self.nb_vars;
        let mut sum = 0;
        for i in k..n {
            let vi = self.vars_by_sum_of_clause_weights[i];
            // 1st line of the equation (when i != j)
            for j in i + 1..n {
                let vj = self.vars_by_sum_of_clause_weights[j];
                let wtt = self.weight(t(vi), t(vj))
                    + self.weight(t(vi), f(vj))
                    + self.weight(f(vi), t(vj));

                let wtf = self.weight(t(vi), t(vj))
                    + self.weight(t(vi), f(vj))
                    + self.weight(f(vi), f(vj));

                let wft = self.weight(t(vi), t(vj))
                    + self.weight(f(vi), t(vj))
                    + self.weight(f(vi), f(vj));

                let wff = self.weight(t(vi), f(vj))
                    + self.weight(f(vi), t(vj))
                    + self.weight(f(vi), f(vj));

                sum += max(max(wtt, wtf), max(wft, wff));
            }
            // 2nd line of the equation: tautological + unit clauses.
            let taut = self.weight(t(vi), f(vi));
            let u_pos = self.weight(t(vi), t(vi));
            let u_neg = self.weight(f(vi), f(vi));
            sum += taut + max(u_pos, u_neg);
        }
        sum
    }

    pub fn fast_upper_bound(&self, state: &State) -> isize {
        let k = state.depth;
        let marginal_benefit = state
            .substates
            .iter()
            .copied()
            .map(|b| b.abs())
            .sum::<isize>();
        marginal_benefit + self.estimates[k] - self.initial + self.nk[k]
    }
}

impl Problem for Max2Sat {
    type State = State;

    fn nb_variables(&self) -> usize {
        self.nb_vars
    }

    fn initial_state(&self) -> State {
        State {
            depth: 0,
            substates: vec![0; self.nb_variables()],
        }
    }

    fn initial_value(&self) -> isize {
        // sum of all tautologies
        self.initial
    }
    fn for_each_in_domain(&self, variable: Variable, _state: &Self::State, f: &mut dyn DecisionCallback) {
        f.apply(Decision { variable, value: T });
        f.apply(Decision { variable, value: F });
    }

    fn transition(&self, state: &State, d: Decision) -> State {
        let k = d.variable;
        let mut ret = state.clone();
        ret.depth += 1;
        ret[k] = 0;

        let vars = self.varset(state);
        if d.value == F {
            for l in vars {
                ret[l] += self.weight(t(k), t(l)) - self.weight(t(k), f(l));
            }
        } else {
            for l in vars {
                ret[l] += self.weight(f(k), t(l)) - self.weight(f(k), f(l));
            }
        }
        ret
    }

    fn transition_cost(&self, state: &State, d: Decision) -> isize {
        let k = d.variable;
        let vars = self.varset(state);
        if d.value == F {
            let res = pos(-state[k]);
            let mut sum = self.weight(f(k), f(k)); // Weight if unit clause
            for l in vars {
                // Those that are satisfied by [[ k = F ]]
                let wff = self.weight(f(k), f(l));
                let wft = self.weight(f(k), t(l));
                // Those that actually depend on the truth value of `l`.
                let wtt = self.weight(t(k), t(l));
                let wtf = self.weight(t(k), f(l));

                sum += (wff + wft) + min(pos(state[l]) + wtt, pos(-state[l]) + wtf);
            }

            res + sum
        } else { /* when d.value == T*/
            let res = pos(state[k]);
            let mut sum = self.weight(t(k), t(k)); // Weight if unit clause
            for l in vars {
                // Those that are satisfied by [[ k = T ]]
                let wtt = self.weight(t(k), t(l));
                let wtf = self.weight(t(k), f(l));
                // Those that actually depend on the truth value of `l`.
                let wff = self.weight(f(k), f(l));
                let wft = self.weight(f(k), t(l));

                sum += (wtf + wtt) + min(pos(state[l]) + wft, pos(-state[l]) + wff);
            }

            res + sum
        }
    }

    fn next_variable(
        &self,
        _: usize,
        next_layer: &mut dyn Iterator<Item = &Self::State>,
    ) -> Option<Variable> {
        if let Some(s) = next_layer.next() {
            let nb_var = self.nb_variables();
            let depth = s.depth;
            if depth < nb_var {
                let free = nb_var - depth;
                Some(self.vars_by_sum_of_clause_weights[free - 1])
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::data::read_instance;

    use super::*;

    #[test]
    fn test_index_state() {
        let state = State {
            depth: 0,
            substates: vec![1, 2, 3, 4],
        };

        assert_eq!(state[Variable(0)], 1);
        assert_eq!(state[Variable(1)], 2);
        assert_eq!(state[Variable(2)], 3);
        assert_eq!(state[Variable(3)], 4);
    }
    #[test]
    fn test_index_mut_state() {
        let mut state = State {
            depth: 0,
            substates: vec![1, 2, 3, 4],
        };

        state[Variable(0)] = 42;
        state[Variable(1)] = 64;
        state[Variable(2)] = 16;
        state[Variable(3)] = 9;

        assert_eq!(state[Variable(0)], 42);
        assert_eq!(state[Variable(1)], 64);
        assert_eq!(state[Variable(2)], 16);
        assert_eq!(state[Variable(3)], 9);
    }

    #[test]
    fn test_initial_value() {
        let id = "debug2.wcnf";
        let problem = instance(id);

        assert_eq!(0, problem.initial_value());
    }

    #[test]
    fn test_next_state() {
        let id = "debug2.wcnf";
        let problem = instance(id);

        //let mut vars   = problem.all_vars();
        let root = problem.initial_state();
        let expected = State {
            depth: 0,
            substates: vec![0, 0, 0],
        };
        assert_eq!(expected, root);

        //vars.remove(Variable(0));
        let dec_f = Decision {
            variable: Variable(0),
            value: F,
        };
        let nod_f = problem.transition(&root, dec_f);

        let expected = State {
            depth: 1,
            substates: vec![0, -4, 3],
        };
        assert_eq!(expected, nod_f);

        let dec_t = Decision {
            variable: Variable(0),
            value: 1,
        };
        let nod_t = problem.transition(&root, dec_t);
        let expected = State {
            depth: 1,
            substates: vec![0, 0, 0],
        };
        assert_eq!(expected, nod_t);
    }

    #[test]
    fn test_rank() {
        let benef = vec![
            -183, -122, -61, -183, -183, -183, -122, -122, -183, -122, -61, -122, 0, -122, -122,
            -122, -122, -122, -183, -122, -61, 0, -122, -61, 0, 0, 0, 0, 0, -244, -61, -183, 0,
            -122, -244, -183, -61, -61, -122, -122, -122, -183, -122, 0, -183, -61, -183, -122,
            -122, -183, -183, -61, -61, -122, 0, 0, 0, 0, 0, 0,
        ];
        let state = State {
            depth: 10,
            substates: benef,
        };

        assert_eq!(5917, state.rank());
    }

    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("../resources/max2sat/")
            .join(id)
    }

    fn instance(id: &str) -> Max2Sat {
        let location = locate(id);
        Max2Sat::new(read_instance(location).expect("could not parse instance"))
    }
}
