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

use std::cmp::{max, min, Ordering};
use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};
use std::ops::{Index, IndexMut};

use ddo::abstraction::dp::Problem;
use ddo::common::{Decision, Domain, Variable, VarSet};

use crate::instance::Weighed2Sat;

const T  : isize      = 1;
const F  : isize      =-1;
const TF : [isize; 2] = [T, F];

const fn v (x: Variable) -> isize { 1 + x.0 as isize}
const fn t (x: Variable) -> isize { v(x) }
const fn f (x: Variable) -> isize {-v(x) }
fn pos(x: isize) -> isize { max(0, x) }


#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct State {
    pub depth    : usize,
    pub substates: Vec<isize>
}

impl Index<Variable> for State {
    type Output = isize;

    fn index(&self, index: Variable) -> &isize {
        self.substates.get(index.0).unwrap()
    }
}
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

#[derive(Debug, Clone)]
pub struct Max2Sat {
    pub nb_vars : usize,
    pub initial : isize,
    pub weights : Vec<isize>,
    pub sum_of_clause_weights: Vec<isize>,
    pub vars_by_sum_of_clause_weights: Vec<Variable>,
}

const fn idx(x: isize) -> usize {
    (x.abs() - 1) as usize
}
fn mk_lit(x: isize) -> usize {
    let sign = if x > 0 { 1 } else { 0 };
    let abs  = (x.abs() - 1) as usize;

    abs + abs + sign
}
impl Max2Sat {
    pub fn new(inst: Weighed2Sat) -> Max2Sat {
        let n = inst.nb_vars;
        let mut ret = Max2Sat{
            nb_vars: n,
            initial: 0,
            weights: vec![0; (2*n)*(2*n)],
            sum_of_clause_weights: vec![0; n],
            vars_by_sum_of_clause_weights: vec![Variable(0); n]
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
        let mut order = (0..n).map(|v| Variable(v)).collect::<Vec<Variable>>();
        order.sort_unstable_by_key(|v| ret.sum_of_clause_weights[v.0]);
        ret.vars_by_sum_of_clause_weights = order;

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
}

impl Problem<State> for Max2Sat {

    fn nb_vars(&self) -> usize {
        self.nb_vars
    }

    fn initial_state(&self) -> State {
        State{depth: 0, substates: vec![0; self.nb_vars()]}
    }

    fn initial_value(&self) -> isize {
        // sum of all tautologies
        self.initial
    }

    fn domain_of<'a>(&self, _state: &'a State, _var: Variable) -> Domain<'a> {
        Domain::Slice(&TF)
    }

    fn transition(&self, state: &State, vars: &VarSet, d: Decision) -> State {
        let k = d.variable;
        let mut ret  = state.clone();
        ret.depth += 1;
        ret[k]     = 0;
        if d.value == F {
            for l in vars.iter() {
                ret[l] += self.weight(t(k), t(l)) - self.weight(t(k), f(l));
            }
        } else {
            for l in vars.iter() {
                ret[l] += self.weight(f(k), t(l)) - self.weight(f(k), f(l));
            }
        }
        ret
    }

    fn transition_cost(&self, state: &State, vars: &VarSet, d: Decision) -> isize {
        let k = d.variable;
        if d.value == F {
            let res = pos(-state[k]);
            let mut sum = self.weight(f(k), f(k)); // Weight if unit clause
            for l in vars.iter() {
                // Those that are satisfied by [[ k = F ]]
                let wff = self.weight(f(k), f(l));
                let wft = self.weight(f(k), t(l));
                // Those that actually depend on the truth value of `l`.
                let wtt = self.weight(t(k), t(l));
                let wtf = self.weight(t(k), f(l));

                sum += (wff + wft) + min(pos( state[l]) + wtt,
                                         pos(-state[l]) + wtf);
            }

            res + sum
        } else /*if d.value == T*/ {
            let res = pos(state[k]);
            let mut sum = self.weight(t(k), t(k)); // Weight if unit clause
            for l in vars.iter() {
                // Those that are satisfied by [[ k = T ]]
                let wtt = self.weight(t(k), t(l));
                let wtf = self.weight(t(k), f(l));
                // Those that actually depend on the truth value of `l`.
                let wff = self.weight(f(k), f(l));
                let wft = self.weight(f(k), t(l));

                sum += (wtf + wtt) + min(pos( state[l]) + wft,
                                         pos(-state[l]) + wff);
            }

            res + sum
        }
    }
}
impl From<File> for Max2Sat {
    fn from(file: File) -> Self {
        BufReader::new(file).into()
    }
}
impl <S: Read> From<BufReader<S>> for Max2Sat {
    fn from(buf: BufReader<S>) -> Self {
        buf.lines().into()
    }
}
impl <B: BufRead> From<Lines<B>> for Max2Sat {
    fn from(lines: Lines<B>) -> Self {
        Max2Sat::new(lines.into())
    }
}


#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_index_state() {
        let state = State{depth: 0, substates: vec![1, 2, 3, 4]};

        assert_eq!(state[Variable(0)], 1);
        assert_eq!(state[Variable(1)], 2);
        assert_eq!(state[Variable(2)], 3);
        assert_eq!(state[Variable(3)], 4);
    }
    #[test]
    fn test_index_mut_state() {
        let mut state = State{depth: 0, substates: vec![1, 2, 3, 4]};

        state[Variable(0)] = 42;
        state[Variable(1)] = 64;
        state[Variable(2)] = 16;
        state[Variable(3)] =  9;

        assert_eq!(state[Variable(0)], 42);
        assert_eq!(state[Variable(1)], 64);
        assert_eq!(state[Variable(2)], 16);
        assert_eq!(state[Variable(3)],  9);
    }

    #[test]
    fn test_initial_value() {
        let id         = "debug2.wcnf";
        let problem    = instance(id);

        assert_eq!(0, problem.initial_value());
    }

    #[test]
    fn test_next_state() {
        let id         = "debug2.wcnf";
        let problem    = instance(id);

        let mut vars   = problem.all_vars();
        let root       = problem.initial_state();
        let expected = State{depth: 0, substates: vec![0, 0, 0]};
        assert_eq!(expected, root);

        vars.remove(Variable(0));
        let dec_f      = Decision{variable: Variable(0), value: F};
        let nod_f      = problem.transition(&root, &vars, dec_f);

        let expected = State{depth: 1, substates: vec![0,-4, 3]};
        assert_eq!(expected, nod_f);

        let dec_t      = Decision{variable: Variable(0), value: 1};
        let nod_t     = problem.transition(&root, &vars, dec_t);
        let expected = State{depth: 1, substates: vec![0, 0, 0]};
        assert_eq!(expected, nod_t);
    }

    #[test]
    fn test_rank() {
        let benef =
        vec![-183, -122, -61, -183, -183, -183, -122, -122, -183, -122, -61, -122,
             0, -122, -122, -122, -122, -122, -183, -122, -61, 0, -122, -61, 0, 0,
             0, 0, 0, -244, -61, -183, 0, -122, -244, -183, -61, -61, -122, -122,
             -122, -183, -122, 0, -183, -61, -183, -122, -122, -183, -183, -61,
             -61, -122, 0, 0, 0, 0, 0, 0];
        let state = State{depth: 10, substates: benef};

        assert_eq!(5917, state.rank());
    }

    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("examples/tests/resources/max2sat/")
            .join(id)
    }

    fn instance(id: &str) -> Max2Sat {
        let location = locate(id);
        File::open(location).expect("File not found").into()
    }
}