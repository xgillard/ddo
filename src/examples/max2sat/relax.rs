use crate::examples::max2sat::model::{Max2Sat, State};
use std::rc::Rc;
use crate::core::abstraction::dp::{Variable, Relaxation, Decision, Problem};
use std::cmp::min;
use crate::core::abstraction::mdd::MDD;

const POSITIVE : u8 = 0x1;
const NEGATIVE : u8 = 0x2;

pub struct Max2SatRelax {
    problem : Rc<Max2Sat>
}

impl Relaxation<State> for Max2SatRelax {
    fn merge_states(&self, states: &[&State]) -> State {
        let mut next = State(vec![0; self.problem.nb_vars()]);
        unimplemented!()
    }

    fn relax_cost(&self, from: &State, to: &State, decision: &Decision) -> i32 {
        unimplemented!()
    }
}

impl Max2SatRelax {
    /// The difference between the absolute benefit of branching on variable l
    /// in the state `u` and `m`.
    fn diff_of_absolute_benefit(&self, l: Variable, u: &State, m: &State) ->  i32 {
        i32::abs(u[l]) - i32::abs(m[l])
    }

    /// Merge the substates of the given components for all of the specified states.
    fn merge_substate(&self, component: Variable, states: &[&State]) -> i32 {
        match self.substate_signs(component, states) {
            POSITIVE => self.minimum_substate(component, states),
            NEGATIVE => self.minimum_abs_value_of_substate(component, states),
            _        => 0
        }
    }

    /// Returns the smallest value of a substate
    fn minimum_substate(&self, component: Variable, states: &[&State]) -> i32 {
        let mut minimum = i32::max_value();
        for s in states {
            minimum = min(minimum, s[component]);
        }
        minimum
    }

    /// Returns the smallest absolute value of a substate
    fn minimum_abs_value_of_substate(&self, component: Variable, states: &[&State]) -> i32 {
        let mut minimum = i32::max_value();
        for s in states {
            minimum = min(minimum, i32::abs(s[component]))
        }
        minimum
    }

     /// Returns a mask with the observed signs ofall the `component`th
     /// substates of the given `states`.
    fn substate_signs(&self, component: Variable, states: &[&State]) -> u8 {
        let mut signs = 0x0_u8;
        for s in states.iter() {
            let substate = s[component];
            if substate > 0 {
                signs |= POSITIVE;
            }
            if substate < 0 {
                signs |= NEGATIVE;
            }
            if signs > 2 {
                return signs;
            }
        }
        signs
    }
}