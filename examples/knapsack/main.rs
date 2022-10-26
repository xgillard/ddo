use ddo::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct KnapsackState {
    depth: usize,
    capacity: usize
}

struct Knapsack {
    capacity: usize,
    profit: Vec<usize>,
    weight: Vec<usize>,
}

impl Problem for Knapsack {
    type State = KnapsackState;

    fn nb_variables(&self) -> usize {
        self.profit.len()
    }
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
    {
        if state.capacity >= self.weight[variable.id()] {
            f.apply(Decision { variable, value: 1 });
            f.apply(Decision { variable, value: 0 });
        } else {
            f.apply(Decision { variable, value: 0 });
        }
    }
    fn initial_state(&self) -> Self::State {
        KnapsackState{ depth: 0, capacity: self.capacity }
    }
    fn initial_value(&self) -> isize {
        0
    }
    fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
        let mut ret = state.clone();
        ret.depth  += 1;
        if dec.value == 1 { ret.capacity -= self.weight[dec.variable.id()] }
        ret
    }
    fn transition_cost(&self, _state: &Self::State, dec: Decision) -> isize {
        self.profit[dec.variable.id()] as isize * dec.value
    }

    fn next_variable(&self, next_layer: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        let n = self.nb_variables();
        next_layer.filter(|s| s.depth < n).next().map(|s| Variable(s.depth))
    }
}

struct KPRelax;
impl Relaxation for KPRelax {
    type State = KnapsackState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        states.max_by_key(|node| node.capacity).copied().unwrap()
    }

    fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
        cost
    }
}

struct KPranking;
impl StateRanking for KPranking {
    type State = KnapsackState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.capacity.cmp(&b.capacity)
    }
}

fn main() {
    let problem = Knapsack {
        capacity: 50,
        profit: vec![60, 100, 120],
        weight: vec![10, 20, 30],
    };
    let relaxation= KPRelax;
    let heuristic= KPranking;
    let width = FixedWidth(100);
    let cutoff = NoCutoff;
    let mut frontier = SimpleFrontier::new(MaxUB::new(&heuristic));

    let mut solver = DefaultSolver::new(&problem, &relaxation, &heuristic, &width, &cutoff, &mut frontier);
    solver.maximize();
    println!("best value {}", solver.best_value().unwrap())
}
