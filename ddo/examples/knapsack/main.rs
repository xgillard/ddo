use std::{path::Path, fs::File, io::{BufReader, BufRead}, time::{Duration, Instant}};

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

struct KPRelax<'a>{pb: &'a Knapsack}
impl Relaxation for KPRelax<'_> {
    type State = KnapsackState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        states.max_by_key(|node| node.capacity).copied().unwrap()
    }

    fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let mut tot = 0;
        for var in state.depth..self.pb.nb_variables() {
            if self.pb.weight[var] <= state.capacity {
                tot += self.pb.profit[var];
            }
        }
        tot as isize
    }
}

struct KPranking;
impl StateRanking for KPranking {
    type State = KnapsackState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.capacity.cmp(&b.capacity)
    }
}

fn read_instance<P: AsRef<Path>>(fname: P) -> Result<Knapsack, std::io::Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    
    let mut is_first = true;
    let mut n = 0;
    let mut count = 0;
    let mut capa = 0;
    let mut profit = vec![];
    let mut weight = vec![];

    for line in f.lines() {
        let line = line?;
        if is_first {
            is_first = false;
            let mut ab = line.split(" ");
            n = ab.next().unwrap().parse().unwrap();
            capa = ab.next().unwrap().parse().unwrap();
        } else {
            if count >= n {
                break;
            }
            let mut ab = line.split(" ");
            profit.push(ab.next().unwrap().parse().unwrap());
            weight.push(ab.next().unwrap().parse().unwrap());
            count += 1;
        }
    }
    Ok(Knapsack { capacity: capa, profit, weight })
}

fn main() {
    //let instance = "/Users/xgillard/Downloads/instances_01_KP(1)/large_scale/knapPI_3_100_1000_1";
    let instance = "/Users/xgillard/Documents/REPO/linfo2266-solutions/data/Knapsack/instance_n100_c500_10_5_10_5_0";
    let problem = read_instance(instance).unwrap();
    let relaxation= KPRelax{pb: &problem};
    let heuristic= KPranking;
    let width = FixedWidth(250);
    let cutoff = TimeBudget::new(Duration::from_secs(15));//NoCutoff;
    let mut frontier = SimpleFrontier::new(MaxUB::new(&heuristic));

    let mut solver = DefaultSolver::new(&problem, &relaxation, &heuristic, &width, &cutoff, &mut frontier);

    let start = Instant::now();
    let Completion{ is_exact, best_value } = solver.maximize();
    
    let duration = start.elapsed();
    let upper_bound = solver.best_upper_bound();
    let lower_bound = solver.best_lower_bound();
    let gap = solver.gap();
    let best_solution  = solver.best_solution().map(|mut decisions|{
        decisions.sort_unstable_by_key(|d| d.variable.id());
        decisions.iter().map(|d| d.value).collect()
    });

    println!("Duration:   {:.3} seconds \nObjective:  {}\nUpper Bnd:  {}\nLower Bnd:  {}\nGap:        {:.3}\nAborted:    {}\nSolution:   {:?}",
            duration.as_secs_f32(), 
            best_value.unwrap_or(-1), 
            upper_bound, 
            lower_bound, 
            gap, 
            !is_exact, 
            best_solution.unwrap_or(vec![]));
}
