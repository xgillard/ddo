use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::common::{Variable, Domain, VarSet, Decision, Node, Edge};
use crate::core::implementation::solver::parallel::ParallelSolver;
use crate::core::abstraction::solver::Solver;
use crate::core::implementation::mdd::builder::mdd_builder;

#[derive(Debug, Clone)]
struct Knapsack {
    capacity: usize,
    profit  : Vec<usize>,
    weight  : Vec<usize>
}
impl Problem<usize> for Knapsack {
    fn nb_vars(&self) -> usize {
        self.profit.len()
    }
    fn initial_state(&self) -> usize {
        self.capacity
    }
    fn initial_value(&self) -> i32 {
        0
    }
    fn domain_of<'a>(&self,
                     _state   :&'a usize,
                     _variable:Variable)
       ->Domain<'a> { vec![0, 1].into() }
    fn transition(&self,s:&usize,_vs:&VarSet,d:Decision)->usize {
        s - self.weight[d.variable.id()]
    }
    fn transition_cost(&self,s:&usize,_vs:&VarSet,d:Decision)->i32 {
        self.profit[d.variable.id()] as i32 * d.value
    }
}

#[derive(Clone)]
struct KPRelax;
impl Relaxation<usize> for KPRelax {
    fn merge_nodes(&self, nodes: &[Node<usize>])-> Node<usize> {
        let lp_info = nodes.iter().map(|n| &n.info).max_by_key(|i| i.lp_len).unwrap();
        let max_capa= nodes.iter().map(|n| n.state).max().unwrap();

        Node::merged(max_capa, lp_info.lp_len, lp_info.lp_arc.clone())
    }
}

fn main() {
    let problem = Knapsack {
        capacity: 42,
        profit  : vec![ 3, 6,  9, 12],
        weight  : vec![10, 7, 21, 14]
    };
    let mdd = mdd_builder(&problem, KPRelax).into_flat();
    let mut solver = ParallelSolver::new(mdd);
    let (optimal, solution) = solver.maximize();
}