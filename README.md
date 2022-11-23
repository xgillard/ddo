# DDO a generic and efficient framework for MDD-based optimization
[![Crates.io](https://img.shields.io/crates/v/ddo)](https://crates.io/crates/ddo)
[![Documentation](https://img.shields.io/badge/Docs.rs-Latest-informational)](https://docs.rs/ddo/)
[![Build](https://github.com/xgillard/ddo/workflows/Build/badge.svg)](https://github.com/xgillard/ddo/actions?query=workflow%3A%22Build%22)
[![Tests](https://github.com/xgillard/ddo/workflows/Tests/badge.svg)](https://github.com/xgillard/ddo/actions?query=workflow%3A%22Tests%22)
[![codecov](https://codecov.io/gh/xgillard/ddo/branch/master/graph/badge.svg)](https://codecov.io/gh/xgillard/ddo)
[![Quality](https://github.com/xgillard/ddo/workflows/Quality%20Assurance/badge.svg)](https://github.com/xgillard/ddo/actions?query=workflow%3A%22Quality+Assurance%22)
![GitHub](https://img.shields.io/github/license/xgillard/ddo)

DDO is a truly generic framework to develop MDD-based combinatorial optimization
solvers in  [Rust](https://www.rust-lang.org/). Its goal is to let you describe
your optimization problem as a dynamic program along with a relaxation.

When the dynamic program of the problem is considered as a transition system,
the relaxation serves the purpose of merging different nodes of the transition
system into an other node standing for them all. In that setup, the sole
condition to ensure the correctness of the optimization algorithm is that the
replacement node must be an over approximation of all what is feasible from the
merged nodes.

***Bonus:***
As a side benefit from using `ddo`, you will be able to exploit all of your
hardware to solve your optimization in parallel.

## Setup
This library is written in stable rust (1.41 at the time of writing). Therefore,
it should be compiled with [cargo](https://doc.rust-lang.org/cargo/index.html)
the rust package manager (installed with your rust toolchain). Thanks to it,
compiling and using ddo will be a breeze no matter the platform you are working
on.

Once you have a rust tool chain installed, using ddo is as simple as adding it
as a dependency to your Cargo.toml file and building your project.

```toml
[dependencies]
ddo = "1.0.0"
```

## Simplistic yet complete example
The following example illustrates what it takes to build a complete solver based
on DDO for the [binary knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem).

The first step is always to describe the optimization problem you try to solve
in terms of a dynamic program. And then, you should also provide a relaxation
for the problem.

From this description of the problem, ddo will derive an MDD which it will use
to find an optimal solution to your problem. However, for a reasonably sized
problem, expanding the complete state space in memory would not be tractable.
Therefore, you should also provide ddo with a *relaxation*. In this context,
the relaxation means a strategy to merge several nodes of the MDD, so as to
make a fake node that is an over approximation of all the merged nodes (state +
longest path, where the longest path is the value of the objective function you
try to optimize).  

### Describe the problem as dynamic program
The first thing to do in this example is to describe the binary knapsack
problem in terms of a dynamic program. That means, describing the states of the
DP model, as well as the DP model itself.

```rust
/// In our DP model, we consider a state that simply consists of the remaining 
/// capacity in the knapsack. Additionally, we also consider the *depth* (number
/// of assigned variables) as part of the state since it useful when it comes to
/// determine the next variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct KnapsackState {
    /// the number of variables that have already been decided upon in the complete
    /// problem.
    depth: usize,
    /// the remaining capacity in the knapsack. That is the maximum load the sack
    /// can bear withouth cracking **given what is already in the sack**.
    capacity: usize
}

/// This structure represents a particular instance of the knapsack problem.
/// This is the sctructure that will implement the knapsack model.
/// 
/// The problem definition is quite easy to understand: there is a knapsack having 
/// a maximum (weight) capacity, and a set of items to chose from. Each of these
/// items having a weight and a profit, the goal is to select the best subset of
/// the items to place them in the sack so as to maximize the profit.
struct Knapsack {
    /// The maximum capacity of the sack (when empty)
    capacity: usize,
    /// the profit of each item
    profit: Vec<usize>,
    /// the weight of each item.
    weight: Vec<usize>,
}

/// For each variable in the decision problem, there are two possible choices:
/// either we take the item in the sack, or we decide to leave it out. This
/// constant is used to indicate that the item is to be taken in the sack.
const TAKE_IT: isize = 1;
/// For each variable in the decision problem, there are two possible choices:
/// either we take the item in the sack, or we decide to leave it out. This
/// constant is used to indicate that the item is to be left out of the sack.
const LEAVE_IT_OUT: isize = 0;

/// This is how you implement the labeled transition system (LTS) semantics of
/// a simple dynamic program solving the knapsack problem. The definition of
/// each of the methods should be pretty clear and easy to grasp. Should you
/// want more details on the role of each of these methods, then you are 
/// encouraged to go checking the documentation of the `Problem` trait.
impl Problem for Knapsack {
    type State = KnapsackState;

    fn nb_variables(&self) -> usize {
        self.profit.len()
    }
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
    {
        if state.capacity >= self.weight[variable.id()] {
            f.apply(Decision { variable, value: TAKE_IT });
            f.apply(Decision { variable, value: LEAVE_IT_OUT });
        } else {
            f.apply(Decision { variable, value: LEAVE_IT_OUT });
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
        if dec.value == TAKE_IT { 
            ret.capacity -= self.weight[dec.variable.id()] 
        }
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
```

### Provide a relaxation for the problem
The relaxation we will define is probably the simplest you can think of.
When one needs to define a new state to replace those exceeding the maximum
width of the MDD, we will simply keep the state with the maximum capacity
as it enables at least all the possibly behaviors feasible with lesser capacities.

Optionally, we could also implement a rough upper bound estimator for our
problem in the relaxation. However, we wont do it in this minimalistic
example since the framework provides you with a default implementation.
If you were to override the default implementation you would need to
implement the `fast_upper_bound()` method of the `Relaxation` trait.

```rust
/// In addition to a dynamic programming (DP) model of the problem you want to solve, 
/// the branch and bound with MDD algorithm (and thus ddo) requires that you provide
/// an additional relaxation allowing to control the maximum amount of space used by
/// the decision diagrams that are compiled. 
/// 
/// That relaxation requires two operations: one to merge several nodes into one 
/// merged node that acts as an over approximation of the other nodes. The second
/// operation is used to possibly offset some weight that would otherwise be lost 
/// to the arcs entering the newly created merged node.
/// 
/// The role of this very simple structure is simply to provide an implementation
/// of that relaxation.
/// 
/// # Note:
/// In addition to the aforementioned two operations, the KPRelax structure implements
/// an optional `fast_upper_bound` method. Whichone provides a useful bound to 
/// prune some portions of the state-space as the decision diagrams are compiled.
/// (aka rough upper bound pruning).
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
```

### State Ranking
The last ingredient which you need to provide in order to create an efficient solver based
on ddo, it a state ranking. That is, an heuristic which is used to compare states in order to
decide which are the most promising states and which are the lesser ones. This way, whenever
an MDD needs to perform a restriction or relaxation, it can simply keep the most promising
nodes and discard the others.
```rust
/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
struct KPranking;
impl StateRanking for KPranking {
    type State = KnapsackState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.capacity.cmp(&b.capacity)
    }
}
```

### Solve the problem
```rust
/// Then instantiate a solver and spawn as many threads as there are hardware
/// threads on the machine to solve the problem.
fn main() {
    // 1. Create an instance of our knapsack problem
    let problem = Knapsack {
        capacity: 50,
        profit  : vec![60, 100, 120],
        weight  : vec![10,  20,  30]
    };

    // 2. Create a relaxation of the problem
    let relaxation = KPRelax;

    // 3. Create a ranking to discriminate the promising and uninteresting states
    let heuristic = KPRanking;

    // 4. Define the policy you will want to use regarding the maximum width of the DD
    let width = FixedWidth(100); // here we mean max 100 nodes per layer

    // 5. Decide of a cutoff heuristic (if you dont want to let the solver run for ever)
    let cutoff = NoCutoff; // might as well be a TimeBudget (or something else)

    // 5. Create the solver fringe
    let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));
    
    // 6. Instanciate your solver
    let mut solver = DefaultSolver::new(
          &problem, 
          &relaxation, 
          &heuristic, 
          &width, 
          &cutoff, 
          &mut fringe);

    // 7. Maximize your objective function
    // the outcome provides the value of the best solution that was found for
    // the problem (if one was found) along with a flag indicating whether or
    // not the solution was proven optimal. Hence an unsatisfiable problem
    // would have `outcome.best_value == None` and `outcome.is_exact` true.
    // The `is_exact` flag will only be false if you explicitly decide to stop
    // searching with an arbitrary cutoff.
    let outcome = solver.maximize();
    // The best solution (if one exist) is retrieved with
    let solution = solver.best_solution();

    // 8. Do whatever you like with the optimal solution.
    assert_eq!(Some(220), outcome.best_value);
    println!("Solution");
    for decision in solution.unwrap().iter() {
        if decision.value == 1 {
            println!("{}", decision.variable.id());
        }
    }
}
```

## Building the more advanced examples (companion binaries)
The source code of the above (simplistic) example is provided in the examples
section of this project. Meanwhile, the examples provide an implementation for
more advanced solvers. Namely, it provides an implementation for:
+   [Maximum Independent Set Problem (MISP)](https://www.wikiwand.com/en/Independent_set_(graph_theory))
+   [Maximum 2 Satisfiability (MAX2SAT)](https://en.wikipedia.org/wiki/Maximum_satisfiability_problem)
+   [Maximum Cut Problem (MCP)](https://en.wikipedia.org/wiki/Maximum_cut)
+   Binary Knapsack

These are again compiled with cargo with the following command:
```cargo build --release --all-targets```. Once the compilation completes, you
will find the desired binaries at :
+   $project/target/release/examples/knapsack
+   $project/target/release/examples/max2sat
+   $project/target/release/examples/mcp
+   $project/target/release/examples/misp

If you have any question regarding the use of these programs, just to `<program> -h`
and it should display an helpful message explaining you how to use it.

### Note
The implementation of MISP, MAX2SAT and MCP correspond to the formulation and
relaxation proposed by [Bergman et al](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2015.0648).

## Citing DDO
If you use DDO, or find it useful for your purpose (research, teaching, business, ...)
please cite:
```
@misc{gillard:20:ddo,
    author       = {Xavier Gillard, Pierre Schaus, Vianney Coppé},
    title        = {Ddo, a generic and efficient framework for MDD-based optimization},
    howpublished = {IJCAI-20},
    year         = {2020},
    note         = {Available from \url{https://github.com/xgillard/ddo}},
}
```

## Changelog
+ Version 0.3.0 adds a cutoff mechanism which may force the solver to stop trying to prove the optimum. Some return types have been adapted to take that possibility into account.

## References
+   David Bergman, Andre A. Cire, Ashish Sabharwal, Samulowitz Horst, Saraswat Vijay, and Willem-Jan and van Hoeve. [Parallel combinatorial optimization with decision diagrams.](https://link.springer.com/chapter/10.1007/978-3-319-07046-9_25) In Helmut Simonis, editor, Integration of AI and OR Techniques in Constraint Programming, volume 8451, pages 351–367. Springer, 2014.
+   David Bergman and Andre A. Cire. [Theoretical insights and algorithmic tools for decision diagram-based optimization.](https://link.springer.com/article/10.1007/s10601-016-9239-9) Constraints, 21(4):533–556, 2016.
+   David Bergman, Andre A. Cire, Willem-Jan van Hoeve, and J. N. Hooker. [Decision Diagrams for Optimization.](https://link.springer.com/book/10.1007%2F978-3-319-42849-9) Springer, 2016.
+   David Bergman, Andre A. Cire, Willem-Jan van Hoeve, and J. N. Hooker. [Discrete optimization with decision diagrams.](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2015.0648) INFORMS Journal on Computing, 28(1):47–66, 2016.
