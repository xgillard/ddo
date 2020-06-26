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
ddo = "0.2.0"
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
problem in terms of a dynamic program. Here, the state of a node, is nothing
more than an unsigned integer (usize). That unsigned integer represents the
remaining capacity of our sack. To do so, you define your own structure and
make sure it implements the `Problem<usize>` trait.

```rust
/// Describe the binary knapsack problem in terms of a dynamic program.
/// Here, the state of a node, is nothing more than an unsigned integer (usize).
/// That unsigned integer represents the remaining capacity of our sack.
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
    fn domain_of<'a>(&self, state: &'a usize, var: Variable) ->Domain<'a> {
        if *state >= self.weight[var.id()] {
            vec![0, 1].into()
        } else {
            vec![0].into()
        }
    }
    fn initial_state(&self) -> usize {
        self.capacity
    }
    fn initial_value(&self) -> isize {
        0
    }
    fn transition(&self, state: &usize, _vars: &VarSet, dec: Decision) -> usize {
        state - (self.weight[dec.variable.id()] * dec.value as usize)
    }
    fn transition_cost(&self, _state: &usize, _vars: &VarSet, dec: Decision) -> isize {
        self.profit[dec.variable.id()] as isize * dec.value
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
implement the `estimate()` method of the `Relaxation` trait.

```rust
#[derive(Debug, Clone)]
struct KPRelax;
impl Relaxation<usize> for KPRelax {
    /// To merge a given selection of states (capacities) we will keep the
    /// maximum capacity. This is an obvious relaxation as it allows us to
    /// put more items in the sack.
    fn merge_states(&self, states: &mut dyn Iterator<Item=&usize>) -> usize {
        // the selection is guaranteed to have at least one state so using
        // unwrap after max to get rid of the wrapping 'Option' is perfectly safe.
        *states.max().unwrap()
    }
    /// When relaxing (merging) the states, we did not run into the risk of
    /// possibly decreasing the maximum objective value reachable from the
    /// components of the merged node. Hence, we dont need to do anything
    /// when relaxing the edge. Still, if we wanted to, we could chose to
    /// return an higher value.
    fn relax_edge(&self, src: &usize, dst: &usize, relaxed: &usize, d: Decision, cost: isize) -> isize {
        cost
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
    // 2. Build an MDD for the given problem and relaxation
    let mdd = mdd_builder(&problem, KPRelax).into_deep();

    // 3. Create a parllel solver on the basis of this MDD (this is how
    //    you can specify the MDD implementation you wish to use to develop
    //    the relaxed and restricted MDDs).
    let mut solver = ParallelSolver::new(mdd);
    // 4. Maximize your objective function
    let (optimal, solution) = solver.maximize();
    // 5. Do whatever you like with the optimal solution.
}
```

## Building the more advanced examples (companion binaries)
The source code of the above (simplistic) example is provided in the examples
section of this project. Meanwhile, the examples provide an implementation for
more advanced solvers. Namely, it provides an implementation for:
+   [Maximum Independent Set Problem (MISP)](https://www.wikiwand.com/en/Independent_set_(graph_theory))
+   [Maximum 2 Satisfiability (MAX2SAT)](https://en.wikipedia.org/wiki/Maximum_satisfiability_problem)
+   [Maximum Cut Problem (MCP)](https://en.wikipedia.org/wiki/Maximum_cut)
+   Unbounded Knapsack

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

## References
+   David Bergman, Andre A. Cire, Ashish Sabharwal, Samulowitz Horst, Saraswat Vijay, and Willem-Jan and van Hoeve. [Parallel combinatorial optimization with decision diagrams.](https://link.springer.com/chapter/10.1007/978-3-319-07046-9_25) In Helmut Simonis, editor, Integration of AI and OR Techniques in Constraint Programming, volume 8451, pages 351–367. Springer, 2014.
+   David Bergman and Andre A. Cire. [Theoretical insights and algorithmic tools for decision diagram-based optimization.](https://link.springer.com/article/10.1007/s10601-016-9239-9) Constraints, 21(4):533–556, 2016.
+   David Bergman, Andre A. Cire, Willem-Jan van Hoeve, and J. N. Hooker. [Decision Diagrams for Optimization.](https://link.springer.com/book/10.1007%2F978-3-319-42849-9) Springer, 2016.
+   David Bergman, Andre A. Cire, Willem-Jan van Hoeve, and J. N. Hooker. [Discrete optimization with decision diagrams.](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2015.0648) INFORMS Journal on Computing, 28(1):47–66, 2016.
