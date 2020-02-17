pub mod graph;
pub mod model;
pub mod relax;


pub mod main;

#[cfg(test)]
mod tests {
    use crate::examples::mcp::graph::Graph;
    use crate::examples::mcp::model::Mcp;
    use crate::examples::mcp::relax::McpRelax;
    use crate::core::abstraction::solver::Solver;
    use crate::core::implementation::mdd::builder::mdd_builder;
    use crate::core::implementation::solver::parallel::BBSolver;
    use crate::core::common::{Decision, Variable};

    use std::path::PathBuf;
    use std::fs::File;

    pub fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("tests/resources/mcp/")
            .join(id)
    }

    pub fn instance(id: &str) -> Mcp {
        let location = locate(id);
        File::open(location).expect("File not found").into()
    }

    pub fn solve(problem: Mcp) -> (i32, Option<Vec<Decision>>) {
        let relax       = McpRelax::new(&problem);
        let mdd         = mdd_builder(&problem, relax).into_flat();
        let mut solver  = BBSolver::new(mdd);

        let (best, sln) = solver.maximize();

        let solution = sln.as_ref().cloned();
        (best, solution)
    }

    fn paper_example_graph() -> Graph {
        let mut graph = Graph::new(4);
        graph.add_bidir_edge(0, 1, 1);
        graph.add_bidir_edge(0, 2, 2);
        graph.add_bidir_edge(0, 3,-2);

        graph.add_bidir_edge(1, 2, 3);
        graph.add_bidir_edge(1, 3,-1);

        graph.add_bidir_edge(2, 3,-1);
        graph
    }

    #[test]
    fn sum_of_neg_edges() {
        let graph = paper_example_graph();
        assert_eq!(graph.sum_of_negative_edges(), -4)
    }

    #[test]
    fn paper_example() {
        let (best, sln) = solve(paper_example_graph().into());
        assert_eq!(best, 4);
        assert!(sln.is_some());

        if let Some(mut ordered_sln) = sln {
            ordered_sln.sort_unstable();
            assert_eq!(ordered_sln, vec![
                Decision{variable: Variable(0), value: 1},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value:-1},
                Decision{variable: Variable(3), value: 1}
            ]);
        }
    }

    #[ignore] #[test] // this is intractable (within a short test)
    fn g1() {
        let problem  = instance("G1.txt");
        let (best, _)= solve(problem);
        println!("{}", best)
    }
}