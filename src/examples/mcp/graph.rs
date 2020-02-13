use std::ops::{Index, IndexMut};
use crate::core::common::Variable;

/// The graph is represented by its adjacency matrix
pub struct Graph {
    pub nb_vertices: usize,
    pub adj_matrix : Vec<i32>
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Graph {nb_vertices: n, adj_matrix: vec![0; n * n]}
    }
    pub fn sum_of_negative_edges(&self) -> i32 {
        // sum is divided by two because the graph should be symmetrical
        self.adj_matrix.iter().cloned()
            .filter(|x| x.is_negative())
            .sum::<i32>() / 2
    }
    pub fn add_bidir_edge(&mut self, x: usize, y: usize, w: i32) {
        self[(x, y)] = w;
        self[(y, x)] = w;
    }
    fn offset(&self, x: usize, y: usize) -> usize {
        x * self.nb_vertices + y
    }
}
impl Index<(Variable, Variable)> for Graph {
    type Output = i32;
    fn index(&self, xy: (Variable, Variable)) -> &i32 {
        self.index((xy.0.id(), xy.1.id()))
    }
}
impl IndexMut<(Variable, Variable)> for Graph {
    fn index_mut(&mut self, xy: (Variable, Variable)) -> &mut i32 {
        self.index_mut((xy.0.id(), xy.1.id()))
    }
}
impl Index<(usize, usize)> for Graph {
    type Output = i32;
    fn index(&self, xy: (usize, usize)) -> &i32 {
        let off = self.offset(xy.0, xy.1);
        &self.adj_matrix[off]
    }
}
impl IndexMut<(usize, usize)> for Graph {
    fn index_mut(&mut self, xy: (usize, usize)) -> &mut i32 {
        let off = self.offset(xy.0, xy.1);
        &mut self.adj_matrix[off]
    }
}
