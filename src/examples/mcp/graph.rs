use std::ops::{Index, IndexMut};
use crate::core::common::Variable;
use std::io::{BufRead, Lines, BufReader, Read};
use regex::Regex;
use std::fs::File;

/// The graph is represented by its adjacency matrix
#[derive(Debug, Clone)]
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

    pub fn from_lines<B: BufRead>(lines: Lines<B>) -> Graph {
        let graph = Regex::new(r"^(?P<vars>\d+)\s+(?P<edges>\d+)$").unwrap();
        let edge  = Regex::new(r"^(?P<src>\d+)\s+(?P<dst>\d+)\s+(?P<w>-?\d+)$").unwrap();

        let mut result = Graph::new(0);
        for line in lines {
            let line = line.unwrap();
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if let Some(caps) = graph.captures(&line) {
                result = Graph::new(caps["vars"].to_string().parse::<usize>().unwrap());
                continue;
            }

            if let Some(caps) = edge.captures(&line) {
                let x = caps["src"].to_string().parse::<usize>().unwrap() - 1;
                let y = caps["dst"].to_string().parse::<usize>().unwrap() - 1;
                let w = caps["w"].to_string().parse::<i32>().unwrap();
                result.add_bidir_edge(x, y, w);
            }
        }
        result
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
impl From<File> for Graph {
    fn from(file: File) -> Graph {
        BufReader::new(file).into()
    }
}
impl <S: Read> From<BufReader<S>> for Graph {
    fn from(buf: BufReader<S>) -> Graph {
        buf.lines().into()
    }
}
impl <B: BufRead> From<Lines<B>> for Graph {
    fn from(lines: Lines<B>) -> Graph {
        Graph::from_lines(lines)
    }
}