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

use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};
use std::ops::{Index, IndexMut};

use regex::Regex;

use ddo::Variable;

/// The graph is represented by its adjacency matrix
#[derive(Debug, Clone)]
pub struct Graph {
    pub nb_vertices: usize,
    pub adj_matrix : Vec<isize>
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Graph {nb_vertices: n, adj_matrix: vec![0; n * n]}
    }
    pub fn sum_of_negative_edges(&self) -> isize {
        // sum is divided by two because the graph should be symmetrical
        self.adj_matrix.iter().cloned()
            .filter(|x| x.is_negative())
            .sum::<isize>() / 2
    }
    pub fn add_bidir_edge(&mut self, x: usize, y: usize, w: isize) {
        self[(x, y)] = w;
        self[(y, x)] = w;
    }

    pub fn from_lines<B: BufRead>(lines: Lines<B>) -> Graph {
        let graph   = Regex::new(r"^(?P<vars>\d+)\s+(?P<edges>\d+)$").unwrap();
        let edge    = Regex::new(r"^(?P<src>\d+)\s+(?P<dst>\d+)\s+(?P<w>-?\d+)$").unwrap();

        let mut result = Graph::new(0);
        for line in lines {
            let line = line.unwrap();
            let line = line.trim();

            if line.is_empty() {
                continue;
            }
            // skip comments
            if line.starts_with("c ") {
                continue;
            }

            if let Some(caps) = graph.captures(line) {
                result = Graph::new(caps["vars"].to_string().parse::<usize>().unwrap());
                continue;
            }

            if let Some(caps) = edge.captures(line) {
                let x = caps["src"].to_string().parse::<usize>().unwrap() - 1;
                let y = caps["dst"].to_string().parse::<usize>().unwrap() - 1;
                let w = caps["w"].to_string().parse::<isize>().unwrap();
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
    type Output = isize;
    fn index(&self, xy: (Variable, Variable)) -> &isize {
        self.index((xy.0.id(), xy.1.id()))
    }
}
impl IndexMut<(Variable, Variable)> for Graph {
    fn index_mut(&mut self, xy: (Variable, Variable)) -> &mut isize {
        self.index_mut((xy.0.id(), xy.1.id()))
    }
}
impl Index<(usize, usize)> for Graph {
    type Output = isize;
    fn index(&self, xy: (usize, usize)) -> &isize {
        let off = self.offset(xy.0, xy.1);
        &self.adj_matrix[off]
    }
}
impl IndexMut<(usize, usize)> for Graph {
    fn index_mut(&mut self, xy: (usize, usize)) -> &mut isize {
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
