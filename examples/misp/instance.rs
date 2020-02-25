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

use bitset_fixed::BitSet;
use std::fs::File;
use std::io::{BufReader, BufRead, Lines, Read};

use regex::Regex;
use std::ops::Not;

#[derive(Debug, Clone)]
pub struct Graph {
    pub nb_vars   : usize,
    pub adj_matrix: Vec<BitSet>,
    pub weights   : Vec<i32>
}

impl Graph {
    pub fn from_lines<B: BufRead>(lines: Lines<B>) -> Graph {
        let comment   = Regex::new(r"^c\s.*$").unwrap();
        let pb_decl   = Regex::new(r"^p\s+edge\s+(?P<vars>\d+)\s+(?P<edges>\d+)$").unwrap();
        let edge_decl = Regex::new(r"^e\s+(?P<src>\d+)\s+(?P<dst>\d+)").unwrap();

        let mut g = Graph{nb_vars: 0, adj_matrix: vec![], weights: vec![]};
        for line in lines {
            let line = line.unwrap();
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if comment.is_match(&line) {
                continue;
            }

            if let Some(caps) = pb_decl.captures(&line) {
                let n = caps["vars"].to_string().parse::<usize>().unwrap();
                g.nb_vars    = n;
                g.adj_matrix = vec![BitSet::new(n); n];
                g.weights    = vec![1; n];
                continue;
            }

            if let Some(caps) = edge_decl.captures(&line) {
                let src = caps["src"].to_string().parse::<usize>().unwrap();
                let dst = caps["dst"].to_string().parse::<usize>().unwrap();

                let src = src-1;
                let dst = dst-1;

                g.adj_matrix[src].set(dst, true);
                g.adj_matrix[dst].set(src, true);

                continue;
            }

            // skip
            panic!(format!("Ill formed \"{}\"", line));
        }

        g
    }

    pub fn complement(&mut self) {
        for i in 0..self.nb_vars {
            self.adj_matrix[i] = (&self.adj_matrix[i]).not();
        }
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
    fn from(lines: Lines<B>) -> Self {
        Self::from_lines(lines)
    }
}