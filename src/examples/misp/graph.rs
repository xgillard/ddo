use bitset_fixed::BitSet;
use std::fs::File;
use std::io::{BufReader, BufRead};

use regex::Regex;
use std::ops::Not;

pub struct Graph {
    pub nb_vars   : usize,
    pub adj_matrix: Vec<BitSet>,
    pub weights   : Vec<i32>
}

impl Graph {

    pub fn from_file(fname: &str) -> Graph {
        let comment = Regex::new(r"^c\s.*$").unwrap();
        let pb_decl = Regex::new(r"^p\s+edge\s+(?P<vars>\d+)\s+(?P<edges>\d+)$").unwrap();
        let edge_decl = Regex::new(r"^e\s+(?P<src>\d+)\s+(?P<dst>\d+)").unwrap();

        let mut g = Graph{nb_vars: 0, adj_matrix: vec![], weights: vec![]};

        let f = File::open(fname).unwrap();
        let f = BufReader::new(f);
        for line in f.lines() {
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