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

//! This module contains everything that is necessary to parse a SOP instance
//! and turn it into a structs usable in Rust. Chances are high that this 
//! module will be of little to no interest to you.

use std::{path::Path, num::ParseIntError, fs::File, io::{BufReader, BufRead}};

use crate::BitSet;

/// This structure represents the SOP instance
#[derive(Debug, Clone)]
pub struct SopInstance {
    /// The number of jobs
    pub nb_jobs: usize, 
    /// This is the distance matrix between any two nodes
    pub distances: Vec<Vec<isize>>,
    /// This vector encodes the precedence constraints for each job
    pub predecessors: Vec<BitSet>,
    /// This vector counts the number of precedence constraints for each job
    pub n_predecessors: Vec<usize>,
}

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// sop instance from file. There can be io errors (file unavailable ?), format error
/// (e.g. the file is not an instance but contains the text of your next paper), 
/// or parse int errors (which are actually a variant of the format errror since it tells 
/// you that the parser expected an integer number but got ... something else).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// There was an io related error
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    /// The parser expected to read somehting that was an integer but got some garbage
    #[error("parse int {0}")]
    ParseInt(#[from] ParseIntError),
}

/// This function is used to read a sop instance from file. It returns either a
/// sop instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<SopInstance, Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    
    let lines = f.lines();

    let mut lc          = 0;
    let mut nb_nodes    = 0;
    let mut distances   = vec![];
    let mut predecessors= vec![];

    let mut edge_weight_section = false;

    for line in lines {
        let line = line.unwrap();
        let line = line.trim();

        // skip header lines
        if line.contains("EDGE_WEIGHT_SECTION") {
            edge_weight_section = true;
            continue;
        } else if !edge_weight_section {
            continue;
        }
        
        // First line is the number of nodes
        if lc == 0 { 
            nb_nodes  = line.split_whitespace().next().unwrap().to_string().parse::<usize>().unwrap();
            distances = vec![vec![0; nb_nodes]; nb_nodes];
            (0..nb_nodes).for_each(|_| predecessors.push(BitSet::empty()));
        }
        // The next 'nb_nodes' lines represent the distances matrix
        else if (1..=nb_nodes).contains(&lc) {
            let i = (lc - 1) as usize;
            for (j, distance) in line.split_whitespace().enumerate() {
                let distance = distance.to_string().parse::<isize>().unwrap();
                distances[i][j] = distance;
                if distance == -1 {
                    predecessors[i].add_inplace(j);
                }
            }
        }
        
        lc += 1;
    }

    let n_predecessors = predecessors.iter().map(|b| b.len()).collect();
    Ok(SopInstance{nb_jobs: nb_nodes, distances, predecessors, n_predecessors})
}
