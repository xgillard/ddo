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

//! This module contains everything that is necessary to parse a TSP+TW instance
//! and turn it into a structs usable in Rust. Chances are high that this 
//! module will be of little to no interest to you.

use std::{f32, fs::File, io::{BufRead, BufReader, Lines, Read}};

/// This structure, represents a timewindow. Basically it is nothing but a 
/// closed time interval
#[derive(Debug, Copy, Clone)]
pub struct TimeWindow {
    pub earliest: usize,
    pub latest  : usize
}
impl TimeWindow {
    /// This is how you create a new time window
    pub fn new(earliest: usize, latest: usize) -> Self {
        Self { earliest, latest }
    }
}

/// This structure represents the TSP with time window instane.
#[derive(Clone)]
pub struct TsptwInstance {
    /// The number of nodes (including depot)
    pub nb_nodes   : u16, 
    /// This is the distance matrix between any two nodes
    pub distances  : Vec<Vec<usize>>,
    /// This vector encodes the time windows to reach any vertex
    pub timewindows: Vec<TimeWindow>
}

impl From<File> for TsptwInstance {
    fn from(file: File) -> Self {
        Self::from(BufReader::new(file))
    }
}
impl <S: Read> From<BufReader<S>> for TsptwInstance {
    fn from(buf: BufReader<S>) -> Self {
        Self::from(buf.lines())
    }
}
impl <B: BufRead> From<Lines<B>> for TsptwInstance {
    fn from(lines: Lines<B>) -> Self {
        let mut lc         = 0;
        let mut nb_nodes   = 0_u16;
        let mut distances  = vec![];
        let mut timewindows= vec![];

        for line in lines {
            let line = line.unwrap();
            let line = line.trim();

            // skip comment lines
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            
           // First line is the number of nodes
           if lc == 0 { 
               nb_nodes  = line.split_whitespace().next().unwrap().to_string().parse::<u16>().unwrap();
               distances = vec![vec![0; nb_nodes as usize]; nb_nodes as usize];
           }
           // The next 'nb_nodes' lines represent the distances matrix
           else if (1..=nb_nodes).contains(&lc) {
               let i = (lc - 1) as usize;
               for (j, distance) in line.split_whitespace().enumerate() {
                    let distance = distance.to_string().parse::<f32>().unwrap();
                    let distance = (distance * 10000.0) as usize;
                    distances[i][j] = distance;
               }
           }
           // Finally, the last 'nb_nodes' lines impose the time windows constraints
           else {
               let mut tokens = line.split_whitespace();
               let earliest   = tokens.next().unwrap().to_string().parse::<f32>().unwrap();
               let latest     = tokens.next().unwrap().to_string().parse::<f32>().unwrap();

               let earliest   = (earliest * 10000.0) as usize;
               let latest     = (latest   * 10000.0) as usize;

               let timewind   = TimeWindow::new(earliest, latest);
               timewindows.push(timewind);
           }
            
            lc += 1;
        }

        TsptwInstance{nb_nodes, distances, timewindows}
    }
}
