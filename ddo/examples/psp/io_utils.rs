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

use std::{num::ParseIntError, path::Path, fs::File, io::{BufReader, BufRead}};

use crate::model::Psp;

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// psp instance from file. There can be io errors (file unavailable ?), format error
/// (e.g. the file is not an instance but contains the text of your next paper), 
/// or parse int errors (which are actually a variant of the format error since it tells 
/// you that the parser expected an integer number but got ... something else).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// There was an io related error
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    /// The parser expected to read something that was an integer but got some garbage
    #[error("parse int {0}")]
    ParseInt(#[from] ParseIntError),
    /// The file was not properly formatted.
    #[error("ill formed instance")]
    Format,
}

/// This function is used to read a psp instance from file. It returns either a
/// psp instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<Psp, Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    
    let mut lines = f.lines();

    let nb_periods = lines.next().ok_or(Error::Format)??.parse::<usize>()?;
    let nb_items   = lines.next().ok_or(Error::Format)??.parse::<usize>()?;
    let _nb_orders = lines.next().ok_or(Error::Format)??.parse::<usize>()?;
    
    let _blank  = lines.next();

    let mut changeover_cost = vec![];

    for line in &mut lines {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            break;
        }

        let costs = line.split_whitespace()
            .map(|x| x.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        
        changeover_cost.push(costs);
    }

    let stocking_cost = lines.next().ok_or(Error::Format)??.split_whitespace()
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();

    let _blank  = lines.next();

    let mut demands = vec![];
    for line in &mut lines {
        let line = line?;
        let line = line.trim();

        if line.is_empty() { // last line is the optimal cost, we don't care here
            break;
        }

        let demands_for_item = line.split_whitespace()
            .map(|n| n.parse::<usize>().unwrap())
            .collect::<Vec<_>>();

        demands.push(demands_for_item);
    }

    let mut prev_demands = vec![ vec![-1; nb_periods + 1] ; nb_items];
    for t in 1..=nb_periods {
        for i in 0..nb_items {
            if demands[i][t-1] > 0 {
                prev_demands[i][t] = (t-1) as isize;
            } else {
                prev_demands[i][t] = prev_demands[i][t-1];
            }
        }
    }

    let mut rem_demands = vec![ vec![0; nb_periods] ; nb_items];
    for t in 0..nb_periods {
        for i in 0..nb_items {
            if t == 0 {
                rem_demands[i][t] = demands[i][t] as isize;
            } else {
                rem_demands[i][t] = rem_demands[i][t-1] + demands[i][t] as isize;
            }
        }
    }
    
    Ok(Psp { 
        n_items: nb_items, 
        horizon: nb_periods, 
        stocking: stocking_cost, 
        changeover: changeover_cost, 
        prev_demands,
        rem_demands 
    })
}
