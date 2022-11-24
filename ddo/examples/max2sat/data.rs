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

use regex::Regex;
use fxhash::FxHashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::errors::Error;

/// The goal of MAX2SAT is to maximize the (total weight of) number of satisfied
/// binary clauses of a SAT problem. This structure represents such a binary clause.
#[derive(Debug, Copy, Clone, Hash, Eq, Ord, PartialOrd, PartialEq)]
pub struct BinaryClause {
    pub a: isize,
    pub b: isize,
}

impl BinaryClause {
    /// Creates the binary clause based on the literals x and y
    pub fn new(x: isize, y: isize) -> BinaryClause {
        BinaryClause {
            a: x.min(y),
            b: x.max(y),
        }
    }
    /// Returns true iff the binary clause is always true.
    pub fn is_tautology(self) -> bool {
        self.a == -self.b
    }
    /// Returns true iff the binary clause only bears on one single literal.
    pub fn is_unit(self) -> bool {
        self.a == self.b
    }
}

/// This structure represents a particular instance of the weighted MAX2SAT problem.
/// By itself, it does not do much but it is a convenient way to read the instance
/// data from a file before processing it.
#[derive(Debug, Clone, Default)]
pub struct Weighed2Sat {
    pub nb_vars: usize,
    pub weights: FxHashMap<BinaryClause, isize>,
}


/// This funciton is used to read an instance from file. It returns either an
/// instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<Weighed2Sat, Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);

    let comment = Regex::new(r"^c\s.*$").unwrap();
    let pb_decl = Regex::new(r"^p\s+wcnf\s+(?P<vars>\d+)\s+(?P<clauses>\d+)").unwrap();
    let bin_decl = Regex::new(r"^(?P<w>-?\d+)\s+(?P<x>-?\d+)\s+(?P<y>-?\d+)\s+0").unwrap();
    let unit_decl = Regex::new(r"^(?P<w>-?\d+)\s+(?P<x>-?\d+)-?\s+0").unwrap();

    let mut instance: Weighed2Sat = Default::default();
    for line in f.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if comment.is_match(line) {
            continue;
        }

        if let Some(caps) = pb_decl.captures(line) {
            let n = caps["vars"].to_string().parse::<usize>()?;
            instance.nb_vars = n;
            continue;
        }

        if let Some(caps) = bin_decl.captures(line) {
            let w = caps["w"].to_string().parse::<isize>()?;
            let x = caps["x"].to_string().parse::<isize>()?;
            let y = caps["y"].to_string().parse::<isize>()?;

            instance.weights.insert(BinaryClause::new(x, y), w);
            continue;
        }
        if let Some(caps) = unit_decl.captures(line) {
            let w = caps["w"].to_string().parse::<isize>()?;
            let x = caps["x"].to_string().parse::<isize>()?;

            instance.weights.insert(BinaryClause::new(x, x), w);
            continue;
        }
    }
    Ok(instance)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_load_from_file_using_trait() {
        let fname = locate("debug2.wcnf");
        let inst: Weighed2Sat = read_instance(fname).expect("Could not parse instance");

        assert_eq!(inst.nb_vars, 3);
        assert_eq!(inst.weights.len(), 4);
    }
    #[test]
    fn test_is_unit() {
        let cla = BinaryClause::new(-1, -1);
        assert!(cla.is_unit())
    }

    fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("../resources/max2sat/")
            .join(id)
    }
}
