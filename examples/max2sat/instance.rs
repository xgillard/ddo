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
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};

#[derive(Debug, Copy, Clone, Hash, Eq, Ord, PartialOrd, PartialEq)]
pub struct BinaryClause {
    pub a: i32,
    pub b: i32
}

impl BinaryClause {
    pub fn new(x: i32, y: i32) -> BinaryClause {
        BinaryClause {a: x.min(y), b: x.max(y)}
    }
    pub fn is_tautology(self) -> bool {
        self.a == -self.b
    }
    pub fn is_unit(self) -> bool {
        self.a == self.b
    }
}

#[derive(Debug, Clone, Default)]
pub struct Weighed2Sat {
    pub nb_vars : usize,
    pub weights : HashMap<BinaryClause, i32>
}

impl Weighed2Sat {
    pub fn from_lines<B: BufRead>(lines: Lines<B>) -> Weighed2Sat {
        let comment   = Regex::new(r"^c\s.*$").unwrap();
        let pb_decl   = Regex::new(r"^p\s+wcnf\s+(?P<vars>\d+)\s+(?P<clauses>\d+)").unwrap();
        let bin_decl  = Regex::new(r"^(?P<w>-?\d+)\s+(?P<x>-?\d+)\s+(?P<y>-?\d+)\s+0").unwrap();
        let unit_decl = Regex::new(r"^(?P<w>-?\d+)\s+(?P<x>-?\d+)-?\s+0").unwrap();

        let mut instance : Weighed2Sat = Default::default();
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
                instance.nb_vars = n;
                continue;
            }

            if let Some(caps)= bin_decl.captures(&line) {
                let w = caps["w"].to_string().parse::<i32>().unwrap();
                let x = caps["x"].to_string().parse::<i32>().unwrap();
                let y = caps["y"].to_string().parse::<i32>().unwrap();

                instance.weights.insert(BinaryClause::new(x, y), w);
                continue;
            }
            if let Some(caps)= unit_decl.captures(&line) {
                let w = caps["w"].to_string().parse::<i32>().unwrap();
                let x = caps["x"].to_string().parse::<i32>().unwrap();

                instance.weights.insert(BinaryClause::new(x, x), w);
                continue;
            }
        }

        instance
    }
}

impl From<File> for Weighed2Sat {
    fn from(file: File) -> Weighed2Sat {
        BufReader::new(file).into()
    }
}
impl <S: Read> From<BufReader<S>> for Weighed2Sat {
    fn from(buf: BufReader<S>) -> Weighed2Sat {
        buf.lines().into()
    }
}
impl <B: BufRead> From<Lines<B>> for Weighed2Sat {
    fn from(lines: Lines<B>) -> Weighed2Sat {
        Weighed2Sat::from_lines(lines)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_load_from_file_using_trait() {
        let fname= locate("debug2.wcnf");
        let inst : Weighed2Sat = File::open(fname).expect("x").into();

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
            .join("examples/tests/resources/max2sat/")
            .join(id)
    }
}