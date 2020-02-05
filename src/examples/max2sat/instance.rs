use std::collections::HashMap;
use regex::Regex;
use std::fs::File;
use std::io::{BufReader, BufRead, Lines, Read};

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

#[derive(Debug, Default)]
pub struct Weighed2Sat {
    pub nb_vars : usize,
    pub weights : HashMap<BinaryClause, i32>
}
impl Weighed2Sat {
    pub fn weight(&self, x: i32, y: i32) -> i32 {
        if let Some(v) = self.weights.get(&BinaryClause::new(x, y)) {
            *v
        } else {
            0
        }
    }
    pub fn from_file(fname: &str) -> Weighed2Sat {
        let f = File::open(fname).unwrap();
        let f = BufReader::new(f);

        Self::from_lines(f.lines())
    }

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
    use super::*;

    #[test]
    fn test_load_from_file() {
        let fname= "/Users/user/Documents/REPO/flatmddsolver/src/test/resources/instances/max2sat/pass.wcnf";
        let inst = Weighed2Sat::from_file(fname);

        assert_eq!(inst.nb_vars, 5);
        assert_eq!(inst.weights.len(), 10);
    }
    #[test]
    fn test_load_from_file_using_trait() {
        let fname= "/Users/user/Documents/REPO/flatmddsolver/src/test/resources/instances/max2sat/debug2.wcnf";
        let inst : Weighed2Sat = File::open(fname).expect("x").into();

        assert_eq!(inst.nb_vars, 3);
        assert_eq!(inst.weights.len(), 4);
    }
    #[test]
    fn test_is_unit() {
        let cla = BinaryClause::new(-1, -1);
        assert!(cla.is_unit())
    }
}