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

#[derive(Debug, Clone)]
pub struct AlpInstance {
    pub nb_classes: usize,
    pub nb_aircrafts: usize,
    pub nb_runways: usize,
    pub classes: Vec<usize>,
    pub target: Vec<isize>,
    pub latest: Vec<isize>,
    pub separation: Vec<Vec<isize>>,
}

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// alp instance from file. There can be io errors (file unavailable ?), format error
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
    /// The file was not properly formatted.
    #[error("ill formed instance")]
    Format,
}

/// This function is used to read an alp instance from file. It returns either an
/// alp instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<AlpInstance, Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    let lines = f.lines();

    let mut data = vec![];
    for line in lines {
        let line = line?;
        let mut numbers = line.split_ascii_whitespace().map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>();
        data.append(&mut numbers);
    }

    let nb_aircrafts = data[0];
    let nb_classes = data[1];
    let nb_runways = data[2];

    let mut target = vec![];
    let mut latest = vec![];
    let mut classes = vec![];

    let mut cnt = 3;
    for _ in 0..nb_aircrafts {
        target.push(data[cnt] as isize);
        latest.push(data[cnt + 1] as isize);
        classes.push(data[cnt + 2]);

        cnt += 3;
    }

    let mut separation = vec![];
    for _ in 0..nb_classes {
        let mut sep = vec![];
        for _ in 0..nb_classes {
            sep.push(data[cnt] as isize);
            cnt += 1;
        }
        separation.push(sep);
    }

    Ok(AlpInstance { nb_classes, nb_aircrafts, nb_runways, classes, target, latest, separation })
}
