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
pub struct SrflpInstance {
    pub nb_departments: usize,
    pub lengths: Vec<isize>,
    pub flows: Vec<Vec<isize>>,
}

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// srflp instance from file. There can be io errors (file unavailable ?), format error
/// (e.g. the file is not an instance but contains the text of your next paper), 
/// or parse int errors (which are actually a variant of the format error since it tells 
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

/// This function is used to read a srflp instance from file. It returns either a
/// lcs instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<SrflpInstance, Error> {
    let instance = String::from(fname.as_ref().to_str().unwrap());
    let clearance = instance.contains("Cl");
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    
    let mut lines = f.lines();
    
    let mut nb_departments = 0;
    let mut lengths = vec![];
    let mut flows = vec![];
    
    let mut i = 0;
    for line in &mut lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        let data = line.replace(',', " ");
        let mut data = data.split_ascii_whitespace();

        if i == 0 {
            nb_departments = data.next().ok_or(Error::Format)?.parse()?;
        } else if i == 1 {
            for _ in 0..nb_departments {
                lengths.push(data.next().ok_or(Error::Format)?.parse()?);
            }
        } else {
            let mut f = vec![];for _ in 0..nb_departments {
                f.push(data.next().ok_or(Error::Format)?.parse()?);
            }
            flows.push(f);
        }

        i += 1;
    }

    if clearance {
        for item in lengths.iter_mut().take(nb_departments) {
            *item += 10;
        }
    }

    Ok(SrflpInstance { nb_departments, lengths, flows })
}
