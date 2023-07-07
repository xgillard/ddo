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

use std::{num::ParseIntError, path::Path, fs::File, io::{BufReader, BufRead}, collections::{BTreeSet, BTreeMap}};

use crate::model::Lcs;

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

/// This function is used to read a lcs instance from file. It returns either a
/// lcs instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<Lcs, Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);
    
    let mut lines = f.lines();

    let params = lines.next().ok_or(Error::Format)??.split_whitespace()
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<_>>();

    if params.len() != 2 {
        return Err(Error::Format);
    }
    
    let n_strings = params[0];
    let n_chars = params[1];

    let mut strings = vec![];
    let mut alphabet = BTreeSet::default();
    
    for (i, line) in (&mut lines).enumerate() {
        let line = line?;
        let mut data = line.split_ascii_whitespace();

        data.next().ok_or(Error::Format)?; // string length
        strings.push(data.next().ok_or(Error::Format)?.to_string());

        for char in strings[i].chars() {
            alphabet.insert(char);
        }
    }

    let mut mapping = BTreeMap::default();
    let mut inverse_mapping = BTreeMap::default();
    for (i, char) in alphabet.iter().enumerate() {
        mapping.insert(i, *char);
        inverse_mapping.insert(*char, i);
    }

    let mut strings = strings.drain(..)
        .map(|s| s.chars().map(|c| *inverse_mapping.get(&c).unwrap()).collect::<Vec<usize>>())
        .collect::<Vec<Vec<usize>>>();

    // take shortest string as reference in DP model to have less layers
    strings.sort_unstable_by_key(|s| s.len());
    
    let mut string_length = vec![];
    let mut next = vec![];
    let mut rem = vec![];

    for i in 0..n_strings {
        string_length.push(strings[i].len());
        
        let mut next_for_string = vec![];
        let mut rem_for_string = vec![];
        for j in 0..n_chars {
            let mut next_for_char = vec![0; strings[i].len() + 1];
            let mut rem_for_char = vec![0; strings[i].len() + 1];

            next_for_char[strings[i].len()] = strings[i].len();
            rem_for_char[strings[i].len()] = 0;

            for (k, char) in strings[i].iter().enumerate().rev() {
                if *char == j {
                    next_for_char[k] = k;
                    rem_for_char[k] = rem_for_char[k + 1] + 1;
                } else {
                    next_for_char[k] = next_for_char[k + 1];
                    rem_for_char[k] = rem_for_char[k + 1];
                }
            }

            next_for_string.push(next_for_char);
            rem_for_string.push(rem_for_char);
        }

        next.push(next_for_string);
        rem.push(rem_for_string);
    }

    Ok(Lcs::new(
        strings,
        n_strings,
        n_chars,
        string_length,
        next,
        rem,
        mapping,
    ))
}
