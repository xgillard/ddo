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

/// This enumeration simply groups the kind of errors that might occur when parsing an
/// instance from file. There can be io errors (file unavailable ?), format error
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

#[derive(Debug, Clone)]
pub struct TalentSchedInstance {
    pub nb_scenes: usize,
    pub nb_actors: usize,
    pub cost: Vec<usize>,
    pub duration: Vec<usize>,
    pub actors: Vec<Vec<usize>>,
}

/// This function is used to read a talentsched instance from file. It returns either a
/// talentsched instance if everything went on well or an error describing the problem.
pub fn read_instance<P: AsRef<Path>>(fname: P) -> Result<TalentSchedInstance, Error> {
    let f = File::open(fname)?;
    let f = BufReader::new(f);

    let mut nb_scenes = 0;
    let mut nb_actors = 0;
    let mut cost = vec![];
    let mut duration = vec![];
    let mut actors = vec![];
    
    let mut lines = f.lines();
    lines.next(); // instance name

    let mut i = 0;
    
    for line in &mut lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        let mut data = line.trim().split_ascii_whitespace();
        if i == 0 {
            nb_scenes = data.next().ok_or(Error::Format)?.parse()?;
            match data.next() {
                Some(val) => {
                    nb_actors = val.parse()?;
                    i += 1;
                },
                None => (),
            }
        } else if i == 1 {
            nb_actors = data.next().ok_or(Error::Format)?.parse()?;
        } else if i < nb_actors + 2 {
            let mut scenes = vec![];
            for _ in 0..nb_scenes {
                scenes.push(data.next().ok_or(Error::Format)?.parse()?);
            }
            actors.push(scenes);
            cost.push(data.next().ok_or(Error::Format)?.parse()?);
        } else {
            for _ in 0..nb_scenes {
                duration.push(data.next().ok_or(Error::Format)?.parse()?);
            }
        }

        i += 1
    }

    Ok(TalentSchedInstance { nb_scenes, nb_actors, cost, duration, actors })
}
