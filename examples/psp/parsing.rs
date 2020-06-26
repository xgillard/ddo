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

use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};

use ddo::common::Matrix;

use crate::model::PSP;

impl From<File> for PSP {
    fn from(file: File) -> PSP {
        BufReader::new(file).into()
    }
}
impl <S: Read> From<BufReader<S>> for PSP {
    fn from(buf: BufReader<S>) -> PSP {
        buf.lines().into()
    }
}
impl <B: BufRead> From<Lines<B>> for PSP {
    fn from(mut lines: Lines<B>) -> PSP {
        let nb_periods = lines.next().unwrap().unwrap().parse::<usize>().unwrap(); // damn you Result !
        let nb_items   = lines.next().unwrap().unwrap().parse::<usize>().unwrap(); // damn you Result !
        let nb_orders  = lines.next().unwrap().unwrap().parse::<usize>().unwrap(); // damn you Result !

        let _blank  = lines.next();
        let mut changeover_cost = Matrix::new_default(nb_items + 1, nb_items + 1, 0);

        let mut i = 0;
        for line in &mut lines {
            let line = line.unwrap();
            let line = line.trim();
            if line.is_empty() {
                break;
            }

            let costs = line.split_whitespace();
            for (other, cost) in costs.enumerate() {
                changeover_cost[(i, other)] = cost.parse::<usize>().unwrap();
            }

            i += 1;
        }

        let stocking_cost = lines.next().unwrap().unwrap().split_whitespace()
            .map(|x| x.parse::<usize>().unwrap())
            .collect::<Vec<usize>>();

        let _blank  = lines.next();

        let mut demands = vec![vec![0; nb_periods]; nb_items];
        i = 0;
        for line in &mut lines {
            let line = line.unwrap();
            let line = line.trim();

            if line.is_empty() { // last line is the optimal cost, we don't care here
                break;
            }

            let demands_for_item = line.split_whitespace()
                .map(|n| n.parse::<usize>().unwrap());

            for (period, demand) in demands_for_item.enumerate() {
                demands[i][period] += demand;
            }

            i += 1;
        }

        let demands_per_item =
            demands.iter().map(|item_dmd| item_dmd.iter().sum::<usize>()).collect::<Vec<usize>>();

        let deadlines =
            demands.iter().map(|item_dmd| {
                item_dmd.iter().enumerate()
                    .filter(|(_period, value)| **value > 0)
                    .map(|(period, _value)| period)
                    .collect::<Vec<usize>>()
            }).collect::<Vec<Vec<usize>>>();

        PSP {
            nb_periods,
            nb_items,
            nb_orders,
            changeover_cost,
            stocking_cost,
            demands_per_item,
            deadlines
        }
    }
}
