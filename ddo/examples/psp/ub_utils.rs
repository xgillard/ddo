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

use smallbitset::Set32;

/// returns the cost the minimum spanning trees for all subset of items
pub fn all_mst(changeover: &Vec<Vec<usize>>) -> Vec<usize> {
    let n_items = changeover.len() as u8;
    let n_poss = 2_u32.pow(n_items as u32);
    let mut ret = vec![];
    for i in 0..n_poss {
        let bs = Set32::from(i);
        ret.push(mst(bs, changeover));
    }
    ret
}

/// minimum spanning tree
pub fn mst(members: Set32, changeover: &[Vec<usize>]) -> usize {
    if members.len() <= 1 {
        0
    } else {
        let mut covered = Set32::empty();
        let mut total = 0;
        for a in members.iter() {
            if covered.contains(a) {
                continue;
            }
            let mut emin = usize::MAX;
            let mut bmin = a;
            for b in members.iter() {
                if a == b {
                    continue;
                }
                let edge = changeover[a][b];
                let edge = edge.min(changeover[b][a]);
                if edge < emin {
                    emin = edge;
                    bmin = b;
                }
            }
            total += emin;
            covered = covered.add(a);
            covered = covered.add(bmin);
        }
        total
    }
}