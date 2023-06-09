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

/// Naive DP solver the 2-strings longest common subsequence problem, used
/// to build a heuristic for the m-strings problem
pub struct LcsDp<'a> {
    pub n_chars: usize,
    pub a: &'a Vec<usize>,
    pub b: &'a Vec<usize>,
}

impl<'a> LcsDp<'a> {
    pub fn solve(&self) -> Vec<Vec<isize>> {
        let mut table = vec![vec![-1; self.b.len() + 1]; self.a.len() + 1];
        for i in 0..=self.a.len() {
            table[i][self.b.len()] = 0;
        }
        for j in 0..=self.b.len() {
            table[self.a.len()][j] = 0;
        }
        self._solve(&mut table, 0, 0);
        table
    }

    fn _solve(&self, table: &mut Vec<Vec<isize>>, i: usize, j: usize) -> isize {
        if table[i][j] != -1 {
            return table[i][j];
        }
        
        table[i][j] = self._solve(table, i + 1, j)
                        .max(self._solve(table, i, j + 1))
                        .max(self._solve(table, i + 1, j + 1) + (self.a[i] == self.b[j]) as isize);

        table[i][j]
    }
}