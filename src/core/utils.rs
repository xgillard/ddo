use bitset_fixed::BitSet;
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;

pub struct BitSetIter {
    bs  : BitSet,
    cur : usize,
}
impl BitSetIter {
    pub fn new(bs: BitSet) -> BitSetIter {
        BitSetIter {bs, cur: 0}
    }
}
impl Iterator for BitSetIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.cur < self.bs.size() {
            if self.bs[self.cur] {
                let x = self.cur;
                self.cur += 1;
                return Some(x);
            }
            self.cur += 1;
        }
        None
    }
}

/// A totally ordered Bitset wrapper. Useful to implement tie break mechanisms
#[derive(Eq, PartialEq)]
pub struct LexBitSet<'a>(pub &'a BitSet);

impl <'a> Ord for LexBitSet<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl <'a> PartialOrd for LexBitSet<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let x = self.0.buffer();
        let y = other.0.buffer();

        for i in 0..x.len() {
            if x[i] != y[i] {
                let mut mask = 1_u64;
                let block_x = x[i];
                let block_y = y[i];
                for _ in 0..64 {
                    let bit_x = block_x & mask;
                    let bit_y = block_y & mask;
                    if bit_x != bit_y {
                        return Some(bit_x.cmp(&bit_y));
                    }
                    mask <<= 1;
                }
            }
        }
        Some(Equal)
    }
}