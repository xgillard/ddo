use bitset_fixed::BitSet;
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::slice::Iter;
use std::iter::Cloned;

pub struct BitSetIter<'a> {
    iter  : Cloned<Iter<'a, u64>>,
    word  : Option<u64>,
    base  : usize,
    offset: usize,
}
impl BitSetIter<'_> {
    pub fn new(bs: &BitSet) -> BitSetIter {
        let mut iter = bs.buffer().iter().cloned();
        let word = iter.next();
        BitSetIter {iter, word, base: 0, offset: 0}
    }
}
impl Iterator for BitSetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(w) = self.word {
            if w == 0 || self.offset >= 64 {
                self.word   = self.iter.next();
                self.base  += 64;
                self.offset = 0;
            } else {
                let mut mask = 1_u64 << self.offset as u64;
                while (w & mask) == 0 && self.offset < 64 {
                    mask <<= 1;
                    self.offset += 1;
                }
                if self.offset < 64 {
                    let ret = Some(self.base + self.offset);
                    self.offset += 1;
                    return ret;
                }
            }
        }
        return None
    }
}

/// A totally ordered Bitset wrapper. Useful to implement tie break mechanisms
#[derive(Eq, PartialEq)]
pub struct LexBitSet<'a>(pub &'a BitSet);

impl Ord for LexBitSet<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl PartialOrd for LexBitSet<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut x = self.0.buffer().iter().cloned();
        let mut y = other.0.buffer().iter().cloned();

        for _ in 0..x.len() {
            let xi = x.next().unwrap();
            let yi = y.next().unwrap();
            if xi != yi {
                let mut mask = 1_u64;
                for _ in 0..64 {
                    let bit_x = xi & mask;
                    let bit_y = yi & mask;
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