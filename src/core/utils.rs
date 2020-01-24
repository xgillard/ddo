use bitset_fixed::BitSet;
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Equal, Less};
use std::marker::PhantomData;

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

pub struct Decreasing<T, CMP>
    where CMP: Fn(&T, &T) -> Ordering {
    cmp : CMP,
    phantom: PhantomData<T>
}
impl <T, CMP> Decreasing<T, CMP>
    where CMP: Fn(&T, &T) -> Ordering {

    pub fn from(f: CMP) -> Decreasing<T, CMP> {
        Decreasing{cmp: f, phantom: PhantomData}
    }

    pub fn compare(&self, a: &T, b: &T) -> Ordering {
        match (self.cmp)(a, b) {
            Ordering::Greater => Less,
            Ordering::Less => Greater,
            Ordering::Equal=> Equal
        }
    }
}