use bitset_fixed::BitSet;

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