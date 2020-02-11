use std::cmp::Ordering;

use bitset_fixed::BitSet;

use crate::core::common::{Node, VarSet};
use crate::core::utils::LexBitSet;

pub fn vars_from_misp_state(n: &Node<BitSet>) -> VarSet {
    VarSet(n.state.clone())
}

pub fn misp_min_lp(a: &Node<BitSet>, b: &Node<BitSet>) -> Ordering {
    a.info.lp_len.cmp(&b.info.lp_len)
        .then_with(|| LexBitSet(&a.state).cmp(&LexBitSet(&b.state)))
}

pub fn misp_ub_order(a : &Node<BitSet>, b: &Node<BitSet>) -> Ordering {
    a.info.ub.cmp(&b.info.ub)
        .then_with(|| a.state.count_ones().cmp(&b.state.count_ones()))
        .then_with(|| a.info.lp_len.cmp(&b.info.lp_len))
        .then_with(|| LexBitSet(&a.state).cmp(&LexBitSet(&b.state)))
}