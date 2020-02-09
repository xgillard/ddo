use std::cmp::Ordering;

use bitset_fixed::BitSet;

use crate::core::abstraction::mdd::Node;
use crate::core::common::VarSet;
use crate::core::utils::LexBitSet;

pub fn vars_from_misp_state(n: &Node<BitSet>) -> VarSet {
    VarSet(n.get_state().clone())
}

pub fn misp_min_lp(a: &Node<BitSet>, b: &Node<BitSet>) -> Ordering {
    a.get_lp_len().cmp(&b.get_lp_len())
        .then_with(|| LexBitSet(a.get_state()).cmp(&LexBitSet(b.get_state())))
}

pub fn misp_ub_order(a : &Node<BitSet>, b: &Node<BitSet>) -> Ordering {
    a.get_ub().cmp(&b.get_ub())
        .then_with(|| a.get_state().count_ones().cmp(&b.get_state().count_ones()))
        .then_with(|| a.get_lp_len().cmp(&b.get_lp_len()))
        .then_with(|| LexBitSet(a.get_state()).cmp(&LexBitSet(b.get_state())))
}