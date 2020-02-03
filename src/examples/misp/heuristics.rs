use crate::core::abstraction::dp::{Problem, VarSet};
use bitset_fixed::BitSet;
use crate::core::abstraction::mdd::Node;
use crate::core::implementation::pooled_mdd::PooledNode;
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less, Equal};
use crate::core::utils::LexBitSet;

pub fn vars_from_misp_state(_pb: &dyn Problem<BitSet>, n: &dyn Node<BitSet>) -> VarSet {
    VarSet(n.get_state().clone())
}

pub fn misp_min_lp(a: &PooledNode<BitSet>, b: &PooledNode<BitSet>) -> Ordering {
    match a.get_lp_len().cmp(&b.get_lp_len()) {
        Ordering::Greater => Greater,
        Ordering::Less    => Less,
        Ordering::Equal   => {
            LexBitSet(a.get_state()).cmp(&LexBitSet(b.get_state()))
        }
    }
}

pub fn misp_ub_order(a : &PooledNode<BitSet>, b: &PooledNode<BitSet>) -> Ordering {
    let by_ub = a.get_ub().cmp(&b.get_ub());
    if by_ub == Equal {
        let by_sz = a.get_state().count_ones().cmp(&b.get_state().count_ones());
        if by_sz == Equal {
            let by_lp_len = a.get_lp_len().cmp(&b.get_lp_len());
            if by_lp_len == Equal {
                LexBitSet(a.get_state()).cmp(&LexBitSet(b.get_state()))
            } else { by_lp_len }
        } else { by_sz }
    } else { by_ub }
}