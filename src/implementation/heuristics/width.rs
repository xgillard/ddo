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

//! This module provides the implementation of various maximum width heuristics.

use crate::{WidthHeuristic, SubProblem};


/// This strategy specifies a fixed maximum width for all the layers of an
/// approximate MDD. This is a *static* heuristic as the width will remain fixed
/// regardless of the approximate MDD to generate.
///
/// # Example
/// Assuming a fixed width of 100, and problem with 5 variables (0..=4). The
/// heuristic will return 100 no matter how many free vars there are left to
/// assign (`free_vars`).
///
/// ```
/// # use ddo::*;
/// #
/// let heuristic   = FixedWidth(100);      // assume a fixed width of 100
/// let mut var_set = VarSet::all(5);       // assume a problem with 5 variables
/// let mdd_type    = MDDType:: Restricted; // assume we're compiling a restricted mdd
/// assert_eq!(100, heuristic.max_width(mdd_type, &var_set));
/// var_set.remove(Variable(1));            // let's say we fixed variables {1, 3, 4}.
/// var_set.remove(Variable(3));            // hence, only variables {0, 2} remain
/// var_set.remove(Variable(4));            // in the set of `free_vars`.
///
/// // still, the heuristic always return 100.
/// assert_eq!(100, heuristic.max_width(mdd_type, &var_set));
/// ```
#[derive(Debug, Copy, Clone)]
pub struct FixedWidth(pub usize);
impl <X> WidthHeuristic<X> for FixedWidth {
    fn max_width(&self, _: &SubProblem<X>) -> usize {
        self.0
    }
}

/// This strategy specifies a variable maximum width for the layers of an
/// approximate MDD. When using this heuristic, each layer of an approximate
/// MDD is allowed to have as many nodes as there are free variables to decide
/// upon.
///
/// # Example
/// Assuming a problem with 5 variables (0..=4). If we are calling this heuristic
/// to derive the maximum allowed width for the layers of an approximate MDD
/// when variables {1, 3, 4} have been fixed, then the set of `free_vars` will
/// contain only the variables {0, 2}. In that case, this strategy will return
/// a max width of two.
///
/// ```
/// # use ddo::*;
/// #
/// let mut var_set = VarSet::all(5);  // assume a problem with 5 variables
/// let mdd_type    = MDDType::Relaxed;// assume we are compiling a relaxed dd
/// var_set.remove(Variable(1));       // variables {1, 3, 4} have been fixed
/// var_set.remove(Variable(3));
/// var_set.remove(Variable(4));       // only variables {0, 2} remain in the set
///
/// assert_eq!(2, NbUnassignedWitdh.max_width(mdd_type, &var_set));
/// ```
#[derive(Default, Debug, Copy, Clone)]
pub struct NbUnassignedWitdh(pub usize);
impl <X> WidthHeuristic<X> for NbUnassignedWitdh {
    fn max_width(&self, x: &SubProblem<X>) -> usize {
        self.0 - x.path.len()
    }
}

/// This strategy acts as a decorator for an other max width heuristic. It
/// multiplies the maximum width of the strategy it delegates to by a constant
/// (configured) factor. It is typically used in conjunction with NbUnassigned
/// to provide a maximum width that allows a certain number of nodes.
/// Using a constant factor of 1 means that this decorator will have absolutely
/// no impact.
/// 
/// # Note:
/// This wrapper forces a minimum width of one. So it is *never*
/// going to return 0 for a value of the max width.
///
/// # Example
/// Here is an example of how to use this strategy to allow 5 nodes per
/// unassigned variable in a layer.
///
/// ```
/// # use ddo::*;
/// #
/// # let mut var_set = VarSet::all(5);  // assume a problem with 5 variables
/// # let mdd_type = MDDType::Restricted;// assume we are compiling a restricted dd
/// # var_set.remove(Variable(1));       // variables {1, 3, 4} have been fixed
/// # var_set.remove(Variable(3));
/// # var_set.remove(Variable(4));       // only variables {0, 2} remain in the set
/// let custom = Times(5, NbUnassignedWitdh);
/// assert_eq!(5 * NbUnassignedWitdh.max_width(mdd_type, &var_set), custom.max_width(mdd_type, &var_set));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Times<X>(pub usize, pub X);

impl <S, X: WidthHeuristic<S>> WidthHeuristic<S> for Times<X> {
    fn max_width(&self, x: &SubProblem<S>) -> usize {
        1.max(self.0 * self.1.max_width(x))
    }
}

/// This strategy acts as a decorator for an other max width heuristic. It
/// divides the maximum width of the strategy it delegates to by a constant
/// (configured) factor. It is typically used in conjunction with NbUnassigned
/// to provide a maximum width that allows a certain number of nodes.
/// Using a constant factor of 1 means that this decorator will have absolutely
/// no impact.
///
/// # Note
/// The maximum width is bounded by one at the very minimum. So it is *never*
/// going to return 0 for a value of the max width.
///
/// # Example
/// Here is an example of how to use this strategy to allow 1 nodes per two
/// unassigned variables in a layer.
///
/// ```
/// # use ddo::*;
/// #
/// # let mut var_set = VarSet::all(5); // assume a problem with 5 variables
/// # let mdd_type = MDDType::Relaxed;  // asume we're developing a relaxed dd 
/// # var_set.remove(Variable(1));      // variables {1, 3, 4} have been fixed
/// # var_set.remove(Variable(3));
/// # var_set.remove(Variable(4));      // only variables {0, 2} remain in the set
/// let custom = DivBy(2, NbUnassignedWitdh);
/// assert_eq!(NbUnassignedWitdh.max_width(mdd_type, &var_set) / 2, custom.max_width(mdd_type, &var_set));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DivBy<X>(pub usize, pub X);

impl <S, X: WidthHeuristic<S>> WidthHeuristic<S> for DivBy<X> {
    fn max_width(&self, x: &SubProblem<S>) -> usize {
        1.max(self.1.max_width(x) / self.0)
    }
}


#[cfg(test)]
mod test_nbunassigned {
    use std::sync::Arc;

    use crate::*;

    #[test]
    fn non_empty() {
        // assume a problem with 5 variables
        let heu = NbUnassignedWitdh(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![Decision{variable: Variable(0), value: 4}]
        };
        assert_eq!(4, heu.max_width(&sub));
    }
    #[test]
    fn all() {
        let heu = NbUnassignedWitdh(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![] // no decision made, all vars are available
        };
        assert_eq!(5, heu.max_width(&sub));
    }
    #[test]
    fn empty() {
        // assume a problem with 5 variables
        let heu = NbUnassignedWitdh(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ]
        };
        assert_eq!(0, heu.max_width(&sub));
    }
}
#[cfg(test)]
mod test_fixedwidth {
    use std::sync::Arc;

    use crate::*;

    #[test]
    fn non_empty() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![
                Decision{variable: Variable(0), value: 0},
                ]
        };
        assert_eq!(5, heu.max_width(&sub));
    }
    #[test]
    fn all() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            path : vec![]
        };
        assert_eq!(5, heu.max_width(&sub));
    }
    #[test]
    fn empty() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ]
        };
        assert_eq!(5, heu.max_width(&sub));
    }
}
#[cfg(test)]
mod test_adapters {
    use std::sync::Arc;

    use crate::*;

    #[test]
    fn test_times() {
        let heu = FixedWidth(5); 
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ]
        };
        assert_eq!(10, Times( 2, heu).max_width(&sub));
        assert_eq!(15, Times( 3, heu).max_width(&sub));
        assert_eq!( 5, Times( 1, heu).max_width(&sub));
        assert_eq!(50, Times(10, heu).max_width(&sub));
    }
    #[test]
    fn test_div_by() {
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ]
        };
        assert_eq!( 2, DivBy( 2, FixedWidth(4)).max_width(&sub));
        assert_eq!( 3, DivBy( 3, FixedWidth(9)).max_width(&sub));
        assert_eq!(10, DivBy( 1, FixedWidth(10)).max_width(&sub));
    }

    #[test]
    fn wrappers_never_return_a_zero_maxwidth() {
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ]
        };
        assert_eq!( 1, Times( 0, FixedWidth(10)).max_width(&sub));
        assert_eq!( 1, Times(10, FixedWidth( 0)).max_width(&sub));
    }

    #[test] #[should_panic]
    fn test_div_by_panics_when_div_by_zero() {
        let sub = SubProblem {
            state: Arc::new('a'),
            value: 10,
            ub   : 100,
            // all vars are assigned max width is 0
            path : vec![
                Decision{variable: Variable(0), value: 0},
                Decision{variable: Variable(1), value: 1},
                Decision{variable: Variable(2), value: 2},
                Decision{variable: Variable(3), value: 3},
                Decision{variable: Variable(4), value: 4},
                ]
        };
        DivBy( 0, FixedWidth(0)).max_width(&sub);
    }
}
