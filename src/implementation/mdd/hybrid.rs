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

//! This module provides the implementation of an hybrid mdd which can benefit
//! from the strengths of its components.

use std::hash::Hash;
use crate::abstraction::mdd::{Config, MDD};
use crate::implementation::mdd::shallow::pooled::PooledMDD;
use crate::implementation::mdd::deep::mdd::DeepMDD;
use crate::common::{FrontierNode, Solution};
use crate::implementation::mdd::MDDType;
use std::marker::PhantomData;
use crate::implementation::mdd::shallow::flat::FlatMDD;

#[derive(Clone)]
struct CompositeMDD<T, C, X, Y>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone,
          X: MDD<T, C>,
          Y: MDD<T, C>
{
    mddtype   : MDDType,
    restricted: X,
    relaxed   : Y,
    _phantom  : PhantomData<(T, C)>
}

impl <T, C, X, Y> MDD<T, C> for CompositeMDD<T, C, X, Y>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone,
          X: MDD<T, C>,
          Y: MDD<T, C>
{
    fn config(&self) -> &C {
        self.relaxed.config()
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.mddtype = MDDType::Exact;
        self.restricted.exact(root, best_lb)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.mddtype = MDDType::Restricted;
        self.restricted.restricted(root, best_lb)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.mddtype = MDDType::Relaxed;
        self.relaxed.relaxed(root, best_lb)
    }

    fn is_exact(&self) -> bool {
        match self.mddtype {
            MDDType::Exact      => self.restricted.is_exact(),
            MDDType::Restricted => self.restricted.is_exact(),
            MDDType::Relaxed    => self.relaxed.is_exact()
        }
    }

    fn best_value(&self) -> isize {
        match self.mddtype {
            MDDType::Exact      => self.restricted.best_value(),
            MDDType::Restricted => self.restricted.best_value(),
            MDDType::Relaxed    => self.relaxed.best_value()
        }
    }

    fn best_solution(&self) -> Option<Solution> {
        match self.mddtype {
            MDDType::Exact      => self.restricted.best_solution(),
            MDDType::Restricted => self.restricted.best_solution(),
            MDDType::Relaxed    => self.relaxed.best_solution()
        }
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        match self.mddtype {
            MDDType::Exact      => self.restricted.for_each_cutset_node(func),
            MDDType::Restricted => self.restricted.for_each_cutset_node(func),
            MDDType::Relaxed    => self.relaxed.for_each_cutset_node(func)
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// HYBRID POOLED - DEEP //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
pub struct HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    composite: CompositeMDD<T, C, PooledMDD<T, C>, DeepMDD<T, C>>
}
impl <T, C> HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    pub fn new(c: C) -> Self {
        HybridPooledDeep {
            composite: CompositeMDD {
                mddtype   : MDDType::Exact,
                restricted: PooledMDD::new(c.clone()),
                relaxed   : DeepMDD::new(c),
                _phantom: PhantomData
            }
        }
    }
}
impl <T, C> MDD<T, C> for HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn config(&self) -> &C {
        self.composite.config()
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.exact(root, best_lb)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.restricted(root, best_lb)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.relaxed(root, best_lb)
    }

    fn is_exact(&self) -> bool {
        self.composite.is_exact()
    }

    fn best_value(&self) -> isize {
        self.composite.best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self.composite.best_solution()
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        self.composite.for_each_cutset_node(func)
    }
}
impl <T, C> From<C> for HybridPooledDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn from(c: C) -> Self {
        Self::new(c)
    }
}

///////////////////////////////////////////////////////////////////////////////
/// HYBRID FLAT - DEEP //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
pub struct HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    composite: CompositeMDD<T, C, FlatMDD<T, C>, DeepMDD<T, C>>
}
impl <T, C> HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    pub fn new(c: C) -> Self {
        HybridFlatDeep {
            composite: CompositeMDD {
                mddtype   : MDDType::Exact,
                restricted: FlatMDD::new(c.clone()),
                relaxed   : DeepMDD::new(c),
                _phantom: PhantomData
            }
        }
    }
}
impl <T, C> MDD<T, C> for HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn config(&self) -> &C {
        self.composite.config()
    }

    fn exact(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.exact(root, best_lb)
    }

    fn restricted(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.restricted(root, best_lb)
    }

    fn relaxed(&mut self, root: &FrontierNode<T>, best_lb: isize) {
        self.composite.relaxed(root, best_lb)
    }

    fn is_exact(&self) -> bool {
        self.composite.is_exact()
    }

    fn best_value(&self) -> isize {
        self.composite.best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self.composite.best_solution()
    }

    fn for_each_cutset_node<F>(&self, func: F) where F: FnMut(FrontierNode<T>) {
        self.composite.for_each_cutset_node(func)
    }
}
impl <T, C> From<C> for HybridFlatDeep<T, C>
    where T: Eq + Hash + Clone,
          C: Config<T> + Clone
{
    fn from(c: C) -> Self {
        Self::new(c)
    }
}

// TODO: Write Doc
// TODO: Test composite
// TODO: Test HybridFlatDeep
// TODO: Test HybridPooledDeep