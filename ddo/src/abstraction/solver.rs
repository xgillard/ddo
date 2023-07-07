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

//! This module defines the `Solver` trait.

use crate::{Decision, Completion};

/// A decision is nothing but a sequence of decision covering all problem
/// variables.
pub type Solution = Vec<Decision>;

/// This is the solver abstraction. It is implemented by a structure that 
/// implements the branch-and-bound with MDD paradigm (or possibly an other
/// optimization algorithm -- currently only branch-and-bound with DD) to
/// find the best possible solution to a given problem.
pub trait Solver {
    /// This method orders the solver to search for the optimal solution among
    /// all possibilities. It returns a structure standing for the outcome of
    /// the attempted maximization. Such a `Completion` may either be marked 
    /// **exact** if the maximization has been carried out until optimality was 
    /// proved. Or it can be inexact, in which case it means that the 
    /// maximization process was stopped because of the satisfaction of some 
    /// cutoff criterion.
    ///
    /// Along with the `is_exact` exact flag, the completion provides an 
    /// optional `best_value` of the maximization problem. Four cases are thus
    /// to be distinguished:
    ///
    /// * When the `is_exact` flag is true, and a `best_value` is present: the
    ///   `best_value` is the maximum value of the objective function.
    /// * When the `is_exact` flag is false and a `best_value` is present, it
    ///   is the best value of the objective function that was known at the time
    ///   of cutoff.
    /// * When the `is_exact` flag is true, and no `best_value` is present: it
    ///   means that the problem admits no feasible solution (UNSAT).
    /// * When the `is_exact` flag is false and no `best_value` is present: it
    ///   simply means that no feasible solution has been found before the 
    ///   cutoff occurred.
    ///
    fn maximize(&mut self) -> Completion;
    /// This method returns the value of the objective function for the best
    /// solution that has been found. It returns `None` when no solution exists
    /// to the problem.
    fn best_value(&self) -> Option<isize>;
    /// This method returns the best solution to the optimization problem.
    /// That is, it returns the vector of decision which maximizes the value 
    /// of the objective function (sum of transition costs + initial value).
    /// It returns `None` when the problem admits no feasible solution.
    fn best_solution(&self) -> Option<Solution>;

    /// Returns the best lower bound that has been identified so far.
    /// In case where no solution has been found, it should return the minimum
    /// value that fits within an isize (-inf).
    fn best_lower_bound(&self) -> isize;
    /// Returns the tightest upper bound that can be guaranteed so far.
    /// In case where no upper bound has been computed, it should return the
    /// maximum value that fits within an isize (+inf).
    fn best_upper_bound(&self) -> isize;

    /// Sets a primal (best known value and solution) of the problem.
    fn set_primal(&mut self, value: isize, solution: Solution);

    /// Computes the optimality gap
    fn gap(&self) -> f32 {
        let ub = self.best_upper_bound();
        let lb = self.best_lower_bound();
        if ub == isize::MAX || lb == isize::MIN {
            1.0
        } else {
            let aub = ub.abs();
            let alb = lb.abs();
            let u = aub.max(alb);
            let l = aub.min(alb);
        
            (u - l) as f32 / u as f32
        }
    }    
}
