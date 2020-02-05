//! This module defines the `Solver` trait.
use crate::core::common::Decision;

/// The solver trait lets you maximize an objective function.
pub trait Solver {
    /// Returns a tuple where the first component is the value of an optimal
    /// solution to the maximization problem, and the second term is the
    /// sequence of decisions making up that solution.
    fn maximize(&mut self) -> (i32, &Option<Vec<Decision>>);
}