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

//! This module defines the types used to encode the state of a node in the 
//! TSPTW problem.

use std::hash::Hash;

use smallbitset::Set256;

use crate::instance::TimeWindow;

/// This represents a state of the problem: 
/// the salesman is at a given position in his tour and a given amount of time
/// has elapsed since he left the depot. Also, he keeps track of the nodes he
/// has already been visiting and the ones which he may still need to visit.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct TsptwState {
    /// This is the current position of the salesman
    pub position : Position,
    /// The amount of time which has elapsed since the salesman left the depot
    pub elapsed  : ElapsedTime,
    /// These are the nodes he still has to visit
    pub must_visit : Set256,
    /// These are the nodes he still might visit but is not forced to
    pub maybe_visit: Option<Set256>,
    /// This is the 'depth' in the tour, the number of cities that have already
    /// been visited
    pub depth: u16
}

/// This represents the postition of the salesman in his tour.
#[derive(Clone, Hash, Eq, PartialEq)]
pub enum Position {
    /// He can either be at an actual position (a true node)
    Node(u16),
    /// Or he can be in one node among a pool of nodes 
    /// (relaxed node == salesman is shroedinger's cat)
    Virtual(Set256),
}

/// This represents a given duration which may either be a fixed amount
/// of time (in the case of an exact node) or a fuzzy amount of time (within
/// a time window) in the case of an inexact node.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ElapsedTime {
    FixedAmount{
        duration: usize
    },
    FuzzyAmount{
        earliest: usize,
        latest  : usize
    }
}
impl ElapsedTime {
    pub fn add_duration(self, d: usize) -> Self {
        match self {
            ElapsedTime::FixedAmount{duration} => 
                ElapsedTime::FixedAmount{duration: duration + d},
            ElapsedTime::FuzzyAmount{earliest,latest} =>
                ElapsedTime::FuzzyAmount{earliest: earliest + d,latest: latest + d}
        }
    }
    pub fn _intersects_with(self, tw: TimeWindow) -> bool {
        match self {
            ElapsedTime::FixedAmount{duration} => 
                duration >= tw.earliest && duration <= tw.latest,
            ElapsedTime::FuzzyAmount{earliest, latest} => 
                   (earliest >= tw.earliest && earliest <= tw.latest)
                || (latest   >= tw.earliest && latest   <= tw.latest),
        }
    }
    pub fn earliest(self) -> usize {
        match self {
            ElapsedTime::FixedAmount{duration} => duration,
            ElapsedTime::FuzzyAmount{earliest,..}=> earliest
        }
    }
    pub fn latest(self) -> usize {
        match self {
            ElapsedTime::FixedAmount{duration} => duration,
            ElapsedTime::FuzzyAmount{latest,..}=> latest
        }
    }
}
