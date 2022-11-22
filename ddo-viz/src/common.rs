// Copyright 2022 Xavier Gillard
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

//! This module contains the objects you will likely want to use when implementing 
//! a visualizable mdd

use ddo::{Decision, NodeFlags};

/// The identifier of a node: it indicates the position of the referenced node 
/// in the ’nodes’ vector of the ’VizMdd’ structure.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NodeId(pub usize);

/// The identifier of an edge: it indicates the position of the referenced edge 
/// in the ’edges’ vector of the ’VizMdd’ structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EdgeId(pub usize);

/// Represents an effective node from the decision diagram
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    /// The length of the longest path between the problem root and this
    /// specific node
    pub value: isize,
    /// The length of the longest path between this node and the terminal node.
    /// 
    /// ### Note
    /// This field is only ever populated after the MDD has been fully unrolled.
    pub value_bot: isize,
    /// The identifier of the last edge on the longest path between the problem 
    /// root and this node if it exists.
    pub best: Option<EdgeId>,
    /// The identifier of the latest edge having been added to the adjacency
    /// list of this node. (Edges, by themselves form a kind of linked structure)
    pub inbound: Option<EdgeId>,
    // The rough upper bound associated to this node
    pub rub: isize,
    /// A group of flag telling if the node is an exact node, if it is a relaxed
    /// node (helps to determine if the best path is an exact path) and if the
    /// node is reachable in a backwards traversal of the MDD starting at the
    /// terminal node.
    pub flags: NodeFlags,
}

/// Materializes one edge a.k.a arc from the decision diagram. It logically 
/// connects two nodes and annotates the link with a decision and a cost.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// The identifier of the node at the ∗∗source∗∗ of this edge.
    /// The destination end of this arc is not mentioned explicitly since it
    /// is simply the node having this edge in its inbound edges list.
    pub from: NodeId,
    /// This is the decision label associated to this edge. It gives the 
    /// information "what variable" is assigned to "what value".
    pub decision: Decision,
    /// This is the transition cost of making this decision from the state
    /// associated with the source node of this edge.
    pub cost: isize,
    /// This is a peculiarity of this design: a node does not maintain a 
    /// explicit adjacency list (only an optional edge id). The rest of the
    /// list is then encoded as a kind of ’linked’ list: each edge knows 
    /// the identifier of the next edge in the adjacency list (if there is
    /// one such edge).
    pub next: Option<EdgeId>,
}
