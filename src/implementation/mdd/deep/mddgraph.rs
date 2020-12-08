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

//! This module implements a data structure which closely matches what I need to
//! implement "deep" MDDs in DDO.
//!
//! It is a graph-like data structure whose implementation was inspired by this
//! [excellent blog post](http://smallcultfollowing.com/babysteps/blog/2015/04/06/modeling-graphs-in-rust-using-vector-indices/)
//!
//! This structure with the indices that are used as pseudo pointers has been
//! chosen because it makes it easy to design a structure that can safely be
//! traversed in both directions.

use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;

use metrohash::MetroHashMap;

use crate::abstraction::heuristics::SelectableNode;
use crate::abstraction::mdd::Config;
use crate::common::Decision;
use crate::implementation::mdd::utils::NodeFlags;

/// The errors related to the graph management
#[allow(dead_code)]
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Error {
    /// Occurs when one tries to access a node, an edge or a layer that
    /// does not exist.
    NoSuchElement
}

/// This is a type safe abstraction of a node index. It serves as a
/// pseudo-pointer to fetch the information (`NodeData`) associated with some
/// node in the `Graph` representation.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeIndex(pub usize);

/// This is a type safe abstraction of an edge index. It serves as a
/// pseudo-pointer to fetch the information (`EdgeData`) associated with some
/// edge of the `Graph` representation.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct EdgeIndex(pub usize);

/// This is a type safe abstraction of a layer index. It serves as a
/// pseudo-pointer to fetch the information (`LayerData`) associated with some
/// layer of the `Graph`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct LayerIndex(pub usize);

/// The graph uses a chained representation for the edge lists associated
/// with each node. Hence, the representation of an edge is split in two
/// parts: `EdgeData` (this structure) which encapsulates a "node" of the
/// edge list and `EdgeState` which really stores the description of an edge.
///
/// # Techical Note
/// This choice was operated in order to avoid the excessive cost incurred by
/// dynamic memory allocation if the edge list were stored as vectors in
/// the nodes.
#[derive(Debug, Copy, Clone)]
pub struct EdgeData {
    pub state : EdgeState,
    pub next  : Option<EdgeIndex>
}

/// This structure really describes what you expect from the definition of an
/// edge: its source (`src`) and destination (`dst`) as well as its associated
/// labels:  `weight` (aka cost) and the `decision` that caused the occurrence
/// of this edge.
///
/// # Note:
/// Because the graph uses a chained representation for the edge lists
/// associated with each node, the representation of an edge is split in two
/// parts: `EdgeData` which basically encapsulates a "node" of the edge list and
/// `EdgeState` (this structure) which really stores the description of an edge.
///
/// This choice was operated in order to avoid the excessive cost incurred by
/// dynamic memory allocation if the edge list were stored as vectors in
/// the nodes.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct EdgeState {
    /// The index of the source node
    pub src: NodeIndex,
    /// The index of the destination node
    pub dst: NodeIndex,
    /// The weight (cost) of taking the transition materialized by this edge.
    pub weight: isize,
    /// The decision that caused the occurrence of this edge.
    pub decision: Decision,
}

/// This structure stores all the information associated with a node of the
/// graph. It is the "fat" description of such a node.
#[derive(Debug, Clone)]
pub struct NodeData<T> {
    /// This is the identifier of the node: a pseudo pointer to itself.
    /// This identifier should be seen as purely internal.
    pub my_id: NodeIndex,
    /// This is a shared reference to the node state as obtained from
    /// the problem description (initial state and/or calls to transition).
    ///
    /// ### Technical Note
    /// A shared state has been used to avoid the need for potentially expensive
    /// state copies (which would have occurred if the state were cloned).
    pub state: Rc<T>,

    /// The length (in terms of the objective function) of the longest path from
    /// the root until this node.
    pub lp_from_top: isize,
    /// The length (in terms of the objective function) of the longest path
    /// between this node and the terminal node of the MDD.
    ///
    /// ### Note
    /// This field is only ever populated after the MDD has been fully unrolled.
    pub lp_from_bot: isize,

    /// A group of flag telling if the node is an exact node, if it is a relaxed
    /// node (helps to determine if the best path is an exact path) and if the
    /// node is reachable in a backwards traversal of the MDD starting at the
    /// terminal node.
    pub flags: NodeFlags,

    /// If present, this is the index of the head of the edge list of this node.
    /// This node stores the incoming edges entering the node.
    pub inbound: Option<EdgeIndex>,
    /// If present, this is the index of the parent sitting on the longest path
    /// between the root and this node (the path whose length is `lp_from_top`.
    pub best_parent: Option<EdgeIndex>,
}
impl <T> NodeData<T> {
    /// Creates a new node associating the index `id` with the given state (`data`).
    /// The argument `relaxed` is a flag telling whether or not the node is a
    /// relaxed node.
    pub fn new(id: NodeIndex, data: Rc<T>, relaxed: bool) -> Self {
        NodeData {
            my_id       : id,
            state       : data,
            lp_from_top : isize::min_value(),
            lp_from_bot : isize::min_value(),
            flags       : NodeFlags::new(relaxed),
            inbound     : None,
            best_parent : None
        }
    }
    /// Returns (an atomically ref counted) reference to the state (from the
    /// problem description) if this node.
    pub fn state_ref(&self) -> Rc<T> {
        Rc::clone(&self.state)
    }
    /// Returns true iff the node is a relaxed node
    pub fn is_relaxed(&self) -> bool {
        self.flags.is_relaxed()
    }
    /// Returns true iff the node is "feasible" in the sense that there exists
    /// a path between this node and the terminal.
    pub fn is_feasible(&self) -> bool {
        self.flags.is_feasible()
    }
    /// Sets the 'exact' flag of this node to the given value.
    /// This is useful to ensure that a node becomes inexact when it used to be
    /// exact but an other path passing through a inexact node produces the
    /// same state.
    pub fn set_exact(&mut self, exact: bool) {
        self.flags.set_exact(exact)
    }
    /// Sets the value of the 'exact' flag to the given value.
    /// This is used during the local bounds computation (backwards traversal)
    /// to skip useless the nodes.
    pub fn set_feasible(&mut self, feasible: bool) {
        self.flags.set_feasible(feasible)
    }
}
/// NodeData represents a node. Thus, if a layer grows too large, the branch and
/// bound algorithm might need to squash the layer and thus to select NodeData
/// for merge or removal.
impl <T> SelectableNode<T> for NodeData<T> {
    /// Returns a reference to the state of this node
    fn state(&self) -> &T {
        self.state.as_ref()
    }
    /// Returns the value of the objective function at this node.
    fn value(&self) -> isize {
        self.lp_from_top
    }
    /// Returns true iff the node is an exact node.
    fn is_exact(&self) -> bool {
        self.flags.is_exact()
    }
}

/// This structure stores the information about a layer: its identifier
/// (depth =  level) as well as the index of the first and last node of the
/// layer in the `nodes` field of the `Graph`.
///
/// # Note:
/// The end field is the position *after* the last node of the layer. Therfore
/// it must be the case that the layer is empty when the its start and end
/// position are equal.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct LayerData {
    /// The identifier of the layer (= its depth).
    /// This identifier is also used as an offset in the list of layers of a
    /// `Graph`.
    pub my_id: LayerIndex,
    /// The position of the first node of this layer (in `graph.nodes`).
    pub start: usize,
    /// The position *after* the last node of this layer (in `graph.nodes`).
    /// Because `end` points *after* the end of the layer, it must be the case
    /// that the layer is empty whenever `start == end`.
    pub end: usize
}

impl LayerData {
    /// Returns the width of the layer, the number of nodes composing this layer.
    pub fn width(&self) -> usize {
        self.end - self.start
    }
}

/// This is the struct that represents an MDD graph.
///
/// It it organized as follows: the graph maintains a vector of `nodes` each of
/// which is identified by a `NodeIndex`. These node know all of their incoming
/// edges. Actually, a node does not hold an immediate reference to its edges,
/// but instead it knows the `EdgeIndex` index of one of its incoming edges.
/// (The node index refers to the position of the edge in the `edges` field of
/// the graph to which the node belongs). And, because edges (`EdgeData`) form
/// a chained structure, a node can iterate over the list of its incoming edges.
///
/// The `layers` information keeps track of a list of `Layers` each of which
/// basically keeps track of the range of nodes (from the `nodes` field)
/// composing it. The `lel` thus simply refers to the identifier of the last
/// exact layer of the graph and that identifier is a position in the `layers`
/// list.
///
/// The _reduced_ MDD aspect of the graph is obtained thanks to the `state`
/// field which associates an user defined state with a `NodeIndex`. That way,
/// if two transitions yields the same state in the same layer, both transition
/// will end up pointing to the same node. Note: in order to keep the size of
/// the `state` map reasonable, it is flushed before the addition of a new layer
/// to the graph.
///
/// # Warnings
///
/// 1. It is important to realize that because nodes and edges can be deleted or
///    merged (because of a restriction or relaxation), their identifiers
///    *cannot be considered* stable. The graph is responsible for the renumbering
///    of shifted nodes but it will not alter the information help in places it
///    does not know. So to make it short... do not keep copies of the identifiers.
///
/// 2. Nodes and edges are always appended to the `nodes` and `edges` list. Hence,
///    because we know that the first layer (LayerIndex(0)) will always contain
///    nothing but the root state (by definition of an MDD), the implementation
///    returns the root node as `nodes[0]`. (When there is a root node).
#[derive(Debug, Clone)]
pub struct Graph<T: Hash + Eq> {
    /// This is the complete list with all the nodes data of the graph.
    /// The position a `NodeIndex` refers to is to be understood as a position
    /// in this vector.
    pub nodes : Vec<NodeData<T>>,
    /// This is the complete list with all the edges data of the graph.
    /// The position an `EdgeIndex` refers to is to be understood as a position
    /// in this vector.
    pub edges : Vec<EdgeData>,
    /// This is the complete list with all the layers of the graph.
    /// The position a `LayerIndex` refers to is to be understood as a position
    /// in this vector.
    pub layers: Vec<LayerData>,
    /// This is the map that associates one state of the current layer to the
    /// node index (also belongs to the current layer) which stores it.
    pub state : MetroHashMap<Rc<T>, NodeIndex>,
    /// If present, this is the identifier of the last exact layer.
    pub lel   : Option<LayerIndex>
}

impl <T: Hash + Eq> Graph<T> {
    /// This creates a new empty graph. By default, it has one empty layer whose
    /// id is `LayerIndex(0)`.
    pub fn new() -> Self {
        Graph {
            nodes : vec![],
            edges : vec![],
            layers: vec![LayerData{my_id: LayerIndex(0), start: 0, end: 0}],
            state : MetroHashMap::default(),
            lel   : None,
        }
    }
    /// This clears the graph to make it reusable at a later time.
    /// It resets the graph state as though it had just been created.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.layers.clear();
        self.state.clear();
        self.lel  = None;

        self.layers.push(LayerData{my_id: LayerIndex(0), start: 0, end: 0});
    }
    /// Returns the root node of the MDD if there is one. Otherwise it raises a
    /// `NoSuchElement` error.
    #[allow(dead_code)]
    pub fn root(&self) -> Result<&NodeData<T>, Error> {
        if self.nodes.is_empty() {
            Err(Error::NoSuchElement)
        } else {
            Ok(&self.nodes[0])
        }
    }
    /// Returns the number of nodes in the graph
    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }
    /// Returns the number of edges in the graph
    #[allow(dead_code)]
    pub fn nb_edges(&self) -> usize {
        self.edges.len()
    }
    /// Returns true iff the MDD is exact
    pub fn is_exact(&self) -> bool {
        self.lel.is_none()
    }
    /// Returns true iff the longest r-t path of the MDD traverses no relaxed
    /// node.
    pub fn has_exact_best_path(&self, node: Option<NodeIndex>) -> bool {
        if let Some(node_id) = node {
            let n = &self.nodes[node_id.0];
            if n.is_exact() {
                true
            } else {
                !n.is_relaxed() && self.has_exact_best_path(n.best_parent.map(|e| self.edges[e.0].state.src))
            }
        } else {
            true
        }
    }
    /// Returns an iterator that goes over all the parents of the node
    /// identified with `node`.
    #[allow(dead_code)]
    pub fn parents(&self, node: NodeIndex) -> Parents<'_, T> {
        Parents {graph: self, next: self.nodes[node.0].inbound}
    }
    /// Creates the root node of the MDD with the given state `s` and `value`.
    ///
    /// ### Warning
    /// This method will *not* check if you lready added a root to the graph.
    /// In that case, it will simply add a new disconnected node to the current
    /// layer but it will not replace the preexisting root. Doing so would
    /// create incohernce in the structure and should be avoided.
    pub fn add_root(&mut self, s: Rc<T>, value: isize) -> NodeIndex {
        let idx = NodeIndex(0);
        let node= NodeData {
            my_id       : idx,
            state       : Rc::clone(&s),
            lp_from_top : value,
            lp_from_bot : isize::min_value(),
            best_parent : None,
            inbound     : None,
            flags       : NodeFlags::new_exact(),
        };

        self.nodes.push(node);
        self.state.insert(s, idx);
        self.layers.last_mut().unwrap().end += 1;
        idx
    }
    /// Creates a node for the state `s` or retrieve the index of a pre-existing
    /// node having a state equal to `s`.
    pub fn add_node(&mut self, s: T) -> NodeIndex {
        let state = &mut self.state;
        let nodes = &mut self.nodes;
        let layers= &mut self.layers;

        let s = Rc::new(s);

        *state.entry(Rc::clone(&s)).or_insert_with(|| {
            let idx = NodeIndex(nodes.len());
            let node= NodeData::new(idx, s, false);
            nodes.push(node);
            layers.last_mut().unwrap().end += 1;
            idx
        })
    }
    /// Appends an edge between the nodes `src` and `dst` labelled with `decision`
    /// and having the given `weight`. This method returns the index of the new
    /// edge.
    pub fn add_edge(&mut self, src: NodeIndex, dst: NodeIndex, decision: Decision, weight: isize) -> EdgeIndex {
        let idx = EdgeIndex(self.edges.len());
        self.edges.push(EdgeData{ state: EdgeState{src, dst, decision, weight}, next: None});
        idx
    }
    /// This method records the branching from the node `orig_id` with the given
    /// `decision`. It creates a fresh node for the `dest` state (or reuses one
    /// if `dest` already belongs to the current layer) and draws an edge of
    /// of the given `weight` between `orig_id` and the new node.
    ///
    /// ### Note:
    /// In case where this branching would create a new longest path to an
    /// already existing node, the length and best parent of the pre-existing
    /// node are updated.
    pub fn branch(&mut self, orig_id: NodeIndex, dest: T, decision: Decision, weight: isize) {
        let dest_id     = self.add_node(dest);

        let edge_id     = self.add_edge(orig_id, dest_id, decision, weight);
        let edge        = &mut self.edges[edge_id.0];

        let orig_node   = &mut self.nodes[orig_id.0];
        let orig_exact  = orig_node.is_exact();
        let orig_lp_len = orig_node.lp_from_top;

        let dest_node = &mut self.nodes[dest_id.0];
        dest_node.set_exact(dest_node.is_exact() && orig_exact);
        edge.next = dest_node.inbound;
        dest_node.inbound  = Some(edge_id);

        if orig_lp_len + weight > dest_node.lp_from_top {
            dest_node.lp_from_top = orig_lp_len + weight;
            dest_node.best_parent = Some(edge_id);
        }
    }
    /// This method adds a new layer to the graph. It must be called after all
    /// all transitions of all nodes of the current layer have been unrolled.
    pub fn add_layer(&mut self) -> LayerIndex {
        let idx = LayerIndex(self.layers.len());
        let nb_nodes = self.nb_nodes();
        self.state.clear();
        self.layers.push(LayerData{my_id: idx, start: nb_nodes, end: nb_nodes});
        idx
    }
    /// Returns the information of the current layer.
    pub fn current_layer(&self) -> LayerData {
        *self.layers.last().unwrap()
    }
    /// Returns a slice of the nodes comprising only those nodes belonging to
    /// the given layer.
    pub fn layer_nodes(&self, layer_id: LayerIndex) -> &[NodeData<T>] {
        let layer = self.layers[layer_id.0];
        self.nodes[layer.start..layer.end].iter().as_slice()
    }
    /// Sorts the nodes of the last layer (the current layer !) using the order
    /// specified by the given function.
    ///
    /// # Warning:
    /// This function sorts the nodes of the current layer. While this should
    /// have no impact for an external observer (layers should be considered as
    /// unordered sets), it *does* have an internal impact. Indeed, the change
    /// of order means that a desynchronization occurs between the nodes and
    /// their identifiers. The nodes are thus renamed. In the node data itself
    /// *and* in the `dst` field of all incoming edges.
    pub fn sort_last_layer<F>(&mut self, f: F)
        where F: FnMut(&NodeData<T>, &NodeData<T>) -> Ordering
    {
        let layer = self.layers.last().unwrap();
        let nodes = &mut self.nodes[layer.start..layer.end];
        let states= &mut self.state;

        nodes.sort_unstable_by(f);

        for (i, n) in nodes.iter_mut().enumerate() {
            let new_id = NodeIndex(layer.start + i);

            let mut inbound = n.inbound;
            while let Some(id) = inbound {
                let edge = &mut self.edges[id.0];
                edge.state.dst = new_id;
                inbound = edge.next;
            }

            n.my_id = new_id;
            *states.get_mut(&n.state).unwrap() = new_id;
        }
    }
    /// Retuns the longest path (sequence of decisions leading to the highest
    /// objective function value) between the root and the node identified with
    /// the node index `n`.
    pub fn longest_path(&self, n: NodeIndex) -> Vec<Decision> {
        let node = &self.nodes[n.0];
        self._longest_path(node)
    }
    /// Retuns the longest path (sequence of decisions leading to the highest
    /// objective function value) and the given `node`.
    fn _longest_path(&self, node: &NodeData<T>) -> Vec<Decision> {
        let mut res = vec![];

        let mut best = node.best_parent;
        while let Some(e_id) = best {
            let edge = &self.edges[e_id.0];
            res.push(edge.state.decision);

            let parent = &self.nodes[edge.state.src.0];
            best = parent.best_parent;
        }
        res
    }
    /// Finds the "best" terminal node. Because our graph structure does not
    /// materializes a "true" terminal node `t`. It must go over all the nodes
    /// from the last layer to determine what would be the best parent of `t`.
    /// This function returns the identifier of that best parent (if such a best
    /// parent exists).
    pub fn find_best_terminal_node(&self) -> Option<NodeIndex> {
        self._find_best_terminal_node().map(|n| n.my_id)
    }
    /// Finds the "best" terminal node. Because our graph structure does not
    /// materializes a "true" terminal node `t`. It must go over all the nodes
    /// from the last layer to determine what would be the best parent of `t`.
    /// This function returns the identifier of that best parent (if such a best
    /// parent exists).
    fn _find_best_terminal_node(&self) -> Option<&NodeData<T>> {
        let final_layer = self.layers.last().unwrap();
        let nodes       = &self.nodes[final_layer.start..final_layer.end];

        nodes.iter().max_by_key(|n| n.lp_from_top)
    }
    /// Returns the best solution encoded in this graph. That is the sequence
    /// of decisions that leads to a node of the last layer having the highest
    /// value. (The value of the "best terminal node").
    #[allow(dead_code)]
    pub fn best_solution(&self) -> Vec<Decision> {
        self._find_best_terminal_node()
            .map_or(vec![], |best| self._longest_path(best))
    }
    /// Records the id of the last exact layer. It only has an effect when
    /// the graph is considered to be still correct. In other words, it will
    /// only remember the index of LEL before the first call to `relax_last()`.
    fn remember_lel(&mut self) {
        if self.lel.is_none() {
            self.lel = Some(LayerIndex(self.layers.len() - 2));
        }
    }
    /// This method restricts the last layer (the current layer !) to make sure
    /// it fits within the maximum "width" size.
    ///
    /// # Note
    /// The removed nodes and all their inbound edges are irremediably lost.
    ///
    /// # Warning
    /// It is your responsibility to make sure the layer is broad enough to be
    /// relaxed. Failing to do so would result in a panic!
    ///
    /// # Warning:
    /// This function sorts the nodes of the current layer. While this should
    /// have no impact for an external observer (layers should be considered as
    /// unordered sets), it *does* have an internal impact. Indeed, the change
    /// of order means that a desynchronization occurs between the nodes and
    /// their identifiers. The nodes are thus renamed. In the node data itself
    /// *and* in the `dst` field of all incoming edges.
    pub fn restrict_last<C: Config<T>>(&mut self, width: usize, config: &C) {
        self.remember_lel();

        let layer = *self.layers.last().unwrap();
        let layer_end = layer.start + width;

        // Select the nodes to be kept/deleted
        self.sort_last_layer(|a, b| config.compare(a, b).reverse());
        let (_keep, squash) = self.nodes.split_at_mut(layer.start + width);

        let mut ids = vec![];
        ids.reserve_exact(squash.len());

        for node in squash.iter() {
            self.state.remove(&node.state);
            ids.push(node.my_id);
        }

        let mut edges_to_delete = vec![];
        for node in ids {
            self.add_edges_of_into(node, &mut edges_to_delete);
        }

        self.delete_all_edges_in_list(&mut edges_to_delete);

        // Drop all squashed nodes (by construction, these are the last ones)
        self.nodes.truncate(layer_end);
        // make sure the layer's end node matches w/ the last existing (kept) node.
        self.layers.last_mut().unwrap().end = layer_end;
    }
    /// This method deletes all edges in the given list of egdes to delete in
    /// a way that minimizes the amount of relabelling to be done.
    fn delete_all_edges_in_list(&mut self, to_delete: &mut Vec<EdgeIndex>) {
        to_delete.sort_unstable_by_key(|id|id.0);

        while !to_delete.is_empty() {
            let del = to_delete.pop().unwrap();

            if !self.edges.is_empty() {
                let last = EdgeIndex(self.edges.len()-1);

                if del != last {
                    self.rename_edge(last, del);
                }
            }

            self.edges.swap_remove(del.0);
        }
    }
    /// Populates the given list `into` with the incoming edges of the requested
    /// `node`. Note, this method *adds* these edges to the given list. It will
    /// *not* ensure that the incoming edges are the sole content of the vector.
    fn add_edges_of_into(&self, node: NodeIndex, into: &mut Vec<EdgeIndex>) {
        let node         = &self.nodes[node.0];
        let mut it       = node.inbound;
        while let Some(eid) = it {
            into.push(eid);
            it = self.edges[eid.0].next;
        }
    }
    /// Renames the edges `eid` with the new name `to`.
    fn rename_edge(&mut self, eid: EdgeIndex, to: EdgeIndex) {
        let edge      = self.edges[eid.0];

        let dest_node = &mut self.nodes[edge.state.dst.0];
        if dest_node.best_parent == Some(eid) {
            dest_node.best_parent = Some(to);
        }
        if dest_node.inbound == Some(eid) {
            dest_node.inbound = Some(to);
        } else {
            let mut it = dest_node.inbound;
            while let Some(current) = it {
                let current_edge = &mut self.edges[current.0];
                if current_edge.next == Some(eid) {
                    current_edge.next = Some(to);
                    break;
                } else {
                    it = current_edge.next;
                }
            }
        }
    }
    /// This method relaxes the last layer (the current layer !) to make sure
    /// it fits within the maximum "width" size. The overdue nodes are merged
    /// according to the configured strategy.
    ///
    /// # Note
    /// The removed nodes and all their inbound edges are irremediably lost.
    ///
    /// # Warning
    /// It is your responsibility to make sure the layer is broad enough to be
    /// relaxed. Failing to do so would result in a panic!
    ///
    /// # Warning
    /// This function will panic if you request a relaxation that would leave
    /// zero nodes in the current layer.
    ///
    /// # Warning
    /// This function sorts the nodes of the current layer. While this should
    /// have no impact for an external observer (layers should be considered as
    /// unordered sets), it *does* have an internal impact. Indeed, the change
    /// of order means that a desynchronization occurs between the nodes and
    /// their identifiers. The nodes are thus renamed. In the node data itself
    /// *and* in the `dst` field of all incoming edges.
    pub fn relax_last<C: Config<T>>(&mut self, width: usize, config: &C) {
        self.remember_lel();

        let layer = *self.layers.last().unwrap();
        let layer_end = layer.start + width - 1;

        // Select the nodes to be merged
        self.sort_last_layer(|a, b| config.compare(a, b).reverse());
        let (keep, squash) = self.nodes.split_at_mut(layer_end);

        // Merge the states of the nodes that must go
        let merged = config.merge_states(&mut squash.iter().map(|n| n.state.as_ref()));
        let merged = Rc::new(merged);

        // Determine the identifier of the new merged node. If the merged state
        // is already known, combine the known state with the merger
        let mut n_id   = NodeIndex(layer_end);
        let mut exists = false;
        if let Some(old) = self.state.get(&merged) {
            if old.0 < layer_end {
                n_id      = *old;
                exists    = true;
            }
        }
        // .. make a default node
        let mut merged_node = NodeData {
            my_id      : n_id,
            state      : Rc::clone(&merged),
            flags      : NodeFlags::new_relaxed(),
            lp_from_top: isize::min_value(),
            lp_from_bot: isize::min_value(),
            inbound    : None,
            best_parent: None,
        };
        // .. combine w/ prev. known state if necessary
        if exists {
            let old = &keep[n_id.0];
            merged_node.lp_from_top = old.lp_from_top;
            merged_node.lp_from_bot = old.lp_from_bot;
            merged_node.inbound     = old.inbound;
            merged_node.best_parent = old.best_parent;
        }

        // Relax all edges and transfer them to the new merged node (op. gamma)
        for node in squash.iter() {
            let mut it = node.inbound;
            while let Some(eid) = it {
                let edge      = &mut self.edges[eid.0];
                let parent    = &keep[edge.state.src.0];
                let src_state = parent.state.as_ref();
                let dst_state = node.state.as_ref();
                let mrg_state = merged_node.state.as_ref();

                edge.state.weight = config.relax_edge(src_state, dst_state, mrg_state, edge.state.decision, edge.state.weight);
                edge.state.dst    = n_id;

                it = edge.next;

                edge.next = merged_node.inbound;
                merged_node.inbound = Some(eid);

                // update the merged node if the relaxed edge improves longest path
                if parent.lp_from_top + edge.state.weight > merged_node.lp_from_top {
                    merged_node.lp_from_top = parent.lp_from_top + edge.state.weight;
                    merged_node.best_parent = Some(eid);
                }
            }

            // The given node is about to be dropped, remove it from the map.
            self.state.remove(&node.state);
        }

        // Drop all squashed nodes (by construction, these are the last ones)
        self.nodes.truncate(layer_end);

        // Save the result of the merger
        if !exists {
            self.state.insert(merged, n_id);
            self.nodes.push(merged_node);
        } else {
            self.nodes[n_id.0] = merged_node;
        }

        // make sure the layer's end node matches w/ the merged one.
        self.layers.last_mut().unwrap().end = if exists { layer_end } else { 1 + layer_end };
    }
}
impl <T: Hash + Eq> Default for Graph<T> {
    /// Creates a default empty graph with a single empty layer (`LayerIndex(0)`)
    fn default() -> Self {
        Self::new()
    }
}
/// This structure implements a tiny reference to some node of a graph.
/// It is used to iterate over the parents of some node.
pub struct NodeRef<'g, T: Hash + Eq> {
    graph: &'g Graph<T>,
    id   : NodeIndex
}
impl <T: Hash + Eq> AsRef<NodeData<T>> for NodeRef<'_, T> {
    fn as_ref(&self) -> &NodeData<T> {
        &self.graph.nodes[self.id.0]
    }
}
impl <T: Hash + Eq> Deref for NodeRef<'_, T> {
    type Target = NodeData<T>;
    fn deref(&self) -> &NodeData<T> {
        self.as_ref()
    }
}

/// This is an iterator to go over all the parents of some given node.
#[derive(Debug, Copy, Clone)]
pub struct Parents<'g, T: Hash + Eq> {
    graph: &'g Graph<T>,
    next : Option<EdgeIndex>
}
impl <'g, T: Hash + Eq> Iterator for Parents<'g, T> {
    type Item = NodeRef<'g, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.map(|idx| {
            let edge = self.graph.edges[idx.0];
            self.next= edge.next;
            NodeRef{graph: self.graph, id: edge.state.src}
        })
    }
}


// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
mod test_node_data {
    use std::rc::Rc;

    use crate::abstraction::heuristics::SelectableNode;
    use crate::implementation::mdd::deep::mddgraph::{NodeData, NodeIndex};

    #[test]
    fn new_can_create_a_new_relaxed_node() {
        let id    = NodeIndex(65);
        let tested= NodeData::new(id, Rc::new(42), true);

        assert_eq!(id,    tested.my_id);
        assert_eq!(42,    *tested.state);
        assert_eq!(true , tested.flags.is_relaxed());
        assert_eq!(false, tested.flags.is_exact());
    }
    #[test]
    fn new_can_create_a_new_exact_node() {
        let id    = NodeIndex(24);
        let tested= NodeData::new(id, Rc::new(7), false);
        assert_eq!(id,    tested.my_id);
        assert_eq!(7 ,    *tested.state);
        assert_eq!(false, tested.flags.is_relaxed());
        assert_eq!(true,  tested.flags.is_exact());
    }
    #[test]
    fn when_created_a_node_is_disconnected_from_graph() {
        let tested= NodeData::new(NodeIndex(65), Rc::new(42), true);

        // infinite negative distance to both poles of the MDD.
        assert_eq!(isize::min_value(), tested.lp_from_top);
        assert_eq!(isize::min_value(), tested.lp_from_bot);
        assert!(tested.inbound.is_none());
        assert!(tested.best_parent.is_none());
    }
    #[test]
    fn state_ref_yields_an_arc_reference_to_the_state() {
        let tested= NodeData::new(NodeIndex(65), Rc::new(42), true);
        assert_eq!(Rc::new(42), tested.state_ref());
    }
    #[test]
    fn state_yields_a_plain_reference_to_the_state() {
        let tested= NodeData::new(NodeIndex(65), Rc::new(42), true);
        assert_eq!(&42, tested.state());
    }
    #[test]
    fn is_relaxed_iff_node_was_relaxed() {
        let tested= NodeData::new(NodeIndex(65), Rc::new(42), true);
        assert!(tested.is_relaxed());

        let tested= NodeData::new(NodeIndex(65), Rc::new(42), false);
        assert!(!tested.is_relaxed());
    }
    #[test]
    fn is_feasible_iff_marked_so() {
        let mut tested= NodeData::new(NodeIndex(65), Rc::new(42), true);
        assert_eq!(false, tested.is_feasible());

        tested.set_feasible(true);
        assert_eq!(true, tested.is_feasible());

        tested.set_feasible(false);
        assert_eq!(false, tested.is_feasible());

        tested.set_feasible(true);
        assert_eq!(true, tested.is_feasible());
    }
    #[test]
    fn is_relaxed_iff_marked_exact_and_not_relaxed() {
        let tested= NodeData::new(NodeIndex(65), Rc::new(42), true);
        assert_eq!(false, tested.is_exact());

        let mut tested= NodeData::new(NodeIndex(65), Rc::new(42), false);
        assert_eq!(true, tested.is_exact());

        tested.set_exact(false);
        assert_eq!(false, tested.is_exact());

        tested.set_exact(true);
        assert_eq!(true, tested.is_exact());

        // it is relaxed: it cant be made exact again
        let mut tested= NodeData::new(NodeIndex(65), Rc::new(42), true);
        assert_eq!(false, tested.is_exact());
        tested.set_exact(true);
        assert_eq!(false, tested.is_exact());
    }
    #[test]
    fn value_returns_the_longest_path_from_the_root() {
        let mut tested= NodeData::new(NodeIndex(65), Rc::new(42), true);
        assert_eq!(isize::min_value(), tested.value());

        tested.lp_from_top = 999;
        assert_eq!(999, tested.value());

        tested.lp_from_bot = 888;
        assert_eq!(999, tested.value());
    }
}

#[cfg(test)]
mod test_layerdata {
    use crate::implementation::mdd::deep::mddgraph::{LayerData, LayerIndex};

    #[test]
    fn when_start_and_end_are_the_same_the_layer_is_empty() {
        let layer = LayerData{my_id: LayerIndex(0), start: 0, end: 0};
        assert_eq!(0, layer.width());

        let layer = LayerData{my_id: LayerIndex(34), start: 240, end: 240};
        assert_eq!(0, layer.width());

        let layer = LayerData{my_id: LayerIndex(7), start: 9, end: 10};
        assert_ne!(0, layer.width());
    }
    #[test]
    fn width_equals_the_number_of_nodes_in_the_layer() {
        let layer = LayerData{my_id: LayerIndex(0), start: 0, end: 0};
        assert_eq!(0, layer.width());

        let layer = LayerData{my_id: LayerIndex(7), start: 9, end: 10};
        assert_eq!(1, layer.width());

        let layer = LayerData{my_id: LayerIndex(9), start: 9, end: 19};
        assert_eq!(10, layer.width());
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod test_graph {
    use std::cmp::Ordering;
    use std::rc::Rc;

    use metrohash::MetroHashMap;
    use mock_it::verify;

    use crate::abstraction::heuristics::SelectableNode;
    use crate::common::{Decision, Variable};
    use crate::implementation::mdd::deep::mddgraph::{EdgeIndex, EdgeState, Graph, LayerData, LayerIndex, NodeIndex};
    use crate::test_utils::MockConfig;

    #[test]
    fn default_graph_is_empty() {
        let g = Graph::<usize>::default();

        assert!(g.root().is_err());
        assert!(g.state.is_empty());
        assert!(g.lel.is_none());
        assert_eq!(0, g.nb_nodes());
        assert_eq!(0, g.nb_edges());
    }
    #[test]
    fn default_graph_has_one_empty_layer() {
        let g = Graph::<usize>::default();

        assert_eq!(1, g.layers.len());
        assert_eq!(0, g.layers[0].width());
    }

    #[test]
    fn a_new_graph_is_empty() {
        let g = Graph::<usize>::new();

        assert!(g.root().is_err());
        assert!(g.state.is_empty());
        assert!(g.lel.is_none());
        assert_eq!(0, g.nb_nodes());
        assert_eq!(0, g.nb_edges());
    }
    #[test]
    fn a_new_graph_has_one_empty_layer() {
        let g = Graph::<usize>::new();

        assert_eq!(1, g.layers.len());
        assert_eq!(0, g.layers[0].width());
    }
    #[test]
    fn clear_resets_the_graph_state_as_if_it_were_fresh() {
        let (_, mut g) = example_graph();
        g.clear();

        assert!(g.root().is_err());
        assert_eq!(0, g.nb_nodes());
        assert_eq!(0, g.nb_edges());
        assert_eq!(1, g.layers.len());
        assert_eq!(0, g.layers[0].width());
    }
    #[test]
    fn add_root_increases_the_width_of_first_layer() {
        let mut g = Graph::new();
        assert_eq!(0, g.layers[0].width());
        g.add_root(Rc::new(24), 12);
        assert_eq!(1, g.layers[0].width());
    }
    #[test]
    fn add_root_increases_the_nodes_count() {
        let mut g = Graph::new();
        assert_eq!(0, g.nb_nodes());
        g.add_root(Rc::new(24), 12);
        assert_eq!(1, g.nb_nodes());
    }
    #[test]
    fn add_root_insert_root_state_in_state_map() {
        let mut g = Graph::new();
        assert!(g.state.is_empty());
        g.add_root(Rc::new(24), 12);
        assert_eq!(1, g.state.len());
    }
    #[test]
    fn add_root_creates_a_new_exact_disconnected_node() {
        let mut g = Graph::new();

        // there is no root before there is one
        assert!(g.root().is_err());

        let idx = g.add_root(Rc::new(24), 12);
        assert_eq!(NodeIndex(0), idx);
        assert_eq!(Ok(24),    g.root().map(|r| *r.state));
        assert_eq!(Ok(12),    g.root().map(|r| r.value()));
        assert_eq!(Ok(true),  g.root().map(|r| r.is_exact()));
        assert_eq!(Ok(false), g.root().map(|r| r.is_relaxed()));
        assert_eq!(Ok(false), g.root().map(|r| r.is_feasible()));
    }
    #[test]
    fn add_root_does_not_override_previous_root() {
        let mut g = Graph::new();

        let idx = g.add_root(Rc::new(24), 12);
        assert_eq!(NodeIndex(0),     idx);
        assert_eq!(Ok(NodeIndex(0)), g.root().map(|r| r.my_id));
        assert_eq!(Ok(24),           g.root().map(|r| *r.state()));
        assert_eq!(Ok(12),           g.root().map(|r| r.value()));
        assert_eq!(Ok(true),         g.root().map(|r| r.is_exact()));
        assert_eq!(Ok(false),        g.root().map(|r| r.is_relaxed()));
        assert_eq!(Ok(false),        g.root().map(|r| r.is_feasible()));

        let idx = g.add_root(Rc::new(56), 43);
        assert_eq!(NodeIndex(0),     idx);
        assert_eq!(Ok(NodeIndex(0)), g.root().map(|r| r.my_id));
        assert_eq!(Ok(24),           g.root().map(|r| *r.state()));
        assert_eq!(Ok(12),           g.root().map(|r| r.value()));
        assert_eq!(Ok(true),         g.root().map(|r| r.is_exact()));
        assert_eq!(Ok(false),        g.root().map(|r| r.is_relaxed()));
        assert_eq!(Ok(false),        g.root().map(|r| r.is_feasible()));
    }
    #[test]
    fn root_returns_the_first_node() {
        let mut g = Graph::new();

        // there is no root before there is one
        assert!(g.root().is_err());
        let idx = g.add_root(Rc::new(24), 12);
        assert_eq!(NodeIndex(0),     idx);
        assert_eq!(idx, g.root().unwrap().my_id);
        assert!(std::ptr::eq(&g.nodes[0], g.root().unwrap()));
    }
    #[test]
    fn nb_node_counts_the_number_of_nodes_in_a_graph() {
        let mut g = Graph::new();
        assert_eq!(0, g.nb_nodes());

        for i in 0..5 {
            g.add_node(i);
            assert_eq!(i+1, g.nb_nodes());
        }

        // Works also when nodes are on multiple layers
        let (_, g) = example_graph();
        assert_eq!(10, g.nb_nodes());
    }
    #[test]
    fn nb_edges_counts_the_number_of_edges_in_a_graph() {
        let mut g = Graph::new();
        assert_eq!(0, g.nb_nodes());

        let r = g.add_root(Rc::new(24), 0);
        g.add_layer();
        let n = g.add_node(12);
        g.add_edge(r, n, Decision{variable: Variable(0), value: 4}, 4);
        assert_eq!(1, g.nb_edges());

        // Works also when nodes are on multiple layers
        let (_, g) = example_graph();
        assert_eq!(13, g.nb_edges());
    }
    #[test]
    fn graph_is_exact_as_long_as_it_has_not_been_relaxed() {
        let mut g = Graph::new();
        assert!(g.is_exact());

        let r = g.add_root(Rc::new(24), 0);
        g.add_layer();
        assert!(g.is_exact());

        let n1 = g.add_node(1);
        g.add_edge(r, n1, Decision{variable: Variable(0), value: 1}, 1);
        assert!(g.is_exact());
        let n2 = g.add_node(2);
        g.add_edge(r, n2, Decision{variable: Variable(0), value: 2}, 2);
        assert!(g.is_exact());

        g.relax_last(1, &MockConfig::default());
        assert!(!g.is_exact());

        let (_, g) = example_graph();
        assert!(g.is_exact());
    }
    #[test]
    fn graph_is_exact_as_long_as_it_has_not_been_restricted() {
        let mut g = Graph::new();
        assert!(g.is_exact());

        let r = g.add_root(Rc::new(24), 0);
        g.add_layer();
        assert!(g.is_exact());

        let n1 = g.add_node(1);
        g.add_edge(r, n1, Decision{variable: Variable(0), value: 1}, 1);
        assert!(g.is_exact());
        let n2 = g.add_node(2);
        g.add_edge(r, n2, Decision{variable: Variable(0), value: 2}, 2);
        assert!(g.is_exact());

        g.restrict_last(1, &MockConfig::default());
        assert!(!g.is_exact());

        let (_, g) = example_graph();
        assert!(g.is_exact());
    }

    #[test]
    fn none_has_a_trivial_exact_best_path() {
        let (_, g) = example_graph();
        assert!(g.has_exact_best_path(None));
    }

    #[test]
    fn node_has_exact_best_path_when_it_is_exact() {
        let (idx, g) = example_graph();
        assert!(g.nodes[idx[&'x'].0].is_exact());
        assert!(g.has_exact_best_path(Some(idx[&'x'])));
    }
    #[test]
    fn node_has_exact_best_path_when_it_is_not_exact_but_its_longest_path_is_exact() {
        let (idx, mut g) = example_graph();
        let z = &mut g.nodes[idx[&'z'].0]; // z is not along the best path of x
        z.flags.set_relaxed(true);

        let b = &mut g.nodes[idx[&'b'].0];
        b.flags.set_exact(false);
        let c = &mut g.nodes[idx[&'c'].0];
        c.flags.set_exact(false);
        let d = &mut g.nodes[idx[&'d'].0];
        d.flags.set_exact(false);
        let e = &mut g.nodes[idx[&'e'].0];
        e.flags.set_exact(false);
        let f = &mut g.nodes[idx[&'f'].0];
        f.flags.set_exact(false);
        let x = &mut g.nodes[idx[&'x'].0];
        x.flags.set_exact(false);

        assert!(!g.nodes[idx[&'x'].0].is_exact());
        assert!(g.has_exact_best_path(Some(idx[&'x'])));
    }
    #[test]
    fn node_has_no_exact_best_path_when_longest_path_crosses_relaxed_node() {
        let (idx, mut g) = example_graph();
        let b = &mut g.nodes[idx[&'b'].0]; // b is on the best path
        b.flags.set_relaxed(true);

        let b = &mut g.nodes[idx[&'b'].0];
        b.flags.set_exact(false);
        let c = &mut g.nodes[idx[&'c'].0];
        c.flags.set_exact(false);
        let d = &mut g.nodes[idx[&'d'].0];
        d.flags.set_exact(false);
        let e = &mut g.nodes[idx[&'e'].0];
        e.flags.set_exact(false);
        let f = &mut g.nodes[idx[&'f'].0];
        f.flags.set_exact(false);
        let x = &mut g.nodes[idx[&'x'].0];
        x.flags.set_exact(false);

        assert!(!g.nodes[idx[&'x'].0].is_exact());
        assert!(!g.has_exact_best_path(Some(idx[&'x'])));
    }
    #[test]
    fn node_has_no_exact_best_path_when_it_is_relaxed() {
        let (idx, mut g) = example_graph();
        let x = &mut g.nodes[idx[&'x'].0]; // b is on the best path
        x.flags.set_relaxed(true);
        assert!(g.nodes[idx[&'x'].0].is_relaxed());
        assert!(!g.has_exact_best_path(Some(idx[&'x'])));
    }
    #[test]
    fn has_exact_best_path_can_be_called_for_non_terminal_node() {
        // when it is exact
        let (idx, g) = example_graph();
        assert!(g.nodes[idx[&'b'].0].is_exact());
        assert!(g.has_exact_best_path(Some(idx[&'b'])));

        // when it is inexact but has exact best path
        let (idx, mut g) = example_graph();
        let z = &mut g.nodes[idx[&'z'].0];
        z.flags.set_relaxed(true);
        let b = &mut g.nodes[idx[&'b'].0];
        b.flags.set_exact(false);
        assert!(!g.nodes[idx[&'b'].0].is_exact());
        assert!(g.has_exact_best_path(Some(idx[&'b'])));

        // when it is inexact and has no exact best
        let (idx, mut g) = example_graph();
        let a = &mut g.nodes[idx[&'a'].0]; // b is on the best path
        a.flags.set_relaxed(true);
        let b = &mut g.nodes[idx[&'b'].0];
        b.flags.set_exact(false);
        assert!(!g.nodes[idx[&'b'].0].is_exact());
        assert!(!g.has_exact_best_path(Some(idx[&'b'])));

        // when it is relaxed
        let (idx, mut g) = example_graph();
        let b = &mut g.nodes[idx[&'b'].0]; // b is on the best path
        b.flags.set_relaxed(true);
        assert!(!g.nodes[idx[&'b'].0].is_exact());
        assert!(!g.has_exact_best_path(Some(idx[&'b'])));
    }
    #[test]
    fn parents_returns_an_iterator_over_the_parent_nodes() {
        let (idx, g) = example_graph();

        // r has no parent
        let parent_ids : Vec<char> = g.parents(NodeIndex(0)).map(|n| *n.state()).collect();
        let no_parents : Vec<char> = vec![];
        assert_eq!(no_parents, parent_ids);

        // b has two parent
        let b_id = idx[&'b'];
        let parents : Vec<char> = vec!['a', 'z'];
        let mut parent_ids : Vec<char> = g.parents(b_id).map(|n| *n.state()).collect();
        parent_ids.sort_unstable_by_key(|n| *n);
        assert_eq!(parents, parent_ids);

        // x has four parent
        let x_id = idx[&'x'];
        let parents : Vec<char> = vec!['c', 'd', 'e', 'f'];
        let mut parent_ids : Vec<char> = g.parents(x_id).map(|n| *n.state()).collect();
        parent_ids.sort_unstable_by_key(|n| *n);
        assert_eq!(parents, parent_ids);
    }

    #[test]
    fn add_node_adds_a_new_node_when_it_does_not_exists() {
        let mut g = Graph::new();
        g.add_root(Rc::new('a'), 0);
        g.add_layer();

        let idx = g.add_node('b');
        assert_eq!(2, g.nb_nodes());
        assert_eq!(NodeIndex(1), idx);
    }
    #[test]
    fn add_node_returns_existing_node_when_one_with_same_state_belongs_to_current_layer() {
        let mut g = Graph::new();
        g.add_root(Rc::new('a'), 0);
        g.add_layer();

        let b_idx = g.add_node('b');
        assert_eq!(2, g.nb_nodes());
        assert_eq!(NodeIndex(1), b_idx);

        let c_idx = g.add_node('c');
        assert_eq!(3, g.nb_nodes());
        assert_eq!(NodeIndex(2), c_idx);

        let b2_idx = g.add_node('b');
        assert_eq!(3, g.nb_nodes());
        assert_eq!(NodeIndex(1), b2_idx);
    }
    #[test]
    fn add_node_performs_no_inter_layer_reduction() {
        let mut g = Graph::new();
        g.add_root(Rc::new('a'), 0);
        g.add_layer();

        let b_idx = g.add_node('b');
        assert_eq!(2, g.nb_nodes());
        assert_eq!(NodeIndex(1), b_idx);

        let c_idx = g.add_node('c');
        assert_eq!(3, g.nb_nodes());
        assert_eq!(NodeIndex(2), c_idx);
        g.add_layer();

        let b2_idx = g.add_node('b');
        assert_eq!(4, g.nb_nodes());
        assert_eq!(NodeIndex(3), b2_idx);
    }

    #[test]
    fn add_edge_just_adds_an_edge_with_given_information() {
        let mut g = Graph::new();
        let a = g.add_root(Rc::new('a'), 0);
        g.add_layer();
        let b = g.add_node('b');
        let c = g.add_node('c');

        let d1 = Decision{variable: Variable(0), value: 0};
        let d2 = Decision{variable: Variable(0), value: 1};

        assert_eq!(0, g.nb_edges());
        let e1 = g.add_edge(a, b, d1, 1);
        assert_eq!(1, g.nb_edges());
        assert_eq!(EdgeState{ src: a, dst: b, decision: d1, weight: 1 }, g.edges[e1.0].state);

        let e2 = g.add_edge(a, c, d2, 3);
        assert_eq!(2, g.nb_edges());
        assert_eq!(EdgeState{ src: a, dst: c, decision: d2, weight: 3 }, g.edges[e2.0].state);

        // that's a duplicate of e1 but we dont care
        let e3 = g.add_edge(a, b, d1, 1);
        assert_eq!(3, g.nb_edges());
        assert_eq!(EdgeState{ src: a, dst: b, decision: d1, weight: 1 }, g.edges[e3.0].state);
    }

    #[test]
    fn branch_creates_target_node_and_adds_an_edge() {
        let mut g = Graph::new();
        let a = g.add_root(Rc::new('a'), 0);
        g.add_layer();

        assert_eq!(1, g.nb_nodes());
        assert_eq!(0, g.nb_edges());

        let d1 = Decision{variable: Variable(0), value: 0};
        g.branch(a, 'b', d1, 1);
        assert_eq!(2, g.nb_nodes());
        assert_eq!(1, g.nb_edges());
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d1, weight: 1}, g.edges[0].state);
    }
    #[test]
    fn branch_reuses_target_node_and_adds_an_edge() {
        let mut g = Graph::new();
        let a = g.add_root(Rc::new('a'), 0);
        g.add_layer();

        assert_eq!(1, g.nb_nodes());
        assert_eq!(0, g.nb_edges());

        let d1 = Decision{variable: Variable(0), value: 0};
        let d2 = Decision{variable: Variable(0), value: 1};

        g.branch(a, 'b', d1, 1);
        assert_eq!(2, g.nb_nodes());
        assert_eq!(1, g.nb_edges());
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d1, weight: 1}, g.edges[0].state);

        g.branch(a, 'b', d2, 2);
        assert_eq!(2, g.nb_nodes());
        assert_eq!(2, g.nb_edges());
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d2, weight: 2}, g.edges[1].state);
    }
    #[test]
    fn when_branch_reuses_target_node_it_updates_inbound_edge_list() {
        let mut g = Graph::new();
        let a = g.add_root(Rc::new('a'), 0);
        g.add_layer();

        assert_eq!(1, g.nb_nodes());
        assert_eq!(0, g.nb_edges());

        let d1 = Decision{variable: Variable(0), value: 0};
        let d2 = Decision{variable: Variable(0), value: 1};

        g.branch(a, 'b', d1, 1);
        assert_eq!(2, g.nb_nodes());
        assert_eq!(1, g.nb_edges());
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d1, weight: 1},
                   g.edges[0].state);
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d1, weight: 1},
                   g.edges[g.nodes[1].inbound.unwrap().0].state);

        g.branch(a, 'b', d2, 2);
        assert_eq!(2, g.nb_nodes());
        assert_eq!(2, g.nb_edges());
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d2, weight: 2},
                   g.edges[1].state);
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d2, weight: 2},
                   g.edges[g.nodes[1].inbound.unwrap().0].state);
        assert_eq!(EdgeState{src: a, dst: NodeIndex(1), decision: d1, weight: 1},
                   g.edges[g.edges[g.nodes[1].inbound.unwrap().0].next.unwrap().0].state);
    }
    #[test]
    fn branch_will_make_target_node_inexact_if_one_of_the_parents_is_inexact() {
        let mut g = Graph::new();
        let a = g.add_root(Rc::new('a'), 0);
        g.add_layer();

        let d1 = Decision{variable: Variable(0), value: 0};
        let d2 = Decision{variable: Variable(0), value: 1};

        g.branch(a, 'b', d1, 1);
        g.branch(a, 'c', d2, 2);
        g.nodes.last_mut().unwrap().flags.set_relaxed(true);
        g.add_layer();

        let b  = NodeIndex(1);
        let c  = NodeIndex(2);
        let d3 = Decision{variable: Variable(1), value: 2};
        let d4 = Decision{variable: Variable(1), value: 3};

        g.branch(b, 'd', d3, 3);
        assert!(g.nodes.last().unwrap().is_exact());
        g.branch(c, 'd', d4, 4);
        assert!(!g.nodes.last().unwrap().is_exact());
    }
    #[test]
    fn branch_will_keep_the_best_parent_and_value_if_it_needs_to_be_updated() {
        let mut g = Graph::new();
        let a = g.add_root(Rc::new('a'), 0);
        g.add_layer();

        let d1 = Decision{variable: Variable(0), value: 0};
        let d2 = Decision{variable: Variable(0), value: 1};

        g.branch(a, 'b', d1, 1);
        g.branch(a, 'c', d2, 2);
        g.nodes.last_mut().unwrap().flags.set_relaxed(true);
        g.add_layer();

        let b  = NodeIndex(1);
        let c  = NodeIndex(2);
        let d3 = Decision{variable: Variable(1), value: 2};
        let d4 = Decision{variable: Variable(1), value: 3};

        g.branch(b, 'd', d3, 3);
        assert_eq!(b, g.edges[g.nodes.last().unwrap().best_parent.unwrap().0].state.src);
        assert_eq!(4, g.nodes.last().unwrap().lp_from_top);

        g.branch(c, 'd', d4, 4);
        assert_eq!(c, g.edges[g.nodes.last().unwrap().best_parent.unwrap().0].state.src);
        assert_eq!(6, g.nodes.last().unwrap().lp_from_top);
    }
    #[test]
    fn add_layer_adds_a_new_layer_starting_at_end_of_previous_one() {
        let mut g = Graph::new();
        g.add_root(Rc::new('a'), 0);

        g.add_layer();
        assert_eq!(1, g.layers.last().unwrap().start);

        g.add_node('x');
        g.add_node('y');
        g.add_node('z');
        g.add_layer();
        assert_eq!(4, g.layers.last().unwrap().start);
    }
    #[test]
    fn add_layer_clears_state_map(){
        let mut g = Graph::new();
        g.add_root(Rc::new('a'), 0);
        assert_eq!(1, g.state.len());

        g.add_layer();
        assert_eq!(0, g.state.len());

        g.add_node('x');
        g.add_node('y');
        g.add_node('z');
        assert_eq!(3, g.state.len());
        g.add_layer();
        assert_eq!(0, g.state.len());
    }
    #[test]
    fn current_layer_returns_current_layer() {
        let mut g = Graph::new();
        assert_eq!(LayerData{my_id: LayerIndex(0), start: 0, end:0}, g.current_layer());
        g.add_root(Rc::new('a'), 0);
        assert_eq!(LayerData{my_id: LayerIndex(0), start: 0, end:1}, g.current_layer());

        g.add_layer();
        assert_eq!(LayerData{my_id: LayerIndex(1), start: 1, end:1}, g.current_layer());
        g.add_node('x');
        g.add_node('y');
        g.add_node('z');
        assert_eq!(LayerData{my_id: LayerIndex(1), start: 1, end:4}, g.current_layer());
        g.add_layer();
        assert_eq!(LayerData{my_id: LayerIndex(2), start: 4, end:4}, g.current_layer());
    }
    #[test]
    fn layer_nodes_returns_a_slice_with_the_appropriate_nodes() {
        let (_, g) = example_graph();

        let expect : Vec<NodeIndex> = vec![NodeIndex(0)];
        let actual : Vec<NodeIndex> = g.layer_nodes(LayerIndex(0)).iter().map(|n| n.my_id).collect();
        assert_eq!(expect, actual);

        let expect : Vec<NodeIndex> = vec![NodeIndex(1), NodeIndex(2)];
        let actual : Vec<NodeIndex> = g.layer_nodes(LayerIndex(1)).iter().map(|n| n.my_id).collect();
        assert_eq!(expect, actual);

        let expect : Vec<NodeIndex> = vec![NodeIndex(3), NodeIndex(4)];
        let actual : Vec<NodeIndex> = g.layer_nodes(LayerIndex(2)).iter().map(|n| n.my_id).collect();
        assert_eq!(expect, actual);

        let expect : Vec<NodeIndex> = vec![NodeIndex(5), NodeIndex(6), NodeIndex(7), NodeIndex(8)];
        let actual : Vec<NodeIndex> = g.layer_nodes(LayerIndex(3)).iter().map(|n| n.my_id).collect();
        assert_eq!(expect, actual);

        let expect : Vec<NodeIndex> = vec![NodeIndex(9)];
        let actual : Vec<NodeIndex> = g.layer_nodes(LayerIndex(4)).iter().map(|n| n.my_id).collect();
        assert_eq!(expect, actual);
    }

    #[test]
    fn remember_lel_has_no_effect_when_lel_is_present() {
        let (_, mut g) = example_graph();
        // assuming that L1 is the LEL
        g.lel = Some(LayerIndex(1));
        // remember_lel will have no impact
        g.remember_lel();
        //
        assert!(g.lel.is_some());
        assert_eq!(LayerIndex(1), g.lel.unwrap());
    }
    #[test]
    fn remember_lel_remembers_the_last_exact_layer() {
        let (_, mut g) = example_graph();

        // assuming we turn the last layer into an inexact one, the LEL would be
        // the forelast layer.
        g.remember_lel();
        //
        assert!(g.lel.is_some());
        assert_eq!(g.lel.unwrap().0, g.current_layer().my_id.0 - 1);
    }

    #[test]
    fn best_terminal() {
        let (ids, g) = example_graph();
        let x = ids[&'x'];

        assert_eq!(Some(x), g.find_best_terminal_node());
    }
    #[test]
    fn best_solution() {
        let (_, g) = example_graph();
        assert_eq!(
            g.best_solution(),
            vec![Decision{variable: Variable(3), value: 1},
                 Decision{variable: Variable(2), value: 4},
                 Decision{variable: Variable(1), value: 1},
                 Decision{variable: Variable(0), value: 0}])
    }

    #[test]
    fn longest_path() {
        let (_, g) = example_graph();
        let x = g.state[&Rc::new('x')];

        assert_eq!(11, g.nodes[x.0].lp_from_top);

        let lp= g.longest_path(x);
        assert_eq!(
            lp,
            vec![Decision{variable: Variable(3), value: 1},
                 Decision{variable: Variable(2), value: 4},
                 Decision{variable: Variable(1), value: 1},
                 Decision{variable: Variable(0), value: 0}]
        )
    }

    #[test]
    fn test_add_node() {
        let (ids, g) = example_graph();

        let a = ids[&'a'];
        let b = ids[&'b'];
        let z = ids[&'z'];

        assert_eq!(vec![z, a], g.parents(b).map(|n|n.my_id).collect::<Vec<NodeIndex>>());
    }

    #[test]
    fn sort_does_not_change_the_graph_for_the_outer_world() {
        let (ids, mut g) = example_graph();
        g.sort_last_layer(|n1, n2| n1.my_id.0.cmp(&n2.my_id.0).reverse());

        let a = ids[&'a'];
        let b = ids[&'b'];
        let z = ids[&'z'];
        assert_eq!(vec![z, a], g.parents(b).map(|n| n.my_id).collect::<Vec<NodeIndex>>());
    }

    #[test]
    fn sort_ensures_that_state_map_is_coherent_with_new_position() {
        let (_, mut g) = example_graph();
        g.sort_last_layer(|n1, n2| n1.my_id.0.cmp(&n2.my_id.0).reverse());

        for (i, n) in g.nodes.iter().enumerate() {
            if g.state.contains_key(n.state.as_ref()) {
                assert_eq!(i, g.state[&n.state].0)
            }
        }
    }

    #[test]
    fn sort_ensures_that_nodes_id_are_coherent_with_their_position() {
        let (_, mut g) = example_graph();
        g.sort_last_layer(|n1, n2| n1.my_id.0.cmp(&n2.my_id.0).reverse());

        for (i, n) in g.nodes.iter().enumerate() {
            assert_eq!(i, n.my_id.0);
        }
    }

    #[test]
    fn sort_ensures_that_end_of_an_edge_is_coherent_with_its_node() {
        let (_, mut g) = example_graph();
        g.sort_last_layer(|n1, n2| n1.my_id.0.cmp(&n2.my_id.0).reverse());

        for (i, n) in g.nodes.iter().enumerate() {
            assert_eq!(i, n.my_id.0);

            if g.state.contains_key(n.state.as_ref()) {
                assert_eq!(i, g.state[n.state.as_ref()].0);
            }

            let mut e = n.inbound;
            while let Some(edge_id) = e {
                let edge = g.edges[edge_id.0];
                assert_eq!(i, edge.state.dst.0);
                e = edge.next;
            }
        }
    }

    #[test]
    fn test_delete_edges() {
        let (ids, mut g) = example_graph();
        let b = ids[&'b'];

        let mut v = vec![];
        g.add_edges_of_into(b, &mut v);
        g.delete_all_edges_in_list(&mut v);

        for n in g.nodes.iter() {
            if n.my_id != b {
                let mut it = n.inbound;
                while let Some(eid) = it {
                    let edge = g.edges[eid.0];
                    assert_eq!(n.my_id, edge.state.dst);
                    it = edge.next;
                }
            }
        }

        for e in g.edges.iter() {
            let dst = e.state.dst;
            let mut it  = e.next;
            while let Some(nxt_id) = it {
                let nxt_e = g.edges[nxt_id.0];
                assert_eq!(dst, nxt_e.state.dst);
                it = nxt_e.next;
            }
        }
    }

    #[test]
    fn restrict_last_remembers_the_last_exact_layer_if_needed() {
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert!(g.lel.is_none());

        // first time, a lel is saved
        g.restrict_last(2, &MockConfig::default());
        assert!(g.lel.is_some());
        assert_eq!(Some(LayerIndex(0)), g.lel);

        // but it is not updated after a subsequent restrict
        g.add_layer();
        g.branch(r_id, 37, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 38, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 39, Decision{variable: Variable(0), value: 3}, 1);
        g.branch(r_id, 40, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 41, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 42, Decision{variable: Variable(0), value: 3}, 1);
        g.restrict_last(2, &MockConfig::default());
        assert!(g.lel.is_some());
        assert_eq!(Some(LayerIndex(0)), g.lel);
    }
    #[test]
    fn restrict_last_makes_the_graph_inexact() {
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert!(g.is_exact());

        // first time, a lel is saved
        g.restrict_last(2, &MockConfig::default());
        assert!(!g.is_exact());
    }
    #[test]
    fn restrict_last_layer_enforces_the_given_width() {
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, g.current_layer().width());

        g.restrict_last(2, &MockConfig::default());
        assert_eq!(2, g.current_layer().width());

        g.restrict_last(1, &MockConfig::default());
        assert_eq!(1, g.current_layer().width());
    }
    #[test]
    fn restrict_last_layer_forgets_the_state_of_deleted_nodes() {
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        let c = MockConfig::default();
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states_before = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states_before.sort_unstable();
        assert_eq!(vec![34, 35, 36], states_before);

        g.restrict_last(2, &c);
        let mut states_after = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states_after.sort_unstable();
        assert_eq!(vec![35, 36], states_after);
    }
    #[test]
    fn restrict_last_layer_will_not_bring_about_new_nodes() {
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        let c = MockConfig::default();
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.restrict_last(2, &c);
        assert_eq!(36, *g.nodes[g.state[&36].0].state());
        assert_eq!( 4,  g.nodes[g.state[&36].0].value());
        assert_eq!(35, *g.nodes[g.state[&35].0].state());
        assert_eq!( 5,  g.nodes[g.state[&35].0].value());
    }
    #[test]
    fn restrict_last_layer_uses_node_selection_heuristic_to_rank_nodes_and_renames_others() {
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        let c = MockConfig::default();
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        // 1. check the appropriate heuristic is used
        let mut states_before = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states_before.sort_unstable();
        assert_eq!(vec![34, 35, 36], states_before);

        //
        let n_34 = g.state[&34];
        let n_35 = g.state[&35];
        let n_36 = g.state[&36];
        assert_eq!(NodeIndex(1), n_34);
        assert_eq!(NodeIndex(2), n_35);
        assert_eq!(NodeIndex(3), n_36);
        assert!(g.nodes.iter().enumerate().all(|(i, n)| i == n.my_id.0));

        g.restrict_last(2, &c);
        assert!(verify(c.compare.was_called_with((35, 34))));
        let mut states_after = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states_after.sort_unstable();
        assert_eq!(vec![35, 36], states_after);

        // 2. Check renaming
        assert!(g.nodes.iter().enumerate().all(|(i, n)| i == n.my_id.0));
        let n_35 = g.state[&35];
        let n_36 = g.state[&36];
        assert_eq!(NodeIndex(1), n_36);
        assert_eq!(NodeIndex(2), n_35);
    }
    #[test]
    fn restrict_last_deletes_the_edges_of_deleted_nodes_and_renames_others() {
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        let c = MockConfig::default();
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(3, g.edges.len());
        assert_eq!(0, g.nodes[1].inbound.unwrap().0); // r->34
        assert_eq!(1, g.nodes[2].inbound.unwrap().0); // r->35
        assert_eq!(2, g.nodes[3].inbound.unwrap().0); // r->36

        g.restrict_last(2, &c);
        assert_eq!(2, g.edges.len());

        // r->36
        assert_eq!(0, g.nodes[1].inbound.unwrap().0);
        assert_eq!(NodeIndex(0), g.edges[0].state.src);
        assert_eq!(NodeIndex(1), g.edges[0].state.dst);
        assert!(g.edges[0].next.is_none());

        // r->35
        assert_eq!(1, g.nodes[2].inbound.unwrap().0);
        assert_eq!(NodeIndex(0), g.edges[1].state.src);
        assert_eq!(NodeIndex(2), g.edges[1].state.dst);
        assert!(g.edges[1].next.is_none());
    }

    #[test]
    fn relax_last_remembers_the_last_exact_layer_if_needed() {
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert!(g.lel.is_none());

        // first time, a lel is saved
        g.relax_last(2, &MockConfig::default());
        assert!(g.lel.is_some());
        assert_eq!(Some(LayerIndex(0)), g.lel);

        // but it is not updated after a subsequent restrict
        g.add_layer();
        g.branch(r_id, 37, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 38, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 39, Decision{variable: Variable(0), value: 3}, 1);
        g.branch(r_id, 40, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 41, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 42, Decision{variable: Variable(0), value: 3}, 1);
        g.relax_last(2, &MockConfig::default());
        assert!(g.lel.is_some());
        assert_eq!(Some(LayerIndex(0)), g.lel);
    }
    #[test]
    fn relax_last_makes_the_graph_inexact() {
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert!(g.is_exact());

        // first time, a lel is saved
        g.relax_last(2, &MockConfig::default());
        assert!(!g.is_exact());
    }
    #[test]
    fn relax_last_layer_enforces_the_given_max_width() {
        let c = MockConfig::default();
        // Merged state is 37 (does not exist prior to relaxation)
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, g.current_layer().width());

        g.relax_last(2, &c);
        let cur = g.layer_nodes(g.current_layer().my_id).iter().map(|n| *n.state()).collect::<Vec<usize>>();
        assert_eq!(vec![36, 37], cur);
        assert_eq!(2, g.current_layer().width());
    }
    #[test]
    fn relax_last_layer_can_produce_lesser_max_width_when_merged_state_already_exists() {
        let c = MockConfig::default();
        // Merged state is 36 (it already exists)
        c.merge_states.given(vec![35, 34]).will_return(36);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, g.current_layer().width());

        g.relax_last(2, &c);
        let cur = g.layer_nodes(g.current_layer().my_id).iter().map(|n| *n.state()).collect::<Vec<usize>>();
        assert_eq!(vec![36], cur);
        assert_eq!(1, g.current_layer().width());
    }
    #[test]
    fn test_relax_last_layer_when_width_is_one() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![36, 35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(3, g.current_layer().width());

        g.relax_last(1, &c);
        let cur = g.layer_nodes(g.current_layer().my_id).iter().map(|n| *n.state()).collect::<Vec<usize>>();
        assert_eq!(vec![37], cur);
        assert_eq!(1, g.current_layer().width());
    }
    #[test]
    fn relax_last_layer_forgets_the_state_of_deleted_nodes() {
        let c = MockConfig::default();
        // Merged state is 37 (does not exist prior to relaxation)
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last(2, &c);
        let mut states = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![36, 37], states);
    }
    #[test]
    fn relax_last_remembers_the_state_of_merged_node() {
        let c = MockConfig::default();
        // Merged state is 37 (does not exist prior to relaxation)
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last(2, &c);
        let mut states = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![36, 37], states);
    }
    #[test]
    fn relax_last_remembers_the_state_of_merged_node_even_if_it_corresponds_to_that_of_one_that_was_deleted() {
        let c = MockConfig::default();
        // Merged state is 35 (selected for deletion)
        c.merge_states.given(vec![35, 34]).will_return(35);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        let mut states = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![34, 35, 36], states);

        g.relax_last(2, &c);
        let mut states = g.state.keys().map(|k| **k).collect::<Vec<usize>>();
        states.sort_unstable();
        assert_eq!(vec![35, 36], states);
    }
    #[test]
    fn relax_last_layer_will_bring_about_one_node() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last(2, &c);
        // 36
        assert_eq!(36, *g.nodes[g.state[&36].0].state());
        assert_eq!( 4,  g.nodes[g.state[&36].0].value());

        // 37 (mock relaxes everything to 0)
        assert_eq!(37, *g.nodes[g.state[&37].0].state());
        assert_eq!( 3,  g.nodes[g.state[&37].0].value());
    }
    #[test]
    fn relax_last_layer_will_bring_about_one_node_unless_merged_state_is_already_known() {
        let c = MockConfig::default();
        // Merged state is 35 (selected for deletion)
        c.merge_states.given(vec![35, 34]).will_return(36);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last(2, &c);
        // 36
        assert_eq!(36, *g.nodes[g.state[&36].0].state());
        assert_eq!( 4,  g.nodes[g.state[&36].0].value());
    }
    #[test]
    fn relax_last_layer_uses_node_selection_heuristic_to_rank_nodes_and_renames_others() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(NodeIndex(3), g.nodes[g.state[&36].0].my_id);

        g.relax_last(2, &c);
        assert!(verify(c.compare.was_called_with((35, 34))));
        assert_eq!(NodeIndex(1), g.nodes[g.state[&36].0].my_id);
        assert_eq!(NodeIndex(2), g.nodes[g.state[&37].0].my_id);
        assert!(g.nodes.iter().enumerate().all(|(i, n)| n.my_id.0 == i));
    }
    #[test]
    fn relax_last_relaxes_the_weight_of_all_redirected_edges() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // relax r-34
        c.relax_edge
            .given((33, 34, 37, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(1);
        // relax r-35
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(0);
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(0);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last(2, &c);
        assert!(verify(c.relax_edge.was_called_with((33, 34, 37, Decision{variable: Variable(0), value: 1}, 3))));
        assert!(verify(c.relax_edge.was_called_with((33, 35, 37, Decision{variable: Variable(0), value: 2}, 2))));
        assert!(verify(c.relax_edge.was_called_with((33, 35, 37, Decision{variable: Variable(0), value: 4}, 4))));
    }

    #[test]
    fn relax_last_redirects_the_edges_of_deleted_nodes_to_the_merged_node() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // relax r-34
        c.relax_edge
            .given((33, 34, 37, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(1);
        // relax r-35
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(0);
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(0);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);
        assert_eq!(4, g.nb_edges());

        g.relax_last(2, &c);
        assert_eq!(4, g.nb_edges());
        // used to go to 34
        assert_eq!(NodeIndex(0), g.edges[0].state.src);
        assert_eq!(NodeIndex(2), g.edges[0].state.dst);
        // used to go to 35
        assert_eq!(NodeIndex(0), g.edges[1].state.src);
        assert_eq!(NodeIndex(2), g.edges[1].state.dst);
        // used to go to 35
        assert_eq!(NodeIndex(0), g.edges[2].state.src);
        assert_eq!(NodeIndex(2), g.edges[2].state.dst);
        // still goes to 36
        assert_eq!(NodeIndex(0), g.edges[3].state.src);
        assert_eq!(NodeIndex(1), g.edges[3].state.dst);
    }
    #[test]
    fn relax_last_will_not_update_best_parent_and_value_when_there_is_no_improvement() {
        let c = MockConfig::default();
        // Merged state is 36
        c.merge_states.given(vec![35, 34]).will_return(36);
        // relax r-34
        c.relax_edge
            .given((33, 34, 36, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(0);
        // relax r-35
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(0);
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(0);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(Some(EdgeIndex(3)), g.nodes[g.state[&36].0].best_parent);
        assert_eq!(4, g.nodes[g.state[&36].0].lp_from_top);

        g.relax_last(2, &c);
        assert_eq!(Some(EdgeIndex(3)), g.nodes[g.state[&36].0].best_parent);
        assert_eq!(4, g.nodes[g.state[&36].0].lp_from_top);
    }
    #[test]
    fn relax_last_updates_best_parent_and_value_if_merged_node_already_exists() {
        let c = MockConfig::default();
        // Merged state is 36
        c.merge_states.given(vec![35, 34]).will_return(36);
        // relax r-34
        c.relax_edge
            .given((33, 34, 36, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(3);
        // relax r-35
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(2);
        c.relax_edge
            .given((33, 35, 36, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(17);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(Some(EdgeIndex(3)), g.nodes[g.state[&36].0].best_parent);
        assert_eq!(4, g.nodes[g.state[&36].0].lp_from_top);

        g.relax_last(2, &c);
        assert_eq!(Some(EdgeIndex(2)), g.nodes[g.state[&36].0].best_parent);
        assert_eq!(20, g.nodes[g.state[&36].0].lp_from_top);
    }
    #[test]
    fn relax_last_updates_best_parent_and_value_relaxed_edge_improve_value() {
        let c = MockConfig::default();
        // Merged state is 37
        c.merge_states.given(vec![35, 34]).will_return(37);
        // relax r-34
        c.relax_edge
            .given((33, 34, 37, Decision{variable: Variable(0), value: 1}, 3))
            .will_return(6);
        // relax r-35
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 2}, 2))
            .will_return(8);
        c.relax_edge
            .given((33, 35, 37, Decision{variable: Variable(0), value: 4}, 4))
            .will_return(10);

        // Node selection orders node in natural order but selection keeps
        // the highest values only. So it should keep 35 and 36.
        c.compare.given((34, 35)).will_return(Ordering::Less);
        c.compare.given((34, 36)).will_return(Ordering::Less);
        c.compare.given((34, 34)).will_return(Ordering::Equal);
        c.compare.given((35, 34)).will_return(Ordering::Greater);
        c.compare.given((35, 36)).will_return(Ordering::Less);
        c.compare.given((35, 35)).will_return(Ordering::Equal);
        c.compare.given((36, 34)).will_return(Ordering::Greater);
        c.compare.given((36, 35)).will_return(Ordering::Greater);
        c.compare.given((36, 36)).will_return(Ordering::Equal);

        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();

        // edge 0
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        // edge 1
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        // edge 2
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        // edge 3
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        assert_eq!(Some(EdgeIndex(3)), g.nodes[g.state[&36].0].best_parent);
        assert_eq!(4, g.nodes[g.state[&36].0].lp_from_top);

        g.relax_last(2, &c);
        assert_eq!(Some(EdgeIndex(2)), g.nodes[g.state[&37].0].best_parent);
        assert_eq!(13, g.nodes[g.state[&37].0].lp_from_top);

        assert_eq!(Some(EdgeIndex(3)), g.nodes[g.state[&36].0].best_parent);
        assert_eq!(4, g.nodes[g.state[&36].0].lp_from_top);
    }

    #[test]
    #[should_panic]
    fn relax_last_panics_if_width_is_0() {
        let c = MockConfig::default();
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last(0, &c);
    }
    #[test]
    #[should_panic]
    fn relax_last_panics_if_layer_is_not_broad_enough() {
        let c = MockConfig::default();
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.relax_last(10, &c);
    }
    #[test]
    #[should_panic]
    fn restrict_last_panics_if_layer_is_not_broad_enough() {
        let c = MockConfig::default();
        let mut g = Graph::new();
        let  r_id = g.add_root(Rc::new(33), 3);
        g.add_layer();
        g.branch(r_id, 34, Decision{variable: Variable(0), value: 1}, 3);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 2}, 2);
        g.branch(r_id, 35, Decision{variable: Variable(0), value: 4}, 4);
        g.branch(r_id, 36, Decision{variable: Variable(0), value: 3}, 1);

        g.restrict_last(10, &c);
    }

    /// This function produces the following example graph.
    /// Its longest path is x0=0,x1=3,x2=4,x3=1.
    /// The lenght of the longest path is 11.
    ///
    /// ```plain
    /// L0  ||                        r
    ///     ||                     -------
    ///     ||                    /       \
    /// L1  ||                x0=0@1     x0=2@0
    ///     ||                /              \
    ///     ||              a                z
    ///     ||             / \             /
    /// L2  ||       x1=2@6  x1=1@3   x1=26@0
    ///     ||         /         \    /
    ///     ||       c             b
    ///     ||            --------------------
    ///     ||           /     |       |      \
    ///     ||      x2=1@1  x2=2@2  x2=3@3  x2=4@4
    ///     ||        /        |       |        \
    /// L3  ||       c         d       e         f
    ///     ||        \        |       |        /
    ///     ||       x3=1@3  x3=1@3  x3=1@3 x3=1@3
    ///     ||          \      |       |     /
    ///     ||           --------------------
    /// L4  ||                     x
    /// ```
    fn example_graph() -> (MetroHashMap<char, NodeIndex>, Graph<char>) {
        let mut g = Graph::new();

        let r_id = g.add_root(Rc::new('r'), 0);
        g.add_layer();

        // L0
        g.branch(r_id, 'a', Decision{variable: Variable(0), value: 0}, 1);
        g.branch(r_id, 'z', Decision{variable: Variable(0), value: 2}, 0);

        // L1
        let a_id  = g.state[&Rc::new('a')];
        let z_id  = g.state[&Rc::new('z')];
        g.add_layer();

        g.branch(a_id, 'b', Decision{variable: Variable(1), value: 1}, 3);
        g.branch(a_id, 'c', Decision{variable: Variable(1), value: 2}, 6);

        g.branch(z_id, 'b', Decision{variable: Variable(1), value:26}, 0);
        // L2
        let b_id  = g.state[&Rc::new('b')];
        let c_id  = g.state[&Rc::new('c')];
        g.add_layer();

        g.branch(b_id, 'c', Decision{variable: Variable(2), value: 1}, 1);
        g.branch(b_id, 'd', Decision{variable: Variable(2), value: 2}, 2);
        g.branch(b_id, 'e', Decision{variable: Variable(2), value: 3}, 3);
        g.branch(b_id, 'f', Decision{variable: Variable(2), value: 4}, 4);

        // L3
        let d_id  = g.state[&Rc::new('d')];
        let e_id  = g.state[&Rc::new('e')];
        let f_id  = g.state[&Rc::new('f')];
        g.add_layer();

        g.branch(c_id, 'x', Decision{variable: Variable(3), value: 1}, 3);
        g.branch(d_id, 'x', Decision{variable: Variable(3), value: 1}, 3);
        g.branch(e_id, 'x', Decision{variable: Variable(3), value: 1}, 3);
        g.branch(f_id, 'x', Decision{variable: Variable(3), value: 1}, 3);

        let mut ids = MetroHashMap::default();
        ids.insert('r', r_id);
        ids.insert('a', a_id);
        ids.insert('z', z_id);
        ids.insert('b', b_id);
        ids.insert('c', c_id);
        ids.insert('d', d_id);
        ids.insert('e', e_id);
        ids.insert('f', f_id);
        ids.insert('x', g.state[&Rc::new('x')]);

        (ids, g)
    }
}
