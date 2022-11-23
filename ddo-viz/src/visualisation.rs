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

//! DDO-Viz is meant to let you visualize the decision diagrams you have compiled

use std::fmt::Debug;

use ddo::{Decision, DecisionDiagram};
use rustc_hash::FxHashMap;
use derive_builder::Builder;
use crate::{common::{Node, Edge, NodeId}};

/// This trait should be implemented by the mdd that can be visualized
pub trait Visualisable : DecisionDiagram {
    /// Returns all the information required to build mdd.
    fn visualisation(&self) -> Visualisation<Self::State>;
}

/// This structure contains all the information one requires to produce an image of
/// some compiled mdd
pub struct Visualisation<'a, T> {
    /// The nodes that have been created when compiling the mdd
    pub nodes:  &'a [Node],
    /// The edges that have been created when compiling the mdd
    pub edges:  &'a [Edge],
    /// The state associated to each node that has been created when compiling the mdd
    pub states: &'a [T],
    /// A flag telling whether or not node[i] has been merged during compilation
    pub merged: &'a [bool],
    /// A flag telling whether or not node[i] has been deleted because of a restriction during compilation
    pub restricted: &'a [bool],
    /// A map that associates the relaxed nodes with the nodes that have been merged to produce the
    /// relaxed node
    pub groups: &'a FxHashMap<NodeId, Vec<NodeId>>,
    /// The nodes composing the final layer. This slice is going to be empty if the dd is infeasible
    pub terminal: &'a [NodeId],
}

/// This is how you configure the output visualisation e.g.
/// if you want to see the RUB, LocB and the nodes that have been merged
#[derive(Debug, Builder)]
pub struct VizConfig {
    /// This flag must be true (default) if you want to see the value of
    /// each node (length of the longest path)
    #[builder(default="true")]
    show_value: bool,
    /// This flag must be true (default) if you want to see the locb of
    /// each node (length of the longest path from the bottom)
    #[builder(default="true")]
    show_locb: bool,
    /// This flag must be true (default) if you want to see the rub of
    /// each node (fast upper bound)
    #[builder(default="true")]
    show_rub: bool,
    /// This flag must be true (default) if you want to see all nodes that
    /// have been deleted because of restrict operations
    #[builder(default="true")]
    show_restricted: bool,
    /// This flag must be true (default) if you want to see all nodes that
    /// have been deleted because of they have been merged into a relaxed node
    #[builder(default="true")]
    show_merged: bool,
}

impl <'a, T> Visualisation<'a, T> where T: Debug + 'a {
    /// This is the method you will want to use in order to create the output image you would like.
    /// Note: the output is going to be a string of (not compiled) 'dot'. This makes it easier for
    /// me to code and gives you the freedom to fiddle with the graph if needed.
    pub fn as_graphviz(&self, config: &VizConfig) -> String {
        let mut out = String::new();

        out.push_str("digraph {\n\tranksep = 3;\n\n");

        // Show all nodes
        for (id, _) in self.nodes.iter().enumerate() {
            if !config.show_merged && self.merged[id] {
                continue;
            }
            if !config.show_restricted && self.restricted[id] {
                continue;
            }
            out.push_str(&self.node(id, config));
            out.push_str(&self.edges_of(id));
        }
        // Add an overlay for the groups of nodes that have been merged
        if config.show_merged {
            for (merged, members) in self.groups.iter() {
                out.push_str(&Self::group(merged, members));
            }
        }

        // Finish the graph with a terminal node
        out.push_str(&self.add_terminal_node());

        out.push_str("}\n");
        out
    }

    /// Creates a string representation of one single node
    fn node(&self, id: usize, config: &VizConfig) -> String {
        let attributes = self.node_attributes(id, config);
        format!("\t{id} [{attributes}];\n")
    }
    /// Creates a string representation of the edges incident to one node
    fn edges_of(&self, id: usize) -> String {
        let mut out = String::new();
        let n = &self.nodes[id];
        let mut e = n.inbound;
        while let Some(eid) = e {
            let ee = self.edges[eid.0];
            out.push_str(&Self::edge(ee.from.0, id, ee.decision, ee.cost, e == n.best));
            
            e = ee.next;
        }
        out
    }
    /// Create a subgraph to visualize a group of nodes that have been merged.
    fn group(owner: &NodeId, members: &Vec<NodeId>) -> String {
        let owner = owner.0;
        let mut out = format!("\tsubgraph cluster_{owner} {{\n");
        out.push_str("\t\tstyle = filled;\n");
        out.push_str("\t\tcolor = \"#ff5050\";\n");

        // owner
        out.push_str(&format!("\t\t{};\n", owner));
        // members
        for x in members {
            out.push_str(&format!("\t\t{};\n", x.0));
        }

        out.push_str("\t};\n");
        out
    }
    /// Adds a terminal node (if the DD is feasible) and draws the edges entering that node from
    /// all the nodes of the terminal layer.
    fn add_terminal_node(&self) -> String {
        let mut out = String::new();
        if !self.terminal.is_empty() {
            let terminal = "\tterminal [shape=\"circle\", label=\"\", style=\"filled\", color=\"black\", group=\"terminal\"];\n";
            out.push_str(terminal);

            let vmax = self.terminal.iter().map(|n| self.nodes[n.0].value).max().unwrap_or(isize::MAX);
            for term in self.terminal.iter() {
                let value = self.nodes[term.0].value;
                if value == vmax {
                    out.push_str(&format!("\t{} -> terminal [penwidth=3];\n", term.0));
                } else {
                    out.push_str(&format!("\t{} -> terminal;\n", term.0));
                }
            }
        }
        out
    }
    /// Creates a string representation of one edge
    fn edge(from: usize, to: usize, decision: Decision, cost: isize, is_best: bool) -> String {
        let width = if is_best { 3 } else { 1 };
        let variable = decision.variable.0;
        let value = decision.value;
        let label = format!("(x{variable} = {value})\\ncost = {cost}");

        format!("\t{from} -> {to} [penwidth={width},label=\"{label}\"];\n")
    }
    /// Creates the list of attributes that are used to configure one node
    fn node_attributes(&self, id: usize, config: &VizConfig) -> String {
        let merged = self.groups.contains_key(&NodeId(id));
        let node = &self.nodes[id];
        let state = &self.states[id];
        let restricted= self.restricted[id];

        let shape = Self::node_shape(merged, restricted);
        let color = Self::node_color(node, merged);
        let peripheries = Self::node_peripheries(node);
        let group = self.node_group(node);
        let label = Self::node_label(node, state, config);

        format!("shape={shape},style=filled,color={color},peripheries={peripheries},group=\"{group}\",label=\"{label}\"")
    }
    /// Determines the group of a node based on the last branching decicion leading to it
    fn node_group(&self, node: &Node) -> String {
        if let Some(eid) = node.best {
            let edge = self.edges[eid.0];
            format!("{}", edge.decision.variable.0)
        } else {
            "root".to_string()
        }
    }
    /// Determines the shape to use when displaying a node
    fn node_shape(merged: bool, restricted: bool) -> &'static str {
        if merged || restricted {
            "diamond"
        } else {
            "circle"
        }
    }
    /// Determines the number of peripheries to draw when displaying a node.
    fn node_peripheries(node: &Node) -> usize {
        if node.flags.is_cutset() {
            2
        } else {
            1
        }
    }
    /// Determines the color of peripheries to draw when displaying a node.
    fn node_color(node: &Node, merged: bool) -> &str {
        if node.flags.is_exact() {
            "\"#99ccff\""
        } else if merged {
            "yellow"
        } else {
            "lightgray"
        }
    }
    /// Creates text label to place inside of the node when displaying it
    fn node_label(node: &Node, state: &T, config: &VizConfig) -> String {
        let mut out = format!("{:?}", state);

        if config.show_value {
            out.push_str(&format!("\\nval: {}", node.value));
        }
        if config.show_locb {
        out.push_str(&format!("\\nlocb: {}", Self::extreme(node.value_bot)));
        }
        if config.show_rub {
            out.push_str(&format!("\\nrub: {}", Self::extreme(node.rub)));
        }

        out
    }
    /// An utility method to replace extreme values with +inf and -inf
    fn extreme(x: isize) -> String {
        match x {
            isize::MAX => "+inf".to_string(),
            isize::MIN => "-inf".to_string(),
            _ => format!("{x}")
        }
    }
}