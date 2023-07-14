//! An implementation of a simple append-only graph, using linked-lists for edges.
extern crate alloc;

use alloc::{collections::VecDeque, vec, vec::Vec};

pub type NodeIdx = usize;

/// Represents a Node, or State, in a graph storing a link to it's first edge
/// and optional data.
#[derive(Default)]
pub struct Node<D> {
    data: D,
    /// The index for the first edge in a linked list of edges.
    first_outgoing_edge: Option<EdgeIdx>,
}

impl<D> Node<D> {
    // Instantiates a new node with the given data and no adjacent edges.
    pub fn new(data: D) -> Self {
        Self {
            data,
            first_outgoing_edge: None,
        }
    }

    // Associates an edge to the node.
    fn with_outgoing_edge_mut(&mut self, edge_idx: EdgeIdx) {
        self.first_outgoing_edge = Some(edge_idx);
    }
}

impl<D> AsRef<D> for Node<D> {
    fn as_ref(&self) -> &D {
        &self.data
    }
}

impl<D> AsMut<D> for Node<D> {
    fn as_mut(&mut self) -> &mut D {
        &mut self.data
    }
}

pub type EdgeIdx = usize;

/// Provides methods for instantiating and interacting with an edge.
pub trait IsEdge {
    fn new(target: NodeIdx) -> Self;
    fn new_with_adjacency(target: NodeIdx, adjacent: Option<EdgeIdx>) -> Self;
    fn next_adjacent_outgoing_edge(&self) -> Option<EdgeIdx>;
}

/// Provides methods for defining a directed edge.
pub trait IsDirectedEdge: IsEdge {
    fn target(&self) -> NodeIdx;
}

pub struct UnconstrainedDirectedEdge {
    target: NodeIdx,

    /// The index for the first edge in a linked list of edges.
    next_outgoing_edge: Option<EdgeIdx>,
}

impl IsEdge for UnconstrainedDirectedEdge {
    fn new(target: NodeIdx) -> Self {
        Self {
            target,
            next_outgoing_edge: None,
        }
    }

    fn new_with_adjacency(target: NodeIdx, adjacent: Option<EdgeIdx>) -> Self {
        Self {
            target,
            next_outgoing_edge: adjacent,
        }
    }

    fn next_adjacent_outgoing_edge(&self) -> Option<EdgeIdx> {
        self.next_outgoing_edge
    }
}

impl IsDirectedEdge for UnconstrainedDirectedEdge {
    fn target(&self) -> NodeIdx {
        self.target
    }
}

/// Graph defines a graph with a given set of nodes and edges.
pub struct Graph<'data, ND, E: IsEdge> {
    nodes: &'data mut Vec<Node<ND>>,
    edges: Vec<E>,
}

impl<'data, D, E: IsEdge> Graph<'data, D, E> {
    /// Instantiates a new graph with a predefined list of nodes and edges.
    pub fn new(nodes: &'data mut Vec<Node<D>>, edges: Vec<E>) -> Self {
        Self { nodes, edges }
    }

    /// Returns the number of nodes in a graph.
    pub fn node_cnt(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in a graph.
    pub fn edge_cnt(&self) -> usize {
        self.edges.len()
    }

    /// Borrows a node by its index if it exists in the graph.
    pub fn get_node(&self, idx: NodeIdx) -> Option<&Node<D>> {
        self.nodes.get(idx)
    }

    /// Mutably borrows a node by its index if it exists in the graph.
    pub fn get_node_mut(&mut self, idx: NodeIdx) -> Option<&mut Node<D>> {
        self.nodes.get_mut(idx)
    }

    /// Borrows an edge by its index if it exists in the graph.
    pub fn get_edge(&self, idx: EdgeIdx) -> Option<&E> {
        self.edges.get(idx)
    }
}

impl<'data, D, E: IsDirectedEdge> Graph<'data, D, E> {
    /// Immutably inserts a new node into a graph. Returning the modified graph
    /// and the index for the newly inserted node.
    pub fn insert_node(mut self, node: Node<D>) -> (Self, NodeIdx) {
        let next_idx = self.node_cnt();
        self.insert_node_mut(node);

        (self, next_idx)
    }

    /// Inserts a node into the graph, returning the index.
    pub fn insert_node_mut(&mut self, node: Node<D>) -> NodeIdx {
        let next_idx = self.node_cnt();
        self.nodes.push(node);

        next_idx
    }

    /// Immutably inserts a link between a source and target node in the graph.
    /// Returning the modified graph and optionally the new edge index if it
    /// could be created.
    pub fn insert_edge(mut self, source: NodeIdx, target: NodeIdx) -> (Self, Option<EdgeIdx>) {
        let new_head_edge_idx = self.insert_edge_mut(source, target);

        (self, new_head_edge_idx)
    }

    /// Inserts a new link between a source and target node, returning the new
    /// index optionally if it can be created.
    pub fn insert_edge_mut(&mut self, source: NodeIdx, target: NodeIdx) -> Option<EdgeIdx> {
        let new_head_edge_idx = self.edge_cnt();

        // short_circuit if target doesn't exist
        self.get_node(target)?;

        let source_node = self.nodes.get_mut(source)?;
        let prev_head_edge_idx = source_node.first_outgoing_edge;
        self.edges
            .push(<E>::new_with_adjacency(target, prev_head_edge_idx));

        source_node.with_outgoing_edge_mut(new_head_edge_idx);
        Some(new_head_edge_idx)
    }

    /// Returns all direct successor nodes from a given node.
    pub fn successors(&self, source: NodeIdx) -> Successors<D, E> {
        let first_outgoing_edge = self.nodes[source].first_outgoing_edge;
        Successors {
            graph: self,
            current_edge_idx: first_outgoing_edge,
        }
    }
}

impl<'data, D, E: IsTraversableEdge> Graph<'data, D, E> {
    /// A helper method for returning a `DepthFirstTraveral` from the root of a
    /// graph.
    pub fn depth_first_traversal(&self) -> DepthFirstTraversal<D, E> {
        DepthFirstTraversal::new(0, self)
    }

    /// A helper method for returning a `BreadthFirstTraveral` from the root of
    /// a graph.
    pub fn breadth_first_traversal(&self) -> BreadthFirstTraversal<D, E> {
        BreadthFirstTraversal::new(0, self)
    }
}

/// Represents an iterator over all direct successors for a given node.
pub struct Successors<'graph, 'data, D, E: IsDirectedEdge> {
    graph: &'graph Graph<'data, D, E>,
    current_edge_idx: Option<EdgeIdx>,
}

impl<'graph, 'data, D, E: IsDirectedEdge> Iterator for Successors<'graph, 'data, D, E> {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current_edge_idx {
            None => None,
            Some(edge_idx) => {
                let edge = &self.graph.edges[edge_idx];
                self.current_edge_idx = edge.next_adjacent_outgoing_edge();
                Some(edge.target())
            }
        }
    }
}

/// Defines a marker trait for any edge that can be traversed.
/// By default, this is implemented for any directed edge.
pub trait IsTraversableEdge: IsDirectedEdge {}
impl<E: IsDirectedEdge> IsTraversableEdge for E {}

/// Provides breadth-first traversal over a graph.
pub struct BreadthFirstTraversal<'graph, 'data, D, E: IsTraversableEdge> {
    visited: Vec<bool>,
    graph: &'graph Graph<'data, D, E>,
    queue: VecDeque<NodeIdx>,
}

impl<'graph, 'data, D, E: IsTraversableEdge> BreadthFirstTraversal<'graph, 'data, D, E> {
    pub fn new(root: NodeIdx, graph: &'graph Graph<'data, D, E>) -> Self {
        let node_cnt = graph.node_cnt();
        let mut queue = VecDeque::with_capacity(node_cnt);
        let visited = vec![false; node_cnt];

        queue.push_back(root);

        Self {
            visited,
            graph,
            queue,
        }
    }
}

impl<'graph, 'data, D, E: IsTraversableEdge> Iterator
    for BreadthFirstTraversal<'graph, 'data, D, E>
{
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.queue.pop_front()?;

        // early exit if visited already
        if self.visited[current] {
            None
        } else {
            self.visited[current] = true;
            let successors = self.graph.successors(current);
            for node in successors {
                if !self.visited[node] {
                    self.queue.push_back(node);
                }
            }

            Some(current)
        }
    }
}

/// Provides depth-first traversal over a graph.
pub struct DepthFirstTraversal<'graph, 'data, D, E: IsTraversableEdge> {
    visited: Vec<bool>,
    graph: &'graph Graph<'data, D, E>,
    stack: Vec<NodeIdx>,
}

impl<'graph, 'data, D, E: IsTraversableEdge> DepthFirstTraversal<'graph, 'data, D, E> {
    pub fn new(root: NodeIdx, graph: &'graph Graph<'data, D, E>) -> Self {
        let node_cnt = graph.node_cnt();
        let mut stack = Vec::with_capacity(node_cnt);
        let visited = vec![false; node_cnt];

        stack.push(root);

        Self {
            visited,
            graph,
            stack,
        }
    }
}

impl<'graph, 'data, D, E: IsTraversableEdge> Iterator for DepthFirstTraversal<'graph, 'data, D, E> {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.stack.pop()?;

        // early exit if visited already
        if self.visited[current] {
            None
        } else {
            self.visited[current] = true;
            let successors = self.graph.successors(current);
            for node in successors {
                if !self.visited[node] {
                    self.stack.push(node);
                }
            }

            Some(current)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_add_nodes() {
        let mut data = Vec::with_capacity(5);
        let mut graph = Graph::<(), UnconstrainedDirectedEdge>::new(&mut data, vec![]);

        for i in 0..5 {
            let node_idx = graph.insert_node_mut(Node::new(()));
            assert_eq!(i, node_idx);
        }

        assert_eq!(5, graph.node_cnt())
    }

    #[test]
    fn should_construct_immutable_graph() {
        let mut data = Vec::new();
        let graph = (0..4).fold(
            Graph::<(), UnconstrainedDirectedEdge>::new(&mut data, vec![]),
            |graph, _| graph.insert_node(Node::new(())).0,
        );

        let graph = [(0, 1), (1, 2), (0, 3), (3, 2)]
            .iter()
            .copied()
            .fold(graph, |graph, (source, target)| {
                graph.insert_edge(source, target).0
            });
        let successor_nodes: Vec<_> = graph.successors(0).collect();
        assert_eq!(&[3, 1], &successor_nodes[..]);
    }

    #[test]
    fn should_fail_to_add_edge_to_non_existent_nodes() {
        let mut data = Vec::new();
        let mut graph = Graph::<(), UnconstrainedDirectedEdge>::new(&mut data, vec![]);

        let n0 = graph.insert_node_mut(Node::new(()));
        let n1 = graph.insert_node_mut(Node::new(()));
        let n2 = 2;
        let n3 = 3;

        assert!(graph.insert_edge_mut(n0, n1).is_some());
        assert!(graph.insert_edge_mut(n0, n2).is_none());
        assert!(graph.insert_edge_mut(n0, n3).is_none());
    }

    #[test]
    fn should_traverse_in_breadth_first_order() {
        let mut data = Vec::new();
        let mut graph = Graph::<(), UnconstrainedDirectedEdge>::new(&mut data, vec![]);

        let n0 = graph.insert_node_mut(Node::new(()));
        let n1 = graph.insert_node_mut(Node::new(()));
        let n2 = graph.insert_node_mut(Node::new(()));
        let n3 = graph.insert_node_mut(Node::new(()));

        graph.insert_edge_mut(n0, n1); // n0 -> n1
        graph.insert_edge_mut(n1, n2); // n1 -> n2
        graph.insert_edge_mut(n0, n3); // n0 -> n3

        // add loops
        graph.insert_edge_mut(n3, n2); // n3 -> n2
        graph.insert_edge_mut(n2, n0); // n3 -> n2

        let bft = BreadthFirstTraversal::new(n0, &graph);
        let iterated_nodes: Vec<_> = bft.collect();

        assert_eq!(&[n0, n3, n1, n2], &iterated_nodes[..]);

        let bfs = graph.breadth_first_traversal();
        let iterated_nodes: Vec<_> = bfs.collect();

        assert_eq!(&[n0, n3, n1, n2], &iterated_nodes[..])
    }

    #[test]
    fn should_traverse_in_depth_first_order() {
        let mut data = Vec::new();
        let mut graph = Graph::<(), UnconstrainedDirectedEdge>::new(&mut data, vec![]);

        let n0 = graph.insert_node_mut(Node::new(()));
        let n1 = graph.insert_node_mut(Node::new(()));
        let n2 = graph.insert_node_mut(Node::new(()));
        let n3 = graph.insert_node_mut(Node::new(()));

        graph.insert_edge_mut(n0, n1); // n0 -> n1
        graph.insert_edge_mut(n1, n2); // n1 -> n2
        graph.insert_edge_mut(n0, n3); // n0 -> n3

        // add loops
        graph.insert_edge_mut(n3, n2); // n3 -> n2
        graph.insert_edge_mut(n2, n0); // n3 -> n2

        let dfs = DepthFirstTraversal::new(n0, &graph);
        let iterated_nodes: Vec<_> = dfs.collect();

        assert_eq!(&[n0, n1, n2, n3], &iterated_nodes[..]);

        let dfs = graph.depth_first_traversal();
        let iterated_nodes: Vec<_> = dfs.collect();

        assert_eq!(&[n0, n1, n2, n3], &iterated_nodes[..])
    }
}
