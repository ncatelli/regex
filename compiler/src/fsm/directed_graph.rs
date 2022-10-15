use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, PartialEq, Eq)]
pub enum GraphErrorKind {
    NodeUndefined,
}

#[derive(Debug, PartialEq, Eq)]
pub struct GraphError {
    /// The type of triggered error.
    kind: GraphErrorKind,
    /// Additional error data.
    data: Option<String>,
}

impl GraphError {
    /// Instantiates a new error.
    pub fn new(kind: GraphErrorKind) -> Self {
        Self { kind, data: None }
    }

    /// Associates additional data with the error, returning the modified error.
    #[allow(unused)]
    pub fn with_data(mut self, data: String) -> Self {
        self.with_data_mut(data);
        self
    }

    /// Associates additional data with the error.
    #[allow(unused)]
    pub fn with_data_mut(&mut self, data: String) {
        self.data = Some(data);
    }
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (data, padding) = if let Some(data) = &self.data {
            (data.as_str(), " ")
        } else {
            ("", "")
        };
        match &self.kind {
            GraphErrorKind::NodeUndefined => {
                write!(f, "invalid operand{}{}", padding, data)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectedEdge<'a, NODE, EDGE>
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    src: &'a NODE,
    dest: &'a NODE,
    edge_value: &'a EDGE,
}

impl<'a, NODE, EDGE> DirectedEdge<'a, NODE, EDGE>
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    pub fn new(src: &'a NODE, dest: &'a NODE, edge_value: &'a EDGE) -> Self {
        Self {
            src,
            dest,
            edge_value,
        }
    }
}

impl<'a, NODE, EDGE> From<DirectedEdge<'a, NODE, EDGE>> for (&'a NODE, &'a NODE, &'a EDGE)
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    fn from(ded: DirectedEdge<'a, NODE, EDGE>) -> Self {
        (ded.src, ded.dest, ded.edge_value)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DirectedEdgeDestination<NODE, EDGEVAL>
where
    NODE: Hash + Eq,
    EDGEVAL: Eq,
{
    dest: NODE,
    edge_value: EDGEVAL,
}

impl<NODE: Hash + Eq, EDGEVAL: Eq> DirectedEdgeDestination<NODE, EDGEVAL> {
    pub fn new(dest: NODE, edge_value: EDGEVAL) -> Self {
        Self { dest, edge_value }
    }
}

impl<NODE: Hash + Eq, EDGEVAL: Clone + Eq> From<DirectedEdgeDestination<NODE, EDGEVAL>>
    for (NODE, EDGEVAL)
{
    fn from(ded: DirectedEdgeDestination<NODE, EDGEVAL>) -> Self {
        (ded.dest, ded.edge_value)
    }
}

pub trait Graph<NODE, EDGE>
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    fn adjacent_mut(&mut self) -> &mut HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGE>>>;
    fn adjacent(&self) -> &HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGE>>>;
    fn add_node(&mut self, node: NODE) -> bool;
    fn add_edge(&mut self, src: NODE, dest: NODE, edge_value: EDGE) -> Result<(), GraphError>;
    fn neighbors(
        &self,
        node: &NODE,
    ) -> Result<&Vec<DirectedEdgeDestination<NODE, EDGE>>, GraphError>;
    fn contains(&self, node: &NODE) -> bool;
    fn nodes(&self) -> Vec<&NODE>;
    fn edges(&self) -> Vec<DirectedEdge<NODE, EDGE>>;
}

pub struct DirectedGraph<NODE, EDGE>
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    adjacency_table: HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGE>>>,
}

impl<NODE, EDGE> DirectedGraph<NODE, EDGE>
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    pub fn new() -> Self {
        Self {
            adjacency_table: HashMap::new(),
        }
    }
}

impl<NODE, EDGE> Default for DirectedGraph<NODE, EDGE>
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<NODE, EDGE> Graph<NODE, EDGE> for DirectedGraph<NODE, EDGE>
where
    NODE: Hash + Eq,
    EDGE: Eq,
{
    fn adjacent_mut(&mut self) -> &mut HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGE>>> {
        &mut self.adjacency_table
    }

    fn adjacent(&self) -> &HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGE>>> {
        &self.adjacency_table
    }

    fn add_node(&mut self, node: NODE) -> bool {
        match self.adjacent().get(&node) {
            None => {
                self.adjacent_mut().insert(node, Vec::new());
                true
            }
            _ => false,
        }
    }

    fn add_edge(&mut self, src: NODE, dest: NODE, edge_value: EDGE) -> Result<(), GraphError> {
        match (self.contains(&src), self.contains(&dest)) {
            (true, false) => Err(GraphError::new(GraphErrorKind::NodeUndefined)
                .with_data("destination node undefined".to_string())),
            (false, true) => Err(GraphError::new(GraphErrorKind::NodeUndefined)
                .with_data("source node undefined".to_string())),
            (false, false) => Err(GraphError::new(GraphErrorKind::NodeUndefined)
                .with_data("source and destination node undefined".to_string())),

            // add the edge if both nodes exist.
            (true, true) => {
                self.adjacent_mut().entry(src).and_modify(|e| {
                    let ded = DirectedEdgeDestination::new(dest, edge_value);
                    e.push(ded);
                });

                Ok(())
            }
        }
    }

    fn neighbors(
        &self,
        node: &NODE,
    ) -> Result<&Vec<DirectedEdgeDestination<NODE, EDGE>>, GraphError> {
        self.adjacent()
            .get(node)
            .ok_or_else(|| GraphError::new(GraphErrorKind::NodeUndefined))
    }

    fn contains(&self, node: &NODE) -> bool {
        self.adjacent().get(node).is_some()
    }

    fn nodes(&self) -> Vec<&NODE> {
        self.adjacent().keys().collect()
    }

    fn edges(&self) -> Vec<DirectedEdge<NODE, EDGE>> {
        self.adjacent()
            .iter()
            .flat_map(|(from_node, from_node_neighbors)| {
                from_node_neighbors
                    .iter()
                    .map(|edge_dest| (&edge_dest.dest, &edge_dest.edge_value))
                    .map(|(to_node, weight)| DirectedEdge::new(from_node, to_node, weight))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_add_directed_nodes_without_edges() {
        let mut graph: DirectedGraph<_, ()> = DirectedGraph::new();
        let nodes = ["a", "b", "c"];

        for node in nodes {
            graph.add_node(node);
        }

        let mut received_nodes = graph.nodes();
        received_nodes.sort();

        let expected: Vec<_> = nodes.iter().collect();

        assert_eq!(&received_nodes, &expected);
    }

    #[test]
    fn should_add_directed_nodes_with_edges() {
        let mut graph = DirectedGraph::new();
        let nodes = ["a", "b", "c"];
        for node in nodes {
            graph.add_node(node);
        }

        let edges = [("a", "b", 1), ("b", "c", 2), ("c", "c", 1), ("c", "c", 2)];
        for (src, dest, edge_val) in edges {
            graph.add_edge(src, dest, edge_val).unwrap();
        }

        let mut received_edges = graph.edges();
        received_edges.sort_by(|a, b| a.src.partial_cmp(b.src).unwrap());

        let expected: Vec<_> = edges
            .iter()
            .map(|(src, dest, edge_val)| DirectedEdge::new(src, dest, edge_val))
            .collect();

        assert_eq!(&received_edges, &expected);
    }
}