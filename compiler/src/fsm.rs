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
    pub fn with_data(mut self, data: String) -> Self {
        self.with_data_mut(data);
        self
    }

    /// Associates additional data with the error.
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

pub struct DirectedEdge<'a, NODE, EDGE>
where
    NODE: Clone + Hash + Eq,
    EDGE: Clone + Eq,
{
    src: &'a NODE,
    dest: &'a NODE,
    edge_value: &'a EDGE,
}

impl<'a, NODE, EDGE> DirectedEdge<'a, NODE, EDGE>
where
    NODE: Clone + Hash + Eq,
    EDGE: Clone + Eq,
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
    NODE: Clone + Hash + Eq,
    EDGE: Clone + Eq,
{
    fn from(ded: DirectedEdge<'a, NODE, EDGE>) -> Self {
        (ded.src, ded.dest, ded.edge_value)
    }
}

pub struct DirectedEdgeDestination<NODE, EDGEVAL>
where
    NODE: Clone + Hash + Eq,
    EDGEVAL: Clone + Eq,
{
    dest: NODE,
    edge_value: EDGEVAL,
}

impl<NODE: Clone + Hash + Eq, EDGEVAL: Clone + Eq> DirectedEdgeDestination<NODE, EDGEVAL> {
    pub fn new(dest: NODE, edge_value: EDGEVAL) -> Self {
        Self { dest, edge_value }
    }
}

impl<NODE: Clone + Hash + Eq, EDGEVAL: Clone + Eq> From<DirectedEdgeDestination<NODE, EDGEVAL>>
    for (NODE, EDGEVAL)
{
    fn from(ded: DirectedEdgeDestination<NODE, EDGEVAL>) -> Self {
        (ded.dest, ded.edge_value)
    }
}

pub trait Graph<NODE, Edge>
where
    NODE: Clone + Hash + Eq,
    Edge: Clone + Eq,
{
    fn adjacency_table_mut(
        &mut self,
    ) -> &mut HashMap<NODE, Vec<DirectedEdgeDestination<NODE, Edge>>>;
    fn adjacency_table(&self) -> &HashMap<NODE, Vec<DirectedEdgeDestination<NODE, Edge>>>;
    fn add_node(&mut self, node: &NODE) -> bool;
    fn add_edge(&mut self, edge: DirectedEdge<NODE, Edge>);
    fn neighbours(
        &self,
        node: &NODE,
    ) -> Result<&Vec<DirectedEdgeDestination<NODE, Edge>>, GraphError>;
    fn contains(&self, node: &NODE) -> bool;
    fn nodes(&self) -> Vec<&NODE>;
    fn edges(&self) -> Vec<DirectedEdge<NODE, Edge>>;
}

pub struct DirectedGraph<NODE, EDGEVAL>
where
    NODE: Clone + Hash + Eq,
    EDGEVAL: Clone + Eq,
{
    adjacency_table: HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGEVAL>>>,
}

impl<NODE, EDGEVAL> DirectedGraph<NODE, EDGEVAL>
where
    NODE: Clone + Hash + Eq,
    EDGEVAL: Clone + Eq,
{
    pub fn new() -> Self {
        Self {
            adjacency_table: HashMap::new(),
        }
    }
}

impl<NODE, EDGEVAL> Default for DirectedGraph<NODE, EDGEVAL>
where
    NODE: Clone + Hash + Eq,
    EDGEVAL: Clone + Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<NODE, EDGEVAL> Graph<NODE, EDGEVAL> for DirectedGraph<NODE, EDGEVAL>
where
    NODE: Clone + Hash + Eq,
    EDGEVAL: Clone + Eq,
{
    fn adjacency_table_mut(
        &mut self,
    ) -> &mut HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGEVAL>>> {
        &mut self.adjacency_table
    }

    fn adjacency_table(&self) -> &HashMap<NODE, Vec<DirectedEdgeDestination<NODE, EDGEVAL>>> {
        &self.adjacency_table
    }

    fn add_node(&mut self, node: &NODE) -> bool {
        match self.adjacency_table().get(node) {
            None => {
                self.adjacency_table_mut().insert(node.clone(), Vec::new());
                true
            }
            _ => false,
        }
    }

    fn add_edge(&mut self, edge: DirectedEdge<NODE, EDGEVAL>) {
        let (src, dest, edge_value) = (edge.src, edge.dest, edge.edge_value);

        self.add_node(src);
        self.add_node(dest);

        self.adjacency_table_mut()
            .entry(src.clone())
            .and_modify(|e| {
                let ded = DirectedEdgeDestination::new(dest.clone(), edge_value.clone());
                e.push(ded);
            });
    }

    fn neighbours(
        &self,
        node: &NODE,
    ) -> Result<&Vec<DirectedEdgeDestination<NODE, EDGEVAL>>, GraphError> {
        match self.adjacency_table().get(node) {
            None => Err(GraphError::new(GraphErrorKind::NodeUndefined)),
            Some(i) => Ok(i),
        }
    }

    fn contains(&self, node: &NODE) -> bool {
        self.adjacency_table().get(node).is_some()
    }

    fn nodes(&self) -> Vec<&NODE> {
        self.adjacency_table().keys().collect()
    }

    fn edges(&self) -> Vec<DirectedEdge<NODE, EDGEVAL>> {
        let mut edges = Vec::new();
        for (from_node, from_node_neighbours) in self.adjacency_table() {
            let destination_tuple_iter = from_node_neighbours
                .iter()
                .map(|edge_dest| (&edge_dest.dest, &edge_dest.edge_value));

            for (to_node, weight) in destination_tuple_iter {
                edges.push(DirectedEdge::new(from_node, to_node, weight));
            }
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_add_directed_nodes_without_edges() {
        let mut graph: DirectedGraph<String, ()> = DirectedGraph::new();
        let nodes = ["a".to_string(), "b".to_string(), "c".to_string()];

        for node in nodes.iter() {
            graph.add_node(node);
        }

        let mut received_nodes = graph.nodes();
        received_nodes.sort();

        let expected: Vec<_> = nodes.iter().collect();

        assert_eq!(&received_nodes, &expected);
    }
}
