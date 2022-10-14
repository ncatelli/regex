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

pub struct DirectedEdge<'a, NODE: Clone + Hash + Eq> {
    src: &'a NODE,
    dest: &'a NODE,
    edge_value: i32,
}

impl<'a, NODE: Clone + Hash + Eq> DirectedEdge<'a, NODE> {
    pub fn new(src: &'a NODE, dest: &'a NODE, edge_value: i32) -> Self {
        Self {
            src,
            dest,
            edge_value,
        }
    }
}

impl<'a, NODE: Clone + Hash + Eq> From<DirectedEdge<'a, NODE>> for (&'a NODE, &'a NODE, i32) {
    fn from(ded: DirectedEdge<'a, NODE>) -> Self {
        (ded.src, ded.dest, ded.edge_value)
    }
}

pub struct DirectedEdgeDestination<NODE: Clone + Hash + Eq> {
    dest: NODE,
    edge_value: i32,
}

impl<NODE: Clone + Hash + Eq> DirectedEdgeDestination<NODE> {
    pub fn new(dest: NODE, edge_value: i32) -> Self {
        Self { dest, edge_value }
    }
}

impl<NODE: Clone + Hash + Eq> From<DirectedEdgeDestination<NODE>> for (NODE, i32) {
    fn from(ded: DirectedEdgeDestination<NODE>) -> Self {
        (ded.dest, ded.edge_value)
    }
}

pub trait Graph<NODE: Clone + Hash + Eq> {
    fn new() -> Self;
    fn adjacency_table_mut(&mut self) -> &mut HashMap<NODE, Vec<DirectedEdgeDestination<NODE>>>;
    fn adjacency_table(&self) -> &HashMap<NODE, Vec<DirectedEdgeDestination<NODE>>>;
    fn add_node(&mut self, node: &NODE) -> bool;
    fn add_edge(&mut self, edge: DirectedEdge<NODE>);
    fn neighbours(&self, node: &NODE) -> Result<&Vec<DirectedEdgeDestination<NODE>>, GraphError>;
    fn contains(&self, node: &NODE) -> bool;
    fn nodes(&self) -> Vec<&NODE>;
    fn edges(&self) -> Vec<DirectedEdge<NODE>>;
}

pub struct DirectedGraph<NODE: Clone + Hash + Eq> {
    adjacency_table: HashMap<NODE, Vec<DirectedEdgeDestination<NODE>>>,
}

impl<NODE: Clone + Hash + Eq> Graph<NODE> for DirectedGraph<NODE> {
    fn new() -> Self {
        Self {
            adjacency_table: HashMap::new(),
        }
    }

    fn adjacency_table_mut(&mut self) -> &mut HashMap<NODE, Vec<DirectedEdgeDestination<NODE>>> {
        &mut self.adjacency_table
    }

    fn adjacency_table(&self) -> &HashMap<NODE, Vec<DirectedEdgeDestination<NODE>>> {
        &self.adjacency_table
    }

    fn add_node(&mut self, node: &NODE) -> bool {
        match self.adjacency_table().get(node) {
            None => {
                self.adjacency_table_mut()
                    .insert((node).clone(), Vec::new());
                true
            }
            _ => false,
        }
    }

    fn add_edge(&mut self, edge: DirectedEdge<NODE>) {
        let (src, dest, edge_value) = (edge.src, edge.dest, edge.edge_value);

        self.add_node(src);
        self.add_node(dest);

        self.adjacency_table_mut()
            .entry(src.clone())
            .and_modify(|e| {
                let ded = DirectedEdgeDestination::new(dest.clone(), edge_value);
                e.push(ded);
            });
    }

    fn neighbours(&self, node: &NODE) -> Result<&Vec<DirectedEdgeDestination<NODE>>, GraphError> {
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

    fn edges(&self) -> Vec<DirectedEdge<NODE>> {
        let mut edges = Vec::new();
        for (from_node, from_node_neighbours) in self.adjacency_table() {
            let destination_tuple_iter = from_node_neighbours
                .iter()
                .map(|edge_dest| (&edge_dest.dest, &edge_dest.edge_value));

            for (to_node, weight) in destination_tuple_iter {
                edges.push(DirectedEdge::new(from_node, to_node, *weight));
            }
        }
        edges
    }
}
