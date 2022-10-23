use std::collections::HashSet;

use crate::fsm::{Language, NFA};
use regex_runtime::Opcode;

mod directed_graph;
use directed_graph::{DirectedGraph, Graph};

#[derive(Debug, Hash, PartialEq, Eq)]
struct State {
    id: usize,
    kind: Terminal,
}

impl State {
    fn new(id: usize, kind: Terminal) -> Self {
        Self { id, kind }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
enum Terminal {
    Acceptor,
    NonAcceptor,
}

impl Terminal {
    fn is_acceptor(&self) -> bool {
        self == &Terminal::Acceptor
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Edge {
    Epsilon,
    MustMatchOneOf(HashSet<char>),
    MustNotMatchOneOf(HashSet<char>),
    Any,
}

impl Edge {
    fn matches(&self, other: Option<&char>) -> bool {
        match (self, other) {
            (Edge::Epsilon, None) => true,
            (Edge::MustMatchOneOf(items), Some(other)) if items.contains(other) => true,
            (Edge::MustNotMatchOneOf(items), Some(other)) if !items.contains(other) => true,
            (Edge::Any, Some(_)) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {}
