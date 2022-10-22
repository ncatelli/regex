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
    MustMatch(char),
    MustMatchAny,
}

struct NFAFromGraph<'a> {
    initial_state: &'a State,
    graph: DirectedGraph<&'a State, &'a Edge>,
}

impl<'a> NFA<'a, State, Edge, char> for NFAFromGraph<'a> {
    fn states(&self) -> std::collections::HashSet<&'a State> {
        self.graph.nodes().into_iter().copied().collect()
    }

    fn initial_state(&self) -> &'a State {
        self.initial_state
    }

    fn final_states(&self) -> std::collections::HashSet<&'a State> {
        self.graph
            .nodes()
            .into_iter()
            .filter(|state| state.kind.is_acceptor())
            .copied()
            .collect()
    }

    fn transition(
        &self,
        current_state: &'a State,
        _next: Option<&char>,
    ) -> crate::fsm::TransitionResult<'a, State> {
        // safe to unwrap, caller guarantees all edges have states if generated
        // from compiled opcode.
        let _edges = self.graph.neighbors(&current_state).unwrap();

        todo!()
    }
}

#[allow(unused)]
fn graph_from_opcode(opcodes: &[Opcode]) -> Result<(), String> {
    todo!()
}
