use crate::fsm::{Language, NFA};
use regex_runtime::Opcode;

mod directed_graph;
use directed_graph::{DirectedGraph, Graph};

#[derive(Debug, Hash, PartialEq, Eq)]
enum State {
    Acceptor,
    NonAcceptor,
}

#[derive(Debug, PartialEq, Eq)]
enum Edge {}

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
            .filter(|state| matches!(state, State::Acceptor))
            .copied()
            .collect()
    }

    fn transition(
        &self,
        _: &'a State,
        _: Option<&<char as Language>::T>,
    ) -> crate::fsm::TransitionResult<'a, State> {
        todo!()
    }
}

#[allow(unused)]
fn graph_from_opcode(opcodes: &[Opcode]) -> Result<(), String> {
    todo!()
}
