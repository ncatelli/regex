use std::collections::{HashMap, HashSet};

use regex_runtime::{Instructions, Opcode};

mod directed_graph;
use directed_graph::{DirectedGraph, Graph};

#[derive(Debug, Hash, PartialEq, Eq)]
struct State {
    id: usize,
    kind: Terminal,
    expr_id: usize,
    save_group_id: Option<usize>,
}

impl State {
    fn new(id: usize, kind: Terminal, expr_id: usize) -> Self {
        Self {
            id,
            kind,
            expr_id,
            save_group_id: None,
        }
    }

    fn with_save_group_id(mut self, save_group_id: usize) -> Self {
        self.save_group_id = Some(save_group_id);
        self
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

struct AttributeGraph {
    graph: DirectedGraph<usize, Edge>,
    states_attrs: HashMap<usize, State>,
}

impl AttributeGraph {
    fn new() -> Self {
        Self {
            graph: Default::default(),
            states_attrs: Default::default(),
        }
    }

    fn add_node(&mut self, node: State) -> bool {
        let state_id = node.id;
        self.states_attrs.insert(state_id, node);
        self.graph.add_node(state_id)
    }
}

fn graph_from_bytecode_program(program: &Instructions) -> Result<AttributeGraph, String> {
    let mut graph = {
        let mut graph = AttributeGraph::new();
        graph.add_node(State::new(usize::MAX, Terminal::NonAcceptor, 0));
        graph
    };
    let mut stack = vec![0_usize];

    for opcode in &program.program {
        // instruction offset, used as a state id.
        let state_id = opcode.offset;
        let parent_state_id = stack
            .last()
            // This should never possibly occur.
            .ok_or_else(|| "state stack is empty".to_string())?;
        let (parent_expr_id, _parent_save_group_id) = graph
            .states_attrs
            .get(&state_id)
            .map(|node| (node.expr_id, node.save_group_id))
            .ok_or_else(|| "unknown state".to_string())?;

        match opcode.opcode {
            Opcode::Any => {
                let new_state = State::new(state_id, Terminal::NonAcceptor, parent_expr_id);
                graph.add_node(new_state);
                stack.push(state_id);
            }
            Opcode::Consume(_) => {
                todo!()
            }
            Opcode::ConsumeSet(_) => {
                todo!()
            }
            Opcode::Epsilon(_) => todo!(),
            Opcode::Split(_) => todo!(),
            Opcode::Jmp(_) => todo!(),
            Opcode::StartSave(_) => todo!(),
            Opcode::EndSave(_) => todo!(),
            Opcode::Match => {
                graph
                    .states_attrs
                    .entry(*parent_state_id)
                    .and_modify(|node| node.kind = Terminal::Acceptor);
            }
            Opcode::Meta(regex_runtime::InstMeta(regex_runtime::MetaKind::SetExpressionId(_))) => {
                todo!()
            }
        }
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {}
