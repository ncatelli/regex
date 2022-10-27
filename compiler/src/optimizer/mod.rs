use std::collections::{HashMap, HashSet};

use regex_runtime::{InstJmp, Instruction, Instructions, Opcode};

mod directed_graph;
use directed_graph::{DirectedGraph, Graph};

#[derive(Debug, Hash, PartialEq, Eq)]
struct State {
    id: usize,
    kind: AcceptState,
    expr_id: usize,
    save_group_id: Option<usize>,
}

impl State {
    fn new(id: usize, kind: AcceptState, expr_id: usize) -> Self {
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
enum AcceptState {
    Acceptor,
    NonAcceptor,
}

impl AcceptState {
    fn is_acceptor(&self) -> bool {
        self == &AcceptState::Acceptor
    }
}

#[derive(Debug, PartialEq, Eq)]
enum EpsilonCondition {}

#[derive(Debug, PartialEq, Eq)]
enum EpsilonAction {
    StartSave(usize),
    EndSave(usize),
    SetExpressionId(u32),
}

#[derive(Debug, PartialEq, Eq)]
enum Edge {
    Epsilon,
    EpsilonWithCondition(EpsilonCondition),
    EpsilonWithAction(EpsilonAction),
    MustMatchOneOf(HashSet<char>),
    MustNotMatchOneOf(HashSet<char>),
    MatchAny,
}

impl Edge {
    fn matches(&self, other: Option<&char>) -> bool {
        match (self, other) {
            (Edge::Epsilon, None) => true,
            (Edge::MustMatchOneOf(items), Some(other)) if items.contains(other) => true,
            (Edge::MustNotMatchOneOf(items), Some(other)) if !items.contains(other) => true,
            (Edge::MatchAny, Some(_)) => true,
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

struct Thread<'a> {
    stack: Vec<usize>,
    instructions: &'a [Instruction],
}

impl<'a> Thread<'a> {
    fn new(stack: Vec<usize>, instructions: &'a [Instruction]) -> Self {
        Self {
            stack,
            instructions,
        }
    }
}

fn graph_from_bytecode(program: &Instructions) -> Result<AttributeGraph, String> {
    let mut graph = {
        let mut graph = AttributeGraph::new();

        for inst in &program.program {
            let state_id = inst.offset;
            let next_state = State::new(state_id, AcceptState::NonAcceptor, 0);

            graph.add_node(next_state);
        }

        graph
    };

    for inst in &program.program {
        let src_state_id = inst.offset;
        let default_dest_state_id = src_state_id + 1;

        match inst.opcode {
            Opcode::Any => {
                graph
                    .graph
                    .add_edge(src_state_id, default_dest_state_id, Edge::MatchAny)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Consume(regex_runtime::InstConsume { value }) => {
                let set = [value].into_iter().collect();

                graph
                    .graph
                    .add_edge(
                        src_state_id,
                        default_dest_state_id,
                        Edge::MustMatchOneOf(set),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Opcode::ConsumeSet(_) => todo!(),
            Opcode::Epsilon(_) => todo!(),
            Opcode::Split(regex_runtime::InstSplit { x_branch, y_branch }) => {
                let x_branch_state_id = x_branch.as_usize();
                let y_branch_state_id = y_branch.as_usize();

                graph
                    .graph
                    .add_edge(src_state_id, x_branch_state_id, Edge::Epsilon)
                    .map_err(|e| e.to_string())?;
                graph
                    .graph
                    .add_edge(src_state_id, y_branch_state_id, Edge::Epsilon)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Jmp(regex_runtime::InstJmp { next }) => {
                let next_state_id = next.as_usize();

                graph
                    .graph
                    .add_edge(src_state_id, next_state_id, Edge::Epsilon)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::StartSave(regex_runtime::InstStartSave { slot_id }) => {
                let edge = Edge::EpsilonWithAction(EpsilonAction::StartSave(slot_id));

                graph
                    .graph
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::EndSave(regex_runtime::InstEndSave { slot_id }) => {
                let edge = Edge::EpsilonWithAction(EpsilonAction::EndSave(slot_id));

                graph
                    .graph
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Meta(regex_runtime::InstMeta(regex_runtime::MetaKind::SetExpressionId(id))) => {
                let edge = Edge::EpsilonWithAction(EpsilonAction::SetExpressionId(id));

                graph
                    .graph
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Match => {
                let terminal_state_id = src_state_id;

                graph
                    .states_attrs
                    .entry(terminal_state_id)
                    .and_modify(|state| state.kind = AcceptState::Acceptor);
            }
        }
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex_runtime::*;

    #[test]
    fn should_generate_terminal_state_for_consuming_instructions() {
        let opcodes = vec![Opcode::Any, Opcode::Any, Opcode::Match];
        let program = Instructions::new(vec![], opcodes);
        let res = graph_from_bytecode(&program);

        assert!(res.is_ok());

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut states = graph.states_attrs.values().collect::<Vec<_>>();
        states.sort_by(|a, b| a.id.cmp(&b.id));

        assert_eq!(
            vec![
                &State::new(0, AcceptState::NonAcceptor, 0),
                &State::new(1, AcceptState::NonAcceptor, 0),
                &State::new(2, AcceptState::Acceptor, 0),
            ],
            states
        )
    }

    #[test]
    fn should_generate_linked_edges_for_consuming_instructions() {
        use directed_graph::DirectedEdge;

        let opcodes = vec![
            Opcode::Any,
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Match,
        ];
        let program = Instructions::new(vec![], opcodes);
        let res = graph_from_bytecode(&program);

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut edges = graph.graph.edges();
        edges.sort_by(|a, b| a.src.cmp(b.src));

        assert_eq!(
            vec![
                DirectedEdge::new(&0, &1, &Edge::MatchAny),
                DirectedEdge::new(&1, &2, &Edge::MustMatchOneOf(['a'].into_iter().collect())),
            ],
            edges
        )
    }

    #[test]
    fn should_generate_diverging_program() {
        use directed_graph::DirectedEdge;

        let opcodes = vec![
            Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(4))),
            Opcode::Any,
            Opcode::Any,
            Opcode::Match,
            Opcode::Any,
            Opcode::Any,
            Opcode::Match,
        ];
        let program = Instructions::new(vec![], opcodes);
        let res = graph_from_bytecode(&program);

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut edges = graph.graph.edges();
        edges.sort_by(|a, b| a.src.cmp(b.src));

        assert_eq!(
            vec![
                DirectedEdge::new(&0, &1, &Edge::Epsilon),
                DirectedEdge::new(&0, &4, &Edge::Epsilon),
                DirectedEdge::new(&1, &2, &Edge::MatchAny),
                DirectedEdge::new(&2, &3, &Edge::MatchAny),
                DirectedEdge::new(&4, &5, &Edge::MatchAny),
                DirectedEdge::new(&5, &6, &Edge::MatchAny),
            ],
            edges
        );

        // entry node plus one for each state.
        assert_eq!(7, graph.graph.nodes().len());
    }
}
