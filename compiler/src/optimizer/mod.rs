use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::Range;

use regex_runtime::*;

mod directed_graph;
mod nfa;
use directed_graph::{DirectedGraph, Graph};
use nfa::{Alphabet, Nfa, TransitionResult};

use self::directed_graph::DirectedEdgeDestination;

#[derive(Debug, Hash, PartialEq, Eq)]
struct State {
    id: usize,
    kind: AcceptState,
}

impl State {
    fn new(id: usize, kind: AcceptState) -> Self {
        Self { id, kind }
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
enum EpsilonAction {
    None,
    StartSave(usize),
    EndSave(usize),
    SetExpressionId(u32),
}

type BoxedTransitionFunc = Box<dyn Fn(Option<char>) -> bool>;

struct Edge {
    action: Option<EpsilonAction>,
    transition_func: BoxedTransitionFunc,
}

impl Edge {
    fn new(func: impl Fn(Option<char>) -> bool + 'static) -> Self {
        Self {
            action: None,
            transition_func: Box::new(func),
        }
    }

    fn new_with_action(
        action: EpsilonAction,
        func: impl Fn(Option<char>) -> bool + 'static,
    ) -> Self {
        Self {
            action: Some(action),
            transition_func: Box::new(func),
        }
    }

    fn is_epsilon(&self) -> bool {
        self.action.is_some()
    }
}

struct AttributeGraph {
    edges: DirectedGraph<usize, Edge>,
    states: HashMap<usize, State>,
}

impl AttributeGraph {
    fn new() -> Self {
        Self {
            edges: Default::default(),
            states: Default::default(),
        }
    }

    fn add_node(&mut self, node: State) -> bool {
        let state_id = node.id;
        self.states.insert(state_id, node);
        self.edges.add_node(state_id)
    }
}

struct NfaFromAttrGraph<'a> {
    graph: &'a AttributeGraph,
}

struct Block {
    span: Range<usize>,
    initial: bool,
    states: Vec<State>,
    edges: Vec<Edge>,
}

fn block_spans_from_instructions(program: &Instructions) -> Vec<Range<usize>> {
    let block_starts = {
        let mut block_starts = [0_usize]
            .into_iter()
            .chain(
                program
                    .program
                    .iter()
                    .flat_map(|program| match program.opcode {
                        Opcode::Split(InstSplit { x_branch, y_branch }) => {
                            vec![x_branch.as_usize(), y_branch.as_usize()]
                        }
                        Opcode::Jmp(InstJmp { next }) => vec![next.as_usize()],
                        _ => vec![],
                    }),
            )
            .chain([program.program.len()].into_iter())
            .collect::<Vec<_>>();
        block_starts.sort();
        block_starts.dedup();
        block_starts
    };

    block_starts
        .as_slice()
        .windows(2)
        .map(|window| window[0]..window[1])
        .collect()
}

fn blocks_from_program(_program: &Instructions) -> Result<AttributeGraph, String> {
    todo!()
}

fn graph_from_runtime_instruction_set(program: &Instructions) -> Result<AttributeGraph, String> {
    let mut graph = {
        let mut graph = AttributeGraph::new();

        for inst in &program.program {
            let state_id = inst.offset;
            let next_state = State::new(state_id, AcceptState::NonAcceptor);

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
                    .edges
                    .add_edge(
                        src_state_id,
                        default_dest_state_id,
                        Edge::new(|next| next.is_some()),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Consume(InstConsume { value }) => {
                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        default_dest_state_id,
                        Edge::new(move |next| next == Some(value)),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Opcode::ConsumeSet(InstConsumeSet { idx }) => {
                let set = program
                    .sets
                    .get(idx)
                    .ok_or_else(|| format!("unknown set index: {}", idx))?;

                let char_set: HashSet<char> = match &set.set {
                    CharacterAlphabet::Range(range) => range.clone().collect(),
                    CharacterAlphabet::Explicit(c) => c.iter().copied().collect(),
                    CharacterAlphabet::Ranges(ranges) => {
                        ranges.iter().flat_map(|r| r.clone()).collect()
                    }
                    CharacterAlphabet::UnicodeCategory(_) => todo!(),
                };

                let edge = match set.membership {
                    SetMembership::Inclusive => Edge::new(move |next| match next {
                        Some(value) => char_set.contains(&value),
                        None => false,
                    }),
                    SetMembership::Exclusive => Edge::new(move |next| match next {
                        Some(value) => !char_set.contains(&value),
                        None => false,
                    }),
                };

                graph
                    .edges
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Epsilon(InstEpsilon { cond: _cond }) => {
                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        default_dest_state_id,
                        Edge::new_with_action(EpsilonAction::None, |_| {
                            // This was the previous handling of the cond.
                            // Edge::EpsilonWithCondition(cond)
                            true
                        }),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Split(InstSplit { x_branch, y_branch }) => {
                let x_branch_state_id = x_branch.as_usize();
                let y_branch_state_id = y_branch.as_usize();

                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        x_branch_state_id,
                        Edge::new_with_action(EpsilonAction::None, |_| true),
                    )
                    .map_err(|e| e.to_string())?;
                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        y_branch_state_id,
                        Edge::new_with_action(EpsilonAction::None, |_| true),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Jmp(InstJmp { next }) => {
                let next_state_id = next.as_usize();

                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        next_state_id,
                        Edge::new_with_action(EpsilonAction::None, |_| true),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Opcode::StartSave(InstStartSave { slot_id }) => {
                let edge = Edge::new_with_action(EpsilonAction::StartSave(slot_id), |_| true);

                graph
                    .edges
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::EndSave(InstEndSave { slot_id }) => {
                let edge = Edge::new_with_action(EpsilonAction::EndSave(slot_id), |_| true);

                graph
                    .edges
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(id))) => {
                let edge = Edge::new_with_action(EpsilonAction::SetExpressionId(id), |_| true);

                graph
                    .edges
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Match => {
                let terminal_state_id = src_state_id;

                graph
                    .states
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

    #[test]
    fn should_generate_terminal_state_for_consuming_instructions() {
        let opcodes = vec![Opcode::Any, Opcode::Any, Opcode::Match];
        let program = Instructions::new(vec![], opcodes);
        let res = graph_from_runtime_instruction_set(&program);

        assert!(res.is_ok());

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut states = graph.states.values().collect::<Vec<_>>();
        states.sort_by(|a, b| a.id.cmp(&b.id));

        assert_eq!(
            vec![
                &State::new(0, AcceptState::NonAcceptor),
                &State::new(1, AcceptState::NonAcceptor),
                &State::new(2, AcceptState::Acceptor),
            ],
            states
        )
    }

    #[test]
    fn should_generate_correct_block_spans() {
        let opcodes = vec![
            Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(4))),
            Opcode::Any,
            Opcode::Consume(InstConsume::new('c')),
            Opcode::Match,
            Opcode::Any,
            Opcode::ConsumeSet(InstConsumeSet::new(0)),
            Opcode::Match,
        ];
        let program = Instructions::new(vec![], opcodes);
        let spans = block_spans_from_instructions(&program);

        assert_eq!(vec![0..1, 1..4, 4..7], spans);
    }
}
