use std::collections::{HashMap, HashSet};
use std::hash::Hash;

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

#[derive(Debug, PartialEq, Eq)]
enum Edge {
    Epsilon(EpsilonAction),
    /// Transitions to the next state if the condition is met. The action is
    /// equivalent to `EpsilonAction::None`.
    EpsilonWithCondition(EpsilonCond),
    MustMatchOneOf(HashSet<char>),
    MustNotMatchOneOf(HashSet<char>),
    MatchAny,
}

impl Edge {
    fn matches(&self, next_input: Option<&char>) -> bool {
        match (self, next_input) {
            (Edge::Epsilon(_), _) => true,
            (Edge::MustMatchOneOf(_), None) | (Edge::MustNotMatchOneOf(_), None) => false,
            (Edge::MustMatchOneOf(items), Some(c)) => items.contains(c),
            (Edge::MustNotMatchOneOf(items), Some(c)) => items.contains(c),
            (Edge::MatchAny, None) => false,
            (Edge::MatchAny, Some(_)) => true,
            (Edge::EpsilonWithCondition(_), None) => todo!(),
            (Edge::EpsilonWithCondition(_), Some(_)) => todo!(),
        }
    }

    fn is_epsilon(&self) -> bool {
        matches!(self, Edge::Epsilon(_) | Edge::EpsilonWithCondition(_))
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

impl<'a> Nfa<'a, State, Edge, char> for NfaFromAttrGraph<'a> {
    fn states(&self) -> HashSet<&State> {
        self.graph.states.values().collect()
    }

    fn initial_state(&self) -> Option<&'a State> {
        self.graph.states.get(&0)
    }

    fn final_states(&self) -> HashSet<&'a State> {
        self.graph
            .states
            .values()
            .filter(|state| state.kind.is_acceptor())
            .collect()
    }

    fn transition(
        &self,
        current_state: &'a State,
        next_input: Option<&<char as Alphabet>::T>,
    ) -> TransitionResult<'a, State> {
        let state_id = current_state.id;
        let edges = self.graph.edges.adjacency_table().get(&state_id);

        match edges {
            None => TransitionResult::NoMatch,
            Some(edges) => {
                let (matching_edges, is_epsilon) = edges
                    .iter()
                    .filter_map(|DirectedEdgeDestination { dest, edge_value }| {
                        let is_epsilon = edge_value.is_epsilon();
                        let matches = edge_value.matches(next_input);
                        matches.then_some((dest, is_epsilon))
                    })
                    .fold(
                        (Vec::new(), false),
                        |(mut destinations, _), matching_dest| {
                            let destination_state = self.graph.states.get(matching_dest.0).unwrap();
                            destinations.push(destination_state);
                            (destinations, matching_dest.1)
                        },
                    );

                if matching_edges.is_empty() {
                    TransitionResult::NoMatch
                } else if is_epsilon {
                    TransitionResult::Epsilon(matching_edges)
                } else {
                    TransitionResult::Match(matching_edges)
                }
            }
        }
    }
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
                    .add_edge(src_state_id, default_dest_state_id, Edge::MatchAny)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Consume(InstConsume { value }) => {
                let set = [value].into_iter().collect();

                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        default_dest_state_id,
                        Edge::MustMatchOneOf(set),
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
                    SetMembership::Inclusive => Edge::MustMatchOneOf(char_set),
                    SetMembership::Exclusive => Edge::MustNotMatchOneOf(char_set),
                };

                graph
                    .edges
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Epsilon(InstEpsilon { cond }) => {
                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        default_dest_state_id,
                        Edge::EpsilonWithCondition(cond),
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
                        Edge::Epsilon(EpsilonAction::None),
                    )
                    .map_err(|e| e.to_string())?;
                graph
                    .edges
                    .add_edge(
                        src_state_id,
                        y_branch_state_id,
                        Edge::Epsilon(EpsilonAction::None),
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
                        Edge::Epsilon(EpsilonAction::None),
                    )
                    .map_err(|e| e.to_string())?;
            }
            Opcode::StartSave(InstStartSave { slot_id }) => {
                let edge = Edge::Epsilon(EpsilonAction::StartSave(slot_id));

                graph
                    .edges
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::EndSave(InstEndSave { slot_id }) => {
                let edge = Edge::Epsilon(EpsilonAction::EndSave(slot_id));

                graph
                    .edges
                    .add_edge(src_state_id, default_dest_state_id, edge)
                    .map_err(|e| e.to_string())?;
            }
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(id))) => {
                let edge = Edge::Epsilon(EpsilonAction::SetExpressionId(id));

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

fn replace_epsilon_condition_with_corresponding_states(
    graph: AttributeGraph,
) -> Result<AttributeGraph, String> {
    let conditional_epsilon_transitions =
        graph
            .edges
            .edges()
            .into_iter()
            .filter_map(|edge| match &edge.edge_value {
                &Edge::EpsilonWithCondition(cond) => Some((cond, edge.src, edge.dest)),
                _ => None,
            });

    for (cond, src, dest) in conditional_epsilon_transitions {}

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
    fn should_generate_linked_edges_for_consuming_instructions() {
        use directed_graph::DirectedEdge;

        let opcodes = vec![
            Opcode::Any,
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Match,
        ];
        let program = Instructions::new(vec![], opcodes);
        let res = graph_from_runtime_instruction_set(&program);

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut edges = graph.edges.edges();
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
            Opcode::Consume(InstConsume::new('c')),
            Opcode::Match,
            Opcode::Any,
            Opcode::ConsumeSet(InstConsumeSet::new(0)),
            Opcode::Match,
        ];
        let program = Instructions::new(
            vec![CharacterSet::inclusive(CharacterAlphabet::Range('a'..='z'))],
            opcodes,
        );
        let res = graph_from_runtime_instruction_set(&program);

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut edges = graph.edges.edges();
        edges.sort_by(|a, b| a.src.cmp(b.src));

        assert_eq!(
            vec![
                DirectedEdge::new(&0, &1, &Edge::Epsilon(EpsilonAction::None)),
                DirectedEdge::new(&0, &4, &Edge::Epsilon(EpsilonAction::None)),
                DirectedEdge::new(&1, &2, &Edge::MatchAny),
                DirectedEdge::new(&2, &3, &Edge::MustMatchOneOf(['c'].into_iter().collect())),
                DirectedEdge::new(&4, &5, &Edge::MatchAny),
                DirectedEdge::new(
                    &5,
                    &6,
                    &Edge::MustMatchOneOf(('a'..='z').into_iter().collect())
                ),
            ],
            edges
        );

        // entry node plus one for each state.
        assert_eq!(7, graph.edges.nodes().len());
    }
}
