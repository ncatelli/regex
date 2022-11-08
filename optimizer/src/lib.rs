#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::Range;

use regex_runtime::*;

mod nfa;
use nfa::{Alphabet, Nfa, TransitionResult};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransitionFuncReturn {
    NoMatch,
    Consuming(usize),
    Epsilon(usize),
}

type BoxedTransitionFunc = Box<dyn Fn(Option<char>) -> TransitionFuncReturn>;

/// Generates a transition via an associated function from a block relative offset.
struct RelativeEdgeTransitionFunc {
    action: Option<EpsilonAction>,
    transition_func: BoxedTransitionFunc,
}

impl RelativeEdgeTransitionFunc {
    fn new(func: impl Fn(Option<char>) -> TransitionFuncReturn + 'static) -> Self {
        Self {
            action: None,
            transition_func: Box::new(func),
        }
    }

    fn new_with_action(
        action: EpsilonAction,
        func: impl Fn(Option<char>) -> TransitionFuncReturn + 'static,
    ) -> Self {
        Self {
            action: Some(action),
            transition_func: Box::new(func),
        }
    }

    /// Converts relative edges to a fixed offset.
    fn into_fixed_offset_edge(self, offset: usize) -> EdgeTransitionFunc {
        let action = self.action;

        let transition_func = move |next| {
            let transition = (*self.transition_func)(next);

            match transition {
                TransitionFuncReturn::Consuming(next) => {
                    TransitionFuncReturn::Consuming(next + offset)
                }
                TransitionFuncReturn::Epsilon(next) => TransitionFuncReturn::Epsilon(next + offset),
                other => other,
            }
        };

        EdgeTransitionFunc {
            offset,
            action,
            transition_func: Box::new(transition_func),
        }
    }
}

/// Generates a transition via an associated function.
struct EdgeTransitionFunc {
    offset: usize,
    action: Option<EpsilonAction>,
    transition_func: BoxedTransitionFunc,
}

impl EdgeTransitionFunc {
    fn is_epsilon(&self) -> bool {
        self.action.is_some()
    }
}

#[derive(Default)]
struct Block {
    span: Range<usize>,
    initial: bool,
    states: Vec<State>,
    edges: Vec<Vec<RelativeEdgeTransitionFunc>>,
}

impl Block {
    fn new(
        span: Range<usize>,
        initial: bool,
        states: Vec<State>,
        edges: Vec<Vec<RelativeEdgeTransitionFunc>>,
    ) -> Self {
        Self {
            span,
            initial,
            states,
            edges,
        }
    }
}

fn block_spans_from_instructions(program: &Instructions) -> Vec<Range<usize>> {
    let program_len = program.program.len();

    let block_starts = {
        let mut block_starts = [0_usize]
            .into_iter()
            .chain(program.program.iter().flat_map(|program| {
                let is_last = (program.offset + 1) == program_len;

                match program.opcode {
                    Opcode::Split(InstSplit { x_branch, y_branch }) => {
                        if is_last {
                            vec![x_branch.as_usize(), y_branch.as_usize()]
                        } else {
                            vec![x_branch.as_usize(), y_branch.as_usize(), program.offset + 1]
                        }
                    }
                    Opcode::Jmp(InstJmp { next }) => {
                        if is_last {
                            vec![next.as_usize()]
                        } else {
                            vec![next.as_usize(), program.offset + 1]
                        }
                    }
                    _ => vec![],
                }
            }))
            .chain([program.program.len()].into_iter())
            .collect::<Vec<_>>();
        block_starts.sort();
        block_starts.dedup();
        block_starts
    };

    if block_starts.len() >= 2 {
        block_starts
            .as_slice()
            .windows(2)
            .map(|window| window[0]..window[1])
            .collect()
    } else {
        vec![0_usize..program.program.len()]
    }
}

/// Finds the block that contains a given id.
fn span_matches_n_block(id: usize, spans: &[Range<usize>]) -> Option<usize> {
    spans
        .iter()
        .enumerate()
        .find_map(|(idx, span)| span.contains(&id).then_some(idx))
}

fn blocks_from_program(program: &Instructions) -> Result<Vec<Block>, String> {
    // slice the instructions into their corresponding blocks.
    let block_spans = block_spans_from_instructions(program);
    let instruction_blocks = block_spans
        .clone()
        .into_iter()
        .map(|span| &program.program[span]);

    let mut blocks = Vec::with_capacity(block_spans.len());
    for instruction_block in instruction_blocks {
        let mut block = {
            // safe to unwrap
            let start = instruction_block.first().map(|inst| inst.offset).unwrap();
            let last = instruction_block
                .last()
                .map(|inst| inst.offset + 1)
                // safe to unwrap
                .unwrap();

            // Entry block will always have an instruction offset of 0.
            let is_initial = start == 0;

            Block {
                span: start..last,
                initial: is_initial,
                states: vec![State::new(0, AcceptState::NonAcceptor)],
                edges: vec![vec![]],
            }
        };

        for inst in instruction_block {
            let state_cnt = block.states.len();

            // This should never underflow. due to above initial state declaration.
            let src_state = state_cnt.checked_sub(1).unwrap();
            let default_dest_state_id = state_cnt;

            match inst.opcode {
                Opcode::Any => {
                    let next_state = State::new(default_dest_state_id, AcceptState::NonAcceptor);

                    block.states.push(next_state);
                    block.edges.push(vec![]);

                    let edge = RelativeEdgeTransitionFunc::new(move |next| {
                        next.map(|_| TransitionFuncReturn::Consuming(default_dest_state_id))
                            .unwrap_or(TransitionFuncReturn::NoMatch)
                    });

                    block.edges[src_state].push(edge);
                }
                Opcode::Consume(InstConsume { value }) => {
                    let next_state = State::new(default_dest_state_id, AcceptState::NonAcceptor);

                    block.states.push(next_state);
                    block.edges.push(vec![]);

                    let edge = RelativeEdgeTransitionFunc::new(move |next| {
                        if next == Some(value) {
                            TransitionFuncReturn::Consuming(default_dest_state_id)
                        } else {
                            TransitionFuncReturn::NoMatch
                        }
                    });

                    block.edges[src_state].push(edge);
                }
                Opcode::ConsumeSet(InstConsumeSet { idx }) => {
                    let next_state = State::new(default_dest_state_id, AcceptState::NonAcceptor);

                    block.states.push(next_state);
                    block.edges.push(vec![]);

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
                        SetMembership::Inclusive => {
                            RelativeEdgeTransitionFunc::new(move |next| match next {
                                Some(value) => {
                                    if char_set.contains(&value) {
                                        TransitionFuncReturn::Consuming(default_dest_state_id)
                                    } else {
                                        TransitionFuncReturn::NoMatch
                                    }
                                }
                                None => TransitionFuncReturn::NoMatch,
                            })
                        }
                        SetMembership::Exclusive => {
                            RelativeEdgeTransitionFunc::new(move |next| match next {
                                Some(value) => {
                                    if !char_set.contains(&value) {
                                        TransitionFuncReturn::Consuming(default_dest_state_id)
                                    } else {
                                        TransitionFuncReturn::NoMatch
                                    }
                                }
                                None => TransitionFuncReturn::NoMatch,
                            })
                        }
                    };

                    block.edges[src_state].push(edge);
                }
                Opcode::Epsilon(InstEpsilon { cond: _cond }) => {
                    todo!()
                }
                Opcode::Split(InstSplit { x_branch, y_branch }) => {
                    let next_state = State::new(default_dest_state_id, AcceptState::NonAcceptor);

                    let x_branch_state_id = x_branch.as_usize();
                    let y_branch_state_id = y_branch.as_usize();

                    // safe to unwrap. All ids should fall within the bounds of a span.
                    let x_branch_block_id = span_matches_n_block(x_branch_state_id, &block_spans)
                        .ok_or_else(|| {
                        format!(
                            "no matching span found for instruction {}",
                            x_branch_state_id
                        )
                    })?;
                    let y_branch_block_id = span_matches_n_block(y_branch_state_id, &block_spans)
                        .ok_or_else(|| {
                        format!(
                            "no matching span found for instruction {}",
                            y_branch_state_id
                        )
                    })?;

                    let x_edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::None,
                        move |_| TransitionFuncReturn::Epsilon(x_branch_block_id),
                    );
                    let y_edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::None,
                        move |_| TransitionFuncReturn::Epsilon(y_branch_block_id),
                    );

                    block.states.push(next_state);
                    block.edges.push(vec![]);

                    block.edges[src_state].push(x_edge);
                    block.edges[src_state].push(y_edge);
                }
                Opcode::Jmp(InstJmp { next }) => {
                    let next = next.as_usize();
                    let next_block_id =
                        span_matches_n_block(next, &block_spans).ok_or_else(|| {
                            format!("no matching span found for instruction {}", next)
                        })?;

                    let edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::None,
                        move |_| TransitionFuncReturn::Epsilon(next_block_id),
                    );

                    block.edges[src_state].push(edge);
                }

                Opcode::StartSave(InstStartSave { slot_id }) => {
                    let next_state = State::new(default_dest_state_id, AcceptState::NonAcceptor);

                    block.states.push(next_state);
                    block.edges.push(vec![]);

                    let edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::StartSave(slot_id),
                        move |_| TransitionFuncReturn::Epsilon(default_dest_state_id),
                    );

                    block.edges[src_state].push(edge);
                }
                Opcode::EndSave(InstEndSave { slot_id }) => {
                    let next_state = State::new(default_dest_state_id, AcceptState::NonAcceptor);

                    block.states.push(next_state);
                    block.edges.push(vec![]);

                    let edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::EndSave(slot_id),
                        move |_| TransitionFuncReturn::Epsilon(default_dest_state_id),
                    );

                    block.edges[src_state].push(edge);
                }
                Opcode::Meta(InstMeta(MetaKind::SetExpressionId(id))) => {
                    let next_state = State::new(default_dest_state_id, AcceptState::NonAcceptor);

                    block.states.push(next_state);
                    block.edges.push(vec![]);

                    let edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::SetExpressionId(id),
                        move |_| TransitionFuncReturn::Epsilon(default_dest_state_id),
                    );

                    block.edges[src_state].push(edge);
                }
                Opcode::Match => {
                    let next_state = State::new(default_dest_state_id, AcceptState::Acceptor);

                    // once in the match state it should never leave.
                    let trapping_edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::None,
                        move |_| TransitionFuncReturn::Epsilon(default_dest_state_id),
                    );

                    block.states.push(next_state);
                    block.edges.push(vec![trapping_edge]);

                    let transition_to_state_edge = RelativeEdgeTransitionFunc::new_with_action(
                        EpsilonAction::None,
                        move |_| TransitionFuncReturn::Epsilon(default_dest_state_id),
                    );

                    block.edges[src_state].push(transition_to_state_edge);
                }
            }
        }

        blocks.push(block);
    }

    Ok(blocks)
}

/// Provides a mapping of states to their corresponding Edge fuctions.
pub struct DirectedGraphWithTransitionFuncs {
    mappings: HashMap<State, Vec<EdgeTransitionFunc>>,
}

impl DirectedGraphWithTransitionFuncs {
    fn new(mappings: HashMap<State, Vec<EdgeTransitionFunc>>) -> Self {
        Self { mappings }
    }

    fn nodes(&self) -> Vec<&State> {
        self.mappings.keys().collect()
    }

    fn adjacency_table(&self) -> &HashMap<State, Vec<EdgeTransitionFunc>> {
        &self.mappings
    }

    fn adjacency_table_mut(&mut self) -> &mut HashMap<State, Vec<EdgeTransitionFunc>> {
        &mut self.mappings
    }

    fn node_by_id(&self, id: usize) -> Option<&State> {
        let state = self
            .mappings
            .get_key_value(&State {
                id,
                kind: AcceptState::NonAcceptor,
            })
            .map(|(k, _)| k);

        if state.is_some() {
            state
        } else {
            self.mappings
                .get_key_value(&State {
                    id,
                    kind: AcceptState::Acceptor,
                })
                .map(|(k, _)| k)
        }
    }

    fn edges_by_id(&self, id: usize) -> Option<&Vec<EdgeTransitionFunc>> {
        let state = self.node_by_id(id)?;

        self.mappings.get(state)
    }
}

struct TransitionTable<ALPHABET: Alphabet> {
    alphabet: std::marker::PhantomData<ALPHABET>,
    mappings: Vec<Vec<Option<usize>>>,
    char_to_index_mapping: HashMap<ALPHABET::T, usize>,
}

impl TransitionTable<char> {
    const EPSILON_COLUMN: usize = char::VARIANT_CNT;

    fn new(states: usize) -> Self {
        let variant_count = char::VARIANT_CNT + 1;

        let mappings = vec![vec![None; states]; variant_count];
        let char_to_index_mapping = char::variants()
            .enumerate()
            .map(|(idx, variant)| (variant, idx))
            .collect();

        Self {
            alphabet: std::marker::PhantomData,
            mappings,
            char_to_index_mapping,
        }
    }

    fn epsilon_column(&self) -> &[Option<usize>] {
        &self.mappings[Self::EPSILON_COLUMN]
    }

    fn non_epsilon_columns(&self) -> &[Vec<Option<usize>>] {
        &self.mappings[0..Self::EPSILON_COLUMN]
    }

    fn update_result(&mut self, state_id: usize, variant: &char, new_val: Option<usize>) {
        let variant_idx = self.char_to_index_mapping.get(variant).unwrap();
        self.mappings[*variant_idx][state_id] = new_val;
    }
}

impl<ALPHABET: Alphabet> AsRef<[Vec<Option<usize>>]> for TransitionTable<ALPHABET> {
    fn as_ref(&self) -> &[Vec<Option<usize>>] {
        &self.mappings
    }
}

/// Generates a directed graph from a given program.
fn graph_from_runtime_instruction_set(
    program: &Instructions,
) -> Result<DirectedGraphWithTransitionFuncs, String> {
    let blocks = blocks_from_program(program)?;
    let (block_offsets, _) = blocks.iter().map(|block| block.states.len()).fold(
        (vec![], 0),
        |(mut acc, offset), block_len| {
            acc.push(offset);
            (acc, offset + block_len)
        },
    );

    let mut states = vec![];
    let mut edges = vec![];
    for (block_idx, block) in blocks.into_iter().enumerate() {
        let block_offset = block_offsets[block_idx];
        let block_states = block.states.into_iter();
        let block_edges = block.edges.into_iter();

        for (state, local_state_edges) in block_states.zip(block_edges) {
            let absolute_idx_state = State::new(block_offset + state.id, state.kind);
            states.push(absolute_idx_state);

            let local_edges = local_state_edges
                .into_iter()
                .map(|edge| edge.into_fixed_offset_edge(block_offset))
                .collect::<Vec<_>>();

            edges.push(local_edges)
        }
    }

    let state_edge_pairs = states.into_iter().zip(edges.into_iter());
    let graph = state_edge_pairs.fold(HashMap::new(), |mut acc, (state, edge)| {
        acc.insert(state, edge);
        acc
    });

    Ok(DirectedGraphWithTransitionFuncs::new(graph))
}

/// Provides the types for constructing an nfa from a type.
pub trait NfaConstructable {
    type Input;
    type Output;
    type Error;

    fn build_nfa(input: Self::Input) -> Result<Self::Output, Self::Error>;
}

pub struct UnicodeNfa<'a> {
    states: HashMap<usize, &'a State>,
    transition_table: TransitionTable<char>,
}

impl<'a> Nfa<'a, State, char> for UnicodeNfa<'a> {
    fn is_final(&self, state: &'a State) -> bool {
        self.final_states().contains(state)
    }

    fn states(&self) -> HashSet<&State> {
        self.states.values().copied().collect()
    }

    fn initial_state(&self) -> Option<&'a State> {
        let initial_id = 0;

        self.states.get(&initial_id).copied()
    }

    fn final_states(&self) -> HashSet<&'a State> {
        self.states
            .values()
            .filter(|state| state.kind == AcceptState::Acceptor)
            .copied()
            .collect()
    }

    fn transition(
        &self,
        current_state: &'a State,
        next_input: Option<&char>,
    ) -> TransitionResult<'a, State> {
        let row_idx = current_state.id;

        match next_input {
            None => {
                let tfr = self.transition_table.epsilon_column()[row_idx];

                match tfr {
                    None => TransitionResult::NoMatch,
                    Some(next) => {
                        let state = self.states.get(&next).unwrap();
                        TransitionResult::Epsilon(vec![state])
                    }
                }
            }
            Some(c) => {
                // safe to guarantee this can be unwrapped for char.
                let column_idx = self.transition_table.char_to_index_mapping.get(c).unwrap();
                let tfr = self.transition_table.non_epsilon_columns()[*column_idx][row_idx];

                match tfr {
                    None => TransitionResult::NoMatch,
                    Some(next) => {
                        let state = self.states.get(&next).unwrap();
                        TransitionResult::Match(vec![state])
                    }
                }
            }
        }
    }
}

impl<'a> NfaConstructable for UnicodeNfa<'a> {
    type Input = &'a DirectedGraphWithTransitionFuncs;
    type Output = Self;
    type Error = String;

    fn build_nfa(input: Self::Input) -> Result<Self::Output, Self::Error> {
        let graph = input;
        let states: HashMap<usize, _> = graph
            .nodes()
            .into_iter()
            .map(|state| (state.id, state))
            .collect();

        let transition_table = {
            let mut transition_table = TransitionTable::<char>::new(states.len());

            // safe to unwrap with above assertion.
            let empty_transition_funcs = vec![];
            for state in states.values() {
                let transition_funcs = graph
                    .edges_by_id(state.id)
                    .unwrap_or(&empty_transition_funcs);

                for c in char::variants() {
                    for transition_func in transition_funcs {
                        let tfr = match (*transition_func.transition_func)(Some(c)) {
                            TransitionFuncReturn::NoMatch => None,
                            TransitionFuncReturn::Consuming(next)
                            | TransitionFuncReturn::Epsilon(next) => Some(next),
                        };

                        transition_table.update_result(state.id, &c, tfr);
                    }
                }
            }

            transition_table
        };

        let unfa = UnicodeNfa {
            states,
            transition_table,
        };

        Ok(unfa)
    }
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
        let mut states = graph.nodes();
        states.sort_by(|a, b| a.id.cmp(&b.id));

        assert_eq!(
            vec![
                &State::new(0, AcceptState::NonAcceptor),
                &State::new(1, AcceptState::NonAcceptor),
                &State::new(2, AcceptState::NonAcceptor),
                &State::new(3, AcceptState::Acceptor),
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
            Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
            Opcode::Match,
        ];
        let program = Instructions::new(vec![], opcodes);
        let spans = block_spans_from_instructions(&program);

        assert_eq!(vec![0..1, 1..2, 2..4, 4..7, 7..8], spans);
    }

    #[test]
    fn should_generate_translation_table_for_alphabet() {
        use super::nfa::{Alphabet, Nfa};
        use super::UnicodeNfa;

        let opcodes = vec![Opcode::Any, Opcode::Any, Opcode::Match];
        let program = Instructions::new(vec![], opcodes);
        let res = graph_from_runtime_instruction_set(&program);

        assert!(res.is_ok());

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let unfa = UnicodeNfa::build_nfa(&graph).unwrap();

        // assert 4 nodes in nfa.
        assert_eq!(4, unfa.states().len());

        let initial_state = unfa.initial_state().unwrap();
        let second_state = unfa
            .states()
            .iter()
            .find(|state| state.id == 1)
            .copied()
            .unwrap();

        // Check all non-epsilon variants
        for letter in char::variants() {
            let transition_from_state_0_to_1 = unfa.transition(initial_state, Some(&letter));
            assert_eq!(
                TransitionResult::Match(vec![second_state]),
                transition_from_state_0_to_1
            );
        }

        // Check Epsilon case
        assert_eq!(
            TransitionResult::NoMatch,
            unfa.transition(initial_state, None),
        );
    }
}
