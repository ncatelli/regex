use std::collections::{HashMap, HashSet};

use regex_runtime::{Instruction, Instructions, Opcode};

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

fn graph_from_bytecode_program(program: &Instructions) -> Result<AttributeGraph, String> {
    let entry_state_id = usize::MAX;
    let initial_thread = Thread::new(vec![entry_state_id], &program.program[0..]);

    let mut current_thread_list = vec![initial_thread];
    let mut next_thread_list = vec![];
    let mut graph = {
        let mut graph = AttributeGraph::new();
        let entry_state = State::new(entry_state_id, AcceptState::NonAcceptor, 0);

        graph.add_node(entry_state);
        graph
    };

    while !current_thread_list.is_empty() {
        'thread: for thread in current_thread_list.iter_mut() {
            for inst in thread.instructions {
                let stack = &mut thread.stack;
                // instruction offset, used as a state id.
                let state_id = inst.offset;
                let parent_state_id = stack
                    .last()
                    .copied()
                    // This should never possibly occur.
                    .ok_or_else(|| "state stack is empty".to_string())?;
                let (parent_expr_id, _parent_save_group_id) = graph
                    .states_attrs
                    .get(&parent_state_id)
                    .map(|node| (node.expr_id, node.save_group_id))
                    .ok_or_else(|| "unknown state".to_string())?;

                match inst.opcode {
                    Opcode::Any => {
                        let new_state =
                            State::new(state_id, AcceptState::NonAcceptor, parent_expr_id);
                        graph.add_node(new_state);
                        stack.push(state_id);

                        let new_edge = Edge::Any;
                        graph
                            .graph
                            .add_edge(parent_state_id, state_id, new_edge)
                            .map_err(|e| e.to_string())?;
                    }
                    Opcode::Consume(regex_runtime::InstConsume { value }) => {
                        let new_state =
                            State::new(state_id, AcceptState::NonAcceptor, parent_expr_id);

                        graph.add_node(new_state);
                        stack.push(state_id);

                        let set = [value].into_iter().collect();
                        let new_edge = Edge::MustMatchOneOf(set);
                        graph
                            .graph
                            .add_edge(parent_state_id, state_id, new_edge)
                            .map_err(|e| e.to_string())?;
                    }
                    Opcode::ConsumeSet(_) => {
                        todo!()
                    }
                    Opcode::Epsilon(_) => todo!(),
                    Opcode::Split(regex_runtime::InstSplit { x_branch, y_branch }) => {
                        let x = x_branch.as_usize();
                        let y = y_branch.as_usize();
                        let x_opcode_start = &program.program[x..];
                        let y_opcode_start = &program.program[y..];

                        next_thread_list.push(Thread::new(stack.clone(), x_opcode_start));
                        next_thread_list.push(Thread::new(stack.clone(), y_opcode_start));
                        break 'thread;
                    }
                    Opcode::Jmp(_) => {}
                    Opcode::StartSave(_) => todo!(),
                    Opcode::EndSave(_) => todo!(),
                    Opcode::Meta(regex_runtime::InstMeta(
                        regex_runtime::MetaKind::SetExpressionId(_),
                    )) => {
                        todo!()
                    }
                    Opcode::Match => {
                        graph
                            .states_attrs
                            .entry(parent_state_id)
                            .and_modify(|node| node.kind = AcceptState::Acceptor);
                        break;
                    }
                }
            }
        }

        std::mem::swap(&mut current_thread_list, &mut next_thread_list);
        next_thread_list.clear();
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex_runtime::*;

    #[test]
    fn empty_program_should_generate_initial_state() {
        let program = Instructions::new(vec![], vec![]);
        let res = graph_from_bytecode_program(&program);

        assert!(res.is_ok());

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        assert_eq!(1, graph.graph.nodes().len());
    }

    #[test]
    fn should_generate_terminal_state_for_consuming_instructions() {
        let opcodes = vec![Opcode::Any, Opcode::Any, Opcode::Match];
        let program = Instructions::new(vec![], opcodes);
        let res = graph_from_bytecode_program(&program);

        assert!(res.is_ok());

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut states = graph.states_attrs.values().collect::<Vec<_>>();
        states.sort_by(|a, b| a.id.cmp(&b.id));

        assert_eq!(
            vec![
                &State::new(0, AcceptState::NonAcceptor, 0),
                &State::new(1, AcceptState::Acceptor, 0),
                &State::new(usize::MAX, AcceptState::NonAcceptor, 0),
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
        let res = graph_from_bytecode_program(&program);

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut edges = graph.graph.edges();
        edges.sort_by(|a, b| a.src.cmp(b.src));

        assert_eq!(
            vec![
                DirectedEdge::new(&0, &1, &Edge::MustMatchOneOf(['a'].into_iter().collect())),
                DirectedEdge::new(&usize::MAX, &0, &Edge::Any),
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
        let res = graph_from_bytecode_program(&program);

        // safe to unwrap with above assertion.
        let graph = res.unwrap();
        let mut edges = graph.graph.edges();
        edges.sort_by(|a, b| a.src.cmp(b.src));

        assert_eq!(
            vec![
                DirectedEdge::new(&1, &2, &Edge::Any),
                DirectedEdge::new(&4, &5, &Edge::Any),
                DirectedEdge::new(&usize::MAX, &1, &Edge::Any),
                DirectedEdge::new(&usize::MAX, &4, &Edge::Any),
            ],
            edges
        );

        // entry node plus 4 consuming nodes.
        assert_eq!(5, graph.graph.nodes().len());
    }
}
