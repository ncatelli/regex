//! Provides methods and types to facilitate the lowering of a parsed ast into
//! a finite state machine.

mod directed_graph;

use std::collections::hash_set::HashSet;
use std::hash::Hash;

pub trait Language {
    type T: Hash + Eq;

    fn variants(&self) -> HashSet<Self::T>;
}

pub trait NFA<'a, STATE, TF, LANG>
where
    STATE: Hash + Eq,
    LANG: Language,
{
    fn states(&self) -> HashSet<&'a STATE>;
    fn initial_state(&self) -> &'a STATE;
    fn final_states(&self) -> HashSet<&'a STATE>;
    fn transition(&self, _: &'a STATE, _: Option<&LANG::T>) -> Option<Vec<&'a STATE>>;
    fn is_final(&self, state: &'a STATE) -> bool {
        self.final_states().contains(state)
    }
}

pub struct ConcreteNFA<'a, STATE, TF, LANG>
where
    STATE: Hash + Eq,
    LANG: Language,
    TF: Fn(&HashSet<&'a STATE>, &'a STATE, Option<&LANG::T>) -> Option<Vec<&'a STATE>>,
{
    language: std::marker::PhantomData<LANG>,
    states: HashSet<&'a STATE>,
    initial_state: &'a STATE,
    final_states: HashSet<&'a STATE>,
    transition_func: TF,
}

impl<'a, STATE, TF, LANG> ConcreteNFA<'a, STATE, TF, LANG>
where
    STATE: Hash + Eq,
    LANG: Language,
    TF: Fn(&HashSet<&'a STATE>, &'a STATE, Option<&LANG::T>) -> Option<Vec<&'a STATE>>,
{
    pub fn try_new(
        states: &[&'a STATE],
        initial_state: &'a STATE,
        final_states: HashSet<&'a STATE>,
        transition_func: TF,
    ) -> Option<Self> {
        let states = states.iter().copied().collect::<HashSet<_, _>>();

        let contains_initial_state = states.contains(initial_state);
        let contains_final_states = final_states
            .iter()
            .all(|final_state| states.contains(final_state));

        if contains_initial_state && contains_final_states {
            Some(Self {
                language: std::marker::PhantomData,
                states,
                initial_state,
                final_states,
                transition_func,
            })
        } else {
            None
        }
    }
}

impl<'a, STATE, TF, LANG> NFA<'a, STATE, TF, LANG> for ConcreteNFA<'a, STATE, TF, LANG>
where
    STATE: Hash + Eq,
    LANG: Language,
    TF: Fn(&HashSet<&'a STATE>, &'a STATE, Option<&LANG::T>) -> Option<Vec<&'a STATE>>,
{
    fn states(&self) -> HashSet<&'a STATE> {
        self.states.clone()
    }

    fn initial_state(&self) -> &'a STATE {
        self.initial_state
    }

    fn final_states(&self) -> HashSet<&'a STATE> {
        self.final_states.clone()
    }

    fn transition(
        &self,
        current_state: &'a STATE,
        next: Option<&<LANG as Language>::T>,
    ) -> Option<Vec<&'a STATE>> {
        (self.transition_func)(&self.states, current_state, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Hash, PartialEq, Eq)]
    enum States {
        A,
        B,
        C,
    }

    #[derive(Debug, Clone, Hash, PartialEq, Eq)]
    enum Alphabet {
        Zero,
        One,
    }

    impl Language for Alphabet {
        type T = Alphabet;

        fn variants(&self) -> HashSet<Self> {
            [Self::Zero, Self::One].into_iter().collect()
        }
    }

    fn ends_in_one_one_transition_func<'a>(
        states: &HashSet<&'a States>,
        current_state: &'a States,
        input: Option<&Alphabet>,
    ) -> Option<Vec<&'a States>> {
        let a = states.get(&States::A).unwrap();
        let b = states.get(&States::B).unwrap();
        let c = states.get(&States::C).unwrap();

        match (current_state, input) {
            (_, Some(Alphabet::Zero)) => Some(vec![a]),
            (States::A, Some(Alphabet::One)) => Some(vec![b]),
            (States::B, Some(Alphabet::One)) => Some(vec![c]),
            (States::C, Some(Alphabet::One)) => Some(vec![c]),
            (state, None) => Some(vec![state]),
        }
    }

    #[test]
    fn should_instantiate_nfa_from_language() {
        let states = vec![&States::A, &States::B, &States::C];
        let final_state = [states[2]].into_iter().collect();

        let nfa = ConcreteNFA::<_, _, Alphabet>::try_new(
            &states,
            states[0],
            final_state,
            ends_in_one_one_transition_func,
        );

        assert!(nfa.is_some())
    }

    #[test]
    fn should_transition_between_nodes() {
        let states = vec![&States::A, &States::B, &States::C];
        let final_state = [states[2]].into_iter().collect();

        let nfa = ConcreteNFA::<_, _, Alphabet>::try_new(
            &states,
            states[0],
            final_state,
            ends_in_one_one_transition_func,
        )
        .unwrap();

        let inputs = [
            Alphabet::Zero,
            Alphabet::One,
            Alphabet::Zero,
            Alphabet::One,
            Alphabet::One,
        ];

        let current_states = inputs
            .iter()
            .fold(vec![nfa.initial_state()], |curr_state, input| {
                curr_state
                    .iter()
                    .filter_map(|&state| nfa.transition(state, Some(input)))
                    .flatten()
                    .collect::<Vec<_>>()
            });

        assert!(current_states.iter().all(|&state| nfa.is_final(state)))
    }
}
