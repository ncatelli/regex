//! Provides methods and types to facilitate the lowering of a parsed ast into
//! a finite state machine.

mod directed_graph;

use std::collections::hash_set::HashSet;
use std::hash::Hash;

pub trait Language {
    type T: Hash + Eq;

    fn variants(&self) -> HashSet<Self::T>;
}

#[derive(Debug, PartialEq, Eq)]
pub enum TransitionResult<'a, STATE>
where
    STATE: Hash + Eq,
{
    Consuming(Vec<&'a STATE>),
    NonConsuming(Vec<&'a STATE>),
    DeadState,
}

impl<'a, STATE> TransitionResult<'a, STATE>
where
    STATE: Hash + Eq,
{
    /// Return the contained value of the `Consuming` or `NonConsuming`
    /// variants.
    ///
    /// # Panics
    /// Panics if the self value equals `DeadState`.
    pub fn unwrap(self) -> Vec<&'a STATE> {
        match self {
            TransitionResult::Consuming(inner) | TransitionResult::NonConsuming(inner) => inner,
            TransitionResult::DeadState => {
                panic!("called `TransitionResult::unwrap()` on a `DeadState` value")
            }
        }
    }

    /// Transforms the `TransitionResult<'a, STATE>> into a
    /// `Result<Vec<&'a STATE>, E>`, mapping `Consuming(inner)` and
    /// `NonConsuming(inner)` to `Ok(inner)` and DeadState to `Err(err())`.
    pub fn ok_or_else<E, F>(self, err: F) -> Result<Vec<&'a STATE>, E>
    where
        F: FnOnce() -> E,
    {
        match self {
            TransitionResult::Consuming(inner) | TransitionResult::NonConsuming(inner) => Ok(inner),
            TransitionResult::DeadState => Err(err()),
        }
    }
}

pub trait NFA<'a, STATE, TF, LANG>
where
    STATE: Hash + Eq,
    LANG: Language,
{
    fn states(&self) -> HashSet<&'a STATE>;
    fn initial_state(&self) -> &'a STATE;
    fn final_states(&self) -> HashSet<&'a STATE>;
    fn transition(&self, _: &'a STATE, _: Option<&LANG::T>) -> TransitionResult<'a, STATE>;
    fn is_final(&self, state: &'a STATE) -> bool {
        self.final_states().contains(state)
    }
}

pub struct ConcreteNFA<'a, STATE, TF, LANG>
where
    STATE: Hash + Eq,
    LANG: Language,
    TF: Fn(&HashSet<&'a STATE>, &'a STATE, Option<&LANG::T>) -> TransitionResult<'a, STATE>,
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
    TF: Fn(&HashSet<&'a STATE>, &'a STATE, Option<&LANG::T>) -> TransitionResult<'a, STATE>,
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
    TF: Fn(&HashSet<&'a STATE>, &'a STATE, Option<&LANG::T>) -> TransitionResult<'a, STATE>,
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
    ) -> TransitionResult<'a, STATE> {
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
    ) -> TransitionResult<'a, States> {
        let a = states.get(&States::A).unwrap();
        let b = states.get(&States::B).unwrap();
        let c = states.get(&States::C).unwrap();

        match (current_state, input) {
            (_, Some(Alphabet::Zero)) => TransitionResult::Consuming(vec![a]),
            (States::A, Some(Alphabet::One)) => TransitionResult::Consuming(vec![b]),
            (States::B, Some(Alphabet::One)) => TransitionResult::Consuming(vec![c]),
            (States::C, Some(Alphabet::One)) => TransitionResult::Consuming(vec![c]),
            (state, None) => TransitionResult::NonConsuming(vec![state]),
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
                    // calculates the transition for a non-epsilon input,
                    // converting the result to an option where `Some`
                    // represents all non-`DeadState` values.
                    .filter_map(|&state| nfa.transition(state, Some(input)).ok_or_else(|| ()).ok())
                    .flatten()
                    .collect::<Vec<_>>()
            });

        assert!(current_states.iter().all(|&state| nfa.is_final(state)))
    }
}
