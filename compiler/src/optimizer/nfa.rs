use std::collections::hash_set::HashSet;
use std::hash::Hash;

pub(crate) trait Alphabet {
    type T: Hash + Eq;

    // Signifies that an element is contained in a language.
    fn contains(&self, item: &Self::T) -> bool;
}

impl Alphabet for char {
    type T = char;

    fn contains(&self, _: &Self::T) -> bool {
        true
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TransitionResult<'a, STATE>
where
    STATE: Hash + Eq,
{
    Match(Vec<&'a STATE>),
    Epsilon(Vec<&'a STATE>),
    NoMatch,
}

#[allow(clippy::upper_case_acronyms)]
pub(crate) trait Nfa<'a, STATE, TF, ALPHABET>
where
    STATE: Hash + Eq,
    ALPHABET: Alphabet,
{
    fn states(&self) -> HashSet<&STATE>;
    fn initial_state(&self) -> Option<&'a STATE>;
    fn final_states(&self) -> HashSet<&'a STATE>;
    /// Takes a state and optional input character and returns the next
    /// transition states.
    fn transition(
        &self,
        current_state: &'a STATE,
        next_input: Option<&ALPHABET::T>,
    ) -> TransitionResult<'a, STATE>;
    fn is_final(&self, state: &'a STATE) -> bool {
        self.final_states().contains(state)
    }
}
