pub trait PatternEvaluatorMut: Sized {
    /// The input interable type to be compared.
    type Item;

    fn initial_state(mut self) -> Self {
        self.initial_state_mut();
        self
    }

    /// Defines the evaluator as being in the initial state. Without resetting
    /// contextual state.
    fn initial_state_mut(&mut self);

    /// Returns a boolean signifying if the match is in a final state.
    fn is_in_accept_state(&self) -> bool;

    /// Attempts to advance to the next state, returning an [Option] signifying
    /// the success of that advance.
    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item>;

    fn matches<I>(&mut self, iter: I) -> bool
    where
        I: Iterator<Item = Self::Item>,
    {
        iter.fold(self.is_in_accept_state(), |_, item| {
            self.advance_mut(&item);

            self.is_in_accept_state()
        })
    }
}

/// Matches a given value.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let mut literal_char = Literal::new('a').initial_state();
///
/// // Advances one character that matches the expected literal.
/// assert_eq!(Some(&'a'), literal_char.advance_mut(&'a'));
/// assert!(literal_char.is_in_accept_state());
///
/// // Fails to match 'b'.
/// literal_char.initial_state_mut();
/// assert!(literal_char.advance_mut(&'b').is_none());
/// assert!(!literal_char.is_in_accept_state());
///
/// literal_char.initial_state_mut();
/// assert!(literal_char.matches("a".chars()));
///
/// literal_char.initial_state_mut();
/// assert!(!literal_char.matches("ab".chars()));
/// ```
pub struct Literal<T> {
    literal: T,
    in_initial_state: bool,
    is_in_final_state: bool,
}

impl<T> Literal<T> {
    #[must_use]
    pub fn new(literal: T) -> Self {
        Self {
            literal,
            in_initial_state: true,
            is_in_final_state: false,
        }
    }
}

impl<T: Eq> PatternEvaluatorMut for Literal<T> {
    type Item = T;

    fn initial_state_mut(&mut self) {
        // clear the acceptor state if set and set as in initial state.
        self.in_initial_state = true;
    }

    fn is_in_accept_state(&self) -> bool {
        self.is_in_final_state
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        let next_matches_literal = next == &self.literal;

        self.is_in_final_state = self.in_initial_state && next_matches_literal;
        self.is_in_final_state.then_some(next)
    }
}

#[cfg(test)]
mod tests {}
