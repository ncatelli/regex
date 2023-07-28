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

/// Matches any given value.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let mut any_char = Any::new().initial_state();
///
/// // Advances one character.
/// assert_eq!(Some(&'a'), any_char.advance_mut(&'a'));
/// assert!(any_char.is_in_accept_state());
///
/// any_char.initial_state_mut();
/// assert!(any_char.matches("a".chars()));
///
/// any_char.initial_state_mut();
/// assert!(!any_char.matches("ab".chars()));
/// ```
pub struct Any<T> {
    ty: std::marker::PhantomData<T>,
    in_initial_state: bool,
    is_in_final_state: bool,
}

impl<T> Any<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ty: std::marker::PhantomData,
            in_initial_state: true,
            is_in_final_state: false,
        }
    }
}

impl<T: Eq> PatternEvaluatorMut for Any<T> {
    type Item = T;

    fn initial_state_mut(&mut self) {
        // clear the acceptor state if set and set as in initial state.
        self.in_initial_state = true;
    }

    fn is_in_accept_state(&self) -> bool {
        self.is_in_final_state
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        self.is_in_final_state = self.in_initial_state;
        self.in_initial_state = !self.is_in_final_state;

        self.is_in_final_state.then_some(next)
    }
}

impl<T> Default for Any<T> {
    fn default() -> Self {
        Self::new()
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
        self.in_initial_state = !self.is_in_final_state;

        self.is_in_final_state.then_some(next)
    }
}

/// Concatenates to matchers, matching them sequentially.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let literal_a = Literal::new('a').initial_state();
/// let literal_b = Literal::new('b').initial_state();
/// let mut concat = Concatenation::new(literal_a, literal_b).initial_state();
///
/// // Advances one character that matches the expected literal.
/// assert!(concat.matches("ab".chars()));
/// assert!(concat.is_in_accept_state());
/// ```
pub struct Concatenation<T, PE1, PE2> {
    ty: std::marker::PhantomData<T>,
    pe1: PE1,
    pe2_started: bool,
    pe2: PE2,
}

impl<T, PE1, PE2> Concatenation<T, PE1, PE2>
where
    PE1: PatternEvaluatorMut<Item = T>,
    PE2: PatternEvaluatorMut<Item = T>,
{
    #[must_use]
    pub fn new(pe1: PE1, pe2: PE2) -> Self {
        Self {
            ty: std::marker::PhantomData,
            pe1,
            pe2_started: false,
            pe2,
        }
    }
}

impl<T, PE1, PE2> PatternEvaluatorMut for Concatenation<T, PE1, PE2>
where
    PE1: PatternEvaluatorMut<Item = T>,
    PE2: PatternEvaluatorMut<Item = T>,
{
    type Item = T;

    fn initial_state_mut(&mut self) {
        // clear the acceptor state if set and set as in initial state.
        self.pe1.initial_state_mut();

        let pe1_in_accept_state = self.pe1.is_in_accept_state();
        self.pe2_started = pe1_in_accept_state;

        if self.pe2_started {
            self.pe2.initial_state_mut();
        }
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe2.is_in_accept_state()
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        if self.pe2_started {
            self.pe2.advance_mut(next)
        } else {
            let pe1_advanced = self.pe1.advance_mut(next);
            self.pe2_started = self.pe1.is_in_accept_state();

            pe1_advanced
        }
    }
}

/// Matches either of two sub-matchers.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let literal_a = Literal::new('a').initial_state();
/// let literal_b = Literal::new('b').initial_state();
/// let mut alt = Alternation::new(literal_a, literal_b).initial_state();
///
/// // matches either a | b.
/// assert!(alt.matches("a".chars()));
/// assert!(alt.is_in_accept_state());
///
/// alt.initial_state_mut();
/// assert!(alt.matches("b".chars()));
/// assert!(alt.is_in_accept_state());
///
/// alt.initial_state_mut();
/// assert!(!alt.matches("c".chars()));
/// assert!(!alt.is_in_accept_state());
///
/// ```
pub struct Alternation<T, PE1, PE2> {
    ty: std::marker::PhantomData<T>,
    pe1: PE1,
    pe2: PE2,
}

impl<T, PE1, PE2> Alternation<T, PE1, PE2>
where
    PE1: PatternEvaluatorMut<Item = T>,
    PE2: PatternEvaluatorMut<Item = T>,
{
    pub fn new(pe1: PE1, pe2: PE2) -> Self {
        Self {
            ty: std::marker::PhantomData,
            pe1,
            pe2,
        }
    }
}
impl<T, PE1, PE2> PatternEvaluatorMut for Alternation<T, PE1, PE2>
where
    PE1: PatternEvaluatorMut<Item = T>,
    PE2: PatternEvaluatorMut<Item = T>,
{
    type Item = T;

    fn initial_state_mut(&mut self) {
        self.pe1.initial_state_mut();
        self.pe2.initial_state_mut();
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe1.is_in_accept_state() || self.pe2.is_in_accept_state()
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        let pe1_advanced = self.pe1.advance_mut(next);
        if pe1_advanced.is_some() {
            pe1_advanced
        } else {
            self.pe2.advance_mut(next)
        }
    }
}

#[cfg(test)]
mod tests {}
