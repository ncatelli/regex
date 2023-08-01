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
        let mut accepted = self.is_in_accept_state();
        for item in iter {
            let advanced = self.advance_mut(&item);
            accepted = self.is_in_accept_state();
            if advanced.is_none() {
                break;
            }
        }

        accepted
    }
}

/// Matches no input.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let mut nothing = Nothing::new().initial_state();
///
/// assert_eq!(None, nothing.advance_mut(&'a'));
/// assert!(!nothing.is_in_accept_state());
///
/// nothing.initial_state_mut();
/// assert!(nothing.matches("".chars()));
///
/// nothing.initial_state_mut();
/// assert!(!nothing.matches("a".chars()));
/// ```
#[derive(Debug, Clone)]
pub struct Nothing<T> {
    ty: std::marker::PhantomData<T>,
    is_in_final_state: bool,
}

impl<T> Nothing<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ty: std::marker::PhantomData,
            is_in_final_state: true,
        }
    }
}

impl<T> PatternEvaluatorMut for Nothing<T> {
    type Item = T;

    fn initial_state_mut(&mut self) {
        self.is_in_final_state = true;
    }

    fn is_in_accept_state(&self) -> bool {
        self.is_in_final_state
    }

    fn advance_mut<'a>(&mut self, _: &'a Self::Item) -> Option<&'a Self::Item> {
        self.is_in_final_state = false;

        None
    }
}

impl<T> Default for Nothing<T> {
    fn default() -> Self {
        Self::new()
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
#[derive(Debug, Clone)]
pub struct Any<T> {
    ty: std::marker::PhantomData<T>,
    in_initial_state: bool,
}

impl<T> Any<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ty: std::marker::PhantomData,
            in_initial_state: true,
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
        !self.in_initial_state
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        self.in_initial_state = !self.in_initial_state;

        (!self.in_initial_state).then_some(next)
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
#[derive(Debug, Clone)]
pub struct Literal<T> {
    literal: T,
    in_initial_state: bool,
}

impl<T> Literal<T> {
    #[must_use]
    pub fn new(literal: T) -> Self {
        Self {
            literal,
            in_initial_state: true,
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
        !self.in_initial_state
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        let next_matches_literal = next == &self.literal;

        self.in_initial_state = !(self.in_initial_state && next_matches_literal);

        (!self.in_initial_state).then_some(next)
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
#[derive(Debug, Clone)]
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
        self.pe2.initial_state_mut();

        self.pe2_started = self.pe1.is_in_accept_state();
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe1.is_in_accept_state() && self.pe2.is_in_accept_state()
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        let pe2_started_and_advanced = self
            .pe2_started
            .then_some(())
            .and_then(|_| self.pe2.advance_mut(next));

        if pe2_started_and_advanced.is_some() {
            pe2_started_and_advanced
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
/// ```
#[derive(Debug, Clone)]
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

/// Matches either zero or one instance of a sub-matcher.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let literal_a = Literal::new('a');
/// let mut zero_or_one = ZeroOrOne::new(literal_a).initial_state();
///
/// // matches either a | b.
/// assert!(zero_or_one.matches("a".chars()));
/// assert!(zero_or_one.is_in_accept_state());
///
/// zero_or_one.initial_state_mut();
/// assert!(zero_or_one.matches("".chars()));
/// assert!(zero_or_one.is_in_accept_state());
///
/// // Ignores the first match, matching the literal.
/// let mut zero_or_one = Concatenation::new(ZeroOrOne::new(Literal::new('a')), Literal::new('a'))
///     .initial_state();
///
/// zero_or_one.initial_state_mut();
/// assert!(zero_or_one.matches("a".chars()));
/// assert!(zero_or_one.is_in_accept_state());
/// ```
#[derive(Debug, Clone)]
pub struct ZeroOrOne<T, PE> {
    ty: std::marker::PhantomData<T>,
    pe: Alternation<T, PE, Nothing<T>>,
}

impl<T, PE> ZeroOrOne<T, PE>
where
    PE: PatternEvaluatorMut<Item = T>,
{
    pub fn new(pe: PE) -> Self {
        Self {
            ty: std::marker::PhantomData,
            pe: Alternation::new(pe, Nothing::new()),
        }
    }
}

impl<T, PE> PatternEvaluatorMut for ZeroOrOne<T, PE>
where
    PE: PatternEvaluatorMut<Item = T>,
{
    type Item = T;

    fn initial_state_mut(&mut self) {
        self.pe.initial_state_mut()
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe.is_in_accept_state()
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        self.pe.advance_mut(next)
    }
}

/// Matches either zero or one instance of a sub-matcher.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let literal_a = Literal::new('a');
/// let mut zero_or_more = ZeroOrMore::new(literal_a).initial_state();
///
/// // matches one
/// assert!(zero_or_more.matches("a".chars()));
/// assert!(zero_or_more.is_in_accept_state());
///
/// // matches zero
/// zero_or_more.initial_state_mut();
/// assert!(zero_or_more.matches("".chars()));
/// assert!(zero_or_more.is_in_accept_state());
///
/// // matches many
/// zero_or_more.initial_state_mut();
/// assert!(zero_or_more.matches("aaa".chars()));
/// assert!(zero_or_more.is_in_accept_state());
/// ```
#[derive(Debug, Clone)]
pub struct ZeroOrMore<T, PE> {
    ty: std::marker::PhantomData<T>,
    pe: PE,
}

impl<T, PE> ZeroOrMore<T, PE>
where
    PE: PatternEvaluatorMut<Item = T>,
{
    pub fn new(pe: PE) -> Self {
        Self {
            ty: std::marker::PhantomData,
            pe,
        }
    }
}

impl<T, PE> PatternEvaluatorMut for ZeroOrMore<T, PE>
where
    PE: PatternEvaluatorMut<Item = T>,
{
    type Item = T;

    fn initial_state_mut(&mut self) {
        self.pe.initial_state_mut();
    }

    fn is_in_accept_state(&self) -> bool {
        true
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        let advanced = self.pe.advance_mut(next);

        if advanced.is_some() && self.is_in_accept_state() {
            // reset the matcher after an accept state
            self.pe.initial_state_mut();
            advanced
        } else {
            None
        }
    }
}

/// Matches either zero or more instance of a sub-matcher.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let literal_a = Literal::new('a');
/// let mut one_or_more = OneOrMore::new(literal_a).initial_state();
///
/// // matches one
/// assert!(one_or_more.matches("a".chars()));
/// assert!(one_or_more.is_in_accept_state());
///
/// // matches many
/// one_or_more.initial_state_mut();
/// assert!(one_or_more.matches("aaa".chars()));
/// assert!(one_or_more.is_in_accept_state());
///
/// // will fail if first doesn't match
/// one_or_more.initial_state_mut();
/// assert!(!one_or_more.matches("baa".chars()));
/// assert!(!one_or_more.is_in_accept_state());
///
/// // will not match zero
/// one_or_more.initial_state_mut();
/// assert!(!one_or_more.matches("".chars()));
/// assert!(!one_or_more.is_in_accept_state());
/// ```
#[derive(Debug, Clone)]
pub struct OneOrMore<T, PE> {
    ty: std::marker::PhantomData<T>,
    pe: Concatenation<T, PE, ZeroOrMore<T, PE>>,
}

impl<T, PE> OneOrMore<T, PE>
where
    PE: PatternEvaluatorMut<Item = T> + Clone,
{
    pub fn new(pe: PE) -> Self {
        Self {
            ty: std::marker::PhantomData,
            pe: Concatenation::new(
                pe.clone().initial_state(),
                ZeroOrMore::new(pe).initial_state(),
            ),
        }
    }
}

impl<T, PE> PatternEvaluatorMut for OneOrMore<T, PE>
where
    PE: PatternEvaluatorMut<Item = T>,
{
    type Item = T;

    fn initial_state_mut(&mut self) {
        self.pe.initial_state_mut();
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe.is_in_accept_state()
    }

    fn advance_mut<'a>(&mut self, next: &'a Self::Item) -> Option<&'a Self::Item> {
        self.pe.advance_mut(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_match_unanchored_expression() {
        let unanchored = ZeroOrMore::new(Any::new());
        let literal = Literal::new('b');

        // equivalent to `b` expression
        let mut expr = Concatenation::new(unanchored, literal).initial_state();

        expr.initial_state_mut();
        assert!(expr.matches("aab".chars()));
        assert!(expr.is_in_accept_state())
    }
}
