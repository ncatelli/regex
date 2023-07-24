pub trait PatternEvaluatorMut {
    /// The input interable type to be compared.
    type Item;

    // Defines the evaluator as being in the initial state.
    fn initial_state_mut(&mut self);

    /// Returns a boolean signifying if the match is in a final state.
    fn is_in_accept_state(&self) -> bool;

    // signifies there are no additional transitions
    fn is_completed(&self) -> bool;

    /// Attempts to advance to the next state, returning a boolean signifying
    /// the success of that advance.
    fn advance_mut(&mut self, next: &Self::Item) -> bool;
}

/// Matches a given value.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let mut literal_char = Literal::new('a');
/// literal_char.initial_state_mut();
/// assert!(literal_char.advance_mut(&'a'));
/// assert!(literal_char.is_in_accept_state());
/// assert!(literal_char.is_completed());
///
/// literal_char.initial_state_mut();
/// assert!(!literal_char.advance_mut(&'b'));
/// assert!(!literal_char.is_in_accept_state());
/// assert!(literal_char.is_completed());
/// ```
pub struct Literal<T> {
    literal: T,
    in_initial_state: bool,
    in_acceptor_state: bool,
    completed: bool,
}

impl<T> Literal<T> {
    #[must_use]
    pub fn new(literal: T) -> Self {
        Self {
            literal,
            in_initial_state: false,
            in_acceptor_state: false,
            completed: false,
        }
    }
}

impl<T: Eq> PatternEvaluatorMut for Literal<T> {
    type Item = T;

    fn initial_state_mut(&mut self) {
        // clear the acceptor state if set and set as in initial state.
        self.in_initial_state = true;
        self.completed = !self.in_initial_state;
        self.in_acceptor_state = false;
    }

    fn is_in_accept_state(&self) -> bool {
        self.in_acceptor_state
    }

    fn is_completed(&self) -> bool {
        self.completed
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.is_completed() {
            return false;
        }

        // transition out of initial state.
        self.in_initial_state = false;
        self.completed = true;
        self.in_acceptor_state = &self.literal == next;
        self.in_acceptor_state
    }
}

/// Matches any single value.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let mut any_char = Any::new();
/// any_char.initial_state_mut();
/// assert!(any_char.advance_mut(&'a'));
/// assert!(any_char.is_in_accept_state());
/// assert!(any_char.is_completed());
/// ```
pub struct Any<T> {
    item_ty: std::marker::PhantomData<T>,
    in_initial_state: bool,
    in_acceptor_state: bool,
    completed: bool,
}

impl<T> Any<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            in_initial_state: true,
            completed: false,
            in_acceptor_state: false,
        }
    }
}

impl<T> PatternEvaluatorMut for Any<T> {
    type Item = T;

    fn initial_state_mut(&mut self) {
        // clear the acceptor state if set and set as in initial state.
        self.in_initial_state = true;
        self.completed = !self.in_initial_state;
        self.in_acceptor_state = false;
    }

    fn is_in_accept_state(&self) -> bool {
        self.in_acceptor_state
    }

    fn is_completed(&self) -> bool {
        self.completed
    }

    fn advance_mut(&mut self, _: &Self::Item) -> bool {
        if self.is_completed() {
            return false;
        }

        // transition out of initial state.
        self.in_initial_state = false;
        self.completed = true;
        self.in_acceptor_state = true;
        self.in_acceptor_state
    }
}

impl<T> Default for Any<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Matches the concatenation of two patterns, similar to a logical and.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
///  // ab
///  let mut concat = Concatenation::new(Literal::new('a'), Literal::new('b'));
///
///  // happy path with "ab" input.
///  concat.initial_state_mut();
///  assert!(concat.advance_mut(&'a'));
///  assert!(concat.advance_mut(&'b'));
///  assert!(concat.is_in_accept_state());
///
///  // happy path with "ab" input.
///  concat.initial_state_mut();
///  assert!(concat.advance_mut(&'a'));
///  assert!(!concat.advance_mut(&'c'));
/// ```
pub struct Concatenation<T, PE1, PE2> {
    item_ty: std::marker::PhantomData<T>,
    pe1: PE1,
    pe1_was_accepted: bool,
    pe1_was_completed: bool,
    pe2: PE2,
}

impl<T, PE1, PE2> Concatenation<T, PE1, PE2> {
    pub fn new(pe1: PE1, pe2: PE2) -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            pe1,
            pe1_was_accepted: false,
            pe1_was_completed: false,
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
        self.pe1.initial_state_mut();
        self.pe1_was_accepted = false;
        self.pe1_was_completed = false;
        self.pe2.initial_state_mut();
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe1_was_accepted && self.pe2.is_in_accept_state()
    }

    fn is_completed(&self) -> bool {
        self.pe1.is_completed() && self.pe2.is_completed()
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.is_completed() {
            return false;
        }

        let pe1_was_accepted = self.pe1.is_in_accept_state();

        let pe1_advanced = if !self.pe1_was_completed {
            self.pe1.advance_mut(next)
        } else {
            false
        };

        if !pe1_advanced && pe1_was_accepted {
            self.pe1_was_accepted = true;
            self.pe1_was_completed = true;
        } else if !pe1_advanced {
            self.pe1_was_completed = true;
        }

        let pe1_was_completed_and_accepted = self.pe1_was_accepted && self.pe1_was_completed;

        let pe2_advanced = self.pe2.advance_mut(next);
        if (pe1_was_completed_and_accepted) && pe2_advanced {
            pe2_advanced
        } else if pe1_advanced && !pe2_advanced {
            self.pe2.initial_state_mut();
            true
        } else {
            false
        }
    }
}

/// Implements regular expression alternation, a logical or.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let mut alternation = Alternation::new(Literal::new('a'), Literal::new('b'));
///
/// // happy path with either 'a' or 'b' input.
/// alternation.initial_state_mut();
/// assert!(alternation.advance_mut(&'a'));
/// assert!(alternation.is_in_accept_state());
///
/// alternation.initial_state_mut();
/// assert!(alternation.advance_mut(&'b'));
/// assert!(alternation.is_in_accept_state());
///
/// // fails with anything else
/// alternation.initial_state_mut();
/// assert!(!alternation.advance_mut(&'c'));
/// ```
pub struct Alternation<T, PE1, PE2> {
    item_ty: std::marker::PhantomData<T>,

    pe1: PE1,
    pe2: PE2,
}

impl<T, PE1, PE2> Alternation<T, PE1, PE2> {
    pub fn new(pe1: PE1, pe2: PE2) -> Self {
        Self {
            item_ty: std::marker::PhantomData,
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

    fn is_completed(&self) -> bool {
        self.pe1.is_completed() && self.pe2.is_completed()
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.is_completed() {
            return false;
        };

        let mut either_advanced = false;

        if !self.pe1.is_completed() && self.pe1.advance_mut(next) {
            either_advanced = true;
        }

        if !self.pe2.is_completed() && self.pe2.advance_mut(next) {
            either_advanced = true;
        }

        either_advanced
    }
}

/// Matches either zero or one sub-expressions.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
/// let mut zero_or_one = ZeroOrOne::new(Literal::new('a'));
///
/// // matches one
///
/// zero_or_one.initial_state_mut();
/// assert!(zero_or_one.advance_mut(&'a'));
/// assert!(zero_or_one.is_in_accept_state());
///
/// // matches zero
///
/// zero_or_one.initial_state_mut();
/// // should not advance but should still be acceptable
/// assert!(!zero_or_one.advance_mut(&'b'));
/// assert!(zero_or_one.is_in_accept_state());
/// ```
pub struct ZeroOrOne<T, PE> {
    item_ty: std::marker::PhantomData<T>,

    pe: PE,
    completed: bool,
}

impl<T, PE> ZeroOrOne<T, PE> {
    pub fn new(pe: PE) -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            pe,
            completed: false,
        }
    }
}

impl<T, PE> PatternEvaluatorMut for ZeroOrOne<T, PE>
where
    PE: PatternEvaluatorMut<Item = T>,
{
    type Item = T;

    fn initial_state_mut(&mut self) {
        self.pe.initial_state_mut();
        self.completed = false;
    }

    fn is_in_accept_state(&self) -> bool {
        let pe_in_accept_state = self.pe.is_in_accept_state();
        let pe_evaluated = self.pe.is_completed();

        // either the pe is in an acceptible state or it has been evaluated and ignored.
        pe_in_accept_state || pe_evaluated
    }

    fn is_completed(&self) -> bool {
        self.pe.is_completed()
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.pe.is_completed() {
            return false;
        };

        self.pe.advance_mut(next)
    }
}

/// Matches either zero or more sub-expressions.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
///
/// // matches one
///
/// let mut zero_or_more = ZeroOrMore::new(Literal::new('a'));
/// zero_or_more.initial_state_mut();
/// // take more than one
/// assert!(zero_or_more.advance_mut(&'a'));
/// assert!(zero_or_more.advance_mut(&'a'));
/// assert!(zero_or_more.advance_mut(&'a'));
/// assert!(zero_or_more.is_in_accept_state());
///
/// // matches zero
///
/// let mut zero_or_more = ZeroOrMore::new(Literal::new('a'));
/// zero_or_more.initial_state_mut();
/// // should not advance but should still be acceptable
/// assert!(!zero_or_more.advance_mut(&'b'));
/// assert!(zero_or_more.is_in_accept_state());
/// ```
pub struct ZeroOrMore<T, PE> {
    item_ty: std::marker::PhantomData<T>,

    pe: PE,
    completed: bool,
    match_count: usize,
}

impl<T, PE> ZeroOrMore<T, PE> {
    pub fn new(pe: PE) -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            pe,
            completed: false,
            match_count: 0,
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
        self.completed = false;
    }

    fn is_in_accept_state(&self) -> bool {
        let pe_in_accept_state = self.pe.is_in_accept_state();
        let pe_evaluated = self.pe.is_completed();

        // either the pe is in an acceptible state or it has been evaluated and ignored.
        pe_in_accept_state || pe_evaluated
    }

    fn is_completed(&self) -> bool {
        self.completed
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.pe.is_in_accept_state() && self.pe.is_completed() {
            self.match_count += 1;

            // reset the sub-expression
            self.pe.initial_state_mut();
        };

        // advance until completed
        if self.pe.advance_mut(next) {
            true
        } else {
            self.completed = true;
            false
        }
    }
}

/// Matches either one or more sub-expressions.
///
/// # Examples
///
/// ```
/// use regex_runtime::matcher::*;
///
///
/// // matches one
///
/// let mut one_or_more = OneOrMore::new(Literal::new('a'));
/// one_or_more.initial_state_mut();
/// // take more than one
/// assert!(one_or_more.advance_mut(&'a'));
/// assert!(one_or_more.advance_mut(&'a'));
/// assert!(one_or_more.advance_mut(&'a'));
/// assert!(one_or_more.is_in_accept_state());
///
/// // matches one
///
/// let mut one_or_more = OneOrMore::new(Literal::new('a'));
/// one_or_more.initial_state_mut();
/// // should not advance after the `b` but should still be acceptable
/// assert!(one_or_more.advance_mut(&'a'));
/// assert!(!one_or_more.advance_mut(&'b'));
/// assert!(one_or_more.is_in_accept_state());
///
/// // fails to match one
///
/// let mut one_or_more = OneOrMore::new(Literal::new('a'));
/// one_or_more.initial_state_mut();
/// // should not advance and shouldn't be accepted
/// assert!(!one_or_more.advance_mut(&'b'));
/// assert!(!one_or_more.is_in_accept_state());
/// ```
pub struct OneOrMore<T, PE> {
    item_ty: std::marker::PhantomData<T>,

    pe: PE,
    completed: bool,
    match_count: usize,
}

impl<T, PE> OneOrMore<T, PE> {
    pub fn new(pe: PE) -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            pe,
            completed: false,
            match_count: 0,
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
        self.completed = false;
    }

    fn is_in_accept_state(&self) -> bool {
        let pe_in_accept_state = self.pe.is_in_accept_state();
        let has_atleast_one_match = self.match_count > 0;

        // it's acceptable if it has atleast one match
        pe_in_accept_state || has_atleast_one_match
    }

    fn is_completed(&self) -> bool {
        self.completed
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.pe.is_in_accept_state() && self.pe.is_completed() {
            self.match_count += 1;

            // reset the sub-expression
            self.pe.initial_state_mut();
        };

        // advance until completed
        if self.pe.advance_mut(next) {
            true
        } else {
            self.completed = true;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_accepted_unanchored_match_expression() {
        let input = "aaaab";

        let literal = Literal::new('a');
        let unanchored_prefix = ZeroOrMore::new(Any::new());
        let mut expression = Alternation::new(unanchored_prefix, literal);

        expression.initial_state_mut();

        for char in input.chars() {
            assert!(expression.advance_mut(&char));
        }

        assert!(expression.is_in_accept_state());
    }
}
