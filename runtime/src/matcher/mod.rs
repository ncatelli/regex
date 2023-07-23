pub trait PatternEvaluatorMut {
    /// The input interable type to be compared.
    type Item;

    // Defines the evaluator as being in the initial state.
    fn initial_state_mut(&mut self);

    /// Returns a boolean signifying if the match is in a final state.
    fn is_in_accept_state(&self) -> bool;

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
///
/// literal_char.initial_state_mut();
/// assert!(!literal_char.advance_mut(&'b'));
/// ```
pub struct Literal<T> {
    literal: T,
    in_initial_state: bool,
    in_acceptor_state: bool,
}

impl<T> Literal<T> {
    #[must_use]
    pub fn new(literal: T) -> Self {
        Self {
            literal,
            in_initial_state: false,
            in_acceptor_state: false,
        }
    }
}

impl<T: Eq> PatternEvaluatorMut for Literal<T> {
    type Item = T;

    fn initial_state_mut(&mut self) {
        // clear the acceptor state if set and set as in initial state.
        self.in_initial_state = true;
        self.in_acceptor_state = false;
    }

    fn is_in_accept_state(&self) -> bool {
        self.in_acceptor_state
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if !self.in_initial_state {
            return false;
        }

        // transition out of initial state.
        self.in_initial_state = false;
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
/// ```
pub struct Any<T> {
    item_ty: std::marker::PhantomData<T>,
    in_initial_state: bool,
    in_acceptor_state: bool,
}

impl<T> Any<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            in_initial_state: false,
            in_acceptor_state: false,
        }
    }
}

impl<T> PatternEvaluatorMut for Any<T> {
    type Item = T;

    fn initial_state_mut(&mut self) {
        // clear the acceptor state if set and set as in initial state.
        self.in_initial_state = true;
        self.in_acceptor_state = false;
    }

    fn is_in_accept_state(&self) -> bool {
        self.in_acceptor_state
    }

    fn advance_mut(&mut self, _: &Self::Item) -> bool {
        if !self.in_initial_state {
            return false;
        }

        // transition out of initial state.
        self.in_initial_state = false;
        self.in_acceptor_state = true;
        self.in_acceptor_state
    }
}

impl<T> Default for Any<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum EvaluationBranch {
    Pe1,
    Pe2,
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
    which: EvaluationBranch,
    pe1: PE1,
    pe2: PE2,
}

impl<T, PE1, PE2> Concatenation<T, PE1, PE2> {
    pub fn new(pe1: PE1, pe2: PE2) -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            which: EvaluationBranch::Pe1,
            pe1,
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
        self.which = EvaluationBranch::Pe1;
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe1.is_in_accept_state() && self.pe2.is_in_accept_state()
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.is_in_accept_state() {
            return false;
        }

        if self.which == EvaluationBranch::Pe1 {
            let advanced = self.pe1.advance_mut(next);
            // check if pe1 has advanced into an accept state.
            let pe1_accepted = self.pe1.is_in_accept_state();
            // if pe1 transitions into an accept state, initialize p2.
            if pe1_accepted {
                // switch to pe2
                self.which = EvaluationBranch::Pe2;
                self.pe2.initial_state_mut();
            }

            advanced
        } else {
            self.pe2.advance_mut(next)
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

    which: EvaluationBranch,
    pe1: PE1,
    pe2: PE2,
}

impl<T, PE1, PE2> Alternation<T, PE1, PE2> {
    pub fn new(pe1: PE1, pe2: PE2) -> Self {
        Self {
            item_ty: std::marker::PhantomData,
            which: EvaluationBranch::Pe1,
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
        self.which = EvaluationBranch::Pe1;
    }

    fn is_in_accept_state(&self) -> bool {
        self.pe1.is_in_accept_state() || self.pe2.is_in_accept_state()
    }

    fn advance_mut(&mut self, next: &Self::Item) -> bool {
        if self.is_in_accept_state() {
            return false;
        };

        if self.which == EvaluationBranch::Pe2 {
            self.pe2.advance_mut(next)
        } else {
            let advanced = self.pe1.advance_mut(next);

            // if pe1 can advance, accept that input
            if advanced {
                advanced
            // otherwise switch to, and evaluate, pe2
            } else {
                self.which = EvaluationBranch::Pe2;
                self.pe2.initial_state_mut();
                self.pe2.advance_mut(next)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_join_two_valid_evaluators() {
        // ab
        let mut concat = Concatenation::new(Literal::new('a'), Literal::new('b'));

        // happy path with "ab" input.
        concat.initial_state_mut();
        assert!(concat.advance_mut(&'a'));
        assert!(concat.advance_mut(&'b'));

        // happy path with "ab" input.
        concat.initial_state_mut();
        assert!(concat.advance_mut(&'a'));
        assert!(!concat.advance_mut(&'c'));
    }
}
