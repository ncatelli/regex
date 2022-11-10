use std::collections::hash_set::HashSet;
use std::hash::Hash;

pub(crate) trait Alphabet {
    type T: Hash + Eq;
    type VARIANTS: ExactSizeIterator;
    const VARIANT_CNT: usize;

    // Signifies that an element is contained in a language.
    fn contains(&self, item: &Self::T) -> bool;
    fn variants() -> Self::VARIANTS;
}

impl Alphabet for char {
    type T = char;
    type VARIANTS = AllUnicodeChars;
    const VARIANT_CNT: usize = AllUnicodeChars::CHAR_CNT;

    fn contains(&self, _: &Self::T) -> bool {
        true
    }

    fn variants() -> Self::VARIANTS {
        Self::VARIANTS::new()
    }
}

pub(crate) struct AllUnicodeChars {
    lower_range: std::ops::Range<u32>,
    upper_range: std::ops::Range<u32>,
}

impl AllUnicodeChars {
    const LOWER_RANGE: std::ops::Range<u32> = 0..0xD800;
    const UPPER_RANGE: std::ops::Range<u32> = 0xE000..(char::MAX as u32 + 1);
    const CHAR_CNT: usize = 0xD800 + 0x102000;

    pub(crate) fn new() -> Self {
        Self {
            lower_range: Self::LOWER_RANGE.clone(),
            upper_range: Self::UPPER_RANGE.clone(),
        }
    }
}

impl Default for AllUnicodeChars {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for AllUnicodeChars {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        self.lower_range
            .next()
            .or_else(|| self.upper_range.next())
            .and_then(char::from_u32)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl DoubleEndedIterator for AllUnicodeChars {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        self.upper_range
            .next_back()
            .or_else(|| self.lower_range.next_back())
            .and_then(char::from_u32)
    }
}

impl ExactSizeIterator for AllUnicodeChars {
    #[inline]
    fn len(&self) -> usize {
        self.lower_range.size_hint().0 + self.upper_range.size_hint().0
    }
}

impl std::iter::FusedIterator for AllUnicodeChars {}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TransitionResult<'a, STATE>
where
    STATE: Hash + Eq,
{
    Match(Vec<&'a STATE>),
    Epsilon(Vec<&'a STATE>),
    NoMatch,
}

pub(crate) trait Nfa<'a, STATE, ALPHABET>
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

#[derive(Debug, Clone)]
pub(crate) struct DotRepr<T> {
    _kind: std::marker::PhantomData<T>,
    data: String,
}

impl<T> DotRepr<T> {
    pub(crate) fn new(data: String) -> Self {
        Self {
            _kind: std::marker::PhantomData,
            data,
        }
    }
}

impl<T: Sized> std::fmt::Display for DotRepr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

pub(crate) trait DotGeneratable: Sized {
    fn to_dot(&self) -> DotRepr<Self>;
}
