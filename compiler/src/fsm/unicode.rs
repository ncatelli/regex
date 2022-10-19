use super::Language;

#[derive(Clone)]
struct UnicodeChars {
    lower_range: std::ops::Range<u32>,
    upper_range: std::ops::Range<u32>,
}

impl UnicodeChars {
    const LOWER_RANGE: std::ops::Range<u32> = 0..0xD800;
    const UPPER_RANGE: std::ops::Range<u32> = 0xE000..(char::MAX as u32 + 1);
    const CHAR_CNT: usize = 0xD800 + 0x102000;

    fn new() -> Self {
        Self {
            lower_range: Self::LOWER_RANGE.clone(),
            upper_range: Self::UPPER_RANGE.clone(),
        }
    }
}

impl Default for UnicodeChars {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for UnicodeChars {
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

impl DoubleEndedIterator for UnicodeChars {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        self.upper_range
            .next_back()
            .or_else(|| self.lower_range.next_back())
            .and_then(char::from_u32)
    }
}

impl ExactSizeIterator for UnicodeChars {
    #[inline]
    fn len(&self) -> usize {
        self.lower_range.size_hint().0 + self.upper_range.size_hint().0
    }
}

impl std::iter::FusedIterator for UnicodeChars {}

impl Language for UnicodeChars {
    type T = char;

    const VARIANT_CNT: usize = Self::CHAR_CNT;

    fn variants(&self) -> std::collections::HashSet<Self::T> {
        self.clone().into_iter().collect()
    }

    /// Returns true signifying that the item is contained in the language set.
    ///
    /// For UnicodeChars, all values of char are valid.
    fn contains(&self, _: &Self::T) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_count_should_match_specified() {
        let cnt = UnicodeChars::CHAR_CNT;
        let calculated_cnt = UnicodeChars::default().count();
        let calculated_key_cnt = UnicodeChars::default().variants().len();

        assert!((cnt == calculated_cnt) && (cnt == calculated_key_cnt))
    }
}
