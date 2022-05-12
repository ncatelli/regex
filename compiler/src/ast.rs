#[derive(Debug, PartialEq)]
pub enum Regex {
    StartOfStringAnchored(Expression),
    Unanchored(Expression),
}

// Expression

#[derive(Debug, PartialEq)]
pub struct Expression(pub Vec<SubExpression>);

#[derive(Debug, PartialEq)]
pub struct SubExpression(pub Vec<SubExpressionItem>);

impl From<SubExpressionItem> for SubExpression {
    fn from(src: SubExpressionItem) -> Self {
        Self(vec![src])
    }
}

#[derive(Debug, PartialEq)]
pub enum SubExpressionItem {
    Match(Match),
    Group(Group),
    Anchor(Anchor),
    Backreference(Integer),
}

pub trait IsSubExpressionItem: Into<SubExpressionItem> {}

impl From<Match> for SubExpressionItem {
    fn from(src: Match) -> Self {
        Self::Match(src)
    }
}

impl From<Group> for SubExpressionItem {
    fn from(src: Group) -> Self {
        Self::Group(src)
    }
}

impl From<Anchor> for SubExpressionItem {
    fn from(src: Anchor) -> Self {
        Self::Anchor(src)
    }
}

impl From<Backreference> for SubExpressionItem {
    fn from(src: Backreference) -> Self {
        Self::Backreference(src.0)
    }
}

// Group
#[derive(Debug, PartialEq)]
pub enum Group {
    Capturing {
        expression: Expression,
    },
    CapturingWithQuantifier {
        expression: Expression,
        quantifier: Quantifier,
    },
    NonCapturing {
        expression: Expression,
    },
    NonCapturingWithQuantifier {
        expression: Expression,
        quantifier: Quantifier,
    },
}

impl IsSubExpressionItem for Group {}

pub struct GroupNonCapturingModifier;

// Matchers

#[derive(Debug, PartialEq)]
pub enum Match {
    WithQuantifier {
        item: MatchItem,
        quantifier: Quantifier,
    },
    WithoutQuantifier {
        item: MatchItem,
    },
}

impl IsSubExpressionItem for Match {}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum MatchItem {
    MatchAnyCharacter,
    MatchCharacterClass(MatchCharacterClass),
    MatchCharacter(MatchCharacter),
}

pub trait IsMatchItem: Into<MatchItem> {}

impl From<MatchAnyCharacter> for MatchItem {
    fn from(_: MatchAnyCharacter) -> Self {
        Self::MatchAnyCharacter
    }
}

impl From<MatchCharacterClass> for MatchItem {
    fn from(src: MatchCharacterClass) -> Self {
        Self::MatchCharacterClass(src)
    }
}

impl From<MatchCharacter> for MatchItem {
    fn from(src: MatchCharacter) -> Self {
        Self::MatchCharacter(src)
    }
}

pub struct MatchAnyCharacter;
impl IsMatchItem for MatchAnyCharacter {}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum MatchCharacterClass {
    CharacterGroup(CharacterGroup),
    CharacterClass(CharacterClass),
    CharacterClassFromUnicodeCategory(CharacterClassFromUnicodeCategory),
}

impl IsMatchItem for MatchCharacterClass {}

pub trait IsMatchCharacterClass: Into<MatchCharacterClass> {}

impl From<CharacterGroup> for MatchCharacterClass {
    fn from(src: CharacterGroup) -> Self {
        Self::CharacterGroup(src)
    }
}

impl From<CharacterClass> for MatchCharacterClass {
    fn from(src: CharacterClass) -> Self {
        Self::CharacterClass(src)
    }
}

impl From<CharacterClassFromUnicodeCategory> for MatchCharacterClass {
    fn from(src: CharacterClassFromUnicodeCategory) -> Self {
        Self::CharacterClassFromUnicodeCategory(src)
    }
}

#[derive(Debug, PartialEq)]
pub struct MatchCharacter(pub Char);
impl IsMatchItem for MatchCharacter {}

// Character Classes

#[derive(Debug, PartialEq)]
pub enum CharacterGroup {
    NegatedItems(Vec<CharacterGroupItem>),
    Items(Vec<CharacterGroupItem>),
}

impl IsMatchCharacterClass for CharacterGroup {}

pub struct CharacterGroupNegativeModifier;

#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum CharacterGroupItem {
    CharacterClass(CharacterClass),
    CharacterClassFromUnicodeCategory(UnicodeCategoryName),
    CharacterRange(Char, Char),
    Char(Char),
}

pub trait IsCharacterGroupItem: Into<CharacterGroupItem> {}

impl From<CharacterClass> for CharacterGroupItem {
    fn from(src: CharacterClass) -> Self {
        Self::CharacterClass(src)
    }
}

impl From<CharacterClassFromUnicodeCategory> for CharacterGroupItem {
    fn from(src: CharacterClassFromUnicodeCategory) -> Self {
        Self::CharacterClassFromUnicodeCategory(src.0)
    }
}

impl From<CharacterRange> for CharacterGroupItem {
    fn from(src: CharacterRange) -> Self {
        let CharacterRange {
            lower_bound,
            upper_bound,
        } = src;

        Self::CharacterRange(lower_bound, upper_bound)
    }
}

impl From<Char> for CharacterGroupItem {
    fn from(src: Char) -> Self {
        Self::Char(src)
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum CharacterClass {
    AnyWord,
    AnyWordInverted,
    AnyDecimalDigit,
    AnyDecimalDigitInverted,
}

impl IsMatchCharacterClass for CharacterClass {}
impl IsCharacterGroupItem for CharacterClass {}

pub trait IsCharacterClass: Into<CharacterClass> {}

impl From<CharacterClassAnyWord> for CharacterClass {
    fn from(_: CharacterClassAnyWord) -> Self {
        Self::AnyWord
    }
}

impl From<CharacterClassAnyWordInverted> for CharacterClass {
    fn from(_: CharacterClassAnyWordInverted) -> Self {
        Self::AnyWordInverted
    }
}

impl From<CharacterClassAnyDecimalDigit> for CharacterClass {
    fn from(_: CharacterClassAnyDecimalDigit) -> Self {
        Self::AnyDecimalDigit
    }
}

impl From<CharacterClassAnyDecimalDigitInverted> for CharacterClass {
    fn from(_: CharacterClassAnyDecimalDigitInverted) -> Self {
        Self::AnyDecimalDigitInverted
    }
}

pub struct CharacterClassAnyWord;
impl IsCharacterClass for CharacterClassAnyWord {}

pub struct CharacterClassAnyWordInverted;
impl IsCharacterClass for CharacterClassAnyWordInverted {}

pub struct CharacterClassAnyDecimalDigit;
impl IsCharacterClass for CharacterClassAnyDecimalDigit {}

pub struct CharacterClassAnyDecimalDigitInverted;
impl IsCharacterClass for CharacterClassAnyDecimalDigitInverted {}

#[derive(Debug, PartialEq)]
pub struct CharacterClassFromUnicodeCategory(pub UnicodeCategoryName);
impl IsMatchCharacterClass for CharacterClassFromUnicodeCategory {}
impl IsCharacterGroupItem for CharacterClassFromUnicodeCategory {}

#[derive(Debug, PartialEq)]
pub struct UnicodeCategoryName(pub Letters);

pub struct CharacterRange {
    lower_bound: Char,
    upper_bound: Char,
}

impl CharacterRange {
    pub fn new(lower_bound: Char, upper_bound: Char) -> Self {
        Self {
            lower_bound,
            upper_bound,
        }
    }
}

impl IsCharacterGroupItem for CharacterRange {}

// Quantifiers

#[derive(Debug, PartialEq)]
pub enum Quantifier {
    Eager(QuantifierType),
    Lazy(QuantifierType),
}

/// Signifies that a type is representative of a Regex Quantifier and can be
/// converted to a `QualifierType` variant.
pub trait IsQuantifierType: Into<QuantifierType> {}

/// Represents all variants of quantifier types
#[derive(Debug, PartialEq)]
pub enum QuantifierType {
    MatchExactRange(Integer),
    MatchAtLeastRange(Integer),
    MatchBetweenRange {
        lower_bound: Integer,
        upper_bound: Integer,
    },
    /// Represents a quantifier representing a match of zero or more of the
    /// preceeding field. Represented by the `*` quantifier.
    ZeroOrMore,
    /// Represents a quantifier representing a match of one or more of the
    /// preceeding field. Represented by the `+` quantifier.
    OneOrMore,
    /// Represents an optional quantifier representing a match of zero or one
    /// field. Represented by the `?` quantifier.
    ZeroOrOne,
}

impl From<ZeroOrMoreQuantifier> for QuantifierType {
    fn from(_: ZeroOrMoreQuantifier) -> Self {
        QuantifierType::ZeroOrMore
    }
}

impl From<OneOrMoreQuantifier> for QuantifierType {
    fn from(_: OneOrMoreQuantifier) -> Self {
        QuantifierType::OneOrMore
    }
}

impl From<ZeroOrOneQuantifier> for QuantifierType {
    fn from(_: ZeroOrOneQuantifier) -> Self {
        QuantifierType::ZeroOrOne
    }
}

impl From<RangeQuantifier> for QuantifierType {
    fn from(src: RangeQuantifier) -> Self {
        let lower_bound = src.lower_bound;
        let upper_bound = src.upper_bound;

        match (lower_bound, upper_bound) {
            (lower, None) => QuantifierType::MatchExactRange(lower.0),
            (lower, Some(None)) => QuantifierType::MatchAtLeastRange(lower.0),
            (lower, Some(Some(upper))) => QuantifierType::MatchBetweenRange {
                lower_bound: lower.0,
                upper_bound: upper.0,
            },
        }
    }
}

/// Represents an optional modifier for the quantifier represented by the `?`
/// quantifier.
pub struct LazyModifier;

/// A Regex Range Qualifier representable by the following three expressions.
/// `{n}`: Match exactly.
/// `{n,}`: Match at least.
/// `{n,m}` Match between range.
pub struct RangeQuantifier {
    lower_bound: RangeQuantifierLowerBound,
    upper_bound: Option<Option<RangeQuantifierUpperBound>>,
}

impl RangeQuantifier {
    pub fn new(
        lower_bound: RangeQuantifierLowerBound,
        upper_bound: Option<Option<RangeQuantifierUpperBound>>,
    ) -> Self {
        Self {
            lower_bound,
            upper_bound,
        }
    }
}

impl IsQuantifierType for RangeQuantifier {}

/// A lower-bound representation of a range qualifier.
/// `{n,m}` representing the `n` in the previous expression.
pub struct RangeQuantifierLowerBound(pub Integer);

/// An upper-bound representation of a range qualifier.
/// `{n,m}` representing the `m` in the previous expression.
pub struct RangeQuantifierUpperBound(pub Integer);

/// Represents a quantifier representing a match of zero or more of the
/// preceeding field. Represented by the `*` quantifier.
pub struct ZeroOrMoreQuantifier;

impl IsQuantifierType for ZeroOrMoreQuantifier {}

/// Represents a quantifier representing a match of one or more of the
/// preceeding field. Represented by the `+` quantifier.
pub struct OneOrMoreQuantifier;

impl IsQuantifierType for OneOrMoreQuantifier {}

/// Represents an optional quantifier representing a match of zero or one
/// field. Represented by the `?` quantifier.
pub struct ZeroOrOneQuantifier;

impl IsQuantifierType for ZeroOrOneQuantifier {}

// Backreference

pub struct Backreference(pub Integer);
impl IsSubExpressionItem for Backreference {}

// Anchors

pub struct StartOfStringAnchor;

pub trait IsAnchor: Into<Anchor> {}

#[derive(Debug, PartialEq)]
pub enum Anchor {
    WordBoundary,
    NonWordBoundary,
    StartOfStringOnly,
    EndOfStringOnlyNonNewline,
    EndOfStringOnly,
    PreviousMatchEnd,
    EndOfString,
}

impl IsSubExpressionItem for Anchor {}

impl From<AnchorWordBoundary> for Anchor {
    fn from(_: AnchorWordBoundary) -> Self {
        Self::WordBoundary
    }
}

impl From<AnchorNonWordBoundary> for Anchor {
    fn from(_: AnchorNonWordBoundary) -> Self {
        Self::NonWordBoundary
    }
}

impl From<AnchorStartOfStringOnly> for Anchor {
    fn from(_: AnchorStartOfStringOnly) -> Self {
        Self::StartOfStringOnly
    }
}

impl From<AnchorEndOfStringOnlyNotNewline> for Anchor {
    fn from(_: AnchorEndOfStringOnlyNotNewline) -> Self {
        Self::EndOfStringOnlyNonNewline
    }
}

impl From<AnchorEndOfStringOnly> for Anchor {
    fn from(_: AnchorEndOfStringOnly) -> Self {
        Self::EndOfStringOnly
    }
}

impl From<AnchorPreviousMatchEnd> for Anchor {
    fn from(_: AnchorPreviousMatchEnd) -> Self {
        Self::PreviousMatchEnd
    }
}

impl From<AnchorEndOfString> for Anchor {
    fn from(_: AnchorEndOfString) -> Self {
        Self::EndOfString
    }
}

/// An anchor representing by "\b".
pub struct AnchorWordBoundary;
impl IsAnchor for AnchorWordBoundary {}

/// An anchor representing by "\B".
pub struct AnchorNonWordBoundary;
impl IsAnchor for AnchorNonWordBoundary {}

/// An anchor representing by "\A".
pub struct AnchorStartOfStringOnly;
impl IsAnchor for AnchorStartOfStringOnly {}

/// An anchor representing by "\z".
pub struct AnchorEndOfStringOnlyNotNewline;
impl IsAnchor for AnchorEndOfStringOnlyNotNewline {}

/// An anchor representing by "\Z".
pub struct AnchorEndOfStringOnly;
impl IsAnchor for AnchorEndOfStringOnly {}

/// An anchor representing by "\G".
pub struct AnchorPreviousMatchEnd;
impl IsAnchor for AnchorPreviousMatchEnd {}

/// An anchor representing by "$".
pub struct AnchorEndOfString;
impl IsAnchor for AnchorEndOfString {}

// Terminals

#[derive(Debug, PartialEq)]
#[repr(transparent)]
pub struct Integer(pub isize);

impl Integer {
    pub fn as_isize(&self) -> isize {
        self.0
    }
}

impl From<Integer> for isize {
    fn from(src: Integer) -> Self {
        src.as_isize()
    }
}

#[derive(Debug, PartialEq)]
#[repr(transparent)]
pub struct Letters(pub Vec<char>);

impl Letters {
    pub fn as_char_slice(&self) -> &[char] {
        &self.0
    }
}

impl AsRef<[char]> for Letters {
    fn as_ref(&self) -> &[char] {
        self.as_char_slice()
    }
}

#[derive(Debug, PartialEq)]
#[repr(transparent)]
pub struct Char(pub char);
impl IsCharacterGroupItem for Char {}

impl Char {
    pub fn as_char(&self) -> char {
        self.0
    }
}

impl AsRef<char> for Char {
    fn as_ref(&self) -> &char {
        &self.0
    }
}

impl From<Char> for char {
    fn from(src: Char) -> char {
        src.as_char()
    }
}
