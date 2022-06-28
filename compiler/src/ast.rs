//! Defines the internal representation of the parsed regex for use within the compiler.

/// Represents a complete regex expression, either anchored or unanchored and
/// functions as the top-level of an AST.
#[derive(Debug, PartialEq)]
pub enum Regex {
    StartOfStringAnchored(Expression),
    Unanchored(Expression),
}

// Expression

/// A single expression consisting of 0 or more sub-expressions.
#[derive(Debug, PartialEq)]
pub struct Expression(pub Vec<SubExpression>);

/// A sub-expression consisting of one or more sub-expression items, being one
/// of:
/// - Match
/// - Group
/// - Anchor
/// - Backreference (In grammar for completeness but not supported.)
#[derive(Debug, PartialEq)]
pub struct SubExpression(pub Vec<SubExpressionItem>);

impl From<SubExpressionItem> for SubExpression {
    fn from(src: SubExpressionItem) -> Self {
        Self(vec![src])
    }
}

/// An inner sub-expression representation being one of the following:
/// - Match
/// - Group
/// - Anchor
/// - Backreference (In grammar for completeness but not supported.)
#[derive(Debug, PartialEq)]
pub enum SubExpressionItem {
    Match(Match),
    Group(Group),
    Anchor(Anchor),
    Backreference(Integer),
}

/// A marker trait for representing a type is a `SubExpressionItem` variant.
pub trait SubExpressionItemConvertible: Into<SubExpressionItem> {}

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

/// Represents a group expression in a regex. Either `(a)` `(?:a)`.
#[derive(Debug, PartialEq)]
pub enum Group {
    // A capturing group: ex. '(a)'
    Capturing {
        expression: Expression,
    },
    // A capturing group with a quantifier: ex. '(a)*'
    CapturingWithQuantifier {
        expression: Expression,
        quantifier: Quantifier,
    },
    // A non-capturing group: ex. '(?:a)'
    NonCapturing {
        expression: Expression,
    },
    // A non-capturing group with a quantifier: ex. '(?:a)*'
    NonCapturingWithQuantifier {
        expression: Expression,
        quantifier: Quantifier,
    },
}

impl SubExpressionItemConvertible for Group {}

/// Representative of the non-capturing group modifier `?:` as seen in a
/// non-capturing group, ex: `(?:a)`.
pub struct GroupNonCapturingModifier;

// Matchers

/// A consuming match consuming against a few match cases defined from its MatchItem.
#[derive(Debug, PartialEq)]
pub enum Match {
    /// A quantified match, ex: `a*`.
    WithQuantifier {
        item: MatchItem,
        quantifier: Quantifier,
    },
    /// An unquantified match, ex: `a`.
    WithoutQuantifier { item: MatchItem },
}

impl SubExpressionItemConvertible for Match {}

/// The inner representation of a match.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum MatchItem {
    /// Represents any unicode character, ex: `.`.
    MatchAnyCharacter,
    /// Represents a unicode character class, examples being:
    /// - character group
    /// - character class
    /// - unicode category
    MatchCharacterClass(MatchCharacterClass),
    /// Represents an explicit character, ex: `a`.
    MatchCharacter(MatchCharacter),
}

/// A marker trait for representing a type is a `MatchItem` variant.
pub trait MatchItemConvertible: Into<MatchItem> {}

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

/// Represents a consuming match of any single character, ex `.`.
pub struct MatchAnyCharacter;
impl MatchItemConvertible for MatchAnyCharacter {}

/// Represents a character class match.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum MatchCharacterClass {
    /// A group, or set, of characters, `[0-9]` or `[abcd]`.
    CharacterGroup(CharacterGroup),
    /// A character class, `\w` or `\d`.
    CharacterClass(CharacterClass),
    /// A unicode category derived class, `\p{Letter}`.
    CharacterClassFromUnicodeCategory(CharacterClassFromUnicodeCategory),
}

impl MatchItemConvertible for MatchCharacterClass {}

/// A marker trait for representing a type is a `MatchCharacterClass` variant.
pub trait MatchCharacterClassConvertible: Into<MatchCharacterClass> {}

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

/// Represents a consuming match against a single unicode character.
#[derive(Debug, PartialEq)]
pub struct MatchCharacter(pub Char);
impl MatchItemConvertible for MatchCharacter {}

// Character Classes

/// A set of characters or groups to match against.
#[derive(Debug, PartialEq)]
pub enum CharacterGroup {
    /// Represents that the matching set is exclusive of the defined group
    /// items, ex: `[^abcd]`
    NegatedItems(Vec<CharacterGroupItem>),
    /// Represents that the matching set is inclusive of the defined group
    /// items, ex: `[abcd]`
    Items(Vec<CharacterGroupItem>),
}

impl MatchCharacterClassConvertible for CharacterGroup {}

/// Represents the negation modifier for a character class `^`.
pub struct CharacterGroupNegativeModifier;

/// The inner representation of a character group.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum CharacterGroupItem {
    /// A character class, `\d` or `\w`.
    CharacterClass(CharacterClass),
    /// A unicode category, `[\p{Letter}]`
    CharacterClassFromUnicodeCategory(UnicodeCategoryName),
    /// A range of characters, `[0-9]` or `[a-z]`.
    CharacterRange(Char, Char),
    /// A single character, `[a]
    Char(Char),
}

/// A marker trait for representing a type is a `CharacterGroupItem` variant.
pub trait CharacterGroupItemConvertible: Into<CharacterGroupItem> {}

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

/// A Regex character class.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, PartialEq)]
pub enum CharacterClass {
    /// An any word match, `\w`.
    AnyWord,
    /// Matches anything but a word, `\W`.
    AnyWordInverted,
    /// Matches any decimal digit, `\d`.
    AnyDecimalDigit,
    /// Anything but a decimal digit, `\D`.
    AnyDecimalDigitInverted,
}

impl MatchCharacterClassConvertible for CharacterClass {}
impl CharacterGroupItemConvertible for CharacterClass {}

/// A marker trait for representing a type is a `CharacterClass` variant.
pub trait CharacterClassConvertible: Into<CharacterClass> {}

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

/// A concrete representation of a `\w` CharacterClass variant.
pub struct CharacterClassAnyWord;
impl CharacterClassConvertible for CharacterClassAnyWord {}

/// A concrete representation of a `\W` CharacterClass variant.
pub struct CharacterClassAnyWordInverted;
impl CharacterClassConvertible for CharacterClassAnyWordInverted {}

/// A concrete representation of a `\d` CharacterClass variant.
pub struct CharacterClassAnyDecimalDigit;
impl CharacterClassConvertible for CharacterClassAnyDecimalDigit {}

/// A concrete representation of a `\D` CharacterClass variant.
pub struct CharacterClassAnyDecimalDigitInverted;
impl CharacterClassConvertible for CharacterClassAnyDecimalDigitInverted {}

/// Represents a unicode category character class.
#[derive(Debug, PartialEq)]
pub struct CharacterClassFromUnicodeCategory(pub UnicodeCategoryName);
impl MatchCharacterClassConvertible for CharacterClassFromUnicodeCategory {}
impl CharacterGroupItemConvertible for CharacterClassFromUnicodeCategory {}

/// An enum representation of possible Unicode General Categories specifiable
/// in the unicode category character class matchers.
#[derive(Debug, PartialEq)]
pub enum UnicodeCategoryName {
    Letter,
    LowercaseLetter,
    UppercaseLetter,
    TitlecaseLetter,
    CasedLetter,
    ModifiedLetter,
    OtherLetter,
    Mark,
    NonSpacingMark,
    SpacingCombiningMark,
    EnclosingMark,
    Separator,
    SpaceSeparator,
    LineSeparator,
    ParagraphSeparator,
    Symbol,
    MathSymbol,
    CurrencySymbol,
    ModifierSymbol,
    OtherSymbol,
    Number,
    DecimalDigitNumber,
    LetterNumber,
    OtherNumber,
    Punctuation,
    DashPunctuation,
    OpenPunctuation,
    ClosePunctuation,
    InitialPunctuation,
    FinalPunctuation,
    ConnectorPunctuation,
    OtherPunctuation,
    Other,
    Control,
    Format,
    PrivateUse,
    Surrogate,
    Unassigned,
}

/// A concrete representation of a character range character group item.
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

impl CharacterGroupItemConvertible for CharacterRange {}

// Quantifiers

/// Represents a quanitifer, ex:
/// - `+`
/// - `*`
/// - `?`
/// - {2}
/// - {2,4}
/// - {2,}
#[derive(Debug, PartialEq)]
pub enum Quantifier {
    /// Represents an eager quantifier, ex. `*`.
    Eager(QuantifierType),
    /// Represents a lazy quantifier, ex. `*?`.
    Lazy(QuantifierType),
}

/// Signifies that a type is representative of a Regex Quantifier and can be
/// converted to a `QualifierType` variant.
pub trait IsQuantifierType: Into<QuantifierType> {}

/// Represents all variants of quantifier types
#[derive(Debug, PartialEq)]
pub enum QuantifierType {
    /// Represents a quantifier matching a specified exact number of elements.
    /// Represented by the `{n}` quantifier where `n` is a positive integer.
    MatchExactRange(Integer),
    /// Represents a quantifier matching atleast a specified number of elements.
    /// Represented by the `{n,}` quantifier where `n` is a positive integer.
    MatchAtLeastRange(Integer),
    /// Represents a quantifier matching within a range of elements `n >= m`.
    /// Represented by the `{n,m}` quantifier where `n` and `m` are positive
    /// integers.
    MatchBetweenRange {
        lower_bound: Integer,
        upper_bound: Integer,
    },
    /// Represents a quantifier matching zero or more of the
    /// preceeding field. Represented by the `*` quantifier.
    ZeroOrMore,
    /// Represents a quantifier matching of one or more of the
    /// preceeding field. Represented by the `+` quantifier.
    OneOrMore,
    /// Represents an optional quantifier matching of zero or one
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

/// Represents an optional modifier represented by the `?`
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

/// A lower-bound of a range qualifier. `{n,m}` representing the `n` in the
/// previous expression.
pub struct RangeQuantifierLowerBound(pub Integer);

/// An upper-bound of a range qualifier. `{n,m}` representing the `m` in the
/// previous expression.
pub struct RangeQuantifierUpperBound(pub Integer);

/// Represents a quantifier matching zero or more of the preceeding field.
/// Represented by the `*` quantifier.
pub struct ZeroOrMoreQuantifier;

impl IsQuantifierType for ZeroOrMoreQuantifier {}

/// Represents a quantifier matching one or more of the preceeding field.
/// Represented by the `+` quantifier.
pub struct OneOrMoreQuantifier;

impl IsQuantifierType for OneOrMoreQuantifier {}

/// Represents an optional quantifier matching zero or one field. Represented
/// by the `?` quantifier.
pub struct ZeroOrOneQuantifier;

impl IsQuantifierType for ZeroOrOneQuantifier {}

// Backreference

/// A Backreference element. This is present for completeness of the spec but
/// is both unsupported and unimplemented in this engine.
pub struct Backreference(pub Integer);
impl SubExpressionItemConvertible for Backreference {}

// Anchors

/// A start of string anchor represented as `^`.
pub struct StartOfStringAnchor;

/// A marker trait for representing a type is a `Anchor` variant.
pub trait AnchorConvertible: Into<Anchor> {}

/// An Anchor element in the AST containing both anchor and boundary elements.
#[derive(Debug, PartialEq)]
pub enum Anchor {
    /// A word boundary, ex. `\b`.
    WordBoundary,
    /// A non-word boundary, ex. `\B`.
    NonWordBoundary,
    /// A start of string only boundary, ex. `\A`.
    StartOfStringOnly,
    /// A non-newline end of string boundary, ex. `\z`.
    EndOfStringOnlyNonNewline,
    /// An end of string only boundary, ex. `\Z`.
    EndOfStringOnly,
    /// An end of previous match boundary, ex. `\G`.
    PreviousMatchEnd,
    /// An end of string anchor boundary, ex. `$`.
    EndOfString,
}

impl SubExpressionItemConvertible for Anchor {}

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

/// A boundary representing by "\b".
pub struct AnchorWordBoundary;
impl AnchorConvertible for AnchorWordBoundary {}

/// A boundary representing by "\B".
pub struct AnchorNonWordBoundary;
impl AnchorConvertible for AnchorNonWordBoundary {}

/// A boundary representing by "\A".
pub struct AnchorStartOfStringOnly;
impl AnchorConvertible for AnchorStartOfStringOnly {}

/// A boundary representing by "\z".
pub struct AnchorEndOfStringOnlyNotNewline;
impl AnchorConvertible for AnchorEndOfStringOnlyNotNewline {}

/// A boundary representing by "\Z".
pub struct AnchorEndOfStringOnly;
impl AnchorConvertible for AnchorEndOfStringOnly {}

/// An boundary representing by "\G".
pub struct AnchorPreviousMatchEnd;
impl AnchorConvertible for AnchorPreviousMatchEnd {}

/// An anchor representing by "$".
pub struct AnchorEndOfString;
impl AnchorConvertible for AnchorEndOfString {}

// Terminals

/// Representative of a positive integer in ast.
#[derive(Debug, PartialEq)]
#[repr(transparent)]
pub struct Integer(pub usize);

impl Integer {
    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<Integer> for usize {
    fn from(src: Integer) -> Self {
        src.as_usize()
    }
}

/// Represents one or more unicode characters.
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

/// Represents a single unicode character.
#[derive(Debug, PartialEq)]
#[repr(transparent)]
pub struct Char(pub char);
impl CharacterGroupItemConvertible for Char {}

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
