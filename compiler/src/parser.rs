//! Provides methods for parsing an input regex string representation into an
//! internal AST.
//!
//! # Example
//!
//! ```
//! use regex_compiler::ast::*;
//! use regex_compiler::parse;
//!
//! assert_eq!(
//!     Ok(Regex::Unanchored(Expression(vec![SubExpression(vec![
//!         SubExpressionItem::Match(Match::WithoutQuantifier {
//!             item: MatchItem::MatchCharacter(MatchCharacter(Char('a')))
//!         }),
//!         SubExpressionItem::Match(Match::WithoutQuantifier {
//!             item: MatchItem::MatchCharacter(MatchCharacter(Char('b')))
//!         }),
//!     ])]))),
//!     parse("ab")
//! )
//! ```

use parcel::parsers::character::{digit, expect_character, expect_str};
use parcel::prelude::v1::*;

use super::ast;

/// Represents an error stemming from parsing of an input string into a regex
/// AST.
#[derive(PartialEq, Eq)]
pub enum ParseErr {
    InvalidRegex,
    Undefined(String),
}

impl std::fmt::Debug for ParseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undefined(err) => write!(f, "undefined parse error: {}", err),
            Self::InvalidRegex => write!(f, "provided regex is invalid",),
        }
    }
}

/// Accepts an input representing a regex pattern, attempting to parse it into
/// a regex AST.
///
/// # Example
///
/// ```
/// use regex_compiler::ast::*;
/// use regex_compiler::parse;
///
/// assert_eq!(
///     Ok(Regex::Unanchored(Expression(vec![SubExpression(vec![
///         SubExpressionItem::Match(Match::WithoutQuantifier {
///             item: MatchItem::MatchCharacter(MatchCharacter(Char('a')))
///         }),
///         SubExpressionItem::Match(Match::WithoutQuantifier {
///             item: MatchItem::MatchCharacter(MatchCharacter(Char('b')))
///         }),
///     ])]))),
///     parse("ab")
/// )
/// ```
pub fn parse<S: AsRef<str>>(input: S) -> Result<ast::Regex, ParseErr> {
    let input: Vec<(usize, char)> = input.as_ref().chars().enumerate().collect();
    parse_enumerated_slice(&input)
}

/// Accepts an enumerated slice of characters from a given input representing a
/// regex pattern, attempting to parse it into an AST.
///
/// # Example
///
/// ```
/// use regex_compiler::ast::*;
/// use regex_compiler::parser::parse_enumerated_slice;
///
/// let input: Vec<(usize, char)> = "ab".chars().enumerate().collect();
///
/// assert_eq!(
///     Ok(Regex::Unanchored(Expression(vec![SubExpression(vec![
///         SubExpressionItem::Match(Match::WithoutQuantifier {
///             item: MatchItem::MatchCharacter(MatchCharacter(Char('a')))
///         }),
///         SubExpressionItem::Match(Match::WithoutQuantifier {
///             item: MatchItem::MatchCharacter(MatchCharacter(Char('b')))
///         }),
///     ])]))),
///     parse_enumerated_slice(&input)
/// )
/// ```
pub fn parse_enumerated_slice(input: &[(usize, char)]) -> Result<ast::Regex, ParseErr> {
    regex()
        .parse(input)
        .map_err(|err| ParseErr::Undefined(format!("unspecified parse error occured: {}", err)))
        .and_then(|ms| match ms {
            MatchStatus::Match { inner, .. } => Ok(inner),
            MatchStatus::NoMatch(..) => Err(ParseErr::InvalidRegex),
        })
}

fn regex<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::Regex> {
    parcel::join(
        parcel::optional(start_of_string_anchor()).map(|anchored| anchored.is_some()),
        expression(),
    )
    .map(|(anchored, expression)| match anchored {
        true => ast::Regex::StartOfStringAnchored(expression),
        false => ast::Regex::Unanchored(expression),
    })
}

// Expression

fn expression<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::Expression> {
    parcel::join(
        subexpression(),
        parcel::zero_or_more(parcel::right(parcel::join(
            expect_character('|'),
            subexpression(),
        ))),
    )
    .map(|(head, tail)| vec![head].into_iter().chain(tail).collect())
    .map(ast::Expression)
}

fn subexpression<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::SubExpression> {
    parcel::one_or_more(subexpression_item()).map(ast::SubExpression)
}

fn subexpression_item<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::SubExpressionItem>
{
    parcel::or(group().map(Into::into), || {
        parcel::or(backreference().map(Into::into), || {
            parcel::or(anchor().map(Into::into), || r#match().map(Into::into))
        })
    })
}

// Group

fn group<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::Group> {
    parcel::right(parcel::join(
        expect_character('('),
        parcel::optional(group_non_capturing_modifier())
            .map(|non_capturing| non_capturing.is_some()),
    ))
    .and_then(|non_capturing| {
        parcel::join(
            expression().map(|expr| expr),
            parcel::right(parcel::join(
                expect_character(')'),
                parcel::optional(quantifier()),
            )),
        )
        .map(move |expr_and_qualifier| (non_capturing, expr_and_qualifier))
    })
    .map(|(is_non_capturing, (expression, quantifier))| {
        match (is_non_capturing, quantifier) {
            (true, None) => ast::Group::NonCapturing { expression },
            (true, Some(quantifier)) => ast::Group::NonCapturingWithQuantifier {
                expression,
                quantifier,
            },
            (false, None) => ast::Group::Capturing { expression },
            (false, Some(quantifier)) => ast::Group::CapturingWithQuantifier {
                expression,
                quantifier,
            },
        }
    })
}

fn group_non_capturing_modifier<'a>(
) -> impl Parser<'a, &'a [(usize, char)], ast::GroupNonCapturingModifier> {
    parcel::join(expect_character('?'), expect_character(':'))
        .map(|_| ast::GroupNonCapturingModifier)
}

// Matchers

fn r#match<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::Match> {
    parcel::join(match_item(), parcel::optional(quantifier())).map(|(match_item, quantifier)| {
        match quantifier {
            Some(quantifier) => ast::Match::WithQuantifier {
                item: match_item,
                quantifier,
            },
            None => ast::Match::WithoutQuantifier { item: match_item },
        }
    })
}

fn match_item<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::MatchItem> {
    parcel::or(match_character_class().map(Into::into), || {
        parcel::or(match_any_character().map(Into::into), || {
            match_character().map(Into::into)
        })
    })
}

fn match_any_character<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::MatchAnyCharacter>
{
    expect_character('.').map(|_| ast::MatchAnyCharacter)
}

fn match_character_class<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::MatchCharacterClass> {
    parcel::or(
        character_group().map(ast::MatchCharacterClass::CharacterGroup),
        || {
            parcel::or(
                character_class().map(ast::MatchCharacterClass::CharacterClass),
                || {
                    character_class_from_unicode_category()
                        .map(ast::MatchCharacterClass::CharacterClassFromUnicodeCategory)
                },
            )
        },
    )
}

fn match_character<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::MatchCharacter> {
    char()
        .predicate(|c| ![')', '|'].contains(&c.as_char()))
        .map(ast::MatchCharacter)
}

// Character Classes

fn character_group<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterGroup> {
    parcel::join(
        parcel::right(parcel::join(
            expect_character('[').map(|c| c),
            parcel::optional(character_group_negative_modifier())
                .map(|negation| negation.is_some()),
        )),
        parcel::left(parcel::join(
            parcel::one_or_more(character_group_item()),
            expect_character(']'),
        )),
    )
    .map(|(negation, character_group_items)| match negation {
        true => ast::CharacterGroup::NegatedItems(character_group_items),
        false => ast::CharacterGroup::Items(character_group_items),
    })
}

fn character_group_negative_modifier<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterGroupNegativeModifier> {
    expect_character('^').map(|_| ast::CharacterGroupNegativeModifier)
}

fn character_group_item<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterGroupItem> {
    parcel::or(character_class().map(Into::into), || {
        parcel::or(
            character_class_from_unicode_category().map(Into::into),
            || {
                parcel::or(character_range().map(Into::into), || {
                    char().predicate(|ast::Char(c)| *c != ']').map(Into::into)
                })
            },
        )
    })
}

fn character_class<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterClass> {
    parcel::or(character_class_any_word().map(Into::into), || {
        parcel::or(character_class_any_word_inverted().map(Into::into), || {
            parcel::or(character_class_any_decimal_digit().map(Into::into), || {
                character_class_any_decimal_digit_inverted().map(Into::into)
            })
        })
    })
}

fn character_class_any_word<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterClassAnyWord> {
    parcel::join(expect_character('\\'), expect_character('w')).map(|_| ast::CharacterClassAnyWord)
}

fn character_class_any_word_inverted<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterClassAnyWordInverted> {
    parcel::join(expect_character('\\'), expect_character('W'))
        .map(|_| ast::CharacterClassAnyWordInverted)
}

fn character_class_any_decimal_digit<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterClassAnyDecimalDigit> {
    parcel::join(expect_character('\\'), expect_character('d'))
        .map(|_| ast::CharacterClassAnyDecimalDigit)
}

fn character_class_any_decimal_digit_inverted<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterClassAnyDecimalDigitInverted> {
    parcel::join(expect_character('\\'), expect_character('D'))
        .map(|_| ast::CharacterClassAnyDecimalDigitInverted)
}

fn character_class_from_unicode_category<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterClassFromUnicodeCategory> {
    parcel::right(parcel::join(
        parcel::join(expect_character('\\'), expect_character('p')),
        parcel::right(parcel::join(
            expect_character('{'),
            parcel::left(parcel::join(unicode_category_name(), expect_character('}'))),
        )),
    ))
    .map(ast::CharacterClassFromUnicodeCategory)
}

fn unicode_category_name<'a>(
) -> impl parcel::Parser<'a, &'a [(usize, char)], ast::UnicodeCategoryName> {
    use parcel::{one_of, or};

    one_of(vec![
        or(expect_str("Lowercase_Letter"), || expect_str("Ll"))
            .map(|_| ast::UnicodeCategoryName::LowercaseLetter),
        or(expect_str("Uppercase_Letter"), || expect_str("Lu"))
            .map(|_| ast::UnicodeCategoryName::UppercaseLetter),
        or(expect_str("Titlecase_Letter"), || expect_str("Lt"))
            .map(|_| ast::UnicodeCategoryName::TitlecaseLetter),
        or(expect_str("Cased_Letter"), || expect_str("L&"))
            .map(|_| ast::UnicodeCategoryName::CasedLetter),
        or(expect_str("Modified_Letter"), || expect_str("Lm"))
            .map(|_| ast::UnicodeCategoryName::ModifiedLetter),
        or(expect_str("Other_Letter"), || expect_str("Lo"))
            .map(|_| ast::UnicodeCategoryName::OtherLetter),
        or(expect_str("Letter"), || expect_str("L")).map(|_| ast::UnicodeCategoryName::Letter),
        or(expect_str("Non_Spacing_Mark"), || expect_str("Mn"))
            .map(|_| ast::UnicodeCategoryName::NonSpacingMark),
        or(expect_str("Spacing_Combining_Mark"), || expect_str("Mc"))
            .map(|_| ast::UnicodeCategoryName::SpacingCombiningMark),
        or(expect_str("Enclosing_Mark"), || expect_str("Me"))
            .map(|_| ast::UnicodeCategoryName::EnclosingMark),
        or(expect_str("Mark"), || expect_str("M")).map(|_| ast::UnicodeCategoryName::Mark),
        or(expect_str("Space_separator"), || expect_str("Zs"))
            .map(|_| ast::UnicodeCategoryName::SpaceSeparator),
        or(expect_str("Line_Separator"), || expect_str("Zl"))
            .map(|_| ast::UnicodeCategoryName::LineSeparator),
        or(expect_str("Paragraph_Separator"), || expect_str("Zp"))
            .map(|_| ast::UnicodeCategoryName::ParagraphSeparator),
        or(expect_str("Separator"), || expect_str("Z"))
            .map(|_| ast::UnicodeCategoryName::Separator),
        or(expect_str("Math_Symbol"), || expect_str("Sm"))
            .map(|_| ast::UnicodeCategoryName::MathSymbol),
        or(expect_str("Currency_Symbol"), || expect_str("Sc"))
            .map(|_| ast::UnicodeCategoryName::CurrencySymbol),
        or(expect_str("Modifier_Symbol"), || expect_str("Sk"))
            .map(|_| ast::UnicodeCategoryName::ModifierSymbol),
        or(expect_str("Other_Symbol"), || expect_str("So"))
            .map(|_| ast::UnicodeCategoryName::OtherSymbol),
        or(expect_str("Symbol"), || expect_str("S")).map(|_| ast::UnicodeCategoryName::Symbol),
        or(expect_str("Decimal_Digit_Number"), || expect_str("Nd"))
            .map(|_| ast::UnicodeCategoryName::DecimalDigitNumber),
        or(expect_str("Letter_Number"), || expect_str("Nl"))
            .map(|_| ast::UnicodeCategoryName::LetterNumber),
        or(expect_str("Other_Number"), || expect_str("No"))
            .map(|_| ast::UnicodeCategoryName::OtherNumber),
        or(expect_str("Number"), || expect_str("N")).map(|_| ast::UnicodeCategoryName::Number),
        or(expect_str("Dash_Punctuation"), || expect_str("Pd"))
            .map(|_| ast::UnicodeCategoryName::DashPunctuation),
        or(expect_str("Open_Punctuation"), || expect_str("Ps"))
            .map(|_| ast::UnicodeCategoryName::OpenPunctuation),
        or(expect_str("Close_Punctuation"), || expect_str("Pe"))
            .map(|_| ast::UnicodeCategoryName::ClosePunctuation),
        or(expect_str("Initial_Punctuation"), || expect_str("Pi"))
            .map(|_| ast::UnicodeCategoryName::InitialPunctuation),
        or(expect_str("Final_Punctuation"), || expect_str("Pf"))
            .map(|_| ast::UnicodeCategoryName::FinalPunctuation),
        or(expect_str("Other_Punctuation"), || expect_str("Po"))
            .map(|_| ast::UnicodeCategoryName::OtherPunctuation),
        or(expect_str("Punctuation"), || expect_str("P"))
            .map(|_| ast::UnicodeCategoryName::Punctuation),
        or(expect_str("Control"), || expect_str("Cc")).map(|_| ast::UnicodeCategoryName::Control),
        or(expect_str("Format"), || expect_str("Cf")).map(|_| ast::UnicodeCategoryName::Format),
        or(expect_str("Private_Use"), || expect_str("Co"))
            .map(|_| ast::UnicodeCategoryName::PrivateUse),
        or(expect_str("Surrogate"), || expect_str("Cs"))
            .map(|_| ast::UnicodeCategoryName::Surrogate),
        or(expect_str("Unassigned"), || expect_str("Cn"))
            .map(|_| ast::UnicodeCategoryName::Unassigned),
        or(expect_str("Other"), || expect_str("C")).map(|_| ast::UnicodeCategoryName::Other),
    ])
}

fn character_range<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::CharacterRange> {
    parcel::join(
        char(),
        parcel::right(parcel::join(expect_character('-'), char())),
    )
    .map(|(lower_bound, upper_bound)| ast::CharacterRange::new(lower_bound, upper_bound))
}

// Quantifiers

/// Represents all variants of regex quantifiers with an optionally lazy modifier.
fn quantifier<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::Quantifier> {
    parcel::join(quantifier_type(), parcel::optional(lazy_modifier())).map(
        |(quantifier_ty, lazy_modifier)| match lazy_modifier {
            Some(_) => ast::Quantifier::Lazy(quantifier_ty),
            None => ast::Quantifier::Eager(quantifier_ty),
        },
    )
}

fn lazy_modifier<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::LazyModifier> {
    expect_character('?').map(|_| ast::LazyModifier)
}

fn quantifier_type<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::QuantifierType> {
    parcel::or(zero_or_more_quantifier().map(Into::into), || {
        parcel::or(one_or_more_quantifier().map(Into::into), || {
            parcel::or(zero_or_one_quantifier().map(Into::into), || {
                range_quantifier().map(Into::into)
            })
        })
    })
}

fn range_quantifier<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::RangeQuantifier> {
    parcel::left(parcel::join(
        parcel::right(parcel::join(
            expect_character('{'),
            parcel::join(
                range_quantifier_lower_bound(),
                parcel::optional(parcel::right(parcel::join(
                    expect_character(','),
                    parcel::optional(range_quantifier_upper_bound()),
                ))),
            ),
        )),
        expect_character('}'),
    ))
    .map(|(lower_bound, upper_bound)| ast::RangeQuantifier::new(lower_bound, upper_bound))
}

fn range_quantifier_lower_bound<'a>(
) -> impl Parser<'a, &'a [(usize, char)], ast::RangeQuantifierLowerBound> {
    integer().map(ast::RangeQuantifierLowerBound)
}

fn range_quantifier_upper_bound<'a>(
) -> impl Parser<'a, &'a [(usize, char)], ast::RangeQuantifierUpperBound> {
    integer().map(ast::RangeQuantifierUpperBound)
}

fn zero_or_more_quantifier<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::QuantifierType> {
    expect_character('*').map(|_| ast::QuantifierType::ZeroOrMore)
}

fn one_or_more_quantifier<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::QuantifierType> {
    expect_character('+').map(|_| ast::QuantifierType::OneOrMore)
}

fn zero_or_one_quantifier<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::QuantifierType> {
    expect_character('?').map(|_| ast::QuantifierType::ZeroOrOne)
}

// Backreferences

fn backreference<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::Backreference> {
    parcel::right(parcel::join(expect_character('\\'), integer())).map(ast::Backreference)
}

// Anchors

fn start_of_string_anchor<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::StartOfStringAnchor> {
    expect_character('^').map(|_| ast::StartOfStringAnchor)
}

fn anchor<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], ast::Anchor> {
    parcel::or(anchor_word_boundary().map(Into::into), || {
        parcel::or(anchor_nonword_boundary().map(Into::into), || {
            parcel::or(anchor_start_of_string_only().map(Into::into), || {
                parcel::or(
                    anchor_end_of_string_only_not_newline().map(Into::into),
                    || {
                        parcel::or(anchor_end_of_string_only().map(Into::into), || {
                            parcel::or(anchor_previous_match_end().map(Into::into), || {
                                anchor_end_of_string().map(Into::into)
                            })
                        })
                    },
                )
            })
        })
    })
}

fn anchor_word_boundary<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::AnchorWordBoundary> {
    parcel::join(expect_character('\\'), expect_character('b')).map(|_| ast::AnchorWordBoundary)
}

fn anchor_nonword_boundary<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::AnchorNonWordBoundary>
{
    parcel::join(expect_character('\\'), expect_character('B')).map(|_| ast::AnchorNonWordBoundary)
}

fn anchor_start_of_string_only<'a>(
) -> impl Parser<'a, &'a [(usize, char)], ast::AnchorStartOfStringOnly> {
    parcel::join(expect_character('\\'), expect_character('A'))
        .map(|_| ast::AnchorStartOfStringOnly)
}

fn anchor_end_of_string_only_not_newline<'a>(
) -> impl Parser<'a, &'a [(usize, char)], ast::AnchorEndOfStringOnlyNotNewline> {
    parcel::join(expect_character('\\'), expect_character('z'))
        .map(|_| ast::AnchorEndOfStringOnlyNotNewline)
}

fn anchor_end_of_string_only<'a>(
) -> impl Parser<'a, &'a [(usize, char)], ast::AnchorEndOfStringOnly> {
    parcel::join(expect_character('\\'), expect_character('Z')).map(|_| ast::AnchorEndOfStringOnly)
}

fn anchor_previous_match_end<'a>(
) -> impl Parser<'a, &'a [(usize, char)], ast::AnchorPreviousMatchEnd> {
    parcel::join(expect_character('\\'), expect_character('G')).map(|_| ast::AnchorPreviousMatchEnd)
}

fn anchor_end_of_string<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::AnchorEndOfString> {
    expect_character('$').map(|_| ast::AnchorEndOfString)
}

// Terminals

fn integer<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::Integer> {
    move |input: &'a [(usize, char)]| {
        let preparsed_input = input;
        let res = parcel::one_or_more(digit(10))
            .map(|digits| {
                let vd: String = digits.into_iter().collect();

                vd.parse::<usize>()
            })
            .parse(input);

        match res {
            Ok(MatchStatus::Match {
                span,
                remainder,
                inner: Ok(int),
            }) => Ok(MatchStatus::Match {
                span,
                remainder,
                inner: ast::Integer(int),
            }),

            Ok(MatchStatus::Match {
                span: _,
                remainder: _,
                inner: Err(_),
            }) => Ok(MatchStatus::NoMatch(preparsed_input)),

            Ok(MatchStatus::NoMatch(remainder)) => Ok(MatchStatus::NoMatch(remainder)),
            Err(e) => Err(e),
        }
    }
}

fn char<'a>() -> impl Parser<'a, &'a [(usize, char)], ast::Char> {
    any_character_escaped().map(ast::Char)
}

fn any_character_escaped<'a>() -> impl Parser<'a, &'a [(usize, char)], char> {
    move |input: &'a [(usize, char)]| match input.get(0..2) {
        Some(&[(escape_pos, '\\'), (to_escape_pos, to_escape)]) => {
            match char_to_escaped_equivalent(to_escape) {
                Some(escaped_char) => Ok(MatchStatus::Match {
                    span: escape_pos..to_escape_pos + 1,
                    remainder: &input[2..],
                    inner: escaped_char,
                }),
                None => Ok(MatchStatus::NoMatch(input)),
            }
        }
        // discard the lookahead.
        Some(&[(next_pos, next), _]) => Ok(MatchStatus::Match {
            span: next_pos..next_pos + 1,
            remainder: &input[1..],
            inner: next,
        }),
        // handle for end of input
        None => match input.get(0..1) {
            // discard the lookahead.
            Some(&[(next_pos, next)]) => Ok(MatchStatus::Match {
                span: next_pos..next_pos + 1,
                remainder: &input[1..],
                inner: next,
            }),
            _ => Ok(MatchStatus::NoMatch(input)),
        },
        _ => Ok(MatchStatus::NoMatch(input)),
    }
}

fn char_to_escaped_equivalent(c: char) -> Option<char> {
    match c {
        'n' => Some('\n'),
        't' => Some('\t'),
        'r' => Some('\r'),
        '\'' => Some('\''),
        '\"' => Some('\"'),
        '\\' => Some('\\'),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_parse_minimal_expression_with_no_errors() {
        let inputs = vec![
            // a basic input string
            "the red pill",
            // A recursive grouping
            "the ((red|blue) pill)",
        ]
        .into_iter();

        for input in inputs {
            let parse_result = parse(input);
            assert!(parse_result.is_ok())
        }
    }

    #[test]
    fn should_parse_compound_match() {
        use ast::*;

        assert_eq!(
            Ok(Regex::Unanchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Match(Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacter(MatchCharacter(Char('a')))
                }),
                SubExpressionItem::Match(Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacter(MatchCharacter(Char('b')))
                }),
            ])]))),
            parse("ab")
        )
    }

    #[test]
    fn should_parse_anchored_match() {
        use ast::*;

        assert_eq!(
            Ok(Regex::StartOfStringAnchored(Expression(vec![
                SubExpression(vec![
                    SubExpressionItem::Match(Match::WithoutQuantifier {
                        item: MatchItem::MatchCharacter(MatchCharacter(Char('a')))
                    }),
                    SubExpressionItem::Match(Match::WithoutQuantifier {
                        item: MatchItem::MatchCharacter(MatchCharacter(Char('b')))
                    }),
                ])
            ]))),
            parse("^ab")
        )
    }

    #[test]
    fn should_parse_alternation() {
        use ast::*;

        assert_eq!(
            Ok(Regex::StartOfStringAnchored(Expression(vec![
                SubExpression(vec![SubExpressionItem::Match(Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacter(MatchCharacter(Char('a')))
                }),]),
                SubExpression(vec![SubExpressionItem::Match(Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacter(MatchCharacter(Char('b')))
                }),])
            ]))),
            parse("^a|b")
        )
    }

    #[test]
    fn should_parse_eager_repetition_quantifiers() {
        use ast::*;

        let inputs = vec![
            (QuantifierType::OneOrMore, "^.+"),
            (QuantifierType::ZeroOrMore, "^.*"),
            (QuantifierType::MatchExactRange(Integer(2)), "^.{2}"),
            (QuantifierType::MatchAtLeastRange(Integer(2)), "^.{2,}"),
            (
                QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                },
                "^.{2,4}",
            ),
        ];

        for (expected_quantifier, input) in inputs {
            assert_eq!(
                Ok(Regex::StartOfStringAnchored(Expression(vec![
                    SubExpression(vec![SubExpressionItem::Match(Match::WithQuantifier {
                        item: MatchItem::MatchAnyCharacter,
                        quantifier: Quantifier::Eager(expected_quantifier),
                    })])
                ]))),
                parse(input)
            )
        }

        // test single character matches
        let inputs = vec![(QuantifierType::MatchExactRange(Integer(2)), "^a{2}")];

        for (expected_quantifier, input) in inputs {
            assert_eq!(
                Ok(Regex::StartOfStringAnchored(Expression(vec![
                    SubExpression(vec![SubExpressionItem::Match(Match::WithQuantifier {
                        item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                        quantifier: Quantifier::Eager(expected_quantifier),
                    })])
                ]))),
                parse(input)
            )
        }
    }

    #[test]
    fn should_parse_lazy_repetition_quantifiers() {
        use ast::*;

        let inputs = vec![
            (QuantifierType::OneOrMore, "^.+?"),
            (QuantifierType::ZeroOrMore, "^.*?"),
            (QuantifierType::MatchExactRange(Integer(2)), "^.{2}?"),
            (QuantifierType::MatchAtLeastRange(Integer(2)), "^.{2,}?"),
            (
                QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                },
                "^.{2,4}?",
            ),
        ];

        for (expected_quantifier, input) in inputs {
            assert_eq!(
                Ok(Regex::StartOfStringAnchored(Expression(vec![
                    SubExpression(vec![SubExpressionItem::Match(Match::WithQuantifier {
                        item: MatchItem::MatchAnyCharacter,
                        quantifier: Quantifier::Lazy(expected_quantifier),
                    })])
                ]))),
                parse(input)
            )
        }

        // test single character matches
        let inputs = vec![(QuantifierType::MatchExactRange(Integer(2)), "^a{2}?")];

        for (expected_quantifier, input) in inputs {
            assert_eq!(
                Ok(Regex::StartOfStringAnchored(Expression(vec![
                    SubExpression(vec![SubExpressionItem::Match(Match::WithQuantifier {
                        item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                        quantifier: Quantifier::Lazy(expected_quantifier),
                    })])
                ]))),
                parse(input)
            )
        }
    }

    #[test]
    fn should_parse_character_class_items() {
        use ast::*;
        let input_output = vec![
            (
                "^\\w",
                Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyWord,
                    )),
                },
            ),
            (
                "^\\w?",
                Match::WithQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyWord,
                    )),
                    quantifier: Quantifier::Eager(QuantifierType::ZeroOrOne),
                },
            ),
            (
                "^\\w*",
                Match::WithQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyWord,
                    )),
                    quantifier: Quantifier::Eager(QuantifierType::ZeroOrMore),
                },
            ),
            (
                "^\\w+",
                Match::WithQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyWord,
                    )),
                    quantifier: Quantifier::Eager(QuantifierType::OneOrMore),
                },
            ),
            (
                "^\\W",
                Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyWordInverted,
                    )),
                },
            ),
            (
                "^\\d",
                Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyDecimalDigit,
                    )),
                },
            ),
            (
                "^\\D",
                Match::WithoutQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyDecimalDigitInverted,
                    )),
                },
            ),
        ];

        for (test_id, (input, class)) in input_output.into_iter().enumerate() {
            let res = parse(input);
            assert_eq!(
                (
                    test_id,
                    Ok(Regex::StartOfStringAnchored(Expression(vec![
                        SubExpression(vec![SubExpressionItem::Match(class)])
                    ])))
                ),
                (test_id, res)
            )
        }
    }

    #[test]
    fn should_parse_character_group_items() {
        use ast::*;

        let input_output = vec![
            (
                "^[a]",
                CharacterGroup::Items(vec![CharacterGroupItem::Char(Char('a'))]),
            ),
            (
                "^[ab]",
                CharacterGroup::Items(vec![
                    CharacterGroupItem::Char(Char('a')),
                    CharacterGroupItem::Char(Char('b')),
                ]),
            ),
            (
                "^[a-z]",
                CharacterGroup::Items(vec![CharacterGroupItem::CharacterRange(
                    Char('a'),
                    Char('z'),
                )]),
            ),
            (
                "^[abc0-9]",
                CharacterGroup::Items(vec![
                    CharacterGroupItem::Char(Char('a')),
                    CharacterGroupItem::Char(Char('b')),
                    CharacterGroupItem::Char(Char('c')),
                    CharacterGroupItem::CharacterRange(Char('0'), Char('9')),
                ]),
            ),
        ];

        for (test_id, (input, output)) in input_output.into_iter().enumerate() {
            let res = parse(input);
            assert_eq!(
                (
                    test_id,
                    Ok(Regex::StartOfStringAnchored(Expression(vec![
                        SubExpression(vec![SubExpressionItem::Match(Match::WithoutQuantifier {
                            item: MatchItem::MatchCharacterClass(
                                MatchCharacterClass::CharacterGroup(output)
                            )
                        }),])
                    ])))
                ),
                (test_id, res)
            )
        }

        // with quantifiers
        let input_output = vec![
            ("^[ab]?", QuantifierType::ZeroOrOne),
            ("^[ab]*", QuantifierType::ZeroOrMore),
            ("^[ab]+", QuantifierType::OneOrMore),
            ("^[ab]{1}", QuantifierType::MatchExactRange(Integer(1))),
            ("^[ab]{1,}", QuantifierType::MatchAtLeastRange(Integer(1))),
            (
                "^[ab]{1,2}",
                QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(1),
                    upper_bound: Integer(2),
                },
            ),
        ];

        for (test_id, (input, quantifier_ty)) in input_output.into_iter().enumerate() {
            let res = parse(input);
            assert_eq!(
                (
                    test_id,
                    Ok(Regex::StartOfStringAnchored(Expression(vec![
                        SubExpression(vec![SubExpressionItem::Match(Match::WithQuantifier {
                            item: MatchItem::MatchCharacterClass(
                                MatchCharacterClass::CharacterGroup(CharacterGroup::Items(vec![
                                    CharacterGroupItem::Char(Char('a')),
                                    CharacterGroupItem::Char(Char('b')),
                                ]))
                            ),
                            quantifier: Quantifier::Eager(quantifier_ty)
                        }),])
                    ])))
                ),
                (test_id, res)
            )
        }
    }

    #[test]
    fn should_parse_group_items() {
        use ast::*;

        let input_output = vec![
            (
                "^(a)",
                Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                    SubExpressionItem::Group(Group::Capturing {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            }),
                        ])]),
                    }),
                ])])),
            ),
            (
                "^(a)?",
                Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                    SubExpressionItem::Group(Group::CapturingWithQuantifier {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            }),
                        ])]),
                        quantifier: Quantifier::Eager(QuantifierType::ZeroOrOne),
                    }),
                ])])),
            ),
            (
                "^(a)(b)",
                Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                    SubExpressionItem::Group(Group::Capturing {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            }),
                        ])]),
                    }),
                    SubExpressionItem::Group(Group::Capturing {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
                            }),
                        ])]),
                    }),
                ])])),
            ),
            (
                "^(a(b))",
                Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                    SubExpressionItem::Group(Group::Capturing {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            }),
                            SubExpressionItem::Group(Group::Capturing {
                                expression: Expression(vec![SubExpression(vec![
                                    SubExpressionItem::Match(Match::WithoutQuantifier {
                                        item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
                                    }),
                                ])]),
                            }),
                        ])]),
                    }),
                ])])),
            ),
            (
                "^(?:a)",
                Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                    SubExpressionItem::Group(Group::NonCapturing {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            }),
                        ])]),
                    }),
                ])])),
            ),
            (
                "^(?:a)?",
                Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                    SubExpressionItem::Group(Group::NonCapturingWithQuantifier {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            }),
                        ])]),
                        quantifier: Quantifier::Eager(QuantifierType::ZeroOrOne),
                    }),
                ])])),
            ),
        ];

        for (test_id, (input, expected_regex_ast)) in input_output.into_iter().enumerate() {
            let res = parse(input);
            assert_eq!((test_id, Ok(expected_regex_ast)), (test_id, res))
        }

        // with quantifiers
        let input_output = vec![
            ("^[ab]?", QuantifierType::ZeroOrOne),
            ("^[ab]*", QuantifierType::ZeroOrMore),
            ("^[ab]+", QuantifierType::OneOrMore),
            ("^[ab]{1}", QuantifierType::MatchExactRange(Integer(1))),
            ("^[ab]{1,}", QuantifierType::MatchAtLeastRange(Integer(1))),
            (
                "^[ab]{1,2}",
                QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(1),
                    upper_bound: Integer(2),
                },
            ),
        ];

        for (test_id, (input, quantifier_ty)) in input_output.into_iter().enumerate() {
            let res = parse(input);
            assert_eq!(
                (
                    test_id,
                    Ok(Regex::StartOfStringAnchored(Expression(vec![
                        SubExpression(vec![SubExpressionItem::Match(Match::WithQuantifier {
                            item: MatchItem::MatchCharacterClass(
                                MatchCharacterClass::CharacterGroup(CharacterGroup::Items(vec![
                                    CharacterGroupItem::Char(Char('a')),
                                    CharacterGroupItem::Char(Char('b')),
                                ]))
                            ),
                            quantifier: Quantifier::Eager(quantifier_ty)
                        }),])
                    ])))
                ),
                (test_id, res)
            )
        }
    }

    #[test]
    fn should_parse_unicode_category() {
        use ast::*;

        let input_to_category_pairing = vec![
            ("\\p{Letter}", UnicodeCategoryName::Letter),
            ("\\p{L}", UnicodeCategoryName::Letter),
            ("\\p{Non_Spacing_Mark}", UnicodeCategoryName::NonSpacingMark),
            ("\\p{Mn}", UnicodeCategoryName::NonSpacingMark),
        ];

        for (test_id, (input, expected_unicode_category)) in
            input_to_category_pairing.into_iter().enumerate()
        {
            assert_eq!(
                (
                    test_id,
                    Ok(Regex::Unanchored(Expression(vec![SubExpression(vec![
                        SubExpressionItem::Match(Match::WithoutQuantifier {
                            item: MatchItem::MatchCharacterClass(
                                MatchCharacterClass::CharacterClassFromUnicodeCategory(
                                    CharacterClassFromUnicodeCategory(expected_unicode_category)
                                )
                            )
                        }),
                    ])])))
                ),
                (test_id, parse(input))
            )
        }
    }

    #[test]
    fn should_parse_any_match() {
        use ast::*;

        assert_eq!(
            Ok(Regex::Unanchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Match(Match::WithoutQuantifier {
                    item: MatchItem::MatchAnyCharacter
                }),
            ])]))),
            parse(".")
        )
    }

    #[test]
    fn should_parse_capture_start_of_string_anchor() {
        use ast::*;

        assert_eq!(
            Ok(Regex::Unanchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::Capturing {
                    expression: Expression(vec![
                        SubExpression(vec![SubExpressionItem::Group(Group::NonCapturing {
                            expression: Expression(vec![SubExpression(vec![
                                SubExpressionItem::Anchor(Anchor::StartOfStringOnly),
                                SubExpressionItem::Match(Match::WithoutQuantifier {
                                    item: MatchItem::MatchCharacter(MatchCharacter(Char('a')))
                                })
                            ])]),
                        }),]),
                        SubExpression(vec![SubExpressionItem::Group(Group::NonCapturing {
                            expression: Expression(vec![SubExpression(vec![
                                SubExpressionItem::Match(Match::WithoutQuantifier {
                                    item: MatchItem::MatchCharacter(MatchCharacter(Char('b')))
                                })
                            ])]),
                        })])
                    ])
                }),
            ])]))),
            parse("((?:\\Aa)|(?:b))")
        )
    }
}
