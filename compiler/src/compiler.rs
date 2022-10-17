//! Provides methods and types to facilitate the compilation of a parsed regex
//! ast into runtime bytecode.
//!
//! # Example
//!
//! ```
//! use regex_compiler::ast::*;
//! use regex_runtime::*;
//! use regex_compiler::compile;
//!
//! // approximate to `ab`
//! let regex_ast = Regex::Unanchored(Expression(vec![SubExpression(vec![
//!     SubExpressionItem::Match(Match::WithoutQuantifier {
//!         item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
//!      }),
//!      SubExpressionItem::Match(Match::WithoutQuantifier {
//!          item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
//!     }),
//! ])]));
//!
//! assert_eq!(
//!     Ok(Instructions::default()
//!         .with_opcodes(vec![
//!             Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
//!             Opcode::Any,
//!             Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
//!             Opcode::Consume(InstConsume::new('a')),
//!             Opcode::Consume(InstConsume::new('b')),
//!             Opcode::Match,
//!         ]).with_fast_forward(FastForward::Char('a'))),
//!     compile(regex_ast)
//! )
//! ```
use super::ast;
use super::opcode::VM;
use regex_runtime::*;

/// Defines a trait for implementing compilation from a regex cast to a
/// lowered output type.
pub trait Lowerable<INPUT, OUTPUT> {
    type Error;

    fn lower(&mut self, input: INPUT) -> Result<OUTPUT, Self::Error>;
}

/// Accepts a parsed AST and attempts to compile it into a runnable bytecode
/// program for use with the regex-runtime crate.
///
/// # Example
///
/// ```
/// use regex_compiler::ast::*;
/// use regex_runtime::*;
/// use regex_compiler::compile;
///
/// // approximate to `ab`
/// let regex_ast = Regex::Unanchored(Expression(vec![SubExpression(vec![
///     SubExpressionItem::Match(Match::WithoutQuantifier {
///         item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
///      }),
///      SubExpressionItem::Match(Match::WithoutQuantifier {
///          item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
///     }),
/// ])]));
///
/// assert_eq!(
///     Ok(Instructions::default()
///         .with_opcodes(vec![
///             Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
///             Opcode::Any,
///             Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
///             Opcode::Consume(InstConsume::new('a')),
///             Opcode::Consume(InstConsume::new('b')),
///             Opcode::Match,
///         ]).with_fast_forward(FastForward::Char('a'))),
///     compile(regex_ast)
/// )
/// ```
pub fn compile(ast: ast::Regex) -> Result<Instructions, String> {
    VM.lower(ast)
}

/// Accepts multiple parsed ASTs and attempts to compile it into a runnable
/// bytecode program for use with the regex-runtime crate.
///
/// This is similar to the `compile` method, with the differencee being the
/// acceptance of multiple expressions.
///
/// # Example
///
/// ```
/// use regex_compiler::ast::*;
/// use regex_runtime::*;
/// use regex_compiler::compile_many;
///
/// // approximate to `ab`
/// // approximate to `^(a)`
/// let regex_ast_anchored =
///     Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
///         SubExpressionItem::Group(Group::Capturing {
///             expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
///                 Match::WithoutQuantifier {
///                     item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
///                 },
///             )])]),
///         }),
///     ])]));
///
/// let regex_ast_unanchored = ['b', 'c'].into_iter().map(|c| {
///     Regex::Unanchored(Expression(vec![SubExpression(vec![
///         SubExpressionItem::Group(Group::Capturing {
///             expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
///                 Match::WithoutQuantifier {
///                     item: MatchItem::MatchCharacter(MatchCharacter(Char(c))),
///                 },
///             )])]),
///         }),
///     ])]))
/// });
///
/// let all_exprs = [regex_ast_anchored]
///     .into_iter()
///     .chain(regex_ast_unanchored)
///     .collect();
///
/// let expected = vec![
///     Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(6))),
///     Opcode::Meta(InstMeta(MetaKind::SetExpressionId(0))),
///     // first anchored expr
///     Opcode::StartSave(InstStartSave::new(0)),
///     Opcode::Consume(InstConsume::new('a')),
///     Opcode::EndSave(InstEndSave::new(0)),
///     Opcode::Match,
///     // unanchored start
///     Opcode::Split(InstSplit::new(InstIndex::from(9), InstIndex::from(7))),
///     Opcode::Any,
///     Opcode::Jmp(InstJmp::new(InstIndex::from(6))),
///     Opcode::Split(InstSplit::new(InstIndex::from(10), InstIndex::from(15))),
///     // first unanchored expr
///     Opcode::Meta(InstMeta(MetaKind::SetExpressionId(1))),
///     Opcode::StartSave(InstStartSave::new(0)),
///     Opcode::Consume(InstConsume::new('b')),
///     Opcode::EndSave(InstEndSave::new(0)),
///     Opcode::Match,
///     // second unanchored expr
///     Opcode::Meta(InstMeta(MetaKind::SetExpressionId(2))),
///     Opcode::StartSave(InstStartSave::new(0)),
///     Opcode::Consume(InstConsume::new('c')),
///     Opcode::EndSave(InstEndSave::new(0)),
///     Opcode::Match,
/// ];
///
/// assert_eq!(
///    Ok(Instructions::default().with_opcodes(expected)),
///    compile_many(all_exprs),
/// );
/// ```
pub fn compile_many(asts: Vec<ast::Regex>) -> Result<Instructions, String> {
    VM.lower(asts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::*;

    #[test]
    fn should_compile_unanchored_character_match() {
        // approximate to `ab`
        let regex_ast = Regex::Unanchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
            }),
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_opcodes(vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::Any,
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('b')),
                    Opcode::Match,
                ])
                .with_fast_forward(FastForward::Char('a'))),
            compile(regex_ast)
        )
    }

    #[test]
    fn should_compile_through_trait() {
        // approximate to `ab`
        let regex_ast = Regex::Unanchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
            }),
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_opcodes(vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::Any,
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('b')),
                    Opcode::Match,
                ])
                .with_fast_forward(FastForward::Char('a'))),
            VM.lower(regex_ast)
        )
    }

    #[test]
    fn should_compile_anchored_character_match() {
        // approximate to `^ab`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
            }),
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::Match,
            ])),
            compile(regex_ast)
        )
    }

    #[test]
    fn should_compile_alternation() {
        // approximate to `^a|b`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![
            SubExpression(vec![SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
            })]),
            SubExpression(vec![SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
            })]),
        ]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(3))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Jmp(InstJmp::new(InstIndex::from(4))),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::Match,
            ])),
            compile(regex_ast)
        )
    }

    #[test]
    fn should_compile_any_character_match() {
        // approximate to `.`
        let regex_ast = Regex::Unanchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchAnyCharacter,
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::Any,
                Opcode::Match,
            ])),
            compile(regex_ast)
        )
    }

    #[test]
    fn should_compile_zero_or_more_quantified_item() {
        // approximate to `^.*`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchAnyCharacter,
                quantifier: Quantifier::Eager(QuantifierType::ZeroOrMore),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(3))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::Match,
            ])),
            compile(regex_ast)
        );

        // approximate to `^a*`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                quantifier: Quantifier::Eager(QuantifierType::ZeroOrMore),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(3))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::Match,
            ])),
            compile(regex_ast)
        )
    }

    #[test]
    fn should_compile_one_or_more_quantified_item() {
        // approximate to `^.+`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchAnyCharacter,
                quantifier: Quantifier::Eager(QuantifierType::OneOrMore),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Any,
                Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(4))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                Opcode::Match,
            ])),
            compile(regex_ast)
        );

        // approximate to `^a+`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                quantifier: Quantifier::Eager(QuantifierType::OneOrMore),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(4))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                Opcode::Match,
            ])),
            compile(regex_ast)
        )
    }

    #[test]
    fn should_compile_match_zero_or_one_item() {
        // approximate to `^.?`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchAnyCharacter,
                quantifier: Quantifier::Eager(QuantifierType::ZeroOrOne),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(2))),
                Opcode::Any,
                Opcode::Match,
            ])),
            compile(regex_ast)
        );

        // approximate to `^a?`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                quantifier: Quantifier::Eager(QuantifierType::ZeroOrOne),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(2))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Match
            ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_exact_match_quantified_item() {
        // approximate to `^.{2}`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchAnyCharacter,
                quantifier: Quantifier::Eager(QuantifierType::MatchExactRange(Integer(2))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_opcodes(vec![Opcode::Any, Opcode::Any, Opcode::Match,])),
            compile(regex_ast)
        );

        // approximate to `^a{2}`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                quantifier: Quantifier::Eager(QuantifierType::MatchExactRange(Integer(2))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Match,
            ])),
            compile(regex_ast)
        )
    }

    #[test]
    fn should_compile_match_at_least_quantified_item() {
        // approximate to `^.{2,}`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchAnyCharacter,
                quantifier: Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(2))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Any,
                Opcode::Any,
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(5))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                Opcode::Match
            ])),
            compile(regex_ast)
        );

        // approximate to `^a{2,}`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                quantifier: Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(2))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(5))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                Opcode::Match
            ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_match_between_quantified_item() {
        // approximate to `^.{2,4}`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchAnyCharacter,
                quantifier: Quantifier::Eager(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                }),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Any,
                Opcode::Any,
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(4))),
                Opcode::Any,
                Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(6))),
                Opcode::Any,
                Opcode::Match
            ])),
            compile(regex_ast)
        );

        // approximate to `^a{2,4}`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                quantifier: Quantifier::Eager(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                }),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(4))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(6))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Match
            ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_character_classes() {
        // approximate to `^\w`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                    CharacterClass::AnyWord,
                )),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Ranges(
                    vec!['a'..='z', 'A'..='Z', '0'..='9', '_'..='_',]
                ))])
                .with_opcodes(vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ])),
            compile(regex_ast)
        );

        // approximate to `^\d`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                    CharacterClass::AnyDecimalDigit,
                )),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Range(
                    '0'..='9'
                ))])
                .with_opcodes(vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_character_classes_with_quantifiers() {
        let quantifier_and_expected_opcodes = vec![
            // approximate to `^\d?`
            (
                Quantifier::Eager(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(2))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^\d??`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(1))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^\d*`
            (
                Quantifier::Eager(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(3))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^\d*?`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^\d+`
            (
                Quantifier::Eager(QuantifierType::OneOrMore),
                vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(4))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^\d+?`
            (
                Quantifier::Lazy(QuantifierType::OneOrMore),
                vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(2))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::Match,
                ],
            ),
        ];

        for (id, (quantifier, expected_opcodes)) in
            quantifier_and_expected_opcodes.into_iter().enumerate()
        {
            let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Match(Match::WithQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(
                        CharacterClass::AnyDecimalDigit,
                    )),
                    quantifier,
                }),
            ])]));

            let res = compile(regex_ast);
            assert_eq!(
                (
                    id,
                    Ok(Instructions::default()
                        .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Range(
                            '0'..='9'
                        ))])
                        .with_opcodes(expected_opcodes))
                ),
                (id, res)
            );
        }
    }

    #[test]
    fn should_compile_single_character_character_group() {
        // approximate to `^[a]`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterGroup(
                    CharacterGroup::Items(vec![CharacterGroupItem::Char(Char('a'))]),
                )),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Explicit(
                    vec!['a']
                ))])
                .with_opcodes(vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match
                ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_compound_character_group() {
        // approximate to `[abz]`
        let regex_ast = Regex::Unanchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterGroup(
                    CharacterGroup::Items(vec![
                        CharacterGroupItem::Char(Char('a')),
                        CharacterGroupItem::Char(Char('b')),
                        CharacterGroupItem::Char(Char('z')),
                    ]),
                )),
            }),
        ])]));

        let expected = Ok(Instructions::default()
            .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Explicit(
                vec!['a', 'b', 'z'],
            ))])
            .with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                Opcode::Match,
            ])
            .with_fast_forward(FastForward::Set(0)));

        assert_eq!(expected, compile(regex_ast));
    }

    #[test]
    fn should_dedup_explicit_character_groups() {
        // approximate to `^[aabbzzzz]`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterGroup(
                    CharacterGroup::Items(vec![
                        CharacterGroupItem::Char(Char('a')),
                        CharacterGroupItem::Char(Char('a')),
                        CharacterGroupItem::Char(Char('b')),
                        CharacterGroupItem::Char(Char('b')),
                        CharacterGroupItem::Char(Char('z')),
                        CharacterGroupItem::Char(Char('z')),
                        CharacterGroupItem::Char(Char('z')),
                        CharacterGroupItem::Char(Char('z')),
                    ]),
                )),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Explicit(
                    vec!['a', 'b', 'z']
                )),])
                .with_opcodes(vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_character_group_range() {
        // approximate to `^[0-9]`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterGroup(
                    CharacterGroup::Items(vec![CharacterGroupItem::CharacterRange(
                        Char('0'),
                        Char('9'),
                    )]),
                )),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Range(
                    '0'..='9'
                )),])
                .with_opcodes(vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_character_groups_with_quantifiers() {
        let quantifier_and_expected_opcodes = vec![
            // approximate to `^[0-9]?`
            (
                Quantifier::Eager(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(2))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^[0-9]??`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(1))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^[0-9]*`
            (
                Quantifier::Eager(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(3))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^[0-9]*?`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^[0-9]+`
            (
                Quantifier::Eager(QuantifierType::OneOrMore),
                vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(4))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^[0-9]+?`
            (
                Quantifier::Lazy(QuantifierType::OneOrMore),
                vec![
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(2))),
                    Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::Match,
                ],
            ),
        ];

        for (id, (quantifier, expected_opcodes)) in
            quantifier_and_expected_opcodes.into_iter().enumerate()
        {
            let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Match(Match::WithQuantifier {
                    item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterGroup(
                        CharacterGroup::Items(vec![CharacterGroupItem::CharacterRange(
                            Char('0'),
                            Char('9'),
                        )]),
                    )),
                    quantifier,
                }),
            ])]));

            let res = compile(regex_ast);
            assert_eq!(
                (
                    id,
                    Ok(Instructions::default()
                        .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Range(
                            '0'..='9'
                        ))])
                        .with_opcodes(expected_opcodes))
                ),
                (id, res)
            );
        }
    }

    #[test]
    fn should_compile_capturing_group() {
        // approximate to `^(a)(b)`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Group(Group::Capturing {
                expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                    Match::WithoutQuantifier {
                        item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                    },
                )])]),
            }),
            SubExpressionItem::Group(Group::Capturing {
                expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                    Match::WithoutQuantifier {
                        item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
                    },
                )])]),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::StartSave(InstStartSave::new(1)),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::EndSave(InstEndSave::new(1)),
                Opcode::Match
            ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_nested_capturing_group() {
        // approximate to `^(a(b))`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
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
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::StartSave(InstStartSave::new(1)),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::EndSave(InstEndSave::new(1)),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match
            ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_non_capturing_group() {
        // approximate to `^(?:a)`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Group(Group::NonCapturing {
                expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                    Match::WithoutQuantifier {
                        item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                    },
                )])]),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default()
                .with_opcodes(vec![Opcode::Consume(InstConsume::new('a')), Opcode::Match])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_quantified_capturing_group() {
        let quantifier_and_expected_opcodes = vec![
            // approximate to `^(a)?`
            (
                Quantifier::Eager(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(3))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a)??`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(2))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a)*`
            (
                Quantifier::Eager(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(4))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a)*?`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(2))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a)+`
            (
                Quantifier::Eager(QuantifierType::OneOrMore),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(5))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a)+?`
            (
                Quantifier::Lazy(QuantifierType::OneOrMore),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(3))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a){2}`
            (
                Quantifier::Eager(QuantifierType::MatchExactRange(Integer(2))),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a){2}?`
            (
                Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(2))),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a){2,}`
            (
                Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(2))),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(6))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(3))),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a){2,}?`
            (
                Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(2))),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(6), InstIndex::from(4))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(3))),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a){2,4}`
            (
                Quantifier::Eager(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                }),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(5))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(6), InstIndex::from(7))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(a){2,4}?`
            (
                Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                }),
                vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(4))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(7), InstIndex::from(6))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ],
            ),
        ];

        for (id, (quantifier, expected_opcodes)) in
            quantifier_and_expected_opcodes.into_iter().enumerate()
        {
            let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::Capturing {
                    expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                        Match::WithQuantifier {
                            item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            quantifier,
                        },
                    )])]),
                }),
            ])]));

            let res = compile(regex_ast);
            assert_eq!(
                (
                    id,
                    Ok(Instructions::default().with_opcodes(expected_opcodes))
                ),
                (id, res)
            );
        }
    }

    #[test]
    fn should_compile_quantified_non_capturing_group() {
        let quantifier_and_expected_opcodes = vec![
            // approximate to `^(?:a)?`
            (
                Quantifier::Eager(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(2))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a)??`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrOne),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(1))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a)*`
            (
                Quantifier::Eager(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(3))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a)*?`
            (
                Quantifier::Lazy(QuantifierType::ZeroOrMore),
                vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a)+`
            (
                Quantifier::Eager(QuantifierType::OneOrMore),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(4))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a)+?`
            (
                Quantifier::Lazy(QuantifierType::OneOrMore),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(2))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a){2}`
            (
                Quantifier::Eager(QuantifierType::MatchExactRange(Integer(2))),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a){2}?`
            (
                Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(2))),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a){2,}`
            (
                Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(2))),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(5))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a){2,}?`
            (
                Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(2))),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(3))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a){2,4}`
            (
                Quantifier::Eager(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                }),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(4))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(6))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Match,
                ],
            ),
            // approximate to `^(?:a){2,4}?`
            (
                Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(2),
                    upper_bound: Integer(4),
                }),
                vec![
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(3))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Split(InstSplit::new(InstIndex::from(6), InstIndex::from(5))),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Match,
                ],
            ),
        ];

        for (id, (quantifier, expected_opcodes)) in
            quantifier_and_expected_opcodes.into_iter().enumerate()
        {
            let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::NonCapturing {
                    expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                        Match::WithQuantifier {
                            item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            quantifier,
                        },
                    )])]),
                }),
            ])]));

            let res = compile(regex_ast);
            assert_eq!(
                (
                    id,
                    Ok(Instructions::default().with_opcodes(expected_opcodes))
                ),
                (id, res)
            );
        }
    }

    #[test]
    fn should_compile_anchor_or_boundary() {
        // approximate to `^(\ba)`
        let regex_ast = Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Anchor(Anchor::WordBoundary),
            SubExpressionItem::Match(Match::WithoutQuantifier {
                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Epsilon(InstEpsilon::new(EpsilonCond::WordBoundary)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Match
            ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_start_of_string_anchor_pattern() {
        // approximate to `((?:\Aa)|(?:b))`
        let regex_ast = Regex::Unanchored(Expression(vec![SubExpression(vec![
            SubExpressionItem::Group(Group::Capturing {
                expression: Expression(vec![
                    SubExpression(vec![SubExpressionItem::Group(Group::NonCapturing {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Anchor(Anchor::StartOfStringOnly),
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                            }),
                        ])]),
                    })]),
                    SubExpression(vec![SubExpressionItem::Group(Group::NonCapturing {
                        expression: Expression(vec![SubExpression(vec![
                            SubExpressionItem::Match(Match::WithoutQuantifier {
                                item: MatchItem::MatchCharacter(MatchCharacter(Char('b'))),
                            }),
                        ])]),
                    })]),
                ]),
            }),
        ])]));

        assert_eq!(
            Ok(Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(8))),
                Opcode::Epsilon(InstEpsilon::new(EpsilonCond::StartOfStringOnly)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Jmp(InstJmp::new(InstIndex::from(9))),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ])),
            compile(regex_ast)
        );
    }

    #[test]
    fn should_compile_many_should_build_correct_single_program() {
        // approximate to `^(a)`
        let regex_ast_anchored =
            Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::Capturing {
                    expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                        Match::WithoutQuantifier {
                            item: MatchItem::MatchCharacter(MatchCharacter(Char('a'))),
                        },
                    )])]),
                }),
            ])]));

        let regex_ast_unanchored = ['b', 'c'].into_iter().map(|c| {
            Regex::Unanchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::Capturing {
                    expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                        Match::WithoutQuantifier {
                            item: MatchItem::MatchCharacter(MatchCharacter(Char(c))),
                        },
                    )])]),
                }),
            ])]))
        });

        let all_exprs = [regex_ast_anchored]
            .into_iter()
            .chain(regex_ast_unanchored)
            .collect();

        let results = compile_many(all_exprs);
        let expected = vec![
            Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(6))),
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(0))),
            // first anchored expr
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // unanchored start
            Opcode::Split(InstSplit::new(InstIndex::from(9), InstIndex::from(7))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(6))),
            Opcode::Split(InstSplit::new(InstIndex::from(10), InstIndex::from(15))),
            // first unanchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(1))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // second unanchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(2))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('c')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ];

        assert_eq!(Ok(Instructions::default().with_opcodes(expected)), results,);
    }

    #[test]
    fn should_compile_many_unanchored_exprs() {
        let ast = ['a', 'b', 'c'].into_iter().map(|c| {
            Regex::Unanchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::Capturing {
                    expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                        Match::WithoutQuantifier {
                            item: MatchItem::MatchCharacter(MatchCharacter(Char(c))),
                        },
                    )])]),
                }),
            ])]))
        });

        let results = compile_many(ast.collect());
        let expected = vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(4))),
            Opcode::Split(InstSplit::new(InstIndex::from(10), InstIndex::from(15))),
            // first expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(0))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // second expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(1))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // third expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(2))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('c')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ];

        assert_eq!(
            &Ok(Instructions::default().with_opcodes(expected)),
            &results,
            "{:#?}",
            &results
        );
    }

    #[test]
    fn should_compile_many_unanchored_and_anchored_exprs() {
        let anchored_ast = ['a', 'b', 'c'].into_iter().map(|c| {
            Regex::StartOfStringAnchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::Capturing {
                    expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                        Match::WithoutQuantifier {
                            item: MatchItem::MatchCharacter(MatchCharacter(Char(c))),
                        },
                    )])]),
                }),
            ])]))
        });

        let unanchored_ast = ['x', 'y', 'z'].into_iter().map(|c| {
            Regex::Unanchored(Expression(vec![SubExpression(vec![
                SubExpressionItem::Group(Group::Capturing {
                    expression: Expression(vec![SubExpression(vec![SubExpressionItem::Match(
                        Match::WithoutQuantifier {
                            item: MatchItem::MatchCharacter(MatchCharacter(Char(c))),
                        },
                    )])]),
                }),
            ])]))
        });

        let expr_asts = anchored_ast.chain(unanchored_ast).collect();

        let results = compile_many(expr_asts);
        let expected = vec![
            Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(18))),
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(2))),
            Opcode::Split(InstSplit::new(InstIndex::from(8), InstIndex::from(13))),
            // first anchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(0))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // second anchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(1))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // third anchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(2))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('c')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // start of unanchored exprs
            Opcode::Split(InstSplit::new(InstIndex::from(21), InstIndex::from(19))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(18))),
            Opcode::Split(InstSplit::new(InstIndex::from(23), InstIndex::from(22))),
            Opcode::Split(InstSplit::new(InstIndex::from(28), InstIndex::from(33))),
            // first unanchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(3))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('x')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // second unanchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(4))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('y')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            // third unanchored expr
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(5))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('z')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ];

        assert_eq!(
            &Ok(Instructions::default().with_opcodes(expected)),
            &results,
            "{:#?}",
            &results
        );
    }
}
