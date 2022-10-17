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
    fn compile_functionality_test() {
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
    fn compile_many_functionality_test() {
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
