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
use regex_runtime::*;

/// A internal representation of the `regex_runtime::Opcode` type, with relative
/// addressing.
///
/// ## Note
/// This type is meant to exist only internally and should be
/// refined to the `regex_runtime::Opcode type
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum RelativeOpcode {
    Any,
    Consume(char),
    ConsumeSet(CharacterSet),
    Epsilon(EpsilonCond),
    Split(i32, i32),
    Jmp(i32),
    StartSave(usize),
    EndSave(usize),
    Match,
    Meta(MetaKind),
}

impl RelativeOpcode {
    fn into_opcode_with_index(self, sets: &mut Vec<CharacterSet>, idx: i32) -> Option<Opcode> {
        match self {
            RelativeOpcode::Any => Some(Opcode::Any),
            RelativeOpcode::Consume(c) => Some(Opcode::Consume(InstConsume::new(c))),
            RelativeOpcode::Epsilon(ec) => Some(Opcode::Epsilon(InstEpsilon::new(ec))),
            RelativeOpcode::Split(rel_x, rel_y) => {
                let signed_idx = idx as i32;
                let x: u32 = (signed_idx + rel_x).try_into().ok()?;
                let y: u32 = (signed_idx + rel_y).try_into().ok()?;

                Some(Opcode::Split(InstSplit::new(
                    InstIndex::from(x),
                    InstIndex::from(y),
                )))
            }
            RelativeOpcode::Jmp(rel_jmp_to) => {
                let jmp_to: u32 = (idx + rel_jmp_to).try_into().ok()?;

                Some(Opcode::Jmp(InstJmp::new(InstIndex::from(jmp_to))))
            }
            RelativeOpcode::StartSave(slot) => Some(Opcode::StartSave(InstStartSave::new(slot))),
            RelativeOpcode::EndSave(slot) => Some(Opcode::EndSave(InstEndSave::new(slot))),
            RelativeOpcode::Match => Some(Opcode::Match),
            RelativeOpcode::ConsumeSet(char_set) => {
                let found = sets.iter().position(|set| set == &char_set);
                let set_idx = match found {
                    Some(set_idx) => set_idx,
                    None => {
                        let set_idx = sets.len();
                        sets.push(char_set);
                        set_idx
                    }
                };

                Some(Opcode::ConsumeSet(InstConsumeSet { idx: set_idx }))
            }
            RelativeOpcode::Meta(kind) => Some(Opcode::Meta(InstMeta::new(kind))),
        }
    }

    fn into_opcode_with_index_unchecked(self, sets: &mut Vec<CharacterSet>, idx: i32) -> Opcode {
        self.into_opcode_with_index(sets, idx).unwrap()
    }
}

type RelativeOpcodes = Vec<RelativeOpcode>;

/// Defines a trait for implementing compilation from a regex cast to a
/// lowered output type.
pub trait ASTCompilable<OUTPUT> {
    type Error;

    fn compile(ast: ast::Regex) -> Result<OUTPUT, Self::Error>;
}

/// Defines a type for representing compilation of a parsed ast to the regex
/// runtime vm bytecode directly.
pub struct VM;

impl ASTCompilable<Instructions> for VM {
    type Error = String;

    fn compile(ast: ast::Regex) -> Result<Instructions, Self::Error> {
        compile(ast)
    }
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
pub fn compile(regex_ast: ast::Regex) -> Result<Instructions, String> {
    let mut save_group_id_counter = 0_usize;

    let suffix = [RelativeOpcode::Match];

    let (anchored, relative_ops): (bool, Result<RelativeOpcodes, _>) = match regex_ast {
        ast::Regex::StartOfStringAnchored(expr) => (
            true,
            expression(&mut save_group_id_counter, expr)
                .map(|expr| expr.into_iter().chain(suffix.into_iter()).collect()),
        ),
        ast::Regex::Unanchored(expr) => (
            false,
            expression(&mut save_group_id_counter, expr).map(|expr| {
                // match anything
                let prefix = [
                    RelativeOpcode::Split(3, 1),
                    RelativeOpcode::Any,
                    RelativeOpcode::Jmp(-2),
                ];

                prefix
                    .into_iter()
                    .chain(expr.into_iter())
                    .chain(suffix.into_iter())
                    .collect()
            }),
        ),
    };

    relative_ops
        .map(convert_relative_ops_to_absolute_ops)
        .map(|(sets, insts)| {
            let first_available_stop = (insts)
                .iter()
                .find(|opcode| opcode.requires_lookahead() || opcode.is_explicit_consuming())
                .cloned();

            match (anchored, first_available_stop) {
                (false, Some(Opcode::Consume(InstConsume { value }))) => {
                    Instructions::new(sets, insts).with_fast_forward(FastForward::Char(value))
                }
                (false, Some(Opcode::ConsumeSet(InstConsumeSet { idx }))) => {
                    // panics if the set is undefined.
                    // This should never happen.
                    let _ = sets.get(idx).unwrap();
                    Instructions::new(sets, insts).with_fast_forward(FastForward::Set(idx))
                }
                // This disables fast-forward for all other cases whcih
                // primarily should be either an anchored or empty program.
                _ => Instructions::new(sets, insts),
            }
        })
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
pub fn compile_many(regex_asts: Vec<ast::Regex>) -> Result<Instructions, String> {
    let mut anchored = Vec::with_capacity(regex_asts.len());
    let mut unanchored = Vec::with_capacity(regex_asts.len());

    // populate the above lists of anchored and unanchored programs.
    for (expr_id, regex_ast) in regex_asts.into_iter().enumerate() {
        // reset save group id to 0
        let mut save_group_id_counter = 0_usize;

        let expr_id = u32::try_from(expr_id).map_err(|e| format!("{:?}", e))?;

        // Start
        let mut relative_expression_opcodes =
            vec![RelativeOpcode::Meta(MetaKind::SetExpressionId(expr_id))];

        match regex_ast {
            ast::Regex::StartOfStringAnchored(expr) => {
                let mut generated_opcodes = expression(&mut save_group_id_counter, expr)?;
                relative_expression_opcodes.append(&mut generated_opcodes);
                relative_expression_opcodes.push(RelativeOpcode::Match);
                anchored.push(relative_expression_opcodes);
            }

            ast::Regex::Unanchored(expr) => {
                let mut generated_opcodes = expression(&mut save_group_id_counter, expr)?;
                relative_expression_opcodes.append(&mut generated_opcodes);
                relative_expression_opcodes.push(RelativeOpcode::Match);
                unanchored.push(relative_expression_opcodes);
            }
        };
    }

    let anchored_expression_cnt = anchored.len();
    let anchored_opcode_cnt = anchored.iter().map(|expr| expr.len()).sum::<usize>();

    let unanchored_expression_cnt = unanchored.len();
    let unanchored_offsets_from_start_of_exprs = generate_offsets_from_start(&unanchored);

    let has_both_anchored_and_unanchored_segments =
        (unanchored_expression_cnt > 0) && (anchored_expression_cnt > 0);

    // generate a reverse stack of all expression start offsets then generate routing split operations.
    let anchored_offset_stack = generate_offsets_from_start(&anchored)
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let anchored_expr_splits = generate_nested_split_expressions(anchored_offset_stack)?;

    // generate a reverse stack of all expression start offsets then generate routing split operations.
    let unanchored_expr_splits = generate_nested_split_expressions(
        unanchored_offsets_from_start_of_exprs
            .into_iter()
            .rev()
            .collect::<Vec<_>>(),
    )?;

    // The initial prefix that jumps into the expression routing splits with
    // a lazy fallover into a consuming `Any` match.
    let unanchored_prefix = vec![
        RelativeOpcode::Split(3, 1),
        RelativeOpcode::Any,
        // jmp to the first unanchored split (start of unanchored)
        RelativeOpcode::Jmp(-2),
    ];

    // existence of unanchored expressions changes the anchored_prefix so this is generated last.
    let anchored_prefix = if has_both_anchored_and_unanchored_segments {
        let start_of_unanchored_exprs =
            i32::try_from(anchored_expr_splits.len() + anchored_opcode_cnt + 1)
                .map_err(|e| e.to_string())?;

        vec![RelativeOpcode::Split(1, start_of_unanchored_exprs)]
    } else {
        vec![]
    };

    let joined_opcodes = anchored_prefix
        .into_iter()
        .chain(anchored_expr_splits.into_iter())
        .chain(anchored.into_iter().flatten())
        .chain(unanchored_prefix.into_iter())
        .chain(unanchored_expr_splits.into_iter())
        .chain(unanchored.into_iter().flatten())
        .collect::<Vec<_>>();

    let (sets, opcodes) = convert_relative_ops_to_absolute_ops(joined_opcodes);

    Ok(Instructions::new(sets, opcodes))
}

/// from a stack of offsets, starting from last expression start first,
/// generate all `Split` operations to jump to the corresponding offsets.
fn generate_nested_split_expressions(
    mut offset_stack: Vec<i32>,
) -> Result<RelativeOpcodes, String> {
    let split_cnt = i32::try_from(offset_stack.len()).map_err(|e| e.to_string())?;

    let mut expr_splits = vec![];

    for offset in (1..split_cnt).into_iter().rev() {
        match offset {
            offset if offset == 1 => {
                // safe to while loop condition asserting 2 elements are present.
                let primary_start = offset_stack.pop().unwrap() + offset;
                let secondary_start = offset_stack.pop().unwrap() + offset;
                expr_splits.push(RelativeOpcode::Split(primary_start, secondary_start));
            }
            offset if offset > 1 => {
                // safe to while loop condition due to guarantee there is atleast 1 element in the queue.
                let primary_start = offset_stack.pop().unwrap() + offset;

                // jump to the next expr, with the secondary branch going to the next split
                expr_splits.push(RelativeOpcode::Split(primary_start, 1));
            }

            // this case should never be encountered due to loop guarantee.
            _ => unreachable!(),
        }
    }

    Ok(expr_splits)
}

/// Takes a slice of expressions, starting at 0 and generates a vector of the
/// start of each expression from 0.
fn generate_offsets_from_start(rel_exprs: &[Vec<RelativeOpcode>]) -> Vec<i32> {
    let mut prev = 0;
    let mut offsets_from_start = Vec::with_capacity(rel_exprs.len());

    for rel_expr in rel_exprs {
        offsets_from_start.push(prev);
        prev += i32::try_from(rel_expr.len()).unwrap();
    }

    offsets_from_start
}

fn convert_relative_ops_to_absolute_ops(
    rel_ops: RelativeOpcodes,
) -> (Vec<CharacterSet>, Vec<Opcode>) {
    let (sets, absolute_insts) = rel_ops
        .into_iter()
        .enumerate()
        // truncate idx to a u32
        .map(|(idx, sets)| {
            (
                idx.try_into()
                    .expect("index overflows a 32-bit signed integer"),
                sets,
            )
        })
        .fold((vec![], vec![]), |(mut sets, mut insts), (idx, opcode)| {
            let absolute_opcode = opcode.into_opcode_with_index_unchecked(&mut sets, idx);
            insts.push(absolute_opcode);

            (sets, insts)
        });

    (sets, absolute_insts)
}

fn expression(
    save_group_id_offset: &mut usize,
    expr: ast::Expression,
) -> Result<RelativeOpcodes, String> {
    let ast::Expression(subexprs) = expr;

    let compiled_subexprs = subexprs
        .into_iter()
        .map(|sub_expr| subexpression(save_group_id_offset, sub_expr))
        .collect::<Result<Vec<_>, _>>()?;

    alternations_for_supplied_relative_opcodes(compiled_subexprs)
}

fn subexpression(
    save_group_id_offset: &mut usize,
    subexpr: ast::SubExpression,
) -> Result<RelativeOpcodes, String> {
    let ast::SubExpression(items) = subexpr;

    items
        .into_iter()
        .map(|subexpr_item| match subexpr_item {
            ast::SubExpressionItem::Match(m) => match_item(m),
            ast::SubExpressionItem::Group(g) => group(save_group_id_offset, g),
            ast::SubExpressionItem::Anchor(a) => anchor(a),
            ast::SubExpressionItem::Backreference(_) => unimplemented!(),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|opcodes| opcodes.into_iter().flatten().collect())
}

macro_rules! generate_range_quantifier_block {
    (eager, $min:expr, $consumer:expr) => {
        $consumer
            .clone()
            .into_iter()
            .cycle()
            .take($consumer.len() * $min as usize)
            .into_iter()
            // jump past end of expression
            .chain(vec![RelativeOpcode::Split(1, (($consumer.len() + 2) as i32))].into_iter())
            .chain($consumer.clone().into_iter())
            // return to split
            .chain(vec![RelativeOpcode::Jmp(-($consumer.len() as i32) - 1)].into_iter())
            .collect()
    };

    (lazy, $min:expr, $consumer:expr) => {
        $consumer
            .clone()
            .into_iter()
            .cycle()
            .take($consumer.len() * $min as usize)
            .into_iter()
            // jump past end of expression
            .chain(vec![RelativeOpcode::Split((($consumer.len() + 2) as i32), 1)].into_iter())
            .chain($consumer.clone().into_iter())
            // return to split
            .chain(vec![RelativeOpcode::Jmp(-($consumer.len() as i32) - 1)].into_iter())
            .collect()
    };

    (eager, $min:expr, $max:expr, $consumer:expr) => {
        $consumer
            .clone()
            .into_iter()
            .cycle()
            .take($consumer.len() * $min as usize)
            .into_iter()
            .chain((0..($max - $min)).flat_map(|_| {
                vec![RelativeOpcode::Split(1, ($consumer.len() as i32) + 1)]
                    .into_iter()
                    .chain($consumer.clone().into_iter())
            }))
            .collect()
    };

    (lazy, $min:expr, $max:expr, $consumer:expr) => {
        $consumer
            .clone()
            .into_iter()
            .cycle()
            .take($consumer.len() * $min as usize)
            .into_iter()
            .chain((0..($max - $min)).flat_map(|_| {
                vec![RelativeOpcode::Split(($consumer.len() as i32) + 1, 1)]
                    .into_iter()
                    .chain($consumer.clone().into_iter())
            }))
            .collect()
    };
}

fn match_item(m: ast::Match) -> Result<RelativeOpcodes, String> {
    use ast::{
        Char, Integer, Match, MatchCharacter, MatchCharacterClass, MatchItem, Quantifier,
        QuantifierType,
    };

    match m {
        // Any character matchers
        Match::WithoutQuantifier {
            item: MatchItem::MatchAnyCharacter,
        } => Ok(vec![RelativeOpcode::Any]),
        Match::WithQuantifier {
            item: MatchItem::MatchAnyCharacter,
            quantifier,
        } => {
            match quantifier {
                Quantifier::Eager(QuantifierType::ZeroOrOne) => Ok(
                    generate_range_quantifier_block!(eager, 0, 1, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Lazy(QuantifierType::ZeroOrOne) => Ok(
                    generate_range_quantifier_block!(lazy, 0, 1, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Eager(QuantifierType::ZeroOrMore) => Ok(
                    generate_range_quantifier_block!(eager, 0, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Lazy(QuantifierType::ZeroOrMore) => Ok(
                    generate_range_quantifier_block!(lazy, 0, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Eager(QuantifierType::OneOrMore) => Ok(
                    generate_range_quantifier_block!(eager, 1, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Lazy(QuantifierType::OneOrMore) => Ok(
                    generate_range_quantifier_block!(lazy, 1, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(cnt)))
                | Quantifier::Eager(QuantifierType::MatchExactRange(Integer(cnt))) => {
                    Ok(vec![RelativeOpcode::Any; cnt as usize])
                }
                Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(cnt))) => Ok(
                    generate_range_quantifier_block!(eager, cnt, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(cnt))) => Ok(
                    generate_range_quantifier_block!(lazy, cnt, vec![RelativeOpcode::Any]),
                ),
                Quantifier::Eager(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(lower),
                    upper_bound: Integer(upper),
                }) => Ok(generate_range_quantifier_block!(
                    eager,
                    lower,
                    upper,
                    vec![RelativeOpcode::Any]
                )),
                Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(lower),
                    upper_bound: Integer(upper),
                }) => Ok(generate_range_quantifier_block!(
                    lazy,
                    lower,
                    upper,
                    vec![RelativeOpcode::Any]
                )),
            }
        }

        // Character matchers
        Match::WithoutQuantifier {
            item: MatchItem::MatchCharacter(MatchCharacter(Char(c))),
        } => Ok(vec![RelativeOpcode::Consume(c)]),
        Match::WithQuantifier {
            item: MatchItem::MatchCharacter(MatchCharacter(Char(c))),
            quantifier,
        } => {
            match quantifier {
                Quantifier::Eager(QuantifierType::ZeroOrOne) => Ok(
                    generate_range_quantifier_block!(eager, 0, 1, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Lazy(QuantifierType::ZeroOrOne) => Ok(
                    generate_range_quantifier_block!(lazy, 0, 1, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Eager(QuantifierType::ZeroOrMore) => Ok(
                    generate_range_quantifier_block!(eager, 0, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Lazy(QuantifierType::ZeroOrMore) => Ok(
                    generate_range_quantifier_block!(lazy, 0, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Eager(QuantifierType::OneOrMore) => Ok(
                    generate_range_quantifier_block!(eager, 1, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Lazy(QuantifierType::OneOrMore) => Ok(
                    generate_range_quantifier_block!(lazy, 1, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(cnt)))
                | Quantifier::Eager(QuantifierType::MatchExactRange(Integer(cnt))) => {
                    Ok(vec![RelativeOpcode::Consume(c); cnt as usize])
                }
                Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(cnt))) => Ok(
                    generate_range_quantifier_block!(eager, cnt, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(cnt))) => Ok(
                    generate_range_quantifier_block!(lazy, cnt, vec![RelativeOpcode::Consume(c)]),
                ),
                Quantifier::Eager(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(lower),
                    upper_bound: Integer(upper),
                }) => Ok(generate_range_quantifier_block!(
                    eager,
                    lower,
                    upper,
                    vec![RelativeOpcode::Consume(c)]
                )),
                Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(lower),
                    upper_bound: Integer(upper),
                }) => Ok(generate_range_quantifier_block!(
                    lazy,
                    lower,
                    upper,
                    vec![RelativeOpcode::Consume(c)]
                )),
            }
        }

        // Character classes
        Match::WithoutQuantifier {
            item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(cc)),
        } => character_class(cc),
        Match::WithQuantifier {
            item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClass(cc)),
            quantifier,
        } => match quantifier {
            Quantifier::Eager(QuantifierType::ZeroOrOne) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(eager, 0, 1, rel_ops)),
            Quantifier::Lazy(QuantifierType::ZeroOrOne) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, 0, 1, rel_ops)),
            Quantifier::Eager(QuantifierType::ZeroOrMore) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(eager, 0, rel_ops)),
            Quantifier::Lazy(QuantifierType::ZeroOrMore) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, 0, rel_ops)),
            Quantifier::Eager(QuantifierType::OneOrMore) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(eager, 1, rel_ops)),
            Quantifier::Lazy(QuantifierType::OneOrMore) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, 1, rel_ops)),
            Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(cnt)))
            | Quantifier::Eager(QuantifierType::MatchExactRange(Integer(cnt))) => {
                character_class(cc).map(|rel_ops| {
                    let multiple_of_len = rel_ops.len() * (cnt as usize);

                    rel_ops.into_iter().cycle().take(multiple_of_len).collect()
                })
            }
            Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                character_class(cc)
                    .map(|rel_ops| generate_range_quantifier_block!(eager, lower, rel_ops))
            }
            Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                character_class(cc)
                    .map(|rel_ops| generate_range_quantifier_block!(lazy, lower, rel_ops))
            }
            Quantifier::Eager(QuantifierType::MatchBetweenRange {
                lower_bound: Integer(lower),
                upper_bound: Integer(upper),
            }) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(eager, lower, upper, rel_ops)),
            Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                lower_bound: Integer(lower),
                upper_bound: Integer(upper),
            }) => character_class(cc)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, lower, upper, rel_ops)),
        },

        // Character groups
        Match::WithoutQuantifier {
            item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterGroup(cg)),
        } => character_group(cg),
        Match::WithQuantifier {
            item: MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterGroup(cg)),
            quantifier,
        } => match quantifier {
            Quantifier::Eager(QuantifierType::ZeroOrOne) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(eager, 0, 1, rel_ops)),
            Quantifier::Lazy(QuantifierType::ZeroOrOne) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, 0, 1, rel_ops)),
            Quantifier::Eager(QuantifierType::ZeroOrMore) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(eager, 0, rel_ops)),
            Quantifier::Lazy(QuantifierType::ZeroOrMore) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, 0, rel_ops)),
            Quantifier::Eager(QuantifierType::OneOrMore) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(eager, 1, rel_ops)),
            Quantifier::Lazy(QuantifierType::OneOrMore) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, 1, rel_ops)),
            Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(cnt)))
            | Quantifier::Eager(QuantifierType::MatchExactRange(Integer(cnt))) => {
                character_group(cg).map(|rel_ops| {
                    let multiple_of_len = rel_ops.len() * (cnt as usize);

                    rel_ops.into_iter().cycle().take(multiple_of_len).collect()
                })
            }
            Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                character_group(cg)
                    .map(|rel_ops| generate_range_quantifier_block!(eager, lower, rel_ops))
            }
            Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                character_group(cg)
                    .map(|rel_ops| generate_range_quantifier_block!(lazy, lower, rel_ops))
            }
            Quantifier::Eager(QuantifierType::MatchBetweenRange {
                lower_bound: Integer(lower),
                upper_bound: Integer(upper),
            }) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(eager, lower, upper, rel_ops)),
            Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                lower_bound: Integer(lower),
                upper_bound: Integer(upper),
            }) => character_group(cg)
                .map(|rel_ops| generate_range_quantifier_block!(lazy, lower, upper, rel_ops)),
        },

        // Unicode categories
        Match::WithoutQuantifier {
            item:
                MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClassFromUnicodeCategory(
                    ast::CharacterClassFromUnicodeCategory(category),
                )),
        } => {
            let set = unicode_category_to_character_set(category);
            Ok(vec![RelativeOpcode::ConsumeSet(set)])
        }
        Match::WithQuantifier {
            item:
                MatchItem::MatchCharacterClass(MatchCharacterClass::CharacterClassFromUnicodeCategory(
                    ast::CharacterClassFromUnicodeCategory(category),
                )),
            quantifier,
        } => {
            let set = unicode_category_to_character_set(category);
            let rel_ops = vec![RelativeOpcode::ConsumeSet(set)];

            let quantified_rel_ops = match quantifier {
                Quantifier::Eager(QuantifierType::ZeroOrOne) => {
                    generate_range_quantifier_block!(eager, 0, 1, rel_ops)
                }
                Quantifier::Lazy(QuantifierType::ZeroOrOne) => {
                    generate_range_quantifier_block!(lazy, 0, 1, rel_ops)
                }
                Quantifier::Eager(QuantifierType::ZeroOrMore) => {
                    generate_range_quantifier_block!(eager, 0, rel_ops)
                }
                Quantifier::Lazy(QuantifierType::ZeroOrMore) => {
                    generate_range_quantifier_block!(lazy, 0, rel_ops)
                }
                Quantifier::Eager(QuantifierType::OneOrMore) => {
                    generate_range_quantifier_block!(eager, 1, rel_ops)
                }
                Quantifier::Lazy(QuantifierType::OneOrMore) => {
                    generate_range_quantifier_block!(lazy, 1, rel_ops)
                }
                Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(cnt)))
                | Quantifier::Eager(QuantifierType::MatchExactRange(Integer(cnt))) => {
                    let multiple_of_len = rel_ops.len() * (cnt as usize);

                    rel_ops.into_iter().cycle().take(multiple_of_len).collect()
                }
                Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                    generate_range_quantifier_block!(eager, lower, rel_ops)
                }
                Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                    generate_range_quantifier_block!(lazy, lower, rel_ops)
                }
                Quantifier::Eager(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(lower),
                    upper_bound: Integer(upper),
                }) => generate_range_quantifier_block!(eager, lower, upper, rel_ops),
                Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                    lower_bound: Integer(lower),
                    upper_bound: Integer(upper),
                }) => generate_range_quantifier_block!(lazy, lower, upper, rel_ops),
            };

            Ok(quantified_rel_ops)
        }
    }
}

fn character_group(cg: ast::CharacterGroup) -> Result<RelativeOpcodes, String> {
    let (negated, items) = match cg {
        ast::CharacterGroup::NegatedItems(cgis) => (true, cgis),
        ast::CharacterGroup::Items(cgis) => (false, cgis),
    };

    let item_cnt = items.len();

    // fold all explicit character alphabets into a single alphabet.
    let (mut explicit, other_alphabets) = items
        .into_iter()
        .map(character_group_item_to_alphabet)
        .fold(
            (Vec::with_capacity(item_cnt), Vec::with_capacity(item_cnt)),
            |(mut chars, mut other), x| {
                match x {
                    ca @ CharacterAlphabet::Range(_)
                    | ca @ CharacterAlphabet::Ranges(_)
                    | ca @ CharacterAlphabet::UnicodeCategory(_) => other.push(ca),
                    CharacterAlphabet::Explicit(mut c) => chars.append(&mut c),
                };

                (chars, other)
            },
        );

    let explicit_alphabet = if explicit.is_empty() {
        vec![]
    } else {
        // deduplicate characters to save traversal time.
        explicit.dedup();
        vec![CharacterAlphabet::Explicit(explicit)]
    };

    // construct a set from the alphabet
    let sets: Vec<RelativeOpcodes> = explicit_alphabet
        .into_iter()
        .chain(other_alphabets)
        .into_iter()
        .map(|alphabet| {
            if negated {
                CharacterSet::exclusive(alphabet)
            } else {
                CharacterSet::inclusive(alphabet)
            }
        })
        .map(|set| vec![RelativeOpcode::ConsumeSet(set)])
        .collect();

    alternations_for_supplied_relative_opcodes(sets)
}

fn character_group_item_to_alphabet(cgi: ast::CharacterGroupItem) -> CharacterAlphabet {
    use ast::Char;

    match cgi {
        ast::CharacterGroupItem::CharacterClassFromUnicodeCategory(_) => unimplemented!(),
        ast::CharacterGroupItem::CharacterClass(cc) => character_class_to_set(cc).set,
        ast::CharacterGroupItem::CharacterRange(Char(lower), Char(upper)) => {
            CharacterAlphabet::Range(lower..=upper)
        }
        ast::CharacterGroupItem::Char(Char(c)) => CharacterAlphabet::Explicit(vec![c]),
    }
}

// character classes

/// A representation of a AnyWordClass character class, in character set format.
pub struct AnyWordClass;

impl AnyWordClass {
    const RANGES: [std::ops::RangeInclusive<char>; 4] =
        ['a'..='z', 'A'..='Z', '0'..='9', '_'..='_'];
}

impl CharacterSetRepresentable for AnyWordClass {}

impl From<AnyWordClass> for CharacterSet {
    fn from(_: AnyWordClass) -> Self {
        CharacterSet::inclusive(CharacterAlphabet::Ranges(AnyWordClass::RANGES.to_vec()))
    }
}

/// A representation of a AnyWordClassInverted character class, in character
/// set format.
pub struct AnyWordClassInverted;

impl CharacterSetRepresentable for AnyWordClassInverted {}

impl From<AnyWordClassInverted> for CharacterSet {
    fn from(_: AnyWordClassInverted) -> Self {
        CharacterSet::exclusive(CharacterAlphabet::Ranges(AnyWordClass::RANGES.to_vec()))
    }
}

/// A representation of a AnyDecimalDigitClass character class, in character
/// set format.
pub struct AnyDecimalDigitClass;

impl AnyDecimalDigitClass {
    const RANGE: std::ops::RangeInclusive<char> = '0'..='9';
}

impl CharacterSetRepresentable for AnyDecimalDigitClass {}

impl From<AnyDecimalDigitClass> for CharacterSet {
    fn from(_: AnyDecimalDigitClass) -> Self {
        CharacterSet::inclusive(CharacterAlphabet::Range(AnyDecimalDigitClass::RANGE))
    }
}

/// A representation of a AnyDecimalDigitClassInverted character class, in
/// character set format.
pub struct AnyDecimalDigitClassInverted;

impl CharacterSetRepresentable for AnyDecimalDigitClassInverted {}

impl From<AnyDecimalDigitClassInverted> for CharacterSet {
    fn from(_: AnyDecimalDigitClassInverted) -> Self {
        CharacterSet::exclusive(CharacterAlphabet::Range(AnyDecimalDigitClass::RANGE))
    }
}

fn character_class(cc: ast::CharacterClass) -> Result<RelativeOpcodes, String> {
    let set = character_class_to_set(cc);

    Ok(vec![RelativeOpcode::ConsumeSet(set)])
}

fn character_class_to_set(cc: ast::CharacterClass) -> CharacterSet {
    match cc {
        ast::CharacterClass::AnyWord => AnyWordClass.into(),
        ast::CharacterClass::AnyWordInverted => AnyWordClassInverted.into(),

        ast::CharacterClass::AnyDecimalDigit => AnyDecimalDigitClass.into(),
        ast::CharacterClass::AnyDecimalDigitInverted => AnyDecimalDigitClassInverted.into(),
    }
}

fn unicode_category_to_character_set(category: ast::UnicodeCategoryName) -> CharacterSet {
    let runtime_category = match category {
        ast::UnicodeCategoryName::Letter => UnicodeCategory::Letter,
        ast::UnicodeCategoryName::LowercaseLetter => UnicodeCategory::LowercaseLetter,
        ast::UnicodeCategoryName::UppercaseLetter => UnicodeCategory::UppercaseLetter,
        ast::UnicodeCategoryName::TitlecaseLetter => UnicodeCategory::TitlecaseLetter,
        ast::UnicodeCategoryName::CasedLetter => UnicodeCategory::CasedLetter,
        ast::UnicodeCategoryName::ModifiedLetter => UnicodeCategory::ModifiedLetter,
        ast::UnicodeCategoryName::OtherLetter => UnicodeCategory::OtherLetter,
        ast::UnicodeCategoryName::Mark => UnicodeCategory::Mark,
        ast::UnicodeCategoryName::NonSpacingMark => UnicodeCategory::NonSpacingMark,
        ast::UnicodeCategoryName::SpacingCombiningMark => UnicodeCategory::SpacingCombiningMark,
        ast::UnicodeCategoryName::EnclosingMark => UnicodeCategory::EnclosingMark,
        ast::UnicodeCategoryName::Separator => UnicodeCategory::Separator,
        ast::UnicodeCategoryName::SpaceSeparator => UnicodeCategory::SpaceSeparator,
        ast::UnicodeCategoryName::LineSeparator => UnicodeCategory::LineSeparator,
        ast::UnicodeCategoryName::ParagraphSeparator => UnicodeCategory::ParagraphSeparator,
        ast::UnicodeCategoryName::Symbol => UnicodeCategory::Symbol,
        ast::UnicodeCategoryName::MathSymbol => UnicodeCategory::MathSymbol,
        ast::UnicodeCategoryName::CurrencySymbol => UnicodeCategory::CurrencySymbol,
        ast::UnicodeCategoryName::ModifierSymbol => UnicodeCategory::ModifierSymbol,
        ast::UnicodeCategoryName::OtherSymbol => UnicodeCategory::OtherSymbol,
        ast::UnicodeCategoryName::Number => UnicodeCategory::Number,
        ast::UnicodeCategoryName::DecimalDigitNumber => UnicodeCategory::DecimalDigitNumber,
        ast::UnicodeCategoryName::LetterNumber => UnicodeCategory::LetterNumber,
        ast::UnicodeCategoryName::OtherNumber => UnicodeCategory::OtherNumber,
        ast::UnicodeCategoryName::Punctuation => UnicodeCategory::Punctuation,
        ast::UnicodeCategoryName::DashPunctuation => UnicodeCategory::DashPunctuation,
        ast::UnicodeCategoryName::OpenPunctuation => UnicodeCategory::OpenPunctuation,
        ast::UnicodeCategoryName::ClosePunctuation => UnicodeCategory::ClosePunctuation,
        ast::UnicodeCategoryName::InitialPunctuation => UnicodeCategory::InitialPunctuation,
        ast::UnicodeCategoryName::FinalPunctuation => UnicodeCategory::FinalPunctuation,
        ast::UnicodeCategoryName::ConnectorPunctuation => UnicodeCategory::ConnectorPunctuation,
        ast::UnicodeCategoryName::OtherPunctuation => UnicodeCategory::OpenPunctuation,
        ast::UnicodeCategoryName::Other => UnicodeCategory::Other,
        ast::UnicodeCategoryName::Control => UnicodeCategory::Control,
        ast::UnicodeCategoryName::Format => UnicodeCategory::Format,
        ast::UnicodeCategoryName::PrivateUse => UnicodeCategory::PrivateUse,
        ast::UnicodeCategoryName::Surrogate => UnicodeCategory::Surrogate,
        ast::UnicodeCategoryName::Unassigned => UnicodeCategory::Unassigned,
    };

    CharacterSet::inclusive(CharacterAlphabet::UnicodeCategory(runtime_category))
}

/// Generates alternations from a block of relative operations.
fn alternations_for_supplied_relative_opcodes(
    rel_ops: Vec<RelativeOpcodes>,
) -> Result<RelativeOpcodes, String> {
    let subexpr_cnt = rel_ops.len();

    let length_of_rel_ops: Vec<_> = rel_ops
        .iter()
        .enumerate()
        .map(|(idx, subexpr)| ((idx + 1 == subexpr_cnt), subexpr))
        .map(|(is_last, subexpr)| {
            // last alternation doesn't require a split prefix and jump suffix
            if is_last {
                subexpr.len()
            } else {
                subexpr.len() + 2
            }
        })
        .collect();

    let total_length_of_compiled_expr: usize = length_of_rel_ops.iter().sum();
    let start_end_offsets_by_subexpr: Vec<(usize, usize)> = length_of_rel_ops
        .iter()
        .fold(
            // add 1 to set end at first instruction of next expr
            (total_length_of_compiled_expr + 1, vec![]),
            |(offset_to_end, mut acc), &subexpr_len| {
                let new_offset_to_end = offset_to_end - subexpr_len;

                acc.push((subexpr_len, new_offset_to_end));
                (new_offset_to_end, acc)
            },
        )
        .1;

    let compiled_ops_with_applied_alternations = rel_ops
        .into_iter()
        .zip(start_end_offsets_by_subexpr.into_iter())
        .enumerate()
        .map(|(idx, (opcodes, (start, end)))| {
            // unwraps should be safe as these should never be able to
            // overflow a 32-bit integer. However if they do I'd like to catch
            // panic.
            (
                u32::try_from(idx).unwrap(),
                opcodes,
                (i32::try_from(start).unwrap(), i32::try_from(end).unwrap()),
            )
        })
        .map(|(idx, opcodes, start_end_offsets)| {
            let optional_next_offsets =
                (((idx as usize) + 1) != subexpr_cnt).then_some(start_end_offsets);
            (optional_next_offsets, opcodes)
        })
        .flat_map(|(start_of_next, ops)| match start_of_next {
            Some((start_of_next_subexpr_offset, end_of_expr_offset)) => {
                [RelativeOpcode::Split(1, start_of_next_subexpr_offset)]
                    .into_iter()
                    .chain(ops.into_iter())
                    .chain([RelativeOpcode::Jmp(end_of_expr_offset)].into_iter())
                    .collect()
            }
            None => ops,
        })
        .collect();

    Ok(compiled_ops_with_applied_alternations)
}

// Groups

fn group(save_group_id_offset: &mut usize, g: ast::Group) -> Result<RelativeOpcodes, String> {
    use ast::{Integer, Quantifier, QuantifierType};

    match g {
        ast::Group::Capturing { expression: expr } => {
            let save_group_id = *save_group_id_offset;
            *save_group_id_offset += 1;
            let save_group_prefix = [RelativeOpcode::StartSave(save_group_id)];
            let save_group_suffix = [RelativeOpcode::EndSave(save_group_id)];

            expression(save_group_id_offset, expr).map(|insts| {
                save_group_prefix
                    .into_iter()
                    .chain(insts.into_iter())
                    .chain(save_group_suffix.into_iter())
                    .collect()
            })
        }
        ast::Group::CapturingWithQuantifier {
            expression: expr,
            quantifier,
        } => {
            let save_group_id = *save_group_id_offset;
            *save_group_id_offset += 1;
            let save_group_prefix = [RelativeOpcode::StartSave(save_group_id)];
            let save_group_suffix = [RelativeOpcode::EndSave(save_group_id)];

            expression(save_group_id_offset, expr)
                .map(|rel_ops| match quantifier {
                    Quantifier::Eager(QuantifierType::ZeroOrOne) => {
                        generate_range_quantifier_block!(eager, 0, 1, rel_ops)
                    }
                    Quantifier::Lazy(QuantifierType::ZeroOrOne) => {
                        generate_range_quantifier_block!(lazy, 0, 1, rel_ops)
                    }
                    Quantifier::Eager(QuantifierType::ZeroOrMore) => {
                        generate_range_quantifier_block!(eager, 0, rel_ops)
                    }
                    Quantifier::Lazy(QuantifierType::ZeroOrMore) => {
                        generate_range_quantifier_block!(lazy, 0, rel_ops)
                    }
                    Quantifier::Eager(QuantifierType::OneOrMore) => {
                        generate_range_quantifier_block!(eager, 1, rel_ops)
                    }
                    Quantifier::Lazy(QuantifierType::OneOrMore) => {
                        generate_range_quantifier_block!(lazy, 1, rel_ops)
                    }
                    Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                        generate_range_quantifier_block!(eager, lower, rel_ops)
                    }
                    Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                        generate_range_quantifier_block!(lazy, lower, rel_ops)
                    }
                    Quantifier::Eager(QuantifierType::MatchBetweenRange {
                        lower_bound: Integer(lower),
                        upper_bound: Integer(upper),
                    }) => generate_range_quantifier_block!(eager, lower, upper, rel_ops),
                    Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                        lower_bound: Integer(lower),
                        upper_bound: Integer(upper),
                    }) => generate_range_quantifier_block!(lazy, lower, upper, rel_ops),
                    Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(cnt)))
                    | Quantifier::Eager(QuantifierType::MatchExactRange(Integer(cnt))) => {
                        let multiple_of_len = rel_ops.len() * (cnt as usize);

                        rel_ops.into_iter().cycle().take(multiple_of_len).collect()
                    }
                })
                .map(|insts: RelativeOpcodes| {
                    save_group_prefix
                        .into_iter()
                        .chain(insts.into_iter())
                        .chain(save_group_suffix.into_iter())
                        .collect()
                })
        }

        ast::Group::NonCapturing { expression: expr } => expression(save_group_id_offset, expr),
        ast::Group::NonCapturingWithQuantifier {
            expression: expr,
            quantifier,
        } => expression(save_group_id_offset, expr).map(|rel_ops| match quantifier {
            Quantifier::Eager(QuantifierType::ZeroOrOne) => {
                generate_range_quantifier_block!(eager, 0, 1, rel_ops)
            }
            Quantifier::Lazy(QuantifierType::ZeroOrOne) => {
                generate_range_quantifier_block!(lazy, 0, 1, rel_ops)
            }
            Quantifier::Eager(QuantifierType::ZeroOrMore) => {
                generate_range_quantifier_block!(eager, 0, rel_ops)
            }
            Quantifier::Lazy(QuantifierType::ZeroOrMore) => {
                generate_range_quantifier_block!(lazy, 0, rel_ops)
            }
            Quantifier::Eager(QuantifierType::OneOrMore) => {
                generate_range_quantifier_block!(eager, 1, rel_ops)
            }
            Quantifier::Lazy(QuantifierType::OneOrMore) => {
                generate_range_quantifier_block!(lazy, 1, rel_ops)
            }
            Quantifier::Eager(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                generate_range_quantifier_block!(eager, lower, rel_ops)
            }
            Quantifier::Lazy(QuantifierType::MatchAtLeastRange(Integer(lower))) => {
                generate_range_quantifier_block!(lazy, lower, rel_ops)
            }
            Quantifier::Eager(QuantifierType::MatchBetweenRange {
                lower_bound: Integer(lower),
                upper_bound: Integer(upper),
            }) => generate_range_quantifier_block!(eager, lower, upper, rel_ops),
            Quantifier::Lazy(QuantifierType::MatchBetweenRange {
                lower_bound: Integer(lower),
                upper_bound: Integer(upper),
            }) => generate_range_quantifier_block!(lazy, lower, upper, rel_ops),
            Quantifier::Lazy(QuantifierType::MatchExactRange(Integer(cnt)))
            | Quantifier::Eager(QuantifierType::MatchExactRange(Integer(cnt))) => {
                let multiple_of_len = rel_ops.len() * (cnt as usize);

                rel_ops.into_iter().cycle().take(multiple_of_len).collect()
            }
        }),
    }
}

// Anchors

fn anchor(a: ast::Anchor) -> Result<RelativeOpcodes, String> {
    let cond = match a {
        ast::Anchor::WordBoundary => EpsilonCond::WordBoundary,
        ast::Anchor::NonWordBoundary => EpsilonCond::NonWordBoundary,
        ast::Anchor::StartOfStringOnly => EpsilonCond::StartOfStringOnly,
        ast::Anchor::EndOfStringOnlyNonNewline => EpsilonCond::EndOfStringOnlyNonNewline,
        ast::Anchor::EndOfStringOnly => EpsilonCond::EndOfStringOnly,
        ast::Anchor::PreviousMatchEnd => EpsilonCond::PreviousMatchEnd,
        ast::Anchor::EndOfString => EpsilonCond::EndOfString,
    };

    Ok(vec![RelativeOpcode::Epsilon(cond)])
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
            VM::compile(regex_ast)
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
