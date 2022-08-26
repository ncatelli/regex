//! Provides a runtime evaluator for a compiled regex bytecode.
//!
//! # Example
//!
//! ```
//! use regex_runtime::*;
//!
//! let progs = vec![
//!     (
//!         Some([SaveGroupSlot::complete(0, 0, 1)]),
//!         Instructions::default().with_opcodes(vec![
//!             Opcode::StartSave(InstStartSave::new(0)),
//!             Opcode::Consume(InstConsume::new('a')),
//!             Opcode::EndSave(InstEndSave::new(0)),
//!             Opcode::Match,
//!         ]),
//!     ),
//!     (
//!         None,
//!         Instructions::default().with_opcodes(vec![
//!             Opcode::StartSave(InstStartSave::new(0)),
//!             Opcode::Consume(InstConsume::new('b')),
//!             Opcode::EndSave(InstEndSave::new(0)),
//!             Opcode::Match,
//!         ]),
//!     ),
//! ];
//!
//! let input = "aab";
//!
//! for (expected_res, prog) in progs {
//!     let res = run::<1>(&prog, input);
//!     assert_eq!(expected_res, res)
//! }
//! ```

use collections_ext::set::sparse::SparseSet;
use std::fmt::{Debug, Display};

pub mod bytecode;
pub use bytecode::from_binary;

/// Represents a defined match group for a pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveGroupSlot {
    /// No valid match for the slot has been found.
    None,
    /// A valid match has been found between the exlusive range of `start..end`.
    Complete {
        /// Stores the matching expression id. For a single expression program
        /// this defaults to `0`.
        expression_id: u32,
        /// The start index of the match.
        start: usize,
        /// The exclusive index of the match. i.e. if the match first index of
        /// a one character match is `0` this will be `1`.
        end: usize,
    },
}

impl SaveGroupSlot {
    /// Returns a boolean representing if the savegroup slot is of the `None`
    /// variant, signifying a match was not found.
    pub fn is_none(&self) -> bool {
        matches!(self, SaveGroupSlot::None)
    }

    /// Returns a boolean representing if the savegroup slot is of the
    /// `Complete` variant, signifying a match was found.
    pub fn is_complete(&self) -> bool {
        !self.is_none()
    }

    /// Returns a completed save group from its constituent parts.
    pub const fn complete(expression_id: u32, start: usize, end: usize) -> Self {
        Self::Complete {
            expression_id,
            start,
            end,
        }
    }
}

impl From<SaveGroup> for SaveGroupSlot {
    fn from(src: SaveGroup) -> Self {
        match src {
            SaveGroup::None => SaveGroupSlot::None,
            SaveGroup::Allocated { .. } => SaveGroupSlot::None,
            SaveGroup::Open { .. } => SaveGroupSlot::None,
            SaveGroup::Complete {
                expression_id,
                start,
                end,
                ..
            } => SaveGroupSlot::complete(expression_id, start, end),
        }
    }
}

impl Default for SaveGroupSlot {
    fn default() -> Self {
        SaveGroupSlot::None
    }
}

/// Represents a Save Group as tracked on an open thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveGroup {
    /// No available slot has been encountered.
    None,
    /// Represents an allocated save slot (for cases where a StartSave
    /// operation have been encountered) but no consuming operations have
    /// occurred for a potential match on a thread.
    Allocated { expression_id: u32, slot_id: usize },
    /// Represents an allocated Slot that has encountered atleast one potential
    /// consuming match but a full match for the slot has not yet been found
    /// (TL;DR pending EndSave).
    Open {
        expression_id: u32,
        slot_id: usize,
        start: usize,
    },
    /// A valid match with a defined start and end has been found.
    Complete {
        expression_id: u32,
        slot_id: usize,
        start: usize,
        end: usize,
    },
}

impl SaveGroup {
    /// Returns true if the SaveGroup is in an allocated state.
    pub fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated { .. })
    }

    /// Instantiates a `SaveGroup::Allocated` for a given slot_id.
    pub fn allocated(expression_id: u32, slot_id: usize) -> Self {
        Self::Allocated {
            slot_id,
            expression_id,
        }
    }

    /// Instantiates a `SaveGroup::Open` for a given slot id and start
    /// position.
    pub fn open(expression_id: u32, slot_id: usize, start: usize) -> Self {
        Self::Open {
            expression_id,
            slot_id,
            start,
        }
    }

    /// Instantiates a `SaveGroup::Complete` for a given slot id, start and end
    /// position.
    pub fn complete(expression_id: u32, slot_id: usize, start: usize, end: usize) -> Self {
        Self::Complete {
            expression_id,
            slot_id,
            start,
            end,
        }
    }
}

/// A thread represents a branch in a patterns evaluation, storing that
/// branches current save state and an instruction pointer.
#[derive(Debug)]
struct Thread<const SG: usize> {
    /// Represents the optional id of the threads enclosing expression
    /// multi-expression programs.
    expression_id: u32,
    save_groups: [SaveGroup; SG],
    inst: InstIndex,
}

impl<const SG: usize> Thread<SG> {
    fn new(save_groups: [SaveGroup; SG], inst: InstIndex) -> Self {
        Self {
            expression_id: 0,
            save_groups,
            inst,
        }
    }

    fn new_with_id(expression_id: u32, save_groups: [SaveGroup; SG], inst: InstIndex) -> Self {
        Self {
            expression_id,
            save_groups,
            inst,
        }
    }

    fn with_expression_id(mut self, expr_id: u32) -> Self {
        self.expression_id = expr_id;
        self
    }
}

/// Stores all active threads and a set representing the evaluation generation.
#[derive(Debug)]
struct Threads<const SG: usize> {
    gen: SparseSet,
    threads: Vec<Thread<SG>>,
}

impl<const SG: usize> Threads<SG> {
    pub fn with_set_size(set_capacity: usize) -> Self {
        let ops = SparseSet::new(set_capacity);
        Self {
            threads: vec![],
            gen: ops,
        }
    }
}

impl<const SG: usize> Default for Threads<SG> {
    fn default() -> Self {
        let ops = SparseSet::new(0);
        Self {
            threads: vec![],
            gen: ops,
        }
    }
}

/// Representative the first consuming match that the runtime can fast-forward
/// to in an input.
#[derive(Debug, PartialEq)]
pub enum FastForward {
    /// Represents a single character.
    Char(char),
    /// Represents a set of characters that could be consuming.
    Set(usize),
    /// Represents that no fast-forward should be performed.
    None,
}

impl Default for FastForward {
    fn default() -> Self {
        Self::None
    }
}

/// Represents a runtime program, consisting of all character sets referenced
/// in an evaluation, all instructions in the program and whether the runtime
/// can fast-forward through an input.
#[derive(Default, Debug, PartialEq)]
pub struct Instructions {
    pub sets: Vec<CharacterSet>,
    pub program: Vec<Instruction>,
    pub fast_forward: FastForward,
}

impl Instructions {
    pub const MAGIC_NUMBER: u16 = 0xF0F0;

    /// Instantiates a program from a predefined list of operations and
    /// character sets. By default fast_forward is disabled.
    #[must_use]
    pub fn new(sets: Vec<CharacterSet>, program: Vec<Opcode>) -> Self {
        Self {
            sets,
            program: program
                .into_iter()
                .enumerate()
                .map(|(id, opcode)| Instruction::new(id, opcode))
                .collect(),
            fast_forward: FastForward::None,
        }
    }

    /// Modifies a program, assigning a new set of opcodes to a program after
    /// up-converting it to a list of instructions.
    pub fn with_opcodes(self, program: Vec<Opcode>) -> Self {
        Self {
            sets: self.sets,
            program: program
                .into_iter()
                .enumerate()
                .map(|(id, opcode)| Instruction::new(id, opcode))
                .collect(),
            fast_forward: self.fast_forward,
        }
    }

    /// Modifies a program, assigning a new list of instructions.
    pub fn with_instructions(self, program: Vec<Instruction>) -> Self {
        Self {
            sets: self.sets,
            program,
            fast_forward: self.fast_forward,
        }
    }

    /// Modifies a program, assigning a new list of character sets.
    pub fn with_sets(self, sets: Vec<CharacterSet>) -> Self {
        Self {
            sets,
            program: self.program,
            fast_forward: self.fast_forward,
        }
    }

    /// Modifies a program, assigning a new fast-forward setting.
    pub fn with_fast_forward(self, fast_forward: FastForward) -> Self {
        Self {
            sets: self.sets,
            program: self.program,
            fast_forward,
        }
    }

    /// Returns the length of the associated program.
    pub fn len(&self) -> usize {
        self.program.len()
    }

    /// Returns a boolean representing if the program contains 0 instructions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a tuple representation of all composing parts of the program.
    pub fn into_raw_parts(self) -> (FastForward, Vec<CharacterSet>, Vec<Instruction>) {
        self.into()
    }

    /// Produces a program from its consituent parts.
    pub fn from_raw_parts(
        fast_forward: FastForward,
        sets: Vec<CharacterSet>,
        program: Vec<Instruction>,
    ) -> Self {
        Self {
            sets,
            program,
            fast_forward,
        }
    }
}

impl Display for Instructions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inst in self.program.iter() {
            writeln!(f, "{:04}: {}", inst.offset, inst.opcode)?
        }

        Ok(())
    }
}

impl std::ops::Index<InstIndex> for Instructions {
    type Output = Opcode;

    fn index(&self, index: InstIndex) -> &Self::Output {
        let idx = index.as_usize();
        &self.program[idx].opcode
    }
}

impl std::ops::IndexMut<InstIndex> for Instructions {
    fn index_mut(&mut self, index: InstIndex) -> &mut Self::Output {
        let idx = index.as_usize();
        &mut self.program[idx].opcode
    }
}

impl AsRef<[Instruction]> for Instructions {
    fn as_ref(&self) -> &[Instruction] {
        &self.program
    }
}

impl From<Instructions> for (FastForward, Vec<CharacterSet>, Vec<Instruction>) {
    fn from(insts: Instructions) -> Self {
        (insts.fast_forward, insts.sets, insts.program)
    }
}

/// A wrapper-type providing a program counter into the runtime program.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstIndex(u32);

impl InstIndex {
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for InstIndex {
    fn from(ptr: u32) -> Self {
        Self(ptr)
    }
}

impl std::ops::Add<u32> for InstIndex {
    type Output = Self;

    fn add(self, rhs: u32) -> Self::Output {
        let new_ptr = self.0 + rhs;

        InstIndex::from(new_ptr)
    }
}

impl std::ops::Add<Self> for InstIndex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let new_ptr = self.0 + rhs.0;

        InstIndex::from(new_ptr)
    }
}

/// A wrapper type for the binary representation of an opcode in little-endian
/// format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpcodeBytecodeRepr(pub [u8; 16]);

/// A runtime instruction, containing an offset and a corresponding instruction.
#[derive(Debug, PartialEq)]
pub struct Instruction {
    // A unique identifier for a given instruction
    pub offset: usize,
    pub opcode: Opcode,
}

impl Instruction {
    /// Instantiates a new instruction from its constituent parts.
    #[must_use]
    pub fn new(offset: usize, opcode: Opcode) -> Self {
        Self { offset, opcode }
    }

    /// Returns the Instruction's enclosed Opcode.
    pub fn into_opcode(self) -> Opcode {
        self.opcode
    }

    /// Returns a tuple representation of all composing parts of the instruction.
    pub fn into_raw_parts(self) -> (usize, Opcode) {
        self.into()
    }

    /// Functionally equivalent to `Self::new` generating an instruction from
    /// its constituent parts.
    pub fn from_raw_parts(id: usize, opcode: Opcode) -> Self {
        Self::new(id, opcode)
    }
}

impl AsRef<Opcode> for Instruction {
    fn as_ref(&self) -> &Opcode {
        &self.opcode
    }
}

impl From<Instruction> for (usize, Opcode) {
    fn from(inst: Instruction) -> Self {
        (inst.offset, inst.opcode)
    }
}

impl From<(usize, Opcode)> for Instruction {
    fn from((id, opcode): (usize, Opcode)) -> Self {
        Self::from_raw_parts(id, opcode)
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04}: {}", self.offset, self.opcode)
    }
}

/// An enum representation of the runtime's operation set.
#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    /// A consume operation, matching any character.
    Any,
    /// A consume operation, matching an explicit character.
    Consume(InstConsume),
    /// A consume operation, matching a character if it falls within the
    /// bounds of a predefined set.
    ConsumeSet(InstConsumeSet),
    /// A non-consuming epsilon transition. Examples of which are
    /// lookahead/lookback operation.
    Epsilon(InstEpsilon),
    /// A non-consuming branch operation. This forks off two threads at the
    /// enclosed offsets.
    Split(InstSplit),
    /// A non-consuming jump operation. This instruction modifies the program
    /// counter in place for the current evaluating thread.
    Jmp(InstJmp),
    /// A non-consuming operation that signifies the start of an enclosed save
    /// group.
    StartSave(InstStartSave),
    /// A non-consuming operation that signifies the end of an enclosed save
    /// group.
    EndSave(InstEndSave),
    /// A non-consuming operation that signifies a match.
    Match,
    /// A non-consuming operation for setting execution metadata.
    Meta(InstMeta),
}

impl Opcode {
    /// Returns true if the opcode represents an input consuming operations.
    #[allow(unused)]
    pub fn is_consuming(&self) -> bool {
        matches!(
            self,
            Opcode::Any | Opcode::Consume(_) | Opcode::ConsumeSet(_)
        )
    }

    /// Returns true if the opcode requires lookahead for evaluation.
    #[allow(unused)]
    pub fn requires_lookahead(&self) -> bool {
        matches!(self, Opcode::Epsilon(InstEpsilon { .. }))
    }

    /// Returns true if the opcode represents an input consuming operations
    /// that represents an explicit value or alphabet.
    #[allow(unused)]
    pub fn is_explicit_consuming(&self) -> bool {
        matches!(self, Opcode::Consume(_) | Opcode::ConsumeSet(_))
    }
}

impl From<Instruction> for Opcode {
    fn from(inst: Instruction) -> Self {
        inst.opcode
    }
}

impl Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Opcode::Any => Display::fmt(&InstAny::new(), f),
            Opcode::Consume(i) => Display::fmt(&i, f),
            Opcode::ConsumeSet(i) => Display::fmt(&i, f),
            Opcode::Epsilon(i) => Display::fmt(&i, f),
            Opcode::Split(i) => Display::fmt(&i, f),
            Opcode::Jmp(i) => Display::fmt(&i, f),
            Opcode::StartSave(i) => Display::fmt(&i, f),
            Opcode::EndSave(i) => Display::fmt(&i, f),
            Opcode::Match => Display::fmt(&InstMatch, f),
            Opcode::Meta(i) => Display::fmt(&i, f),
        }
    }
}

/// A concrete representation of the Any Opcode.
#[derive(Debug, PartialEq)]
pub struct InstAny;

impl InstAny {
    pub const OPCODE_BINARY_REPR: u64 = 1;

    /// Instantiates a new `InstAny`.
    pub const fn new() -> Self {
        Self
    }
}

impl Default for InstAny {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for InstAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Any")
    }
}

/// A concrete representation of the Consume Opcode.
#[derive(Debug, Clone, PartialEq)]
pub struct InstConsume {
    /// An expected unicode character required to match.
    pub value: char,
}

impl InstConsume {
    pub const OPCODE_BINARY_REPR: u64 = 2;

    /// Instantiates a new `InstConsume` with the expected matching char.
    #[must_use]
    pub fn new(value: char) -> Self {
        Self { value }
    }
}

impl Display for InstConsume {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Consume: {:?}", self.value)
    }
}

/// Represents a type that can be used as a comparative character set.
trait CharacterRangeSetVerifiable {
    fn in_set(&self, value: char) -> bool;

    fn not_in_set(&self, value: char) -> bool {
        !self.in_set(value)
    }
}

impl CharacterRangeSetVerifiable for std::ops::RangeInclusive<char> {
    fn in_set(&self, value: char) -> bool {
        self.contains(&value)
    }
}

impl CharacterRangeSetVerifiable for char {
    fn in_set(&self, value: char) -> bool {
        *self == value
    }
}

impl<CRSV: CharacterRangeSetVerifiable> CharacterRangeSetVerifiable for Vec<CRSV> {
    fn in_set(&self, value: char) -> bool {
        self.iter().any(|r| r.in_set(value))
    }
}

/// Defines that a given type can be converted into a character set.
pub trait CharacterSetRepresentable: Into<CharacterSet> {}

/// Representing a runtime-dispatchable set of characters by associating a sets
/// membership to a character alphabet.
#[derive(Debug, Clone, PartialEq)]
pub struct CharacterSet {
    pub membership: SetMembership,
    pub set: CharacterAlphabet,
}

impl CharacterSet {
    pub const MAGIC_NUMBER: u16 = 0x1A1A;

    /// Instantiates a character set from its consituent parts.
    pub fn new(membership: SetMembership, set: CharacterAlphabet) -> Self {
        Self { membership, set }
    }

    /// Instantiates an inclusive character set from a passed alphabet.
    pub fn inclusive(set: CharacterAlphabet) -> Self {
        Self {
            membership: SetMembership::Inclusive,
            set,
        }
    }

    /// Instantiates an exclusive character set from a passed alphabet.
    pub fn exclusive(set: CharacterAlphabet) -> Self {
        Self {
            membership: SetMembership::Exclusive,
            set,
        }
    }

    /// Inverts a character sets membership, retaining its defined alphabet.
    pub fn invert_membership(self) -> Self {
        let Self { membership, set } = self;

        Self {
            membership: match membership {
                SetMembership::Inclusive => SetMembership::Exclusive,
                SetMembership::Exclusive => SetMembership::Inclusive,
            },
            set,
        }
    }
}

impl CharacterRangeSetVerifiable for CharacterSet {
    fn in_set(&self, value: char) -> bool {
        match &self.membership {
            SetMembership::Inclusive => self.set.in_set(value),
            SetMembership::Exclusive => self.set.not_in_set(value),
        }
    }
}

/// Represents a runtime dispatchable set of characters.
#[derive(Debug, Clone, PartialEq)]
pub enum CharacterAlphabet {
    /// Represents a range of values i.e. `0-9`, `a-z`, `A-Z`, etc...
    Range(std::ops::RangeInclusive<char>),
    /// Represents an explicitly defined set of values. i.e. `[a,b,z]`, `[1,2,7]`
    Explicit(Vec<char>),
    /// Represents a set of range of values i.e. `[0-9a-zA-Z]`,  etc...
    Ranges(Vec<std::ops::RangeInclusive<char>>),
    /// Represents a unicode category.
    UnicodeCategory(UnicodeCategory),
}

impl CharacterAlphabet {
    /// Joins a group of character sets into a single `Ranges` variant character set.
    #[deprecated = "Removing to eliminate the risk of runtime fallibility from a unicode category."]
    pub fn join(sets: Vec<Self>) -> CharacterAlphabet {
        let ranges = sets
            .into_iter()
            .flat_map(|set| match set {
                CharacterAlphabet::Range(r) => vec![r],
                CharacterAlphabet::Ranges(ranges) => ranges,
                CharacterAlphabet::Explicit(explicit_chars) => {
                    explicit_chars.into_iter().map(|c| c..=c).collect()
                }
                CharacterAlphabet::UnicodeCategory(_) => todo!(),
            })
            .collect();

        CharacterAlphabet::Ranges(ranges)
    }
}

impl CharacterRangeSetVerifiable for CharacterAlphabet {
    fn in_set(&self, value: char) -> bool {
        match self {
            CharacterAlphabet::Range(r) => r.in_set(value),
            CharacterAlphabet::Explicit(v) => v.in_set(value),
            CharacterAlphabet::Ranges(ranges) => ranges.in_set(value),
            CharacterAlphabet::UnicodeCategory(category) => category.in_set(value),
        }
    }
}

/// Represents a unicode general category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnicodeCategory {
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

impl CharacterRangeSetVerifiable for UnicodeCategory {
    #![allow(clippy::match_like_matches_macro)]
    fn in_set(&self, value: char) -> bool {
        use unicode_categories::{HumanReadableCategory, UnicodeCategorizable};

        let char_category =
            if let Some(hrc) = value.unicode_category().map(HumanReadableCategory::from) {
                hrc
            } else {
                return false;
            };

        match (self, char_category) {
            (UnicodeCategory::Letter, HumanReadableCategory::LetterLowercase)
            | (UnicodeCategory::Letter, HumanReadableCategory::LetterUppercase)
            | (UnicodeCategory::Letter, HumanReadableCategory::LetterTitlecase)
            | (UnicodeCategory::Letter, HumanReadableCategory::LetterModifier)
            | (UnicodeCategory::Letter, HumanReadableCategory::LetterOther)
            | (UnicodeCategory::LowercaseLetter, HumanReadableCategory::LetterLowercase)
            | (UnicodeCategory::UppercaseLetter, HumanReadableCategory::LetterUppercase)
            | (UnicodeCategory::TitlecaseLetter, HumanReadableCategory::LetterTitlecase)
            | (UnicodeCategory::ModifiedLetter, HumanReadableCategory::LetterModifier)
            | (UnicodeCategory::OtherLetter, HumanReadableCategory::LetterOther) => true,
            (UnicodeCategory::CasedLetter, HumanReadableCategory::LetterLowercase)
            | (UnicodeCategory::CasedLetter, HumanReadableCategory::LetterUppercase)
            | (UnicodeCategory::CasedLetter, HumanReadableCategory::LetterTitlecase) => true,
            (UnicodeCategory::Mark, HumanReadableCategory::MarkNonspacing)
            | (UnicodeCategory::Mark, HumanReadableCategory::MarkSpacingCombining)
            | (UnicodeCategory::Mark, HumanReadableCategory::MarkEnclosing)
            | (UnicodeCategory::NonSpacingMark, HumanReadableCategory::MarkNonspacing)
            | (
                UnicodeCategory::SpacingCombiningMark,
                HumanReadableCategory::MarkSpacingCombining,
            )
            | (UnicodeCategory::EnclosingMark, HumanReadableCategory::MarkEnclosing) => true,
            (UnicodeCategory::Number, HumanReadableCategory::NumberDecimalDigit)
            | (UnicodeCategory::Number, HumanReadableCategory::NumberLetter)
            | (UnicodeCategory::Number, HumanReadableCategory::NumberOther)
            | (UnicodeCategory::DecimalDigitNumber, HumanReadableCategory::NumberDecimalDigit)
            | (UnicodeCategory::LetterNumber, HumanReadableCategory::NumberLetter)
            | (UnicodeCategory::OtherNumber, HumanReadableCategory::NumberOther) => true,
            (UnicodeCategory::Separator, HumanReadableCategory::SeperatorSpace)
            | (UnicodeCategory::Separator, HumanReadableCategory::SeperatorLine)
            | (UnicodeCategory::Separator, HumanReadableCategory::SeperatorParagraph)
            | (UnicodeCategory::SpaceSeparator, HumanReadableCategory::SeperatorSpace)
            | (UnicodeCategory::LineSeparator, HumanReadableCategory::SeperatorLine)
            | (UnicodeCategory::ParagraphSeparator, HumanReadableCategory::SeperatorParagraph) => {
                true
            }
            (UnicodeCategory::Symbol, HumanReadableCategory::SymbolMath)
            | (UnicodeCategory::Symbol, HumanReadableCategory::SymbolCurrency)
            | (UnicodeCategory::Symbol, HumanReadableCategory::SymbolModifier)
            | (UnicodeCategory::Symbol, HumanReadableCategory::SymbolOther)
            | (UnicodeCategory::MathSymbol, HumanReadableCategory::SymbolMath)
            | (UnicodeCategory::CurrencySymbol, HumanReadableCategory::SymbolCurrency)
            | (UnicodeCategory::ModifierSymbol, HumanReadableCategory::SymbolModifier)
            | (UnicodeCategory::OtherSymbol, HumanReadableCategory::SymbolOther) => true,
            (UnicodeCategory::Punctuation, HumanReadableCategory::PunctuationDash)
            | (UnicodeCategory::Punctuation, HumanReadableCategory::PunctuationOpen)
            | (UnicodeCategory::Punctuation, HumanReadableCategory::PunctuationClose)
            | (UnicodeCategory::Punctuation, HumanReadableCategory::PunctuationInnerQuote)
            | (UnicodeCategory::Punctuation, HumanReadableCategory::PunctuationFinalQuote)
            | (UnicodeCategory::Punctuation, HumanReadableCategory::PunctuationConnector)
            | (UnicodeCategory::Punctuation, HumanReadableCategory::PunctuationOther)
            | (UnicodeCategory::DashPunctuation, HumanReadableCategory::PunctuationDash)
            | (UnicodeCategory::OpenPunctuation, HumanReadableCategory::PunctuationOpen)
            | (UnicodeCategory::ClosePunctuation, HumanReadableCategory::PunctuationClose)
            | (UnicodeCategory::InitialPunctuation, HumanReadableCategory::PunctuationInnerQuote)
            | (UnicodeCategory::FinalPunctuation, HumanReadableCategory::PunctuationFinalQuote)
            | (
                UnicodeCategory::ConnectorPunctuation,
                HumanReadableCategory::PunctuationConnector,
            )
            | (UnicodeCategory::OtherPunctuation, HumanReadableCategory::PunctuationOther) => true,
            (UnicodeCategory::Other, HumanReadableCategory::OtherControl)
            | (UnicodeCategory::Other, HumanReadableCategory::OtherFormat)
            | (UnicodeCategory::Other, HumanReadableCategory::OtherPrivateUse)
            | (UnicodeCategory::Other, HumanReadableCategory::OtherSurrogate)
            | (UnicodeCategory::Other, HumanReadableCategory::OtherNotAssigned)
            | (UnicodeCategory::Control, HumanReadableCategory::OtherControl)
            | (UnicodeCategory::Format, HumanReadableCategory::OtherFormat)
            | (UnicodeCategory::PrivateUse, HumanReadableCategory::OtherPrivateUse)
            | (UnicodeCategory::Surrogate, HumanReadableCategory::OtherSurrogate)
            | (UnicodeCategory::Unassigned, HumanReadableCategory::OtherNotAssigned) => true,

            // All others do not match
            _ => false,
        }
    }
}

/// Denotes whether a given set is inclusive or exclusive to a match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetMembership {
    /// States that a set is inclusive of a value, i.e. the value is a member of
    /// the set.
    Inclusive,
    /// States that a set is exclusive of a value, i.e. the value is not a
    /// member of the set.
    Exclusive,
}

/// ConsumeSet provides richer matching patterns than the more constrained
/// Consume or Any instructions allowing for the matching from a set of
/// characters. This functions as a brevity tool to prevent long alternations.
#[derive(Debug, Clone, PartialEq)]
pub struct InstConsumeSet {
    /// A offset id to a predefined set in the current program.
    pub idx: usize,
}

impl InstConsumeSet {
    pub const OPCODE_BINARY_REPR: u64 = 3;

    /// Instantiate a new `InstConsumeSet` with a passed index.
    pub fn new(idx: usize) -> Self {
        Self::member_of(idx)
    }

    /// An alias method to `new`
    pub fn member_of(idx: usize) -> Self {
        Self { idx }
    }
}

impl Display for InstConsumeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConsumeSet: {{{:04}}}", self.idx)
    }
}

/// A condition in which an Epsilon transition boundary can be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsilonCond {
    WordBoundary,
    NonWordBoundary,
    StartOfStringOnly,
    EndOfStringOnlyNonNewline,
    EndOfStringOnly,
    PreviousMatchEnd,
    EndOfString,
}

/// An internal representation of an `Epsilon` opcode.
#[derive(Debug, Clone, PartialEq)]
pub struct InstEpsilon {
    pub cond: EpsilonCond,
}

impl InstEpsilon {
    pub const OPCODE_BINARY_REPR: u64 = 4;

    /// Instantiates a new `InstEpsilon` from a passed condition.
    pub fn new(cond: EpsilonCond) -> Self {
        Self { cond }
    }
}

impl Display for InstEpsilon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cond = match self.cond {
            EpsilonCond::WordBoundary => "Word Boundary",
            EpsilonCond::NonWordBoundary => "Non-Word Boundary",
            EpsilonCond::StartOfStringOnly => "Start of String Only",
            EpsilonCond::EndOfStringOnlyNonNewline => "End of String Only Non-Newline",
            EpsilonCond::EndOfStringOnly => "End of String Only",
            EpsilonCond::PreviousMatchEnd => "Previous Match End",
            EpsilonCond::EndOfString => "End of String",
        };

        write!(f, "Epsilon: {{{}}}", cond)
    }
}

/// An internal representation of the `Split` opcode.
#[derive(Debug, Clone, PartialEq)]
pub struct InstSplit {
    pub x_branch: InstIndex,
    pub y_branch: InstIndex,
}

impl InstSplit {
    pub const OPCODE_BINARY_REPR: u64 = 5;

    /// Instantiates a new `InstSplit` from two branching program counter
    /// values.
    #[must_use]
    pub fn new(x: InstIndex, y: InstIndex) -> Self {
        Self {
            x_branch: x,
            y_branch: y,
        }
    }
}

impl Display for InstSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Split: ({:04}), ({:04})",
            self.x_branch.as_u32(),
            self.y_branch.as_u32()
        )
    }
}

/// An internal representation of the `Jmp` opcode.
#[derive(Debug, Clone, PartialEq)]
pub struct InstJmp {
    pub next: InstIndex,
}

impl InstJmp {
    pub const OPCODE_BINARY_REPR: u64 = 6;

    /// Instnatiates a new `InstJump` from a past program counter value.
    pub fn new(next: InstIndex) -> Self {
        Self { next }
    }
}

impl Display for InstJmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JumpAbs: ({:04})", self.next.as_u32())
    }
}

/// An internal representation of the `StartSave` opcode.
#[derive(Debug, Clone, PartialEq)]
pub struct InstStartSave {
    pub slot_id: usize,
}

impl InstStartSave {
    pub const OPCODE_BINARY_REPR: u64 = 7;

    /// Instantiates a new `InstStartSave` from a passed slot id.
    #[must_use]
    pub fn new(slot_id: usize) -> Self {
        Self { slot_id }
    }
}

impl Display for InstStartSave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StartSave[{:04}]", self.slot_id,)
    }
}

/// An internal representation of the `EndSave` opcode.
#[derive(Debug, Clone, PartialEq)]
pub struct InstEndSave {
    pub slot_id: usize,
}

impl InstEndSave {
    pub const OPCODE_BINARY_REPR: u64 = 8;

    /// Instantiates a new `InstEndSave` from a passed slot id.
    #[must_use]
    pub fn new(slot_id: usize) -> Self {
        Self { slot_id }
    }
}

impl Display for InstEndSave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EndSave[{:04}]", self.slot_id,)
    }
}

/// An internal representation of the `Match` opcode.
#[derive(Debug, PartialEq)]
pub struct InstMatch;

impl InstMatch {
    pub const OPCODE_BINARY_REPR: u64 = 9;
}

impl Display for InstMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Match",)
    }
}

/// An internal representation of the `Meta` opcode.
#[derive(Debug, Clone, PartialEq)]
pub struct InstMeta(pub MetaKind);

impl InstMeta {
    pub const OPCODE_BINARY_REPR: u64 = 10;

    pub fn new(kind: MetaKind) -> Self {
        Self(kind)
    }
}

impl Display for InstMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            MetaKind::SetExpressionId(id) => write!(f, "Meta[SetExprID: ({})]", id),
        }
    }
}

/// Represents the kind of Metadata operation to trigger.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaKind {
    /// Sets expression id on a thread. This _ONLY_ comes into play in
    /// multi-expression runs.
    SetExpressionId(u32),
}

/// A Window storing both the previous and next value of an input.
#[derive(Debug, Clone, Copy)]
struct Window<T: Copy> {
    buffer: [Option<T>; 3],
}

impl<T: Copy> Window<T> {
    fn new(buffer: [Option<T>; 3]) -> Self {
        Self { buffer }
    }

    fn to_array(self) -> [Option<T>; 3] {
        self.buffer
    }

    fn previous(&self) -> Option<T> {
        self.buffer[0]
    }

    fn current(&self) -> Option<T> {
        self.buffer[1]
    }

    fn next(&self) -> Option<T> {
        self.buffer[2]
    }

    fn push(&mut self, next: Option<T>) {
        // replace the item that will rotate around.
        self.buffer.as_mut()[0] = next;
        self.buffer.as_mut().rotate_left(1);
    }
}

impl<T: Copy> AsRef<[Option<T>]> for Window<T> {
    fn as_ref(&self) -> &[Option<T>] {
        self.buffer.as_slice()
    }
}

impl<T: Copy> AsMut<[Option<T>]> for Window<T> {
    fn as_mut(&mut self) -> &mut [Option<T>] {
        self.buffer.as_mut_slice()
    }
}

impl<T: Copy> From<Window<T>> for [Option<T>; 3] {
    fn from(window: Window<T>) -> Self {
        window.to_array()
    }
}

impl<T: Copy> Default for Window<T> {
    fn default() -> Self {
        Self {
            buffer: Default::default(),
        }
    }
}

/// A buffered iterator that provides a lookahead and lookback for a given input.
struct CharsWithLookAheadAndLookBack<I>
where
    I: Iterator<Item = char>,
{
    buffer: Window<char>,
    iter: I,
}

impl<I> CharsWithLookAheadAndLookBack<I>
where
    I: Iterator<Item = char>,
{
    fn new(mut iter: I) -> Self {
        let lookahead = match iter.next() {
            Some(c) => {
                let first = Some(c);
                [None, None, first]
            }
            None => [None, None, None],
        };

        let buffer = Window::new(lookahead);

        Self { buffer, iter }
    }
}

impl<I> Iterator for CharsWithLookAheadAndLookBack<I>
where
    I: Iterator<Item = char>,
{
    type Item = Window<char>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.push(self.iter.next());

        if self.buffer.current().is_some() {
            Some(self.buffer)
        } else {
            None
        }
    }
}

fn add_thread<const SG: usize>(
    program: &[Instruction],
    save_groups: &mut [SaveGroupSlot; SG],
    mut thread_list: Threads<SG>,
    t: Thread<SG>,
    sp: usize,
    window: [Option<char>; 3],
) -> Threads<SG> {
    let inst_idx = t.inst;
    let default_next_inst_idx = inst_idx + 1;

    // Don't visit states we've already added.
    let inst = match program.get(inst_idx.as_usize()) {
        // if the thread is already defined, return.
        Some(inst) if thread_list.gen.contains(&inst.offset) => return thread_list,
        // if it's the end of the program without a match instruction, return.
        None => return thread_list,
        // Otherwise add the new thread.
        Some(inst) => {
            thread_list.gen.insert(inst.offset);
            inst
        }
    };

    let opcode = &inst.opcode;
    let [lookback, current_char, lookahead] = window;
    match opcode {
        Opcode::Split(InstSplit { x_branch, y_branch }) => {
            let x = *x_branch;
            let y = *y_branch;
            thread_list = add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new_with_id(t.expression_id, t.save_groups, x),
                sp,
                window,
            );

            add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new_with_id(t.expression_id, t.save_groups, y),
                sp,
                window,
            )
        }
        Opcode::Jmp(InstJmp { next }) => add_thread(
            program,
            save_groups,
            thread_list,
            Thread::new_with_id(t.expression_id, t.save_groups, *next),
            sp,
            window,
        ),
        Opcode::StartSave(InstStartSave { slot_id }) => {
            let mut groups = t.save_groups;
            groups[*slot_id] = SaveGroup::allocated(t.expression_id, *slot_id);

            add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new_with_id(t.expression_id, groups, default_next_inst_idx),
                sp,
                window,
            )
        }
        Opcode::EndSave(InstEndSave { slot_id }) => {
            let closed_save = match t.save_groups.get(*slot_id) {
                Some(SaveGroup::Open {
                    expression_id,
                    slot_id,
                    start,
                }) => SaveGroup::complete(*expression_id, *slot_id, *start, sp),

                // if the save group is not open, return none.
                Some(sg) => *sg,
                None => panic!("index out of range"),
            };

            let thread_save_group_slot = SaveGroupSlot::from(closed_save);

            match (save_groups.get(*slot_id), thread_save_group_slot) {
                // Save a valid match.
                (Some(SaveGroupSlot::None), thread_save_group_slot) => {
                    save_groups[*slot_id] = thread_save_group_slot;
                }

                // if the match is a better match from the same root, choose it.
                (
                    Some(SaveGroupSlot::Complete {
                        start: global_start,
                        ..
                    }),
                    SaveGroupSlot::Complete { start: t_start, .. },
                ) if t_start == *global_start => {
                    save_groups[*slot_id] = thread_save_group_slot;
                }

                // already matched, do nothing.
                (Some(SaveGroupSlot::Complete { .. }), SaveGroupSlot::Complete { .. }) => (),

                // save group slot will guaranteed to be `Complete` from the
                // above check.
                (Some(_), SaveGroupSlot::None) => unreachable!(),
                (None, _) => panic!("save group slot out of range."),
            };

            let mut thread_save_group = t.save_groups;
            thread_save_group[*slot_id] = closed_save;

            add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new_with_id(t.expression_id, thread_save_group, default_next_inst_idx),
                sp,
                window,
            )
        }

        // cover empty initial-state
        Opcode::Epsilon(InstEpsilon { .. }) if window == [None, None, None] => add_thread(
            program,
            save_groups,
            thread_list,
            Thread::new_with_id(t.expression_id, t.save_groups, default_next_inst_idx),
            sp,
            window,
        ),

        Opcode::Epsilon(InstEpsilon { cond })
            if matches!(cond, EpsilonCond::WordBoundary)
                || matches!(cond, EpsilonCond::NonWordBoundary) =>
        {
            {
                let lookback_is_whitespace = lookback.map(|c| c.is_whitespace()).unwrap_or(true);
                let current_is_whitespace = current_char.map(|c| c.is_whitespace()).unwrap_or(true);

                // Place is a boundary if both lookback and current are either
                // both whitespace or both chars.
                let is_boundary = current_is_whitespace ^ lookback_is_whitespace;

                match (cond, is_boundary) {
                    (EpsilonCond::WordBoundary, true) | (EpsilonCond::NonWordBoundary, false) => {
                        add_thread(
                            program,
                            save_groups,
                            thread_list,
                            Thread::new_with_id(
                                t.expression_id,
                                t.save_groups,
                                default_next_inst_idx,
                            ),
                            sp,
                            window,
                        )
                    }
                    (EpsilonCond::WordBoundary, false) | (EpsilonCond::NonWordBoundary, true) => {
                        thread_list
                    }

                    // due to the above guard, no other cases should be encounterable
                    _ => unreachable!(),
                }
            }
        }

        Opcode::Epsilon(InstEpsilon {
            cond: EpsilonCond::StartOfStringOnly,
        }) => {
            let start_of_string = sp == 0 && lookback.is_none();

            if start_of_string {
                add_thread(
                    program,
                    save_groups,
                    thread_list,
                    Thread::new_with_id(t.expression_id, t.save_groups, default_next_inst_idx),
                    sp,
                    window,
                )
            } else {
                thread_list
            }
        }

        Opcode::Epsilon(InstEpsilon { cond })
            if matches!(cond, EpsilonCond::EndOfString)
                || matches!(cond, EpsilonCond::EndOfStringOnly)
                || matches!(cond, EpsilonCond::EndOfStringOnlyNonNewline) =>
        {
            let end_of_input = match cond {
                EpsilonCond::EndOfStringOnlyNonNewline => current_char.is_none(),
                EpsilonCond::EndOfStringOnly | EpsilonCond::EndOfString => {
                    current_char.is_none() || (current_char == Some('\n') && lookahead.is_none())
                }
                _ => unreachable!(),
            };

            if end_of_input {
                add_thread(
                    program,
                    save_groups,
                    thread_list,
                    Thread::new_with_id(t.expression_id, t.save_groups, default_next_inst_idx),
                    sp,
                    window,
                )
            } else {
                thread_list
            }
        }

        Opcode::Epsilon(InstEpsilon {
            cond: EpsilonCond::PreviousMatchEnd,
        }) => {
            // If the end of any matches is the current SP, return true.
            let is_end_of_last_match = t
                .save_groups
                .iter()
                .filter_map(|s| match s {
                    SaveGroup::Complete { end, .. } => Some(*end),
                    _ => None,
                })
                .any(|end_pointer| end_pointer == sp);

            if is_end_of_last_match {
                add_thread(
                    program,
                    save_groups,
                    thread_list,
                    Thread::new_with_id(t.expression_id, t.save_groups, default_next_inst_idx),
                    sp,
                    window,
                )
            } else {
                thread_list
            }
        }

        // Meta instructions
        Opcode::Meta(InstMeta(MetaKind::SetExpressionId(expr_id))) => add_thread(
            program,
            save_groups,
            thread_list,
            Thread::new_with_id(*expr_id, t.save_groups, default_next_inst_idx),
            sp,
            window,
        ),

        // catch all for threads that can't be live-evaluated.
        _ => {
            thread_list.threads.push(t);
            thread_list
        }
    }
}

/// Executes a given program against an input. If a match is found an
/// `Optional` vector of savegroups is returned. A match occurs if all
/// savegroup slots are marked complete and pattern match is found.
pub fn run<const SG: usize>(program: &Instructions, input: &str) -> Option<[SaveGroupSlot; SG]> {
    use core::mem::swap;

    let mut input_idx = 0;
    let mut input_iter = CharsWithLookAheadAndLookBack::new(input.chars())
        .enumerate()
        .skip_while(
            |(_, window)| match (window.current(), &program.fast_forward) {
                (None, _) | (_, FastForward::None) => false,
                (Some(c), FastForward::Char(first_match)) => c != *first_match,
                (Some(c), FastForward::Set(set_idx)) => program.sets[*set_idx].not_in_set(c),
            },
        );

    let sets = &program.sets;
    let instructions = program.as_ref();

    let program_len = instructions.len();
    let mut current_thread_list = Threads::with_set_size(program_len);
    let mut next_thread_list = Threads::with_set_size(program_len);

    // a running tracker of found matches
    let mut matches = 0;
    let mut sub = [SaveGroupSlot::None; SG];

    // input and eoi tracker.
    let (mut done, mut current_window) = match input_iter.next() {
        Some((idx, window)) => {
            let lookback = window.previous();
            let current = window.current();
            let lookahead = window.next();

            input_idx = idx;
            (false, [lookback, current, lookahead])
        }
        None => (true, [None, None, None]),
    };

    // add the initial thread
    current_thread_list = add_thread(
        instructions,
        &mut sub,
        current_thread_list,
        Thread::new([SaveGroup::None; SG], InstIndex::from(0)),
        0,
        current_window,
    );

    'outer: while !done && !current_thread_list.threads.is_empty() {
        let next_char = current_window[1];
        done = next_char.is_none();
        let (next_input_idx, next_window) = match input_iter.next() {
            Some((idx, window)) => {
                let lookback = window.previous();
                let current = window.current();
                let lookahead = window.next();

                (idx, [lookback, current, lookahead])
            }
            None => (input_idx + 1, [None, None, None]),
        };

        for thread in current_thread_list.threads.iter() {
            let thread_save_groups = thread.save_groups;
            let inst_idx = thread.inst;
            let default_next_inst_idx = inst_idx + 1;
            let opcode = instructions.get(inst_idx.as_usize()).map(|i| &i.opcode);

            match opcode {
                Some(Opcode::Any) if next_char.is_none() => {
                    break;
                }
                Some(Opcode::Any) => {
                    let thread_local_save_group = thread_save_groups;

                    next_thread_list = add_thread(
                        instructions,
                        &mut sub,
                        next_thread_list,
                        Thread::new_with_id(
                            thread.expression_id,
                            thread_local_save_group,
                            default_next_inst_idx,
                        ),
                        next_input_idx,
                        next_window,
                    );
                }

                Some(Opcode::Consume(InstConsume { value })) if Some(*value) == next_char => {
                    let mut thread_local_save_group = thread_save_groups;

                    for thr in thread_local_save_group.iter_mut() {
                        if let SaveGroup::Allocated {
                            expression_id,
                            slot_id,
                        } = thr
                        {
                            *thr = SaveGroup::open(*expression_id, *slot_id, input_idx);
                        }
                    }

                    next_thread_list = add_thread(
                        instructions,
                        &mut sub,
                        next_thread_list,
                        Thread::new(thread_local_save_group, default_next_inst_idx)
                            .with_expression_id(thread.expression_id),
                        next_input_idx,
                        next_window,
                    );
                }

                Some(Opcode::ConsumeSet(InstConsumeSet { idx: set_idx }))
                    if next_char.map_or(false, |c| {
                        sets.get(*set_idx).map_or(false, |set| set.in_set(c))
                    }) =>
                {
                    let mut thread_local_save_group = thread_save_groups;
                    for thr in thread_local_save_group.iter_mut() {
                        if let SaveGroup::Allocated {
                            expression_id,
                            slot_id,
                        } = thr
                        {
                            *thr = SaveGroup::open(*expression_id, *slot_id, input_idx);
                        }
                    }

                    next_thread_list = add_thread(
                        instructions,
                        &mut sub,
                        next_thread_list,
                        Thread::<SG>::new_with_id(
                            thread.expression_id,
                            thread_local_save_group,
                            default_next_inst_idx,
                        ),
                        next_input_idx,
                        next_window,
                    );
                }

                Some(Opcode::Match) => {
                    matches += 1;
                    continue;
                }
                None => {
                    break 'outer;
                }
                _ => continue,
            }
        }

        current_window = next_window;
        input_idx = next_input_idx;
        swap(&mut current_thread_list, &mut next_thread_list);
        next_thread_list.threads.clear();
        next_thread_list.gen.clear();
    }

    // Signifies all savegroups are satisfied
    let all_complete = sub.iter().all(|sg| sg.is_complete());
    if matches > 0 && all_complete {
        Some(sub)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_evaluate_simple_linear_match_expression() {
        let progs = vec![
            (
                Some([SaveGroupSlot::complete(0, 0, 1)]),
                Instructions::default().with_opcodes(vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ]),
            ),
            (
                None,
                Instructions::default().with_opcodes(vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('b')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ]),
            ),
        ];

        let input = "aab";

        for (expected_res, prog) in progs {
            let res = run::<1>(&prog, input);
            assert_eq!(expected_res, res)
        }
    }

    #[test]
    fn should_evaluate_alternation_expression() {
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Split(InstSplit::new(InstIndex::from(2), InstIndex::from(4))),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Jmp(InstJmp::new(InstIndex::from(5))),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);
        let input_output = vec![
            ("a", Some([SaveGroupSlot::complete(0, 0, 1)])),
            ("b", Some([SaveGroupSlot::complete(0, 0, 1)])),
            ("ab", Some([SaveGroupSlot::complete(0, 0, 1)])),
            ("ba", Some([SaveGroupSlot::complete(0, 0, 1)])),
            ("c", None),
        ];

        for (test_id, (input, expected_res)) in input_output.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_evaluate_set_match_expression() {
        let progs = vec![
            (
                Some([SaveGroupSlot::complete(0, 0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(1))),
            (
                Some([SaveGroupSlot::complete(0, 0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(3)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(2))),
            (
                Some([SaveGroupSlot::complete(0, 0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(4)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(5))),
            (
                Some([SaveGroupSlot::complete(0, 0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(7)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(6))),
            (
                Some([SaveGroupSlot::complete(0, 0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(8)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(9))),
        ];

        let input = "aab";

        for (test_id, (expected_res, consume_set_inst)) in progs.into_iter().enumerate() {
            let prog = Instructions::default()
                .with_sets(vec![
                    CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
                    CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
                    CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['x', 'y', 'z'])),
                    CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['x', 'y', 'z'])),
                    CharacterSet::inclusive(CharacterAlphabet::Range('a'..='z')),
                    CharacterSet::exclusive(CharacterAlphabet::Range('a'..='z')),
                    CharacterSet::inclusive(CharacterAlphabet::Range('x'..='z')),
                    CharacterSet::exclusive(CharacterAlphabet::Range('x'..='z')),
                    CharacterSet::inclusive(CharacterAlphabet::UnicodeCategory(
                        UnicodeCategory::Letter,
                    )),
                    CharacterSet::exclusive(CharacterAlphabet::UnicodeCategory(
                        UnicodeCategory::Letter,
                    )),
                ])
                .with_opcodes(vec![
                    Opcode::StartSave(InstStartSave::new(0)),
                    consume_set_inst,
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ]);
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_evaluate_eager_character_class_zero_or_one_expression() {
        let tests = vec![
            (None, "aab"),
            (Some([SaveGroupSlot::complete(0, 0, 1)]), "1ab"),
            (Some([SaveGroupSlot::complete(0, 0, 2)]), "123"),
        ];

        // `^\d\d?` | `^[0-9][0-9]?`
        let prog = Instructions::default()
            .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Range(
                '0'..='9',
            ))])
            .with_opcodes(vec![
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(4))),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, expected_res), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_eager_character_class_zero_or_more_expression() {
        let tests = vec![
            (None, "aab"),
            (Some([SaveGroupSlot::complete(0, 0, 1)]), "1ab"),
            (Some([SaveGroupSlot::complete(0, 0, 3)]), "123"),
        ];

        // `^\d\d*` | `^[0-9][0-9]*`
        let prog = Instructions::default()
            .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Range(
                '0'..='9',
            ))])
            .with_opcodes(vec![
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(5))),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, expected_res), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_eager_character_class_one_or_more_expression() {
        let tests = vec![
            (None, "aab"),
            (Some([SaveGroupSlot::complete(0, 0, 1)]), "1ab"),
            (Some([SaveGroupSlot::complete(0, 0, 3)]), "123"),
        ];

        // `^\d+` | `^[0-9]+`
        let prog = Instructions::default()
            .with_sets(vec![CharacterSet::inclusive(CharacterAlphabet::Range(
                '0'..='9',
            ))])
            .with_opcodes(vec![
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(5))),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, expected_res), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_eager_unicode_category_one_or_more_expression() {
        let tests = vec![
            (None, "123"),
            (Some([SaveGroupSlot::complete(0, 0, 1)]), "a12"),
            (Some([SaveGroupSlot::complete(0, 0, 3)]), "aab"),
        ];

        // `^\p{Letter}+`
        let prog = Instructions::default()
            .with_sets(vec![CharacterSet::inclusive(
                CharacterAlphabet::UnicodeCategory(UnicodeCategory::Letter),
            )])
            .with_opcodes(vec![
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(5))),
                Opcode::ConsumeSet(InstConsumeSet::new(0)),
                Opcode::Jmp(InstJmp::new(InstIndex::from(2))),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, expected_res), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_consecutive_diverging_match_expression() {
        let progs = vec![
            (
                [SaveGroupSlot::complete(0, 0, 2)],
                Instructions::default().with_opcodes(vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::Any,
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ]),
            ),
            (
                [SaveGroupSlot::complete(0, 1, 3)],
                Instructions::default().with_opcodes(vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::Any,
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('a')),
                    Opcode::Consume(InstConsume::new('b')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ]),
            ),
        ];

        let input = "aab";

        for (test_num, (expected_res, prog)) in progs.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_num, Some(expected_res)), (test_num, res))
        }
    }

    #[test]
    fn should_match_first_match_in_unanchored_expression() {
        let (save_group, prog) = (
            [SaveGroupSlot::complete(0, 0, 2)],
            Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ]),
        );

        let input = "aaaab";

        let res = run::<1>(&prog, input);
        assert_eq!(Some(save_group), res)
    }

    #[test]
    fn should_evaluate_multiple_save_groups_expression() {
        // (aa)(b)
        let (expected_res, prog) = (
            [
                SaveGroupSlot::complete(0, 0, 2),
                SaveGroupSlot::complete(0, 2, 3),
            ],
            Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::StartSave(InstStartSave::new(1)),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::EndSave(InstEndSave::new(1)),
                Opcode::Match,
            ]),
        );

        let input = "aab";

        let res = run::<2>(&prog, input);
        assert_eq!(Some(expected_res), res)
    }

    #[test]
    fn should_evaluate_eager_match_zero_or_one_expression() {
        let tests = vec![
            ([SaveGroupSlot::complete(0, 0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 0, 3)], "aaab"),
            ([SaveGroupSlot::complete(0, 0, 3)], "aaaab"),
        ];

        // `^(aaa?)`
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(5))),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            Opcode::Match,
        ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, Some(expected_res)), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_lazy_match_zero_or_one_expression() {
        let tests = vec![
            (None, "aab"),
            (Some([SaveGroupSlot::complete(0, 0, 3)]), "aaab"),
            (Some([SaveGroupSlot::complete(0, 0, 4)]), "aaaab"),
        ];

        // `^(aaa??)`
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Split(InstSplit::new(InstIndex::from(5), InstIndex::from(4))),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            Opcode::Match,
        ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, expected_res), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_eager_match_exact_quantifier_expression() {
        let tests = vec![
            ([SaveGroupSlot::complete(0, 0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 0, 2)], "aaab"),
        ];

        let prog = Instructions::new(
            vec![],
            vec![
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ],
        );

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, Some(expected_res)), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_eager_match_atleast_quantifier_expression() {
        let tests = vec![
            ([SaveGroupSlot::complete(0, 0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 0, 3)], "aaab"),
        ];

        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(6))),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Jmp(InstJmp::new(InstIndex::from(3))),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, Some(expected_res)), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_eager_match_between_quantifier_expression() {
        let tests = vec![
            ([SaveGroupSlot::complete(0, 0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 0, 3)], "aaab"),
            ([SaveGroupSlot::complete(0, 0, 4)], "aaaab"),
        ];

        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(5))),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Split(InstSplit::new(InstIndex::from(6), InstIndex::from(7))),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
            Opcode::Match,
        ]);

        for (case_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((case_id, Some(expected_res)), (case_id, res));
        }
    }

    #[test]
    fn should_evaluate_nested_group_expression() {
        // ^(a(b))
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::StartSave(InstStartSave::new(1)),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::EndSave(InstEndSave::new(1)),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        let res = run::<2>(&prog, "ab");
        assert_eq!(
            Some([
                SaveGroupSlot::complete(0, 0, 2),
                SaveGroupSlot::complete(0, 1, 2),
            ]),
            res
        );
    }

    #[test]
    fn should_evaluate_nested_quantified_group_expression() {
        // ^(a(b){2,})
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::StartSave(InstStartSave::new(1)),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::Split(InstSplit::new(InstIndex::from(6), InstIndex::from(8))),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::Jmp(InstJmp::new(InstIndex::from(5))),
            Opcode::EndSave(InstEndSave::new(1)),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        let res = run::<2>(&prog, "abbb");
        assert_eq!(
            Some([
                SaveGroupSlot::complete(0, 0, 4),
                SaveGroupSlot::complete(0, 1, 4),
            ]),
            res
        );
    }

    #[test]
    fn should_match_start_of_string_only_anchor() {
        let tests = vec![
            (None, "caa"),
            (Some([SaveGroupSlot::complete(0, 0, 1)]), "baab"),
            (Some([SaveGroupSlot::complete(0, 0, 1)]), "aab"),
            (Some([SaveGroupSlot::complete(0, 3, 4)]), " aab"),
            (Some([SaveGroupSlot::complete(0, 0, 1)]), "baab\naab"),
        ];

        // ((?:\Aa)|(?:b))
        let prog = Instructions::default().with_opcodes(vec![
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
        ]);

        for (test_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_follow_word_boundary_epsilon_transition_on_start_of_input_or_whitespace() {
        let tests = vec![
            (None, "baab"),
            (Some([SaveGroupSlot::complete(0, 0, 2)]), "aab"),
            (Some([SaveGroupSlot::complete(0, 0, 2)]), "aaab"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), " aab"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), " aaaab"),
            (Some([SaveGroupSlot::complete(0, 5, 7)]), "baab\naab"),
        ];

        // (\baa)
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Epsilon(InstEpsilon::new(EpsilonCond::WordBoundary)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        for (test_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_follow_word_boundary_epsilon_transition_on_end_of_input_or_whitespace() {
        let tests = vec![
            (None, "baab"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), "baa"),
            (Some([SaveGroupSlot::complete(0, 2, 4)]), "baaa"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), "baa "),
            (Some([SaveGroupSlot::complete(0, 2, 4)]), "baaa "),
        ];

        // (aa\b)
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Epsilon(InstEpsilon::new(EpsilonCond::WordBoundary)),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        for (test_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_follow_nonword_boundary_on_character_wrapped_matches() {
        let tests = vec![
            (None, "aab"),
            (None, " aab"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), "baab"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), "aaab"),
            (Some([SaveGroupSlot::complete(0, 2, 4)]), " aaaab"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), "baab\naab"),
        ];

        // (\Baa)
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Epsilon(InstEpsilon::new(EpsilonCond::NonWordBoundary)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        for (test_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_follow_end_of_string_epsilon_on_end_of_input() {
        let tests = vec![
            (None, "baab"),
            (None, "baa "),
            (None, "baaa "),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), "baa"),
            (Some([SaveGroupSlot::complete(0, 2, 4)]), "baaa"),
            // match on trailing newline
            (Some([SaveGroupSlot::complete(0, 2, 4)]), "baaa\n"),
        ];

        // (aa$)
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Epsilon(InstEpsilon::new(EpsilonCond::EndOfString)),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        for (test_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_enforce_non_newline_eoi_anchor() {
        let tests = vec![
            (None, "baab"),
            (None, "baa "),
            (None, "baaa "),
            (None, "baaa\n"),
            (Some([SaveGroupSlot::complete(0, 1, 3)]), "baa"),
            (Some([SaveGroupSlot::complete(0, 2, 4)]), "baaa"),
        ];

        // (aa$)
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Epsilon(InstEpsilon::new(EpsilonCond::EndOfStringOnlyNonNewline)),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        for (test_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<1>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_follow_end_of_last_match_epsilon() {
        let tests = vec![
            (None, "aa"),
            (
                Some([
                    SaveGroupSlot::complete(0, 0, 2),
                    SaveGroupSlot::complete(0, 0, 1),
                ]),
                "ab",
            ),
        ];

        // ^((a)\Gb)
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::StartSave(InstStartSave::new(1)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(1)),
            Opcode::Epsilon(InstEpsilon::new(EpsilonCond::PreviousMatchEnd)),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        for (test_id, (expected_res, input)) in tests.into_iter().enumerate() {
            let res = run::<2>(&prog, input);
            assert_eq!((test_id, expected_res), (test_id, res))
        }
    }

    #[test]
    fn should_retain_a_fixed_opcode_size() {
        use core::mem::size_of;

        assert_eq!(16, size_of::<Opcode>())
    }

    #[test]
    fn should_print_test_instructions() {
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::Consume(InstConsume::new('a')),
            Opcode::Consume(InstConsume::new('b')),
            Opcode::Match,
            Opcode::Match,
        ]);

        assert_eq!(
            "0000: Consume: 'a'
0001: Consume: 'b'
0002: Match
0003: Match\n",
            prog.to_string()
        )
    }

    #[test]
    fn should_correctly_handle_indexing_over_unicode() {
        // (b)
        let (expected_res, prog) = (
            [SaveGroupSlot::complete(0, 1, 2)],
            Instructions::default().with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ]),
        );

        let input = "\u{00A0}b";

        let res = run::<1>(&prog, input);
        assert_eq!(Some(expected_res), res)
    }

    #[test]
    fn should_handle_fast_forwarding_match() {
        // (d)
        let (expected_res, prog) = (
            [SaveGroupSlot::complete(0, 3, 4)],
            Instructions::default()
                .with_opcodes(vec![
                    Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                    Opcode::Any,
                    Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                    Opcode::StartSave(InstStartSave::new(0)),
                    Opcode::Consume(InstConsume::new('d')),
                    Opcode::EndSave(InstEndSave::new(0)),
                    Opcode::Match,
                ])
                .with_fast_forward(FastForward::Char('d')),
        );

        let input = "abcd";

        let res = run::<1>(&prog, input);
        assert_eq!(Some(expected_res), res)
    }

    #[test]
    fn should_match_explicit_char_set_middle() {
        // [124]
        let prog = Instructions::default()
            .with_sets(vec![
                CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['1'])),
                CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['2'])),
                CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['4'])),
            ])
            .with_opcodes(vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(6))),
                Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
                Opcode::Jmp(InstJmp::new(InstIndex::from(10))),
                Opcode::Split(InstSplit::new(InstIndex::from(7), InstIndex::from(9))),
                Opcode::ConsumeSet(InstConsumeSet::member_of(1)),
                Opcode::Jmp(InstJmp::new(InstIndex::from(10))),
                Opcode::ConsumeSet(InstConsumeSet::member_of(2)),
                Opcode::Match,
            ]);

        let input = "2";

        let res = run::<0>(&prog, input);
        assert!(res.is_some())
    }

    #[test]
    fn should_allow_setting_of_expression_id_via_meta_expression() {
        // (a)
        let prog = Instructions::default().with_opcodes(vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::Meta(InstMeta(MetaKind::SetExpressionId(1))),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::Consume(InstConsume::new('a')),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Match,
        ]);

        let input = "bcdea";

        let res = run::<1>(&prog, input);
        assert_eq!(Some([SaveGroupSlot::complete(1, 4, 5)]), res)
    }

    #[test]
    fn should_evaluate_multi_expression_program() {
        let prog = Instructions::default().with_opcodes(vec![
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
        ]);

        // match first expr
        assert_eq!(
            Some([SaveGroupSlot::complete(0, 0, 1)]),
            run::<1>(&prog, "abc")
        );

        // match second expr
        assert_eq!(
            Some([SaveGroupSlot::complete(1, 5, 6)]),
            run::<1>(&prog, "defalbno")
        );

        // match third expr
        assert_eq!(
            Some([SaveGroupSlot::complete(2, 3, 4)]),
            run::<1>(&prog, "zxyc")
        );
    }

    #[test]
    fn should_evaluate_multi_expression_unanchored_program() {
        let prog = Instructions::default().with_opcodes(vec![
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
        ]);

        // match first expr
        assert_eq!(
            Some([SaveGroupSlot::complete(0, 3, 4)]),
            run::<1>(&prog, "zxya")
        );

        // match second expr
        assert_eq!(
            Some([SaveGroupSlot::complete(1, 3, 4)]),
            run::<1>(&prog, "zxyb")
        );

        // match third expr
        assert_eq!(
            Some([SaveGroupSlot::complete(2, 3, 4)]),
            run::<1>(&prog, "zxyc")
        );
    }
}
