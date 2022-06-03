use collections_ext::set::sparse::SparseSet;
use std::fmt::{Debug, Display};

/// Represents a defined match group for a pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveGroupSlot {
    None,
    Complete { start: usize, end: usize },
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
    pub const fn complete(start: usize, end: usize) -> Self {
        Self::Complete { start, end }
    }
}

impl From<SaveGroup> for SaveGroupSlot {
    fn from(src: SaveGroup) -> Self {
        match src {
            SaveGroup::None => SaveGroupSlot::None,

            SaveGroup::Allocated { .. } => SaveGroupSlot::None,
            SaveGroup::Open { .. } => SaveGroupSlot::None,
            SaveGroup::Complete { start, end, .. } => SaveGroupSlot::Complete { start, end },
        }
    }
}

impl Default for SaveGroupSlot {
    fn default() -> Self {
        SaveGroupSlot::None
    }
}

/// Represents a Save Group as tracked on an open thread
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveGroup {
    None,
    Allocated {
        slot_id: usize,
    },
    Open {
        slot_id: usize,
        start: usize,
    },
    Complete {
        slot_id: usize,
        start: usize,
        end: usize,
    },
}

impl SaveGroup {
    pub fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated { .. })
    }

    pub fn allocated(slot_id: usize) -> Self {
        Self::Allocated { slot_id }
    }

    pub fn open(slot_id: usize, start: usize) -> Self {
        Self::Open { slot_id, start }
    }

    pub fn complete(slot_id: usize, start: usize, end: usize) -> Self {
        Self::Complete {
            slot_id,
            start,
            end,
        }
    }
}

#[derive(Debug)]
pub struct Thread<const SG: usize> {
    save_groups: [SaveGroup; SG],
    inst: InstIndex,
}

impl<const SG: usize> Thread<SG> {
    pub fn new(save_groups: [SaveGroup; SG], inst: InstIndex) -> Self {
        Self { save_groups, inst }
    }
}

#[derive(Debug)]
pub struct Threads<const SG: usize> {
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

#[derive(Debug, PartialEq)]
pub enum FastForward {
    Char(char),
    Set(CharacterSet),
    None,
}

impl Default for FastForward {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct Instructions {
    pub sets: Vec<CharacterSet>,
    pub program: Vec<Instruction>,
    pub fast_forward: FastForward,
}

impl Instructions {
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

    pub fn with_instructions(self, program: Vec<Instruction>) -> Self {
        Self {
            sets: self.sets,
            program,
            fast_forward: self.fast_forward,
        }
    }

    pub fn with_sets(self, sets: Vec<CharacterSet>) -> Self {
        Self {
            sets,
            program: self.program,
            fast_forward: self.fast_forward,
        }
    }

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
            writeln!(f, "{:04}: {}", inst.id, inst.opcode)?
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

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstIndex(u32);

impl InstIndex {
    #[inline]
    fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    fn as_usize(self) -> usize {
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

#[derive(Debug, PartialEq)]
pub struct Instruction {
    // A unique identifier for a given instruction
    pub id: usize,
    pub opcode: Opcode,
}

impl Instruction {
    #[must_use]
    pub fn new(id: usize, opcode: Opcode) -> Self {
        Self { id, opcode }
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
        (inst.id, inst.opcode)
    }
}

impl From<(usize, Opcode)> for Instruction {
    fn from((id, opcode): (usize, Opcode)) -> Self {
        Self::from_raw_parts(id, opcode)
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04}: {}", self.id, self.opcode)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    Any,
    Consume(InstConsume),
    ConsumeSet(InstConsumeSet),
    Epsilon(InstEpsilon),
    Split(InstSplit),
    Jmp(InstJmp),
    StartSave(InstStartSave),
    EndSave(InstEndSave),
    Match,
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
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct InstAny;

impl InstAny {
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

#[derive(Debug, Clone, PartialEq)]
pub struct InstConsume {
    pub value: char,
}

impl InstConsume {
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
    membership: SetMembership,
    set: CharacterAlphabet,
}

impl CharacterSet {
    pub fn inclusive(set: CharacterAlphabet) -> Self {
        Self {
            membership: SetMembership::Inclusive,
            set,
        }
    }

    pub fn exclusive(set: CharacterAlphabet) -> Self {
        Self {
            membership: SetMembership::Exclusive,
            set,
        }
    }

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
}

impl CharacterAlphabet {
    /// Joins a group of character sets into a single `Ranges` variant character set.
    pub fn join(sets: Vec<Self>) -> CharacterAlphabet {
        let ranges = sets
            .into_iter()
            .flat_map(|set| match set {
                CharacterAlphabet::Range(r) => vec![r],
                CharacterAlphabet::Ranges(ranges) => ranges,
                CharacterAlphabet::Explicit(explicit_chars) => {
                    explicit_chars.into_iter().map(|c| c..=c).collect()
                }
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
    pub idx: usize,
}

impl InstConsumeSet {
    pub fn new(idx: usize) -> Self {
        Self::member_of(idx)
    }

    pub fn member_of(idx: usize) -> Self {
        Self { idx }
    }
}

impl Display for InstConsumeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConsumeSet: {{{:04}}}", self.idx)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstSplit {
    x_branch: InstIndex,
    y_branch: InstIndex,
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct InstEpsilon {
    pub cond: EpsilonCond,
}

impl InstEpsilon {
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

impl InstSplit {
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

#[derive(Debug, Clone, PartialEq)]
pub struct InstJmp {
    next: InstIndex,
}

impl InstJmp {
    pub fn new(next: InstIndex) -> Self {
        Self { next }
    }
}

impl Display for InstJmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JumpAbs: ({:04})", self.next.as_u32())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstStartSave {
    slot_id: usize,
}

impl InstStartSave {
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

#[derive(Debug, Clone, PartialEq)]
pub struct InstEndSave {
    slot_id: usize,
}

impl InstEndSave {
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

#[derive(Debug, PartialEq)]
pub struct InstMatch;

impl Display for InstMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Match",)
    }
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
        Some(inst) if thread_list.gen.contains(&inst.id) => return thread_list,
        // if it's the end of the program without a match instruction, return.
        None => return thread_list,
        // Otherwise add the new thread.
        Some(inst) => {
            thread_list.gen.insert(inst.id);
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
                Thread::new(t.save_groups, x),
                sp,
                window,
            );

            add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new(t.save_groups, y),
                sp,
                window,
            )
        }
        Opcode::Jmp(InstJmp { next }) => add_thread(
            program,
            save_groups,
            thread_list,
            Thread::new(t.save_groups, *next),
            sp,
            window,
        ),
        Opcode::StartSave(InstStartSave { slot_id }) => {
            let mut groups = t.save_groups;
            groups[*slot_id] = SaveGroup::Allocated { slot_id: *slot_id };

            add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new(groups, default_next_inst_idx),
                sp,
                window,
            )
        }
        Opcode::EndSave(InstEndSave { slot_id }) => {
            let closed_save = match t.save_groups.get(*slot_id) {
                Some(SaveGroup::Open { slot_id, start }) => SaveGroup::Complete {
                    slot_id: *slot_id,
                    start: *start,
                    end: sp,
                },

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
                Thread::new(thread_save_group, default_next_inst_idx),
                sp,
                window,
            )
        }

        // cover empty initial-state
        Opcode::Epsilon(InstEpsilon { .. }) if window == [None, None, None] => add_thread(
            program,
            save_groups,
            thread_list,
            Thread::new(t.save_groups, default_next_inst_idx),
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
                            Thread::new(t.save_groups, default_next_inst_idx),
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
                    Thread::new(t.save_groups, default_next_inst_idx),
                    sp,
                    window,
                )
            } else {
                thread_list
            }
        }

        Opcode::Epsilon(InstEpsilon {
            cond: EpsilonCond::EndOfString,
        }) => {
            let end_of_input =
                current_char.is_none() || (current_char == Some('\n') && lookahead.is_none());

            if end_of_input {
                add_thread(
                    program,
                    save_groups,
                    thread_list,
                    Thread::new(t.save_groups, default_next_inst_idx),
                    sp,
                    window,
                )
            } else {
                thread_list
            }
        }

        // catch-all todo state
        Opcode::Epsilon(InstEpsilon { .. }) => todo!(),

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
                (Some(c), FastForward::Set(first_match)) => first_match.not_in_set(c),
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
                        Thread::new(thread_local_save_group, default_next_inst_idx),
                        next_input_idx,
                        next_window,
                    );
                }

                Some(Opcode::Consume(InstConsume { value })) if Some(*value) == next_char => {
                    let mut thread_local_save_group = thread_save_groups;
                    for thr in thread_local_save_group
                        .iter_mut()
                        .filter(|t| t.is_allocated())
                    {
                        if let SaveGroup::Allocated { slot_id } = thr {
                            *thr = SaveGroup::open(*slot_id, input_idx);
                        }
                    }

                    next_thread_list = add_thread(
                        instructions,
                        &mut sub,
                        next_thread_list,
                        Thread::new(thread_local_save_group, default_next_inst_idx),
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
                    for thr in thread_local_save_group
                        .iter_mut()
                        .filter(|t| t.is_allocated())
                    {
                        if let SaveGroup::Allocated { slot_id } = thr {
                            *thr = SaveGroup::open(*slot_id, input_idx);
                        }
                    }

                    next_thread_list = add_thread(
                        instructions,
                        &mut sub,
                        next_thread_list,
                        Thread::<SG>::new(thread_local_save_group, default_next_inst_idx),
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
                Some([SaveGroupSlot::complete(0, 1)]),
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
            ("a", Some([SaveGroupSlot::complete(0, 1)])),
            ("b", Some([SaveGroupSlot::complete(0, 1)])),
            ("ab", Some([SaveGroupSlot::complete(0, 1)])),
            ("ba", Some([SaveGroupSlot::complete(0, 1)])),
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
                Some([SaveGroupSlot::complete(0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(0)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(1))),
            (
                Some([SaveGroupSlot::complete(0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(3)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(2))),
            (
                Some([SaveGroupSlot::complete(0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(4)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(5))),
            (
                Some([SaveGroupSlot::complete(0, 1)]),
                Opcode::ConsumeSet(InstConsumeSet::member_of(7)),
            ),
            (None, Opcode::ConsumeSet(InstConsumeSet::member_of(6))),
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
            (Some([SaveGroupSlot::complete(0, 1)]), "1ab"),
            (Some([SaveGroupSlot::complete(0, 2)]), "123"),
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
            (Some([SaveGroupSlot::complete(0, 1)]), "1ab"),
            (Some([SaveGroupSlot::complete(0, 3)]), "123"),
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
            (Some([SaveGroupSlot::complete(0, 1)]), "1ab"),
            (Some([SaveGroupSlot::complete(0, 3)]), "123"),
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
    fn should_evaluate_consecutive_diverging_match_expression() {
        let progs = vec![
            (
                [SaveGroupSlot::complete(0, 2)],
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
                [SaveGroupSlot::complete(1, 3)],
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
            [SaveGroupSlot::complete(0, 2)],
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
            [SaveGroupSlot::complete(0, 2), SaveGroupSlot::complete(2, 3)],
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
            ([SaveGroupSlot::complete(0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 3)], "aaab"),
            ([SaveGroupSlot::complete(0, 3)], "aaaab"),
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
            (Some([SaveGroupSlot::complete(0, 3)]), "aaab"),
            (Some([SaveGroupSlot::complete(0, 4)]), "aaaab"),
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
            ([SaveGroupSlot::complete(0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 2)], "aaab"),
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
            ([SaveGroupSlot::complete(0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 3)], "aaab"),
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
            ([SaveGroupSlot::complete(0, 2)], "aab"),
            ([SaveGroupSlot::complete(0, 3)], "aaab"),
            ([SaveGroupSlot::complete(0, 4)], "aaaab"),
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
            Some([SaveGroupSlot::complete(0, 2), SaveGroupSlot::complete(1, 2),]),
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
            Some([SaveGroupSlot::complete(0, 4), SaveGroupSlot::complete(1, 4),]),
            res
        );
    }

    #[test]
    fn should_match_start_of_string_only_anchor() {
        let tests = vec![
            (None, "caa"),
            (Some([SaveGroupSlot::complete(0, 1)]), "baab"),
            (Some([SaveGroupSlot::complete(0, 1)]), "aab"),
            (Some([SaveGroupSlot::complete(3, 4)]), " aab"),
            (Some([SaveGroupSlot::complete(0, 1)]), "baab\naab"),
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
            (Some([SaveGroupSlot::complete(0, 2)]), "aab"),
            (Some([SaveGroupSlot::complete(0, 2)]), "aaab"),
            (Some([SaveGroupSlot::complete(1, 3)]), " aab"),
            (Some([SaveGroupSlot::complete(1, 3)]), " aaaab"),
            (Some([SaveGroupSlot::complete(5, 7)]), "baab\naab"),
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
            (Some([SaveGroupSlot::complete(1, 3)]), "baa"),
            (Some([SaveGroupSlot::complete(2, 4)]), "baaa"),
            (Some([SaveGroupSlot::complete(1, 3)]), "baa "),
            (Some([SaveGroupSlot::complete(2, 4)]), "baaa "),
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
            (Some([SaveGroupSlot::complete(1, 3)]), "baab"),
            (Some([SaveGroupSlot::complete(1, 3)]), "aaab"),
            (Some([SaveGroupSlot::complete(2, 4)]), " aaaab"),
            (Some([SaveGroupSlot::complete(1, 3)]), "baab\naab"),
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
            (Some([SaveGroupSlot::complete(1, 3)]), "baa"),
            (Some([SaveGroupSlot::complete(2, 4)]), "baaa"),
            // match on trailing newline
            (Some([SaveGroupSlot::complete(2, 4)]), "baaa\n"),
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
            [SaveGroupSlot::complete(1, 2)],
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
            [SaveGroupSlot::complete(3, 4)],
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
}
