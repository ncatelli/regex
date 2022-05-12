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

#[derive(Default, Debug, PartialEq)]
pub struct Instructions {
    sets: Vec<CharacterSet>,
    program: Vec<Instruction>,
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
        }
    }

    pub fn with_instructions(self, program: Vec<Instruction>) -> Self {
        Self {
            sets: self.sets,
            program,
        }
    }

    pub fn with_sets(self, sets: Vec<CharacterSet>) -> Self {
        Self {
            sets,
            program: self.program,
        }
    }

    pub fn len(&self) -> usize {
        self.program.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    id: usize,
    opcode: Opcode,
}

impl Instruction {
    #[must_use]
    pub fn new(id: usize, opcode: Opcode) -> Self {
        Self { id, opcode }
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
    Split(InstSplit),
    Jmp(InstJmp),
    StartSave(InstStartSave),
    EndSave(InstEndSave),
    Match,
}

impl Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Opcode::Match => Display::fmt(&InstMatch, f),
            Opcode::Consume(i) => Display::fmt(&i, f),
            Opcode::ConsumeSet(i) => Display::fmt(&i, f),
            Opcode::Split(i) => Display::fmt(&i, f),
            Opcode::Any => Display::fmt(&InstAny::new(), f),
            Opcode::Jmp(i) => Display::fmt(&i, f),
            Opcode::StartSave(i) => Display::fmt(&i, f),
            Opcode::EndSave(i) => Display::fmt(&i, f),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct InstMatch;

impl Display for InstMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Match",)
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
    value: char,
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

fn get_at(input: &str, idx: usize) -> Option<char> {
    input[idx..].chars().next()
}

fn add_thread<const SG: usize>(
    program: &[Instruction],
    save_groups: &mut [SaveGroupSlot; SG],
    mut thread_list: Threads<SG>,
    t: Thread<SG>,
    sp: usize,
    input: &str,
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
                input,
            );

            add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new(t.save_groups, y),
                sp,
                input,
            )
        }
        Opcode::Jmp(InstJmp { next }) => add_thread(
            program,
            save_groups,
            thread_list,
            Thread::new(t.save_groups, *next),
            sp,
            input,
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
                input,
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
            let next = inst_idx + 1;

            save_groups[*slot_id] = SaveGroupSlot::from(closed_save);
            let mut thread_save_group = t.save_groups;
            thread_save_group[*slot_id] = closed_save;

            add_thread(
                program,
                save_groups,
                thread_list,
                Thread::new(thread_save_group, next),
                sp,
                input,
            )
        }
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

    let sets = &program.sets;
    let instructions = program.as_ref();

    let input_len = input.len();
    let program_len = instructions.len();

    let mut input_idx = 0;
    let mut current_thread_list = Threads::with_set_size(program_len);
    let mut next_thread_list = Threads::with_set_size(program_len);
    // a running tracker of found matches
    let mut matches = 0;

    let mut sub = [SaveGroupSlot::None; SG];

    current_thread_list = add_thread(
        instructions,
        &mut sub,
        current_thread_list,
        Thread::new([SaveGroup::None; SG], InstIndex::from(0)),
        input_idx,
        input,
    );

    'outer: while input_idx <= input_len {
        for thread in current_thread_list.threads.iter() {
            let thread_save_groups = thread.save_groups;
            let next_char = get_at(input, input_idx);
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
                        input_idx + 1,
                        input,
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
                        input_idx + 1,
                        input,
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
                        input_idx + 1,
                        input,
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

        input_idx += 1;
        swap(&mut current_thread_list, &mut next_thread_list);
        next_thread_list.threads.clear();
        next_thread_list.gen.clear();

        if current_thread_list.threads.is_empty() {
            break 'outer;
        }
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
}
