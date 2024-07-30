//! Provides the traits and functions for converting a runtime program to its
//! corresponding binary representation.

use regex_runtime::*;

/// Accepts a parsed AST and attempts to compile it into a runnable bytecode
/// program for use with the regex-runtime crate.
///
/// # Example
///
/// ```
/// use regex_compiler::to_binary;
/// use regex_runtime::{Instructions, Opcode};
///
/// let input = Instructions::new(vec![], vec![Opcode::Any, Opcode::Match]);
/// let expected_output = vec![
///     240, 240, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
///     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0,
///     0, 0, 0, 0, 0, 0, 0, 0, 0, 0
/// ];
///
///
/// let generated_bytecode = to_binary(&input);
/// assert_eq!(
///     Ok(expected_output),
///     generated_bytecode
/// );
/// ```
pub fn to_binary(insts: &Instructions) -> Result<Vec<u8>, String> {
    Ok(insts.to_bytecode())
}

/// Merges two arrays.
///
/// # Safety
/// Caller guarantees that the `N` parameter is EXACTLY half of `M` parameter.
fn merge_arrays<const N: usize, const M: usize>(first: [u8; N], second: [u8; N]) -> [u8; M] {
    let mut output_arr = [0; M];

    for (idx, val) in first.into_iter().chain(second.into_iter()).enumerate() {
        output_arr[idx] = val;
    }

    output_arr
}

/// Represents a conversion trait to a given opcode's binary little-endian
/// representation.
pub trait ToBytecode {
    // the bytecode representable type.
    type Output;

    fn to_bytecode(&self) -> Self::Output;
}

impl ToBytecode for Instructions {
    type Output = Vec<u8>;

    fn to_bytecode(&self) -> Self::Output {
        // header is 256-bits
        const HEADER_LEN: u32 = 32;

        let set_cnt: u32 = self
            .sets
            .len()
            .try_into()
            .expect("set count overflows 32-bit integer");
        let inst_cnt: u32 = self
            .program
            .len()
            .try_into()
            .expect("program count overflows 32-bit integer");

        let (ff_variant, ff_value) = match self.fast_forward {
            FastForward::None => (0u16, 0u32),
            FastForward::Char(c) => (1u16, c as u32),
            FastForward::Set(idx) => {
                let set_idx: u32 = idx
                    .try_into()
                    .expect("program count overflows 32-bit integer");
                (2u16, set_idx)
            }
        };

        let set_bytes: Vec<u8> = self.sets.iter().flat_map(|s| s.to_bytecode()).collect();
        let instruction_bytes = self
            .program
            .iter()
            .map(|inst| inst.to_bytecode())
            .flat_map(|or| or.0);

        let inst_offset = u32::try_from(set_bytes.len())
            .map(|len| HEADER_LEN + len)
            .expect("set bytes overflows 32-bit integer");

        let header_bytes: [u8; 2] = Self::MAGIC_NUMBER.to_le_bytes();
        let lower_32_bits: [u8; 4] = merge_arrays(header_bytes, ff_variant.to_le_bytes());
        let lower_64_bits: [u8; 8] = merge_arrays(lower_32_bits, set_cnt.to_le_bytes());
        let middle_64_bits: [u8; 8] =
            merge_arrays(inst_cnt.to_le_bytes(), inst_offset.to_le_bytes());
        let lower_128_bits: [u8; 16] = merge_arrays(lower_64_bits, middle_64_bits);

        let ff_value_or_offset_and_unused: [u8; 8] = merge_arrays(ff_value.to_le_bytes(), [0u8; 4]);
        let unused = [0u8; 8];
        let upper_128_bits: [u8; 16] = merge_arrays(ff_value_or_offset_and_unused, unused);

        let header: [u8; 32] = merge_arrays(lower_128_bits, upper_128_bits);

        header
            .into_iter()
            .chain(set_bytes)
            .chain(instruction_bytes)
            .collect()
    }
}

impl ToBytecode for Instruction {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        self.opcode.to_bytecode()
    }
}

impl ToBytecode for Opcode {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        match self {
            Opcode::Any => InstAny.to_bytecode(),
            Opcode::Consume(ic) => ic.to_bytecode(),
            Opcode::ConsumeSet(ics) => ics.to_bytecode(),
            Opcode::Epsilon(ie) => ie.to_bytecode(),
            Opcode::Split(is) => is.to_bytecode(),
            Opcode::Jmp(ij) => ij.to_bytecode(),
            Opcode::StartSave(iss) => iss.to_bytecode(),
            Opcode::EndSave(ies) => ies.to_bytecode(),
            Opcode::Match => InstMatch.to_bytecode(),
            Opcode::Meta(im) => im.to_bytecode(),
        }
    }
}

impl ToBytecode for InstAny {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = 0u64.to_le_bytes();

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstConsume {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let char_repr = self.value as u64;

        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = char_repr.to_le_bytes();

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for CharacterSet {
    type Output = Vec<u8>;

    fn to_bytecode(&self) -> Self::Output {
        /// all bytes need to be aligned on a 16-byte (128-bit) boundary.
        const ALIGNMENT: u32 = 16;

        let header = Self::MAGIC_NUMBER;
        let membership: u16 = match self.membership {
            SetMembership::Inclusive => 0u16,
            SetMembership::Exclusive => 1u16,
        };
        let (set_variant, len) = match self.set {
            CharacterAlphabet::Range(_) => (0u16, 1),
            CharacterAlphabet::Explicit(ref chars) => (1, chars.len()),
            CharacterAlphabet::Ranges(ref ranges) => (2, ranges.len()),
            CharacterAlphabet::UnicodeCategory(_) => (3, 1),
        };

        let truncated_alphabet_cnt: u32 = len
            .try_into()
            .expect("character alphabet count overflows 32-bit integer");

        let variant_and_membership = (membership << 2) | (set_variant);
        let variant_and_membership_bytes = variant_and_membership.to_le_bytes();
        let lower_32_bytes = merge_arrays(header.to_le_bytes(), variant_and_membership_bytes);
        let char_set_header: [u8; 8] =
            merge_arrays(lower_32_bytes, truncated_alphabet_cnt.to_le_bytes());

        let entries = self.set.to_bytecode();

        // the size in bytes of all the entries plus the char set header (64-bits)
        let char_set_byte_cnt = u32::try_from(entries.len())
            .map(|v| v + 8)
            // should be safe to assume this fits within 32 bytes. panic otherwise
            .expect("char_set overflows 32-bit integer");
        let align_up_to = (char_set_byte_cnt + (ALIGNMENT - 1)) & ALIGNMENT.wrapping_neg();
        // similarly, it shoudl be safe to assume this fits within a usize,
        // any case where it can't for a platform it should just panic.
        let padding = usize::try_from(align_up_to - char_set_byte_cnt)
            .expect("aligned char set overflows 32-bit integer");

        char_set_header
            .into_iter()
            .chain(entries)
            .chain([0u8].into_iter().cycle().take(padding))
            .collect()
    }
}

impl ToBytecode for CharacterAlphabet {
    type Output = Vec<u8>;

    fn to_bytecode(&self) -> Self::Output {
        match self {
            CharacterAlphabet::Range(r) => {
                let start = *(r.start()) as u32;
                let end = *(r.end()) as u32;

                let merged: [u8; 8] = merge_arrays(start.to_le_bytes(), end.to_le_bytes());
                merged.to_vec()
            }
            CharacterAlphabet::Explicit(chars) => chars
                .iter()
                .map(|c| *c as u32)
                .flat_map(|c| c.to_le_bytes())
                .collect(),
            CharacterAlphabet::Ranges(ranges) => ranges
                .iter()
                .flat_map(|range| {
                    let start = *(range.start()) as u32;
                    let end = *(range.end()) as u32;

                    let merged: [u8; 8] = merge_arrays(start.to_le_bytes(), end.to_le_bytes());
                    merged.to_vec()
                })
                .collect(),
            CharacterAlphabet::UnicodeCategory(category) => {
                let category_id = match category {
                    UnicodeCategory::Letter => 0u32,
                    UnicodeCategory::LowercaseLetter => 1u32,
                    UnicodeCategory::UppercaseLetter => 2u32,
                    UnicodeCategory::TitlecaseLetter => 3u32,
                    UnicodeCategory::CasedLetter => 4u32,
                    UnicodeCategory::ModifiedLetter => 5u32,
                    UnicodeCategory::OtherLetter => 6u32,
                    UnicodeCategory::Mark => 7u32,
                    UnicodeCategory::NonSpacingMark => 8u32,
                    UnicodeCategory::SpacingCombiningMark => 9u32,
                    UnicodeCategory::EnclosingMark => 10u32,
                    UnicodeCategory::Separator => 11u32,
                    UnicodeCategory::SpaceSeparator => 12u32,
                    UnicodeCategory::LineSeparator => 13u32,
                    UnicodeCategory::ParagraphSeparator => 14u32,
                    UnicodeCategory::Symbol => 15u32,
                    UnicodeCategory::MathSymbol => 16u32,
                    UnicodeCategory::CurrencySymbol => 17u32,
                    UnicodeCategory::ModifierSymbol => 18u32,
                    UnicodeCategory::OtherSymbol => 19u32,
                    UnicodeCategory::Number => 20u32,
                    UnicodeCategory::DecimalDigitNumber => 21u32,
                    UnicodeCategory::LetterNumber => 22u32,
                    UnicodeCategory::OtherNumber => 23u32,
                    UnicodeCategory::Punctuation => 24u32,
                    UnicodeCategory::DashPunctuation => 25u32,
                    UnicodeCategory::OpenPunctuation => 26u32,
                    UnicodeCategory::ClosePunctuation => 27u32,
                    UnicodeCategory::InitialPunctuation => 28u32,
                    UnicodeCategory::FinalPunctuation => 29u32,
                    UnicodeCategory::ConnectorPunctuation => 30u32,
                    UnicodeCategory::OtherPunctuation => 31u32,
                    UnicodeCategory::Other => 32u32,
                    UnicodeCategory::Control => 33u32,
                    UnicodeCategory::Format => 34u32,
                    UnicodeCategory::PrivateUse => 35u32,
                    UnicodeCategory::Surrogate => 36u32,
                    UnicodeCategory::Unassigned => 37u32,
                }
                .to_le_bytes();

                let padded_array: [u8; 8] = merge_arrays(category_id, [0u8; 4]);
                padded_array.to_vec()
            }
        }
    }
}

impl ToBytecode for InstConsumeSet {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = (self.idx as u64).to_le_bytes();

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstEpsilon {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let cond_uint_repr: u64 = match self.cond {
            EpsilonCond::WordBoundary => 0,
            EpsilonCond::NonWordBoundary => 1,
            EpsilonCond::StartOfStringOnly => 2,
            EpsilonCond::EndOfStringOnlyNonNewline => 3,
            EpsilonCond::EndOfStringOnly => 4,
            EpsilonCond::PreviousMatchEnd => 5,
            EpsilonCond::EndOfString => 6,
        };

        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = cond_uint_repr.to_le_bytes();

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstSplit {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let x_bytes = self.x_branch.as_u32().to_le_bytes();
        let y_bytes = self.y_branch.as_u32().to_le_bytes();

        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = merge_arrays(x_bytes, y_bytes);

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstJmp {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        // pad out the next inst index from 4 to 8 bytes.
        let padded_next_inst: u64 = self.next.as_u32().into();

        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = padded_next_inst.to_le_bytes();

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstStartSave {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let slot_id = self.slot_id as u64;

        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = slot_id.to_le_bytes();

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstEndSave {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let slot_id = self.slot_id as u64;

        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = slot_id.to_le_bytes();

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstMatch {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = [0u8; 8];

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

impl ToBytecode for InstMeta {
    type Output = OpcodeBytecodeRepr;

    fn to_bytecode(&self) -> Self::Output {
        let first = Self::OPCODE_BINARY_REPR.to_le_bytes();
        let second = match self.0 {
            MetaKind::SetExpressionId(id) => merge_arrays([0u8; 4], id.to_le_bytes()),
        };

        OpcodeBytecodeRepr(merge_arrays(first, second))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_encode_header_to_correct_bytecode_representation() {
        let input_output = [
            // minimal functionality test.
            (
                Instructions::new(vec![], vec![Opcode::Any, Opcode::Match]),
                vec![
                    240, 240, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            // multiple sets and fast-forward
            (
                Instructions::new(
                    vec![
                        CharacterSet::inclusive(CharacterAlphabet::Range('a'..='z')),
                        CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a'])),
                    ],
                    vec![
                        Opcode::Any,
                        Opcode::ConsumeSet(InstConsumeSet::new(1)),
                        Opcode::Match,
                    ],
                )
                .with_fast_forward(FastForward::Set(0)),
                vec![
                    240, 240, 2, 0, 2, 0, 0, 0, 3, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 26, 26, 0, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0, 26,
                    26, 1, 0, 1, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
        ];

        for (test_case, (char_set, expected_output)) in input_output.into_iter().enumerate() {
            let generated_bytecode = char_set.to_bytecode();

            // assert the bytecode is 16-byte (128-bit) aligned
            assert!((test_case, generated_bytecode.len() % 16) == (test_case, 0));

            // assert the generated output matches the expected output
            assert_eq!(
                (test_case, expected_output),
                (test_case, generated_bytecode)
            );
        }
    }

    #[test]
    fn should_encode_character_sets_to_correct_bytecode_representation() {
        let input_output = [
            (
                CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a'])),
                vec![26, 26, 1, 0, 1, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['a'])),
                vec![26, 26, 5, 0, 1, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
                vec![26, 26, 1, 0, 2, 0, 0, 0, 97, 0, 0, 0, 98, 0, 0, 0],
            ),
            (
                CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
                vec![26, 26, 5, 0, 2, 0, 0, 0, 97, 0, 0, 0, 98, 0, 0, 0],
            ),
            (
                CharacterSet::inclusive(CharacterAlphabet::Range('a'..='z')),
                vec![26, 26, 0, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
            ),
            (
                CharacterSet::exclusive(CharacterAlphabet::Range('a'..='z')),
                vec![26, 26, 4, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
            ),
            (
                CharacterSet::inclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
                vec![26, 26, 2, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
            ),
            (
                CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
                vec![26, 26, 6, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
            ),
            (
                CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
                vec![26, 26, 6, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
            ),
            (
                CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z', 'A'..='Z'])),
                vec![
                    26, 26, 6, 0, 2, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0, 65, 0, 0, 0, 90, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                CharacterSet::exclusive(CharacterAlphabet::UnicodeCategory(
                    UnicodeCategory::DecimalDigitNumber,
                )),
                vec![26, 26, 7, 0, 1, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                CharacterSet::inclusive(CharacterAlphabet::UnicodeCategory(
                    UnicodeCategory::DecimalDigitNumber,
                )),
                vec![26, 26, 3, 0, 1, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];

        for (test_case, (char_set, expected_output)) in input_output.into_iter().enumerate() {
            let generated_bytecode = char_set.to_bytecode();

            // assert the bytecode is 16-byte (128-bit) aligned
            assert!((test_case, generated_bytecode.len() % 16) == (test_case, 0));

            // assert the generated output matches the expected output
            assert_eq!(
                (test_case, expected_output),
                (test_case, generated_bytecode)
            );
        }
    }

    #[test]
    fn should_encode_instruction_into_expected_bytecode_representation() {
        let input_output = [
            (
                Opcode::Any,
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                Opcode::Consume(InstConsume::new('a')),
                [2, 0, 0, 0, 0, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                Opcode::ConsumeSet(InstConsumeSet::new(2)),
                [3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                Opcode::Epsilon(InstEpsilon::new(EpsilonCond::EndOfStringOnlyNonNewline)),
                [4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(256))),
                [5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            ),
            (
                Opcode::Jmp(InstJmp::new(InstIndex::from(1))),
                [6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                Opcode::StartSave(InstStartSave::new(1)),
                [7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                Opcode::EndSave(InstEndSave::new(1)),
                [8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                Opcode::Match,
                [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];

        for (test_case, (opcode, expected_output)) in input_output.into_iter().enumerate() {
            let generated_bytecode = opcode.to_bytecode();
            let expected_bytecode = OpcodeBytecodeRepr(expected_output);

            assert_eq!(
                (test_case, expected_bytecode),
                (test_case, generated_bytecode)
            );
        }
    }
}
