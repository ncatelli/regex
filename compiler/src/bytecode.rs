use regex_runtime::*;

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
            FastForward::None => (0u8, 0u32),
            FastForward::Char(c) => (1u8, c as u32),
            FastForward::Set(idx) => (2u8, idx as u32),
        };

        let set_bytes: Vec<u8> = self.sets.iter().flat_map(|s| s.to_bytecode()).collect();
        let instruction_bytes = self
            .program
            .iter()
            .map(|inst| inst.to_bytecode())
            .flat_map(|or| or.0);

        let inst_offset = HEADER_LEN + (set_bytes.len() as u32);

        let header_bytes: [u8; 2] = (Self::MAGIC_NUMBER as u16).to_le_bytes();
        let lower_32_bits: [u8; 4] = merge_arrays(header_bytes, (ff_variant as u16).to_le_bytes());
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
            .chain(set_bytes.into_iter())
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
        let header = Self::MAGIC_NUMBER;
        let membership = match self.membership {
            SetMembership::Inclusive => 0u8,
            SetMembership::Exclusive => 1u8,
        };
        let (set_variant, len) = match self.set {
            CharacterAlphabet::Range(_) => (0u8, 1),
            CharacterAlphabet::Explicit(ref chars) => (1u8, chars.len()),
            CharacterAlphabet::Ranges(ref ranges) => (2u8, ranges.len()),
            CharacterAlphabet::UnicodeCategory(_) => (3, 1),
        };

        let truncated_alphabet_cnt: u32 = len
            .try_into()
            .expect("character alphabet count overflows 32-bit integer");

        let variant_and_membership = ((membership as u16) << 2) | (set_variant as u16);
        let variant_and_membership_bytes = variant_and_membership.to_le_bytes();
        let lower_32_bytes = merge_arrays(header.to_le_bytes(), variant_and_membership_bytes);
        let lower_64_bytes: [u8; 8] =
            merge_arrays(lower_32_bytes, truncated_alphabet_cnt.to_le_bytes());

        let entries = self.set.to_bytecode();

        let set_block: [u8; 16] = merge_arrays(lower_64_bytes, [0u8; 8]);
        set_block.into_iter().chain(entries.into_iter()).collect()
    }
}

impl ToBytecode for CharacterAlphabet {
    type Output = Vec<u8>;

    fn to_bytecode(&self) -> Self::Output {
        /// all bytes need to be aligned on a 16-byte (128-bit) boundary.
        const ALIGNMENT: u32 = 16;

        match self {
            CharacterAlphabet::Range(r) => {
                let start = *(r.start()) as u32;
                let end = *(r.end()) as u32;

                let merged: [u8; 8] = merge_arrays(start.to_le_bytes(), end.to_le_bytes());
                merged.to_vec()
            }
            CharacterAlphabet::Explicit(chars) => {
                let char_cnt = chars.len() as u32;
                let byte_cnt: u32 = char_cnt * 4;
                let align_up_to = (byte_cnt + (ALIGNMENT - 1)) & ALIGNMENT.wrapping_neg();
                let padding = (align_up_to - byte_cnt) as usize;

                chars
                    .iter()
                    .map(|c| *c as u32)
                    .flat_map(|c| c.to_le_bytes())
                    .chain([0u8].into_iter().cycle().take(padding))
                    .collect()
            }
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
                    UnicodeCategory::SpaceSeparator => 13u32,
                    UnicodeCategory::LineSeparator => 14u32,
                    UnicodeCategory::ParagraphSeparator => 15u32,
                    UnicodeCategory::Symbol => 16u32,
                    UnicodeCategory::MathSymbol => 17u32,
                    UnicodeCategory::CurrencySymbol => 18u32,
                    UnicodeCategory::ModifierSymbol => 19u32,
                    UnicodeCategory::OtherSymbol => 20u32,
                    UnicodeCategory::Number => 21u32,
                    UnicodeCategory::DecimalDigitNumber => 22u32,
                    UnicodeCategory::LetterNumber => 23u32,
                    UnicodeCategory::OtherNumber => 24u32,
                    UnicodeCategory::Punctuation => 25u32,
                    UnicodeCategory::DashPunctuation => 26u32,
                    UnicodeCategory::OpenPunctuation => 27u32,
                    UnicodeCategory::ClosePunctuation => 28u32,
                    UnicodeCategory::InitialPunctuation => 29u32,
                    UnicodeCategory::FinalPunctuation => 30u32,
                    UnicodeCategory::ConnectorPunctuation => 31u32,
                    UnicodeCategory::OtherPunctuation => 32u32,
                    UnicodeCategory::Other => 33u32,
                    UnicodeCategory::Control => 34u32,
                    UnicodeCategory::Format => 35u32,
                    UnicodeCategory::PrivateUse => 36u32,
                    UnicodeCategory::Surrogate => 37u32,
                    UnicodeCategory::Unassigned => 38u32,
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
        let padded_next_inst = self.next.as_u32() as u64;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_encode_instruction_into_expected_bytecode_representation() {
        use super::ToBytecode;

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
