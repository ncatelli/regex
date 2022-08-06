//! Provides utilities for deserializing a binary representation of the
//! bytecode.

#[derive(Debug, PartialEq, Eq)]
pub enum BytecodeConversionError {
    CharacterEncodingError(u64),
    IntegerConversionTo32Bit(u64),
    IntegerConversionToUsize(u64),
    OutOfBoundsEpsilonValue(u64),
    ByteWidthMismatch { expected: usize, received: usize },
    ValueMismatch(String),
}

impl std::fmt::Display for BytecodeConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CharacterEncodingError(val) => {
                write!(f, "unable to encode value {} as character", val)
            }
            Self::IntegerConversionTo32Bit(val) => {
                write!(f, "unable to convert {} to 32-bit value", val)
            }
            Self::IntegerConversionToUsize(val) => {
                write!(f, "unable to convert {} to ptr sized value", val)
            }
            Self::OutOfBoundsEpsilonValue(val) => {
                write!(f, "epsilon condition id {} out of range", val)
            }
            Self::ByteWidthMismatch { expected, received } => {
                write!(
                    f,
                    "byte-width mismatch, expected {}, received {}",
                    expected, received
                )
            }
            Self::ValueMismatch(e) => write!(f, "value mismatch: {}", e),
        }
    }
}

/// Represents a conversion trait from a given opcodes binary little-endian
/// representation into it's intermediary state.
pub trait FromBytecode<B: AsRef<[u8]>> {
    // The output type of a successful match.
    type Output;
    // An alternate error type.
    type Error;

    fn from_bytecode(bin: B) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, PartialEq, Eq)]
enum FastForwardVariant {
    None,
    Char,
    Set,
}

impl TryFrom<u8> for FastForwardVariant {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Char),
            2 => Ok(Self::Set),
            _ => Err(()),
        }
    }
}

impl<B: AsRef<[u8]>> FromBytecode<B> for crate::Instructions {
    type Output = Self;

    type Error = String;

    fn from_bytecode(bin: B) -> Result<Self::Output, Self::Error> {
        const CHUNK_32BIT: usize = 4;
        const CHUNK_128BIT: usize = 16;

        // header is 32bits x 8, this takes 5, as the last 3 32-bit values are unused.
        let mut header = bin
            .as_ref()
            .chunks_exact(CHUNK_32BIT)
            // safe to unwrap due to exact chunk guarantee.
            .map(|v| TryInto::<[u8; 4]>::try_into(v).unwrap())
            .take(5);

        let ff_variant = if let Some([0xF0, 0xF0, ff_variant, _]) = header.next() {
            FastForwardVariant::try_from(ff_variant)
                .map_err(|_| format!("fast-forward variant out of range: {}", ff_variant))
        } else {
            Err("invalid header".to_string())
        }?;

        let set_cnt = header
            .next()
            .ok_or_else(|| "unexpected end of header".to_string())
            .map(u32::from_le_bytes)
            .and_then(|val| usize::try_from(val).map_err(|e| e.to_string()))?;
        let inst_cnt = header
            .next()
            .ok_or_else(|| "unexpected end of header".to_string())
            .map(u32::from_le_bytes)
            .and_then(|val| usize::try_from(val).map_err(|e| e.to_string()))?;
        let inst_offset = header
            .next()
            .ok_or_else(|| "unexpected end of header".to_string())
            .map(u32::from_le_bytes)
            .and_then(|val| usize::try_from(val).map_err(|e| e.to_string()))?;
        let ff_value = match (ff_variant, header.next()) {
            (FastForwardVariant::None, Some(_)) => Ok(crate::FastForward::None),
            (FastForwardVariant::Char, Some(bytes)) => char::from_u32(u32::from_le_bytes(bytes))
                .map(crate::FastForward::Char)
                .ok_or_else(|| "unable to deserialize fast-forward char".to_string()),
            (FastForwardVariant::Set, Some(bytes)) => usize::try_from(u32::from_le_bytes(bytes))
                .map(crate::FastForward::Set)
                .map_err(|_| "unable to deserialize fast-forward char".to_string()),
            (_, None) => Err("unexpected end of header".to_string()),
        }?;

        let inst_bytes_start = inst_offset;
        let inst_bytes = bin
            .as_ref()
            .get(inst_bytes_start..)
            .ok_or_else(|| "invalid instruction headers".to_string())?;

        let insts = inst_bytes
            .chunks_exact(CHUNK_128BIT)
            .take(inst_cnt)
            .map(crate::Opcode::from_bytecode)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        let mut next_set_start = 32;
        let set_bytes_end = set_cnt * 128 + next_set_start;

        let mut cnt = 0;
        let mut set_bytes = bin
            .as_ref()
            .get(next_set_start..set_bytes_end)
            .ok_or_else(|| "invalid character set headers".to_string())?;
        let mut sets = Vec::with_capacity(set_cnt);

        // loop over the character sets, reborrow the binary from each sets
        // next starting position until either the set count is hit or the end
        // of the byte slice is hit.
        while (cnt < set_cnt) && (next_set_start < set_bytes_end) {
            let (set, bytes_consumed) =
                crate::CharacterSet::from_bytecode(set_bytes).map_err(|e| e.to_string())?;

            sets.push(set);
            cnt += 1;
            next_set_start += bytes_consumed;
            set_bytes = bin
                .as_ref()
                .get(next_set_start..set_bytes_end)
                .ok_or_else(|| "invalid character set headers".to_string())?;
        }

        let program = crate::Instructions::new(sets, insts).with_fast_forward(ff_value);

        Ok(program)
    }
}

/// Represents a runtime dispatchable set of characters variant. Functionally
/// this is equivalent to `CharacterAlphabet` sans the enclosed values.
#[derive(Debug, PartialEq, Eq)]
enum CharacterAlphabetVariant {
    /// Represents a range of values i.e. `0-9`, `a-z`, `A-Z`, etc...
    Range,
    /// Represents an explicitly defined set of values. i.e. `[a,b,z]`, `[1,2,7]`
    Explicit,
    /// Represents a set of range of values i.e. `[0-9a-zA-Z]`,  etc...
    Ranges,
    /// Represents a unicode category.
    UnicodeCategory,
}

impl TryFrom<u8> for CharacterAlphabetVariant {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Range),
            1 => Ok(Self::Explicit),
            2 => Ok(Self::Ranges),
            3 => Ok(Self::UnicodeCategory),
            _ => Err(()),
        }
    }
}

fn decode_range_alphabet(data: [u8; 8]) -> Result<std::ops::RangeInclusive<char>, ()> {
    // safe to unwrap due to array constraint
    let lower = data[0..4]
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
        .and_then(char::from_u32)
        .ok_or(())?;
    let upper = data[4..8]
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
        .and_then(char::from_u32)
        .ok_or(())?;

    Ok(lower..=upper)
}

fn decode_unicode_category_alphabet(data: [u8; 8]) -> Result<crate::UnicodeCategory, ()> {
    let variant = u64::from_le_bytes(data);

    match variant {
        0 => Some(crate::UnicodeCategory::Letter),
        1 => Some(crate::UnicodeCategory::LowercaseLetter),
        2 => Some(crate::UnicodeCategory::UppercaseLetter),
        3 => Some(crate::UnicodeCategory::TitlecaseLetter),
        4 => Some(crate::UnicodeCategory::CasedLetter),
        5 => Some(crate::UnicodeCategory::ModifiedLetter),
        6 => Some(crate::UnicodeCategory::OtherLetter),
        7 => Some(crate::UnicodeCategory::Mark),
        8 => Some(crate::UnicodeCategory::NonSpacingMark),
        9 => Some(crate::UnicodeCategory::SpacingCombiningMark),
        10 => Some(crate::UnicodeCategory::EnclosingMark),
        11 => Some(crate::UnicodeCategory::Separator),
        12 => Some(crate::UnicodeCategory::SpaceSeparator),
        13 => Some(crate::UnicodeCategory::LineSeparator),
        14 => Some(crate::UnicodeCategory::ParagraphSeparator),
        15 => Some(crate::UnicodeCategory::Symbol),
        16 => Some(crate::UnicodeCategory::MathSymbol),
        17 => Some(crate::UnicodeCategory::CurrencySymbol),
        18 => Some(crate::UnicodeCategory::ModifierSymbol),
        19 => Some(crate::UnicodeCategory::OtherSymbol),
        20 => Some(crate::UnicodeCategory::Number),
        21 => Some(crate::UnicodeCategory::DecimalDigitNumber),
        22 => Some(crate::UnicodeCategory::LetterNumber),
        23 => Some(crate::UnicodeCategory::OtherNumber),
        24 => Some(crate::UnicodeCategory::Punctuation),
        25 => Some(crate::UnicodeCategory::DashPunctuation),
        26 => Some(crate::UnicodeCategory::OpenPunctuation),
        27 => Some(crate::UnicodeCategory::ClosePunctuation),
        28 => Some(crate::UnicodeCategory::InitialPunctuation),
        29 => Some(crate::UnicodeCategory::FinalPunctuation),
        30 => Some(crate::UnicodeCategory::ConnectorPunctuation),
        31 => Some(crate::UnicodeCategory::OtherPunctuation),
        32 => Some(crate::UnicodeCategory::Other),
        33 => Some(crate::UnicodeCategory::Control),
        34 => Some(crate::UnicodeCategory::Format),
        35 => Some(crate::UnicodeCategory::PrivateUse),
        36 => Some(crate::UnicodeCategory::Surrogate),
        37 => Some(crate::UnicodeCategory::Unassigned),
        _ => None,
    }
    .ok_or(())
}

fn align_up<const ALIGNMENT: usize>(val: usize) -> usize {
    (val + (ALIGNMENT - 1)) & ALIGNMENT.wrapping_neg()
}

impl<B: AsRef<[u8]>> FromBytecode<B> for crate::CharacterSet {
    type Output = (Self, usize);
    type Error = BytecodeConversionError;

    fn from_bytecode(bin: B) -> Result<Self::Output, Self::Error> {
        const CHUNK_SIZE: usize = 8usize;
        let data = bin.as_ref();

        // chunk as 8-bytes.
        let mut chunked_data = data.chunks_exact(CHUNK_SIZE);

        // unpack the set header
        let (membership, variant, entry_cnt) =
            if let Some([26, 26, membership_and_variant, _unused, entry_cnt @ ..]) =
                chunked_data.next()
            {
                let membership = if (membership_and_variant & 0b100) == 0 {
                    crate::SetMembership::Inclusive
                } else {
                    crate::SetMembership::Exclusive
                };
                let variant = CharacterAlphabetVariant::try_from((membership_and_variant) & 0b11)
                    // safe to unwrap due to `&` truncation.
                    .unwrap();
                let entry_cnt = entry_cnt
                    .try_into()
                    .map(u32::from_le_bytes)
                    .map_err(|_| BytecodeConversionError::IntegerConversionTo32Bit(0))?;

                Ok((membership, variant, entry_cnt))
            } else {
                Err(BytecodeConversionError::ValueMismatch(
                    "invalid character set header".to_string(),
                ))
            }?;

        let (alphabet, consumed_bytes) = match (variant, entry_cnt) {
            (CharacterAlphabetVariant::Range, 1) => {
                let alphabet = chunked_data
                    .take(1)
                    // safe to unwrap, constrained to 8-byte chunks above.
                    .map(|slice| TryInto::<[u8; CHUNK_SIZE]>::try_into(slice).unwrap())
                    .map(|arr| decode_range_alphabet(arr).map(crate::CharacterAlphabet::Range))
                    .next();

                if let Some(Ok(alphabet)) = alphabet {
                    Ok((alphabet, 16))
                } else {
                    Err(BytecodeConversionError::ValueMismatch(
                        "alphabet variant out of range".to_string(),
                    ))
                }
            }
            (CharacterAlphabetVariant::Explicit, cnt) => {
                let element_cnt = usize::try_from(cnt)
                    .map_err(|_| BytecodeConversionError::IntegerConversionToUsize(cnt.into()))?;

                let alphabet = chunked_data
                    // break the data into 32-bit chunks
                    .flat_map(|d| {
                        // safe to unwrap due to chunk guarantee
                        let lower = TryInto::<[u8; 4]>::try_into(&d[0..4]).unwrap();
                        let upper = TryInto::<[u8; 4]>::try_into(&d[4..8]).unwrap();

                        [lower, upper].into_iter()
                    })
                    .take(element_cnt)
                    .map(u32::from_le_bytes)
                    .map(char::from_u32)
                    .collect::<Option<Vec<_>>>();

                // 32-bit elements aligned on 128-bit boundaries plus a 64-bit header
                let aligned_set_size = align_up::<16>(element_cnt * 4 + 8);

                if let Some(alphabet) = alphabet {
                    Ok((
                        crate::CharacterAlphabet::Explicit(alphabet),
                        aligned_set_size,
                    ))
                } else {
                    Err(BytecodeConversionError::ValueMismatch(
                        "alphabet variant out of range".to_string(),
                    ))
                }
            }
            (CharacterAlphabetVariant::Ranges, cnt) => {
                let element_cnt = usize::try_from(cnt)
                    .map_err(|_| BytecodeConversionError::IntegerConversionToUsize(cnt.into()))?;

                let alphabet = chunked_data
                    .take(element_cnt)
                    // safe to unwrap, constrained to 8-byte chunks above.
                    .map(|slice| TryInto::<[u8; CHUNK_SIZE]>::try_into(slice).unwrap())
                    .map(decode_range_alphabet)
                    .collect::<Result<Vec<_>, ()>>();

                // 64-bit elements aligned on 128-bit boundaries plus a 64-bit header
                let aligned_set_size = align_up::<16>(element_cnt * 8 + 8);

                if let Ok(alphabet) = alphabet {
                    Ok((crate::CharacterAlphabet::Ranges(alphabet), aligned_set_size))
                } else {
                    Err(BytecodeConversionError::ValueMismatch(
                        "alphabet variant out of range".to_string(),
                    ))
                }
            }
            (CharacterAlphabetVariant::UnicodeCategory, 1) => {
                let alphabet = chunked_data
                    .take(1)
                    .map(|slice| TryInto::<[u8; CHUNK_SIZE]>::try_into(slice).unwrap())
                    .map(|arr| {
                        decode_unicode_category_alphabet(arr)
                            .map(crate::CharacterAlphabet::UnicodeCategory)
                    })
                    .next();

                if let Some(Ok(alphabet)) = alphabet {
                    Ok((alphabet, 16))
                } else {
                    Err(BytecodeConversionError::ValueMismatch(
                        "alphabet variant out of range".to_string(),
                    ))
                }
            }
            _ => Err(BytecodeConversionError::ValueMismatch(
                "alphabet variant out of range".to_string(),
            )),
        }?;

        Ok((
            crate::CharacterSet::new(membership, alphabet),
            consumed_bytes,
        ))
    }
}

impl<B: AsRef<[u8]>> FromBytecode<B> for crate::Opcode {
    type Output = Self;
    type Error = BytecodeConversionError;

    fn from_bytecode(bin: B) -> Result<Self::Output, Self::Error> {
        use crate::*;

        let data = bin.as_ref();

        let variant = data
            .get(0..8)
            .and_then(|slice| TryInto::<[u8; 8]>::try_into(slice).ok())
            .map(u64::from_le_bytes);
        let operand = data
            .get(8..16)
            .and_then(|slice| TryInto::<[u8; 8]>::try_into(slice).ok())
            .map(u64::from_le_bytes);

        match (variant, operand) {
            (Some(_), None) | (None, None) => Err(BytecodeConversionError::ByteWidthMismatch {
                expected: 16,
                received: data.len(),
            }),
            (Some(InstAny::OPCODE_BINARY_REPR), Some(0)) => Ok(Opcode::Any),
            (Some(InstConsume::OPCODE_BINARY_REPR), Some(char_value)) => u32::try_from(char_value)
                .ok()
                .and_then(char::from_u32)
                .map(|c| Opcode::Consume(InstConsume::new(c)))
                .ok_or(BytecodeConversionError::CharacterEncodingError(char_value)),
            (Some(InstConsumeSet::OPCODE_BINARY_REPR), Some(set_id)) => usize::try_from(set_id)
                .ok()
                .map(|set| Opcode::ConsumeSet(InstConsumeSet::new(set)))
                .ok_or(BytecodeConversionError::IntegerConversionTo32Bit(set_id)),

            (Some(InstEpsilon::OPCODE_BINARY_REPR), Some(epsilon_kind)) => {
                let cond = match epsilon_kind {
                    0 => Ok(EpsilonCond::WordBoundary),
                    1 => Ok(EpsilonCond::NonWordBoundary),
                    2 => Ok(EpsilonCond::StartOfStringOnly),
                    3 => Ok(EpsilonCond::EndOfStringOnlyNonNewline),
                    4 => Ok(EpsilonCond::EndOfStringOnly),
                    5 => Ok(EpsilonCond::PreviousMatchEnd),
                    6 => Ok(EpsilonCond::EndOfString),
                    other => Err(BytecodeConversionError::OutOfBoundsEpsilonValue(other)),
                }?;

                Ok(Opcode::Epsilon(InstEpsilon::new(cond)))
            }

            (Some(InstSplit::OPCODE_BINARY_REPR), Some(_)) => {
                // should be safe to unwrap due to & truncation
                let x_branch = data
                    .get(8..12)
                    .and_then(|slice| TryInto::<[u8; 4]>::try_into(slice).ok())
                    .map(u32::from_le_bytes)
                    .map(InstIndex::from)
                    .unwrap();
                let y_branch = data
                    .get(12..16)
                    .and_then(|slice| TryInto::<[u8; 4]>::try_into(slice).ok())
                    .map(u32::from_le_bytes)
                    .map(InstIndex::from)
                    .unwrap();

                Ok(Opcode::Split(InstSplit::new(x_branch, y_branch)))
            }
            (Some(InstJmp::OPCODE_BINARY_REPR), Some(idx)) => u32::try_from(idx)
                .ok()
                .map(InstIndex::from)
                .map(|inst_idx| Opcode::Jmp(InstJmp::new(inst_idx)))
                .ok_or(BytecodeConversionError::IntegerConversionTo32Bit(idx)),
            (Some(InstStartSave::OPCODE_BINARY_REPR), Some(slot_id)) => usize::try_from(slot_id)
                .ok()
                .map(|slot| Opcode::StartSave(InstStartSave::new(slot)))
                .ok_or(BytecodeConversionError::IntegerConversionTo32Bit(slot_id)),
            (Some(InstEndSave::OPCODE_BINARY_REPR), Some(slot_id)) => usize::try_from(slot_id)
                .ok()
                .map(|slot| Opcode::EndSave(InstEndSave::new(slot)))
                .ok_or(BytecodeConversionError::IntegerConversionTo32Bit(slot_id)),
            (Some(InstMatch::OPCODE_BINARY_REPR), Some(0)) => Ok(Opcode::Match),
            (Some(_), Some(_)) => todo!(),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::binary::FromBytecode;
    use crate::*;

    #[test]
    fn should_decode_bytecode_into_expected_program() {
        let input_output = [(
            vec![
                240, 240, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            Instructions::new(vec![], vec![Opcode::Any, Opcode::Match]),
        )];

        for (test_case, (bin, expected_output)) in input_output.into_iter().enumerate() {
            let decoded_program = crate::Instructions::from_bytecode(bin);

            // assert the generated output matches the expected output
            assert_eq!(
                (test_case, Ok(expected_output)),
                (test_case, decoded_program)
            );
        }
    }

    #[test]
    fn should_decode_bytecode_into_expected_opcode() {
        let input_output = [
            (
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::Any),
            ),
            (
                [2, 0, 0, 0, 0, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::Consume(InstConsume::new('a'))),
            ),
            (
                [3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::ConsumeSet(InstConsumeSet::new(2))),
            ),
            (
                [4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::Epsilon(InstEpsilon::new(
                    EpsilonCond::EndOfStringOnlyNonNewline,
                ))),
            ),
            (
                [5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                Ok(Opcode::Split(InstSplit::new(
                    InstIndex::from(1),
                    InstIndex::from(256),
                ))),
            ),
            (
                [6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::Jmp(InstJmp::new(InstIndex::from(1)))),
            ),
            (
                [7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::StartSave(InstStartSave::new(1))),
            ),
            (
                [8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::EndSave(InstEndSave::new(1))),
            ),
            (
                [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                Ok(Opcode::Match),
            ),
        ];

        for (test_case, (bin, expected_output)) in input_output.into_iter().enumerate() {
            let decoded_opcode = Opcode::from_bytecode(bin.to_vec());

            assert_eq!((test_case, expected_output), (test_case, decoded_opcode));
        }
    }

    #[test]
    fn should_decode_character_set_bytecode_to_correct_internal_representation() {
        let input_output = [
            (
                vec![26, 26, 1, 0, 1, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0],
                (
                    CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a'])),
                    16,
                ),
            ),
            (
                vec![26, 26, 5, 0, 1, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0],
                (
                    CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['a'])),
                    16,
                ),
            ),
            (
                vec![26, 26, 1, 0, 2, 0, 0, 0, 97, 0, 0, 0, 98, 0, 0, 0],
                (
                    CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
                    16,
                ),
            ),
            (
                vec![26, 26, 5, 0, 2, 0, 0, 0, 97, 0, 0, 0, 98, 0, 0, 0],
                (
                    CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
                    16,
                ),
            ),
            (
                vec![26, 26, 0, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                (
                    CharacterSet::inclusive(CharacterAlphabet::Range('a'..='z')),
                    16,
                ),
            ),
            (
                vec![26, 26, 4, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                (
                    CharacterSet::exclusive(CharacterAlphabet::Range('a'..='z')),
                    16,
                ),
            ),
            (
                vec![26, 26, 2, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                (
                    CharacterSet::inclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
                    16,
                ),
            ),
            (
                vec![26, 26, 6, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                (
                    CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
                    16,
                ),
            ),
            (
                vec![26, 26, 6, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                (
                    CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
                    16,
                ),
            ),
            (
                vec![
                    26, 26, 6, 0, 2, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0, 65, 0, 0, 0, 90, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ],
                (
                    CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z', 'A'..='Z'])),
                    32,
                ),
            ),
            (
                vec![26, 26, 7, 0, 1, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0],
                (
                    CharacterSet::exclusive(CharacterAlphabet::UnicodeCategory(
                        UnicodeCategory::DecimalDigitNumber,
                    )),
                    16,
                ),
            ),
            (
                vec![26, 26, 3, 0, 1, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0],
                (
                    CharacterSet::inclusive(CharacterAlphabet::UnicodeCategory(
                        UnicodeCategory::DecimalDigitNumber,
                    )),
                    16,
                ),
            ),
        ];

        for (test_case, (bin, expected_output)) in input_output.into_iter().enumerate() {
            let output = CharacterSet::from_bytecode(&bin);

            // assert the generated output matches the expected output
            assert_eq!((test_case, Ok(expected_output)), (test_case, output));
        }
    }
}
