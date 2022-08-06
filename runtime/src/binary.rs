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

impl<B: AsRef<[u8]>> FromBytecode<B> for crate::CharacterSet {
    type Output = Self;
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

        let alphabet = match (variant, entry_cnt) {
            (CharacterAlphabetVariant::Range, 1) => {
                let alphabet = chunked_data
                    .take(1)
                    // safe to unwrap, constrained to 8-byte chunks above.
                    .map(|slice| TryInto::<[u8; CHUNK_SIZE]>::try_into(slice).unwrap())
                    .map(|arr| decode_range_alphabet(arr).map(crate::CharacterAlphabet::Range))
                    .next();

                if let Some(Ok(alphabet)) = alphabet {
                    Ok(alphabet)
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

                if let Some(alphabet) = alphabet {
                    Ok(crate::CharacterAlphabet::Explicit(alphabet))
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

                if let Ok(alphabet) = alphabet {
                    Ok(crate::CharacterAlphabet::Ranges(alphabet))
                } else {
                    Err(BytecodeConversionError::ValueMismatch(
                        "alphabet variant out of range".to_string(),
                    ))
                }
            }
            (CharacterAlphabetVariant::UnicodeCategory, 1) => todo!(),
            _ => Err(BytecodeConversionError::ValueMismatch(
                "alphabet variant out of range".to_string(),
            )),
        }?;

        Ok(crate::CharacterSet::new(membership, alphabet))
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
                CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a'])),
            ),
            (
                vec![26, 26, 5, 0, 1, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0],
                CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['a'])),
            ),
            (
                vec![26, 26, 1, 0, 2, 0, 0, 0, 97, 0, 0, 0, 98, 0, 0, 0],
                CharacterSet::inclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
            ),
            (
                vec![26, 26, 5, 0, 2, 0, 0, 0, 97, 0, 0, 0, 98, 0, 0, 0],
                CharacterSet::exclusive(CharacterAlphabet::Explicit(vec!['a', 'b'])),
            ),
            (
                vec![26, 26, 0, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                CharacterSet::inclusive(CharacterAlphabet::Range('a'..='z')),
            ),
            (
                vec![26, 26, 4, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                CharacterSet::exclusive(CharacterAlphabet::Range('a'..='z')),
            ),
            (
                vec![26, 26, 2, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                CharacterSet::inclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
            ),
            (
                vec![26, 26, 6, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
            ),
            (
                vec![26, 26, 6, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0],
                CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z'])),
            ),
            (
                vec![
                    26, 26, 6, 0, 2, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0, 65, 0, 0, 0, 90, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ],
                CharacterSet::exclusive(CharacterAlphabet::Ranges(vec!['a'..='z', 'A'..='Z'])),
            ),
            // unimplemented yet
            /*
            (
                vec![26, 26, 7, 0, 1, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0],
                CharacterSet::exclusive(CharacterAlphabet::UnicodeCategory(
                    UnicodeCategory::DecimalDigitNumber,
                )),
            ),
            (
                vec![26, 26, 3, 0, 1, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0],
                CharacterSet::inclusive(CharacterAlphabet::UnicodeCategory(
                    UnicodeCategory::DecimalDigitNumber,
                )),
            ),*/
        ];

        for (test_case, (bin, expected_output)) in input_output.into_iter().enumerate() {
            let output = CharacterSet::from_bytecode(&bin);

            // assert the generated output matches the expected output
            assert_eq!((test_case, Ok(expected_output)), (test_case, output));
        }
    }
}
