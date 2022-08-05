//! Provides utilities for deserializing a binary representation of the
//! bytecode.

#[derive(Debug, PartialEq, Eq)]
pub enum BytecodeConversionError {
    CharacterEncodingError(u64),
    IntegerConversionTo32Bit(u64),
    OutOfBoundsEpsilonValue(u64),
    ByteWidthMismatch { expected: usize, received: usize },
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

impl<B: AsRef<[u8]>> FromBytecode<B> for super::Opcode {
    type Output = Self;
    type Error = BytecodeConversionError;

    fn from_bytecode(bin: B) -> Result<Self::Output, Self::Error> {
        use super::*;

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
}
