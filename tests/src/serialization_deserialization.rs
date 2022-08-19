use regex_compiler::bytecode::ToBytecode;
use regex_runtime::bytecode::FromBytecode;
use regex_runtime::*;

#[test]
fn should_preserve_expected_semantics_from_parse_through_serialization() {
    use regex_runtime::{CharacterAlphabet, CharacterSet, FastForward, SetMembership};
    let expr = "\"([a-z]+)\"";

    let insts = regex_compiler::parse(expr)
        .map_err(|e| format!("{:?}", e))
        .and_then(regex_compiler::compile)
        .unwrap();

    let expected_program = Instructions::new(
        vec![CharacterSet::new(
            SetMembership::Inclusive,
            CharacterAlphabet::Range('a'..='z'),
        )],
        vec![
            Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
            Opcode::Any,
            Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
            Opcode::Consume(InstConsume::new('"')),
            Opcode::StartSave(InstStartSave::new(0)),
            Opcode::ConsumeSet(InstConsumeSet::new(0)),
            Opcode::Split(InstSplit::new(InstIndex::from(7), InstIndex::from(9))),
            Opcode::ConsumeSet(InstConsumeSet::new(0)),
            Opcode::Jmp(InstJmp::new(InstIndex::from(6))),
            Opcode::EndSave(InstEndSave::new(0)),
            Opcode::Consume(InstConsume::new('"')),
            Opcode::Match,
        ],
    )
    .with_fast_forward(FastForward::Char('"'));

    assert_eq!(&insts, &expected_program);

    let generated_bytecode = insts.to_bytecode();

    let deserialized_insts =
        regex_runtime::Instructions::from_bytecode(generated_bytecode).unwrap();

    assert_eq!(&insts, &deserialized_insts);

    let input = "\"hello\"";
    assert_eq!(
        Some([SaveGroupSlot::complete(0, 1, 6)]),
        regex_runtime::run::<1>(&insts, input)
    );

    assert_eq!(
        Some([SaveGroupSlot::complete(0, 1, 6)]),
        regex_runtime::run::<1>(&deserialized_insts, input)
    );
}

#[test]
fn should_preserve_equivalent_representation_in_bytecode_encoding() {
    let input_output = [
        Instructions::new(vec![], vec![Opcode::Any, Opcode::Match]),
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
        Instructions::new(
            vec![],
            vec![
                Opcode::Split(InstSplit::new(InstIndex::from(3), InstIndex::from(1))),
                Opcode::Any,
                Opcode::Jmp(InstJmp::new(InstIndex::from(0))),
                Opcode::Split(InstSplit::new(InstIndex::from(4), InstIndex::from(6))),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::Jmp(InstJmp::new(InstIndex::from(3))),
                Opcode::Match,
            ],
        ),
        Instructions::new(
            vec![],
            vec![
                Opcode::Any,
                Opcode::StartSave(InstStartSave::new(0)),
                Opcode::Consume(InstConsume::new('a')),
                Opcode::StartSave(InstStartSave::new(1)),
                Opcode::Consume(InstConsume::new('b')),
                Opcode::EndSave(InstEndSave::new(1)),
                Opcode::EndSave(InstEndSave::new(0)),
                Opcode::Match,
            ],
        ),
    ];

    for (test_case, program) in input_output.into_iter().enumerate() {
        let generated_bytecode = program.to_bytecode();
        let deserialized_program_result =
            regex_runtime::Instructions::from_bytecode(generated_bytecode);

        // assert the generated output matches the expected output
        assert_eq!(
            (test_case, Ok(program)),
            (test_case, deserialized_program_result)
        );
    }
}
