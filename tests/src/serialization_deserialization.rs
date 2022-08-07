use regex_compiler::bytecode::ToBytecode;
use regex_runtime::bytecode::FromBytecode;
use regex_runtime::*;

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
                Opcode::Consume(InstConsume::new('a')),
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
