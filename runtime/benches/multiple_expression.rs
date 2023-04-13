use criterion::{black_box, criterion_group, criterion_main, Criterion};
use regex_runtime::*;

pub fn multiple_expression_match(c: &mut Criterion) {
    let input = "abc";

    let prog = Instructions::default().with_opcodes(vec![
        Opcode::Split(InstSplit::new(InstIndex::from(1), InstIndex::from(2))),
        Opcode::Split(InstSplit::new(InstIndex::from(7), InstIndex::from(12))),
        // first anchored expr
        Opcode::Meta(InstMeta(MetaKind::SetExpressionId(0))),
        Opcode::StartSave(InstStartSave::new(0)),
        Opcode::Consume(InstConsume::new('a')),
        Opcode::EndSave(InstEndSave::new(0)),
        Opcode::Match,
        // second anchored expr
        Opcode::Meta(InstMeta(MetaKind::SetExpressionId(1))),
        Opcode::StartSave(InstStartSave::new(0)),
        Opcode::Consume(InstConsume::new('b')),
        Opcode::EndSave(InstEndSave::new(0)),
        Opcode::Match,
        // third anchored expr
        Opcode::Meta(InstMeta(MetaKind::SetExpressionId(2))),
        Opcode::StartSave(InstStartSave::new(0)),
        Opcode::Consume(InstConsume::new('c')),
        Opcode::EndSave(InstEndSave::new(0)),
        Opcode::Match,
    ]);

    c.bench_function("match multiple expression input", |b| {
        let expected = [
            SaveGroupSlot::complete(0, 0, 1),
            SaveGroupSlot::complete(1, 0, 1),
            SaveGroupSlot::complete(2, 0, 1),
        ];

        b.iter(|| {
            let mut start_offset = 0;
            for (test_id, expected_match) in expected.into_iter().enumerate() {
                if let Some(
                    [SaveGroupSlot::Complete {
                        start,
                        end,
                        expression_id,
                    }],
                ) = run::<1>(&prog, black_box(&input[start_offset..]))
                {
                    assert_eq!(
                        (test_id, &expected_match),
                        (
                            test_id,
                            &SaveGroupSlot::Complete {
                                expression_id,
                                start,
                                end,
                            }
                        )
                    );

                    start_offset += end - start;
                } else {
                    panic!("failed")
                }
            }
        });
    });
}

criterion_group!(benches, multiple_expression_match);
criterion_main!(benches);
