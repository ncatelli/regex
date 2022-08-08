use criterion::{criterion_group, criterion_main, Criterion};
use regex_compiler::*;
use regex_runtime::*;

pub fn serialization_of_program_into_bytecode(c: &mut Criterion) {
    let prog = Instructions::new(
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
    .with_fast_forward(FastForward::Set(0));

    c.bench_function("serialize instruction to binary format", |b| {
        b.iter(|| {
            let res = criterion::black_box(to_binary(&prog));
            assert!(res.is_ok())
        })
    });
}

criterion_group!(benches, serialization_of_program_into_bytecode);
criterion_main!(benches);
