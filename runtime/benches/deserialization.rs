use criterion::{criterion_group, criterion_main, Criterion};
use regex_runtime::*;

pub fn deserialization_of_bytecode_into_a_ir_program(c: &mut Criterion) {
    let bytecode_arr = [
        240, 240, 2, 0, 2, 0, 0, 0, 3, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 26, 26, 0, 0, 1, 0, 0, 0, 97, 0, 0, 0, 122, 0, 0, 0, 26, 26, 1, 0, 1, 0, 0, 0, 97,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let bin = bytecode_arr.as_slice();

    c.bench_function("deserialize binary to internal representation", |b| {
        b.iter(|| {
            let res = criterion::black_box(bytecode::from_binary(bin));
            assert!(res.is_ok())
        })
    });
}

criterion_group!(benches, deserialization_of_bytecode_into_a_ir_program);
criterion_main!(benches);
