use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use regex_compiler::*;

fn pad_input_to_length_with(prefix: &str, suffix: &str, pad_str: &str, len: usize) -> String {
    let prefix_len = prefix.chars().count();
    let suffix_len = suffix.chars().count();
    let req_padding = len - suffix_len;

    if suffix_len > len || prefix_len > len || (suffix_len + prefix_len) > len {
        "".to_string()
    } else {
        prefix
            .chars()
            .chain(pad_str.chars().cycle().take(req_padding))
            .chain(suffix.chars())
            .collect()
    }
}

pub fn exponential_input_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern length compilation comparison");
    let pad = "ab";

    (1..10)
        .map(|exponent| 2usize.pow(exponent))
        .map(|input_len| (pad_input_to_length_with("^", "", pad, input_len), input_len))
        .map(|(input, len)| (input.chars().enumerate().collect::<Vec<_>>(), len))
        .for_each(|(input, sample_size)| {
            group.throughput(Throughput::Elements(sample_size as u64));
            group.bench_with_input(
                BenchmarkId::new("pattern input length of size", sample_size),
                &input,
                |b, input| {
                    b.iter(|| {
                        let res = parse(input).map(compile);
                        assert!(res.is_ok())
                    })
                },
            );
        })
}

criterion_group!(benches, exponential_input_size_comparison);
criterion_main!(benches);
