use criterion::{criterion_group, criterion_main, Criterion};
use csv::Reader;
use lrtc::{classify, CompressionAlgorithm};
use std::fs::File;
use std::vec::Vec;

pub fn criterion_benchmark(c: &mut Criterion) {
    let imdb = File::open("./data/imdb.csv").unwrap();
    let mut reader = Reader::from_reader(imdb);

    let mut content = Vec::with_capacity(50000);
    let mut label = Vec::with_capacity(50000);
    for record in reader.records() {
        content.push(record.as_ref().unwrap()[0].to_string());
        label.push(record.unwrap()[1].to_string());
    }
    c.bench_function("classify zstd", |b| {
        b.iter(|| {
            classify(
                content[0..1000].to_vec(),
                label[0..1000].to_vec(),
                content[40000..41000].to_vec(),
                3i32,
                CompressionAlgorithm::Zstd,
                5usize,
            )
        })
    });
    c.bench_function("classify gzip", |b| {
        b.iter(|| {
            classify(
                content[0..1000].to_vec(),
                label[0..1000].to_vec(),
                content[40000..41000].to_vec(),
                3i32,
                CompressionAlgorithm::Gzip,
                5usize,
            )
        })
    });
    c.bench_function("classify zlib", |b| {
        b.iter(|| {
            classify(
                content[0..1000].to_vec(),
                label[0..1000].to_vec(),
                content[40000..41000].to_vec(),
                3i32,
                CompressionAlgorithm::Zlib,
                5usize,
            )
        })
    });
    c.bench_function("classify deflate", |b| {
        b.iter(|| {
            classify(
                content[0..1000].to_vec(),
                label[0..1000].to_vec(),
                content[40000..41000].to_vec(),
                3i32,
                CompressionAlgorithm::Deflate,
                5usize,
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
}
criterion_main!(benches);
