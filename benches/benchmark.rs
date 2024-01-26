use std::fs::File;
use std::vec::Vec;

use criterion::{criterion_group, criterion_main, Criterion};
use csv::Reader;
use rayon::prelude::*;

use lrtc::{Classifier, CompressionAlgorithm};

pub fn criterion_benchmark(c: &mut Criterion) {
    let imdb = File::open("./data/imdb.csv").unwrap();
    let mut reader = Reader::from_reader(imdb);

    let mut content = Vec::with_capacity(50000);
    let mut label = Vec::with_capacity(50000);
    for record in reader.records() {
        content.push(record.as_ref().unwrap()[0].to_string());
        label.push(record.unwrap()[1].to_string());
    }
    let mut classifier = Classifier::new(CompressionAlgorithm::Zstd, 3);
    classifier.train(&content[0..1000], &label[0..1000]);
    c.bench_function("classify zstd", |b| {
        b.iter(|| {
            let _x: Vec<_> = content[40_000..40_250]
                .par_iter()
                .map(|query| classifier.classify(query, 5))
                .collect();
        })
    });
    let mut classifier = Classifier::new(CompressionAlgorithm::Gzip, 3);
    classifier.train(&content[0..1000], &label[0..1000]);
    c.bench_function("classify gzip", |b| {
        b.iter(|| {
            let _x: Vec<_> = content[40_000..40_250]
                .par_iter()
                .map(|query| classifier.classify(query, 5))
                .collect();
        })
    });
    let mut classifier = Classifier::new(CompressionAlgorithm::Zlib, 3);
    classifier.train(&content[0..1000], &label[0..1000]);
    c.bench_function("classify zlib", |b| {
        b.iter(|| {
            let _x: Vec<_> = content[40_000..40_250]
                .par_iter()
                .map(|query| classifier.classify(query, 5))
                .collect();
        })
    });
    let mut classifier = Classifier::new(CompressionAlgorithm::Deflate, 3);
    classifier.train(&content[0..1000], &label[0..1000]);
    c.bench_function("classify deflate", |b| {
        b.iter(|| {
            let _x: Vec<_> = content[40_000..40_250]
                .par_iter()
                .map(|query| classifier.classify(query, 5))
                .collect();
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
}
criterion_main!(benches);
