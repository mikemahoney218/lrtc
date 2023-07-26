# lrtc: Low-resource text classification

This crate is a Rust implementation of the low-resource text classification
method introduced in Jiang et al. (2023). This implementation allows you to
choose from gzip, zstd, zlib, or deflate compression algorithms, at various
levels of compression.

```rust
use lrtc::{CompressionAlgorithm, classify};

let training = vec!["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
let training_labels = vec!["normal".to_string(), "godzilla".into(),];
let queries = vec!["another normal sentence".to_string(), "godzilla eats marshes in August".into(),];
// Using a compression level of 3, and 1 nearest neighbor:
println!("{:?}", classify(&training, &training_labels, &queries, 3i32, CompressionAlgorithm::Gzip, 1usize));
```

This method seems to perform decently well for relatively sparse training sets,
and does not require the same amount of tuning as neural net methods.

```rust
use csv::Reader;
use lrtc::{classify, CompressionAlgorithm};
use std::fs::File;
let imdb = File::open("./data/imdb.csv").unwrap();
let mut reader = Reader::from_reader(imdb);

    let imdb = File::open("./data/imdb.csv").unwrap();
    let mut reader = Reader::from_reader(imdb);

    let mut content = Vec::with_capacity(50000);
    let mut label = Vec::with_capacity(50000);
    for record in reader.records() {
        content.push(record.as_ref().unwrap()[0].to_string());
        label.push(record.unwrap()[1].to_string());
    }

let predictions = classify(
    &content[0..1000],
    &label[0..1000],
    &content[40000..50000],
    3i32,
    CompressionAlgorithm::Zstd,
    5usize,
)

let correct = predictions
    .iter()
    .zip(label[40000..50000].to_vec().iter())
    .filter(|(a, b)| a == b)
    .count();
println!("{}", correct as f64 / 1000f64)
// 0.623
```

## References
Zhiying Jiang, Matthew Yang, Mikhail Tsirlin, Raphael Tang, Yiqin Dai, and Jimmy Lin. 
2023. “Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors. 
In Findings of the Association for Computational Linguistics: ACL 2023, pages 6810–6828, Toronto, Canada. 
Association for Computational Linguistics. <https://aclanthology.org/2023.findings-acl.426>