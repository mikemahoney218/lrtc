#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! Rust implementation of low-resource text classification
//!
//! This crate is a Rust implementation of Jiang et al (2023),
//! using text compressors to efficiently classify text snippets
//! via k-nearest neighbors.
//!
//! Full method citation:
//! Zhiying Jiang, Matthew Yang, Mikhail Tsirlin, Raphael Tang, Yiqin Dai, and Jimmy Lin.
//! 2023. “Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors.
//! In Findings of the Association for Computational Linguistics: ACL 2023, pages 6810–6828, Toronto, Canada.
//! Association for Computational Linguistics. <https://aclanthology.org/2023.findings-acl.426>
//!
//! # Examples
//!
//! ```
//! use lrtc::{CompressionAlgorithm, classify};
//!
//! let training = vec!["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
//! let training_labels = vec!["normal".to_string(), "godzilla".into(),];
//! let queries = vec!["another normal sentence".to_string(), "godzilla eats marshes in August".into(),];
//! // Using a compression level of 3, and 1 nearest neighbor:
//! println!("{:?}", classify(training, training_labels, queries, 3i32, CompressionAlgorithm::Gzip, 1usize));
//! ```

use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
use flate2::Compression;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::io::Write;
use std::string::String;
use zstd::bulk::compress;

/// Training data struct
///
/// This struct pairs training content with its label, and optionally with its length when compressed.
///
/// # Examples
///
/// ```
/// use lrtc::{TrainingData};
///
/// let out = TrainingData {label: "godzilla", content: "godzilla ate mars in June", compressed_length: None};
/// println!{"{:?}", out};
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingData<'a> {
    /// The class label of each observation. These are the values that will be returned.
    pub label: &'a str,
    /// The text content of the observation. This is the value distance calculations rely on.
    pub content: &'a str,
    /// The length of `content` when compressed. This obviously depends on the algorithm used
    /// (and compression level), and therefore should generally be `None` when manually creating
    /// TrainingData objects -- this field will be automatically populated by `ncd()`.
    pub compressed_length: Option<usize>,
}

/// NCD struct
///
/// This struct pairs training labels with their calculated NCD for a single query string.
/// This enables easier sorting and nearest-neighbor matching.
///
/// # Examples
///
/// ```
/// use lrtc::NCD;
///
/// let out = NCD {label: "godzilla", ncd: 0.5f64};
/// println!{"{:?}", out};
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NCD<'a> {
    /// The class label of the original training observation. These are the values that will be returned.
    pub label: &'a str,
    /// The NCD between the query point and the original training observation. Lower values imply closer strings.
    pub ncd: f64,
}

/// Available compression algorithms
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CompressionAlgorithm {
    /// Facebook's zstd library, provided by zstd
    Zstd,
    /// The classic gzip algorithm, provided by flate2
    Gzip,
    /// The zlib-ng algorithm, provided by flate2
    Zlib,
    /// The classic deflate algorithm, provided by flate2
    Deflate,
}

/// Calculate the length of an input string once compressed
///
/// Currently this function uses zstd for compression, at level `level`.
///
/// # Examples
///
/// ```
/// use lrtc::{CompressionAlgorithm, compressed_length};
///
/// let out = compressed_length(&"godzilla eats marshes in August".to_string(), 3i32, &CompressionAlgorithm::Zstd);
/// println!{"{:?}", out};
/// ```
pub fn compressed_length(training: &str, level: i32, algorithm: &CompressionAlgorithm) -> usize {
    let compressed = match algorithm {
        &CompressionAlgorithm::Zstd => compress(training.as_bytes(), level).unwrap(),
        &CompressionAlgorithm::Gzip => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(training.as_bytes()).unwrap();
            encoder.finish().unwrap()
        }
        &CompressionAlgorithm::Zlib => {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(training.as_bytes()).unwrap();
            encoder.finish().unwrap()
        }
        &CompressionAlgorithm::Deflate => {
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(training.as_bytes()).unwrap();
            encoder.finish().unwrap()
        }
    };
    compressed.len()
}

/// Calculate a vector of NCD values for a given query
///
/// # Examples
///
/// ```
/// use lrtc::{TrainingData, compressed_length, ncd, CompressionAlgorithm};
///
/// let training = vec!["some normal sentence", "godzilla ate mars in June",];
/// let training_labels = vec!["normal", "godzilla",];
/// let query = "another normal sentence";
/// let training_data = training
///        .iter()
///        .zip(training_labels.iter())
///        .map(|(content, label)| TrainingData {
///            label: label.clone(),
///            content: content.clone(),
///            compressed_length: Some(compressed_length(content, 3i32, &CompressionAlgorithm::Gzip)),
///        })
///        .collect::<Vec<TrainingData>>();
///
/// let out = ncd(&training_data, &query, 3i32, &CompressionAlgorithm::Zstd);
/// println!{"{:?}", out};
/// ```
pub fn ncd<'a>(
    training_data: &'a Vec<TrainingData<'a>>,
    query: &'a str,
    level: i32,
    algorithm: &'a CompressionAlgorithm,
) -> Vec<NCD<'a>> {
    let len_training = training_data
        .par_iter()
        .map(|td| {
            td.compressed_length
                .unwrap_or_else(|| compressed_length(&td.content, level, algorithm))
        })
        .collect::<Vec<usize>>();
    let len_query = compressed_length(query, level, algorithm);

    let len_combo = training_data
        .par_iter()
        .map(|td| compressed_length(&format!("{} {}", td.content, query), level, algorithm))
        .collect::<Vec<usize>>();

    let mins = len_training
        .par_iter()
        .map(|train_length| *min(train_length, &len_query))
        .collect::<Vec<usize>>();

    let maxes = len_training
        .par_iter()
        .map(|train_length| *max(train_length, &len_query))
        .collect::<Vec<usize>>();

    len_combo
        .par_iter()
        .zip(mins.par_iter())
        .map(|(c, m)| c - m)
        .collect::<Vec<usize>>()
        .par_iter()
        .zip(maxes.par_iter())
        .map(|(n, d)| *n as f64 / *d as f64)
        .collect::<Vec<f64>>()
        .par_iter()
        .zip(training_data.par_iter())
        .map(|(ncd, td)| NCD {
            label: td.label,
            ncd: *ncd,
        })
        .collect()
}

/// Classify sentences based on their distance from a set of labeled training data.
///
/// # Examples
///
/// ```
/// use lrtc::{CompressionAlgorithm, classify};
///
/// let training = vec!["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
/// let training_labels = vec!["normal".to_string(), "godzilla".into(),];
/// let queries = vec!["another normal sentence".to_string(), "godzilla eats marshes in August".into(),];
/// // Using a compression level of 3, and 1 nearest neighbor:
/// println!("{:?}", classify(training, training_labels, queries, 3i32, CompressionAlgorithm::Gzip, 1usize));
/// ```
pub fn classify(
    training: Vec<String>,
    training_labels: Vec<String>,
    queries: Vec<String>,
    level: i32,
    algorithm: CompressionAlgorithm,
    k: usize,
) -> Vec<String> {
    let training_data = training
        .par_iter()
        .zip(training_labels.par_iter())
        .map(|(content, label)| TrainingData {
            label: label,
            content: content,
            compressed_length: Some(compressed_length(content, level, &algorithm)),
        })
        .collect::<Vec<TrainingData>>();

    queries
        .par_iter()
        .map(|query| {
            let mut ncds = ncd(&training_data, query, level, &algorithm);

            ncds.sort_by(|a, b| a.ncd.total_cmp(&b.ncd));
            ncds[0..k]
                .iter()
                .map(|x| x.label)
                .collect::<Vec<&str>>()
                .iter()
                .fold(HashMap::<String, usize>::new(), |mut m, x| {
                    *m.entry(x.to_string()).or_default() += 1;
                    m
                })
                .into_par_iter()
                .max_by_key(|(_, v)| *v)
                .map(|(x, _)| x)
                .unwrap()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use csv::Reader;
    use std::fs::File;

    #[test]
    fn test_classification() {
        let training = vec![
            "some normal sentence".to_string(),
            "godzilla ate mars in June".into(),
        ];
        let training_labels = vec!["a".to_string(), "b".into()];
        let queries = vec![
            "another normal sentence".to_string(),
            "godzilla eats marshes in August".into(),
        ];

        assert_eq!(
            classify(
                training,
                training_labels,
                queries,
                3i32,
                CompressionAlgorithm::Gzip,
                1usize
            ),
            vec!["a".to_string(), "b".into()]
        );
    }

    #[test]
    fn csv_classifications() {
        let imdb = File::open("./data/imdb.csv").unwrap();
        let mut reader = Reader::from_reader(imdb);

        let mut content = Vec::with_capacity(50000);
        let mut label = Vec::with_capacity(50000);
        for record in reader.records() {
            content.push(record.as_ref().unwrap()[0].to_string());
            label.push(record.unwrap()[1].to_string());
        }

        let predictions = classify(
            content[0..5000].to_vec(),
            label[0..5000].to_vec(),
            content[5000..6000].to_vec(),
            3i32,
            CompressionAlgorithm::Zstd,
            1usize,
        );
        let correct = predictions
            .iter()
            .zip(label[5000..6000].to_vec().iter())
            .filter(|(a, b)| a == b)
            .count();
        assert_eq!(correct, 685usize)
    }
}
