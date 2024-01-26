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
//! use lrtc::{Classifier, CompressionAlgorithm};
//!
//! let training = vec!["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
//! let training_labels = vec!["normal".to_string(), "godzilla".into(),];
//! let queries = vec!["another normal sentence".to_string(), "godzilla eats marshes in August".into(),];
//! // Using a compression level of 3, and 1 nearest neighbor:
//! let mut classifier = Classifier::new(CompressionAlgorithm::Gzip, 3);
//! classifier.train(&training, &training_labels);
//! println!("{:?}", classifier.classify(&queries[0], 1usize));
//! ```
use std::cmp::{max, min};
use std::collections::HashMap;
use std::io::Write;
use std::string::String;

use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
use flate2::Compression;
use serde::{Deserialize, Serialize};
use zstd::bulk::compress;

/// Configuration of text classifier
pub struct Classifier {
    /// Compression algorithm
    algorithm: CompressionAlgorithm,
    /// Compression level
    level: i32,
    /// The class label of each observation. These are the values that will be returned.
    labels: Vec<String>,
    /// The text content of the observation. This is the value distance calculations rely on.
    content: Vec<String>,
    /// The length of `content` when compressed. This obviously depends on the algorithm used
    /// (and compression level).
    compressed_lengths: Vec<usize>,
}

impl Classifier {
    /// Configure classifier
    pub fn new(algorithm: CompressionAlgorithm, level: i32) -> Classifier {
        Classifier {
            algorithm,
            level,
            labels: Vec::new(),
            content: Vec::new(),
            compressed_lengths: Vec::new(),
        }
    }
    /// Train the classifier on samples, can also be done incrementally
    pub fn train<S: AsRef<str>>(&mut self, content: &[S], labels: &[S]) {
        for (label, content) in labels.iter().zip(content.iter()) {
            let l = label.as_ref().to_string();
            let c = content.as_ref().to_string();
            self.compressed_lengths
                .push(compressed_length(&c, self.level, &self.algorithm));
            self.content.push(c);
            self.labels.push(l);
        }
    }
    /// Classify sentences based on their distance from a set of labeled training data.
    ///
    /// # Examples
    ///
    /// ```
    /// use lrtc::{Classifier, CompressionAlgorithm};
    ///
    /// let training = ["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
    /// let training_labels = ["normal".to_string(), "godzilla".into(),];
    /// let queries = ["another normal sentence".to_string(), "godzilla eats marshes in August".into(),];
    /// // Using a compression level of 3, and 1 nearest neighbor:
    /// let mut classifier = Classifier::new(CompressionAlgorithm::Gzip, 3);
    /// classifier.train(&training, &training_labels);
    /// println!("{:?}", classifier.classify(&queries[0], 1usize));
    /// ```
    pub fn classify(&self, query: &str, k: usize) -> String {
        let mut ncds = self.ncd(query.as_ref());

        ncds.sort_by(|a, b| a.ncd.total_cmp(&b.ncd));
        ncds[0..k]
            .iter()
            .map(|x| x.label)
            .fold(HashMap::<String, usize>::new(), |mut m, x| {
                *m.entry(x.to_string()).or_default() += 1;
                m
            })
            .into_iter()
            .max_by_key(|(_, v)| *v)
            .map(|(x, _)| x)
            .unwrap()
    }
    /// Calculate a vector of NCD values for a given query
    ///
    /// # Examples
    ///
    /// ```
    /// use lrtc::{Classifier, compressed_length, CompressionAlgorithm};
    ///
    /// let training = vec!["some normal sentence", "godzilla ate mars in June",];
    /// let training_labels = vec!["normal", "godzilla",];
    /// let query = "another normal sentence";
    /// let mut classifier = Classifier::new(CompressionAlgorithm::Gzip, 3);
    /// classifier.train(&training, &training_labels);
    /// println!("{:?}", classifier.classify(query, 1usize));
    /// let out = classifier.ncd(&query);
    /// println!{"{:?}", out};
    /// ```
    pub fn ncd(&self, query: &str) -> Vec<NCD<'_>> {
        let len_training = &self.compressed_lengths;
        let len_query = compressed_length(query, self.level, &self.algorithm);

        let len_combo = self
            .content
            .iter()
            .map(|content| {
                compressed_length(
                    &format!("{} {}", content, query),
                    self.level,
                    &self.algorithm,
                )
            })
            .collect::<Vec<usize>>();

        let minmaxs = len_training
            .iter()
            .map(|train_length| {
                (
                    *min(train_length, &len_query),
                    *max(train_length, &len_query),
                )
            })
            .collect::<Vec<(usize, usize)>>();

        len_combo
            .iter()
            .zip(minmaxs.iter())
            .zip(self.labels.iter())
            .map(|((c, (min, max)), label)| {
                let ncd = c.abs_diff(*min) as f64 / *max as f64;
                NCD { label, ncd }
            })
            .collect()
    }
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
#[derive(Serialize, Deserialize, Debug)]
pub struct NCD<'a> {
    /// The class label of the original training observation. These are the values that will be returned.
    pub label: &'a str,
    /// The NCD between the query point and the original training observation. Lower values imply closer strings.
    pub ncd: f64,
}

/// Available compression algorithms
#[derive(Serialize, Deserialize, Debug)]
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
/// let out = compressed_length("godzilla eats marshes in August", 3i32, &CompressionAlgorithm::Zstd);
/// println!{"{:?}", out};
/// ```
pub fn compressed_length(training: &str, level: i32, algorithm: &CompressionAlgorithm) -> usize {
    let compressed = match algorithm {
        CompressionAlgorithm::Zstd => compress(training.as_bytes(), level).unwrap(),
        CompressionAlgorithm::Gzip => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(training.as_bytes()).unwrap();
            encoder.finish().unwrap()
        }
        CompressionAlgorithm::Zlib => {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(training.as_bytes()).unwrap();
            encoder.finish().unwrap()
        }
        CompressionAlgorithm::Deflate => {
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(training.as_bytes()).unwrap();
            encoder.finish().unwrap()
        }
    };
    compressed.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use csv::Reader;
    use rayon::prelude::*;
    use std::fs::File;

    #[test]
    fn test_classification() {
        let training = [
            "some normal sentence".to_string(),
            "godzilla ate mars in June".into(),
        ];
        let training_labels = ["a".to_string(), "b".into()];
        let queries = [
            "another normal sentence".to_string(),
            "godzilla eats marshes in August".into(),
        ];

        let mut classifier = Classifier::new(CompressionAlgorithm::Gzip, 3);
        classifier.train(&training[..], &training_labels[..]);
        let predictions: Vec<String> = queries
            .par_iter()
            .map(|query| classifier.classify(query, 1))
            .collect();
        assert_eq!(predictions, vec!["a".to_string(), "b".into()]);
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

        let mut classifier = Classifier::new(CompressionAlgorithm::Zstd, 3);
        classifier.train(&content[0..5000], &label[0..5000]);
        let predictions: Vec<String> = content[5000..6000]
            .par_iter()
            .map(|query| classifier.classify(query, 1))
            .collect();
        let correct = predictions
            .iter()
            .zip(label[5000..6000].iter())
            .filter(|(a, b)| a == b)
            .count();
        assert_eq!(correct, 685usize)
    }
}
