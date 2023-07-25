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
//! let training = vec!["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
//! let training_labels = vec!["normal".to_string(), "godzilla".into(),];
//! let queries = vec!["another normal sentence".to_string(), "godzilla eats marshes in August".into(),];
//! // Using a compression level of 3, and 1 nearest neighbor:
//! println!("{:?}", classify(training, training_labels, queries, 3i32, 1usize));
//! ```

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::string::String;
use zstd::bulk::compress;

/// Training data struct
///
/// This struct pairs training content with its label, and optionally with its length when compressed.
///
/// # Examples
///
/// ```
/// let out = TrainingData {label = "godzilla".to_string(), content = "godzilla ate mars in June".to_string()};
/// println!{"{:?}", out};
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingData {
    label: String,
    content: String,
    compressed_length: Option<usize>,
}

/// NCD struct
///
/// This struct pairs training labels with their calculated NCD for a single query string.
/// This enables easier sorting and nearest-neighbor matching.
///
/// # Examples
///
/// ```
/// let out = NCD {label = "godzilla".to_string(), ncd = 0.5f64};
/// println!{"{:?}", out};
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NCD {
    label: String,
    ncd: f64,
}

/// Calculate the length of an input string once compressed
///
/// Currently this function uses zstd for compression, at level `level`.
///
/// # Examples
///
/// ```
/// let out = compressed_length(&"godzilla eats marshes in August".to_string(), 3i32);
/// println!{"{:?}", out};
/// ```
pub fn compressed_length(training: &String, level: i32) -> usize {
    let compressed = compress(training.as_bytes(), level).unwrap();
    compressed.len()
}

/// Calculate a vector of NCD values for a given query
///
/// # Examples
///
/// ```
/// let training = vec!["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
/// let training_labels = vec!["normal".to_string(), "godzilla".into(),];
/// let query = "another normal sentence".to_string();
/// let training_data = training
///        .iter()
///        .zip(training_labels.iter())
///        .map(|(content, label)| TrainingData {
///            label: label.clone(),
///            content: content.clone(),
///            compressed_length: Some(compressed_length(content, level)),
///        })
///        .collect::<Vec<TrainingData>>()
///
/// let out = ncd(training_data, query, 3i32)
/// println!{"{:?}", out};
/// ```
pub fn ncd(training_data: Vec<TrainingData>, query: String, level: i32) -> Vec<NCD> {
    let len_training = training_data
        .par_iter()
        .map(|td| {
            td.compressed_length
                .unwrap_or_else(|| compressed_length(&td.content, level))
        })
        .collect::<Vec<usize>>();
    let len_query = compressed_length(&query, level);

    let len_combo = training_data
        .par_iter()
        .map(|td| compressed_length(&format!("{} {}", td.content, query), level))
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
            label: td.label.clone(),
            ncd: *ncd,
        })
        .collect()
}

/// Classify sentences based on their distance from a set of labeled training data.
///
/// # Examples
///
/// ```
/// let training = vec!["some normal sentence".to_string(), "godzilla ate mars in June".into(),];
/// let training_labels = vec!["normal".to_string(), "godzilla".into(),];
/// let queries = vec!["another normal sentence".to_string(), "godzilla eats marshes in August".into(),];
/// // Using a compression level of 3, and 1 nearest neighbor:
/// println!("{:?}", classify(training, training_labels, queries, 3i32, 1usize));
/// ```
pub fn classify(
    training: Vec<String>,
    training_labels: Vec<String>,
    queries: Vec<String>,
    level: i32,
    k: usize,
) -> Vec<String> {
    let training_data = training
        .par_iter()
        .zip(training_labels.par_iter())
        .map(|(content, label)| TrainingData {
            label: label.clone(),
            content: content.clone(),
            compressed_length: Some(compressed_length(content, level)),
        })
        .collect::<Vec<TrainingData>>();

    queries
        .par_iter()
        .map(|query| {
            let mut ncds = ncd(training_data.clone(), query.clone(), level);

            ncds.sort_by(|a, b| a.ncd.total_cmp(&b.ncd));
            ncds[0..k]
                .iter()
                .map(|x| x.label.clone())
                .collect::<Vec<String>>()
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
            classify(training, training_labels, queries, 3i32, 1usize),
            vec!["a".to_string(), "b".into()]
        );
    }
}
