#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{borrow::Cow, cmp::Ordering};

use anyhow::Result;
use candle_core::{DType, Device, Error as E, Tensor};
use itertools::Itertools;
use tokenizers::{InputSequence, PaddingParams, PaddingStrategy, Tokenizer};

use crate::embedding::bert::{BertModel, BertPipeline};

use super::SearchResult;

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

/// Get the indexes of requests most similar to the query. In decreasing order
pub fn compute_most_similar(
    device: &Device,
    query: &str,
    results: Vec<&SearchResult>,
    BertPipeline { model, tokenizer }: &mut BertPipeline,
) -> Result<Vec<usize>> {
    let normalize_embeddings = false;

    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    let mut mean_similarities = Vec::new();
    for result in results {
        let mean_content_similarity = {
            let content = &result.content;
            let chunks = content
                .chars()
                .chunks(4096)
                .into_iter()
                .map(|chunk| chunk.collect::<String>())
                .collect::<Vec<_>>();
            let sentences = [vec![query.to_string()], chunks].concat();
            #[cfg(feature = "metal")]
            let similarities = objc::rc::autoreleasepool(|| -> Result<Vec<f32>> {
                compute_similarities(model, tokenizer, device, sentences, normalize_embeddings)
            })?;
            #[cfg(not(feature = "metal"))]
            let similarities =
                compute_similarities(model, tokenizer, device, sentences, normalize_embeddings)?;
            similarities.iter().sum::<f32>() / similarities.len() as f32
        };

        let title_similarity = {
            let title = &result.title;
            let sentences = vec![query.to_string(), title.to_string()];
            #[cfg(feature = "metal")]
            let similarities = objc::rc::autoreleasepool(|| -> Result<Vec<f32>> {
                compute_similarities(model, tokenizer, device, sentences, normalize_embeddings)
            })?;
            #[cfg(not(feature = "metal"))]
            let similarities =
                compute_similarities(model, tokenizer, device, sentences, normalize_embeddings)?;
            similarities.iter().sum::<f32>() / similarities.len() as f32
        };
        mean_similarities.push(title_similarity * 2. + mean_content_similarity);
    }

    let mut indexed: Vec<(usize, f32)> = mean_similarities.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Less));
    let ordered_indexes: Vec<usize> = indexed.into_iter().map(|(i, _)| i).collect();

    Ok(ordered_indexes)
}

fn compute_similarities(
    model: &BertModel,
    tokenizer: &Tokenizer,
    device: &Device,
    sentences: Vec<String>,
    normalize_embeddings: bool,
) -> Result<Vec<f32>> {
    let n_sentences = sentences.len();
    let sentences_batched = sentences
        .iter()
        .map(|s| InputSequence::Raw(Cow::from(s)))
        .collect::<Vec<_>>();

    let tokens = tokenizer
        .encode_batch(sentences_batched.to_vec(), true)
        .map_err(E::msg)?;
    let mut embeddings_all = Vec::new();
    for tokens in tokens.chunks(2) {
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?
            .to_dtype(DType::F32)?;
        embeddings_all.push(embeddings)
    }
    let embeddings = Tensor::cat(&embeddings_all, 0)?;
    drop(embeddings_all);

    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    let embeddings = if normalize_embeddings {
        normalize_l2(&embeddings)?
    } else {
        embeddings
    };

    let query_embedding = embeddings.get(0)?;
    let mut similarities = vec![];
    for j in 1..n_sentences {
        let e_j = embeddings.get(j)?;
        let sum_ij = (&query_embedding * &e_j)?.sum_all()?.to_scalar::<f32>()?;
        let sum_i2 = (&query_embedding * &query_embedding)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
        let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
        similarities.push(cosine_similarity)
    }

    Ok(similarities)
}
