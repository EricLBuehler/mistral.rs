#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{cmp::Ordering, sync::Arc};

use anyhow::{Context, Result};
use bm25::{Embedder as Bm25Embedder, EmbedderBuilder, Language, Scorer};
use candle_core::{DType, Device, Error as E};
use mistralrs_quant::log::once_log_info;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;

use crate::pipeline::ForwardInputsResult;
use crate::{
    embedding_models::inputs_processor::{make_prompt_chunk, ModelInputs},
    engine::SearchEmbeddingModel,
    get_mut_arcmutex, AutoDeviceMapParams, DeviceMapSetting, EmbeddingLoaderBuilder,
    EmbeddingSpecificConfig, ModelDType, Pipeline, TokenSource,
};

use super::SearchResult;

const EMBEDDING_BATCH: usize = 64;
/// Target chunk size in tokens. Smaller chunks give better granularity and
/// reduce total tokens sent to the embedding model.
const CHUNK_TARGET_TOKENS: usize = 500;
/// Number of top BM25-scored chunks to pass to the neural re-ranker.
const BM25_TOP_K: usize = 20;

pub struct SearchPipeline {
    model: Arc<TokioMutex<dyn Pipeline + Send + Sync>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    has_causal_attention: bool,
    max_seq_len: usize,
}

#[derive(Debug, Clone)]
struct DocumentChunk {
    prompt: String,
    content: String,
    token_len: usize,
}

#[derive(Debug, Clone)]
pub struct ScoredChunk {
    pub result_index: usize,
    pub content: String,
    pub token_len: usize,
    pub score: f32,
}

impl SearchPipeline {
    pub fn new(model: SearchEmbeddingModel, runner_device: &Device) -> anyhow::Result<Self> {
        let model_id = model.hf_model_id().to_string();

        once_log_info(format!("Loading embedding model ({model_id})."));

        let loader = EmbeddingLoaderBuilder::new(
            EmbeddingSpecificConfig::default(),
            None,
            Some(model_id.clone()),
        )
        .build(None);

        let pipeline = loader.load_model_from_hf(
            None,
            TokenSource::CacheToken,
            &ModelDType::Auto,
            runner_device,
            true,
            DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
            None,
            None,
        )?;

        let guard = get_mut_arcmutex!(pipeline);
        let tokenizer = guard
            .tokenizer()
            .with_context(|| "Embedding model did not expose a tokenizer")?
            .clone();
        let device = guard.device();
        let max_seq_len = guard.get_metadata().max_seq_len;
        drop(guard);

        Ok(Self {
            model: pipeline,
            tokenizer,
            device,
            has_causal_attention: false,
            max_seq_len,
        })
    }

    fn embed(&mut self, prompts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        let encoded: Vec<(usize, Vec<u32>)> = prompts
            .iter()
            .enumerate()
            .map(|(idx, prompt)| {
                let encoding = self
                    .tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(E::msg)?;
                Ok((idx, encoding.get_ids().to_vec()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut outputs = vec![Vec::new(); prompts.len()];
        // Batch all prompts together — make_prompt_chunk handles variable-length
        // sequences via flash attention cumulative sequence lengths.
        for chunk_entries in encoded.chunks(EMBEDDING_BATCH) {
            let slices: Vec<&[u32]> = chunk_entries
                .iter()
                .map(|(_, ids)| ids.as_slice())
                .collect();
            let chunk = make_prompt_chunk(
                0,
                slices,
                &self.device,
                None,
                self.has_causal_attention,
                None,
            )?;
            let inputs = Box::new(ModelInputs {
                input_ids: chunk.input,
                flash_meta: chunk.flash_meta,
            });
            let mut pipeline = get_mut_arcmutex!(self.model);
            let ForwardInputsResult::Embeddings { embeddings } =
                pipeline.forward_inputs(inputs, false)?
            else {
                anyhow::bail!("Embedding pipeline returned non-embedding output");
            };
            drop(pipeline);
            let vecs = embeddings
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            for ((idx, _), embedding) in chunk_entries.iter().zip(vecs) {
                outputs[*idx] = embedding;
            }
        }

        Ok(outputs)
    }
}

impl SearchPipeline {
    fn chunk_document(&self, sanitized_title: &str, text: &str) -> Result<Vec<DocumentChunk>> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        let prefix_prompt = format_document_prompt(sanitized_title, "");
        let prefix_tokens = self
            .tokenizer
            .encode(prefix_prompt.as_str(), true)
            .map_err(E::msg)?
            .len();
        if self.max_seq_len <= prefix_tokens {
            return Ok(vec![DocumentChunk {
                prompt: format_document_prompt(sanitized_title, trimmed),
                content: trimmed.to_string(),
                token_len: self.tokenizer.encode(trimmed, false).map_err(E::msg)?.len(),
            }]);
        }

        let budget = CHUNK_TARGET_TOKENS
            .min(self.max_seq_len - prefix_tokens)
            .max(1);
        let encoding = self.tokenizer.encode(trimmed, false).map_err(E::msg)?;
        let token_count = encoding.len();
        if token_count == 0 {
            return Ok(Vec::new());
        }

        let offsets = encoding.get_offsets();
        let mut chunks = Vec::new();
        let mut start_idx = 0;

        while start_idx < token_count {
            let mut end_idx = (start_idx + budget).min(token_count);
            let mut accepted: Option<(DocumentChunk, usize)> = None;

            while end_idx > start_idx {
                let range = offsets[start_idx].0..offsets[end_idx - 1].1;
                let chunk_slice = trimmed[range.clone()].trim();
                if chunk_slice.is_empty() {
                    end_idx -= 1;
                    continue;
                }

                let candidate = format_document_prompt(sanitized_title, chunk_slice);
                let candidate_len = self
                    .tokenizer
                    .encode(candidate.as_str(), true)
                    .map_err(E::msg)?
                    .len();

                if candidate_len <= self.max_seq_len {
                    let token_len = end_idx - start_idx;
                    accepted = Some((
                        DocumentChunk {
                            prompt: candidate,
                            content: chunk_slice.to_string(),
                            token_len,
                        },
                        end_idx,
                    ));
                    break;
                }
                end_idx -= 1;
            }

            if let Some((chunk, next_idx)) = accepted {
                chunks.push(chunk);
                start_idx = next_idx;
                continue;
            }

            let single_range = offsets[start_idx].0..offsets[start_idx].1;
            let chunk_slice = trimmed[single_range].trim();
            if !chunk_slice.is_empty() {
                chunks.push(DocumentChunk {
                    prompt: format_document_prompt(sanitized_title, chunk_slice),
                    content: chunk_slice.to_string(),
                    token_len: 1,
                });
            }
            start_idx += 1;
        }

        Ok(chunks)
    }
}

fn sanitize_title(title: &str) -> String {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        "none".to_string()
    } else {
        trimmed.to_string()
    }
}

fn format_query_prompt(query: &str) -> String {
    format!("task: search result | query: {}", query.trim())
}

fn format_document_prompt(title: &str, text: &str) -> String {
    format!("title: {title} | text: {}", text.trim())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn rank_document_chunks(
    query: &str,
    results: &[SearchResult],
    pipeline: &mut SearchPipeline,
) -> Result<Vec<ScoredChunk>> {
    if results.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Chunk all documents.
    let mut bindings: Vec<(usize, DocumentChunk)> = Vec::new();
    for (result_index, result) in results.iter().enumerate() {
        let title = sanitize_title(&result.title);
        bindings.extend(
            pipeline
                .chunk_document(&title, &result.content)?
                .into_iter()
                .map(|chunk| (result_index, chunk)),
        );
    }

    if bindings.is_empty() {
        return Ok(Vec::new());
    }

    // 2. BM25 pre-filter: score all chunks by keyword overlap (CPU, instant), keep only top-K for the expensive neural re-ranking step.
    let chunk_texts: Vec<&str> = bindings.iter().map(|(_, c)| c.content.as_str()).collect();
    let bm25_embedder: Bm25Embedder =
        EmbedderBuilder::with_fit_to_corpus(Language::English, &chunk_texts).build();

    let mut bm25_scorer = Scorer::<usize>::new();
    for (i, text) in chunk_texts.iter().enumerate() {
        bm25_scorer.upsert(&i, bm25_embedder.embed(text));
    }

    let bm25_query = bm25_embedder.embed(query);
    let mut bm25_scores: Vec<(usize, f32)> = (0..chunk_texts.len())
        .filter_map(|i| bm25_scorer.score(&i, &bm25_query).map(|s| (i, s)))
        .collect();
    bm25_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let top_indices: Vec<usize> = bm25_scores
        .iter()
        .take(BM25_TOP_K)
        .map(|(i, _)| *i)
        .collect();

    tracing::info!(
        "Search: {} chunks from {} results, BM25 selected top {}",
        bindings.len(),
        results.len(),
        top_indices.len()
    );

    if top_indices.is_empty() {
        return Ok(Vec::new());
    }

    // 3. Neural re-ranking: embed query + only the BM25 top-K chunks.
    let query_embedding = pipeline
        .embed(&[format_query_prompt(query)])?
        .into_iter()
        .next()
        .context("Failed to generate embedding for search query")?;

    let top_prompts: Vec<String> = top_indices
        .iter()
        .map(|&i| bindings[i].1.prompt.clone())
        .collect();
    let top_embeddings = pipeline.embed(&top_prompts)?;

    let mut scored: Vec<ScoredChunk> = top_indices
        .iter()
        .zip(top_embeddings)
        .map(|(&i, embedding)| {
            let (result_index, ref chunk) = bindings[i];
            let score = cosine_similarity(&query_embedding, &embedding);
            ScoredChunk {
                result_index,
                content: chunk.content.clone(),
                token_len: chunk.token_len,
                score,
            }
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Less));
    Ok(scored)
}
