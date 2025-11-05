#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{cmp::Ordering, sync::Arc};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Error as E};
use itertools::Itertools;
use mistralrs_quant::log::once_log_info;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;

use crate::pipeline::ForwardInputsResult;
use crate::{
    embedding_models::inputs_processor::{make_prompt_chunk, ModelInputs},
    engine::BertEmbeddingModel,
    get_mut_arcmutex, AutoDeviceMapParams, DeviceMapSetting, EmbeddingLoaderBuilder,
    EmbeddingSpecificConfig, ModelDType, Pipeline, TokenSource,
};

use super::SearchResult;

const DEFAULT_EMBED_MODEL_ID: &str = "google/embeddinggemma-300m";
const DOC_CHARS_PER_CHUNK: usize = 4096;
const EMBEDDING_BATCH: usize = 8;

pub struct SearchPipeline {
    model: Arc<TokioMutex<dyn Pipeline + Send + Sync>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    has_causal_attention: bool,
}

impl SearchPipeline {
    pub fn new(model: BertEmbeddingModel, runner_device: &Device) -> anyhow::Result<Self> {
        let model_id = match model {
            BertEmbeddingModel::EmbeddingGemma300M => DEFAULT_EMBED_MODEL_ID.to_string(),
            BertEmbeddingModel::Custom(id) => id,
        };

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
        drop(guard);

        Ok(Self {
            model: pipeline,
            tokenizer,
            device,
            has_causal_attention: false,
        })
    }

    fn embed(&mut self, prompts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        use std::collections::BTreeMap;
        let mut by_len: BTreeMap<usize, Vec<(usize, Vec<u32>)>> = BTreeMap::new();
        for (idx, prompt) in prompts.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(E::msg)?;
            let ids = encoding.get_ids().to_vec();
            by_len.entry(ids.len()).or_default().push((idx, ids));
        }

        let mut outputs = vec![Vec::new(); prompts.len()];
        for (_, sequences) in by_len {
            for chunk_entries in sequences.chunks(EMBEDDING_BATCH) {
                let slices: Vec<&[u32]> = chunk_entries
                    .iter()
                    .map(|(_, ids)| ids.as_slice())
                    .collect();
                let chunk =
                    make_prompt_chunk(0, slices, &self.device, None, self.has_causal_attention)?;
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
                for ((idx, _), embedding) in chunk_entries.iter().zip(vecs.into_iter()) {
                    outputs[*idx] = embedding;
                }
            }
        }

        Ok(outputs)
    }
}

fn chunk_by_chars(text: &str) -> Vec<String> {
    text.chars()
        .chunks(DOC_CHARS_PER_CHUNK)
        .into_iter()
        .map(|chunk| chunk.collect::<String>())
        .collect()
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

/// Get the indexes of requests most similar to the query. In decreasing order
pub fn compute_most_similar(
    _device: &Device,
    query: &str,
    results: Vec<&SearchResult>,
    pipeline: &mut SearchPipeline,
) -> Result<Vec<usize>> {
    if results.is_empty() {
        return Ok(Vec::new());
    }

    let query_embedding = pipeline
        .embed(&[format_query_prompt(query)])?
        .into_iter()
        .next()
        .context("Failed to generate embedding for search query")?;

    let mut scored = Vec::with_capacity(results.len());
    for (idx, result) in results.iter().enumerate() {
        let title = sanitize_title(&result.title);

        let content_chunks = chunk_by_chars(&result.content);
        let content_prompts: Vec<String> = content_chunks
            .iter()
            .map(|chunk| format_document_prompt(&title, chunk))
            .collect();
        let content_embeddings = pipeline.embed(&content_prompts)?;
        let mean_content_similarity = if content_embeddings.is_empty() {
            0.0
        } else {
            content_embeddings
                .iter()
                .map(|emb| cosine_similarity(&query_embedding, emb))
                .sum::<f32>()
                / content_embeddings.len() as f32
        };

        let title_similarity = if result.title.trim().is_empty() {
            0.0
        } else {
            let title_prompt = format_document_prompt(&title, &result.title);
            let title_emb = pipeline
                .embed(&[title_prompt])?
                .into_iter()
                .next()
                .context("Failed to generate embedding for result title")?;
            cosine_similarity(&query_embedding, &title_emb)
        };

        let score = (2.0 * title_similarity) + mean_content_similarity;
        scored.push((idx, score));
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Less));
    Ok(scored.into_iter().map(|(idx, _)| idx).collect())
}
