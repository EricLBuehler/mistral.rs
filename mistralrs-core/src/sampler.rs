#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use core::f64;
use std::collections::HashMap;

use candle_core::{DType, Device, Error, IndexOp, Result, Tensor, D};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::layers_masker::masked_fill;

#[derive(Clone, Debug)]
/// Stop sequences or ids.
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

#[derive(Clone, Debug)]
/// Sampling params are used to control sampling.
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub top_n_logprobs: usize,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_toks: Option<StopTokens>,
    pub max_len: Option<usize>,
    pub logits_bias: Option<HashMap<u32, f32>>,
    pub n_choices: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            stop_toks: None,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
        }
    }
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Top-n logprobs element
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub token: u32,
    pub logprob: f32,
    pub bytes: String,
    pub top_logprobs: Option<Vec<TopLogprob>>,
}

#[derive(Clone)]
pub struct SamplingMetadata<'a> {
    pub output_tokens_tensor: Tensor,
    pub presence_penalties: Tensor,
    pub freq_penalties: Tensor,
    pub topk: Tensor,
    pub topp: Tensor,
    pub minp: Tensor,
    pub temperature: Tensor,
    pub tokenizers: Vec<&'a Tokenizer>,
}

pub struct NewSampler;

impl NewSampler {
    pub fn sample_on_gpu(
        &self,
        logits: Tensor,
        metadata: SamplingMetadata,
    ) -> Result<Vec<Logprobs>> {
        let logits = logits.to_dtype(DType::F32)?.squeeze(1)?;

        let logits = apply_penalties(
            logits,
            &metadata.output_tokens_tensor,
            &metadata.presence_penalties,
            &metadata.freq_penalties,
        )?;

        let logits = logits.broadcast_div(&metadata.temperature.to_dtype(DType::F32)?)?;

        let logits = apply_topk_topp(&logits, &metadata.topp, &metadata.topk)?;
        let logits = apply_minp(&logits, &metadata.minp)?;

        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let log_probs = probs.log()?;

        let sampled = sample_gumbel(&logits)?;

        let sampled_toks = sampled.to_vec1::<u32>()?;
        let logprobs = log_probs.to_vec1::<f32>()?;

        let mut logprobs_collector = Vec::new();
        for (tok, (logprob, tokenizer)) in sampled_toks
            .into_iter()
            .zip(logprobs.into_iter().zip(metadata.tokenizers))
        {
            logprobs_collector.push(Logprobs {
                token: tok,
                logprob,
                top_logprobs: None, // TODO
                bytes: tokenizer
                    .decode(&[tok], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            });
        }
        Ok(logprobs_collector)
    }
}

fn get_bin_counts_and_mask(
    tokens: &Tensor,
    vocab_size: usize,
    num_seqs: usize,
) -> Result<(Tensor, Tensor)> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L179
    let bin_counts = Tensor::zeros((num_seqs, vocab_size + 1), DType::I64, tokens.device())?;
    let bin_counts = if !tokens.dims().contains(&0) {
        bin_counts.scatter_add(tokens, &tokens.ones_like()?, 1)?
    } else {
        bin_counts
    };
    let bin_counts = bin_counts.i((.., ..vocab_size))?;
    let mask = bin_counts.gt(0f64)?;
    Ok((bin_counts, mask))
}

fn apply_penalties(
    mut logits: Tensor,
    output_tokens_tensor: &Tensor,
    presence_penalties: &Tensor,
    freq_penalties: &Tensor,
) -> Result<Tensor> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L243
    let (num_seqs, vocab_size) = logits.dims2()?;
    let (output_bin_counts, output_mask) =
        get_bin_counts_and_mask(output_tokens_tensor, vocab_size, num_seqs)?;

    logits = (logits - freq_penalties.broadcast_mul(&output_bin_counts.to_dtype(DType::F32)?)?)?;
    logits = (logits - presence_penalties.broadcast_mul(&output_mask.to_dtype(DType::F32)?)?)?;
    Ok(logits)
}

fn apply_topk_topp(logits: &Tensor, p: &Tensor, k: &Tensor) -> Result<Tensor> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L266
    let (logits_sort, logits_idx) = logits.to_device(&Device::Cpu)?.sort_last_dim(true)?;
    // TODO(EricLBuehler): Can we avoid this GPU <> CPU sync? This is the big one.
    let logits_sort = logits_sort.to_device(logits.device())?;
    let logits_idx = logits_idx.to_device(logits.device())?;

    // Apply topk
    let topk_mask =
        (k.to_dtype(DType::F32)?.neg()? + logits_sort.dim(1)? as f64)?.to_dtype(DType::U32)?;
    // Get all topk values
    let topk_mask = logits_sort.gather(&topk_mask, 1)?;
    let topk_mask = logits_sort.broadcast_lt(&topk_mask)?;
    let logits_sort = masked_fill(&logits_sort, &topk_mask, f64::NEG_INFINITY)?;

    // Apply topp
    let probs_sort = candle_nn::ops::softmax_last_dim(&logits_sort)?;
    let probs_sum = probs_sort.cumsum(D::Minus1)?;
    let topp_mask = probs_sum.broadcast_le(&(p.unsqueeze(1)?.neg()? + 1f64)?)?;
    // At least one
    // Equivalent of Pytorch `topp_mask[:, -1] = False`
    let topp_mask = topp_mask.slice_assign(
        &[&.., &(topk_mask.dim(1)? - 1..)],
        &Tensor::zeros(
            (topk_mask.dim(0)?, 1),
            topp_mask.dtype(),
            topp_mask.device(),
        )?,
    )?;
    let logits_sort = masked_fill(&logits_sort, &topp_mask, f64::NEG_INFINITY)?;

    // Resort the probs
    let src = Tensor::arange(0, logits_idx.dim(D::Minus1)? as u32, logits_idx.device())?;
    let logits_idx_inv = logits_idx
        .zeros_like()?
        .scatter_add(&logits_idx, &src, D::Minus1)?;

    logits_sort.gather(&logits_idx_inv, D::Minus1)
}

fn apply_minp(logits: &Tensor, minp: &Tensor) -> Result<Tensor> {
    // https://github.com/vllm-project/vllm/blob/740374d456a638df98ffbc7d9dab328752330e62/vllm/model_executor/layers/sampler.py#L298
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let top_probs = probs.max_keepdim(D::Minus1)?;
    let scaled_minp = (minp.unsqueeze(1)? * top_probs)?;
    let toks_to_remove = probs.lt(&scaled_minp)?;
    masked_fill(logits, &toks_to_remove, f64::NEG_INFINITY)
}

// (bs, vocab) -> (bs)
/// Approximation of multinomial
fn sample_gumbel(probs: &Tensor) -> Result<Tensor> {
    // https://github.com/lucidrains/speculative-decoding/blob/main/speculative_decoding/speculative_decoding.py#L36
    let uniform_dist = Tensor::rand(0f32, 1f32, probs.shape(), probs.device())?;
    let gumbel_noise = uniform_dist.log()?.neg()?.log()?.neg()?;
    // Already applied temperature, no need to again
    candle_nn::ops::softmax_last_dim(&(probs + gumbel_noise)?)?.argmax(D::Minus1)
}

mod tests {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use tokenizers::Tokenizer;

    #[allow(dead_code)]
    fn get_tokenizer() -> Tokenizer {
        let api = ApiBuilder::new().with_progress(true).build().unwrap();
        let api = api.repo(Repo::with_revision(
            "EricB/mistralrs_tests".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = api.get("tokenizer.json").unwrap();
        Tokenizer::from_file(tokenizer_filename).unwrap()
    }
}
