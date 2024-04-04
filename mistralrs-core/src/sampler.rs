#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::ops::Add;

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_ext::F;

#[derive(Clone, Debug)]
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_n_logprobs: usize,
    pub freq_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_toks: Option<StopTokens>,
    pub max_len: Option<usize>,
    pub logits_bias: Option<HashMap<u32, f32>>,
    pub n_choices: usize,

    // mistral.rs
    pub top_k: Option<i64>,
}

struct BinCountsAndMask {
    bin_counts: Tensor,
    mask: Tensor,
}

/// Sampler for sampling.
#[derive(Clone)]
pub struct Sampler {
    params: SamplingParams,
    vocab_size: usize,
}

#[derive(Debug, Clone)]
// Top-n logprobs element
pub struct TopLogprob {
    pub token: Tensor,
    pub logprob: f32,
    pub bytes: String,
}

#[derive(Debug, Clone)]
pub struct Logprobs {
    pub token: Tensor,
    pub logprob: f32,
    pub bytes: String,
    pub top_logprobs: Vec<TopLogprob>,
}

impl Sampler {
    pub fn new(_seed: u64, params: SamplingParams, vocab_size: usize) -> Self {
        Self { params, vocab_size }
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    /// If `freq_penalty.is_some()` or `presence_penalty.is_some()`, then `penalty_ctxt` must be provided.
    pub fn sample(&mut self, logits: Tensor, _penalty_ctxt: Option<&Tensor>) -> Result<Logprobs> {
        let logits = logits.to_dtype(DType::F32)?;

        let next_token = match self.params.temperature {
            None => logits.argmax(D::Minus1)?,
            Some(temperature) => {
                // Apply penalties
                let logits = self.apply_penalties(logits)?;

                // Apply temperature scaling
                let logits = (&logits / temperature)?;
                let logits = candle_nn::ops::softmax_last_dim(&logits)?;

                // Apply topk, topp
                //let logits = self.apply_topk_topp(logits)?;
                logits.argmax(D::Minus1)?
            }
        };

        Ok(Logprobs {
            token: next_token,
            logprob: 1.0,
            bytes: "".to_string(),
            top_logprobs: vec![],
        })
    }

    // https://github.com/vllm-project/vllm/blob/2ff767b51301e07d1e0ad5887eb26e104e2b3a8a/vllm/model_executor/layers/sampler.py#L85
    fn get_bin_counts_and_mask(
        &self,
        logits: &Tensor,
        vocab_size: usize,
    ) -> Result<BinCountsAndMask> {
        // Use vocab_size + 1 to account for padding
        let bin_counts = Tensor::zeros(vocab_size, DType::I64, logits.device())?
            .scatter_add(&logits, &logits.ones_like()?, 0)?
            .i((.., ..vocab_size))?;
        let mask = bin_counts.ge(0i64)?;
        Ok(BinCountsAndMask { bin_counts, mask })
    }

    fn apply_penalties(&self, logits: Tensor) -> Result<Tensor> {
        let BinCountsAndMask { bin_counts, mask } =
            self.get_bin_counts_and_mask(&logits, self.vocab_size)?;
        dbg!(&bin_counts);
        dbg!(&mask);
        dbg!(&logits);
        let presence_penalties =
            Tensor::new(self.params.presence_penalty.unwrap(), logits.device())?;
        let freq_penalties = Tensor::new(self.params.freq_penalty.unwrap(), logits.device())?;
        let logits = (logits
            - &presence_penalties
                .unsqueeze(1)?
                .broadcast_mul(&bin_counts.to_dtype(DType::F32)?)?)?;
        let logits = (logits - &freq_penalties.unsqueeze(1)?.broadcast_mul(&mask.to_dtype(DType::F32)?)?)?;
        Ok(logits)
    }

    fn sort_ascending(&self, _logits: &Tensor) -> (Tensor, Tensor) {
        todo!()
    }

    fn apply_topk_topp(&self, logits: Tensor) -> Result<Tensor> {
        let (logits_sort, logits_idx) = self.sort_ascending(&logits);

        // TOPK
        // Apply topk
        let topk = self.params.top_k.unwrap_or(self.vocab_size as i64);
        let k = Tensor::new(topk, logits.device())?.repeat(logits.dims1()?)?;
        let topk_mask = k.neg()?.add(logits_sort.dim(0)? as f64)?;
        // Get all the topk values
        let topk_mask = logits_sort.gather(&topk_mask, 1)?;
        let topk_mask = logits_sort.broadcast_lt(&topk_mask)?;
        let logits_sort = F::masked_fill(&logits_sort, &topk_mask, f64::NEG_INFINITY)?;

        // TOPP
        let probs_sort = candle_nn::ops::softmax_last_dim(&logits_sort)?;
        let probs_sum = probs_sort.cumsum(D::Minus1)?;
        let p = Tensor::new(topk, logits.device())?.repeat(logits.dims1()?)?;
        let topp_mask = probs_sum.broadcast_le(&(p.neg()? + 1.)?)?;
        // at least one
        topp_mask.slice_assign(
            &[(topp_mask.dim(0)? - 1)..],
            &Tensor::new(0f64, logits.device())?,
        )?;
        let logits_sort = F::masked_fill(&logits_sort, &topp_mask, f64::NEG_INFINITY)?;

        // Re sort probs
        let src = Tensor::arange(0, logits_idx.dim(D::Minus1)? as i64, logits.device())?
            .expand(logits_idx.shape())?;
        let logits_idx_inv = F::scatter(
            &unsafe { logits_idx.empty_like()? },
            &logits_idx,
            &src,
            D::Minus1,
        )?;

        logits_sort.gather(&logits_idx_inv, D::Minus1)
    }
}
