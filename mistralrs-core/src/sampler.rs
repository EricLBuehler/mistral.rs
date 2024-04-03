#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use candle_core::{DType, Result, Tensor, D};
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<i64>,
    pub top_p: Option<f64>,
    pub top_n_logprobs: usize,
    pub freq_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_toks: Option<StopTokens>,
    pub max_len: Option<usize>,
    pub logits_bias: Option<HashMap<u32, f32>>,
    pub n_choices: usize,
}

/// Sampler for sampling.
#[derive(Clone)]
pub struct Sampler {
    rng: rand::rngs::StdRng,
    params: SamplingParams,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(seed: u64, params: SamplingParams) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            params,
        }
    }

    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    /// If `freq_penalty.is_some()` or `presence_penalty.is_some()`, then `penalty_ctxt` must be provided.
    pub fn sample(&mut self, logits: &Tensor, _penalty_ctxt: Option<&Tensor>) -> Result<Logprobs> {
        let logits = logits.to_dtype(DType::F32)?;

        let next_token = match self.params.temperature {
            None => logits.argmax(D::Minus1)?,
            Some(temperature) => {
                let logits = (&logits / temperature)?;
                let logits = candle_nn::ops::softmax_last_dim(&logits)?;
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
}
