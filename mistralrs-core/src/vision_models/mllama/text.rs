use candle_core::{Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Activation, Embedding, Linear, Module, VarBuilder};

use crate::layers::RmsNorm;

use super::config::MLlamaTextConfig;

struct MLlamaTextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl MLlamaTextMlp {
    fn new(cfg: &MLlamaTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down_proj.forward(
            &self
                .act
                .forward(&self.gate_proj.forward(xs)?)?
                .broadcast_mul(&self.up_proj.forward(xs)?)?,
        )
    }
}

pub(super) struct MLlamaTextModel {
    embed_tokens: Embedding,
    lm_head: Linear,
    norm: RmsNorm,
}

impl MLlamaTextModel {
    pub(super) fn new(cfg: &MLlamaTextConfig, vb: VarBuilder) -> Result<Self> {
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        let vb = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size + 8, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        todo!()
    }
}
