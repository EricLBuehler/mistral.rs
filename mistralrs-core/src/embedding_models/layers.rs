use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::Module;
use serde::Deserialize;

/// Pooling layer
#[derive(Deserialize, Debug, Clone)]
pub struct Pooling {
    pub word_embedding_dimension: usize,
    pub pooling_mode_cls_token: bool,
    pub pooling_mode_mean_tokens: bool,
    pub pooling_mode_max_tokens: bool,
    pub pooling_mode_mean_sqrt_len_tokens: bool,
    pub pooling_mode_weightedmean_tokens: bool,
    pub pooling_mode_lasttoken: bool,
    pub include_prompt: bool,
}

impl Module for Pooling {
    // https://github.com/huggingface/sentence-transformers/blob/85ec64559f4414aa536eca4bf53538291e0a333f/sentence_transformers/models/Pooling.py#L26
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if !self.include_prompt {
            candle_core::bail!("Only support include_prompt==true");
        }
        if xs.dim(D::Minus1)? != self.word_embedding_dimension {
            candle_core::bail!("xs does not match the expected embedding dimension.");
        }

        let mut outputs = Vec::new();
        if self.pooling_mode_cls_token {
            unimplemented!();
        }
        if self.pooling_mode_mean_tokens {
            // Assume full attention mask. Otherwise this must be updated

            let sum_embeddings = xs.sum(1)?;
            outputs.push(sum_embeddings);
        }
        if self.pooling_mode_max_tokens {
            unimplemented!();
        }
        if self.pooling_mode_mean_sqrt_len_tokens {
            unimplemented!();
        }
        if self.pooling_mode_weightedmean_tokens {
            unimplemented!();
        }
        if self.pooling_mode_lasttoken {
            outputs.push(xs.i((.., xs.dim(D::Minus2)? - 1, ..))?);
        }

        Tensor::cat(&outputs, 1)
    }
}

/// Normalize layer
#[derive(Deserialize, Debug, Clone)]
pub struct Normalize;

impl Module for Normalize {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm = (xs.sqr()?.sum(1)? + 1e-12)?.sqrt()?;

        xs.broadcast_div(&norm.unsqueeze(D::Minus1)?)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub enum DenseActivation {
    #[serde(alias = "torch.nn.modules.linear.Identity")]
    Identity,
}

/// Dense layer
#[derive(Deserialize, Debug, Clone)]
pub struct Dense {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub activation_function: DenseActivation,
}
