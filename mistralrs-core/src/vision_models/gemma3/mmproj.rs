use candle_core::{Result, Tensor};
use candle_nn::Module;
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::{AvgPool2d, RmsNorm};

use super::config::Gemma3Config;

pub struct Gemma3MultiModalProjector {
    mm_input_projection_weight: Tensor,
    mm_soft_emb_norm: RmsNorm,
    patches_per_image: usize,
    avg_pool: AvgPool2d,
}

impl Gemma3MultiModalProjector {
    pub fn new(cfg: &Gemma3Config, vb: ShardedVarBuilder) -> Result<Self> {
        let mm_input_projection_weight = vb.get(
            (cfg.vision_config.hidden_size, cfg.text_config.hidden_size),
            "mm_input_projection_weight",
        )?;
        let mm_soft_emb_norm = RmsNorm::new_gemma(
            cfg.vision_config.hidden_size,
            cfg.vision_config.layer_norm_eps,
            vb.pp("mm_soft_emb_norm"),
        )?;

        let patches_per_image = cfg.vision_config.image_size / cfg.vision_config.patch_size;
        let tokens_per_side = cfg.mm_tokens_per_image.isqrt();
        let kernel_size = patches_per_image / tokens_per_side;
        let avg_pool = AvgPool2d::new(kernel_size, kernel_size);

        Ok(Self {
            mm_input_projection_weight,
            mm_soft_emb_norm,
            patches_per_image,
            avg_pool,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, _, seqlen) = xs.dims3()?;

        let mut reshaped_vision_outputs = xs.transpose(1, 2)?;
        reshaped_vision_outputs = reshaped_vision_outputs.reshape((
            bs,
            seqlen,
            self.patches_per_image,
            self.patches_per_image,
        ))?;
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()?;

        let mut pooled_vision_outputs = self.avg_pool.forward(&reshaped_vision_outputs)?;
        pooled_vision_outputs = pooled_vision_outputs.flatten_from(2)?;
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)?;

        let normed_vision_outputs = self.mm_soft_emb_norm.forward(&pooled_vision_outputs)?;

        normed_vision_outputs.broadcast_matmul(&self.mm_input_projection_weight)
    }
}
