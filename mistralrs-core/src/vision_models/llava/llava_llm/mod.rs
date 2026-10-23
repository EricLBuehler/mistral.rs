#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use candle_core::{DType, Device, Result, Tensor};

use crate::pipeline::{IsqModel, ModelForwardContext, NormalModel};

fn rope_positions(
    ctx: &mut ModelForwardContext<'_>,
    device: &Device,
    seq_len: usize,
) -> Result<Tensor> {
    ctx.text_positions(device, seq_len)?
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))
}

pub(crate) trait LLaVALLM: IsqModel + NormalModel + Sync + Send {
    //Normal model without anymoe, but add embed and forward_input_embed. This is only a temporary solution. Finally when the rope problem solved for normal LLM models, we should refactor this.
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn forward_input_embed(
        &self,
        input_ids: &Tensor,  // only for masking
        input_embed: Tensor, // we don't want to clone, so we pass it in
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor>;
}

#[derive(Debug)]
pub(crate) struct OrdinaryRoPE;

impl OrdinaryRoPE {
    fn create_parameters(
        n_elem: usize,
        max_seq_len: usize,
        rope_theta: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Result::Ok((cos, sin))
    }
    fn forward(x: &Tensor, positions: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        crate::layers::apply_rotary_q(x, cos, sin, positions, true)
    }
}
pub(crate) mod llama;
pub(crate) mod mistral;

pub use llama::Llama;
pub use mistral::Model as Mistral;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};

    use super::*;

    #[test]
    fn ordinary_batch_expands_each_sequence_offset() -> Result<()> {
        let offsets = [4, 12];
        let context_lens = [(0, 1), (0, 1)];
        let position_ids = [3, 3];
        let flash_params = FlashParams::empty(true);
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash_params);

        assert_eq!(
            rope_positions(&mut ctx, &Device::Cpu, 3)?.to_vec1::<u32>()?,
            vec![4, 5, 6, 12, 13, 14]
        );
        Ok(())
    }

    #[test]
    fn packed_batch_uses_ragged_token_positions() -> Result<()> {
        let offsets = [4, 12];
        let context_lens = [(2, 1), (1, 1)];
        let position_ids = [3, 2];
        let packed_positions = Tensor::new(&[4u32, 5, 6, 12, 13], &Device::Cpu)?;
        let mut metadata = PagedAttentionInputMetadata::dummy(&Device::Cpu)?;
        metadata.rope_positions = Some(HashMap::from([(Device::Cpu.location(), packed_positions)]));
        let mut flash_params = FlashParams::empty(true);
        flash_params.packed = true;
        let kv_cache = Vec::new();
        let mut ctx = ModelForwardContext::new(
            &offsets,
            &context_lens,
            &position_ids,
            Some((kv_cache.as_slice(), &metadata)),
            &flash_params,
        );

        assert_eq!(
            rope_positions(&mut ctx, &Device::Cpu, 5)?.to_vec1::<u32>()?,
            vec![4, 5, 6, 12, 13]
        );
        Ok(())
    }
}
