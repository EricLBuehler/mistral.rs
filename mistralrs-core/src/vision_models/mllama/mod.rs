#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod config;
mod inputs_processor;
mod text;
mod vision;

use std::{any::Any, sync::Arc};

pub(crate) use config::{MLlamaConfig, MLlamaRopeScaling, MLlamaRopeType, MLlamaTextConfig};
use config::{MLlamaVisionConfig, VisionActivation};
pub(crate) use inputs_processor::MLlamaProcessor;
use text::MLlamaTextModel;
use vision::MLlamaVisionModel;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};
use mistralrs_quant::QuantMethod;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::FusedBiasLinear,
    layers_masker::masked_fill,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        Cache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
};

fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    // For metal
    assert!(xs.dtype().is_float());
    let indices = Tensor::new(
        (0..xs.dim(dim)?)
            .flat_map(|i| vec![i as u32; repeats])
            .collect::<Vec<_>>(),
        xs.device(),
    )?;
    xs.index_select(&indices, dim)
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L99
fn prepare_cross_attention_mask(
    cross_attention_mask: &Tensor,
    num_vision_tokens: usize,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let bs = cross_attention_mask.dim(0)?;
    let text_total_length = cross_attention_mask.dim(1)?;
    let mut cross_attn_mask = repeat_interleave(
        &cross_attention_mask.to_dtype(DType::F32)?,
        num_vision_tokens,
        3,
    )?;
    dbg!(&cross_attn_mask);
    cross_attn_mask
        .to_dtype(DType::F32)?
        .write_npy("m-repeati.npy")?;
    cross_attn_mask = cross_attn_mask.reshape((bs, text_total_length, ()))?;
    cross_attn_mask = cross_attn_mask.unsqueeze(1)?;

    // Invert the mask
    let inverted_cross_attn_mask = (1. - cross_attn_mask)?;
    dbg!(&inverted_cross_attn_mask);
    inverted_cross_attn_mask
        .to_dtype(DType::F32)?
        .write_npy("m-invmask.npy")?;
    const NEG_INF_VALUE: f32 = -3.3895313892515355e+38;
    dbg!(&NEG_INF_VALUE);
    dbg!(&inverted_cross_attn_mask);
    inverted_cross_attn_mask
        .ne(0.)?
        .to_dtype(DType::F32)?
        .write_npy("m-boolmask.npy")?;
    cross_attn_mask = masked_fill(
        &inverted_cross_attn_mask,
        &inverted_cross_attn_mask.ne(0.)?,
        NEG_INF_VALUE,
    )?;
    dbg!(&cross_attn_mask);
    cross_attn_mask
        .to_dtype(DType::F32)?
        .write_npy("m-fillmask.npy")?;

    // Apply full-row bias which return 4d tensor of shape (b, h, s1, 1) where
    // value is 0 if a full row in cross attn mask's last dimension contains
    // negative infinity values, otherwise it's 1
    let full_text_row_masked_out_mask = cross_attn_mask
        .ne(NEG_INF_VALUE)?
        .sum(D::Minus1)?
        .ne(0.)?
        .unsqueeze(D::Minus1)?;

    cross_attn_mask = cross_attn_mask
        .broadcast_mul(&full_text_row_masked_out_mask.to_dtype(cross_attn_mask.dtype())?)?
        .to_dtype(DType::F32)?
        .to_dtype(dtype)?;

    Ok((cross_attn_mask, full_text_row_masked_out_mask))
}

pub(crate) struct MLlamaModel {
    vision_model: MLlamaVisionModel,
    language_model: MLlamaTextModel,
    multi_modal_projector: FusedBiasLinear,
    hidden_size: usize,
    dtype: DType,
}

impl MLlamaModel {
    pub(crate) fn new(
        cfg: &MLlamaConfig,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let real_dev = normal_loading_metadata.real_device.clone();
        // This vision model is very sensitive.
        let vision_model_dtype = if vb.dtype() == DType::F16 {
            DType::F32
        } else {
            vb.dtype()
        };
        Ok(Self {
            vision_model: MLlamaVisionModel::new(
                &cfg.vision_config,
                vb.pp("vision_model")
                    .set_device(real_dev.clone())
                    .set_dtype(vision_model_dtype),
            )?,
            language_model: MLlamaTextModel::new(
                &cfg.text_config,
                vb.pp("language_model"),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            multi_modal_projector: linear(
                cfg.vision_config.vision_output_dim,
                cfg.text_config.hidden_size,
                vb.pp("multi_modal_projector")
                    .set_device(real_dev.clone())
                    .set_dtype(vision_model_dtype),
            )?
            .try_into()?,
            hidden_size: cfg.text_config.hidden_size,
            dtype: vb.dtype(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        aspect_ratio_mask: Option<&Tensor>,
        aspect_ratio_ids: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let cross_attn_states = if let Some(pixel_values) = pixel_values {
            let Some(aspect_ratio_mask) = aspect_ratio_mask else {
                candle_core::bail!("`aspect_ratio_mask` must be specified if `pixel_values` is.");
            };
            let Some(aspect_ratio_ids) = aspect_ratio_ids else {
                candle_core::bail!("`aspect_ratio_ids` must be specified if `pixel_values` is.");
            };
            pixel_values
                .to_dtype(DType::F32)?
                .write_npy("m-pixel_values.npy")?;
            let pixel_values = Tensor::read_npy("t-pixel_values.npy")?
                .to_dtype(pixel_values.dtype())?
                .to_device(pixel_values.device())?;
            dbg!(&aspect_ratio_ids, &aspect_ratio_mask);
            aspect_ratio_ids
                .to_dtype(DType::F32)?
                .write_npy("m-aspect_ratio_ids.npy")?;
            aspect_ratio_mask
                .to_dtype(DType::F32)?
                .write_npy("m-aspect_ratio_mask.npy")?;
            let vision_outputs =
                self.vision_model
                    .forward(&pixel_values, aspect_ratio_ids, aspect_ratio_mask)?;
            let cross_attention_states = self
                .multi_modal_projector
                .forward(&vision_outputs.flatten(0, 1)?)?
                .reshape(((), vision_outputs.dim(D::Minus2)?, self.hidden_size))?
                .to_dtype(self.dtype)?;
            println!("Saving...");
            dbg!(&cross_attention_states);
            cross_attention_states
                .to_dtype(DType::F32)?
                .write_npy("m-cross_attention_states.npy")?;
            println!("Saved!");

            // let cross_attention_states = Tensor::read_npy("t-cross_attention_states.npy")?
            //     .to_dtype(cross_attention_states.dtype())?
            //     .to_device(cross_attention_states.device())?;
            Some(cross_attention_states)
        } else {
            None
        };

        let (cross_attn_mask, full_text_row_masked_out_mask) =
            if let Some(cross_attn_mask) = cross_attn_mask {
                let (cmask, fmask) = prepare_cross_attention_mask(
                    cross_attn_mask,
                    self.vision_model.num_patches,
                    self.dtype,
                )?;
                println!("Saving...");
                dbg!(&cmask, &fmask);
                cmask.to_dtype(DType::F32)?.write_npy("m-cmask.npy")?;
                fmask.to_dtype(DType::F32)?.write_npy("m-fmask.npy")?;
                println!("Saved!");
                (Some(cmask), Some(fmask))
            } else {
                (None, None)
            };

        self.language_model.forward(
            input_ids,
            cross_attn_states.as_ref(),
            cross_attn_mask.as_ref(),
            full_text_row_masked_out_mask.as_ref(),
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }
}

pub(crate) struct MLlamaSpecificArgs {
    pub aspect_ratio_ids: Option<Tensor>,
    pub aspect_ratio_mask: Option<Tensor>,
    pub cross_attn_mask: Option<Tensor>,
}

impl VisionModel for MLlamaModel {
    fn cache(&self) -> &Cache {
        &self.language_model.cache
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.language_model.cfg
    }
    fn device(&self) -> &Device {
        &self.language_model.device
    }
    fn has_conv2d(&self) -> bool {
        true
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_position_embeddings
    }
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        _metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let MLlamaSpecificArgs {
            aspect_ratio_ids,
            aspect_ratio_mask,
            cross_attn_mask,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `MLlamaSpecificArgs`");
        self.forward_inner(
            input_ids,
            pixel_values.as_ref(),
            aspect_ratio_mask.as_ref(),
            aspect_ratio_ids.as_ref(),
            cross_attn_mask.as_ref(),
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }
}

impl IsqModel for MLlamaModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.language_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("multi_modal_projector")
            .add(&self.multi_modal_projector);
        uvb.pp("language_model")
            .extend(self.language_model.residual_tensors());
        uvb.pp("vision_model")
            .extend(self.vision_model.residual_tensors());

        uvb.to_safetensors()
    }
}

impl AnyMoeBaseModelMixin for MLlamaModel {}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::repeat_interleave;

    #[test]
    fn test_repeat_interleave() -> Result<()> {
        let input = Tensor::new(
            vec![vec![vec![1f32, 2., 3.], vec![4f32, 5., 6.]]],
            &Device::Cpu,
        )?;

        let repeat_interleaved = repeat_interleave(&input, 2, 2)?;
        assert_eq!(
            repeat_interleaved.to_vec3::<f32>()?,
            vec![vec![
                vec![1., 1., 2., 2., 3., 3.],
                vec![4., 4., 5., 5., 6., 6.]
            ]]
        );

        Ok(())
    }
}
