use std::any::Any;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    layers::CausalMasker,
    layers_masker::{CausalMaskConfig, PastKvLenCache},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
    },
};

use text::HunyuanVLTextModel;
use vision::HunyuanVLVisionModel;

pub(crate) mod config;
pub(crate) mod inputs_processor;
mod layout;
mod rope;
mod text;
mod vision;

pub(crate) use config::Config;
pub(crate) use inputs_processor::HunyuanVLProcessor;

pub struct HunyuanVLModel {
    text: HunyuanVLTextModel,
    vision: HunyuanVLVisionModel,
}

pub(crate) struct HunyuanVLVisionSpecificArgs {
    pub position_ids: Option<Tensor>,
    pub image_grid_thw: Option<Tensor>,
    pub continuous_img_pad: Vec<Vec<(usize, usize)>>,
}

impl HunyuanVLModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision = HunyuanVLVisionModel::new(
            &cfg.vision_config,
            vb.pp("vit")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        // A top-level quantization_config takes precedence over the nested text_config one.
        let mut text_config = cfg.text_config.clone();
        if cfg.quantization_config.is_some() {
            text_config.quantization_config = cfg.quantization_config.clone();
        }
        let text = HunyuanVLTextModel::new(
            &text_config,
            vb,
            cfg.tie_word_embeddings,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self { text, vision })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: Option<Tensor>,
        pixel_values: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let seqlen_offsets = ctx.seqlen_offsets();
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.dtype,
            &CausalMaskConfig {
                sliding_window: self.text.cfg.sliding_window,
                ..Default::default()
            },
        )?;

        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let (_batch_size, _seq_len, hidden_dim) = input_embeds.dims3()?;
        let device = input_embeds.device().clone();

        if let Some(pixel_values) = pixel_values {
            let Some(ref image_grid_thw) = image_grid_thw else {
                candle_core::bail!("pixel_values require image_grid_thw");
            };
            let image_embeds = self
                .vision
                .forward(&pixel_values, image_grid_thw)?
                .to_device(&device)?
                .to_dtype(self.text.dtype)?;
            let total_expected: usize = continuous_img_pad
                .iter()
                .flat_map(|spans| spans.iter().map(|(s, e)| e - s))
                .sum();
            if image_embeds.dim(0)? != total_expected {
                candle_core::bail!(
                    "Image embedding length {} does not match placeholder tokens {}",
                    image_embeds.dim(0)?,
                    total_expected
                );
            }
            let mut offset = 0;
            for (batch, spans) in continuous_img_pad.iter().enumerate() {
                for &(start, end) in spans {
                    let len = end - start;
                    input_embeds = input_embeds.slice_assign(
                        &[batch..batch + 1, start..end, 0..hidden_dim],
                        &image_embeds.narrow(0, offset, len)?.unsqueeze(0)?,
                    )?;
                    offset += len;
                }
            }
        }

        let position_ids = match position_ids {
            Some(positions) => positions,
            None => {
                let (batch, seq_len) = input_ids.dims2()?;
                crate::vision_models::text_position_ids(input_ids, seqlen_offsets)?
                    .to_dtype(candle_core::DType::I64)?
                    .reshape((1, batch, seq_len))?
                    .repeat((layout::POSITION_PLANES, 1, 1))?
            }
        };
        self.text
            .forward_embeds(input_embeds, &attention_mask, &position_ids, ctx)
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for HunyuanVLModel {}

impl crate::speculative::SpeculativeTargetMixin for HunyuanVLModel {}

impl crate::block_diffusion::BlockDiffusionMixin for HunyuanVLModel {}

impl MultimodalModel for HunyuanVLModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let HunyuanVLVisionSpecificArgs {
            position_ids,
            image_grid_thw,
            continuous_img_pad,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `HunyuanVLVisionSpecificArgs`");
        self.forward(
            input_ids,
            position_ids,
            pixel_values,
            image_grid_thw,
            continuous_img_pad,
            ctx,
        )
    }

    fn cache(&self) -> &EitherCache {
        &self.text.cache
    }

    fn device(&self) -> &Device {
        &self.text.device
    }

    fn max_seq_len(&self) -> usize {
        self.text.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.text.cfg
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(HunyuanVLVisionSpecificArgs {
            position_ids: None,
            image_grid_thw: None,
            continuous_img_pad: vec![vec![]],
        })
    }
}

impl IsqModel for HunyuanVLModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut residual = self.text.residual_tensors();
        residual.extend(
            self.vision
                .residual_tensors()
                .into_iter()
                .map(|(name, tensor)| (format!("vit.{name}"), tensor)),
        );
        residual
    }
}
