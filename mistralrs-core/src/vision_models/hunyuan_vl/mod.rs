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
mod rope;
mod text;
mod vision;

pub(crate) use config::Config;
pub(crate) use inputs_processor::HunyuanVLProcessor;

pub struct HunyuanVLModel {
    text: HunyuanVLTextModel,
    vision: HunyuanVLVisionModel,
    spatial_merge_size: usize,
}

pub(crate) struct HunyuanVLVisionSpecificArgs {
    pub input_ids_full: Tensor,
    pub image_grid_thw: Option<Tensor>,
    pub rope_img_grid_thw: Option<Tensor>,
    pub seqlens: Vec<usize>,
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
        let text = HunyuanVLTextModel::new(cfg, vb, normal_loading_metadata, attention_mechanism)?;
        Ok(Self {
            text,
            vision,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }

    fn get_position_ids(
        &self,
        input_ids_full: &Tensor,
        image_grid_thw: Option<&Tensor>,
        seqlens: &[usize],
        continuous_img_pad: &[Vec<(usize, usize)>],
        input_ids: &Tensor,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        if image_grid_thw.is_none() {
            if seq_len == 1 {
                if let Some(rope_positions) = ctx.cache().rope_positions(input_ids.device()) {
                    if rope_positions.dim(0)? != batch {
                        candle_core::bail!(
                            "rope positions shape {:?} is incompatible with input shape {:?}",
                            rope_positions.shape(),
                            input_ids.shape()
                        );
                    }
                    return Ok(rope_positions.reshape((1, batch, 1))?.repeat((4, 1, 1))?);
                }
            }

            let mut data = Vec::with_capacity(4 * batch * seq_len);
            for _ in 0..4 {
                for offset in ctx.seqlen_offsets() {
                    let offset = *offset;
                    for pos in offset..offset + seq_len {
                        data.push(pos as i64);
                    }
                }
            }
            return Tensor::from_vec(data, (4, batch, seq_len), input_ids.device());
        }

        let (_, full_len) = input_ids_full.dims2()?;
        let mut data = vec![vec![vec![0i64; full_len]; batch]; 4];
        for b in 0..batch {
            let len = seqlens.get(b).copied().unwrap_or(full_len).min(full_len);
            for pos in 0..len {
                for plane in data.iter_mut().take(4) {
                    plane[b][pos] = pos as i64;
                }
            }
        }

        let mut grid_index = 0usize;
        let grid = image_grid_thw.unwrap().to_vec2::<u32>()?;
        for (batch_idx, spans) in continuous_img_pad.iter().enumerate() {
            for &(start, end) in spans {
                let Some(g) = grid.get(grid_index) else {
                    candle_core::bail!("Not enough image_grid_thw entries for HunyuanVL positions");
                };
                grid_index += 1;
                let patch_h = (g[1] as usize) / self.spatial_merge_size;
                let patch_w = (g[2] as usize) / self.spatial_merge_size;
                let interior = patch_h * (patch_w + 1);
                if end.saturating_sub(start) != interior + 2 {
                    candle_core::bail!(
                        "HunyuanVL image span length {} does not match expected {}",
                        end.saturating_sub(start),
                        interior + 2
                    );
                }
                let mut offset = start + 1;
                for h in 0..patch_h {
                    for w in 0..=patch_w {
                        data[1][batch_idx][offset] = w as i64;
                        data[2][batch_idx][offset] = h as i64;
                        data[3][batch_idx][offset] = 0;
                        offset += 1;
                    }
                }
            }
        }

        let mut flat = Vec::with_capacity(4 * batch * full_len);
        for plane in data.iter().take(4) {
            for row in plane.iter().take(batch) {
                flat.extend_from_slice(row);
            }
        }
        let full_positions = Tensor::from_vec(flat, (4, batch, full_len), input_ids.device())?;
        let mut indices = Vec::with_capacity(4 * batch * seq_len);
        for _ in 0..4 {
            for offset in ctx.seqlen_offsets() {
                let offset = *offset;
                for pos in offset..offset + seq_len {
                    indices.push(u32::try_from(pos).map_err(candle_core::Error::wrap)?);
                }
            }
        }
        let indices = Tensor::from_vec(indices, (4, batch, seq_len), input_ids.device())?;
        full_positions.gather(&indices, 2)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        pixel_values: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        rope_img_grid_thw: Option<Tensor>,
        seqlens: Vec<usize>,
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
            for range in crate::vision_models::chunked_multimodal_ranges(
                &continuous_img_pad,
                seqlen_offsets,
                input_ids.dim(1)?,
            ) {
                let chunk = image_embeds.narrow(0, range.embed_start, range.len)?;
                input_embeds = input_embeds.slice_assign(
                    &[
                        range.batch..range.batch + 1,
                        range.local_start..range.local_end,
                        0..hidden_dim,
                    ],
                    &chunk.unsqueeze(0)?,
                )?;
            }
        }

        let position_ids = self.get_position_ids(
            input_ids_full,
            rope_img_grid_thw.as_ref().or(image_grid_thw.as_ref()),
            &seqlens,
            &continuous_img_pad,
            input_ids,
            ctx,
        )?;
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
            input_ids_full,
            image_grid_thw,
            rope_img_grid_thw,
            seqlens,
            continuous_img_pad,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `HunyuanVLVisionSpecificArgs`");
        self.forward(
            input_ids,
            &input_ids_full,
            pixel_values,
            image_grid_thw,
            rope_img_grid_thw,
            seqlens,
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

    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(HunyuanVLVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            image_grid_thw: None,
            rope_img_grid_thw: None,
            seqlens: vec![input_ids.dims()[1]],
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
