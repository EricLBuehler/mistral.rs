//! Qwen4Exp multimodal wrapper: the Qwen4Exp text decoder with the reused Qwen3-VL
//! vision tower and merger, matching the reference converter's "unmodified Qwen3-VL
//! ViT" claim.
//!
//! Multimodal inputs arrive as embeddings, so the original token IDs (with image
//! placeholder tokens preserved) are passed to the text decoder for PLE hashing,
//! matching the reference behavior of hashing the image token id on embedding-only
//! batches. Video inputs and deepstack visual embeds fail closed until their
//! position and PLE behavior is validated against the reference.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{QuantizedConfig, ShardedVarBuilder};
use serde::Deserialize;

use super::{
    config::Config as TextConfig,
    text::{self, MropePositionOverride},
};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::AttentionMask,
    paged_attention::{
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
        NormalModel,
    },
    vision_models::{
        qwen3_5::config::VisionConfig,
        qwen3_vl::{
            concatenate_visual_items, get_rope_index, vision::Qwen3VLVisionModel,
            Qwen3VLVisionSpecificArgs, VisualEncoder,
        },
    },
};

/// Multimodal configuration: the Qwen4Exp text config plus the Qwen3-VL vision config
/// and the vision token ids from the original checkpoint config.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MultimodalConfig {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

pub struct Qwen4ExpModel {
    text: text::Model,
    vision: Qwen3VLVisionModel,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Qwen4ExpModel {
    pub(crate) fn new(
        cfg: &MultimodalConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // Support both original HuggingFace naming (model.visual.*) and MLX naming
        // (vision_tower.*), matching the Qwen3.5 wrapper.
        let vision_vb = if vb.contains_tensor("vision_tower.patch_embed.proj.weight") {
            vb.pp("vision_tower")
        } else {
            vb.pp("model").pp("visual")
        }
        .without_lora_registry();
        let vision = Qwen3VLVisionModel::new(
            &cfg.vision_config,
            vision_vb.set_device(normal_loading_metadata.real_device.clone()),
        )?;
        // Top-level quantization_config takes precedence over the text config's own, and
        // the top-level tie_word_embeddings overrides the text config's default.
        let mut text_config = cfg.text_config.clone();
        if cfg.quantization_config.is_some() {
            text_config.quantization_config = cfg.quantization_config.clone();
        }
        text_config.tie_word_embeddings =
            cfg.tie_word_embeddings || text_config.tie_word_embeddings;
        let text = text::Model::new(
            &text_config,
            vb,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            vision_start_token_id: cfg.vision_start_token_id,
            vision_end_token_id: cfg.vision_end_token_id,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        args: Qwen3VLVisionSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Qwen3VLVisionSpecificArgs {
            input_ids_full,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            rope_img_grid_thw,
            rope_vid_grid_thw,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            image_hashes,
            video_hashes: _,
            packed_layout,
            prompt_position_ids: _,
        } = args;
        if packed_layout.is_some() {
            candle_core::bail!("Qwen4Exp multimodal packed prefill is not supported yet");
        }
        if pixel_values_videos.is_some() || video_grid_thw.is_some() {
            candle_core::bail!(
                "Qwen4Exp multimodal video inputs are not supported yet; their MRoPE position and PLE behavior is pending validation"
            );
        }
        let seqlen_offsets = ctx.seqlen_offsets().to_vec();
        let dtype = self.text.dtype();
        let mut input_embeds = self
            .text
            .embed_tokens()
            .embedding_forward(input_ids, dtype)?;
        let device = input_embeds.device().clone();

        let mut has_images = false;
        if let Some(pixel_values) = &pixel_values {
            has_images = true;
            let Some(grid) = image_grid_thw.as_ref() else {
                candle_core::bail!("pixel_values require image_grid_thw");
            };
            let mut pixel_values = pixel_values.clone();
            let ndim = pixel_values.dims().len();
            if ndim > 2 {
                let last_dim = pixel_values.dim(ndim - 1)?;
                pixel_values = pixel_values.reshape(((), last_dim))?;
            }
            let (image_embeds, deepstack_embeds) = if image_hashes.is_empty() {
                self.vision.forward(&pixel_values, grid)?
            } else {
                let per_image =
                    VisualEncoder::new(&self.vision, &self.encoder_cache, self.spatial_merge_size)
                        .encode(&pixel_values, grid, &image_hashes, CacheModality::Image)?;
                concatenate_visual_items(&per_image)?
            };
            if !deepstack_embeds.is_empty() {
                candle_core::bail!(
                    "Qwen4Exp multimodal does not yet apply deepstack visual embeds; use an mmproj without deepstack layers"
                );
            }
            let image_embeds = image_embeds.to_device(&device)?.to_dtype(dtype)?;
            let (_, _, hidden_dim) = input_embeds.dims3()?;
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
            let mut offset = 0usize;
            for (batch, spans) in continuous_img_pad.iter().enumerate() {
                for &(start, end) in spans {
                    let len = end - start;
                    let chunk = image_embeds.narrow(0, offset, len)?;
                    offset += len;
                    input_embeds = input_embeds.slice_assign(
                        &[batch..batch + 1, start..end, 0..hidden_dim],
                        &chunk.unsqueeze(0)?,
                    )?;
                }
            }
            let _ = continuous_vid_pad;
        }

        // Image inputs need multimodal MRoPE positions. Preserve the temporal, height, and
        // width components so both QSA main attention and the indexer rotate their designated
        // interleaved frequency pairs without reducing image positions to a scalar.
        let positions_override = if has_images {
            let rope_img = rope_img_grid_thw.or(image_grid_thw.clone());
            let rope_vid = rope_vid_grid_thw.or(video_grid_thw.clone());
            let max_seqlens = *seqlens
                .iter()
                .max()
                .ok_or_else(|| candle_core::Error::msg("seqlens is empty"))?;
            let mut ropeidx_attn_mask_bs = Vec::new();
            for len in &seqlens {
                ropeidx_attn_mask_bs.push(Tensor::new(
                    [vec![1f32; *len], vec![0f32; max_seqlens - len]].concat(),
                    input_ids.device(),
                )?);
            }
            let ropeidx_attn_mask = Tensor::stack(&ropeidx_attn_mask_bs, 0)?;
            let (position_ids, mrope_position_deltas) = get_rope_index(
                &input_ids_full,
                rope_img.as_ref(),
                rope_vid.as_ref(),
                &AttentionMask::Custom(ropeidx_attn_mask),
                self.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            )?;
            let position_ids = crate::vision_models::mrope_position_ids_for_input(
                &position_ids,
                &mrope_position_deltas,
                input_ids,
                &seqlen_offsets,
            )?
            .to_dtype(DType::U32)?
            .to_vec3::<u32>()?;
            Some(MropePositionOverride::Sectioned(position_ids))
        } else {
            None
        };

        self.text
            .forward_embeds(input_embeds, input_ids, positions_override, ctx)
    }
}

impl MultimodalModel for Qwen4ExpModel {
    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        false
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = model_specific_args
            .downcast::<Qwen3VLVisionSpecificArgs>()
            .map_err(|_| {
                candle_core::Error::msg(
                    "Qwen4Exp multimodal requires `Qwen3VLVisionSpecificArgs` model args",
                )
            })?;
        self.forward(input_ids, pixel_values, *args, ctx)
    }

    fn cache(&self) -> &EitherCache {
        NormalModel::cache(&self.text)
    }

    fn device(&self) -> &Device {
        NormalModel::device(&self.text)
    }

    fn max_seq_len(&self) -> usize {
        NormalModel::max_seq_len(&self.text)
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        false
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for_args(&self, _model_specific_args: &dyn Any) -> bool {
        false
    }

    fn config(&self) -> &ModelConfigMetadata {
        NormalModel::config(&self.text)
    }

    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(NormalModel::config(&self.text).clone())
    }

    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        let (batch_size, seq_len) = input_ids.dims2().expect("input ids must be rank 2");
        Box::new(Qwen3VLVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            pixel_values_videos: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![seq_len; batch_size],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            image_hashes: vec![],
            video_hashes: vec![],
            packed_layout: None,
            prompt_position_ids: None,
        })
    }

    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(&self.encoder_cache)
    }

    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("encoder cache poisoned")
                .counters(),
        )
    }
}

impl IsqModel for Qwen4ExpModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        IsqModel::residual_tensors(&self.text)
    }
}

impl AnyMoeBaseModelMixin for Qwen4ExpModel {}

impl crate::speculative::SpeculativeTargetMixin for Qwen4ExpModel {}

impl crate::block_diffusion::BlockDiffusionMixin for Qwen4ExpModel {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Result, Tensor};
    use mistralrs_quant::ShardedSafeTensors;

    use super::*;
    use crate::layers::Activation;
    use crate::models::qwen4_exp::text::tests::{fixture_tensors, loading_metadata, ple_config};
    use crate::pipeline::{
        text_models_inputs_processor::FlashParams, RecurrentBatchKind, RecurrentMetadata,
    };

    /// Tiny Qwen3-VL vision tower whose merger outputs the text hidden width.
    fn vision_config() -> VisionConfig {
        VisionConfig {
            depth: 1,
            hidden_size: 4,
            out_hidden_size: 8, // the text fixture's hidden size
            hidden_act: Activation::Silu,
            intermediate_size: 6,
            num_heads: 1, // head_dim 4, required by the vision rotary width
            in_chans: 3,
            patch_size: 1,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            num_position_embeddings: 4, // 2x2 grid, must be a perfect square
            deepstack_visual_indexes: vec![],
        }
    }

    fn multimodal_config() -> MultimodalConfig {
        MultimodalConfig {
            // PLE-enabled text config so image placeholder tokens reach PLE hashing.
            text_config: ple_config(),
            vision_config: vision_config(),
            image_token_id: 10,
            video_token_id: 11,
            vision_start_token_id: 12,
            vision_end_token_id: 13,
            tie_word_embeddings: false,
            quantization_config: None,
        }
    }

    fn weightn(shape: &[usize], seed: usize) -> Result<Tensor> {
        let count: usize = shape.iter().product();
        let data = (0..count)
            .map(|index| (((index * 2_654_435_761 + seed) % 2_000) as f32 / 1_000.0) - 1.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, shape, &Device::Cpu)
    }

    fn put(tensors: &mut HashMap<String, Tensor>, name: String, shape: &[usize], seed: usize) {
        tensors.insert(name, weightn(shape, seed).expect("vision fixture weight"));
    }

    /// Build the Qwen3-VL vision tensors the wrapper reads under `model.visual.*`.
    fn add_vision_tensors(tensors: &mut HashMap<String, Tensor>, cfg: &VisionConfig) -> Result<()> {
        let prefix = "model.visual";
        let vh = cfg.hidden_size;
        let merged = vh * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let mut seed = 9_000usize;
        let mut next = || {
            seed += 1;
            seed
        };

        put(
            tensors,
            format!("{prefix}.patch_embed.proj.weight"),
            &[
                vh,
                cfg.in_chans,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ],
            next(),
        );
        put(
            tensors,
            format!("{prefix}.patch_embed.proj.bias"),
            &[vh],
            next(),
        );
        put(
            tensors,
            format!("{prefix}.pos_embed.weight"),
            &[cfg.num_position_embeddings, vh],
            next(),
        );
        for block in 0..cfg.depth {
            let block = format!("{prefix}.blocks.{block}");
            for norm in ["norm1", "norm2"] {
                put(tensors, format!("{block}.{norm}.weight"), &[vh], next());
                put(tensors, format!("{block}.{norm}.bias"), &[vh], next());
            }
            put(
                tensors,
                format!("{block}.attn.qkv.weight"),
                &[vh * 3, vh],
                next(),
            );
            put(tensors, format!("{block}.attn.qkv.bias"), &[vh * 3], next());
            put(
                tensors,
                format!("{block}.attn.proj.weight"),
                &[vh, vh],
                next(),
            );
            put(tensors, format!("{block}.attn.proj.bias"), &[vh], next());
            put(
                tensors,
                format!("{block}.mlp.linear_fc1.weight"),
                &[cfg.intermediate_size, vh],
                next(),
            );
            put(
                tensors,
                format!("{block}.mlp.linear_fc1.bias"),
                &[cfg.intermediate_size],
                next(),
            );
            put(
                tensors,
                format!("{block}.mlp.linear_fc2.weight"),
                &[vh, cfg.intermediate_size],
                next(),
            );
            put(
                tensors,
                format!("{block}.mlp.linear_fc2.bias"),
                &[vh],
                next(),
            );
        }
        put(
            tensors,
            format!("{prefix}.merger.norm.weight"),
            &[vh],
            next(),
        );
        put(tensors, format!("{prefix}.merger.norm.bias"), &[vh], next());
        put(
            tensors,
            format!("{prefix}.merger.linear_fc1.weight"),
            &[merged, merged],
            next(),
        );
        put(
            tensors,
            format!("{prefix}.merger.linear_fc1.bias"),
            &[merged],
            next(),
        );
        put(
            tensors,
            format!("{prefix}.merger.linear_fc2.weight"),
            &[cfg.out_hidden_size, merged],
            next(),
        );
        put(
            tensors,
            format!("{prefix}.merger.linear_fc2.bias"),
            &[cfg.out_hidden_size],
            next(),
        );
        Ok(())
    }

    fn build_multimodal_model(cfg: &MultimodalConfig) -> Result<Qwen4ExpModel> {
        let mut tensors = fixture_tensors(&cfg.text_config)?;
        add_vision_tensors(&mut tensors, &cfg.vision_config)?;
        let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu);
        Qwen4ExpModel::new(
            cfg,
            vb,
            true,
            loading_metadata(cfg.text_config.num_hidden_layers)?,
            AttentionImplementation::Eager,
        )
    }

    fn ids_tensor(tokens: &[u32]) -> Result<Tensor> {
        Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu)
    }

    /// One wrapper-level prefill forward as `MultimodalModel::forward` with explicit slots.
    fn run_wrapper_forward(
        model: &Qwen4ExpModel,
        tokens: &[u32],
        pixel_values: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        img_pad: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let flash = FlashParams::empty(true);
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        model
            .cache()
            .hybrid()
            .set_physical_state_indices_with_host(Some(indices.clone()), Some(vec![0]));
        let args = Qwen3VLVisionSpecificArgs {
            input_ids_full: ids_tensor(tokens)?,
            pixel_values_videos: None,
            image_grid_thw,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![tokens.len()],
            continuous_img_pad: vec![img_pad],
            continuous_vid_pad: vec![],
            image_hashes: vec![],
            video_hashes: vec![],
            packed_layout: None,
            prompt_position_ids: None,
        };
        let offsets = [0usize];
        let context_lens = [(0usize, tokens.len())];
        let position_ids = [tokens.len()];
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(
                    RecurrentBatchKind::Prefill,
                    indices,
                    None,
                )));
        MultimodalModel::forward(
            model,
            &ids_tensor(tokens)?,
            pixel_values,
            Box::new(args),
            &mut ctx,
        )
    }

    fn reset_state(model: &Qwen4ExpModel) -> Result<()> {
        model.cache().hybrid().reset()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_vec3::<f32>()?;
        let b = b.to_vec3::<f32>()?;
        let mut diff = 0f32;
        for (a_rows, b_rows) in a.iter().zip(b.iter()) {
            for (a_row, b_row) in a_rows.iter().zip(b_rows.iter()) {
                for (a_val, b_val) in a_row.iter().zip(b_row.iter()) {
                    diff = diff.max((a_val - b_val).abs());
                }
            }
        }
        Ok(diff)
    }

    /// Fixture coherence: the vision merger width matches the text decoder, the positional
    /// embedding count is a perfect square, and the text config keeps PLE enabled so image
    /// placeholder tokens reach PLE hashing.
    #[test]
    fn multimodal_fixture_dimensions_line_up() {
        let cfg = multimodal_config();
        assert_eq!(
            cfg.vision_config.out_hidden_size,
            cfg.text_config.hidden_size
        );
        let side = (cfg.vision_config.num_position_embeddings as f64)
            .sqrt()
            .round() as usize;
        assert_eq!(side * side, cfg.vision_config.num_position_embeddings);
        assert_eq!(cfg.image_token_id, 10);
        assert_eq!(cfg.text_config.ple_layer_ids, vec![0]);
        assert_eq!(cfg.text_config.image_token_id, None);
    }

    /// The Phase 9 synthetic image-embedding replacement test: the wrapper substitutes the
    /// vision output into exactly the placeholder span, keeps the original token ids (with
    /// the image token) for PLE hashing, and produces the same logits as manually composing
    /// vision + substitution + `forward_embeds`. A 1x1 merged image grid yields uniform
    /// (text-like) MRoPE positions, which is the only image shape supported today.
    #[test]
    fn image_embedding_replacement_matches_manual_composition() -> Result<()> {
        let cfg = multimodal_config();
        cfg.text_config.validate()?;
        let model = build_multimodal_model(&cfg)?;
        let hidden = cfg.text_config.hidden_size;
        // vision_start(12) text image(10) vision_end(13) text; grid 1x2x2 merges to one token.
        let tokens = [1u32, 12, 10, 13, 2];

        let grid = Tensor::from_vec(vec![1u32, 2, 2], (1, 3), &Device::Cpu)?;
        // Each patch row carries in_chans * temporal * patch * patch channel values.
        let patch_channels = cfg.vision_config.in_chans
            * cfg.vision_config.temporal_patch_size
            * cfg.vision_config.patch_size
            * cfg.vision_config.patch_size;
        let pixel_values = weightn(&[4, patch_channels], 777)?;
        let image_logits = run_wrapper_forward(
            &model,
            &tokens,
            Some(pixel_values.clone()),
            Some(grid.clone()),
            vec![(2, 3)],
        )?;
        assert_eq!(
            image_logits.dims(),
            [1, tokens.len(), cfg.text_config.vocab_size]
        );

        reset_state(&model)?;

        // Identical token ids without pixel values must differ: the image embeddings
        // replaced the placeholder embedding at index 2.
        let text_logits = run_wrapper_forward(&model, &tokens, None, None, vec![])?;
        assert!(
            max_abs_diff(&image_logits, &text_logits)? > 1e-4,
            "image logits must differ from text-only logits"
        );

        reset_state(&model)?;

        // Manual composition parity, mirroring the wrapper's own steps.
        let dtype = model.text.dtype();
        let mut embeds = model
            .text
            .embed_tokens()
            .embedding_forward(&ids_tensor(&tokens)?, dtype)?;
        let (image_embeds, deepstack) = model.vision.forward(&pixel_values, &grid)?;
        assert!(deepstack.is_empty(), "no deepstack layers in this fixture");
        let image_embeds = image_embeds.to_device(&Device::Cpu)?.to_dtype(dtype)?;
        assert_eq!(image_embeds.dims(), [1, hidden]);
        embeds = embeds.slice_assign(&[0..1, 2..3, 0..hidden], &image_embeds.unsqueeze(0)?)?;
        // Uniform positions derived by the wrapper: one text token, the merged image token,
        // then the trailing text tokens.
        let positions = Some(vec![vec![0u32, 1, 2, 3, 4]]);
        let flash = FlashParams::empty(true);
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        model
            .cache()
            .hybrid()
            .set_physical_state_indices_with_host(Some(indices.clone()), Some(vec![0]));
        let offsets = [0usize];
        let context_lens = [(0usize, tokens.len())];
        let position_ids = [tokens.len()];
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(
                    RecurrentBatchKind::Prefill,
                    indices,
                    None,
                )));
        let manual = model.text.forward_embeds(
            embeds,
            &ids_tensor(&tokens)?,
            positions.map(MropePositionOverride::Scalar),
            &mut ctx,
        )?;
        let diff = max_abs_diff(&image_logits, &manual)?;
        assert!(
            diff < 1e-5,
            "wrapper output must match manual composition, got diff {diff}"
        );
        Ok(())
    }

    /// Genuinely 2D image positions retain their independent height/width MRoPE sections.
    #[test]
    fn nonuniform_image_positions_run_with_sectioned_mrope() -> Result<()> {
        let cfg = multimodal_config();
        let model = build_multimodal_model(&cfg)?;
        let tokens = [1u32, 12, 10, 10, 13, 2];
        // Grid 1x2x4 merges into two placeholder tokens with distinct height/width positions.
        let grid = Tensor::from_vec(vec![1u32, 2, 4], (1, 3), &Device::Cpu)?;
        let patch_channels = cfg.vision_config.in_chans
            * cfg.vision_config.temporal_patch_size
            * cfg.vision_config.patch_size
            * cfg.vision_config.patch_size;
        let pixel_values = weightn(&[8, patch_channels], 888)?;
        let logits = run_wrapper_forward(
            &model,
            &tokens,
            Some(pixel_values),
            Some(grid),
            vec![(2, 4)],
        )?;
        assert_eq!(logits.dims(), [1, tokens.len(), cfg.text_config.vocab_size]);
        Ok(())
    }
}
