#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) mod config;
mod inputs_processor;
mod text;
mod vision;

use std::any::Any;
use std::sync::Arc;

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision_models::kimi_k25::{
        config::Config,
        text::KimiK25TextModel,
        vision::MoonViT3D,
    },
};

pub use inputs_processor::KimiK25Processor;

// ── Model-Specific Args ──

pub struct KimiK25SpecificArgs {
    pub grid_thws: Vec<(usize, usize, usize)>,
}

// ── PatchMergerMLP Projector ──

struct PatchMergerMLP {
    pre_norm: LayerNorm,
    proj_0: Linear,
    proj_2: Linear,
    merged_hidden_size: usize,
}

impl PatchMergerMLP {
    fn new(cfg: &config::VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let merge_area = cfg.merge_kernel_size[0] * cfg.merge_kernel_size[1]; // 4
        let merged_hidden_size = cfg.mm_hidden_size * merge_area; // 4608

        let norm_cfg = LayerNormConfig {
            eps: cfg.projector_ln_eps,
            ..Default::default()
        };
        let pre_norm = layers::layer_norm(cfg.mm_hidden_size, norm_cfg, vb.pp("pre_norm"))?;

        // proj is a Sequential: [Linear(4608, 4608), GELU(), Linear(4608, text_hidden_size)]
        // Weights: proj.0 and proj.2 (proj.1 is GELU activation, no weights)
        let proj_0 = layers::linear(merged_hidden_size, merged_hidden_size, vb.pp("proj").pp("0"))?;
        let proj_2 = layers::linear(merged_hidden_size, cfg.text_hidden_size, vb.pp("proj").pp("2"))?;

        Ok(Self {
            pre_norm,
            proj_0,
            proj_2,
            merged_hidden_size,
        })
    }

    /// Forward pass. Input: list of tensors, each (num_merged_patches, merge_area, mm_hidden_size).
    /// Output: list of tensors, each (num_merged_patches, text_hidden_size).
    fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut outputs = Vec::with_capacity(features.len());
        for feat in features {
            // feat: (num_patches, merge_area, mm_hidden_size)
            let num_patches = feat.dim(0)?;
            let normed = self.pre_norm.forward(feat)?;
            // Flatten merge_area * mm_hidden_size -> merged_hidden_size
            let flat = normed.reshape((num_patches, self.merged_hidden_size))?;
            let projected = self.proj_0.forward(&flat)?.gelu_erf()?;
            let projected = self.proj_2.forward(&projected)?;
            outputs.push(projected);
        }
        Ok(outputs)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("pre_norm").add(&self.pre_norm);
        uvb.pp("proj").pp("0").add(&self.proj_0);
        uvb.pp("proj").pp("2").add(&self.proj_2);
        uvb.to_safetensors()
    }
}

// ── Top-Level Model ──

pub struct KimiK25Model {
    vision_tower: MoonViT3D,
    mm_projector: PatchMergerMLP,
    text: KimiK25TextModel,
    media_placeholder_token_id: u32,
}

impl KimiK25Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision_tower = MoonViT3D::new(&cfg.vision_config, vb.pp("vision_tower"))?;
        let mm_projector = PatchMergerMLP::new(&cfg.vision_config, vb.pp("mm_projector"))?;
        let text = KimiK25TextModel::new(
            &cfg.text_config,
            vb.pp("language_model"),
            normal_loading_metadata,
            attention_mechanism,
        )?;

        Ok(Self {
            vision_tower,
            mm_projector,
            text,
            media_placeholder_token_id: cfg.media_placeholder_token_id as u32,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        grid_thws: Vec<(usize, usize, usize)>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut input_embeds = self.text.embed_tokens(input_ids)?;

        if let Some(pixel_values) = pixel_values {
            if pixel_values.dim(0)? > 0 && input_ids.dims()[1] > 1 {
                // Extract vision features
                let vision_features = self.vision_tower.forward(&pixel_values, &grid_thws)?;
                // Project to text hidden size
                let projected_features = self.mm_projector.forward(&vision_features)?;

                // Merge image features into text embeddings at media_placeholder positions
                let (batch_size, _seq_len, hidden_dim) = input_embeds.dims3()?;

                // For each batch, find placeholder positions and replace with projected features
                let mut feature_offset = 0;
                for b in 0..batch_size {
                    let batch_ids = input_ids.i((b, ..))?;
                    let batch_ids_vec: Vec<u32> = batch_ids.to_vec1()?;

                    for (img_idx, proj_feat) in projected_features.iter().enumerate() {
                        if feature_offset + img_idx >= projected_features.len() {
                            break;
                        }
                        let num_tokens = proj_feat.dim(0)?;

                        // Find contiguous run of placeholder tokens for this image
                        let mut placeholder_positions = Vec::new();
                        for (pos, &token_id) in batch_ids_vec.iter().enumerate() {
                            if token_id == self.media_placeholder_token_id {
                                placeholder_positions.push(pos);
                            }
                        }

                        // Replace placeholder embeddings with projected features
                        if placeholder_positions.len() >= num_tokens {
                            let start = placeholder_positions[0];
                            input_embeds = input_embeds.slice_assign(
                                &[b..b + 1, start..start + num_tokens, 0..hidden_dim],
                                &proj_feat.unsqueeze(0)?,
                            )?;
                        }
                    }
                    feature_offset += projected_features.len();
                }
            }
        }

        self.text.forward_input_embeds(
            input_embeds,
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

// ── Trait Implementations ──

impl IsqModel for KimiK25Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.text.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = Vec::new();

        // Text model residual tensors (under "language_model" prefix)
        for (name, tensor) in self.text.residual_tensors() {
            tensors.push((format!("language_model.{name}"), tensor));
        }

        // Vision tower residual tensors
        for (name, tensor) in self.vision_tower.residual_tensors() {
            tensors.push((format!("vision_tower.{name}"), tensor));
        }

        // MM projector residual tensors
        for (name, tensor) in self.mm_projector.residual_tensors() {
            tensors.push((format!("mm_projector.{name}"), tensor));
        }

        tensors
    }
}

impl crate::pipeline::VisionModel for KimiK25Model {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let KimiK25SpecificArgs { grid_thws } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `KimiK25SpecificArgs`");

        self.forward(
            input_ids,
            pixel_values,
            grid_thws,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }

    fn cache(&self) -> &EitherCache {
        &self.text.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.text.cache
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
        Box::new(KimiK25SpecificArgs {
            grid_thws: Vec::new(),
        })
    }
}

impl AnyMoeBaseModelMixin for KimiK25Model {}
