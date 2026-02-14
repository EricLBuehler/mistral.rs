#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    device_map::DeviceMapper,
    layers::{self, Activation, RmsNorm},
    models,
    ops::SplitOp,
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    AnyMoeConfig, AnyMoeExpertType,
};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
pub use config::Mistral3Config;
pub use inputs_processor::Mistral3Processor;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use models::mistral::Model as Mistral;
use vision::Mistral3VisionModel;

mod config;
mod inputs_processor;
mod vision;

struct Mistral3PatchMerger {
    merging_layer: Linear,
    spatial_merge_size: usize,
    patch_size: usize,
}

impl Mistral3PatchMerger {
    fn new(cfg: &Mistral3Config, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            merging_layer: layers::linear_no_bias(
                cfg.vision_config.hidden_size * cfg.spatial_merge_size.pow(2),
                cfg.vision_config.hidden_size,
                vb.pp("merging_layer"),
            )?,
            spatial_merge_size: cfg.spatial_merge_size,
            patch_size: cfg.vision_config.patch_size,
        })
    }

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(u32, u32)>) -> Result<Tensor> {
        let image_sizes = image_sizes
            .iter()
            .map(|&(h, w)| (h as usize / self.patch_size, w as usize / self.patch_size))
            .collect::<Vec<_>>();

        let tokens_per_image = image_sizes.iter().map(|&(h, w)| h * w).collect::<Vec<_>>();
        let d = image_features.dim(D::Minus1)?;

        let mut permuted_tensor = Vec::new();

        for (image_index, image_tokens) in image_features
            .split(&tokens_per_image, 0)?
            .iter()
            .enumerate()
        {
            let (h, w) = image_sizes[image_index];
            let image_grid = image_tokens
                .reshape((h, w, d))?
                .permute((2, 0, 1))?
                .unsqueeze(0)?;
            // Equiv of:
            // torch.nn.functional.unfold(image_grid, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size)
            let grid = {
                // The first unfold extracts sliding windows along the height (dim=2),
                // and the second unfolds the width (dim=3).
                let patches = image_grid
                    .unfold(2, self.spatial_merge_size, self.spatial_merge_size)?
                    .unfold(3, self.spatial_merge_size, self.spatial_merge_size)?;
                // patches now has shape: (N, C, n_H, n_W, K, K) where n_H = (H - K) // K + 1 and n_W = (W - K) // K + 1

                let patches = patches.permute((0, 1, 4, 5, 2, 3))?;
                patches.contiguous()?.reshape((
                    1,
                    d * self.spatial_merge_size * self.spatial_merge_size,
                    (),
                ))?
            };
            let grid = grid
                .reshape((d * self.spatial_merge_size.pow(2), ()))?
                .t()?;
            permuted_tensor.push(grid);
        }

        let image_features = Tensor::cat(&permuted_tensor, 0)?;

        self.merging_layer.forward(&image_features)
    }
}

struct Mistral3MultiModalProjector {
    norm: RmsNorm,
    linear_1: Linear,
    linear_2: Linear,
    act: Activation,
    patch_merger: Mistral3PatchMerger,
}

impl Mistral3MultiModalProjector {
    fn new(cfg: &Mistral3Config, vb: ShardedVarBuilder) -> Result<Self> {
        let norm = RmsNorm::new(
            cfg.vision_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("norm"),
        )?;
        // let num_feature_layers = match &cfg.vision_feature_layer {
        //     Either::Left(_) => 1,
        //     Either::Right(r) => r.len(),
        // };
        let num_feature_layers = 1;
        let linear_1 = layers::linear_b(
            cfg.vision_config.hidden_size * num_feature_layers,
            cfg.text_config.hidden_size,
            cfg.multimodal_projector_bias,
            vb.pp("linear_1"),
        )?;
        let linear_2 = layers::linear_b(
            cfg.text_config.hidden_size,
            cfg.text_config.hidden_size,
            cfg.multimodal_projector_bias,
            vb.pp("linear_2"),
        )?;
        let patch_merger = Mistral3PatchMerger::new(cfg, vb.pp("patch_merger"))?;
        Ok(Self {
            norm,
            linear_1,
            linear_2,
            act: cfg.projector_hidden_act,
            patch_merger,
        })
    }

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(u32, u32)>) -> Result<Tensor> {
        let mut hidden_states = self.norm.forward(image_features)?;
        hidden_states = self.patch_merger.forward(&hidden_states, image_sizes)?;
        hidden_states = self.linear_1.forward(&hidden_states)?.apply(&self.act)?;
        self.linear_2.forward(&hidden_states)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("norm").add(&self.norm);
        uvb.pp("linear_1").add(&self.linear_1);
        uvb.pp("linear_2").add(&self.linear_2);
        uvb.pp("patch_merger")
            .pp("merging_layer")
            .add(&self.patch_merger.merging_layer);

        uvb.to_safetensors()
    }
}

pub struct Mistral3Model {
    text_model: Mistral,
    vision_model: Mistral3VisionModel,
    mmproj: Mistral3MultiModalProjector,
    cfg: Mistral3Config,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Mistral3Model {
    pub fn new(
        cfg: &Mistral3Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision_model = Mistral3VisionModel::new(
            &cfg.vision_config,
            vb.pp("vision_tower"),
            &normal_loading_metadata,
        )?;
        let mmproj = Mistral3MultiModalProjector::new(
            cfg,
            vb.pp("multi_modal_projector")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let text_model = Mistral::new(
            &cfg.text_config,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;

        // For get_image_features, assuming this for best efficiency.
        assert_eq!(cfg.vision_feature_layer, -1);

        Ok(Self {
            vision_model,
            text_model,
            mmproj,
            cfg: cfg.clone(),
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn get_image_features(
        &self,
        image_features: &Tensor,
        image_sizes: Vec<(u32, u32)>,
    ) -> Result<Tensor> {
        let image_outputs = self
            .vision_model
            .forward(image_features, image_sizes.clone())?;
        let selected_image_feature = image_outputs;
        self.mmproj
            .forward(&selected_image_feature.squeeze(0)?, image_sizes)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        image_hashes: &[u64],
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        image_sizes: Option<Vec<(u32, u32)>>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut input_embeds = self.text_model.get_input_embeddings(input_ids)?;

        if let Some(pixel_values) = pixel_values {
            let special_image_mask = input_ids
                .eq(self.cfg.image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;
            let mask_flat = special_image_mask.flatten_all()?;
            // Nonzero before vision model to allow async processing all the way through logits.
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            let image_sizes = image_sizes.unwrap();
            let dtype = self.vision_model.dtype();
            let pixel_values = pixel_values.to_dtype(dtype)?;

            // Per-image caching (Mistral3 has variable patch counts per image).
            let n_images = image_hashes.len();
            let image_features = if n_images > 0 {
                let mut per_image: Vec<Tensor> = Vec::with_capacity(n_images);
                let mut miss_indices = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(hash) {
                            per_image.push(cached[0].clone());
                        } else {
                            per_image.push(Tensor::zeros(
                                1,
                                candle_core::DType::F32,
                                pixel_values.device(),
                            )?);
                            miss_indices.push(i);
                        }
                    }
                }
                if !miss_indices.is_empty() {
                    // Encode only misses, one at a time (variable resolution).
                    for &idx in &miss_indices {
                        let single_pv = pixel_values.get(idx)?.unsqueeze(0)?;
                        let single_size = vec![image_sizes[idx]];
                        let feats = self.get_image_features(&single_pv, single_size)?;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(image_hashes[idx], vec![feats.clone()]);
                        }
                        per_image[idx] = feats;
                    }
                }
                Tensor::cat(&per_image, 0)?
            } else {
                self.get_image_features(&pixel_values, image_sizes)?
            };

            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_features.flatten_all()?;

            let current_vals = x_flat.gather(&indices, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

            input_embeds = x_flat.reshape(input_embeds.shape())?;
        }

        self.text_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl IsqModel for Mistral3Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let (mut tensors, mapper) = self.text_model.get_layers();
        tensors.extend(self.vision_model.get_layers());
        (tensors, mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("multi_modal_projector")
            .extend(self.mmproj.residual_tensors());
        uvb.pp("vision_tower")
            .extend(self.vision_model.residual_tensors());
        uvb.pp("language_model")
            .extend(self.text_model.residual_tensors());

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        self.text_model.imatrix_names()
    }
}

#[derive(Default)]
pub struct Mistral3SpecificArgs {
    pub image_sizes: Option<Vec<(u32, u32)>>,
    pub image_hashes: Vec<u64>,
}

impl VisionModel for Mistral3Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn std::any::Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let Mistral3SpecificArgs {
            image_sizes,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Mistral3SpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            &image_hashes,
            seqlen_offsets,
            context_lens,
            image_sizes,
            metadata,
            flash_params,
        )
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Mistral3SpecificArgs::default())
    }
    fn cache(&self) -> &EitherCache {
        self.text_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.text_model.cache_mut()
    }
    fn device(&self) -> &Device {
        self.text_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.text_model.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.text_model.config()
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

impl AnyMoeBaseModelMixin for Mistral3Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.text_model.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.text_model.get_mlps_mut()
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        self.text_model.create_anymoe_layers(
            additional_vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}
