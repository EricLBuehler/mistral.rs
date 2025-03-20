#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{
    layers::{self, Activation, RmsNorm},
    models,
    ops::{NonZeroOp, SplitOp},
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        NormalLoadingMetadata,
    },
};
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module};
use config::Mistral3Config;
use mistralrs_quant::ShardedVarBuilder;
use models::mistral::Model as Mistral;
use vision::VisionModel;

mod config;
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

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(usize, usize)>) -> Result<Tensor> {
        let image_sizes = image_sizes
            .iter()
            .map(|&(h, w)| (h / self.patch_size, w / self.patch_size))
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

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut hidden_states = self.norm.forward(image_features)?;
        hidden_states = self.patch_merger.forward(&hidden_states, image_sizes)?;
        hidden_states = self.linear_1.forward(&hidden_states)?.apply(&self.act)?;
        self.linear_2.forward(&hidden_states)
    }
}

pub struct Mistral3Model {
    text_model: Mistral,
    vision_model: VisionModel,
    mmproj: Mistral3MultiModalProjector,
    cfg: Mistral3Config,
}

impl Mistral3Model {
    pub fn new(
        cfg: &Mistral3Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision_model = VisionModel::new(&cfg.vision_config, vb.pp("vision_tower"))?;
        let text_model = Mistral::new(
            &cfg.text_config,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let mmproj = Mistral3MultiModalProjector::new(cfg, vb.pp("multi_modal_projector"))?;

        // For get_image_features, assuming this for best efficiency.
        assert_eq!(cfg.vision_feature_layer, -1);

        Ok(Self {
            vision_model,
            text_model,
            mmproj,
            cfg: cfg.clone(),
        })
    }

    fn get_image_features(
        &self,
        image_features: &Tensor,
        image_sizes: Vec<(usize, usize)>,
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
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        image_sizes: Option<Vec<(usize, usize)>>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut input_embeds = self.text_model.get_input_embeddings(input_ids)?;

        if let Some(pixel_values) = pixel_values {
            let image_sizes = image_sizes.unwrap();
            let image_features = self.get_image_features(&pixel_values, image_sizes)?;

            let special_image_mask = input_ids
                .eq(self.cfg.image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;

            let mask_flat = special_image_mask.flatten_all()?;
            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_features.flatten_all()?;

            let indices = mask_flat.nonzero()?.squeeze(1)?;
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
