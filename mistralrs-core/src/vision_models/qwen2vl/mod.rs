use std::{any::Any, sync::Arc};

use candle_core::{Context, Device, Result, Tensor, D};
use candle_nn::VarBuilder;
use mistralrs_quant::QuantMethod;
use text::Qwen2VLTextModel;
use vision::Qwen2VLVisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    dummy_paged_attention::ModelConfigMetadata,
    layers::CausalMasker,
    layers_masker::PastKvLenCache,
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        Cache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
};

mod config;
mod text;
mod vision;

pub(crate) use config::Config;

pub struct Qwen2VLModel {
    text: Qwen2VLTextModel,
    vision: Qwen2VLVisionModel,
    image_token_id: usize,
    video_token_id: usize,
}

impl Qwen2VLModel {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            // TODO!
            candle_core::bail!("Sliding window is unsupported for now!");
        }
        let text = Qwen2VLTextModel::new(
            cfg,
            vb.clone(),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let vision = Qwen2VLVisionModel::new(&cfg.vision_config, vb.pp("vision"))?;
        Ok(Self {
            text,
            vision,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        pixel_values_videos: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        video_grid_thw: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (input_embeds, attention_mask) = if pixel_values.is_some()
            || pixel_values_videos.is_some()
        {
            let mut xs = self.text.embed_tokens(input_ids)?;

            let cache = self.text.cache.lock();
            let attention_mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
                input_ids,
                metadata
                    .as_ref()
                    .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                    .unwrap_or(&*cache as &dyn PastKvLenCache),
                self.text.cfg.sliding_window,
                xs.dtype(),
                self.text.cfg.num_attn_heads,
            )?;

            if let Some(pixel_values) = pixel_values {
                let image_embeds = self.vision.forward(
                    &pixel_values,
                    &image_grid_thw.context("pixel_values require image_grid_thw")?,
                )?;
                let image_mask = input_ids
                    .eq(self.image_token_id as f64)?
                    .unsqueeze(D::Minus1)?
                    .broadcast_as(xs.shape())?;
                xs = image_mask.where_cond(&image_embeds, &xs)?;
            }

            if let Some(pixel_values_videos) = pixel_values_videos {
                let video_embeds = self.vision.forward(
                    &pixel_values_videos,
                    &video_grid_thw.context("pixel_values_videos require video_grid_thw")?,
                )?;
                let video_mask = input_ids
                    .eq(self.video_token_id as f64)?
                    .unsqueeze(D::Minus1)?
                    .broadcast_as(xs.shape())?;
                xs = video_mask.where_cond(&video_embeds, &xs)?;
            }

            (xs, attention_mask)
        } else {
            let xs = self.text.embed_tokens(input_ids)?;
            (xs, None)
        };

        self.text.forward_embeds(
            input_embeds,
            attention_mask.as_ref(),
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

pub(crate) struct Qwen2VLVisionSpecificArgs {
    image_grid_thw: Option<Tensor>,
    video_grid_thw: Option<Tensor>,
    pixel_values_video: Option<Tensor>,
}

impl VisionModel for Qwen2VLModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Qwen2VLVisionSpecificArgs {
            image_grid_thw,
            video_grid_thw,
            pixel_values_video,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Qwen2VLVisionSpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            pixel_values_video,
            image_grid_thw,
            video_grid_thw,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn cache(&self) -> &Cache {
        &self.text.cache
    }
    fn device(&self) -> &Device {
        &self.text.device
    }
    fn max_seq_len(&self) -> usize {
        self.text.max_seq_len
    }
    fn has_conv2d(&self) -> bool {
        true
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.text.cfg
    }
}

impl IsqModel for Qwen2VLModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        todo!()
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        todo!()
    }
}

impl AnyMoeBaseModelMixin for Qwen2VLModel {}
