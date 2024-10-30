use std::{any::Any, sync::Arc};

use candle_core::{Context, DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use mistralrs_quant::QuantMethod;
use text::Qwen2VLTextModel;
use vision::Qwen2VLVisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    dummy_paged_attention::ModelConfigMetadata,
    layers::CausalMasker,
    layers_masker::{masked_fill, PastKvLenCache},
    ops::NonZeroOp,
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        Cache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
};

mod config;
mod inputs_processor;
mod text;
mod vision;

pub(crate) use config::Config;

pub struct Qwen2VLModel {
    text: Qwen2VLTextModel,
    vision: Qwen2VLVisionModel,
    image_token_id: usize,
    video_token_id: usize,
    vision_start_token_id: usize,
    spatial_merge_size: usize,
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
            vision_start_token_id: cfg.vision_start_token_id,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }

    /// (position_ids, mrope_position_deltas)
    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mut position_ids = Tensor::zeros(
                (3, input_ids.dim(0)?, input_ids.dim(1)?),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            let mut mrope_position_deltas = Vec::new();

            let mut image_index = 0;
            let mut video_index = 0;
            for (i, mut input_ids) in total_input_ids
                .chunk(input_ids.dim(0)?, 0)?
                .into_iter()
                .enumerate()
            {
                if let Some(attention_mask) = attention_mask {
                    // transformers uses ==1 because their attn mask is 1 for not masked, we use bias where 0 is not masked
                    input_ids = input_ids.gather(
                        &attention_mask.i(i)?.eq(0.)?.nonzero()?.squeeze(D::Minus1)?,
                        0,
                    )?
                }
                let vision_start_indices = input_ids
                    .eq(self.vision_start_token_id as f64)?
                    .nonzero()?
                    .squeeze(1)?;
                let vision_tokens = input_ids.gather(&(vision_start_indices + 1.)?, 0)?;
                let image_nums = vision_tokens
                    .eq(self.image_token_id as f64)?
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()? as usize;
                let vision_nums = vision_tokens
                    .eq(self.video_token_id as f64)?
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()? as usize;
                let input_tokens = input_ids.to_vec1::<u32>()?;

                let mut llm_pos_ids: Vec<Tensor> = Vec::new();
                let mut st = 0;
                let (mut remain_images, mut remain_videos) = (image_nums, vision_nums);
                for _ in 0..(image_nums + vision_nums) {
                    let ed_image = if input_tokens.contains(&(self.image_token_id as u32))
                        && remain_images > 0
                    {
                        input_tokens[st..]
                            .iter()
                            .position(|p| *p == self.image_token_id as u32)
                            .context("Image token not found")?
                    } else {
                        input_tokens.len() + 1
                    };
                    let ed_video = if input_tokens.contains(&(self.video_token_id as u32))
                        && remain_videos > 0
                    {
                        input_tokens[st..]
                            .iter()
                            .position(|p| *p == self.video_token_id as u32)
                            .context("Image token not found")?
                    } else {
                        input_tokens.len() + 1
                    };
                    let (ed, llm_grid_t, h, w) = if ed_image < ed_video {
                        let t = image_grid_thw.as_ref().unwrap().i((image_index, 0))?;
                        let h = image_grid_thw.as_ref().unwrap().i((image_index, 1))?;
                        let w = image_grid_thw.as_ref().unwrap().i((image_index, 2))?;
                        image_index += 1;
                        remain_images -= 1;
                        (
                            ed_image,
                            t.to_scalar::<u32>()?,
                            h.to_scalar::<u32>()?,
                            w.to_scalar::<u32>()?,
                        )
                    } else {
                        let t = video_grid_thw.as_ref().unwrap().i((video_index, 0))?;
                        let h = video_grid_thw.as_ref().unwrap().i((video_index, 1))?;
                        let w = video_grid_thw.as_ref().unwrap().i((video_index, 2))?;
                        video_index += 1;
                        remain_videos -= 1;
                        (
                            ed_video,
                            t.to_scalar::<u32>()?,
                            h.to_scalar::<u32>()?,
                            w.to_scalar::<u32>()?,
                        )
                    };
                    let llm_grid_h = h / self.spatial_merge_size as u32;
                    let llm_grid_w = w / self.spatial_merge_size as u32;
                    let text_len = ed - st;

                    let st_idx = if llm_pos_ids.len() > 0 {
                        let last = llm_pos_ids.last().unwrap();
                        last.max(0)?.to_scalar::<u32>()? + 1
                    } else {
                        0
                    };
                    llm_pos_ids.push(
                        Tensor::arange(st_idx, text_len as u32 + st_idx, input_ids.device())?
                            .unsqueeze(D::Minus1)?
                            .repeat((3, 1))?,
                    );

                    let t_idx = Tensor::arange(0, llm_grid_t, input_ids.device())?
                        .reshape(((), 1))?
                        .repeat((1, llm_grid_h as usize * llm_grid_w as usize))?
                        .flatten_all()?;
                    let h_idx = Tensor::arange(0, llm_grid_h, input_ids.device())?
                        .reshape((1, (), 1))?
                        .repeat((llm_grid_t as usize, 1, llm_grid_w as usize))?
                        .flatten_all()?;
                    let w_idx = Tensor::arange(0, llm_grid_w, input_ids.device())?
                        .reshape((1, 1, ()))?
                        .repeat((llm_grid_t as usize, llm_grid_h as usize, 1))?
                        .flatten_all()?;
                    llm_pos_ids.push(
                        (Tensor::stack(&[t_idx, h_idx, w_idx], 0)?
                            + (text_len + st_idx as usize) as f64)?,
                    );
                    st = ed + (llm_grid_t * llm_grid_h * llm_grid_w) as usize;
                }

                if st < input_tokens.len() {
                    let st_idx = if llm_pos_ids.len() > 0 {
                        let last = llm_pos_ids.last().unwrap();
                        last.max(0)?.to_scalar::<u32>()? + 1
                    } else {
                        0
                    };
                    let text_len = (input_tokens.len() - st) as u32;
                    llm_pos_ids.push(
                        Tensor::arange(st_idx, text_len + st_idx, input_ids.device())?
                            .reshape((1, ()))?
                            .repeat((3, 1))?,
                    );
                }

                let llm_positions = Tensor::cat(&llm_pos_ids, 1)?.reshape((3, ()))?;
                // transformers uses ==1 because their attn mask is 1 for not masked, we use bias where 0 is not masked
                let positions_mask = attention_mask.as_ref().unwrap().i(i)?.eq(0f64)?;
                position_ids = position_ids.slice_assign(
                    &[&.., &i, &..],
                    &positions_mask
                        .where_cond(&llm_positions, &position_ids.i((.., i, ..))?)?
                        .unsqueeze(1)?,
                )?;
                mrope_position_deltas.push(
                    llm_positions.max(0)?.max(0)?.to_scalar::<u32>()? + 1
                        - total_input_ids.i(i)?.dim(0)? as u32,
                );
            }
            let mrope_position_deltas_len = mrope_position_deltas.len();
            let mrope_position_deltas = Tensor::from_vec(
                mrope_position_deltas,
                (mrope_position_deltas_len,),
                input_ids.device(),
            )?
            .unsqueeze(1)?;
            Ok((position_ids, mrope_position_deltas))
        } else if let Some(attention_mask) = attention_mask {
            // transformers uses ==1 because their attn mask is 1 for not masked, we use bias where 0 is not masked
            // convert to transformers compat here!
            let attention_mask = attention_mask.ne(0f64)?;
            let position_ids = (attention_mask.to_dtype(DType::I64)?.cumsum(D::Minus1)? - 1f64)?;
            let position_ids = masked_fill(&position_ids, &attention_mask.eq(0f64)?, 1i64)?;
            let position_ids = position_ids.unsqueeze(0)?.repeat((3, 1, 1))?;

            let max_position_ids = position_ids.max(0)?.max_keepdim(D::Minus1)?;
            let mrope_position_deltas =
                ((max_position_ids + 1.)? - attention_mask.dim(D::Minus1)? as f64)?;

            Ok((position_ids, mrope_position_deltas))
        } else {
            let position_ids = Tensor::arange(0u32, input_ids.dim(1)? as u32, input_ids.device())?
                .reshape((1, 1, ()))?
                .repeat((3, input_ids.dim(0)?, 1))?;
            let mrope_position_deltas = Tensor::zeros(
                (input_ids.dim(0)?, 1),
                input_ids.dtype(),
                input_ids.device(),
            )?;

            Ok((position_ids, mrope_position_deltas))
        }
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
                    image_grid_thw
                        .as_ref()
                        .context("pixel_values require image_grid_thw")?,
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
                    video_grid_thw
                        .as_ref()
                        .context("pixel_values_videos require video_grid_thw")?,
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

        let (mut position_ids, mrope_position_deltas) = self.get_rope_index(
            input_ids,
            image_grid_thw.as_ref(),
            video_grid_thw.as_ref(),
            attention_mask.as_ref(),
        )?;

        if attention_mask.is_some() {
            position_ids = (position_ids + mrope_position_deltas.unsqueeze(0)?)?;
        }

        self.text.forward_embeds(
            input_embeds,
            attention_mask.as_ref(),
            &position_ids,
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
