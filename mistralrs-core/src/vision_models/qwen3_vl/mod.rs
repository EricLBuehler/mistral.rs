#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use text::Qwen3VLTextModel;
use vision::Qwen3VLVisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::CausalMasker,
    layers_masker::{masked_fill, PastKvLenCache},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
};

mod config;
mod inputs_processor;
mod text;
mod vision;

pub(crate) use config::Config;
pub(crate) use inputs_processor::Qwen3VLProcessor;

pub struct Qwen3VLModel {
    text: Qwen3VLTextModel,
    vision: Qwen3VLVisionModel,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
}

impl Qwen3VLModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision = Qwen3VLVisionModel::new(
            &cfg.vision_config,
            vb.pp("model")
                .pp("visual")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let text = Qwen3VLTextModel::new(
            &cfg.text_config,
            vb.clone(),
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// (position_ids, mrope_position_deltas)
    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        attention_mask_indices: Option<&Tensor>,
        input_ids_searching: Vec<Vec<u32>>,
        image_nums: Vec<usize>,
        video_nums: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mut position_ids = Tensor::zeros(
                (3, input_ids.dim(0)?, input_ids.dim(1)?),
                DType::I64,
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
                if let Some(attention_mask_indices) = attention_mask_indices {
                    input_ids = input_ids
                        .i(i)?
                        .to_dtype(DType::F32)?
                        .index_select(&attention_mask_indices.squeeze(0)?, 0)?
                        .to_dtype(input_ids.dtype())?;
                }
                let image_nums = image_nums[i];
                let vision_nums = video_nums[i];

                let mut llm_pos_ids: Vec<Tensor> = Vec::new();
                let mut max_last_llm_pos_ids = None;
                let mut max_llm_pos_ids = 0;
                let mut st = 0;
                let (mut remain_images, mut remain_videos) = (image_nums, vision_nums);
                for _ in 0..(image_nums + vision_nums) {
                    let ed_image = if input_ids_searching[i].contains(&self.image_token_id)
                        && remain_images > 0
                    {
                        input_ids_searching[i][st..]
                            .iter()
                            .position(|&t| t == self.image_token_id)
                            .unwrap()
                            + st
                    } else {
                        input_ids.dim(0)? + 1
                    };
                    let ed_video = if input_ids_searching[i].contains(&self.video_token_id)
                        && remain_videos > 0
                    {
                        input_ids_searching[i][st..]
                            .iter()
                            .position(|&t| t == self.video_token_id)
                            .unwrap()
                            + st
                    } else {
                        input_ids.dim(0)? + 1
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

                    let st_idx = max_last_llm_pos_ids.unwrap_or(0);
                    // max_last_llm_pos_ids = Some(text_len as i64 + st_idx);
                    max_llm_pos_ids = max_llm_pos_ids.max(text_len as i64 + st_idx);
                    llm_pos_ids.push(
                        Tensor::arange(st_idx, text_len as i64 + st_idx, input_ids.device())?
                            .unsqueeze(0)?
                            .repeat((3, 1))?,
                    );

                    let t_idx = Tensor::arange(0, llm_grid_t as i64, input_ids.device())?
                        .reshape(((), 1))?
                        .repeat((1, llm_grid_h as usize * llm_grid_w as usize))?
                        .flatten_all()?;
                    let h_idx = Tensor::arange(0, llm_grid_h as i64, input_ids.device())?
                        .reshape((1, (), 1))?
                        .repeat((llm_grid_t as usize, 1, llm_grid_w as usize))?
                        .flatten_all()?;
                    let w_idx = Tensor::arange(0, llm_grid_w as i64, input_ids.device())?
                        .reshape((1, 1, ()))?
                        .repeat((llm_grid_t as usize, llm_grid_h as usize, 1))?
                        .flatten_all()?;
                    max_last_llm_pos_ids = Some(
                        *[llm_grid_t, llm_grid_h, llm_grid_w].iter().max().unwrap() as i64
                            + (text_len as i64 + st_idx),
                    );
                    max_llm_pos_ids = max_llm_pos_ids.max(max_last_llm_pos_ids.unwrap());
                    llm_pos_ids.push(
                        (Tensor::stack(&[t_idx, h_idx, w_idx], 0)?.to_dtype(DType::F32)?
                            + (text_len + st_idx as usize) as f64)?
                            .to_dtype(DType::I64)?,
                    );
                    st = ed + (llm_grid_t * llm_grid_h * llm_grid_w) as usize;
                }

                if st < input_ids.dim(0)? {
                    let st_idx = max_last_llm_pos_ids.unwrap_or(0);
                    let text_len = (input_ids.dim(0)? - st) as u32;
                    // max_last_llm_pos_ids = Some(text_len as i64 + st_idx);
                    max_llm_pos_ids = max_llm_pos_ids.max(text_len as i64 + st_idx);
                    llm_pos_ids.push(
                        Tensor::arange(st_idx, text_len as i64 + st_idx, input_ids.device())?
                            .reshape((1, ()))?
                            .repeat((3, 1))?,
                    );
                }

                let llm_positions = Tensor::cat(&llm_pos_ids, 1)?.reshape((3, ()))?;
                let positions_mask = attention_mask
                    .as_ref()
                    .unwrap()
                    .i(i)?
                    .eq(1f64)?
                    .unsqueeze(0)?
                    .repeat((3, 1))?;

                position_ids = position_ids.slice_assign(
                    &[0..position_ids.dim(0)?, i..i + 1, 0..position_ids.dim(2)?],
                    &positions_mask
                        .where_cond(&llm_positions, &position_ids.i((.., i, ..))?)?
                        .unsqueeze(1)?,
                )?;
                mrope_position_deltas
                    .push(max_llm_pos_ids + 1 - total_input_ids.i(i)?.dim(0)? as i64);
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
            let position_ids = (attention_mask.to_dtype(DType::F32)?.cumsum(D::Minus1)? - 1f64)?;
            let position_ids = masked_fill(&position_ids, &attention_mask.eq(0f64)?, 1i64)?;
            let position_ids = position_ids.unsqueeze(0)?.repeat((3, 1, 1))?;

            let max_position_ids = position_ids.max(0)?.max_keepdim(D::Minus1)?;
            let mrope_position_deltas =
                ((max_position_ids + 1.)? - attention_mask.dim(D::Minus1)? as f64)?;

            Ok((
                position_ids.to_dtype(DType::I64)?,
                mrope_position_deltas.to_dtype(DType::I64)?,
            ))
        } else {
            let position_ids = Tensor::arange(0i64, input_ids.dim(1)? as i64, input_ids.device())?
                .reshape((1, 1, ()))?
                .repeat((3, input_ids.dim(0)?, 1))?;
            let mrope_position_deltas =
                Tensor::zeros((input_ids.dim(0)?, 1), DType::I64, input_ids.device())?;

            Ok((position_ids, mrope_position_deltas))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        pixel_values: Option<Tensor>,
        pixel_values_videos: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        video_grid_thw: Option<Tensor>,
        seqlens: Vec<usize>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        continuous_vid_pad: Vec<Vec<(usize, usize)>>,
        input_ids_searching: Vec<Vec<u32>>,
        image_nums: Vec<usize>,
        video_nums: Vec<usize>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.cfg.sliding_window,
            self.text.dtype,
            self.text.cfg.num_attn_heads,
        )?;

        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let (batch_size, seq_len, hidden_dim) = input_embeds.dims3()?;
        let device = input_embeds.device().clone();

        let mut image_mask_opt: Option<Tensor> = None;
        let mut video_mask_opt: Option<Tensor> = None;
        let mut deepstack_image_opt: Option<Vec<Tensor>> = None;
        let mut deepstack_video_opt: Option<Vec<Tensor>> = None;

        if let Some(pixel_values) = &pixel_values {
            let Some(image_grid_thw_ref) = image_grid_thw.as_ref() else {
                candle_core::bail!("pixel_values require image_grid_thw");
            };
            let mut pixel_values = pixel_values.clone();
            let dims = pixel_values.dims();
            if dims.len() == 3 {
                pixel_values = pixel_values.reshape((dims[0] * dims[1], dims[2]))?;
            }
            let (image_embeds, deepstack_image_embeds) =
                self.vision.forward(&pixel_values, image_grid_thw_ref)?;
            let image_embeds = image_embeds.to_device(&device)?.to_dtype(self.text.dtype)?;
            let mut deepstack_image_embeds = deepstack_image_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(self.text.dtype))
                .collect::<Result<Vec<_>>>()?;

            let mut offset = 0usize;
            let mut image_mask =
                Tensor::zeros((batch_size, seq_len), DType::F32, input_ids.device())?;
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

            for (batch, spans) in continuous_img_pad.iter().enumerate() {
                for &(start, end) in spans {
                    let len = end - start;
                    let chunk = image_embeds.narrow(0, offset, len)?;
                    offset += len;
                    input_embeds = input_embeds.slice_assign(
                        &[batch..batch + 1, start..end, 0..hidden_dim],
                        &chunk.unsqueeze(0)?,
                    )?;
                    let ones = Tensor::ones((1, len), DType::F32, input_ids.device())?;
                    image_mask = image_mask.slice_assign(&[batch..batch + 1, start..end], &ones)?;
                }
            }
            image_mask_opt = Some(image_mask.to_dtype(DType::U8)?);
            deepstack_image_opt = Some(deepstack_image_embeds.drain(..).collect());
        }

        if let Some(pixel_values_videos) = &pixel_values_videos {
            let Some(video_grid_thw_ref) = video_grid_thw.as_ref() else {
                candle_core::bail!("pixel_values_videos require video_grid_thw");
            };
            let mut pixel_values = pixel_values_videos.clone();
            let dims = pixel_values.dims();
            if dims.len() == 3 {
                pixel_values = pixel_values.reshape((dims[0] * dims[1], dims[2]))?;
            }
            let (video_embeds, deepstack_video_embeds) =
                self.vision.forward(&pixel_values, video_grid_thw_ref)?;
            let video_embeds = video_embeds.to_device(&device)?.to_dtype(self.text.dtype)?;
            let mut deepstack_video_embeds = deepstack_video_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(self.text.dtype))
                .collect::<Result<Vec<_>>>()?;

            let mut offset = 0usize;
            let mut video_mask =
                Tensor::zeros((batch_size, seq_len), DType::F32, input_ids.device())?;
            let total_expected: usize = continuous_vid_pad
                .iter()
                .flat_map(|spans| spans.iter().map(|(s, e)| e - s))
                .sum();
            if video_embeds.dim(0)? != total_expected {
                candle_core::bail!(
                    "Video embedding length {} does not match placeholder tokens {}",
                    video_embeds.dim(0)?,
                    total_expected
                );
            }

            for (batch, spans) in continuous_vid_pad.iter().enumerate() {
                for &(start, end) in spans {
                    let len = end - start;
                    let chunk = video_embeds.narrow(0, offset, len)?;
                    offset += len;
                    input_embeds = input_embeds.slice_assign(
                        &[batch..batch + 1, start..end, 0..hidden_dim],
                        &chunk.unsqueeze(0)?,
                    )?;
                    let ones = Tensor::ones((1, len), DType::F32, input_ids.device())?;
                    video_mask = video_mask.slice_assign(&[batch..batch + 1, start..end], &ones)?;
                }
            }
            video_mask_opt = Some(video_mask.to_dtype(DType::U8)?);
            deepstack_video_opt = Some(deepstack_video_embeds.drain(..).collect());
        }

        let (visual_pos_masks, deepstack_visual_embeds) = match (
            image_mask_opt,
            deepstack_image_opt,
            video_mask_opt,
            deepstack_video_opt,
        ) {
            (Some(image_mask), Some(image_deepstack), Some(video_mask), Some(video_deepstack)) => {
                let combined =
                    (image_mask.to_dtype(DType::F32)? + video_mask.to_dtype(DType::F32)?)?;
                let visual_mask = combined.gt(0f32)?.to_dtype(DType::U8)?;
                let visual_indices = visual_mask.flatten_all()?.nonzero()?.squeeze(1)?;
                let visual_indices_vec = visual_indices.to_vec1::<i64>()?;

                let image_flat = image_mask
                    .flatten_all()?
                    .to_dtype(DType::U8)?
                    .to_vec1::<u8>()?;
                let num_visual = visual_indices_vec.len();
                if image_deepstack.len() != video_deepstack.len() {
                    candle_core::bail!(
                        "DeepStack image layers ({}) do not match video layers ({})",
                        image_deepstack.len(),
                        video_deepstack.len()
                    );
                }
                let mut combined_layers = Vec::with_capacity(image_deepstack.len());
                for (img_layer, vid_layer) in image_deepstack.iter().zip(video_deepstack.iter()) {
                    let mut rows = Vec::with_capacity(num_visual);
                    let mut img_offset = 0usize;
                    let mut vid_offset = 0usize;
                    for &idx in &visual_indices_vec {
                        let idx = idx as usize;
                        if image_flat[idx] != 0 {
                            rows.push(img_layer.i(img_offset)?);
                            img_offset += 1;
                        } else {
                            rows.push(vid_layer.i(vid_offset)?);
                            vid_offset += 1;
                        }
                    }
                    if img_offset != img_layer.dim(0)? || vid_offset != vid_layer.dim(0)? {
                        candle_core::bail!(
                                "DeepStack feature alignment failed for images ({}/{}) or videos ({}/{})",
                                img_offset,
                                img_layer.dim(0)?,
                                vid_offset,
                                vid_layer.dim(0)?
                            );
                    }
                    let row_refs: Vec<&Tensor> = rows.iter().collect();
                    combined_layers.push(Tensor::stack(&row_refs, 0)?);
                }
                (Some(visual_mask), Some(combined_layers))
            }
            (Some(image_mask), Some(image_deepstack), _, _) => {
                (Some(image_mask), Some(image_deepstack))
            }
            (_, _, Some(video_mask), Some(video_deepstack)) => {
                (Some(video_mask), Some(video_deepstack))
            }
            _ => (None, None),
        };

        let mut ropeidx_attn_mask_bs = Vec::new();
        let max_seqlens = *seqlens.iter().max().unwrap();
        for len in &seqlens {
            ropeidx_attn_mask_bs.push(Tensor::new(
                [vec![1f32; *len], vec![0f32; max_seqlens - len]].concat(),
                input_ids.device(),
            )?);
        }
        let ropeidx_attn_mask = Tensor::stack(&ropeidx_attn_mask_bs, 0)?;
        let mut ropeidx_attn_mask_indices_bs = Vec::new();
        for (len, offset) in seqlens.iter().zip(seqlen_offsets) {
            ropeidx_attn_mask_indices_bs.push(Tensor::from_vec(
                (*offset as i64..(*len as i64 + *offset as i64)).collect(),
                (*len,),
                input_ids.device(),
            )?);
        }
        let ropeidx_attn_mask_indices = Tensor::stack(&ropeidx_attn_mask_indices_bs, 0)?;

        let ropeidx_input_ids = if attention_mask.is_some() {
            input_ids
        } else {
            input_ids_full
        };
        let (position_ids, mrope_position_deltas) = self.get_rope_index(
            ropeidx_input_ids,
            image_grid_thw.as_ref(),
            video_grid_thw.as_ref(),
            Some(&ropeidx_attn_mask),
            Some(&ropeidx_attn_mask_indices),
            input_ids_searching,
            image_nums,
            video_nums,
        )?;

        let position_ids = if attention_mask.is_some() {
            position_ids
        } else {
            let mut position_ids = Tensor::new(
                seqlen_offsets.iter().map(|x| *x as i64).collect::<Vec<_>>(),
                input_ids.device(),
            )?
            .reshape((1, (), 1))?
            .repeat((3, 1, 1))?;

            position_ids = position_ids.broadcast_add(&mrope_position_deltas.unsqueeze(0)?)?;

            position_ids
        };

        let out = self.text.forward_embeds(
            input_embeds,
            attention_mask.as_ref(),
            seqlen_offsets,
            &position_ids,
            context_lens,
            flash_params,
            visual_pos_masks.as_ref(),
            deepstack_visual_embeds
                .as_ref()
                .map(|embeds| embeds.as_slice()),
        )?;
        Ok(out)
    }
}

pub(crate) struct Qwen3VLVisionSpecificArgs {
    input_ids_full: Tensor,
    image_grid_thw: Option<Tensor>, // Some when pixel values are provided
    video_grid_thw: Option<Tensor>, // Some when pixel values are provided
    seqlens: Vec<usize>,
    continuous_img_pad: Vec<Vec<(usize, usize)>>,
    continuous_vid_pad: Vec<Vec<(usize, usize)>>,
    input_ids_searching: Vec<Vec<u32>>,
    image_nums: Vec<usize>,
    video_nums: Vec<usize>,
}

impl VisionModel for Qwen3VLModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Qwen3VLVisionSpecificArgs {
            input_ids_full,
            image_grid_thw,
            video_grid_thw,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            input_ids_searching,
            image_nums,
            video_nums,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Qwen3VLVisionSpecificArgs`");
        let (pixel_values, pixel_values_video) = match (&image_grid_thw, &video_grid_thw) {
            (Some(_), None) => (pixel_values, None),
            (None, Some(_)) => (None, pixel_values),
            (None, None) => (None, None),
            (Some(_), Some(_)) => {
                candle_core::bail!("Images and videos cannot be provided together.")
            }
        };
        self.forward(
            input_ids,
            &input_ids_full,
            pixel_values,
            pixel_values_video,
            image_grid_thw,
            video_grid_thw,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            input_ids_searching,
            image_nums,
            video_nums,
            seqlen_offsets,
            context_lens,
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
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        assert_eq!(input_ids.dims()[0], 1);
        Box::new(Qwen3VLVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            image_grid_thw: None,
            video_grid_thw: None,
            seqlens: vec![input_ids.dims()[1]],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            input_ids_searching: vec![vec![]; input_ids.dims()[0]],
            image_nums: vec![0; input_ids.dims()[0]],
            video_nums: vec![0; input_ids.dims()[0]],
        })
    }
}

impl IsqModel for Qwen3VLModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.text.get_layers()
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.text.residual_tensors()
    }
}

impl AnyMoeBaseModelMixin for Qwen3VLModel {}
