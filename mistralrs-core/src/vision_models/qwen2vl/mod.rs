#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{Context, DType, Device, IndexOp, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use text::Qwen2VLTextModel;
use vision::Qwen2VLVisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::CausalMasker,
    layers_masker::{masked_fill, PastKvLenCache},
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
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
pub(crate) use inputs_processor::Qwen2VLProcessor;

pub struct Qwen2VLModel {
    text: Qwen2VLTextModel,
    vision: Qwen2VLVisionModel,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Qwen2VLModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            // TODO!
            candle_core::bail!("Sliding window is unsupported for now!");
        }
        // Support both HuggingFace naming (visual.*) and MLX naming (vision_tower.*)
        let vision_vb = if vb.contains_tensor("vision_tower.patch_embed.proj.weight") {
            vb.pp("vision_tower")
        } else {
            vb.pp("visual")
        };
        let vision = Qwen2VLVisionModel::new(
            &cfg.vision_config,
            vision_vb.set_device(normal_loading_metadata.real_device.clone()),
            &normal_loading_metadata.mapper.get_comm_for(0)?,
        )?;
        let text = Qwen2VLTextModel::new(
            cfg,
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
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
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
        rope_img_grid_thw: Option<Tensor>,
        rope_vid_grid_thw: Option<Tensor>,
        seqlens: Vec<usize>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        continuous_vid_pad: Vec<Vec<(usize, usize)>>,
        input_ids_searching: Vec<Vec<u32>>,
        image_nums: Vec<usize>,
        video_nums: Vec<usize>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        image_hashes: &[u64],
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.cfg.sliding_window,
            self.text.dtype,
            self.text.cfg.num_attn_heads,
        )?;

        let input_embeds = if pixel_values.is_some() || pixel_values_videos.is_some() {
            let mut xs = self.text.embed_tokens(input_ids)?;

            if let Some(pixel_values) = pixel_values {
                let mut pixel_values = pixel_values;
                let ndim = pixel_values.dims().len();
                if ndim > 2 {
                    let last_dim = pixel_values.dim(ndim - 1)?;
                    pixel_values = pixel_values.reshape(((), last_dim))?;
                }
                let grid_thw = image_grid_thw
                    .as_ref()
                    .context("pixel_values require image_grid_thw")?;

                let image_embeds = if !image_hashes.is_empty() {
                    let n_images = image_hashes.len();
                    let grid_data = grid_thw.to_vec2::<u32>()?;
                    let patches_per_image: Vec<usize> = grid_data
                        .iter()
                        .map(|row| row[0] as usize * row[1] as usize * row[2] as usize)
                        .collect();

                    let mut per_image: Vec<Option<Tensor>> = vec![None; n_images];
                    let mut miss_indices = Vec::new();
                    {
                        let mut guard = self
                            .encoder_cache
                            .lock()
                            .expect("encoder cache lock poisoned");
                        for (i, &hash) in image_hashes.iter().enumerate() {
                            if let Some(cached) = guard.get(hash) {
                                per_image[i] = Some(cached[0].clone());
                            } else {
                                miss_indices.push(i);
                            }
                        }
                    }

                    if miss_indices.is_empty() {
                        let parts: Vec<Tensor> =
                            per_image.into_iter().map(|t| t.unwrap()).collect();
                        Tensor::cat(&parts, 0)?
                    } else {
                        // Collect miss pixel slices and grid rows
                        let mut miss_pixel_slices = Vec::new();
                        let mut miss_grid_rows = Vec::new();
                        let mut offset = 0usize;
                        for (i, &n_patches) in patches_per_image.iter().enumerate() {
                            if miss_indices.contains(&i) {
                                miss_pixel_slices.push(pixel_values.narrow(0, offset, n_patches)?);
                                miss_grid_rows.push(grid_thw.i(i)?);
                            }
                            offset += n_patches;
                        }
                        let miss_pixels = Tensor::cat(&miss_pixel_slices, 0)?;
                        let miss_grid = Tensor::stack(&miss_grid_rows, 0)?;

                        let encoded = self.vision.forward(&miss_pixels, &miss_grid)?;

                        // Compute output tokens per miss image (after spatial merge)
                        let merge = self.spatial_merge_size as u32;
                        let miss_output_tokens: Vec<usize> = miss_indices
                            .iter()
                            .map(|&i| {
                                let row = &grid_data[i];
                                (row[0] as usize)
                                    * (row[1] as usize / merge as usize)
                                    * (row[2] as usize / merge as usize)
                            })
                            .collect();

                        // Split encoded output and cache per-image
                        let mut enc_offset = 0usize;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            for (j, &orig_idx) in miss_indices.iter().enumerate() {
                                let n_out = miss_output_tokens[j];
                                let single = encoded.narrow(0, enc_offset, n_out)?;
                                enc_offset += n_out;
                                guard.insert(image_hashes[orig_idx], vec![single.clone()]);
                                per_image[orig_idx] = Some(single);
                            }
                        }

                        let parts: Vec<Tensor> =
                            per_image.into_iter().map(|t| t.unwrap()).collect();
                        Tensor::cat(&parts, 0)?
                    }
                } else {
                    self.vision.forward(&pixel_values, grid_thw)?
                }
                .to_dtype(self.text.dtype)?;

                for (batch, batch_ids) in continuous_img_pad.into_iter().enumerate() {
                    let mut last_end = 0;
                    for (start, end) in batch_ids {
                        xs = xs.slice_assign(
                            &[batch..batch + 1, start..end, 0..xs.dim(2)?],
                            &image_embeds
                                .i((last_end..last_end + (end - start), ..))?
                                .unsqueeze(0)?,
                        )?;
                        last_end = end - start;
                    }
                }
            }

            if let Some(pixel_values_videos) = pixel_values_videos {
                let video_embeds = self.vision.forward(
                    &pixel_values_videos,
                    video_grid_thw
                        .as_ref()
                        .context("pixel_values_videos require video_grid_thw")?,
                )?;

                for (batch, batch_ids) in continuous_vid_pad.into_iter().enumerate() {
                    let mut last_end = 0;
                    for (start, end) in batch_ids {
                        xs = xs.slice_assign(
                            &[batch..batch + 1, start..end, 0..xs.dim(2)?],
                            &video_embeds
                                .i((last_end..last_end + (end - start), ..))?
                                .unsqueeze(0)?,
                        )?;
                        last_end = end - start;
                    }
                }
            }

            xs
        } else {
            self.text.embed_tokens(input_ids)?
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
            rope_img_grid_thw.as_ref(),
            rope_vid_grid_thw.as_ref(),
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
            &position_ids,
            context_lens,
            flash_params,
        )?;
        Ok(out)
    }
}

pub(crate) struct Qwen2VLVisionSpecificArgs {
    input_ids_full: Tensor,
    image_grid_thw: Option<Tensor>, // Some when pixel values are provided
    video_grid_thw: Option<Tensor>, // Some when pixel values are provided
    /// Complete image grid THW for ALL images in the full sequence (including prefix-cached ones).
    /// Used for MRoPE position computation. Falls back to `image_grid_thw` if None.
    pub rope_img_grid_thw: Option<Tensor>,
    /// Complete video grid THW for ALL videos in the full sequence (including prefix-cached ones).
    pub rope_vid_grid_thw: Option<Tensor>,
    seqlens: Vec<usize>,
    continuous_img_pad: Vec<Vec<(usize, usize)>>,
    continuous_vid_pad: Vec<Vec<(usize, usize)>>,
    input_ids_searching: Vec<Vec<u32>>,
    image_nums: Vec<usize>,
    video_nums: Vec<usize>,
    pub image_hashes: Vec<u64>,
}

impl VisionModel for Qwen2VLModel {
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
        let Qwen2VLVisionSpecificArgs {
            input_ids_full,
            image_grid_thw,
            video_grid_thw,
            rope_img_grid_thw,
            rope_vid_grid_thw,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            input_ids_searching,
            image_nums,
            video_nums,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Qwen2VLVisionSpecificArgs`");
        let (pixel_values, pixel_values_video) = match (&image_grid_thw, &video_grid_thw) {
            (Some(_), None) => (pixel_values, None),
            (None, Some(_)) => (None, pixel_values),
            (None, None) => (None, None),
            (Some(_), Some(_)) => {
                candle_core::bail!("Images and videos cannot be provided together.")
            }
        };
        // Use the complete grid (covering all images/videos including prefix-cached ones)
        // for MRoPE position computation. Falls back to current-frame grid.
        let rope_img = rope_img_grid_thw.or(image_grid_thw.clone());
        let rope_vid = rope_vid_grid_thw.or(video_grid_thw.clone());
        self.forward(
            input_ids,
            &input_ids_full,
            pixel_values,
            pixel_values_video,
            image_grid_thw,
            video_grid_thw,
            rope_img,
            rope_vid,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            input_ids_searching,
            image_nums,
            video_nums,
            seqlen_offsets,
            context_lens,
            &image_hashes,
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
        Box::new(Qwen2VLVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            image_grid_thw: None,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![input_ids.dims()[1]],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            input_ids_searching: vec![vec![]; input_ids.dims()[0]],
            image_nums: vec![0; input_ids.dims()[0]],
            video_nums: vec![0; input_ids.dims()[0]],
            image_hashes: vec![],
        })
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

impl IsqModel for Qwen2VLModel {
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

impl AnyMoeBaseModelMixin for Qwen2VLModel {}
