#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor, D};
use config::Gemma4Config;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use text::TextModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    paged_attention::{
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, MultimodalModel, NormalLoadingMetadata,
    },
    speculative::{
        SpeculativeAttachInfo, SpeculativeConfig, SpeculativeProposalBatch,
        SpeculativeProposeBatchCtx, SpeculativeProposer,
    },
    utils::unvarbuilder::UnVarBuilder,
};

pub(crate) mod audio;
pub(crate) mod audio_processing;
pub mod config;
pub(crate) mod inputs_processor;
mod mtp;
mod multimodal_embedding;
pub(crate) mod text;
pub mod vision;

pub(crate) use inputs_processor::Gemma4Processor;

#[derive(Default)]
pub struct Gemma4SpecificArgs {
    pub audio_mel: Option<Tensor>,
    pub audio_mel_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
    pub image_cached_tokens: Vec<usize>,
    pub image_sizes: Vec<(u32, u32)>,
    pub audio_hashes: Vec<u64>,
    pub audio_cached_tokens: Vec<usize>,
    pub video_pixel_values: Option<Tensor>,
    pub video_hashes: Vec<u64>,
    pub video_cached_tokens: Vec<usize>,
    pub video_sizes: Vec<(u32, u32)>,
}

pub struct Gemma4Model {
    language_model: TextModel,
    vision_tower: vision::VisionTower,
    embed_vision: multimodal_embedding::Gemma4MultimodalEmbedder,
    audio_tower: Option<audio::AudioModel>,
    embed_audio: Option<multimodal_embedding::Gemma4MultimodalEmbedder>,
    cfg: Gemma4Config,
    vision_dtype: DType,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
    mtp: Mutex<Option<mtp::Gemma4MtpRuntime>>,
}

impl Gemma4Model {
    fn trim_cached_prefix_tokens(features: Tensor, cached_tokens: usize) -> Result<Tensor> {
        if cached_tokens == 0 {
            return Ok(features);
        }
        let total_tokens = features.dim(0)?;
        if cached_tokens >= total_tokens {
            return features.narrow(0, total_tokens, 0);
        }
        features.narrow(0, cached_tokens, total_tokens - cached_tokens)
    }

    pub fn new(
        cfg: &Gemma4Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb = vb.pp("model");

        let vision_dtype = if vb.dtype() == DType::F16 {
            DType::F32
        } else {
            vb.dtype()
        };
        let audio_dtype = DType::F32;

        let vision_tower = vision::VisionTower::new(
            &cfg.vision_config,
            normal_loading_metadata
                .mapper
                .set_nm_device(vb.pp("vision_tower"), false)
                .set_dtype(vision_dtype),
        )?;

        let vis_hidden = cfg.vision_config.hidden_size;
        let text_hidden = cfg.text_config.hidden_size;
        let embed_vision = multimodal_embedding::Gemma4MultimodalEmbedder::new(
            vis_hidden,
            text_hidden,
            cfg.vision_config.rms_norm_eps,
            normal_loading_metadata
                .mapper
                .set_nm_device(vb.pp("embed_vision"), false)
                .set_dtype(vision_dtype),
        )?;

        let (audio_tower, embed_audio) = if let Some(ref audio_cfg) = cfg.audio_config {
            let tower = audio::AudioModel::new(
                audio_cfg,
                normal_loading_metadata
                    .mapper
                    .set_nm_device(vb.pp("audio_tower"), false)
                    .set_dtype(audio_dtype),
            )?;
            let audio_hidden = audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size);
            let embed = multimodal_embedding::Gemma4MultimodalEmbedder::new(
                audio_hidden,
                text_hidden,
                audio_cfg.rms_norm_eps,
                normal_loading_metadata
                    .mapper
                    .set_nm_device(vb.pp("embed_audio"), false)
                    .set_dtype(audio_dtype),
            )?;
            (Some(tower), Some(embed))
        } else {
            (None, None)
        };

        let language_model = TextModel::new(
            &cfg.text_config,
            Some(cfg.image_token_id),
            Some(cfg.video_token_id),
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;

        Ok(Self {
            language_model,
            vision_tower,
            embed_vision,
            audio_tower,
            embed_audio,
            cfg: cfg.clone(),
            vision_dtype,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
            mtp: Mutex::new(None),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        audio_mel: Option<&Tensor>,
        audio_mel_mask: Option<&Tensor>,
        image_hashes: &[u64],
        image_cached_tokens: &[usize],
        image_sizes: &[(u32, u32)],
        audio_hashes: &[u64],
        audio_cached_tokens: &[usize],
        video_pixel_values: Option<&Tensor>,
        video_hashes: &[u64],
        video_cached_tokens: &[usize],
        video_sizes: &[(u32, u32)],
    ) -> Result<Tensor> {
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;

        if let Some(ref pixel_values) = pixel_values {
            let image_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(self.cfg.image_token_id as f64)?;
            let image_mask_expanded = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;
            let indices = image_mask_expanded.flatten_all()?.nonzero()?.squeeze(1)?;

            let n_images = pixel_values.dim(0)?;
            let crop_image = |pv: Tensor, idx: usize| -> Result<Tensor> {
                if let Some((h, w)) = image_sizes.get(idx).copied() {
                    let (h, w) = (h as usize, w as usize);
                    pv.narrow(2, 0, h)?.narrow(3, 0, w)
                } else {
                    Ok(pv)
                }
            };
            let image_embeds = if !image_hashes.is_empty() && image_hashes.len() == n_images {
                let mut per_image: Vec<Option<Tensor>> = vec![None; n_images];
                let mut miss_indices = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(CacheModality::Image, hash) {
                            per_image[i] = Some(cached[0].clone());
                        } else {
                            miss_indices.push(i);
                        }
                    }
                }
                if !miss_indices.is_empty() {
                    for &idx in &miss_indices {
                        let single_pv = crop_image(pixel_values.get(idx)?.unsqueeze(0)?, idx)?;
                        let vision_features = self
                            .vision_tower
                            .forward(&[single_pv.to_dtype(self.vision_dtype)?])?;
                        let feats = self
                            .embed_vision
                            .forward(&vision_features)?
                            .to_dtype(input_embeds.dtype())?
                            .squeeze(0)?;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(
                                CacheModality::Image,
                                image_hashes[idx],
                                vec![feats.clone()],
                            );
                        }
                        per_image[idx] = Some(feats);
                    }
                }
                let parts: Vec<Tensor> = per_image.into_iter().map(|t| t.unwrap()).collect();
                Self::trim_cached_prefix_tokens(
                    Tensor::cat(&parts, 0)?,
                    image_cached_tokens.first().copied().unwrap_or(0),
                )?
            } else {
                let per_image_tensors: Vec<Tensor> = (0..n_images)
                    .map(|i| {
                        pixel_values
                            .get(i)
                            .and_then(|t| t.unsqueeze(0))
                            .and_then(|t| crop_image(t, i))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let vision_features = self.vision_tower.forward(
                    &per_image_tensors
                        .iter()
                        .map(|t| t.to_dtype(self.vision_dtype))
                        .collect::<Result<Vec<_>>>()?,
                )?;
                let embeds = self
                    .embed_vision
                    .forward(&vision_features)?
                    .to_dtype(input_embeds.dtype())?
                    .squeeze(0)?;
                Self::trim_cached_prefix_tokens(
                    embeds,
                    image_cached_tokens.first().copied().unwrap_or(0),
                )?
            };

            if indices.dim(0)? > 0 {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = image_embeds.flatten_all()?;
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (src_flat - current_vals)?;
                x_flat = x_flat.scatter_add(&indices, &diff, 0)?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        if let (
            Some(audio_mel),
            Some(audio_mel_mask),
            Some(ref audio_tower),
            Some(ref embed_audio),
        ) = (
            audio_mel,
            audio_mel_mask,
            &self.audio_tower,
            &self.embed_audio,
        ) {
            let audio_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(self.cfg.audio_token_id as f64)?;
            let audio_mask_expanded = audio_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;
            let indices = audio_mask_expanded.flatten_all()?.nonzero()?.squeeze(1)?;

            let n_audio = audio_mel.dim(0)?;
            let audio_embeds = if !audio_hashes.is_empty() && audio_hashes.len() == n_audio {
                let mut per_audio: Vec<Option<Tensor>> = vec![None; n_audio];
                let mut miss_indices = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in audio_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(CacheModality::Audio, hash) {
                            per_audio[i] = Some(cached[0].clone());
                        } else {
                            miss_indices.push(i);
                        }
                    }
                }
                if !miss_indices.is_empty() {
                    for &idx in &miss_indices {
                        let single_mel = audio_mel.get(idx)?.unsqueeze(0)?;
                        let single_mask = audio_mel_mask.get(idx)?.unsqueeze(0)?;
                        let (audio_features, enc_mask) =
                            audio_tower.forward(&single_mel, &single_mask)?;
                        let valid = enc_mask.eq(0.0)?;
                        let valid_indices =
                            valid.squeeze(0)?.flatten_all()?.nonzero()?.squeeze(1)?;
                        let valid_features = audio_features
                            .squeeze(0)?
                            .contiguous()?
                            .index_select(&valid_indices, 0)?;
                        let feats = embed_audio
                            .forward(&valid_features.unsqueeze(0)?)?
                            .to_dtype(input_embeds.dtype())?
                            .squeeze(0)?;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(
                                CacheModality::Audio,
                                audio_hashes[idx],
                                vec![feats.clone()],
                            );
                        }
                        per_audio[idx] = Some(feats);
                    }
                }
                let parts: Vec<Tensor> = per_audio.into_iter().map(|t| t.unwrap()).collect();
                Self::trim_cached_prefix_tokens(
                    Tensor::cat(&parts, 0)?,
                    audio_cached_tokens.first().copied().unwrap_or(0),
                )?
            } else {
                let (audio_features, enc_mask) = audio_tower.forward(audio_mel, audio_mel_mask)?;
                let valid = enc_mask.eq(0.0)?;
                let batch = audio_features.dim(0)?;
                let mut all_feats = Vec::new();
                for b in 0..batch {
                    let valid_indices = valid.get(b)?.flatten_all()?.nonzero()?.squeeze(1)?;
                    let feats = audio_features
                        .get(b)?
                        .contiguous()?
                        .index_select(&valid_indices, 0)?;
                    all_feats.push(feats);
                }
                let audio_feats = Tensor::cat(&all_feats, 0)?.unsqueeze(0)?;
                let embeds = embed_audio
                    .forward(&audio_feats)?
                    .to_dtype(input_embeds.dtype())?
                    .squeeze(0)?;
                Self::trim_cached_prefix_tokens(
                    embeds,
                    audio_cached_tokens.first().copied().unwrap_or(0),
                )?
            };

            if indices.dim(0)? > 0 {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = audio_embeds.flatten_all()?;
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (src_flat - current_vals)?;
                x_flat = x_flat.scatter_add(&indices, &diff, 0)?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        // ── Video embedding (same vision tower as images) ──────────────
        if let Some(vid_pixel_values) = video_pixel_values {
            let video_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(self.cfg.video_token_id as f64)?;
            let video_mask_expanded = video_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;
            let indices = video_mask_expanded.flatten_all()?.nonzero()?.squeeze(1)?;

            let n_frames = vid_pixel_values.dim(0)?;
            let crop_frame = |pv: Tensor, idx: usize| -> Result<Tensor> {
                if let Some((h, w)) = video_sizes.get(idx).copied() {
                    let (h, w) = (h as usize, w as usize);
                    pv.narrow(2, 0, h)?.narrow(3, 0, w)
                } else {
                    Ok(pv)
                }
            };

            let video_embeds = if !video_hashes.is_empty() && video_hashes.len() == n_frames {
                let mut per_frame: Vec<Option<Tensor>> = vec![None; n_frames];
                let mut miss_indices = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in video_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(CacheModality::Video, hash) {
                            per_frame[i] = Some(cached[0].clone());
                        } else {
                            miss_indices.push(i);
                        }
                    }
                }
                if !miss_indices.is_empty() {
                    for &idx in &miss_indices {
                        let single_pv = crop_frame(vid_pixel_values.get(idx)?.unsqueeze(0)?, idx)?;
                        let vision_features = self
                            .vision_tower
                            .forward(&[single_pv.to_dtype(self.vision_dtype)?])?;
                        let feats = self
                            .embed_vision
                            .forward(&vision_features)?
                            .to_dtype(input_embeds.dtype())?
                            .squeeze(0)?;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(
                                CacheModality::Video,
                                video_hashes[idx],
                                vec![feats.clone()],
                            );
                        }
                        per_frame[idx] = Some(feats);
                    }
                }
                let parts: Vec<Tensor> = per_frame.into_iter().map(|t| t.unwrap()).collect();
                // Sum all per-frame cached counts — unlike images (where fully-cached
                // ones are skipped), ALL video frames are sent when not fully cached,
                // so we must trim the total cached prefix from the concatenated features.
                let total_cached: usize = video_cached_tokens.iter().copied().sum();
                Self::trim_cached_prefix_tokens(Tensor::cat(&parts, 0)?, total_cached)?
            } else {
                let per_frame_tensors: Vec<Tensor> = (0..n_frames)
                    .map(|i| {
                        vid_pixel_values
                            .get(i)
                            .and_then(|t| t.unsqueeze(0))
                            .and_then(|t| crop_frame(t, i))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let vision_features = self.vision_tower.forward(
                    &per_frame_tensors
                        .iter()
                        .map(|t| t.to_dtype(self.vision_dtype))
                        .collect::<Result<Vec<_>>>()?,
                )?;
                let embeds = self
                    .embed_vision
                    .forward(&vision_features)?
                    .to_dtype(input_embeds.dtype())?
                    .squeeze(0)?;
                let total_cached: usize = video_cached_tokens.iter().copied().sum();
                Self::trim_cached_prefix_tokens(embeds, total_cached)?
            };

            if indices.dim(0)? > 0 {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = video_embeds.flatten_all()?;
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (src_flat - current_vals)?;
                x_flat = x_flat.scatter_add(&indices, &diff, 0)?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        let ple_vocab_limit = self
            .cfg
            .text_config
            .vocab_size_per_layer_input
            .unwrap_or(self.cfg.text_config.vocab_size);
        let ple_zeros = input_ids.zeros_like()?;
        let ple_inputs_mask = input_ids.lt(ple_vocab_limit as f64)?;
        let ple_input_ids = ple_inputs_mask.where_cond(input_ids, &ple_zeros)?;
        let non_image_mask = input_ids.ne(self.cfg.image_token_id as f64)?;
        let ple_input_ids = non_image_mask.where_cond(&ple_input_ids, &ple_zeros)?;
        let non_audio_mask = input_ids.ne(self.cfg.audio_token_id as f64)?;
        let ple_input_ids = non_audio_mask.where_cond(&ple_input_ids, &ple_zeros)?;
        let non_video_mask = input_ids.ne(self.cfg.video_token_id as f64)?;
        let ple_input_ids = non_video_mask.where_cond(&ple_input_ids, &ple_zeros)?;

        self.language_model.forward_embeds(
            input_ids,
            &ple_input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
            pixel_values.is_some() || video_pixel_values.is_some(),
        )
    }
}

impl IsqModel for Gemma4Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let (tensors, mapper) = self.language_model.get_layers();
        (tensors, mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_model = uvb.pp("model");

        let uvb_language = uvb_model.pp("language_model");
        uvb_language.extend(self.language_model.residual_tensors());

        let uvb_vision = uvb_model.pp("vision_tower");
        uvb_vision.extend(self.vision_tower.residual_tensors());

        if let Some(ref audio) = self.audio_tower {
            let uvb_audio = uvb_model.pp("audio_tower");
            uvb_audio.extend(audio.residual_tensors());
        }

        let uvb_embed_vision = uvb_model.pp("embed_vision");
        uvb_embed_vision.extend(self.embed_vision.residual_tensors());

        if let Some(ref embed_audio) = self.embed_audio {
            let uvb_embed_audio = uvb_model.pp("embed_audio");
            uvb_embed_audio.extend(embed_audio.residual_tensors());
        }

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        self.language_model.imatrix_names()
    }
}

impl MultimodalModel for Gemma4Model {
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
        let args = model_specific_args
            .downcast::<Gemma4SpecificArgs>()
            .expect("Downcast to Gemma4SpecificArgs failed");

        self.forward(
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
            args.audio_mel.as_ref(),
            args.audio_mel_mask.as_ref(),
            &args.image_hashes,
            &args.image_cached_tokens,
            &args.image_sizes,
            &args.audio_hashes,
            &args.audio_cached_tokens,
            args.video_pixel_values.as_ref(),
            &args.video_hashes,
            &args.video_cached_tokens,
            &args.video_sizes,
        )
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Gemma4SpecificArgs::default())
    }

    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }

    fn cache_mut(&mut self) -> &mut EitherCache {
        self.language_model.cache_mut()
    }

    fn device(&self) -> &Device {
        self.language_model.device()
    }

    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }

    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }

    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.language_model.model_config_like()
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

impl crate::speculative::SpeculativeTargetMixin for Gemma4Model {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> candle_core::Result<Option<SpeculativeAttachInfo>> {
        let SpeculativeConfig::Mtp(config) = config else {
            *self.mtp.lock().expect("MTP mutex poisoned") = None;
            return Ok(None);
        };
        let assistant = config.model.clone();
        let runtime = mtp::Gemma4MtpRuntime::load(
            config,
            &self.cfg.text_config,
            self.language_model.device(),
            self.language_model.device_mapper(),
            false,
        )?;
        let attach_info = SpeculativeAttachInfo::mtp(assistant, runtime.proposal_len());
        *self.mtp.lock().expect("MTP mutex poisoned") = Some(runtime);
        Ok(Some(attach_info))
    }

    fn has_speculative_proposer(&self) -> bool {
        self.mtp.lock().is_ok_and(|mtp| mtp.is_some())
    }

    fn speculative_proposal_len(&self) -> Option<usize> {
        self.mtp
            .lock()
            .ok()
            .and_then(|mtp| mtp.as_ref().map(SpeculativeProposer::proposal_len))
    }

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> candle_core::Result<Option<SpeculativeProposalBatch>> {
        let embedder = |token: &Tensor| self.language_model.embed_tokens(token);
        let mut guard = self.mtp.lock().expect("MTP mutex poisoned");
        let Some(runtime) = guard.as_mut() else {
            return Ok(None);
        };
        runtime.propose(ctx, Some(&embedder)).map(Some)
    }

    fn speculative_target_hiddens(
        &self,
        rows: &[(usize, usize)],
    ) -> candle_core::Result<Option<Tensor>> {
        let hidden = self.language_model.last_spec_hidden().ok_or_else(|| {
            candle_core::Error::Msg(
                "MTP target hidden state was not captured before proposal.".to_string(),
            )
        })?;
        if rows.is_empty() {
            return Ok(None);
        }
        match hidden.dims() {
            [batch, row_count, _] => {
                let mut gathered = Vec::with_capacity(rows.len());
                for &(batch_idx, row) in rows {
                    if batch_idx >= *batch {
                        candle_core::bail!(
                            "MTP hidden batch {batch_idx} is out of range for {batch}"
                        );
                    }
                    if row >= *row_count {
                        candle_core::bail!(
                            "MTP hidden row {row} is out of range for {row_count} rows"
                        );
                    }
                    gathered.push(hidden.narrow(0, batch_idx, 1)?.narrow(1, row, 1)?);
                }
                Tensor::cat(&gathered, 0).map(Some)
            }
            [row_count, _] => {
                let mut gathered = Vec::with_capacity(rows.len());
                for &(batch_idx, row) in rows {
                    if batch_idx != 0 {
                        candle_core::bail!(
                            "MTP hidden batch {batch_idx} is out of range for single-batch hidden state"
                        );
                    }
                    if row >= *row_count {
                        candle_core::bail!(
                            "MTP hidden row {row} is out of range for {row_count} rows"
                        );
                    }
                    gathered.push(hidden.narrow(0, row, 1)?.unsqueeze(0)?);
                }
                Tensor::cat(&gathered, 0).map(Some)
            }
            shape => candle_core::bail!("MTP hidden state has unsupported shape {shape:?}"),
        }
    }
}

impl AnyMoeBaseModelMixin for Gemma4Model {}
