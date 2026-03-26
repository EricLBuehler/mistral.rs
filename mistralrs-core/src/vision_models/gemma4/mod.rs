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
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigLike,
        ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
};

pub(crate) mod audio;
pub(crate) mod audio_processing;
pub mod config;
pub(crate) mod inputs_processor;
mod multimodal_embedding;
pub(crate) mod text;
pub mod vision;

pub(crate) use inputs_processor::Gemma4Processor;

#[derive(Default)]
pub struct Gemma4SpecificArgs {
    pub audio_mel: Option<Tensor>,
    pub audio_mel_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
    pub image_sizes: Vec<(u32, u32)>,
    pub audio_hashes: Vec<u64>,
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
}

impl Gemma4Model {
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
        image_sizes: &[(u32, u32)],
        audio_hashes: &[u64],
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
                        if let Some(cached) = guard.get(hash) {
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
                            guard.insert(image_hashes[idx], vec![feats.clone()]);
                        }
                        per_image[idx] = Some(feats);
                    }
                }
                let parts: Vec<Tensor> = per_image.into_iter().map(|t| t.unwrap()).collect();
                Tensor::cat(&parts, 0)?
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
                self.embed_vision
                    .forward(&vision_features)?
                    .to_dtype(input_embeds.dtype())?
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
                        if let Some(cached) = guard.get(hash) {
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
                            guard.insert(audio_hashes[idx], vec![feats.clone()]);
                        }
                        per_audio[idx] = Some(feats);
                    }
                }
                let parts: Vec<Tensor> = per_audio.into_iter().map(|t| t.unwrap()).collect();
                Tensor::cat(&parts, 0)?
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
                embeds
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

        let ple_vocab_limit = self
            .cfg
            .text_config
            .vocab_size_per_layer_input
            .unwrap_or(self.cfg.text_config.vocab_size);
        let ple_zeros = input_ids.zeros_like()?;
        let ple_inputs_mask = input_ids.lt(ple_vocab_limit as f64)?;
        let ple_input_ids = ple_inputs_mask.where_cond(input_ids, &ple_zeros)?;
        // Match HF: per-layer token embeddings only see hard text tokens.
        // Soft multimodal placeholder slots are zeroed out before
        // get_per_layer_inputs(), even though their token ids are in-vocab.
        let non_image_mask = input_ids.ne(self.cfg.image_token_id as f64)?;
        let ple_input_ids = non_image_mask.where_cond(&ple_input_ids, &ple_zeros)?;
        let non_audio_mask = input_ids.ne(self.cfg.audio_token_id as f64)?;
        let ple_input_ids = non_audio_mask.where_cond(&ple_input_ids, &ple_zeros)?;

        self.language_model.forward_embeds(
            input_ids,
            &ple_input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
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
        let (mut tensors, mapper) = self.language_model.get_layers();
        // tensors.extend(self.vision_tower.get_isq_layers());
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
        uvb_embed_vision
            .pp("embedding_post_projection_norm")
            .add(&self.embed_vision.embedding_post_projection_norm);

        if let Some(ref embed_audio) = self.embed_audio {
            let uvb_embed_audio = uvb_model.pp("embed_audio");
            uvb_embed_audio
                .pp("embedding_post_projection_norm")
                .add(&embed_audio.embedding_post_projection_norm);
        }

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        // Start with text model names
        let mut names = self.language_model.imatrix_names()?;

        // Vision tower layers
        for _ in 0..self.vision_tower.num_encoder_layers() {
            // q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
            for _ in 0..7 {
                names.push(None);
            }
        }

        // Audio tower layers
        if let Some(ref audio) = self.audio_tower {
            for _block in audio.conformer.iter() {
                // 11 quantized layers per conformer block
                for _ in 0..11 {
                    names.push(None);
                }
            }
            // SSCP input projection
            names.push(None);
            // Output projection
            if audio.output_proj.is_some() {
                names.push(None);
            }
        }

        // embed_vision projection
        names.push(None);
        // embed_audio projection
        if self.embed_audio.is_some() {
            names.push(None);
        }

        Ok(names)
    }
}

impl VisionModel for Gemma4Model {
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
            &args.image_sizes,
            &args.audio_hashes,
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

impl AnyMoeBaseModelMixin for Gemma4Model {}
