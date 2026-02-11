use std::{cmp::Ordering, fs::File, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Module;
use either::Either;
use hf_hub::api::sync::{Api, ApiError, ApiRepo};
use image::{imageops::FilterType, DynamicImage, RgbImage};
use indexmap::IndexMap;
use mistralrs_quant::ShardedVarBuilder;
use serde::Deserialize;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{info, warn};

use crate::{
    diffusion_models::{
        clip::text::{ClipConfig, ClipTextTransformer},
        flux,
        t5::{self, T5EncoderModel},
        DiffusionGenerationParams,
    },
    pipeline::{
        chat_template::{apply_chat_template_to, ChatTemplate, ChatTemplateValue},
        DiffusionModel,
    },
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
};

use super::sampling::FlowMatchScheduleConfig;
use super::{
    autoencoder::AutoEncoder,
    autoencoder_kl::AutoEncoderKL,
    model::Flux,
    model_flux2::Flux2,
    text_encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig},
};

/// VAE wrapper that supports both FLUX.1 and FLUX.2 autoencoder formats
#[derive(Debug, Clone)]
pub enum FluxVae {
    /// Original FLUX.1 autoencoder format
    Flux1(AutoEncoder),
    /// FLUX.2 diffusers-style AutoencoderKL format
    Flux2(AutoEncoderKL),
}

impl FluxVae {
    pub fn decode(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        match self {
            FluxVae::Flux1(vae) => vae.decode(xs),
            FluxVae::Flux2(vae) => vae.decode(xs),
        }
    }

    #[allow(dead_code)]
    pub fn encode(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        match self {
            FluxVae::Flux1(vae) => vae.encode(xs),
            FluxVae::Flux2(vae) => vae.encode(xs),
        }
    }

    pub fn denormalize_packed(
        &self,
        xs: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        match self {
            FluxVae::Flux1(_) => Ok(xs.clone()),
            FluxVae::Flux2(vae) => vae.denormalize_packed(xs),
        }
    }

    pub fn normalize_packed(
        &self,
        xs: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        match self {
            FluxVae::Flux1(_) => Ok(xs.clone()),
            FluxVae::Flux2(vae) => vae.normalize_packed(xs),
        }
    }
}

/// Transformer model wrapper supporting both FLUX.1 and FLUX.2
pub enum FluxTransformer {
    Flux1(Flux),
    Flux2(Flux2),
}

impl FluxTransformer {
    pub fn compute_pe(&self, txt_ids: &Tensor, img_ids: &Tensor) -> Result<Tensor> {
        match self {
            FluxTransformer::Flux1(model) => model.compute_pe(txt_ids, img_ids),
            FluxTransformer::Flux2(model) => model.compute_pe(txt_ids, img_ids),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        pe: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            FluxTransformer::Flux1(model) => {
                model.forward(img, img_ids, txt, txt_ids, timesteps, y, guidance, pe)
            }
            FluxTransformer::Flux2(model) => {
                // FLUX.2 doesn't use guidance or y (CLIP pooled embedding)
                model.forward(img, img_ids, txt, txt_ids, timesteps, pe)
            }
        }
    }
}

const T5_XXL_SAFETENSOR_FILES: &[&str] =
    &["t5_xxl-shard-0.safetensors", "t5_xxl-shard-1.safetensors"];

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperShift {
    pub base_shift: f64,
    pub max_shift: f64,
    pub guidance_scale: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperConfig {
    pub num_steps: usize,
    pub guidance_config: Option<FluxStepperShift>,
    pub is_guidance: bool,
}

impl FluxStepperConfig {
    pub fn default_for_guidance(has_guidance: bool) -> Self {
        if has_guidance {
            Self {
                num_steps: 50,
                guidance_config: Some(FluxStepperShift {
                    base_shift: 0.5,
                    max_shift: 1.15,
                    guidance_scale: 4.0,
                }),
                is_guidance: true,
            }
        } else {
            Self {
                num_steps: 4,
                guidance_config: None,
                is_guidance: false,
            }
        }
    }
}

pub struct FluxStepper {
    cfg: FluxStepperConfig,
    /// Tokenizer for text encoding (T5 for FLUX.1, Qwen3 for FLUX.2)
    text_tok: Tokenizer,
    clip_tok: Option<Tokenizer>,
    clip_text: Option<ClipTextTransformer>,
    /// Qwen3 text encoder for FLUX.2 (None for FLUX.1)
    qwen3_encoder: Option<Qwen3TextEncoder>,
    flux2_chat_template: Option<ChatTemplateValue>,
    flux2_max_seq_len: usize,
    flux_model: FluxTransformer,
    flux_vae: FluxVae,
    is_guidance: bool,
    is_flux2: bool,
    rope_axes: usize,
    flux2_schedule_cfg: FlowMatchScheduleConfig,
    device: Device,
    dtype: DType,
    preview_sender: Option<UnboundedSender<Vec<DynamicImage>>>,
    api: Api,
    silent: bool,
    offloaded: bool,
    /// Number of latent channels (16 for FLUX.1, 32 for FLUX.2)
    latent_channels: usize,
}

fn get_t5_tokenizer(api: &Api) -> anyhow::Result<Tokenizer> {
    let tokenizer_filename = api
        .model("EricB/t5_tokenizer".to_string())
        .get("t5-v1_1-xxl.tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    Ok(tokenizer)
}

fn get_t5_model(
    api: &Api,
    dtype: DType,
    device: &Device,
    silent: bool,
    offloaded: bool,
) -> candle_core::Result<T5EncoderModel> {
    let repo = api.repo(hf_hub::Repo::with_revision(
        "EricB/t5-v1_1-xxl-enc-only".to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));

    let vb = from_mmaped_safetensors(
        T5_XXL_SAFETENSOR_FILES
            .iter()
            .map(|f| repo.get(f))
            .collect::<std::result::Result<Vec<_>, ApiError>>()
            .map_err(candle_core::Error::msg)?,
        vec![],
        Some(dtype),
        device,
        vec![None],
        silent,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;
    let config_filename = repo.get("config.json").map_err(candle_core::Error::msg)?;
    let config = std::fs::read_to_string(config_filename)?;
    let config: t5::Config = serde_json::from_str(&config).map_err(candle_core::Error::msg)?;

    t5::T5EncoderModel::load(vb, &config, device, offloaded)
}

fn get_clip_model_and_tokenizer(
    api: &Api,
    device: &Device,
    silent: bool,
) -> anyhow::Result<(ClipTextTransformer, Tokenizer)> {
    let repo = api.repo(hf_hub::Repo::model(
        "openai/clip-vit-large-patch14".to_string(),
    ));

    let model_file = repo.get("model.safetensors")?;
    let vb = from_mmaped_safetensors(
        vec![model_file],
        vec![],
        None,
        device,
        vec![None],
        silent,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;
    let config_file = repo.get("config.json")?;
    let config: ClipConfig = serde_json::from_reader(File::open(config_file)?)?;
    let config = config.text_config;
    let model = ClipTextTransformer::new(vb.pp("text_model"), &config)?;

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    Ok((model, tokenizer))
}

fn get_tokenization(tok: &Tokenizer, prompts: Vec<String>, device: &Device) -> Result<Tensor> {
    Tensor::new(
        tok.encode_batch(prompts, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect::<Vec<_>>(),
        device,
    )
}

fn get_qwen3_tokenization_with_mask(
    tok: &Tokenizer,
    prompts: Vec<String>,
    device: &Device,
    chat_template: Option<&ChatTemplateValue>,
) -> Result<(Tensor, Tensor)> {
    // Tokenizer is pre-configured with padding/truncation at FluxStepper init
    let mut rendered = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        if let Some(template) = chat_template {
            let mut msg = IndexMap::new();
            msg.insert("role".to_string(), Either::Left("user".to_string()));
            msg.insert("content".to_string(), Either::Left(prompt));
            let text = apply_chat_template_to(
                vec![msg],
                true,
                Some(false),
                None,
                template,
                None,
                None,
                None,
                Vec::new(),
            )
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            rendered.push(text);
        } else {
            rendered.push(prompt);
        }
    }

    let encodings = tok
        .encode_batch(rendered, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let input_ids = encodings
        .iter()
        .map(|e| e.get_ids().to_vec())
        .collect::<Vec<_>>();
    let attention_mask = encodings
        .iter()
        .map(|e| e.get_attention_mask().to_vec())
        .collect::<Vec<_>>();
    Ok((
        Tensor::new(input_ids, device)?,
        Tensor::new(attention_mask, device)?,
    ))
}

/// Helper to parse the model.safetensors.index.json and get shard file paths
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// Load Qwen3 text encoder and tokenizer for FLUX.2
fn get_qwen3_encoder_and_tokenizer(
    api: &ApiRepo,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> anyhow::Result<(Qwen3TextEncoder, Tokenizer, Option<ChatTemplateValue>)> {
    info!("Loading Qwen3 text encoder for FLUX.2");

    // Get the index file to find all shard files
    let index_file = api.get("text_encoder/model.safetensors.index.json")?;
    let index_content = std::fs::read_to_string(&index_file)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    // Get unique shard files
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    info!("Loading {} text encoder shards", shard_files.len());

    // Download all shard files
    let shard_paths: Vec<std::path::PathBuf> = shard_files
        .iter()
        .map(|f| api.get(&format!("text_encoder/{}", f)))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // Load into VarBuilder
    let vb = from_mmaped_safetensors(
        shard_paths,
        vec![],
        Some(dtype),
        device,
        vec![None],
        silent,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;

    // Load config
    let config_file = api.get("text_encoder/config.json")?;
    let config_content = std::fs::read_to_string(&config_file)?;
    let config: Qwen3TextEncoderConfig = serde_json::from_str(&config_content)?;

    info!(
        hidden_size = config.hidden_size,
        num_heads = config.num_attention_heads,
        num_kv_heads = config.num_key_value_heads,
        head_dim = ?config.head_dim,
        head_dim_effective = config.head_dim(),
        "Qwen3 text encoder config"
    );

    // Build encoder
    let encoder = Qwen3TextEncoder::new(&config, vb)?;

    // Load tokenizer (in tokenizer/ directory, not text_encoder/)
    let tokenizer_file = api.get("tokenizer/tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    info!(
        "Qwen3 text encoder loaded (output dim: {})",
        encoder.output_dim()
    );

    let chat_template = load_flux2_chat_template(api).unwrap_or(None);
    Ok((encoder, tokenizer, chat_template))
}

fn load_flux2_scheduler_config(api: &ApiRepo) -> anyhow::Result<FlowMatchScheduleConfig> {
    let config_path = api.get("scheduler/scheduler_config.json")?;
    let config = std::fs::read_to_string(config_path)?;
    let cfg: FlowMatchScheduleConfig = serde_json::from_str(&config)?;
    Ok(cfg)
}

fn load_flux2_chat_template(api: &ApiRepo) -> anyhow::Result<Option<ChatTemplateValue>> {
    let template_path = api.get("tokenizer/chat_template.jinja");
    if let Ok(path) = template_path {
        let template = std::fs::read_to_string(path)?;
        return Ok(Some(ChatTemplateValue(Either::Left(template))));
    }

    let config_path = api.get("tokenizer_config.json");
    if let Ok(path) = config_path {
        let config = std::fs::read_to_string(path)?;
        let template: ChatTemplate = serde_json::from_str(&config)?;
        return Ok(template.chat_template);
    }

    Ok(None)
}

impl FluxStepper {
    fn preprocess_flux2_image(&self, image: &DynamicImage) -> Result<Tensor> {
        let mut image = image.to_rgb8();
        let (w, h) = image.dimensions();
        let new_w = (w / 16).max(1) * 16;
        let new_h = (h / 16).max(1) * 16;
        if new_w != w || new_h != h {
            image = image::imageops::resize(&image, new_w, new_h, FilterType::Lanczos3);
        }
        let data = image.into_raw();
        let mut data_f = Vec::with_capacity(data.len());
        for v in data {
            data_f.push(v as f32 / 255.0);
        }
        let t = Tensor::from_vec(data_f, (new_h as usize, new_w as usize, 3), self.device())?;
        let t = t.permute((2, 0, 1))?;
        let t = ((t * 2.0)? - 1.0)?;
        t.unsqueeze(0)?.to_dtype(self.dtype)
    }

    fn build_flux2_image_conditioning(
        &self,
        images: &[DynamicImage],
        batch_size: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !self.is_flux2 || images.is_empty() {
            return Ok(None);
        }

        let mut packed_latents = Vec::with_capacity(images.len());
        let mut img_ids = Vec::with_capacity(images.len());
        let dev = self.device();
        let scale = 10u32;

        for (idx, image) in images.iter().enumerate() {
            let image_tensor = self.preprocess_flux2_image(image)?;
            let latents = self.flux_vae.encode(&image_tensor)?;
            let patched = flux::sampling::patchify_latents(&latents)?;
            let normalized = self.flux_vae.normalize_packed(&patched)?;
            let packed = flux::sampling::pack_latents(&normalized)?;
            packed_latents.push(packed.squeeze(0)?);

            let (_, _, h, w) = normalized.dims4()?;
            let t_ids = Tensor::full(scale + scale * idx as u32, (h, w), dev)?;
            let h_ids = Tensor::arange(0u32, h as u32, dev)?
                .reshape(((), 1))?
                .broadcast_as((h, w))?;
            let w_ids = Tensor::arange(0u32, w as u32, dev)?
                .reshape((1, ()))?
                .broadcast_as((h, w))?;
            let l_ids = Tensor::full(0u32, (h, w), dev)?;
            let ids = Tensor::stack(&[t_ids, h_ids, w_ids, l_ids], 2)?
                .reshape((1, h * w, 4))?
                .to_dtype(self.dtype)?; // Match state.img_ids dtype
            img_ids.push(ids);
        }

        let packed = Tensor::cat(&packed_latents, 0)?.unsqueeze(0)?;
        let packed = packed.repeat((batch_size, 1, 1))?;
        let ids = Tensor::cat(&img_ids, 1)?.repeat((batch_size, 1, 1))?;
        Ok(Some((packed, ids)))
    }
    pub fn new(
        cfg: FluxStepperConfig,
        (flux_vb, flux_cfg): (ShardedVarBuilder, &flux::model::Config),
        (flux_ae_vb, flux_ae_cfg): (ShardedVarBuilder, &flux::autoencoder::Config),
        model_id: &str,
        dtype: DType,
        device: &Device,
        silent: bool,
        offloaded: bool,
    ) -> anyhow::Result<Self> {
        let api = Api::new()?;

        let latent_channels = flux_ae_cfg.latent_channels;

        // Detect FLUX.2 by latent_channels (32) or attention_head_dim in config
        let is_flux2 = latent_channels == 32 || flux_cfg.is_flux2();
        let rope_axes = flux_cfg.axes_dims_rope.len();

        // Load text encoder and tokenizer based on model type
        let (
            text_tokenizer,
            qwen3_encoder,
            clip_encoder,
            clip_tokenizer,
            chat_template,
            flux2_schedule_cfg,
        ) = if is_flux2 {
            info!("FLUX.2 detected - loading Qwen3 text encoder");
            let model_repo = api.repo(hf_hub::Repo::model(model_id.to_string()));
            let (encoder, mut tokenizer, template) =
                get_qwen3_encoder_and_tokenizer(&model_repo, dtype, device, silent)?;
            // Configure padding/truncation once rather than cloning per forward pass
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(512),
                ..Default::default()
            }));
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 512,
                    strategy: TruncationStrategy::LongestFirst,
                    ..Default::default()
                }))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let sched_cfg = load_flux2_scheduler_config(&model_repo).unwrap_or_default();
            (tokenizer, Some(encoder), None, None, template, sched_cfg)
        } else {
            info!("Loading T5 XXL tokenizer.");
            let t5_tokenizer = get_t5_tokenizer(&api)?;
            info!("Loading CLIP model and tokenizer.");
            let (enc, tok) = get_clip_model_and_tokenizer(&api, device, silent)?;
            (
                t5_tokenizer,
                None,
                Some(enc),
                Some(tok),
                None,
                FlowMatchScheduleConfig::default(),
            )
        };

        // Load VAE
        let flux_vae = if is_flux2 {
            info!("Using FLUX.2 VAE (AutoEncoderKL with latent_channels=32)");
            let flux2_cfg = flux::autoencoder_kl::Config::from(flux_ae_cfg);
            FluxVae::Flux2(AutoEncoderKL::new(&flux2_cfg, flux_ae_vb)?)
        } else {
            info!("Using FLUX.1 VAE (latent_channels=16)");
            FluxVae::Flux1(AutoEncoder::new(flux_ae_cfg, flux_ae_vb)?)
        };

        // Load transformer model
        let flux_model = if is_flux2 {
            info!("Loading FLUX.2 Transformer (Flux2Transformer2DModel)");
            FluxTransformer::Flux2(Flux2::new(flux_cfg, flux_vb, device.clone(), dtype, offloaded)?)
        } else {
            info!("Loading FLUX.1 Transformer");
            FluxTransformer::Flux1(Flux::new(flux_cfg, flux_vb, device.clone(), offloaded)?)
        };

        Ok(Self {
            cfg,
            text_tok: text_tokenizer,
            clip_tok: clip_tokenizer,
            clip_text: clip_encoder,
            qwen3_encoder,
            flux2_chat_template: chat_template,
            flux2_max_seq_len: 512,
            flux_model,
            flux_vae,
            is_guidance: cfg.is_guidance,
            is_flux2,
            rope_axes,
            flux2_schedule_cfg,
            device: device.clone(),
            dtype,
            preview_sender: None,
            api,
            silent,
            offloaded,
            latent_channels,
        })
    }
}

impl DiffusionModel for FluxStepper {
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
        images: Option<Vec<DynamicImage>>,
    ) -> Result<Tensor> {
        if params.height % 16 != 0 || params.width % 16 != 0 {
            candle_core::bail!(
                "height ({}) and width ({}) must be multiples of 16",
                params.height,
                params.width
            );
        }
        let preview_interval = params.preview_interval.unwrap_or(0);
        let prompt_len = prompts.len();
        // Get text embeddings based on model type
        let text_embed = if self.is_flux2 {
            // FLUX.2: Use Qwen3 text encoder
            let Some(qwen3_encoder) = self.qwen3_encoder.as_ref() else {
                candle_core::bail!("Qwen3 encoder required for FLUX.2 but not loaded");
            };
            let (input_ids, attention_mask) = get_qwen3_tokenization_with_mask(
                &self.text_tok,
                prompts.clone(),
                &self.device,
                self.flux2_chat_template.as_ref(),
            )?;
            info!("Running Qwen3 text encoder");
            qwen3_encoder
                .forward(&input_ids, &attention_mask)?
                .to_dtype(self.dtype)?
        } else {
            // FLUX.1: Use T5 (hotloaded)
            let mut t5_input_ids = get_tokenization(&self.text_tok, prompts.clone(), &self.device)?;
            if !self.is_guidance {
                match t5_input_ids.dim(1)?.cmp(&256) {
                    Ordering::Greater => {
                        candle_core::bail!("T5 embedding length greater than 256, please shrink the prompt or use the -dev (with guidance distillation) version.")
                    }
                    Ordering::Less | Ordering::Equal => {
                        t5_input_ids = t5_input_ids.pad_with_zeros(
                            D::Minus1,
                            0,
                            256 - t5_input_ids.dim(1)?,
                        )?;
                    }
                }
            }
            info!("Hotloading T5 XXL model.");
            let mut t5_encoder = get_t5_model(
                &self.api,
                self.dtype,
                &self.device,
                self.silent,
                self.offloaded,
            )?;
            t5_encoder.forward(&t5_input_ids)?
        };

        let img = flux::sampling::get_noise(
            text_embed.dim(0)?,
            params.height,
            params.width,
            self.latent_channels,
            self.device(),
        )?
        .to_dtype(self.dtype)?;

        // Create state - FLUX.2 doesn't use CLIP embeddings
        let state = if self.is_flux2 {
            info!("Using FLUX.2 state (Qwen3 embeddings, no CLIP)");
            flux::sampling::State::new_flux2(&text_embed, &img, self.rope_axes)?
        } else {
            // FLUX.1 requires CLIP embeddings
            let clip_tok = self
                .clip_tok
                .as_ref()
                .expect("CLIP tokenizer required for FLUX.1");
            let clip_text = self
                .clip_text
                .as_ref()
                .expect("CLIP model required for FLUX.1");
            let clip_input_ids = get_tokenization(clip_tok, prompts.clone(), &self.device)?;
            let clip_embed = clip_text.forward(&clip_input_ids)?.to_dtype(self.dtype)?;
            flux::sampling::State::new(&text_embed, &clip_embed, &img, self.rope_axes)?
        };

        // Use per-request params if provided, otherwise fall back to model defaults.
        // FLUX.2 needs 20-28 steps (28 default); schnell default of 4 is far too few.
        let default_steps = if self.is_flux2 { 28 } else { self.cfg.num_steps };
        let num_steps = params.num_steps.unwrap_or(default_steps);
        let timesteps = if self.is_flux2 {
            flux::sampling::get_schedule_flux2(
                num_steps,
                state.img.dims()[1],
                &self.flux2_schedule_cfg,
            )
        } else {
            flux::sampling::get_schedule(
                num_steps,
                self.cfg
                    .guidance_config
                    .map(|s| (state.img.dims()[1], s.base_shift, s.max_shift)),
            )
        };

        let image_cond = self.build_flux2_image_conditioning(
            images.as_deref().unwrap_or_default(),
            state.img.dim(0)?,
        )?;

        let img = if self.is_flux2 {
            let cfg_scale = params.guidance_scale.unwrap_or(0.0);
            let do_cfg = cfg_scale > 0.0;
            let neg_text = if do_cfg {
                let neg_prompt = params.negative_prompt.clone().unwrap_or_default();
                let neg_prompts = vec![neg_prompt; prompt_len];
                let (neg_ids, neg_mask) = get_qwen3_tokenization_with_mask(
                    &self.text_tok,
                    neg_prompts,
                    &self.device,
                    self.flux2_chat_template.as_ref(),
                )?;
                let Some(enc) = self.qwen3_encoder.as_ref() else {
                    candle_core::bail!("Qwen3 encoder required for FLUX.2 but not loaded");
                };
                Some(enc.forward(&neg_ids, &neg_mask)?.to_dtype(self.dtype)?)
            } else {
                None
            };

            let mut img = state.img.clone();
            let b_sz = img.dim(0)?;
            let dev = self.device.clone();
            let total_steps = timesteps.len().saturating_sub(1);

            // Precompute PE once (img_ids/txt_ids are static across steps)
            let pe = if let Some((_, cond_ids)) = &image_cond {
                let combined_ids = Tensor::cat(&[&state.img_ids, cond_ids], 1)?;
                self.flux_model.compute_pe(&state.txt_ids, &combined_ids)?
            } else {
                self.flux_model.compute_pe(&state.txt_ids, &state.img_ids)?
            };

            // Precompute all timestep vectors
            let t_vecs: Vec<Tensor> = timesteps
                .iter()
                .map(|t| Tensor::full(*t as f32, b_sz, &dev))
                .collect::<Result<Vec<_>>>()?;

            for (step_idx, window) in timesteps.windows(2).enumerate() {
                let (t_curr, t_prev) = match window {
                    [a, b] => (a, b),
                    _ => continue,
                };
                let t_vec = &t_vecs[step_idx];
                let (img_in, seq_len) = if let Some((cond_img, _)) = &image_cond {
                    let img_in = Tensor::cat(&[&img, cond_img], 1)?;
                    (img_in, img.dim(1)?)
                } else {
                    (img.clone(), img.dim(1)?)
                };

                let pred = self.flux_model.forward(
                    &img_in,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    t_vec,
                    &state.vec,
                    None,
                    Some(&pe),
                )?;
                let mut pred = if let Some(neg_txt) = &neg_text {
                    let neg_pred = self.flux_model.forward(
                        &img_in,
                        &state.img_ids,
                        neg_txt,
                        &state.txt_ids,
                        t_vec,
                        &state.vec,
                        None,
                        Some(&pe),
                    )?;
                    (&neg_pred + ((&pred - &neg_pred)? * cfg_scale)?)?
                } else {
                    pred
                };
                if image_cond.is_some() {
                    pred = pred.narrow(1, 0, seq_len)?;
                }
                img = (img + pred * (t_prev - t_curr))?;

                if preview_interval > 0
                    && step_idx + 1 < total_steps
                    && (step_idx + 1) % preview_interval == 0
                {
                    if let Err(err) = self.send_preview(&img, &params) {
                        warn!(error = %err, "Failed to send image preview");
                    }
                }
            }
            img
        } else if let Some(guidance_cfg) = &self.cfg.guidance_config {
            let guidance_scale = params.guidance_scale.unwrap_or(guidance_cfg.guidance_scale);
            flux::sampling::denoise(
                &mut self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
                guidance_scale,
            )?
        } else {
            flux::sampling::denoise_no_guidance(
                &mut self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
            )?
        };

        let img = if self.is_flux2 {
            let packed = flux::sampling::unpack_packed(&img, params.height, params.width)?;
            let packed = self.flux_vae.denormalize_packed(&packed)?;
            let latent_img = flux::sampling::unpatchify_packed(&packed)?;
            self.flux_vae.decode(&latent_img)?
        } else {
            let latent_img = flux::sampling::unpack(&img, params.height, params.width)?;
            self.flux_vae.decode(&latent_img)?
        };

        let normalized_img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;

        Ok(normalized_img)
    }

    fn set_preview_sender(&mut self, sender: Option<UnboundedSender<Vec<DynamicImage>>>) {
        self.preview_sender = sender;
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_seq_len(&self) -> usize {
        if self.is_guidance {
            usize::MAX
        } else if self.is_flux2 {
            self.flux2_max_seq_len
        } else {
            256
        }
    }
}

impl FluxStepper {
    fn decode_latents_to_tensor(
        &self,
        latents: &Tensor,
        params: &DiffusionGenerationParams,
    ) -> Result<Tensor> {
        let img = if self.is_flux2 {
            let packed = flux::sampling::unpack_packed(latents, params.height, params.width)?;
            let packed = self.flux_vae.denormalize_packed(&packed)?;
            let latent_img = flux::sampling::unpatchify_packed(&packed)?;
            self.flux_vae.decode(&latent_img)?
        } else {
            let latent_img = flux::sampling::unpack(latents, params.height, params.width)?;
            self.flux_vae.decode(&latent_img)?
        };

        ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)
    }

    fn tensor_to_images(&self, img: &Tensor) -> Result<Vec<DynamicImage>> {
        let (_b, c, h, w) = img.dims4()?;
        if c != 3 {
            candle_core::bail!("Expected 3 channels in image output");
        }
        let mut images = Vec::new();
        for b_img in img.chunk(img.dim(0)?, 0)? {
            let flattened = b_img.squeeze(0)?.permute((1, 2, 0))?.flatten_all()?;
            images.push(DynamicImage::ImageRgb8(
                RgbImage::from_raw(w as u32, h as u32, flattened.to_vec1::<u8>()?).ok_or(
                    candle_core::Error::Msg("RgbImage has invalid capacity.".to_string()),
                )?,
            ));
        }
        Ok(images)
    }

    fn send_preview(&self, latents: &Tensor, params: &DiffusionGenerationParams) -> Result<()> {
        let Some(sender) = &self.preview_sender else {
            return Ok(());
        };
        let preview = self.decode_latents_to_tensor(latents, params)?;
        let images = self.tensor_to_images(&preview)?;
        let _ = sender.send(images);
        Ok(())
    }
}
