use std::{cmp::Ordering, fs::File, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Module;
use hf_hub::api::sync::{Api, ApiError};
use mistralrs_quant::ShardedVarBuilder;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    diffusion_models::{
        clip::text::{ClipConfig, ClipTextTransformer},
        flux,
        t5::{self, T5EncoderModel},
        DiffusionGenerationParams,
    },
    pipeline::DiffusionModel,
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
};

use super::{autoencoder::AutoEncoder, model::Flux};

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
    t5_tok: Tokenizer,
    clip_tok: Tokenizer,
    clip_text: ClipTextTransformer,
    clip_max_seq_len: usize,
    flux_model: Flux,
    flux_vae: AutoEncoder,
    is_guidance: bool,
    device: Device,
    dtype: DType,
    api: Api,
    silent: bool,
    offloaded: bool,
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
) -> anyhow::Result<(ClipTextTransformer, Tokenizer, usize)> {
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
    let max_position_embeddings = config.max_position_embeddings;
    let model = ClipTextTransformer::new(vb.pp("text_model"), &config)?;

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    // CLIP has a hard limit of `max_position_embeddings` (typically 77) tokens.
    // Ensure the tokenizer truncates to that length so prompts that exceed it
    // don't cause a narrow-out-of-bounds panic in ClipTextEmbeddings::forward.
    let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
        max_length: max_position_embeddings,
        ..Default::default()
    }));

    Ok((model, tokenizer, max_position_embeddings))
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

impl FluxStepper {
    pub fn new(
        cfg: FluxStepperConfig,
        (flux_vb, flux_cfg): (ShardedVarBuilder, &flux::model::Config),
        (flux_ae_vb, flux_ae_cfg): (ShardedVarBuilder, &flux::autoencoder::Config),
        dtype: DType,
        device: &Device,
        silent: bool,
        offloaded: bool,
    ) -> anyhow::Result<Self> {
        let api = Api::new()?;

        info!("Loading T5 XXL tokenizer.");
        let t5_tokenizer = get_t5_tokenizer(&api)?;
        info!("Loading CLIP model and tokenizer.");
        let (clip_encoder, clip_tokenizer, clip_max_seq_len) =
            get_clip_model_and_tokenizer(&api, device, silent)?;

        Ok(Self {
            cfg,
            t5_tok: t5_tokenizer,
            clip_tok: clip_tokenizer,
            clip_text: clip_encoder,
            clip_max_seq_len,
            flux_model: Flux::new(flux_cfg, flux_vb, device.clone(), offloaded)?,
            flux_vae: AutoEncoder::new(flux_ae_cfg, flux_ae_vb)?,
            is_guidance: cfg.is_guidance,
            device: device.clone(),
            dtype,
            api,
            silent,
            offloaded,
        })
    }
}

impl DiffusionModel for FluxStepper {
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> Result<Tensor> {
        let mut t5_input_ids = get_tokenization(&self.t5_tok, prompts.clone(), &self.device)?;
        if !self.is_guidance {
            match t5_input_ids.dim(1)?.cmp(&256) {
                Ordering::Greater => {
                    candle_core::bail!("T5 embedding length greater than 256, please shrink the prompt or use the -dev (with guidance distillation) version.")
                }
                Ordering::Less | Ordering::Equal => {
                    t5_input_ids =
                        t5_input_ids.pad_with_zeros(D::Minus1, 0, 256 - t5_input_ids.dim(1)?)?;
                }
            }
        }

        let t5_embed = {
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

        let mut clip_input_ids = get_tokenization(&self.clip_tok, prompts, &self.device)?;

        // Safety truncation: CLIP's position embeddings are fixed at
        // `max_position_embeddings` (77).  If the tokenizer didn't truncate
        // (e.g. missing truncation config), clamp here to avoid a
        // narrow-out-of-bounds panic in ClipTextEmbeddings::forward.
        if clip_input_ids.dim(1)? > self.clip_max_seq_len {
            info!(
                "CLIP input length {} exceeds max_position_embeddings ({}), truncating.",
                clip_input_ids.dim(1)?,
                self.clip_max_seq_len
            );
            clip_input_ids = clip_input_ids.narrow(1, 0, self.clip_max_seq_len)?;
        }

        let clip_embed = self
            .clip_text
            .forward(&clip_input_ids)?
            .to_dtype(self.dtype)?;

        let img = flux::sampling::get_noise(
            t5_embed.dim(0)?,
            params.height,
            params.width,
            self.device(),
        )?
        .to_dtype(self.dtype)?;

        let state = flux::sampling::State::new(&t5_embed, &clip_embed, &img)?;
        let timesteps = flux::sampling::get_schedule(
            self.cfg.num_steps,
            self.cfg
                .guidance_config
                .map(|s| (state.img.dims()[1], s.base_shift, s.max_shift)),
        );

        let img = if let Some(guidance_cfg) = &self.cfg.guidance_config {
            flux::sampling::denoise(
                &mut self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
                guidance_cfg.guidance_scale,
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

        let latent_img = flux::sampling::unpack(&img, params.height, params.width)?;

        let img = self.flux_vae.decode(&latent_img)?;

        let normalized_img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;

        Ok(normalized_img)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_seq_len(&self) -> usize {
        if self.is_guidance {
            usize::MAX
        } else {
            256
        }
    }
}
