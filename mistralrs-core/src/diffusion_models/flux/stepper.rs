use std::{cmp::Ordering, fs::File};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    diffusion_models::{
        clip::text::{ClipConfig, ClipTextTransformer},
        flux,
        t5::{self, T5EncoderModel},
    },
    pipeline::DiffusionModel,
};

use super::{autoencoder::AutoEncoder, model::Flux};

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperShift {
    pub base_shift: f64,
    pub max_shift: f64,
    pub guidance_scale: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperConfig {
    pub height: usize,
    pub width: usize,
    pub num_steps: usize,
    pub guidance_config: Option<FluxStepperShift>,
    pub is_guidance: bool,
}

impl FluxStepperConfig {
    pub fn default_for_guidance(has_guidance: bool) -> Self {
        if has_guidance {
            Self {
                height: 1380,
                width: 768,
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
                height: 1380,
                width: 768,
                num_steps: 4,
                guidance_config: None,
                is_guidance: true,
            }
        }
    }
}

pub struct FluxStepper {
    cfg: FluxStepperConfig,
    t5_tok: Tokenizer,
    t5: T5EncoderModel,
    clip_tok: Tokenizer,
    clip_text: ClipTextTransformer,
    flux_model: Flux,
    flux_vae: AutoEncoder,
    is_guidance: bool,
    device: Device,
}

fn get_t5_model_and_tokenizr(
    api: &Api,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<(T5EncoderModel, Tokenizer)> {
    let repo = api.repo(hf_hub::Repo::with_revision(
        "google/t5-v1_1-xxl".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/2".to_string(),
    ));

    let model_file = repo.get("model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)? };
    let config_filename = repo.get("config.json")?;
    let config = std::fs::read_to_string(config_filename)?;
    let config: t5::Config = serde_json::from_str(&config)?;
    let model = t5::T5EncoderModel::load(vb, &config)?;

    let tokenizer_filename = api
        .model("EricB/t5_tokenizer".to_string())
        .get("t5-v1_1-xxl.tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    Ok((model, tokenizer))
}

fn get_clip_model_and_tokenizer(
    api: &Api,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<(ClipTextTransformer, Tokenizer)> {
    let repo = api.repo(hf_hub::Repo::model(
        "openai/clip-vit-large-patch14".to_string(),
    ));

    let model_file = repo.get("model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)? };
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

impl FluxStepper {
    pub fn new(
        cfg: FluxStepperConfig,
        flux_vb: VarBuilder,
        flux_cfg: &flux::model::Config,
        flux_ae_vb: VarBuilder,
        flux_ae_cfg: &flux::autoencoder::Config,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let api = Api::new()?;

        info!("Loading T5 XXL model and tokenizer.");
        let (t5_encoder, t5_tokenizer) = get_t5_model_and_tokenizr(&api, dtype, &Device::Cpu)?;
        info!("Loading CLIP model and tokenizer.");
        let (clip_encoder, clip_tokenizer) =
            get_clip_model_and_tokenizer(&api, dtype, &Device::Cpu)?;

        Ok(Self {
            cfg,
            t5_tok: t5_tokenizer,
            t5: t5_encoder,
            clip_tok: clip_tokenizer,
            clip_text: clip_encoder,
            flux_model: Flux::new(flux_cfg, flux_vb)?,
            flux_vae: AutoEncoder::new(flux_ae_cfg, flux_ae_vb)?,
            is_guidance: cfg.is_guidance,
            device: device.clone(),
        })
    }
}

impl DiffusionModel for FluxStepper {
    fn forward(&self, prompts: Vec<String>) -> Result<Tensor> {
        let mut t5_input_ids = get_tokenization(&self.t5_tok, prompts.clone(), &Device::Cpu)?;
        if !self.is_guidance {
            match t5_input_ids.dim(1)?.cmp(&256) {
                Ordering::Greater => {
                    candle_core::bail!("T5 embedding length greater than 256, please shrink the prompt or use the -dev (with guidance distillation) version.")
                }
                Ordering::Less => {
                    t5_input_ids =
                        t5_input_ids.pad_with_zeros(D::Minus1, 0, 256 - t5_input_ids.dim(1)?)?;
                }
                Ordering::Equal => (),
            }
        }

        let t5_embed = self.t5.forward(&t5_input_ids)?.to_device(&self.device)?;

        let clip_input_ids = get_tokenization(&self.clip_tok, prompts, &Device::Cpu)?;
        let clip_embed = self
            .clip_text
            .forward(&clip_input_ids)?
            .to_device(&self.device)?;

        let img = flux::sampling::get_noise(
            t5_embed.dim(0)?,
            self.cfg.height,
            self.cfg.width,
            self.device(),
        )?;
        let state = flux::sampling::State::new(&t5_embed, &clip_embed, &img)?;
        let timesteps = flux::sampling::get_schedule(
            self.cfg.num_steps,
            self.cfg
                .guidance_config
                .map(|s| (state.img.dims()[1], s.base_shift, s.max_shift)),
        );

        let img = if let Some(guidance_cfg) = &self.cfg.guidance_config {
            flux::sampling::denoise(
                &self.flux_model,
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
                &self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
            )?
        };
        let latent_img = flux::sampling::unpack(&img, self.cfg.height, self.cfg.width)?;

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
