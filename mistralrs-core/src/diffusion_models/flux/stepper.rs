use std::fs::File;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::diffusion_models::{
    clip::text::{Activation, ClipConfig, ClipTextTransformer},
    flux,
    t5::{self, T5EncoderModel},
};

use super::{autoencoder::AutoEncoder, model::Flux};

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperShift {
    base_shift: f64,
    max_shift: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperConfig {
    height: usize,
    width: usize,
    num_samples: usize,
    num_steps: usize,
    shift: Option<FluxStepperShift>,
    guidance_scale: f64,
}

pub struct FluxStepper {
    cfg: FluxStepperConfig,
    t5_tok: Tokenizer,
    t5: T5EncoderModel,
    clip_tok: Tokenizer,
    clip_text: ClipTextTransformer,
    flux_model: Flux,
    flux_vae: AutoEncoder,
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
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
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

        let (t5_encoder, t5_tokenizer) = get_t5_model_and_tokenizr(&api, dtype, device)?;
        let (clip_encoder, clip_tokenizer) = get_clip_model_and_tokenizer(&api, dtype, device)?;

        Ok(Self {
            cfg,
            t5_tok: t5_tokenizer,
            t5: t5_encoder,
            clip_tok: clip_tokenizer,
            clip_text: clip_encoder,
            flux_model: Flux::new(&flux_cfg, flux_vb)?,
            flux_vae: AutoEncoder::new(&flux_ae_cfg, flux_ae_vb)?,
        })
    }

    pub fn forward(&self, prompts: Vec<String>, device: &Device) -> Result<Tensor> {
        let t5_input_ids = get_tokenization(&self.t5_tok, prompts.clone(), device)?;
        let t5_embed = self.t5.forward(&t5_input_ids)?;

        let clip_input_ids = get_tokenization(&self.t5_tok, prompts, device)?;
        let clip_embed = self.clip_text.forward(&clip_input_ids)?;

        let img = flux::sampling::get_noise(1, self.cfg.height, self.cfg.width, device)?;
        let state = flux::sampling::State::new(&t5_embed, &clip_embed, &img)?;
        let timesteps = flux::sampling::get_schedule(
            self.cfg.num_steps,
            self.cfg
                .shift
                .map(|s| (state.img.dims()[1], s.base_shift, s.max_shift)),
        );

        let img = flux::sampling::denoise(
            &self.flux_model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            self.cfg.guidance_scale,
        )?;
        let latent_img = flux::sampling::unpack(&img, self.cfg.height, self.cfg.width)?;

        let img = self.flux_vae.decode(&latent_img)?;

        let normalized_img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;

        Ok(normalized_img)
    }
}
