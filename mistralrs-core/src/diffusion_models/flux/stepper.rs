use candle_core::{DType, Result, Tensor};
use candle_nn::Module;

use crate::diffusion_models::{clip::text::ClipTextTransformer, flux, t5::T5EncoderModel};

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
    t5: T5EncoderModel,
    clip_text: ClipTextTransformer,
    flux_model: Flux,
    flux_vae: AutoEncoder,
}

impl FluxStepper {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let t5_embed = self.t5.forward(input_ids)?;
        let clip_embed = self.clip_text.forward(&input_ids)?;

        let device = input_ids.device();
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
