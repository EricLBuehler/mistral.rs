use audio::{apply_audio_delay, build_delay_indices};
use candle_core::{DType, Device, Result, Tensor};
use config::DiaConfig;
use mistralrs_quant::ShardedVarBuilder;
use model::DiaModel;

mod audio;
mod config;
mod model;

pub struct DiaPipeline {
    model: DiaModel,
    cfg: DiaConfig,
    device: Device,
}

impl DiaPipeline {
    pub fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            device: vb.device().clone(),
            model: DiaModel::new(cfg, vb)?,
            cfg: cfg.clone(),
        })
    }

    fn prepare_audio_prompt(&self) -> Result<(Tensor, usize)> {
        let num_channels = self.cfg.data.channels;
        let audio_pad_value = self.cfg.data.audio_pad_value;
        let audio_bos_value = self.cfg.data.audio_bos_value;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_delay_pattern = *delay_pattern.iter().max().unwrap() as usize;

        let prefill =
            (Tensor::ones((1, num_channels), DType::I32, &self.device)? * audio_bos_value as f64)?;
        let prefill_step = 1;

        let delay_pad_tensor =
            Tensor::ones((max_delay_pattern, num_channels), DType::I32, &self.device)?.neg()?;
        let prefill = Tensor::cat(&[prefill, delay_pad_tensor], 0)?;

        let delay_precomp = build_delay_indices(
            1,
            prefill.dim(0)?,
            num_channels,
            delay_pattern,
            &self.device,
        )?;

        let prefill = apply_audio_delay(
            &prefill.unsqueeze(0)?,
            audio_pad_value,
            audio_bos_value,
            &delay_precomp,
        )?;

        Ok((prefill, prefill_step))
    }

    pub fn generate(&self, text: &str) {
        let max_tokens: Option<usize> = None;

        let audio_pad_value = self.cfg.data.audio_pad_value;
        let audio_eos_value = self.cfg.data.audio_eos_value;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_tokens = max_tokens.unwrap_or(self.cfg.data.audio_length);
        let max_delay_pattern = delay_pattern.iter().max().unwrap();
    }
}
