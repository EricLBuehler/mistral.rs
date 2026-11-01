use std::sync::Arc;

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;
use rubato::Resampler;
use rustfft::{num_complex::Complex32, FftPlanner};

pub(crate) const SAMPLE_RATE: u32 = 16_000;
pub(crate) const MEL_BINS: usize = 128;
pub(crate) const ENCODER_STRIDE: usize = 8;
const FFT_SIZE: usize = 400;
const HOP_LENGTH: usize = 160;
const LOG_FLOOR: f32 = -12.0;
const NORMALIZATION_OFFSET: f32 = 4.0;
const NORMALIZATION_DIVISOR: f32 = 4.0;
const SLANEY_LINEAR_STEP: f64 = 200.0 / 3.0;
const SLANEY_LOG_START_HZ: f64 = 1000.0;
const SLANEY_LOG_START_MEL: f64 = 15.0;
const SLANEY_LOG_RANGE: f64 = 6.4;
const SLANEY_LOG_STEPS: f64 = 27.0;
const RESAMPLE_SINC_LENGTH: usize = 256;
const RESAMPLE_OVERSAMPLING: usize = 256;
const RESAMPLE_CUTOFF: f32 = 0.95;
const MAX_RESAMPLE_RATIO: f64 = 2.0;

#[derive(Clone)]
pub(crate) struct AudioProcessor {
    window: Arc<[f32]>,
    filters: Arc<[Vec<f32>]>,
}

impl AudioProcessor {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub(crate) fn new() -> Self {
        let max_mel = SLANEY_LOG_START_MEL
            + (f64::from(SAMPLE_RATE) / 2.0 / SLANEY_LOG_START_HZ).ln()
                / (SLANEY_LOG_RANGE.ln() / SLANEY_LOG_STEPS);
        let frequencies = (0..MEL_BINS + 2)
            .map(|index| {
                let mel = index as f64 * max_mel / (MEL_BINS + 1) as f64;
                if mel < SLANEY_LOG_START_MEL {
                    mel * SLANEY_LINEAR_STEP
                } else {
                    SLANEY_LOG_START_HZ
                        * ((mel - SLANEY_LOG_START_MEL)
                            * (SLANEY_LOG_RANGE.ln() / SLANEY_LOG_STEPS))
                            .exp()
                }
            })
            .collect::<Vec<_>>();
        let filters = frequencies
            .windows(3)
            .map(|band| {
                (0..FFT_SIZE / 2 + 1)
                    .map(|bin| {
                        let hz = bin as f64 * f64::from(SAMPLE_RATE) / FFT_SIZE as f64;
                        let rising = (hz - band[0]) / (band[1] - band[0]);
                        let falling = (band[2] - hz) / (band[2] - band[1]);
                        (rising.min(falling).max(0.0) * 2.0 / (band[2] - band[0])) as f32
                    })
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let window = (0..FFT_SIZE)
            .map(|index| {
                0.5 * (1.0 - (std::f32::consts::TAU * index as f32 / FFT_SIZE as f32).cos())
            })
            .collect::<Vec<_>>();
        Self {
            window: window.into(),
            filters: filters.into(),
        }
    }

    pub(crate) fn num_encoder_tokens(num_frames: usize) -> usize {
        num_frames.div_ceil(ENCODER_STRIDE)
    }

    pub(crate) fn num_tokens_for_audio(audio: &AudioInput) -> Result<usize> {
        anyhow::ensure!(audio.sample_rate > 0, "Audio sample rate must be positive");
        let samples = audio.samples.len() / usize::from(audio.channels.max(1));
        let samples = samples * SAMPLE_RATE as usize / audio.sample_rate as usize;
        let frames = samples.saturating_sub(FFT_SIZE).div_ceil(HOP_LENGTH) + 1;
        Ok(Self::num_encoder_tokens(frames))
    }

    pub(crate) fn process_audio(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let mut samples = audio.to_mono();
        if audio.sample_rate != SAMPLE_RATE && !samples.is_empty() {
            samples = resample(&samples, audio.sample_rate)?;
        }
        self.process_samples(&samples, device)
    }

    fn process_samples(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        let frames = samples.len().saturating_sub(FFT_SIZE).div_ceil(HOP_LENGTH) + 1;
        let mut padded = vec![0.0; FFT_SIZE + (frames - 1) * HOP_LENGTH];
        padded[..samples.len()].copy_from_slice(samples);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        let mut spectrum = vec![Complex32::default(); FFT_SIZE];
        let mut features = vec![0.0; MEL_BINS * frames];
        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            for (index, value) in spectrum.iter_mut().enumerate() {
                *value = Complex32::new(padded[start + index] * self.window[index], 0.0);
            }
            fft.process_with_scratch(&mut spectrum, &mut scratch);
            for (mel, filter) in self.filters.iter().enumerate() {
                let power = filter
                    .iter()
                    .zip(&spectrum)
                    .map(|(&weight, value)| weight * value.norm_sqr())
                    .sum::<f32>();
                features[mel * frames + frame] =
                    (power.log10().max(LOG_FLOOR) + NORMALIZATION_OFFSET) / NORMALIZATION_DIVISOR;
            }
        }
        Ok(Tensor::from_vec(features, (1, MEL_BINS, frames), device)?)
    }
}

fn resample(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    anyhow::ensure!(source_rate > 0, "Audio sample rate must be positive");
    let expected_length = samples.len() * SAMPLE_RATE as usize / source_rate as usize;
    let parameters = rubato::SincInterpolationParameters {
        sinc_len: RESAMPLE_SINC_LENGTH,
        f_cutoff: RESAMPLE_CUTOFF,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: RESAMPLE_OVERSAMPLING,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    let mut resampler = rubato::SincFixedIn::<f32>::new(
        f64::from(SAMPLE_RATE) / f64::from(source_rate),
        MAX_RESAMPLE_RATIO,
        parameters,
        samples.len(),
        1,
    )?;
    let delay = resampler.output_delay();
    let mut output = resampler.process(&[samples.to_vec()], None)?.remove(0);
    while output.len() < expected_length + delay {
        output.extend(resampler.process_partial::<Vec<f32>>(None, None)?.remove(0));
    }
    Ok(output[delay..delay + expected_length].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_mel_matches_torch_frontend() -> Result<()> {
        let samples = (0u16..960)
            .map(|index| (f32::from(index % 17) - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let features = AudioProcessor::new().process_samples(&samples, &Device::Cpu)?;
        let actual = features.squeeze(0)?.to_vec2::<f32>()?;
        // Generated with the HF Slaney filter bank and torch.stft(center=False).
        for (mel, frame, expected) in [
            (0, 0, -0.718_921_4),
            (1, 0, -0.621_356_96),
            (8, 0, -0.591_097_5),
            (32, 0, 0.276_995_72),
            (64, 1, -0.177_094_58),
            (100, 2, -0.553_293_94),
            (127, 4, 0.302_608_43),
        ] {
            assert!(
                (actual[mel][frame] - expected).abs() < 2e-4,
                "mel={mel}, frame={frame}: {} != {expected}",
                actual[mel][frame]
            );
        }
        Ok(())
    }

    #[test]
    fn planning_matches_resampling_and_downmix() -> Result<()> {
        let processor = AudioProcessor::new();
        for sample_rate in [8_000, 16_000, 48_000] {
            let audio = AudioInput {
                samples: vec![0.0; 2 * 1441],
                sample_rate,
                channels: 2,
            };
            let features = processor.process_audio(&audio, &Device::Cpu)?;
            assert_eq!(
                AudioProcessor::num_tokens_for_audio(&audio)?,
                AudioProcessor::num_encoder_tokens(features.dim(2)?)
            );
        }
        Ok(())
    }

    #[test]
    fn silence_and_frame_boundaries() -> Result<()> {
        let processor = AudioProcessor::new();
        for (samples, expected_frames) in [(0, 1), (399, 1), (400, 1), (401, 2), (560, 2), (561, 3)]
        {
            let features = processor.process_samples(&vec![0.0; samples], &Device::Cpu)?;
            assert_eq!(features.dims(), &[1, MEL_BINS, expected_frames]);
            assert!(features
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&x| x == -2.0));
        }
        assert_eq!(AudioProcessor::num_encoder_tokens(8), 1);
        assert_eq!(AudioProcessor::num_encoder_tokens(9), 2);
        Ok(())
    }
}
