use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;
use rubato::Resampler;
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::vision_models::preprocessor_config::PreProcessorConfig;

const DEFAULT_MAX_AUDIO_SAMPLES: usize = 480_000;
const DEFAULT_PAD_TO_MULTIPLE_OF: usize = 128;

pub struct AudioProcessor {
    target_sample_rate: u32,
    feature_size: usize,
    fft_length: usize,
    frame_length: usize,
    hop_length: usize,
    preemphasis: f32,
    preemphasis_htk_flavor: bool,
    mel_floor: f32,
    dither: f32,
    input_scale_factor: f32,
    padding_value: f32,
    per_bin_mean: Option<Vec<f64>>,
    per_bin_stddev: Option<Vec<f64>>,
    max_audio_samples: usize,
    pad_to_multiple_of: usize,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
}

impl AudioProcessor {
    pub fn new(config: &PreProcessorConfig) -> Self {
        let target_sample_rate = config.sampling_rate.unwrap_or(16_000) as u32;
        let frame_length = config.frame_length.unwrap_or(320);
        let hop_length = config.hop_length.unwrap_or(160);
        let fft_overdrive = config.fft_overdrive.unwrap_or(false);
        let fft_length = config.fft_length.unwrap_or_else(|| {
            let base_fft = frame_length.next_power_of_two();
            if fft_overdrive {
                base_fft * 2
            } else {
                base_fft
            }
        });
        let feature_size = config.feature_size.unwrap_or(128);
        let min_frequency = config.min_frequency.unwrap_or(0.0) as f32;
        let max_frequency = config
            .max_frequency
            .unwrap_or(f64::from(target_sample_rate) / 2.0) as f32;
        let arg = std::f32::consts::PI * 2.0 / frame_length as f32;
        let window = (0..frame_length)
            .map(|idx| 0.5 - 0.5 * (arg * idx as f32).cos())
            .collect::<Vec<_>>();
        let mel_filters = Self::create_mel_filterbank(
            feature_size,
            fft_length,
            target_sample_rate as f32,
            min_frequency,
            max_frequency,
        );

        Self {
            target_sample_rate,
            feature_size,
            fft_length,
            frame_length,
            hop_length,
            preemphasis: config.preemphasis.unwrap_or(0.0) as f32,
            preemphasis_htk_flavor: config.preemphasis_htk_flavor.unwrap_or(true),
            mel_floor: config.mel_floor.unwrap_or(1e-3) as f32,
            dither: config.dither.unwrap_or(0.0) as f32,
            input_scale_factor: config.input_scale_factor.unwrap_or(1.0) as f32,
            padding_value: config.padding_value.unwrap_or(0.0) as f32,
            per_bin_mean: config.per_bin_mean.clone(),
            per_bin_stddev: config.per_bin_stddev.clone(),
            max_audio_samples: DEFAULT_MAX_AUDIO_SAMPLES,
            pad_to_multiple_of: DEFAULT_PAD_TO_MULTIPLE_OF,
            window,
            mel_filters,
        }
    }

    pub fn process_audios(
        &self,
        audio_inputs: &[AudioInput],
        device: &Device,
    ) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let processed = audio_inputs
            .iter()
            .map(|audio| self.prepare_audio(audio))
            .collect::<Result<Vec<_>>>()?;

        let max_len = processed.iter().map(std::vec::Vec::len).max().unwrap_or(0);
        let padded_len = self.round_up_to_multiple(max_len, self.pad_to_multiple_of);

        let mut mel_batches = Vec::with_capacity(processed.len());
        let mut valid_masks = Vec::with_capacity(processed.len());
        let mut valid_frame_counts = Vec::with_capacity(processed.len());
        let mut max_frames = 0usize;

        for samples in processed {
            let mut padded_samples = vec![self.padding_value; padded_len];
            padded_samples[..samples.len()].copy_from_slice(&samples);

            let mut sample_mask = vec![false; padded_len];
            sample_mask[..samples.len()].fill(true);

            let (mel, valid_mask) = self.extract_spectrogram(&padded_samples, &sample_mask)?;
            max_frames = max_frames.max(mel.len());
            valid_frame_counts.push(valid_mask.iter().filter(|&&is_valid| is_valid).count());
            mel_batches.push(mel);
            valid_masks.push(valid_mask);
        }

        let batch_size = mel_batches.len();
        let mut mel_data = Vec::<f32>::with_capacity(batch_size * max_frames * self.feature_size);
        let mut mask_data = Vec::<f32>::with_capacity(batch_size * max_frames);

        for (mel, valid_mask) in mel_batches.into_iter().zip(valid_masks.into_iter()) {
            for frame in &mel {
                mel_data.extend_from_slice(frame);
            }
            let pad_frames = max_frames.saturating_sub(mel.len());
            mel_data.extend(std::iter::repeat_n(0.0, pad_frames * self.feature_size));

            mask_data.extend(
                valid_mask
                    .into_iter()
                    .map(|is_valid| if is_valid { 0.0f32 } else { 1.0f32 }),
            );
            mask_data.extend(std::iter::repeat_n(1.0f32, pad_frames));
        }

        let mel_tensor = Tensor::from_vec(
            mel_data,
            (batch_size, max_frames, self.feature_size),
            device,
        )?;
        let mask_tensor = Tensor::from_vec(mask_data, (batch_size, max_frames), device)?;

        Ok((mel_tensor, mask_tensor, valid_frame_counts))
    }

    fn prepare_audio(&self, audio_input: &AudioInput) -> Result<Vec<f32>> {
        let mono_samples = audio_input.to_mono();
        let scaled_samples: Vec<f32> = mono_samples
            .iter()
            .map(|&sample| sample * self.input_scale_factor)
            .collect();

        let dithered_samples = if self.dither > 0.0 {
            use rand_distr::{Distribution, Normal};

            let mut rng = rand::rng();
            let normal = Normal::new(0.0, self.dither as f64).unwrap();
            scaled_samples
                .iter()
                .map(|&sample| sample + normal.sample(&mut rng) as f32)
                .collect()
        } else {
            scaled_samples
        };

        let mut resampled = if audio_input.sample_rate != self.target_sample_rate {
            self.resample(
                &dithered_samples,
                audio_input.sample_rate,
                self.target_sample_rate,
            )?
        } else {
            dithered_samples
        };

        if resampled.len() > self.max_audio_samples {
            resampled.truncate(self.max_audio_samples);
        }

        Ok(resampled)
    }

    fn round_up_to_multiple(&self, value: usize, multiple: usize) -> usize {
        if value == 0 || multiple == 0 {
            value
        } else {
            value.div_ceil(multiple) * multiple
        }
    }

    fn extract_spectrogram(
        &self,
        waveform: &[f32],
        attention_mask: &[bool],
    ) -> Result<(Vec<Vec<f32>>, Vec<bool>)> {
        // Semicausal time-padding: prepend frame_length/2 zeros so the first
        // STFT frame is centered at t=0, matching HF's time_padding='semicausal'.
        let pad_left = self.frame_length / 2;
        let mut padded_waveform = vec![0.0f32; pad_left + waveform.len()];
        padded_waveform[pad_left..].copy_from_slice(waveform);
        let mut padded_mask = vec![false; pad_left + attention_mask.len()];
        padded_mask[pad_left..].copy_from_slice(attention_mask);
        let waveform = &padded_waveform[..];
        let attention_mask = &padded_mask[..];

        let frame_size_for_unfold = self.frame_length + 1;
        if waveform.len() < frame_size_for_unfold {
            return Ok((Vec::new(), Vec::new()));
        }

        let num_frames = (waveform.len() - frame_size_for_unfold) / self.hop_length + 1;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.fft_length);
        let mut mel_features = Vec::with_capacity(num_frames);
        let mut valid_mask = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            let raw_frame = &waveform[start..start + frame_size_for_unfold];
            let mut frame = Vec::with_capacity(self.frame_length);

            if self.preemphasis > 0.0 {
                if self.preemphasis_htk_flavor {
                    frame.push(raw_frame[0] * (1.0 - self.preemphasis));
                    for idx in 1..self.frame_length {
                        frame.push(raw_frame[idx] - self.preemphasis * raw_frame[idx - 1]);
                    }
                } else {
                    for idx in 0..self.frame_length {
                        frame.push(raw_frame[idx + 1] - self.preemphasis * raw_frame[idx]);
                    }
                }
            } else {
                frame.extend_from_slice(&raw_frame[..self.frame_length]);
            }

            let mut windowed = frame
                .iter()
                .zip(self.window.iter())
                .map(|(sample, window)| Complex32::new(sample * window, 0.0))
                .collect::<Vec<_>>();
            windowed.resize(self.fft_length, Complex32::new(0.0, 0.0));
            fft.process(&mut windowed);

            let magnitude = windowed[..self.fft_length / 2 + 1]
                .iter()
                .map(|complex| complex.norm())
                .collect::<Vec<_>>();

            let mut mel_frame = vec![0.0f32; self.feature_size];
            for (mel_idx, filter) in self.mel_filters.iter().enumerate() {
                let mut sum = 0.0f32;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    sum += magnitude[freq_idx] * coeff;
                }
                let mut value = (sum + self.mel_floor).ln();
                if let Some(mean) = self
                    .per_bin_mean
                    .as_ref()
                    .and_then(|mean| mean.get(mel_idx))
                {
                    value -= *mean as f32;
                }
                if let Some(stddev) = self
                    .per_bin_stddev
                    .as_ref()
                    .and_then(|stddev| stddev.get(mel_idx))
                    .copied()
                {
                    if stddev != 0.0 {
                        value /= stddev as f32;
                    }
                }
                mel_frame[mel_idx] = value;
            }

            mel_features.push(mel_frame);
            valid_mask.push(attention_mask[start + frame_size_for_unfold - 1]);
        }

        Ok((mel_features, valid_mask))
    }

    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }

        let expected_len = samples.len() * to_rate as usize / from_rate as usize;

        let sinc = rubato::SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2,
        };

        let mut resampler = rubato::SincFixedIn::<f32>::new(
            to_rate as f64 / from_rate as f64,
            2.0,
            sinc,
            samples.len(),
            1,
        )?;

        let delay = resampler.output_delay();
        let mut result = resampler.process(&[samples.to_vec()], None)?[0].clone();
        let target_len_with_delay = expected_len + delay;

        while result.len() < target_len_with_delay {
            let flushed = resampler.process_partial::<Vec<f32>>(None, None)?;
            let tail = &flushed[0];
            if tail.is_empty() {
                break;
            }
            result.extend_from_slice(tail);
        }

        if result.is_empty() {
            return Ok(Vec::new());
        }

        // Rubato exposes the filter latency via `output_delay()`, but trimming
        // it here shifts the waveform earlier than HF's audio loading path
        // (`librosa`/`soxr`) and causes the downstream Gemma4 mel frames to
        // drift. For HF parity we keep the leading samples and only truncate
        // to the expected resampled length.
        Ok(result.drain(..expected_len.min(result.len())).collect())
    }

    fn create_mel_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        min_frequency: f32,
        max_frequency: f32,
    ) -> Vec<Vec<f32>> {
        let n_freqs = n_fft / 2 + 1;
        let all_freqs = (0..n_freqs)
            .map(|idx| idx as f32 * sample_rate / n_fft as f32)
            .collect::<Vec<_>>();

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let min_mel = hz_to_mel(min_frequency);
        let max_mel = hz_to_mel(max_frequency);
        let mut f_pts = Vec::with_capacity(n_mels + 2);
        for idx in 0..n_mels + 2 {
            let mel = min_mel + (max_mel - min_mel) * idx as f32 / (n_mels + 1) as f32;
            f_pts.push(mel_to_hz(mel));
        }

        let mut filterbank = vec![vec![0.0; n_freqs]; n_mels];
        for idx in 0..n_mels {
            let left_freq = f_pts[idx];
            let center_freq = f_pts[idx + 1];
            let right_freq = f_pts[idx + 2];

            for (freq_idx, &freq) in all_freqs.iter().enumerate() {
                if freq >= left_freq && freq <= center_freq && center_freq > left_freq {
                    filterbank[idx][freq_idx] = (freq - left_freq) / (center_freq - left_freq);
                } else if freq > center_freq && freq <= right_freq && right_freq > center_freq {
                    filterbank[idx][freq_idx] = (right_freq - freq) / (right_freq - center_freq);
                }
            }
        }

        filterbank
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use mistralrs_audio::AudioInput;

    use crate::vision_models::preprocessor_config::PreProcessorConfig;

    use super::AudioProcessor;

    #[test]
    fn batched_audio_marks_padded_frames_invalid() {
        let processor = AudioProcessor::new(&PreProcessorConfig {
            sampling_rate: Some(16_000),
            frame_length: Some(320),
            hop_length: Some(160),
            feature_size: Some(128),
            fft_overdrive: Some(false),
            min_frequency: Some(0.0),
            max_frequency: Some(8_000.0),
            preemphasis: Some(0.0),
            mel_floor: Some(1e-3),
            ..Default::default()
        });

        let long = AudioInput {
            samples: vec![0.0; 3_200],
            sample_rate: 16_000,
            channels: 1,
        };
        let short = AudioInput {
            samples: vec![0.0; 1_600],
            sample_rate: 16_000,
            channels: 1,
        };

        let (_mel, mask, valid_frame_counts) = processor
            .process_audios(&[long, short], &Device::Cpu)
            .unwrap();

        assert_eq!(valid_frame_counts.len(), 2);
        assert!(valid_frame_counts[0] > valid_frame_counts[1]);

        let short_mask = mask.get(1).unwrap().to_vec1::<f32>().unwrap();
        assert!(short_mask
            .iter()
            .skip(valid_frame_counts[1])
            .all(|&value| value == 1.0));
    }

    #[test]
    fn resampling_flushes_tail_and_keeps_expected_length() {
        let processor = AudioProcessor::new(&PreProcessorConfig {
            sampling_rate: Some(16_000),
            ..Default::default()
        });

        let samples = vec![0.0f32; 400_384];
        let resampled = processor.resample(&samples, 44_100, 16_000).unwrap();

        assert_eq!(resampled.len(), 145_264);
    }
}
