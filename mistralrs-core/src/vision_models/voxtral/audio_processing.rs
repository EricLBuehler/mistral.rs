#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;
use rubato::Resampler;
use rustfft::{num_complex::Complex32, FftPlanner};

use super::config::AudioEncodingArgs;

/// Number of silence tokens to left-pad audio (matches voxmlx reference).
const N_LEFT_PAD_TOKENS: usize = 32;
/// Number of silence tokens to right-pad audio (matches voxmlx reference).
const N_RIGHT_PAD_TOKENS: usize = 17;

/// Whisper-style mel spectrogram processor for Voxtral audio encoder.
pub struct VoxtralAudioProcessor {
    sampling_rate: u32,
    frame_rate: f32,
    num_mel_bins: usize,
    hop_length: usize,
    window_size: usize,
    global_log_mel_max: f32,
}

impl VoxtralAudioProcessor {
    pub fn new(cfg: &AudioEncodingArgs) -> Self {
        Self {
            sampling_rate: cfg.sampling_rate,
            frame_rate: cfg.frame_rate as f32,
            num_mel_bins: cfg.num_mel_bins,
            hop_length: cfg.hop_length,
            window_size: cfg.window_size,
            global_log_mel_max: cfg.global_log_mel_max as f32,
        }
    }

    pub fn new_from_processor(other: &Self) -> Self {
        Self {
            sampling_rate: other.sampling_rate,
            frame_rate: other.frame_rate,
            num_mel_bins: other.num_mel_bins,
            hop_length: other.hop_length,
            window_size: other.window_size,
            global_log_mel_max: other.global_log_mel_max,
        }
    }

    /// Number of samples per streaming token (sampling_rate / frame_rate).
    fn samples_per_token(&self) -> usize {
        (self.sampling_rate as f32 / self.frame_rate) as usize
    }

    /// Process audio input into a mel spectrogram tensor.
    /// Left-pads with 32 tokens of silence and right-pads with 17 tokens of silence
    /// to match the reference implementation.
    /// Returns [1, T, num_mel_bins] tensor.
    pub fn process_audio(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let mono = audio.to_mono();

        // Resample if necessary
        let samples = if audio.sample_rate != self.sampling_rate {
            self.resample(&mono, audio.sample_rate, self.sampling_rate)?
        } else {
            mono
        };

        // Pad audio with silence: left_pad + audio + right_pad
        let spt = self.samples_per_token();
        let left_pad = N_LEFT_PAD_TOKENS * spt;
        let right_pad = N_RIGHT_PAD_TOKENS * spt;
        let mut padded = vec![0.0f32; left_pad + samples.len() + right_pad];
        padded[left_pad..left_pad + samples.len()].copy_from_slice(&samples);

        let mel = self.compute_mel_spectrogram(&padded)?;
        let num_frames = mel.len();
        if num_frames == 0 {
            anyhow::bail!("Audio too short to produce mel frames");
        }

        let data: Vec<f32> = mel.into_iter().flatten().collect();

        let tensor = Tensor::from_vec(data, (1, num_frames, self.num_mel_bins), device)?;
        Ok(tensor)
    }

    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }
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
        let result = resampler.process(&[samples.to_vec()], None)?;
        Ok(result[0].clone())
    }

    /// Centered STFT mel spectrogram matching `torch.stft(center=True)`.
    /// Applies reflection padding of n_fft//2 on each side, then drops the last STFT frame.
    fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        let n_fft = self.window_size;
        let hop = self.hop_length;
        let n_freqs = n_fft / 2 + 1;
        let pad = n_fft / 2;

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Reflection-pad the input by n_fft//2 on each side (matching torch.stft center=True)
        let padded_len = pad + samples.len() + pad;
        let mut padded = vec![0.0f32; padded_len];
        // Left reflection: samples[pad], samples[pad-1], ..., samples[1]
        for (i, p) in padded.iter_mut().enumerate().take(pad) {
            let src_idx = (pad - i).min(samples.len() - 1);
            *p = samples[src_idx];
        }
        // Center: copy original samples
        padded[pad..pad + samples.len()].copy_from_slice(samples);
        // Right reflection: samples[len-2], samples[len-3], ...
        for i in 0..pad {
            let src_idx = samples.len().saturating_sub(2 + i);
            padded[pad + samples.len() + i] = samples[src_idx];
        }

        let total_frames = (padded_len - n_fft) / hop + 1;
        // Drop last frame (matching HF: stft[..., :-1])
        let num_frames = total_frames.saturating_sub(1);

        // Hann window (periodic: w[n] = 0.5*(1 - cos(2*pi*n/N)))
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();

        let mel_filters = self.create_mel_filterbank(n_fft)?;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        let mut mel_features = Vec::with_capacity(num_frames);
        let log_mel_floor = self.global_log_mel_max - 8.0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;

            let mut buf: Vec<Complex32> = padded[start..start + n_fft]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex32::new(s * w, 0.0))
                .collect();

            fft.process(&mut buf);

            let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

            let mut mel_frame = vec![0.0f32; self.num_mel_bins];
            for (mel_idx, filter) in mel_filters.iter().enumerate() {
                let mut sum = 0.0f32;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < power.len() {
                        sum += power[freq_idx] * coeff;
                    }
                }
                let log_val = sum.max(1e-10).log10();
                let clamped = log_val.max(log_mel_floor);
                mel_frame[mel_idx] = (clamped + 4.0) / 4.0;
            }

            mel_features.push(mel_frame);
        }

        Ok(mel_features)
    }

    /// Slaney mel scale: Hz to mel.
    fn hertz_to_mel(freq: f32) -> f32 {
        const MIN_LOG_HERTZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 27.0 / 1.856_298; // 27.0 / ln(6.4)
        if freq >= MIN_LOG_HERTZ {
            MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * LOGSTEP
        } else {
            3.0 * freq / 200.0
        }
    }

    /// Slaney mel scale: mel to Hz.
    fn mel_to_hertz(mel: f32) -> f32 {
        const MIN_LOG_HERTZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 1.856_298 / 27.0; // ln(6.4) / 27.0
        if mel >= MIN_LOG_MEL {
            MIN_LOG_HERTZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
        } else {
            200.0 * mel / 3.0
        }
    }

    /// Create Slaney-style mel filterbank matching `mistral_common.audio.mel_filter_bank`.
    /// Returns `[n_mels][n_freqs]` with Slaney energy normalization.
    fn create_mel_filterbank(&self, n_fft: usize) -> Result<Vec<Vec<f32>>> {
        let n_freqs = n_fft / 2 + 1;
        let sr = self.sampling_rate as f32;
        let n_mels = self.num_mel_bins;

        // FFT bin frequencies: linspace(0, sr/2, n_freqs)
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * (sr / 2.0) / (n_freqs - 1) as f32)
            .collect();

        // Mel filter center frequencies (n_mels + 2 points)
        let mel_min = Self::hertz_to_mel(0.0);
        let mel_max = Self::hertz_to_mel(sr / 2.0);
        let filter_freqs: Vec<f32> = (0..n_mels + 2)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
                Self::mel_to_hertz(mel)
            })
            .collect();

        // Differences between adjacent filter frequencies
        let filter_diff: Vec<f32> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

        // Triangular filterbank (matching _create_triangular_filter_bank)
        let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];
        for m in 0..n_mels {
            for (j, &fft_f) in fft_freqs.iter().enumerate() {
                let slope_left = fft_f - filter_freqs[m];
                let slope_right = filter_freqs[m + 2] - fft_f;
                let down = slope_left / filter_diff[m]; // rising slope
                let up = slope_right / filter_diff[m + 1]; // falling slope
                filterbank[m][j] = 0.0f32.max(down.min(up));
            }
        }

        // Slaney energy normalization: constant energy per channel
        for m in 0..n_mels {
            let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
            for val in filterbank[m].iter_mut().take(n_freqs) {
                *val *= enorm;
            }
        }

        Ok(filterbank)
    }
}
