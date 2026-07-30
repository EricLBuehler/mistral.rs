use anyhow::Result;
use candle_core::{Device, Tensor};
use rubato::Resampler;
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::AudioInput;

// === Configuration constants ===
// NOTE: All configuration values are now loaded from the preprocessor config.
// Defaults match the HuggingFace Transformers implementation.

pub struct AudioProcessor {
    target_sample_rate: u32,
    fft_overdrive: bool,
    feature_size: usize,
    frame_length: usize,
    hop_length: usize,
    min_frequency: f32,
    max_frequency: f32,
    preemphasis: f32,
    mel_floor: f32,
    dither: f32,
    input_scale_factor: f32,
    per_bin_mean: Option<Vec<f64>>,
    per_bin_stddev: Option<Vec<f64>>,
}

impl AudioProcessor {
    pub fn new(config: &PreProcessorConfig) -> Self {
        // Load from config with defaults matching transformers implementation
        let target_sample_rate = config.sampling_rate.unwrap_or(16000) as u32;
        let frame_length = config.frame_length.unwrap_or(512);
        let hop_length = config.hop_length.unwrap_or(160);

        Self {
            target_sample_rate,
            fft_overdrive: config.fft_overdrive.unwrap_or(true),
            feature_size: config.feature_size.unwrap_or(128),
            frame_length,
            hop_length,
            min_frequency: config.min_frequency.unwrap_or(125.0) as f32,
            max_frequency: config.max_frequency.unwrap_or(7600.0) as f32,
            preemphasis: config.preemphasis.unwrap_or(0.97) as f32,
            mel_floor: config.mel_floor.unwrap_or(1e-5) as f32,
            dither: config.dither.unwrap_or(0.0) as f32,
            input_scale_factor: config.input_scale_factor.unwrap_or(1.0) as f32,
            per_bin_mean: config.per_bin_mean.clone(),
            per_bin_stddev: config.per_bin_stddev.clone(),
        }
    }

    pub fn process_audio(
        &self,
        audio_input: &AudioInput,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // Convert to mono
        let mono_samples = audio_input.to_mono();

        // Apply input scaling
        let scaled_samples: Vec<f32> = mono_samples
            .iter()
            .map(|&s| s * self.input_scale_factor)
            .collect();

        // Apply dithering if configured
        let dithered_samples = if self.dither > 0.0 {
            use rand_distr::{Distribution, Normal};
            let mut rng = rand::rng();
            let normal = Normal::new(0.0, self.dither as f64).unwrap();
            scaled_samples
                .iter()
                .map(|&s| s + normal.sample(&mut rng) as f32)
                .collect()
        } else {
            scaled_samples
        };

        // Resample if necessary
        let resampled = if audio_input.sample_rate != self.target_sample_rate {
            self.resample(
                &dithered_samples,
                audio_input.sample_rate,
                self.target_sample_rate,
            )?
        } else {
            dithered_samples
        };

        // Use frame parameters from config
        let frame_length = self.frame_length;
        let hop_length = self.hop_length;

        // Compute mel-spectrogram
        let mel_spectrogram = self.compute_mel_spectrogram(
            &resampled,
            frame_length,
            hop_length,
            self.target_sample_rate,
        )?;

        // Apply per-bin normalization if configured
        let normalized_spectrogram = if self.per_bin_mean.is_some() || self.per_bin_stddev.is_some()
        {
            self.normalize_mel_spectrogram(mel_spectrogram)?
        } else {
            mel_spectrogram
        };

        // Convert to tensors
        let num_frames = normalized_spectrogram.len();
        let mel_data: Vec<f32> = normalized_spectrogram.into_iter().flatten().collect();
        let mel_tensor = Tensor::from_vec(mel_data, (1, num_frames, self.feature_size), device)?;

        // Create mask (all valid for now)
        let mask = Tensor::zeros((1, num_frames), candle_core::DType::F32, device)?;

        Ok((mel_tensor, mask))
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

        let samples_vec = vec![samples.to_vec()];
        let result = resampler.process(&samples_vec, None)?;
        Ok(result[0].clone())
    }

    #[allow(dead_code)]
    fn apply_preemphasis(&self, samples: &[f32], coeff: f32) -> Vec<f32> {
        // Retained for potential future use (e.g. non-HTK pre-emphasis) but not
        // called in the current implementation.
        if samples.is_empty() {
            return vec![];
        }

        let mut out = Vec::with_capacity(samples.len());
        out.push(samples[0] * (1.0 - coeff));
        for i in 1..samples.len() {
            out.push(samples[i] - coeff * samples[i - 1]);
        }
        out
    }

    fn compute_mel_spectrogram(
        &self,
        samples: &[f32],
        frame_length: usize,
        hop_length: usize,
        sample_rate: u32,
    ) -> Result<Vec<Vec<f32>>> {
        // FFT size (with overdrive if enabled)
        let mut n_fft = frame_length.next_power_of_two();
        if self.fft_overdrive {
            n_fft *= 2;
        }

        // Create FFT planner
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        // === Hann window (same formulation as the reference implementation) ===
        let window: Vec<f64> = (0..frame_length)
            .map(|n| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / frame_length as f64).cos())
            })
            .collect();

        // Create mel filterbank
        let mel_filters =
            self.create_mel_filterbank(self.feature_size, n_fft, sample_rate as f32)?;

        // Process frames
        // We replicate the exact frame logic used in
        // `Gemma3nAudioFeatureExtractor` from the Python reference.
        // frame_length_py == self.frame_length (512) and Python computes
        // `frame_size_for_unfold = frame_length + 1` to be able to reference
        // the previous sample when applying pre-emphasis.  Instead of copying
        // a full additional sample we simply look back one position when
        // computing each point **inside the same frame**.  The resulting
        // values are identical to the reference implementation.

        let frame_size_for_pe = frame_length + 1; // 513
        if samples.len() < frame_size_for_pe {
            return Ok(Vec::new());
        }

        let num_frames = (samples.len() - frame_size_for_pe) / hop_length + 1;
        let mut mel_features = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;
            let raw_frame = &samples[start..start + frame_size_for_pe];

            // === Pre-emphasis (HTK flavour) ===
            let mut frame: Vec<f32> = Vec::with_capacity(frame_length);
            // First sample – scaled, no look-back.
            frame.push(raw_frame[0] * (1.0 - self.preemphasis));
            // Remaining samples use the previous raw sample within the frame.
            for i in 1..frame_length {
                frame.push(raw_frame[i] - self.preemphasis * raw_frame[i - 1]);
            }

            // === Window ===
            let mut windowed: Vec<Complex32> = frame
                .iter()
                .zip(window.iter())
                .map(|(s, w)| Complex32::new(s * *w as f32, 0.0))
                .collect();

            // === FFT ===
            windowed.resize(n_fft, Complex32::new(0.0, 0.0));
            fft.process(&mut windowed);

            // Positive frequencies magnitude (length n_fft/2 + 1)
            let magnitude: Vec<f32> = windowed[0..n_fft / 2 + 1]
                .iter()
                .map(|c| c.norm())
                .collect();

            // === Mel filter-bank projection ===
            let mut mel_frame = vec![0.0f32; self.feature_size];
            for (mel_idx, filter) in mel_filters.iter().enumerate() {
                let mut sum = 0.0f32;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < magnitude.len() {
                        sum += magnitude[freq_idx] * coeff;
                    }
                }
                mel_frame[mel_idx] = (sum.max(self.mel_floor)).ln();
            }

            mel_features.push(mel_frame);
        }

        Ok(mel_features)
    }

    fn create_mel_filterbank(
        &self,
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
    ) -> Result<Vec<Vec<f32>>> {
        let n_freqs = n_fft / 2 + 1;

        // Create frequency bins (actual frequencies for each FFT bin)
        let all_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate / n_fft as f32)
            .collect();

        // Mel scale conversion functions
        let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
        let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) };

        // Create mel points in Hz (not bin indices!)
        let min_mel = hz_to_mel(self.min_frequency);
        let max_mel = hz_to_mel(self.max_frequency);

        // Linear spacing in mel scale
        let mut f_pts = Vec::with_capacity(n_mels + 2);
        for i in 0..n_mels + 2 {
            let mel = min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32;
            f_pts.push(mel_to_hz(mel));
        }

        // Create triangular filters
        let mut filterbank = vec![vec![0.0; n_freqs]; n_mels];

        for i in 0..n_mels {
            let left_freq = f_pts[i];
            let center_freq = f_pts[i + 1];
            let right_freq = f_pts[i + 2];

            // Apply triangular filter based on actual frequencies
            for (j, &freq) in all_freqs.iter().enumerate() {
                if freq >= left_freq && freq <= center_freq {
                    // Rising edge
                    if center_freq > left_freq {
                        filterbank[i][j] = (freq - left_freq) / (center_freq - left_freq);
                    }
                } else if freq > center_freq && freq <= right_freq {
                    // Falling edge
                    if right_freq > center_freq {
                        filterbank[i][j] = (right_freq - freq) / (right_freq - center_freq);
                    }
                }
            }
        }

        Ok(filterbank)
    }

    fn normalize_mel_spectrogram(&self, mel_spectrogram: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        let mut normalized = mel_spectrogram;

        // Apply per-bin mean subtraction
        if let Some(ref mean) = self.per_bin_mean {
            for frame in normalized.iter_mut() {
                for (i, val) in frame.iter_mut().enumerate() {
                    if i < mean.len() {
                        *val -= mean[i] as f32;
                    }
                }
            }
        }

        // Apply per-bin stddev division
        if let Some(ref stddev) = self.per_bin_stddev {
            for frame in normalized.iter_mut() {
                for (i, val) in frame.iter_mut().enumerate() {
                    if i < stddev.len() && stddev[i] != 0.0 {
                        *val /= stddev[i] as f32;
                    }
                }
            }
        }

        Ok(normalized)
    }
}
