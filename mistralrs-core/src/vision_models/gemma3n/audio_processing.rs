use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;
use rustfft::{num_complex::Complex32, FftPlanner};
use apodize::{hanning_iter};
use rubato::Resampler;

const FRAME_LENGTH_MS: f32 = 32.0;
const HOP_LENGTH_MS: f32 = 10.0;
const MIN_FREQUENCY: f32 = 125.0;
const MAX_FREQUENCY: f32 = 7600.0;
const PREEMPHASIS: f32 = 0.97;
const MEL_FLOOR: f32 = 1e-5;
const FEATURE_SIZE: usize = 128; // Number of mel bins

pub struct AudioProcessor {
    target_sample_rate: u32,
    fft_overdrive: bool,
}

impl AudioProcessor {
    pub fn new() -> Self {
        Self {
            target_sample_rate: 16000,
            fft_overdrive: true,
        }
    }

    pub fn process_audio(&self, audio_input: &AudioInput, device: &Device) -> Result<(Tensor, Tensor)> {
        // Convert to mono
        let mono_samples = audio_input.to_mono();
        
        // Resample if necessary
        let resampled = if audio_input.sample_rate != self.target_sample_rate {
            self.resample(&mono_samples, audio_input.sample_rate, self.target_sample_rate)?
        } else {
            mono_samples
        };

        // Calculate frame parameters
        let frame_length = (FRAME_LENGTH_MS * self.target_sample_rate as f32 / 1000.0) as usize;
        let hop_length = (HOP_LENGTH_MS * self.target_sample_rate as f32 / 1000.0) as usize;
        
        // Apply preemphasis
        let preemphasized = self.apply_preemphasis(&resampled, PREEMPHASIS);
        
        // Compute mel spectrogram
        let mel_spectrogram = self.compute_mel_spectrogram(
            &preemphasized,
            frame_length,
            hop_length,
            self.target_sample_rate,
        )?;
        
        // Convert to tensors
        let num_frames = mel_spectrogram.len();
        let mel_data: Vec<f32> = mel_spectrogram.into_iter().flatten().collect();
        let mel_tensor = Tensor::from_vec(mel_data, (1, num_frames, FEATURE_SIZE), device)?;
        
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

    fn apply_preemphasis(&self, samples: &[f32], coeff: f32) -> Vec<f32> {
        if samples.is_empty() {
            return vec![];
        }

        let mut preemphasized = Vec::with_capacity(samples.len());
        
        // HTK-style: first sample is unmodified
        preemphasized.push(samples[0]);
        
        // Apply filter to remaining samples
        for i in 1..samples.len() {
            preemphasized.push(samples[i] - coeff * samples[i - 1]);
        }
        
        preemphasized
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
        
        // Create Hanning window
        let window: Vec<f64> = hanning_iter(frame_length).collect();
        
        // Create mel filterbank
        let mel_filters = self.create_mel_filterbank(FEATURE_SIZE, n_fft, sample_rate as f32)?;
        
        // Process frames
        let num_frames = (samples.len() - frame_length) / hop_length + 1;
        let mut mel_features = Vec::with_capacity(num_frames);
        
        for i in 0..num_frames {
            let start = i * hop_length;
            let end = start + frame_length;
            if end > samples.len() {
                break;
            }
            
            // Apply window
            let mut windowed: Vec<Complex32> = samples[start..end]
                .iter()
                .zip(window.iter())
                .map(|(s, w)| Complex32::new(s * *w as f32, 0.0))
                .collect();
            
            // Pad to FFT size
            windowed.resize(n_fft, Complex32::new(0.0, 0.0));
            
            // Apply FFT
            fft.process(&mut windowed);
            
            // Take magnitude spectrum of positive frequencies
            let magnitude: Vec<f32> = windowed[0..n_fft / 2 + 1]
                .iter()
                .map(|c| c.norm())
                .collect();
            
            // Apply mel filterbank
            let mut mel_frame = vec![0.0; FEATURE_SIZE];
            for (mel_idx, filter) in mel_filters.iter().enumerate() {
                let mut sum = 0.0;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < magnitude.len() {
                        sum += magnitude[freq_idx] * magnitude[freq_idx] * coeff;
                    }
                }
                // Log with floor to avoid log(0)
                mel_frame[mel_idx] = (sum.max(MEL_FLOOR)).ln();
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
        let fmax = sample_rate / 2.0;
        
        // Mel scale conversion
        let hz_to_mel = |hz: f32| -> f32 {
            2595.0 * (1.0 + hz / 700.0).log10()
        };
        
        let mel_to_hz = |mel: f32| -> f32 {
            700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
        };
        
        // Create mel points
        let min_mel = hz_to_mel(MIN_FREQUENCY);
        let max_mel = hz_to_mel(fmax.min(MAX_FREQUENCY));
        
        let mut mel_points = Vec::with_capacity(n_mels + 2);
        for i in 0..n_mels + 2 {
            let mel = min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32;
            mel_points.push(mel_to_hz(mel));
        }
        
        // Convert to FFT bin indices
        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&hz| ((n_fft as f32 * hz / sample_rate).round() as usize).min(n_freqs - 1))
            .collect();
        
        // Create triangular filters
        let mut filterbank = vec![vec![0.0; n_freqs]; n_mels];
        
        for i in 0..n_mels {
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];
            
            // Rising edge
            for j in left..center {
                if center > left {
                    filterbank[i][j] = (j - left) as f32 / (center - left) as f32;
                }
            }
            
            // Falling edge
            for j in center..right {
                if right > center {
                    filterbank[i][j] = 1.0 - (j - center) as f32 / (right - center) as f32;
                }
            }
            
            // Peak
            if center < n_freqs {
                filterbank[i][center] = 1.0;
            }
        }
        
        Ok(filterbank)
    }
}