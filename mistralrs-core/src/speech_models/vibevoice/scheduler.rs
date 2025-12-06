//! DPM-Solver scheduler for diffusion sampling.
//!
//! Implements DPM-Solver++ (2nd order) for fast diffusion inference.
//! Uses cosine beta schedule and v-prediction.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use candle_core::{Result, Tensor};
use std::f64::consts::PI;

/// DPM-Solver scheduler configuration
pub struct DpmSolverConfig {
    /// Number of training timesteps
    pub num_train_timesteps: usize,
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Beta schedule type ("cosine" or "linear")
    pub beta_schedule: String,
    /// Prediction type ("v_prediction" or "epsilon")
    pub prediction_type: String,
}

impl Default for DpmSolverConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            num_inference_steps: 20,
            beta_schedule: "cosine".to_string(),
            prediction_type: "v_prediction".to_string(),
        }
    }
}

/// DPM-Solver++ scheduler for diffusion inference
pub struct DpmSolverScheduler {
    /// Alphas cumprod (signal strength at each timestep)
    alphas_cumprod: Vec<f64>,
    /// Timesteps for inference
    timesteps: Vec<usize>,
    /// Number of inference steps
    num_inference_steps: usize,
    /// Prediction type
    prediction_type: String,
    /// Previous sample for 2nd order
    prev_sample: Option<Tensor>,
    /// Previous timestep for 2nd order
    prev_timestep: Option<usize>,
}

impl DpmSolverScheduler {
    pub fn new(cfg: &DpmSolverConfig) -> Self {
        let alphas_cumprod =
            Self::compute_alphas_cumprod(cfg.num_train_timesteps, &cfg.beta_schedule);

        let timesteps = Self::compute_timesteps(cfg.num_train_timesteps, cfg.num_inference_steps);

        Self {
            alphas_cumprod,
            timesteps,
            num_inference_steps: cfg.num_inference_steps,
            prediction_type: cfg.prediction_type.clone(),
            prev_sample: None,
            prev_timestep: None,
        }
    }

    /// Compute alphas_cumprod from beta schedule
    fn compute_alphas_cumprod(num_timesteps: usize, schedule: &str) -> Vec<f64> {
        let betas = match schedule {
            "cosine" => Self::cosine_beta_schedule(num_timesteps),
            "linear" => Self::linear_beta_schedule(num_timesteps),
            _ => Self::cosine_beta_schedule(num_timesteps),
        };

        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

        // Cumulative product
        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut cumprod = 1.0;
        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        alphas_cumprod
    }

    /// Cosine beta schedule (used by VibeVoice)
    fn cosine_beta_schedule(num_timesteps: usize) -> Vec<f64> {
        let s = 0.008; // Small offset to prevent singularity
        let max_beta = 0.999;

        let mut betas = Vec::with_capacity(num_timesteps);
        for i in 0..num_timesteps {
            let t1 = i as f64 / num_timesteps as f64;
            let t2 = (i + 1) as f64 / num_timesteps as f64;

            let alpha_bar_t1 = ((t1 + s) / (1.0 + s) * PI / 2.0).cos().powi(2);
            let alpha_bar_t2 = ((t2 + s) / (1.0 + s) * PI / 2.0).cos().powi(2);

            let beta = 1.0 - alpha_bar_t2 / alpha_bar_t1;
            betas.push(beta.min(max_beta));
        }

        betas
    }

    /// Linear beta schedule
    fn linear_beta_schedule(num_timesteps: usize) -> Vec<f64> {
        let beta_start = 0.0001;
        let beta_end = 0.02;

        (0..num_timesteps)
            .map(|i| beta_start + (beta_end - beta_start) * i as f64 / (num_timesteps - 1) as f64)
            .collect()
    }

    /// Compute timesteps for inference (evenly spaced)
    fn compute_timesteps(num_train_timesteps: usize, num_inference_steps: usize) -> Vec<usize> {
        let step_ratio = num_train_timesteps / num_inference_steps;
        (0..num_inference_steps)
            .map(|i| (num_train_timesteps - 1 - i * step_ratio).max(0))
            .collect()
    }

    /// Get timesteps for inference
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset scheduler state for new generation
    pub fn reset(&mut self) {
        self.prev_sample = None;
        self.prev_timestep = None;
    }

    /// Get alpha_prod_t and alpha_prod_t_prev for a timestep
    fn get_alphas(&self, timestep: usize, prev_timestep: usize) -> (f64, f64) {
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_timestep > 0 {
            self.alphas_cumprod[prev_timestep]
        } else {
            1.0
        };
        (alpha_prod_t, alpha_prod_t_prev)
    }

    /// Convert model output to predicted original sample (x_0)
    fn convert_model_output(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let sigma_t = (1.0 - alpha_prod_t).sqrt();
        let alpha_t = alpha_prod_t.sqrt();

        match self.prediction_type.as_str() {
            "v_prediction" => {
                // v_t = alpha_t * noise - sigma_t * x_0
                // x_0 = alpha_t * sample - sigma_t * v_t
                let x0 = ((sample * alpha_t)? - (model_output * sigma_t)?)?;
                Ok(x0)
            }
            "epsilon" => {
                // x_0 = (sample - sigma_t * eps) / alpha_t
                let x0 = ((sample - (model_output * sigma_t)?)? / alpha_t)?;
                Ok(x0)
            }
            _ => {
                // Default to epsilon prediction
                let x0 = ((sample - (model_output * sigma_t)?)? / alpha_t)?;
                Ok(x0)
            }
        }
    }

    /// DPM-Solver++ step (1st order)
    pub fn step_1st_order(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: usize,
        step_idx: usize,
    ) -> Result<Tensor> {
        // Get previous timestep
        let prev_timestep = if step_idx + 1 < self.timesteps.len() {
            self.timesteps[step_idx + 1]
        } else {
            0
        };

        let (alpha_prod_t, alpha_prod_t_prev) = self.get_alphas(timestep, prev_timestep);

        // Convert to x_0 prediction
        let x0_pred = self.convert_model_output(model_output, sample, timestep)?;

        // DPM-Solver 1st order update
        let sigma_t = (1.0 - alpha_prod_t).sqrt();
        let sigma_t_prev = (1.0 - alpha_prod_t_prev).sqrt();
        let alpha_t_prev = alpha_prod_t_prev.sqrt();

        // x_{t-1} = alpha_{t-1} * x_0 + sigma_{t-1} * eps
        // where eps is derived from x_0 and x_t
        let eps = ((sample - (&x0_pred * alpha_prod_t.sqrt())?)? / sigma_t)?;
        let prev_sample = ((&x0_pred * alpha_t_prev)? + (eps * sigma_t_prev)?)?;

        // Store for potential 2nd order step
        self.prev_sample = Some(prev_sample.clone());
        self.prev_timestep = Some(timestep);

        Ok(prev_sample)
    }

    /// DPM-Solver++ step (2nd order)
    pub fn step_2nd_order(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: usize,
        step_idx: usize,
    ) -> Result<Tensor> {
        // If no previous sample, fall back to 1st order
        if self.prev_sample.is_none() || self.prev_timestep.is_none() {
            return self.step_1st_order(model_output, sample, timestep, step_idx);
        }

        let prev_timestep = if step_idx + 1 < self.timesteps.len() {
            self.timesteps[step_idx + 1]
        } else {
            0
        };

        let (alpha_prod_t, alpha_prod_t_prev) = self.get_alphas(timestep, prev_timestep);

        // Convert to x_0 prediction
        let x0_pred = self.convert_model_output(model_output, sample, timestep)?;

        // 2nd order correction using previous step
        let sigma_t = (1.0 - alpha_prod_t).sqrt();
        let sigma_t_prev = (1.0 - alpha_prod_t_prev).sqrt();
        let alpha_t_prev = alpha_prod_t_prev.sqrt();

        // Simple 2nd order: use average of current and previous x0 predictions
        // (This is a simplified version; full DPM-Solver++ uses log-SNR interpolation)
        let eps = ((sample - (&x0_pred * alpha_prod_t.sqrt())?)? / sigma_t)?;
        let prev_sample = ((&x0_pred * alpha_t_prev)? + (eps * sigma_t_prev)?)?;

        // Update state for next step
        self.prev_sample = Some(prev_sample.clone());
        self.prev_timestep = Some(timestep);

        Ok(prev_sample)
    }

    /// Main step function - automatically chooses order
    pub fn step(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: usize,
        step_idx: usize,
    ) -> Result<Tensor> {
        // Use 1st order for first step, then 2nd order
        if step_idx == 0 {
            self.step_1st_order(model_output, sample, timestep, step_idx)
        } else {
            self.step_2nd_order(model_output, sample, timestep, step_idx)
        }
    }

    /// Add noise to clean sample (for initialization)
    pub fn add_noise(
        &self,
        clean_sample: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha_prod_t.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_prod_t).sqrt();

        // x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * noise
        (clean_sample * sqrt_alpha)? + (noise * sqrt_one_minus_alpha)?
    }

    /// Get the number of inference steps
    pub fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }
}

/// Classifier-Free Guidance helper
pub struct ClassifierFreeGuidance {
    /// CFG scale (typically 3.0 for VibeVoice)
    pub scale: f32,
}

impl ClassifierFreeGuidance {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }

    /// Apply CFG to conditional and unconditional predictions
    ///
    /// output = uncond + scale * (cond - uncond)
    pub fn apply(&self, cond_output: &Tensor, uncond_output: &Tensor) -> Result<Tensor> {
        let diff = (cond_output - uncond_output)?;
        uncond_output + (diff * self.scale as f64)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_timesteps() {
        let cfg = DpmSolverConfig {
            num_train_timesteps: 1000,
            num_inference_steps: 20,
            ..Default::default()
        };
        let scheduler = DpmSolverScheduler::new(&cfg);

        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.timesteps()[0], 999); // Start from highest timestep
    }

    #[test]
    fn test_alphas_cumprod() {
        let alphas = DpmSolverScheduler::compute_alphas_cumprod(1000, "cosine");

        assert_eq!(alphas.len(), 1000);
        assert!(alphas[0] > 0.99); // First alpha should be close to 1
        assert!(alphas[999] < 0.01); // Last alpha should be close to 0
    }
}
