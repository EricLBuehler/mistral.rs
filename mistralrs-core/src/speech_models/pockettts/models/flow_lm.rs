use crate::speech_models::pockettts::models::transformer::StreamingTransformer;
use crate::speech_models::pockettts::modules::mlp::{LayerNorm, ModulationParams, SimpleMLPAdaLN};
use crate::speech_models::pockettts::voice_state::ModelState;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub fn lsd_decode(
    flow_net: &SimpleMLPAdaLN,
    modulations: &[Vec<ModulationParams>],
    x_0: &Tensor,
) -> Result<Tensor> {
    let mut current = x_0.clone();
    let num_steps = modulations.len();

    let step_factor = 1.0 / num_steps as f64;
    for step_mod in modulations {
        // Use forward_step_cached with pre-computed modulation batch for this ODE step
        let flow_dir = flow_net.forward_step_cached(&current, step_mod)?;
        current = (current + flow_dir.affine(step_factor, 0.0)?)?;
    }
    Ok(current)
}

#[derive(Clone)]
pub struct FlowLMModel {
    pub flow_net: SimpleMLPAdaLN,
    pub transformer: StreamingTransformer,
    pub input_linear: Linear,
    pub out_norm: LayerNorm,
    pub out_eos: Linear,
    pub bos_emb: Tensor,
    pub emb_mean: Tensor,
    pub emb_std: Tensor,
    pub ldim: usize,
    pub dim: usize,
    pub noise_clamp: Option<f32>,
}

fn sample_noise(
    device: &candle_core::Device,
    shape: (usize, usize),
    temp: f32,
    clamp: Option<f32>,
) -> Result<Tensor> {
    let std = temp.sqrt();
    let noise = Tensor::randn(0.0f32, std, shape, device)?;
    match clamp {
        None => Ok(noise),
        Some(limit) => noise.clamp(-limit, limit),
    }
}

impl FlowLMModel {
    pub fn new(
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        ldim: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_linear = candle_nn::linear_no_bias(ldim, dim, vb.pp("input_linear"))?;
        let out_norm = LayerNorm::new(dim, 1e-5, true, vb.pp("out_norm"))?;
        let out_eos = candle_nn::linear(dim, 1, vb.pp("out_eos"))?;
        let bos_emb = vb.get(ldim, "bos_emb")?;
        let emb_mean = vb.get(ldim, "emb_mean")?;
        let emb_std = vb.get(ldim, "emb_std")?;

        Ok(Self {
            flow_net,
            transformer,
            input_linear,
            out_norm,
            out_eos,
            bos_emb,
            emb_mean,
            emb_std,
            ldim,
            dim,
            noise_clamp: None, // Default to no clamp
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        sequence: &Tensor,
        text_embeddings: &Tensor,
        model_state: &mut ModelState,
        time_embeddings: &Tensor,
        temp: f32,
        eos_threshold: f32,
        step: usize,
    ) -> Result<(Tensor, bool)> {
        // sequence is [B, T, ldim]
        // text_embeddings is [B, S, dim]

        // Handle BOS (if NaN, use bos_emb) - simplistic check for NaN
        // In Candle we can use `Tensor::where_cond`
        // But for now let's assume sequence passed in doesn't have NaNs or handled upstream.
        // Original: sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)

        // Let's assume BOS is handled by caller for now or if sequence empty.

        let x = self.input_linear.forward(sequence)?;
        let s_len = text_embeddings.dims()[1];

        // Cat text embeddings and sequence embeddings only if text_embeddings is not empty
        let transformer_out_pre_norm = if s_len > 0 {
            let input = Tensor::cat(&[text_embeddings, &x], 1)?;
            let mut out = self.transformer.forward(&input, model_state, step)?;
            // Remove prefix (text embeddings length)
            out = out.narrow(1, s_len, out.dims()[1] - s_len)?;
            out
        } else {
            self.transformer.forward(&x, model_state, step)?
        };

        let transformer_out = self.out_norm.forward(&transformer_out_pre_norm)?;

        // Only use the last frame for generation
        let last_frame = transformer_out
            .narrow(1, transformer_out.dims()[1] - 1, 1)?
            .squeeze(1)?;

        let eos_score = self
            .out_eos
            .forward(&last_frame)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar::<f32>()?;
        let is_eos = eos_score > eos_threshold;

        // Generate noise with optional clamping
        let noise = sample_noise(
            last_frame.device(),
            (last_frame.dims()[0], self.ldim),
            temp,
            self.noise_clamp,
        )?;

        // Pre-compute all modulations for this frame's ODE steps (8 steps * N blocks) in batch
        let c_emb = self.flow_net.embed_condition(&last_frame)?;
        let modulations = self
            .flow_net
            .precompute_modulations(&c_emb, time_embeddings)?;

        let next_latent = lsd_decode(&self.flow_net, &modulations, &noise)?;

        Ok((next_latent, is_eos))
    }
}
