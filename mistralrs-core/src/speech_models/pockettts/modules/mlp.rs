use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub type StepFn = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

#[derive(Clone)]
pub struct RMSNorm {
    alpha: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(dim, "alpha")?;
        Ok(Self { alpha, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        // Python's "RMSNorm" uses x.var() which IS mean((x - mean)²), NOT standard RMSNorm
        // We must match Python exactly for parity
        let var = x.var_keepdim(candle_core::D::Minus1)?;
        let inv_rms = (var + self.eps)?.sqrt()?.recip()?;
        let normalized = x.broadcast_mul(&inv_rms)?;
        normalized.broadcast_mul(&self.alpha)?.to_dtype(x_dtype)
    }
}

#[derive(Clone)]
pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<Self> {
        let (weight, bias) = if affine {
            let weight = vb.get(dim, "weight")?;
            let bias = vb.get(dim, "bias")?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        // candle_nn::LayerNorm::new takes (weight, bias, eps)
        // If not affine, we can pass None for weight/bias but candle_nn::LayerNorm expects Tensor if present.
        // Actually candle_nn has a layer_norm function.
        // Let's use it.
        let weight = weight.unwrap_or_else(|| Tensor::ones(dim, vb.dtype(), vb.device()).unwrap());
        let bias = bias.unwrap_or_else(|| Tensor::zeros(dim, vb.dtype(), vb.device()).unwrap());

        Ok(Self {
            inner: candle_nn::LayerNorm::new(weight, bias, eps),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

#[derive(Clone)]
pub struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    pub fn new(channels: usize, _init: f32, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(channels, "scale")?;
        Ok(Self { scale })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

#[derive(Clone)]
pub struct TimestepEmbedder {
    lin1: Linear,
    lin2: Linear,
    norm: RMSNorm,
    freqs: Tensor,
}

impl TimestepEmbedder {
    pub fn new(
        hidden_size: usize,
        frequency_embedding_size: usize,
        max_period: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let lin1 = candle_nn::linear(frequency_embedding_size, hidden_size, vb.pp("mlp.0"))?;
        let lin2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        let norm = RMSNorm::new(hidden_size, 1e-5, vb.pp("mlp.3"))?;

        let half = frequency_embedding_size / 2;
        let ds = Tensor::arange(0u32, half as u32, vb.device())?.to_dtype(DType::F32)?;
        let freqs = ds
            .affine(-(max_period.ln() as f64) / half as f64, 0.0)?
            .exp()?
            .to_dtype(vb.dtype())?; // Pre-convert to model dtype

        Ok(Self {
            lin1,
            lin2,
            norm,
            freqs,
        })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // t is [B], freqs is [half]
        // We need args to be [B, half] for MLP to process
        let t = if t.dims().len() == 1 {
            t.unsqueeze(1)? // [B] -> [B, 1]
        } else {
            t.clone()
        };
        // args = t * freqs: [B, 1] * [half] -> [B, half]
        let args = t.broadcast_mul(&self.freqs)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        // [B, half] cat [B, half] -> [B, frequency_embedding_size]
        let mut x = Tensor::cat(&[cos, sin], candle_core::D::Minus1)?;

        // Forward through MLP sequence: lin1 -> silu -> lin2 -> norm
        x = self.lin1.forward(&x)?;
        x = x.silu()?;
        x = self.lin2.forward(&x)?;
        x = self.norm.forward(&x)?;

        Ok(x)
    }
}

pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

#[derive(Clone)]
pub struct ModulationParams {
    pub shift: Tensor,
    pub scale: Tensor,
    pub gate: Option<Tensor>,
}

#[derive(Clone)]
pub struct ResBlock {
    in_ln: LayerNorm,
    mlp_lin1: Linear,
    mlp_lin2: Linear,
    ada_ln_lin: Linear,
}

impl ResBlock {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let in_ln = LayerNorm::new(channels, 1e-6, true, vb.pp("in_ln"))?;
        let mlp_lin1 = candle_nn::linear(channels, channels, vb.pp("mlp.0"))?;
        let mlp_lin2 = candle_nn::linear(channels, channels, vb.pp("mlp.2"))?;
        let ada_ln_lin = candle_nn::linear(channels, 3 * channels, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            in_ln,
            mlp_lin1,
            mlp_lin2,
            ada_ln_lin,
        })
    }

    pub fn forward(&self, x: &Tensor, modulation: &ModulationParams) -> Result<Tensor> {
        let mut h = self.in_ln.forward(x)?;
        h = modulate(&h, &modulation.shift, &modulation.scale)?;
        h = self.mlp_lin1.forward(&h)?.silu()?;
        h = self.mlp_lin2.forward(&h)?;

        if let Some(gate) = &modulation.gate {
            x + h.broadcast_mul(gate)
        } else {
            x + h
        }
    }
}

#[derive(Clone)]
pub struct FinalLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_lin: Linear,
}

impl FinalLayer {
    pub fn new(model_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = LayerNorm::new(model_channels, 1e-6, false, vb.pp("norm_final"))?;
        let linear = candle_nn::linear(model_channels, out_channels, vb.pp("linear"))?;
        let ada_ln_lin = candle_nn::linear(
            model_channels,
            2 * model_channels,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_lin,
        })
    }

    pub fn forward(&self, x: &Tensor, modulation: &ModulationParams) -> Result<Tensor> {
        let h = modulate(
            &self.norm_final.forward(x)?,
            &modulation.shift,
            &modulation.scale,
        )?;
        self.linear.forward(&h)
    }
}

#[derive(Clone)]
pub struct SimpleMLPAdaLN {
    time_embeds: Vec<TimestepEmbedder>,
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
    num_time_conds: usize,
}

impl SimpleMLPAdaLN {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
        max_period: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::new(
                model_channels,
                256,
                max_period,
                vb.pp(format!("time_embed.{}", i)),
            )?);
        }

        let cond_embed = candle_nn::linear(cond_channels, model_channels, vb.pp("cond_embed"))?;
        let input_proj = candle_nn::linear(in_channels, model_channels, vb.pp("input_proj"))?;

        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks.push(ResBlock::new(
                model_channels,
                vb.pp(format!("res_blocks.{}", i)),
            )?);
        }

        let final_layer = FinalLayer::new(model_channels, out_channels, vb.pp("final_layer"))?;

        Ok(Self {
            time_embeds,
            cond_embed,
            input_proj,
            res_blocks,
            final_layer,
            num_time_conds,
        })
    }

    pub fn forward(&self, c: &Tensor, s: &Tensor, t: &Tensor, x: &Tensor) -> Result<Tensor> {
        let c_emb = self.embed_condition(c)?;
        self.forward_step(x, &c_emb, s, t)
    }

    pub fn embed_condition(&self, c: &Tensor) -> Result<Tensor> {
        self.cond_embed.forward(c)
    }

    pub fn forward_step(
        &self,
        x: &Tensor,
        c_emb: &Tensor,
        s: &Tensor,
        t: &Tensor,
    ) -> Result<Tensor> {
        let y = (self.time_embeds[0].forward(s)? + self.time_embeds[1].forward(t)?)?;
        let t_combined = (y / self.num_time_conds as f64)?;

        // Compute modulations on the fly for non-cached call
        let mod_vec = self.precompute_modulations(c_emb, &t_combined)?;
        self.forward_step_cached(x, &mod_vec[0])
    }
}

impl SimpleMLPAdaLN {
    pub fn compute_time_embeddings(
        &self,
        num_steps: usize,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut embeddings = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            let s = i as f64 / num_steps as f64;
            let t = (i + 1) as f64 / num_steps as f64;

            // 1D Tensors [1]
            let s_tensor = Tensor::new(&[s as f32], device)?.to_dtype(dtype)?;
            let t_tensor = Tensor::new(&[t as f32], device)?.to_dtype(dtype)?;

            let t0 = self.time_embeds[0].forward(&s_tensor)?;
            let t1 = self.time_embeds[1].forward(&t_tensor)?;
            let t_combined = ((t0 + t1)? / self.num_time_conds as f64)?;
            embeddings.push(t_combined);
        }
        // stack of [1, 512] -> [num_steps, 1, 512]
        // squeeze(1) -> [num_steps, 512]
        Tensor::stack(&embeddings, 0)?.squeeze(1)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn precompute_modulations(
        &self,
        c_emb: &Tensor,
        time_embeddings: &Tensor,
    ) -> Result<Vec<Vec<ModulationParams>>> {
        // c_emb: [1, 512], time_embeddings: [8, 512]
        let num_steps = time_embeddings.dim(0)?;
        let y = time_embeddings.broadcast_add(c_emb)?; // [8, 512]
        let y_silu = y.silu()?;

        let mut all_step_modulations =
            vec![Vec::with_capacity(self.res_blocks.len() + 1); num_steps];

        // ResBlocks
        for block in &self.res_blocks {
            let mod_batch = block.ada_ln_lin.forward(&y_silu)?; // [8, 1536]
            let dim = mod_batch.dim(candle_core::D::Minus1)? / 3;

            for s in 0..num_steps {
                let modulation = mod_batch.narrow(0, s, 1)?; // [1, 1536]
                let shift = modulation.narrow(candle_core::D::Minus1, 0, dim)?;
                let scale = modulation.narrow(candle_core::D::Minus1, dim, dim)?;
                let gate = modulation.narrow(candle_core::D::Minus1, 2 * dim, dim)?;
                all_step_modulations[s].push(ModulationParams {
                    shift,
                    scale,
                    gate: Some(gate),
                });
            }
        }

        // Final layer
        let mod_batch = self.final_layer.ada_ln_lin.forward(&y_silu)?; // [8, 1024]
        let dim = mod_batch.dim(candle_core::D::Minus1)? / 2;
        for s in 0..num_steps {
            let modulation = mod_batch.narrow(0, s, 1)?; // [1, 1024]
            let shift = modulation.narrow(candle_core::D::Minus1, 0, dim)?;
            let scale = modulation.narrow(candle_core::D::Minus1, dim, dim)?;
            all_step_modulations[s].push(ModulationParams {
                shift,
                scale,
                gate: None,
            });
        }

        Ok(all_step_modulations)
    }

    pub fn forward_step_cached(
        &self,
        x: &Tensor,
        modulations: &[ModulationParams],
    ) -> Result<Tensor> {
        let mut x = self.input_proj.forward(x)?;

        for (i, block) in self.res_blocks.iter().enumerate() {
            x = block.forward(&x, &modulations[i])?;
        }

        self.final_layer
            .forward(&x, &modulations[self.res_blocks.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    #[test]
    fn test_rmsnorm_parity() -> Result<()> {
        let device = Device::Cpu;
        let mut map = HashMap::new();
        map.insert(
            "alpha".to_string(),
            Tensor::ones((4,), DType::F32, &device)?,
        );
        let vb = VarBuilder::from_tensors(map, DType::F32, &device);
        let norm = RMSNorm::new(4, 1e-5, vb)?;

        // Input: [[1.0, 2.0, 3.0, 4.0]]
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
        let y = norm.forward(&x)?;

        // Python's "RMSNorm" uses x.var() = mean((x - mean)²)
        // mean = 2.5, var = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 3 = 1.6667 (Bessel)
        // rsqrt(1.6667 + 1e-5) ≈ 0.7746
        // output = x * 0.7746 = [0.7746, 1.5492, 2.3238, 3.0984]
        let expected = Tensor::new(&[[0.7746f32, 1.5492, 2.3238, 3.0984]], &device)?;

        let diff = (y - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "RMSNorm parity failed: diff={}", diff);
        Ok(())
    }
}
