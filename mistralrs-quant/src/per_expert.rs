//! Per-expert weights kept as one view per expert instead of a fused stack.
//!
//! A stacked-expert checkpoint normally gets fused into `[num_experts, out, in]` at load,
//! which costs the whole model in anonymous memory on top of the mapped file. For a
//! memory-mapped safetensors checkpoint the individual per-expert tensors are already
//! zero-copy views of the mapping, so keeping them un-fused leaves the model in reclaimable
//! page cache and bounds the resident set to the experts a token actually routes to.
//!
//! The gather is the same arithmetic as the fused path, only the row addressing differs, so
//! the two agree bit-for-bit; see `moe_gather_cpu::tests`.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

use crate::moe_gather_cpu;
use crate::{
    IsqPlanParams, IsqRequest, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedSerde,
};

/// One `[out_features, in_features]` view per expert, indexed directly by the router's output.
pub struct PerExpertLinear {
    experts: Arc<Vec<Tensor>>,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    device: Device,
}

impl PerExpertLinear {
    pub fn new(experts: Vec<Tensor>, out_features: usize, in_features: usize) -> Result<Self> {
        let first = experts
            .first()
            .ok_or_else(|| candle_core::Error::msg("PerExpertLinear needs at least one expert"))?;
        let dtype = first.dtype();
        let device = first.device().clone();
        for slab in &experts {
            if slab.dtype() != dtype {
                candle_core::bail!("PerExpertLinear: expert slabs disagree on dtype")
            }
            if !slab.device().same_device(&device) {
                candle_core::bail!("PerExpertLinear: expert slabs are on different devices")
            }
        }
        Ok(Self {
            experts: Arc::new(experts),
            out_features,
            in_features,
            dtype,
            device,
        })
    }

    pub fn experts(&self) -> &[Tensor] {
        &self.experts
    }

    /// `(n_tokens, in) @ gather(slabs, ids).T` as `(n_tokens, top_k, out_features)`, with every
    /// one of a token's `top_k` slots reading the same activation row.
    fn gather_rows(&self, a: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let (n_tokens, in_features) = a.dims2()?;
        if in_features != self.in_features {
            candle_core::bail!(
                "PerExpertLinear: input has {in_features} features, weights have {}",
                self.in_features
            )
        }
        let top_k = ids.elem_count() / n_tokens.max(1);
        if top_k == 0 || !ids.elem_count().is_multiple_of(n_tokens) {
            candle_core::bail!(
                "PerExpertLinear: {} ids for {n_tokens} tokens",
                ids.elem_count()
            )
        }
        let ids: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?.to_vec();

        // The token's activation is read by all of its slots, so the rows are passed once and
        // the kernel addresses them by `slot / top_k`. Expanding them instead would allocate
        // and copy `top_k * in_features` floats on every projection of every layer, which at
        // batch 1 is 150 short-lived allocations per decode step.
        let a_rows = a.to_dtype(DType::F32)?.contiguous()?;
        let a_rows = a_rows.flatten_all()?.to_vec1::<f32>()?;

        let out = moe_gather_cpu::gather_gemv_per_expert(
            &a_rows,
            &self.experts,
            self.out_features,
            self.in_features,
            &ids,
            top_k,
        )?;
        Tensor::from_vec(out, (n_tokens, top_k, self.out_features), &self.device)?
            .to_dtype(self.dtype)
    }
}

impl std::fmt::Debug for PerExpertLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerExpertLinear")
            .field("experts", &self.experts.len())
            .field("out_features", &self.out_features)
            .field("in_features", &self.in_features)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl QuantizedSerde for PerExpertLinear {
    fn isq_serde_supported(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "per-expert-linear"
    }
}

impl QuantMethod for PerExpertLinear {
    fn new(_method: QuantMethodConfig) -> Result<Self> {
        candle_core::bail!("PerExpertLinear is constructed directly, not from a QuantMethodConfig")
    }

    /// The fused stack would be the "weight" here, and there deliberately is not one: the
    /// point of this type is that the stack is never built.
    fn dequantize_w(&self) -> Result<Tensor> {
        candle_core::bail!("PerExpertLinear has no fused weight to dequantize")
    }

    fn forward_raw(&self, _a: &Tensor) -> Result<Tensor> {
        candle_core::bail!("PerExpertLinear is only usable through gather_forward")
    }

    /// Shapes are the ones the stacked CPU path already hands a dense expert projection:
    /// `(batch, seq, 1, 1, hidden)` with `(batch, seq, top_k)` indices for the gate and up
    /// projections, and `(batch, seq, top_k, hidden)` for the down projection.
    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        match a.dims() {
            &[b, s, 1, 1, h] => {
                let (_ib, _is, k) = indices.dims3()?;
                let a_flat = a.reshape((b * s, h))?;
                let ids = indices.reshape((b * s, k))?;
                self.gather_rows(&a_flat, &ids)?
                    .reshape((b, s, k, self.out_features))
            }
            &[b, s, k, h] => {
                let (_ib, _is, ik) = indices.dims3()?;
                if ik != k {
                    candle_core::bail!("PerExpertLinear: input has {k} slots but indices have {ik}")
                }
                // Per-slot inputs rather than per-token: the activation row is already the
                // slot's own, so no broadcast is needed.
                let a_flat = a.reshape((b * s * k, h))?;
                let ids = indices.reshape((b * s * k,))?;
                self.gather_rows(&a_flat, &ids)?
                    .reshape((b, s, k, self.out_features))
            }
            dims => candle_core::bail!("PerExpertLinear: unsupported input shape {dims:?}"),
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.device.clone())
    }

    fn plan_isq(&self, _request: &IsqRequest) -> Result<IsqPlanParams> {
        candle_core::bail!("PerExpertLinear does not support ISQ")
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("PerExpertLinear does not support add_delta_w")
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &std::sync::atomic::AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("PerExpertLinear does not support ISQ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slabs(experts: usize, out: usize, in_features: usize, dtype: DType) -> Result<Vec<Tensor>> {
        (0..experts)
            .map(|_| Tensor::rand(-1f32, 1f32, (out, in_features), &Device::Cpu)?.to_dtype(dtype))
            .collect()
    }

    #[test]
    fn rejects_inconsistent_slabs() -> Result<()> {
        let mixed = vec![
            Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?,
            Tensor::zeros((4, 4), DType::BF16, &Device::Cpu)?,
        ];
        assert!(PerExpertLinear::new(mixed, 4, 4).is_err());
        assert!(PerExpertLinear::new(vec![], 4, 4).is_err());
        Ok(())
    }

    /// The whole point is that the fused and un-fused layouts agree, so this compares against
    /// the fused path rather than a hand-written reference.
    #[test]
    fn matches_a_fused_unquant_linear_on_the_cpu_shapes() -> Result<()> {
        use candle_nn::Linear;

        let (experts, out, hidden, k) = (8usize, 32usize, 16usize, 4usize);
        let b = 2usize;
        let seq = 3usize;
        let n = b * seq;

        for dtype in [DType::F32, DType::BF16] {
            let per_expert = slabs(experts, out, hidden, dtype)?;
            let layer = PerExpertLinear::new(per_expert.clone(), out, hidden)?;
            let stack = Tensor::stack(&per_expert, 0)?;
            let dense = crate::UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
                stack, None,
            )))?;

            // The stacked CPU path hands both gathers (batch, seq, top_k) ids.
            let ids = Tensor::from_vec(
                (0..n * k)
                    .map(|i| (i * 3 % experts) as u32)
                    .collect::<Vec<_>>(),
                (b, seq, k),
                &Device::Cpu,
            )?;
            let x = Tensor::rand(-1f32, 1f32, (n, hidden), &Device::Cpu)?.to_dtype(dtype)?;

            // (batch, seq, 1, 1, hidden): the shape the stacked CPU gate/up gathers receive.
            let got = layer.gather_forward(&x.reshape((b, seq, 1, 1, hidden))?, &ids)?;
            let want = dense.gather_forward(&x.reshape((b, seq, 1, 1, hidden))?, &ids)?;
            assert_eq!(got.dims(), want.dims(), "{dtype:?}");
            assert_close(&got, &want, "{dtype:?}")?;

            // (batch, seq, top_k, hidden): the per-slot shape the down projection receives.
            let per_slot =
                Tensor::rand(-1f32, 1f32, (b, seq, k, hidden), &Device::Cpu)?.to_dtype(dtype)?;
            let got = layer.gather_forward(&per_slot, &ids)?;
            let want = dense.gather_forward(&per_slot, &ids)?;
            assert_eq!(got.dims(), want.dims(), "{dtype:?}");
            assert_close(&got, &want, "{dtype:?}")?;
        }
        Ok(())
    }

    fn assert_close(got: &Tensor, want: &Tensor, ctx: &str) -> Result<()> {
        let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let want = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got.len(), want.len(), "{ctx}");
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let mut worst = 0f32;
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((g - w).abs());
        }
        // Identical inputs and an identical kernel: the layouts differ only in how a row is
        // addressed, so this is exact for f32 and a rounding match for bf16.
        let tol = 1e-6 * scale;
        assert!(worst <= tol, "{ctx}: worst {worst} > {tol}");
        Ok(())
    }
}
