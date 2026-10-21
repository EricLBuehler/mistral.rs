//! Per-expert weights that live in one quantised stack and are dequantised on demand.
//!
//! A GGUF checkpoint keeps the expert stack as a single quantised block. Dequantising the
//! whole stack on every forward costs the entire model per step, which is what
//! `todo/mistralrs/02` records. This resolves only the experts a token routes to.
//!
//! The quantised stack is one contiguous block of bytes with the expert axis outermost, so the
//! per-expert stride is `storage_size_in_bytes() / n_experts` and a slice of that many bytes
//! dequantises to exactly the values the whole stack's slice of the same expert dequantises
//! to. `QTensor::new` checks the shape against the block size, so a slice that cannot be
//! represented is rejected rather than silently mis-dequantised.
//!
//! The gather is the same arithmetic as the fused path, only the row addressing differs, so
//! the two agree bit-for-bit; see `moe_gather_cpu::tests`.

use std::borrow::Cow;
use std::sync::Arc;

use candle_core::quantized::{QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor};

use crate::moe_gather_cpu::{self, NO_REPEAT};

/// A linear whose experts are one quantised stack, dequantised per selected expert.
pub struct PerExpertQuantizedLinear {
    qtensor: Arc<QTensor>,
    n_experts: usize,
    out_features: usize,
    in_features: usize,
    device: Device,
}

impl PerExpertQuantizedLinear {
    /// `qtensor` is the quantised stack, shaped `[n_experts, out_features, in_features]`.
    pub fn new(
        qtensor: Arc<QTensor>,
        n_experts: usize,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        let dims = qtensor.shape().dims().to_vec();
        if dims.len() != 3 {
            candle_core::bail!(
                "PerExpertQuantizedLinear: expected [experts, out, in], got {dims:?}"
            );
        }
        if dims[0] != n_experts || dims[1] != out_features || dims[2] != in_features {
            candle_core::bail!(
                "PerExpertQuantizedLinear: stack is {dims:?}, expected [{n_experts}, {out_features}, {in_features}]"
            );
        }
        let device = qtensor.device();
        Ok(Self {
            qtensor,
            n_experts,
            out_features,
            in_features,
            device,
        })
    }

    /// The dequantised slabs for the experts `ids` names, in ascending expert order, plus the
    /// same ids remapped onto that order.
    ///
    /// The remap is what lets the gather read a compacted slab list: it only ever indexes the
    /// experts it was told about.
    fn resolve(&self, ids: &[u32]) -> Result<(Vec<Tensor>, Vec<u32>)> {
        let mut wanted: Vec<usize> = ids
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| i < self.n_experts)
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        if wanted.is_empty() {
            candle_core::bail!("PerExpertQuantizedLinear: no expert was routed to");
        }

        let total = self.qtensor.storage_size_in_bytes();
        if !total.is_multiple_of(self.n_experts) {
            candle_core::bail!(
                "PerExpertQuantizedLinear: {total} bytes do not divide across {} experts",
                self.n_experts
            );
        }
        let stride = total / self.n_experts;

        let bytes = self.qtensor.data()?;
        let dtype = self.qtensor.dtype();
        let mut slabs = Vec::with_capacity(wanted.len());
        for &e in &wanted {
            let lo = e * stride;
            let Some(slab) = bytes.get(lo..lo + stride) else {
                candle_core::bail!("PerExpertQuantizedLinear: expert {e} is out of range");
            };
            let storage = QStorage::from_data(Cow::Borrowed(slab), &self.device, dtype)?;
            let q = QTensor::new(
                storage,
                (self.out_features, self.in_features),
            )?;
            slabs.push(q.dequantize(&self.device)?);
        }

        let mut dense = vec![u32::MAX; self.n_experts];
        for (row, &e) in wanted.iter().enumerate() {
            dense[e] = row as u32;
        }
        let remapped: Vec<u32> = ids
            .iter()
            .map(|&i| {
                if (i as usize) < self.n_experts {
                    dense[i as usize]
                } else {
                    u32::MAX
                }
            })
            .collect();
        Ok((slabs, remapped))
    }

    /// The gather over flat activation rows, one row per slot or per token depending on
    /// `repeat`.
    ///
    /// `repeat` is how many slots share one activation row: the top-k of a token share the
    /// token's row, so the rows are passed once and the kernel addresses them by
    /// `slot / repeat`.
    pub fn gather_slots(
        &self,
        a: &[f32],
        ids: &[u32],
        repeat: usize,
    ) -> Result<Vec<f32>> {
        let (slabs, dense) = self.resolve(ids)?;
        moe_gather_cpu::gather_gemv_per_expert(
            a,
            &slabs,
            self.out_features,
            self.in_features,
            &dense,
            repeat,
        )
    }

    /// `(n_tokens, 1|top_k, in) @ gather(experts, ids).T` as `(n_tokens, top_k, out_features)`.
    ///
    /// The two input forms differ in which activation a slot reads: with one row per token
    /// every one of the token's slots reads that row, and with one row per slot each slot
    /// reads its own. The gather takes the rows and a repeat count, so the broadcast form
    /// passes the token's row once with the repeat count, and the expanded form passes the
    /// rows already expanded.
    pub fn gather_rows(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let (n_tokens, xt, in_features) = x.dims3()?;
        if in_features != self.in_features {
            candle_core::bail!(
                "PerExpertQuantizedLinear: input has {in_features} features, weights have {}",
                self.in_features
            );
        }
        let top_k = ids.elem_count() / n_tokens.max(1);
        if top_k == 0 || !ids.elem_count().is_multiple_of(n_tokens) {
            candle_core::bail!(
                "PerExpertQuantizedLinear: {} ids for {n_tokens} tokens",
                ids.elem_count()
            );
        }
        let ids_flat: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?.to_vec();

        let (a_rows, repeat) = if xt == 1 {
            (x.reshape((n_tokens, in_features))?, top_k)
        } else if xt == top_k {
            (x.reshape((n_tokens * top_k, in_features))?, NO_REPEAT)
        } else {
            candle_core::bail!(
                "PerExpertQuantizedLinear: input has {xt} slots per token, expected 1 or {top_k}"
            );
        };
        let a: Vec<f32> = a_rows
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let out = self.gather_slots(&a, &ids_flat, repeat)?;
        Tensor::from_vec(
            out,
            (n_tokens, top_k, self.out_features),
            &self.device,
        )
    }
}

impl std::fmt::Debug for PerExpertQuantizedLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerExpertQuantizedLinear")
            .field("n_experts", &self.n_experts)
            .field("out_features", &self.out_features)
            .field("in_features", &self.in_features)
            .field("dtype", &self.qtensor.dtype())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::IndexOp;

    /// A per-expert slice of a quantised stack must dequantise to exactly the values the
    /// whole stack's slice of the same expert dequantises to. The slicing is what keeps the
    /// dequantisation scoped, so a wrong stride or a wrong shape would show up here as
    /// different values rather than as a wrong answer later.
    #[test]
    fn a_slice_dequantises_to_the_whole_stack_slice() -> Result<()> {
        let dev = Device::Cpu;
        let (n_expert, n, k) = (4usize, 8usize, 64usize);
        // Distinctive values, so a wrong slab is obvious rather than merely different.
        let mut data = vec![0f32; n_expert * n * k];
        for e in 0..n_expert {
            for i in 0..n * k {
                data[e * n * k + i] = (e as f32) * 1000.0 + (i as f32) * 0.5;
            }
        }
        let src = Tensor::from_vec(data, (n_expert, n, k), &dev)?;
        let q = Arc::new(QTensor::quantize(&src, GgmlDType::Q4_0)?);

        let linear = PerExpertQuantizedLinear::new(q.clone(), n_expert, n, k)?;
        let whole = q.dequantize(&dev)?;

        // The slices the linear would resolve, and the whole stack's own slices to compare.
        let ids: Vec<u32> = (0..n_expert as u32).collect();
        let (slabs, _dense) = linear.resolve(&ids)?;
        assert_eq!(slabs.len(), n_expert);
        for (e, slab) in slabs.iter().enumerate() {
            let want = whole.i(e..e + 1)?.flatten_all()?.to_vec1::<f32>()?;
            let got = slab.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(want.len(), got.len());
            assert_eq!(want, got, "expert {e}: slice differs from the whole stack");
        }
        Ok(())
    }

    /// The scoped path only helps if it resolves the experts the gather names and nothing
    /// else, so the byte count it touches has to track the routing rather than the stack.
    #[test]
    fn resolve_only_reads_the_named_experts() -> Result<()> {
        let dev = Device::Cpu;
        let (n_expert, n, k) = (16usize, 8usize, 64usize);
        let src = Tensor::from_vec(vec![0.5f32; n_expert * n * k], (n_expert, n, k), &dev)?;
        let q = Arc::new(QTensor::quantize(&src, GgmlDType::Q4_0)?);
        let linear = PerExpertQuantizedLinear::new(q, n_expert, n, k)?;

        let ids = vec![3u32, 3, 7, 11];
        let (slabs, dense) = linear.resolve(&ids)?;
        assert_eq!(slabs.len(), 3, "three distinct experts were named");
        // The dense ids point at the compacted slabs in ascending expert order.
        assert_eq!(dense, vec![0, 0, 1, 2]);
        Ok(())
    }
}
