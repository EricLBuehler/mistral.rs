//! Metal-resident stacked GGUF routed-expert weights with a bounded gather.
//!
//! The full-stack fallback in [`super::cpu`] dequantizes every expert onto the
//! compute device, which exhausts GPU memory for large MoE checkpoints. This
//! module keeps stacked routed experts `[experts, n, k]` as packed GGUF bytes
//! on the Metal device and executes routed gathers with
//! [`crate::metal_kernels::call_indexed_moe_gemv`], which reads only the
//! selected experts' blocks. The stack is never dequantized or materialized
//! densely. GGML dtypes the kernel does not implement fail closed at
//! construction with an actionable error instead of failing at prompt time.

use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{
    backend::BackendStorage, quantized::GgmlDType, DType, Device, MetalStorage, Result, Shape,
    Storage, Tensor, D,
};

use super::{archive::GgufArchive, mmap};
use crate::{
    metal_kernels, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
    UqffTensor,
};

/// Stacked routed-expert weights `[experts, n, k]` resident on a Metal device
/// as packed GGUF bytes.
#[derive(Debug)]
pub struct GgufMetalExperts {
    bytes: Tensor,
    dtype: GgmlDType,
    dims: Vec<usize>,
    bias: Option<Tensor>,
}

impl GgufMetalExperts {
    /// Build from a stacked GGUF archive tensor, uploading the packed bytes to
    /// the Metal device once at load time. The archive stays untouched and the
    /// bytes are never read back except by explicit full-dequantize operations.
    pub fn new_from_archive(
        archive: &GgufArchive,
        tensor_name: &str,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let info = archive.tensor_info(tensor_name)?;
        let dtype = info.dtype().candle_dtype()?;
        let dims = info.shape().to_vec();
        let data = archive.tensor_data(tensor_name)?;
        let bytes = data.bytes();
        let expected = mmap::packed_byte_len(mmap::elem_count(&dims)?, dtype)?;
        if bytes.len() != expected {
            candle_core::bail!(
                "stacked-expert GGUF tensor `{tensor_name}` has {} packed bytes, expected {expected}",
                bytes.len()
            );
        }
        let bytes = Tensor::from_vec(bytes.to_vec(), (bytes.len(),), device)?;
        Self::new(dtype, dims, bytes, bias, Some(tensor_name))
    }

    /// Validated constructor from resident bytes (shared with tests).
    pub fn new(
        dtype: GgmlDType,
        dims: Vec<usize>,
        bytes: Tensor,
        bias: Option<Tensor>,
        tensor_name: Option<&str>,
    ) -> Result<Self> {
        let name = tensor_name.unwrap_or("<stacked-expert-bytes>");
        let [experts, n, k] = dims.as_slice() else {
            candle_core::bail!("stacked-expert GGUF tensor `{name}` must be rank 3, got {dims:?}");
        };
        let (experts, n, k) = (*experts, *n, *k);
        if experts == 0 || n == 0 || k == 0 {
            candle_core::bail!(
                "stacked-expert GGUF tensor `{name}` has an empty dimension [{experts}, {n}, {k}]"
            );
        }
        match dtype {
            GgmlDType::Q2K | GgmlDType::Q4K | GgmlDType::Q6K => {
                if !k.is_multiple_of(256) {
                    candle_core::bail!(
                        "stacked-expert GGUF tensor `{name}` uses {dtype:?} with K={k}; \
                         the Metal indexed-MoE kernel requires K divisible by 256"
                    );
                }
            }
            GgmlDType::Q4_0 | GgmlDType::Q8_0 => {
                if !k.is_multiple_of(32) {
                    candle_core::bail!(
                        "stacked-expert GGUF tensor `{name}` uses {dtype:?} with K={k}; \
                         the Metal indexed-MoE kernel requires K divisible by 32"
                    );
                }
            }
            other => candle_core::bail!(
                "stacked-expert GGUF tensor `{name}` uses {other:?}, which has no bounded Metal \
                 indexed-MoE kernel; routed experts would be fully dequantized on the device. \
                 Load with `--cpu` so the experts stay mmap-backed, or use a quant whose routed \
                 experts use Q2_K, Q4_0, Q4_K, Q6_K, or Q8_0"
            ),
        }
        if !matches!(bytes.device(), Device::Metal(_)) {
            candle_core::bail!(
                "stacked-expert GGUF tensor `{name}` requires Metal-resident bytes, got {:?}",
                bytes.device()
            );
        }
        if let Some(bias) = &bias {
            match bias.dims() {
                [b_n] if *b_n == n => {}
                [b_experts, b_n] if *b_experts == experts && *b_n == n => {}
                other => candle_core::bail!(
                    "stacked-expert GGUF tensor `{name}` bias {other:?} does not match \
                     experts [{experts}, {n}, {k}]"
                ),
            }
        }
        Ok(Self {
            bytes,
            dtype,
            dims,
            bias,
        })
    }

    fn gather_forward_metal(&self, input: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (experts, n, k) = (self.dims[0], self.dims[1], self.dims[2]);
        // Normalize the MoE block's activation/index layouts to per-pair rows.
        let (x3, ids2, leading) = match *input.dims() {
            [b, s, xt, h] => {
                let (ib, is, t) = indices.dims3()?;
                if ib != b || is != s {
                    candle_core::bail!(
                        "GGUF expert gather activations {:?} and indices {:?} disagree on the \
                         leading batch dimensions",
                        input.dims(),
                        indices.dims()
                    );
                }
                (
                    input.reshape((b * s, xt, h))?,
                    indices.reshape((b * s, t))?,
                    Some((b, s, t)),
                )
            }
            [b, s, 1, 1, h] => {
                let (ib, is, t) = indices.dims3()?;
                if ib != b || is != s {
                    candle_core::bail!(
                        "GGUF expert gather activations {:?} and indices {:?} disagree on the \
                         leading batch dimensions",
                        input.dims(),
                        indices.dims()
                    );
                }
                (
                    input.reshape((b * s, 1, h))?,
                    indices.reshape((b * s, t))?,
                    Some((b, s, t)),
                )
            }
            [t, _, _] => {
                let (t2, _) = indices.dims2()?;
                if t2 != t {
                    candle_core::bail!(
                        "GGUF expert gather activations {:?} and indices {:?} disagree on the \
                         token count",
                        input.dims(),
                        indices.dims()
                    );
                }
                (input.clone(), indices.clone(), None)
            }
            _ => candle_core::bail!(
                "unsupported activation shape {:?} for the GGUF stacked-expert gather",
                input.dims()
            ),
        };
        let (tokens, xt, h) = x3.dims3()?;
        let topk = ids2.dim(1)?;
        if h != k {
            candle_core::bail!(
                "GGUF expert gather activations {:?} are incompatible with expert weights {:?}",
                x3.dims(),
                self.dims
            );
        }
        if xt != 1 && xt != topk {
            candle_core::bail!(
                "GGUF expert gather has {xt} activation rows per token, expected 1 or top-k {topk}"
            );
        }
        let flat_ids = ids2.to_dtype(DType::U32)?.contiguous()?.flatten_all()?;
        let max_id = flat_ids.max(D::Minus1)?.to_scalar::<u32>()? as usize;
        if max_id >= experts {
            candle_core::bail!("GGUF expert index {max_id} is out of range for {experts} experts");
        }
        let pairs = flat_ids.elem_count();
        let x = x3.to_dtype(DType::F32)?.contiguous()?;

        let (x_storage, x_layout) = x.storage_and_layout();
        let Storage::Metal(x_storage) = &*x_storage else {
            candle_core::bail!("GGUF expert gather requires Metal activations");
        };
        let (w_storage, w_layout) = self.bytes.storage_and_layout();
        let Storage::Metal(w_storage) = &*w_storage else {
            candle_core::bail!("GGUF stacked experts must be Metal-resident");
        };
        let (i_storage, i_layout) = flat_ids.storage_and_layout();
        let Storage::Metal(i_storage) = &*i_storage else {
            candle_core::bail!("GGUF expert gather requires Metal indices");
        };

        let device = x_storage.device();
        // Scope the encoder: it holds the shared Metal command-encoder state, so
        // no candle tensor op (e.g. the bias `index_select` below) may run while
        // it is alive.
        let out_buffer = {
            let encoder = device.command_encoder()?;
            encoder.set_label("gguf-indexed-moe-gemv");
            let out_buffer =
                device.new_buffer(pairs * n, DType::F32, "gguf-indexed-moe-gemv-out")?;
            metal_kernels::call_indexed_moe_gemv(
                device.device(),
                &encoder,
                metal_kernels::Kernels::global(),
                self.dtype,
                (w_storage.buffer(), w_layout.start_offset()),
                (
                    x_storage.buffer(),
                    x_layout.start_offset() * x_storage.dtype().size_in_bytes(),
                ),
                (
                    i_storage.buffer(),
                    i_layout.start_offset() * i_storage.dtype().size_in_bytes(),
                ),
                &out_buffer,
                n,
                k,
                topk,
                xt == topk,
                pairs,
            )
            .map_err(candle_core::Error::wrap)?;
            out_buffer
        };

        let mut out_shape = match leading {
            Some((b, s, t)) => vec![b, s, t],
            None => vec![tokens, topk],
        };
        out_shape.push(n);
        let mut out = Tensor::from((
            Storage::Metal(MetalStorage::new(
                out_buffer,
                device.clone(),
                pairs * n,
                DType::F32,
            )),
            Shape::from((pairs, n)),
        ))
        .reshape(Shape::from(out_shape.clone()))?;
        if let Some(bias) = &self.bias {
            let bias = bias.to_dtype(DType::F32)?;
            if bias.rank() == 2 {
                let selected = bias
                    .index_select(&flat_ids, 0)?
                    .reshape(Shape::from(out_shape.clone()))?;
                out = out.broadcast_add(&selected)?;
            } else {
                out = out.broadcast_add(&bias)?;
            }
        }
        Ok(out)
    }
}

impl QuantMethod for GgufMetalExperts {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("GgufMetalExperts must be constructed from a GgufArchive")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        // Explicit-operation only: copy the packed bytes back and dequantize on
        // the CPU. Never called by the routed MoE forward path.
        let bytes = self.bytes.to_device(&Device::Cpu)?.to_vec1::<u8>()?;
        let mut output = vec![0f32; mmap::elem_count(&self.dims)?];
        mmap::dequantize(self.dtype, &bytes, &mut output)?;
        Tensor::from_vec(output, Shape::from(self.dims.clone()), &Device::Cpu)
    }

    fn forward_raw(&self, _input: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "stacked GGUF experts on Metal support routed `gather_forward` only; a dense \
             forward would require every expert"
        )
    }

    fn gather_forward_raw(&self, input: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.gather_forward_metal(input, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F32)
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, self.bytes.device().clone())
    }

    fn plan_isq(&self, _request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        candle_core::bail!(
            "ISQ of Metal-resident stacked GGUF experts is not supported; run without ISQ or \
             keep the routed experts CPU mmap-backed"
        )
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("LoRA deltas on Metal-resident stacked GGUF experts are not supported")
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() && device.is_cpu() {
            return Ok(self);
        }
        candle_core::bail!(
            "ISQ/transfer of Metal-resident stacked GGUF experts is not supported; run without \
             ISQ or keep the routed experts CPU mmap-backed"
        )
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

impl QuantizedSerde for GgufMetalExperts {
    fn name(&self) -> &'static str {
        "gguf"
    }

    fn isq_serde_supported(&self) -> bool {
        false
    }

    fn uqff_type(&self) -> Option<IsqType> {
        None
    }

    fn serialize_uqff(&self, _prefix: &str, _ty: IsqType) -> Result<Vec<UqffTensor>> {
        candle_core::bail!(
            "serializing Metal-resident stacked GGUF experts to UQFF is not supported"
        )
    }
}

#[cfg(test)]
mod metal_tests {
    use super::mmap::packed_byte_len;
    use super::*;
    use crate::gguf::GgufMatMul;

    fn metal_device() -> Option<Device> {
        Device::new_metal(0).ok()
    }

    fn patterned(rows: usize, cols: usize, offset: usize) -> Result<Tensor> {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let v = ((i * 2654435761 + offset * 7919) % 2000) as f32 / 2000.0;
                v - 0.5
            })
            .collect();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu)
    }

    struct Fixture {
        method: GgufMetalExperts,
        /// Exact F32 dequantization of the quantized weights, `[E * n, k]`.
        reference: Vec<f32>,
        n: usize,
        k: usize,
    }

    fn fixture(
        dtype: GgmlDType,
        experts: usize,
        n: usize,
        k: usize,
        device: &Device,
        bias: Option<Tensor>,
    ) -> Result<Fixture> {
        let w_cpu = patterned(experts * n, k, 7)?;
        let quantized = candle_core::quantized::QTensor::quantize(&w_cpu, dtype)?;
        let bytes = quantized.data()?.to_vec();
        let expected_len = packed_byte_len(experts * n * k, dtype)?;
        assert_eq!(bytes.len(), expected_len);
        let bytes = Tensor::from_vec(bytes, (expected_len,), device)?;
        let method = GgufMetalExperts::new(
            dtype,
            vec![experts, n, k],
            bytes,
            bias,
            Some("blk.0.test_exps.weight"),
        )?;
        let reference = quantized
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Fixture {
            method,
            reference,
            n,
            k,
        })
    }

    fn expected(
        fixture: &Fixture,
        ids: &[u32],
        x: &[f32],
        topk: usize,
        x_per_pair: bool,
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let k = fixture.k;
        let mut out = vec![0f32; ids.len() * fixture.n];
        for (pair, &e) in ids.iter().enumerate() {
            let e = e as usize;
            let token = if x_per_pair { pair } else { pair / topk };
            for r in 0..fixture.n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += fixture.reference[(e * fixture.n + r) * k + kk] * x[token * k + kk];
                }
                if let Some(bias) = bias {
                    acc += bias[e * fixture.n + r];
                }
                out[pair * fixture.n + r] = acc;
            }
        }
        out
    }

    fn assert_close(got: &Tensor, expected: &[f32], tol: f32) -> Result<()> {
        let got = got
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(got.len(), expected.len());
        let max_diff = got
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < tol,
            "max abs diff {max_diff} exceeds tolerance {tol}"
        );
        Ok(())
    }

    const IDS: &[u32] = &[0, 2, 4, 1, 3, 2, 0, 4, 3, 1, 4, 2, 1, 3, 0];

    #[test]
    fn indexed_moe_matches_dequantized_reference() -> Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        for dtype in [
            GgmlDType::Q2K,
            GgmlDType::Q4_0,
            GgmlDType::Q4K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
        ] {
            let (experts, n, k) = (5usize, 48usize, 256usize);
            let fixture = fixture(dtype, experts, n, k, &device, None)?;
            let tokens = 5usize;
            let topk = 3usize;
            assert_eq!(IDS.len(), tokens * topk);
            let x_cpu = patterned(tokens, k, 91)?;
            let x = x_cpu.reshape((1, tokens, 1, 1, k))?.to_device(&device)?;
            let ids = Tensor::from_vec(IDS.to_vec(), (1, tokens, topk), &device)?;
            let out = fixture.method.gather_forward(&x, &ids)?;
            assert_eq!(out.dims(), &[1, tokens, topk, n]);
            let expected = expected(
                &fixture,
                IDS,
                &x_cpu.flatten_all()?.to_vec1::<f32>()?,
                topk,
                false,
                None,
            );
            assert_close(&out, &expected, 2e-3)?;
        }
        Ok(())
    }

    #[test]
    fn indexed_moe_rank3_and_decode_shapes() -> Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let (experts, n, k) = (4usize, 32usize, 256usize);
        let fixture = fixture(GgmlDType::Q8_0, experts, n, k, &device, None)?;

        // Rank-3 activations with rank-2 indices, several tokens.
        let tokens = 4usize;
        let topk = 2usize;
        let x_cpu = patterned(tokens, k, 13)?;
        let x = x_cpu.reshape((tokens, 1, k))?.to_device(&device)?;
        let ids = Tensor::from_vec(vec![3u32, 1, 0, 2, 2, 2, 1, 0], (tokens, topk), &device)?;
        let out = fixture.method.gather_forward_raw(&x, &ids)?;
        assert_eq!(out.dims(), &[tokens, topk, n]);
        let expected = expected(
            &fixture,
            &[3, 1, 0, 2, 2, 2, 1, 0],
            &x_cpu.flatten_all()?.to_vec1::<f32>()?,
            topk,
            false,
            None,
        );
        assert_close(&out, &expected, 2e-3)?;

        // Single-token decode.
        let x = x_cpu
            .narrow(0, 0, 1)?
            .reshape((1, 1, k))?
            .to_device(&device)?;
        let ids = Tensor::from_vec(vec![2u32, 0], (1, topk), &device)?;
        let out = fixture.method.gather_forward_raw(&x, &ids)?;
        assert_eq!(out.dims(), &[1, topk, n]);
        Ok(())
    }

    #[test]
    fn indexed_moe_per_pair_activations() -> Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let (experts, n, k) = (3usize, 24usize, 256usize);
        let fixture = fixture(GgmlDType::Q4K, experts, n, k, &device, None)?;
        let topk = 2usize;
        // One activation row per (token, expert) pair: xt == topk.
        let x_cpu = patterned(topk, k, 29)?;
        let x = x_cpu.reshape((1, 1, topk, k))?.to_device(&device)?;
        let ids = Tensor::from_vec(vec![1u32, 2], (1, 1, topk), &device)?;
        let out = fixture.method.gather_forward(&x, &ids)?;
        assert_eq!(out.dims(), &[1, 1, topk, n]);
        let expected = expected(
            &fixture,
            &[1, 2],
            &x_cpu.flatten_all()?.to_vec1::<f32>()?,
            topk,
            true,
            None,
        );
        assert_close(&out, &expected, 2e-3)?;
        Ok(())
    }

    #[test]
    fn indexed_moe_expert_bias_selected_per_id() -> Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let (experts, n, k) = (3usize, 24usize, 256usize);
        let bias_cpu = patterned(experts, n, 55)?;
        let bias = bias_cpu.to_device(&device)?;
        let fixture = fixture(GgmlDType::Q8_0, experts, n, k, &device, Some(bias))?;
        let tokens = 3usize;
        let topk = 2usize;
        let x_cpu = patterned(tokens, k, 61)?;
        let x = x_cpu.reshape((1, tokens, 1, 1, k))?.to_device(&device)?;
        let ids = Tensor::from_vec(vec![2u32, 0, 1, 2, 0, 1], (1, tokens, topk), &device)?;
        let out = fixture.method.gather_forward(&x, &ids)?;
        let expected = expected(
            &fixture,
            &[2, 0, 1, 2, 0, 1],
            &x_cpu.flatten_all()?.to_vec1::<f32>()?,
            topk,
            false,
            Some(&bias_cpu.flatten_all()?.to_vec1::<f32>()?),
        );
        assert_close(&out, &expected, 2e-3)?;
        Ok(())
    }

    #[test]
    fn stacked_expert_rejects_unsupported_dtype() {
        let Some(device) = metal_device() else {
            return;
        };
        let bytes = Tensor::from_vec(vec![0u8; 1024], (1024,), &device).unwrap();
        let err =
            GgufMetalExperts::new(GgmlDType::Q3K, vec![2, 4, 256], bytes, None, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("no bounded Metal indexed-MoE kernel"),
            "{err}"
        );
    }

    #[test]
    fn stacked_expert_rejects_unaligned_k() {
        let Some(device) = metal_device() else {
            return;
        };
        let bytes = Tensor::from_vec(vec![0u8; 16], (16,), &device).unwrap();
        let err =
            GgufMetalExperts::new(GgmlDType::Q2K, vec![1, 4, 128], bytes, None, None).unwrap_err();
        assert!(err.to_string().contains("divisible by 256"), "{err}");
        let bytes = Tensor::from_vec(vec![0u8; 16], (16,), &device).unwrap();
        let err =
            GgufMetalExperts::new(GgmlDType::Q4_0, vec![1, 4, 48], bytes, None, None).unwrap_err();
        assert!(err.to_string().contains("divisible by 32"), "{err}");
    }

    #[test]
    fn stacked_expert_rejects_out_of_range_ids() -> Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let fixture = fixture(GgmlDType::Q8_0, 2, 8, 256, &device, None)?;
        let x = patterned(1, 256, 3)?
            .reshape((1, 1, 256))?
            .to_device(&device)?;
        let ids = Tensor::from_vec(vec![99u32], (1, 1), &device)?;
        let err = fixture.method.gather_forward_raw(&x, &ids).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
        Ok(())
    }

    #[test]
    fn gguf_qtensor_metal_gather_fails_closed() -> Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        // A structurally-bound (non-direct) stacked expert tensor still takes the
        // owned GgufMatMul path; its gather must fail closed instead of
        // dequantizing the whole stack on the device.
        let w = patterned(3 * 8, 256, 3)?.reshape((3, 8, 256))?;
        let qt_cpu = candle_core::quantized::QTensor::quantize(&w, GgmlDType::Q3K)?;
        let storage =
            candle_core::quantized::QStorage::from_data(qt_cpu.data()?, &device, GgmlDType::Q3K)?;
        let qt = candle_core::quantized::QTensor::new(storage, vec![3, 8, 256])?;
        let method = GgufMatMul::from_qtensor(qt, None);
        let x = patterned(2, 256, 5)?.to_device(&device)?;
        let ids = Tensor::from_vec(vec![0u32, 1, 2, 0], (2, 2), &device)?;
        let err = method.gather_forward(&x, &ids).unwrap_err();
        assert!(err.to_string().contains("fully dequantize"), "{err}");
        Ok(())
    }
}
