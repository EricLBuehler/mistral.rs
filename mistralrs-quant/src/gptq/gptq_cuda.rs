// Portions of this file are adapted from the vLLM project
// (https://github.com/vllm-project/vllm)
// Licensed under the Apache License 2.0
// Copyright contributors to the vLLM project

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{
    cuda::{
        cudarc::{
            cublas::{result::hgemm, sys::cublasOperation_t},
            driver::{CudaSlice, DevicePtr},
        },
        CudaStorageSlice, WrapErr,
    },
    Context, CudaStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use half::f16;

use crate::{
    gptq::marlin_backend::{marlin_matmul, marlin_weight_repack},
    utils::get_cuda_device,
    DummyLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig,
    QuantizedSerde, ShardedVarBuilder,
};

use super::{
    ffi::{
        gemm_half_q_half_alt, gemm_half_q_half_cuda_part, reconstruct_exllama, reconstruct_gptq,
    },
    marlin_ffi::HAVE_MARLIN_KERNELS,
};

const MAX_Q_GEMM_ROWS_8BIT: i32 = 24;
const MAX_Q_GEMM_ROWS: i32 = 50;
const MAX_ALT_GEMM_ROWS: i32 = 8;
const BLOCK_M_SIZE_MAX: i32 = 8;

thread_local! {
    static ENGINE_TMP_DQS: RefCell<HashMap<usize, CudaSlice<f16>>> = RefCell::new(HashMap::new());
}

/// Get a temporary dequantization buffer for the current engine thread
fn get_tmp_dq_buffer(len: usize) -> Option<*mut f16> {
    ENGINE_TMP_DQS.with(|buffers| {
        buffers
            .borrow()
            .get(&len)
            .map(|slice| slice.device_ptr(slice.stream()).0 as *mut f16)
    })
}

/// Insert a temporary dequantization buffer for the current engine thread
fn insert_tmp_dq_buffer(len: usize, buffer: CudaSlice<f16>) {
    ENGINE_TMP_DQS.with(|buffers| {
        buffers.borrow_mut().insert(len, buffer);
    });
}

#[derive(Debug)]
pub struct GptqLayer {
    q_weight: Tensor,       // u32
    qzeros: Option<Tensor>, // u32
    scales: Tensor,         // f16
    bias: Option<Tensor>,   // f16
    g_idx: Option<Tensor>,  // i32
    bits: i32,
    use_exllama: bool,
    workspace: Option<Tensor>,
    is_marlin: bool,
    is_awq: bool,
}

impl GptqLayer {
    // https://github.com/vllm-project/vllm/blob/966fe72141e8365721840b7ababfb78601c23ead/csrc/quantization/gptq/q_gemm.cu#L1490
    // https://github.com/vllm-project/vllm/blob/966fe72141e8365721840b7ababfb78601c23ead/csrc/quantization/gptq/q_gemm.cu#L1823
    fn gptq_gemm(
        &self,
        a: Tensor,
        g_idx: &Tensor,
        qzeros: &Tensor,
        groups: i32,
        use_exllama: bool,
    ) -> Result<Tensor> {
        if !a.is_contiguous() {
            candle_core::bail!(
                "Expected `a` to be contiguous, got strides {:?}",
                a.layout().stride()
            )
        }

        let dev = get_cuda_device(&a)?;

        let cublas_handle = match a.device() {
            Device::Cuda(dev) => dev.cublas_handle(),
            _ => unreachable!(), // invariant enforced earlier
        };

        // a
        let (a, a_l) = a.storage_and_layout();
        let a = match &*a {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("a must be a cuda tensor"),
        };
        let (a, _a_guard) = crate::utils::slice_ptr(a.as_cuda_slice::<f16>()?, a_l.start_offset());
        let a_ptr = a as *const f16;

        // i32 -> u32 is intentional
        // qweight
        let (q_weight, _) = self.q_weight.storage_and_layout();
        let q_weight = match &*q_weight {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("q_weight must be a cuda tensor"),
        };
        let (q_weight, _q_weight_guard) = crate::utils::slice_ptr(
            q_weight.as_cuda_slice::<i32>()?,
            self.q_weight.layout().start_offset(),
        );
        let b_q_weight = q_weight as *const u32;

        // i32 -> u32 is intentional
        // qzeros
        let (qzeros, qzeros_l) = qzeros.storage_and_layout();
        let qzeros = match &*qzeros {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("qzeros must be a cuda tensor"),
        };
        let (qzeros, _qzeros_guard) =
            crate::utils::slice_ptr(qzeros.as_cuda_slice::<i32>()?, qzeros_l.start_offset());
        let b_qzeros = qzeros as *const u32;

        // scales
        let (scales, _) = self.scales.storage_and_layout();
        let scales = match &*scales {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("scales must be a cuda tensor"),
        };
        let (scales, _scales_guard) = crate::utils::slice_ptr(
            scales.as_cuda_slice::<f16>()?,
            self.scales.layout().start_offset(),
        );
        let b_scales = scales as *const f16;

        // g_idx
        let (g_idx, g_idx_l) = g_idx.storage_and_layout();
        let g_idx = match &*g_idx {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("g_idx must be a cuda tensor"),
        };
        let (g_idx, _g_idx_guard) =
            crate::utils::slice_ptr(g_idx.as_cuda_slice::<i32>()?, g_idx_l.start_offset());
        let b_g_idx = g_idx as *const i32;

        let c_shape = Shape::from_dims(&[a_l.dims()[0], self.q_weight.dims()[1]]);

        let (m, n, k) = (
            c_shape.dims()[0] as i32,
            c_shape.dims()[1] as i32,
            a_l.dims()[1] as i32,
        );

        let c = unsafe { dev.alloc::<f16>(c_shape.elem_count())? };

        let (c_ptr, c_guard) = c.device_ptr(c.stream());
        let c_ptr = c_ptr as *mut f16;

        let len = (self.q_weight.dims()[0] * 32 / self.bits as usize) * self.q_weight.dims()[1];
        let temp_dq_ptr =
            get_tmp_dq_buffer(len).expect("Temporary dequantization buffer not found");

        let use_reconstruct = if use_exllama {
            (self.bits == 8 && m > MAX_Q_GEMM_ROWS_8BIT) || (self.bits != 8 && m > MAX_Q_GEMM_ROWS)
        } else {
            // The 2/3-bit kernels are somehow slower than dequant + gemm baseline, so
            // we disabled them for now.
            self.bits < 4 || m > MAX_ALT_GEMM_ROWS
        };

        if use_reconstruct {
            // Reconstruct FP16 matrix, then cuBLAS
            let reconstruct_kernel = if use_exllama {
                reconstruct_exllama
            } else {
                reconstruct_gptq
            };
            unsafe {
                reconstruct_kernel(
                    b_q_weight,
                    b_qzeros,
                    b_scales,
                    b_g_idx,
                    temp_dq_ptr,
                    k,
                    n,
                    groups,
                    self.bits,
                )
            };

            let alpha = f16::from_f32_const(1.0);
            let beta = f16::from_f32_const(0.0);

            unsafe {
                hgemm(
                    *cublas_handle.handle(),
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha,
                    temp_dq_ptr as *const _,
                    n,
                    a_ptr as *const _,
                    k,
                    &beta,
                    c_ptr,
                    n,
                )
                .w()?
            };
        } else if use_exllama {
            let max_chunks = m / BLOCK_M_SIZE_MAX;
            let last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
            let last_chunk_size = m - last_chunk;

            if max_chunks > 0 {
                unsafe {
                    gemm_half_q_half_cuda_part(
                        a_ptr as *const _,
                        b_q_weight,
                        b_qzeros,
                        b_scales,
                        b_g_idx,
                        c_ptr,
                        last_chunk,
                        n,
                        k,
                        BLOCK_M_SIZE_MAX,
                        groups,
                        self.bits,
                    )
                }
            }
            if last_chunk_size > 0 {
                unsafe {
                    gemm_half_q_half_cuda_part(
                        a_ptr.add((last_chunk * k) as usize),
                        b_q_weight,
                        b_qzeros,
                        b_scales,
                        b_g_idx,
                        c_ptr.add((last_chunk * n) as usize),
                        last_chunk_size,
                        n,
                        k,
                        last_chunk_size,
                        groups,
                        self.bits,
                    )
                }
            }
        } else {
            unsafe {
                gemm_half_q_half_alt(
                    a_ptr as *const _,
                    b_q_weight,
                    b_qzeros,
                    b_scales,
                    b_g_idx,
                    c_ptr,
                    m,
                    n,
                    k,
                    self.bits,
                )
            }
        }

        drop(c_guard);

        let storage = CudaStorage {
            slice: CudaStorageSlice::F16(c),
            device: dev.clone(),
        };
        let storage = Storage::Cuda(storage);

        Ok(Tensor::from((storage, c_shape)))
    }
}

impl QuantMethod for GptqLayer {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::GptqAwq {
                bits,
                use_exllama,
                q_weight,
                qzeros,
                scales,
                g_idx,
                bias,
                workspace,
                is_marlin,
                is_awq,
            } => {
                if workspace.is_none() {
                    let dev = get_cuda_device(&q_weight)?;
                    let len = (q_weight.dims()[0] * 32 / bits as usize) * q_weight.dims()[1];
                    // SAFETY: used in the kernel as a tmp space, just preallocating it here.
                    if get_tmp_dq_buffer(len).is_none() {
                        let buffer = unsafe { dev.alloc::<f16>(len)? };
                        insert_tmp_dq_buffer(len, buffer);
                    }
                }
                Ok(Self {
                    q_weight,
                    qzeros,
                    scales,
                    g_idx,
                    bits,
                    use_exllama,
                    bias,
                    workspace,
                    is_marlin,
                    is_awq,
                })
            }
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => {
                unreachable!()
            }
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        // TODO
        candle_core::bail!("GptqLayer cannot be dequantized!");
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        // https://github.com/vllm-project/vllm/blob/ba991d5c84adbc0685075af88333c688ddb06011/vllm/model_executor/layers/quantization/gptq.py#L200
        let out_shape = Shape::from_dims(
            &[
                &a.dims()[..a.dims().len() - 1],
                &[self.q_weight.dim(D::Minus1)?],
            ]
            .concat(),
        );
        let reshaped_a = a.reshape(((), a.dim(D::Minus1)?))?;
        if !reshaped_a.device().is_cuda() {
            candle_core::bail!("Expected CUDA input to GptqLayer");
        }

        let out = match (self.g_idx.as_ref(), self.qzeros.as_ref(), self.is_marlin) {
            (Some(g_idx), Some(qzeros), false) => self
                .gptq_gemm(
                    reshaped_a,
                    g_idx,
                    qzeros,
                    qzeros.dim(0)? as i32,
                    self.use_exllama,
                )?
                .reshape(out_shape)?,
            (_, _, true) => marlin_matmul(
                a,
                &self.q_weight,
                &self.scales,
                &self.qzeros,
                self.workspace.as_ref().context("Workspace required")?,
                self.bits,
                self.is_awq,
            )?,
            _ => unreachable!(),
        };

        if let Some(bias) = &self.bias {
            out.broadcast_add(bias)
        } else {
            Ok(out)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F16)
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("GPTQ quantization does not support adding weight delta.")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.scales.dtype(), self.scales.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("GPTQ quantization does not support ISQ.")
    }
}

impl QuantizedSerde for GptqLayer {
    fn name(&self) -> &'static str {
        "gptq"
    }
}

macro_rules! pack_factor {
    ($bits:expr) => {
        32 / $bits
    };
}

pub fn gptq_linear(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let QuantizedConfig::GptqAwq {
        bits,
        group_size,
        checkpoint_format,
        is_awq,
    } = config
    else {
        candle_core::bail!("Unexpected quantization config.")
    };

    let is_awq = *is_awq;
    // Handle the case where the layer is dummy (no tensors)
    if !vb.contains_tensor("qweight")
        || !vb.contains_tensor("qzeros")
        || !vb.contains_tensor("scales")
        || !is_awq && !vb.contains_tensor("g_idx")
    {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

    let marlin_compatible = *bits == 4 || *bits == 8;
    let marlin_format = checkpoint_format
        .as_ref()
        .is_some_and(|fmt| fmt == "marlin")
        && HAVE_MARLIN_KERNELS;

    let qw_shape = if !is_awq {
        //quantized gptq (k/pack_factor, n) format
        (
            in_dim / pack_factor!(bits) / if marlin_format { 2 } else { 1 },
            out_dim * if marlin_format { 2 } else { 1 },
        )
    } else {
        //quantized awq (k, n/pack_factor) format
        (
            in_dim * if marlin_format { 2 } else { 1 },
            out_dim / pack_factor!(bits) / if marlin_format { 2 } else { 1 },
        )
    };

    let qweight = vb.get_with_hints_dtype(
        qw_shape,
        if marlin_format { "B" } else { "qweight" },
        Default::default(),
        DType::I32,
    )?;
    let scale_and_zero_size = in_dim / group_size;
    let scales = vb.get_with_hints_dtype(
        (scale_and_zero_size, out_dim),
        if marlin_format { "s" } else { "scales" },
        Default::default(),
        DType::F16,
    )?;
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get_with_hints_dtype((out_dim,), "bias", Default::default(), DType::F16)?)
    } else {
        None
    };
    let workspace = Tensor::zeros(out_dim / pack_factor!(bits), DType::U32, vb.device())?;

    let config = if marlin_format {
        QuantMethodConfig::GptqAwq {
            bits: *bits as i32,
            use_exllama: false,
            q_weight: qweight,
            qzeros: None,
            scales,
            g_idx: None,
            bias,
            workspace: Some(workspace),
            is_marlin: true,
            is_awq,
        }
    } else {
        fn get_scale_perms() -> (Vec<u32>, Vec<u32>) {
            let mut scale_perm: Vec<u32> = Vec::new();
            for i in 0..8 {
                scale_perm.extend((0..8).map(|j| i + 8 * j));
            }
            let mut scale_perm_single: Vec<u32> = Vec::new();
            for i in 0..4 {
                scale_perm_single.extend([0, 1, 8, 9, 16, 17, 24, 25].iter().map(|&j| 2 * i + j));
            }
            (scale_perm, scale_perm_single)
        }

        fn marlin_permute_scales(
            s: &Tensor,
            size_k: usize,
            size_n: usize,
            group_size: i32,
            _num_bits: u32,
        ) -> Result<Tensor> {
            let (scale_perm, scale_perm_single) = get_scale_perms();
            let s = if (group_size as usize) < size_k && group_size != -1 {
                let s = s.reshape(((), scale_perm.len()))?;
                let scale_perm_tensor =
                    Tensor::from_slice(&scale_perm, scale_perm.len(), s.device())?;
                s.index_select(&scale_perm_tensor, 1)?
            } else {
                let s = s.reshape(((), scale_perm_single.len()))?;
                let scale_perm_single_tensor =
                    Tensor::from_slice(&scale_perm_single, scale_perm_single.len(), s.device())?;
                s.index_select(&scale_perm_single_tensor, 1)?
            };

            let s = s.reshape(((), size_n))?.contiguous()?;
            Ok(s)
        }

        let qzeros = vb.get_with_hints_dtype(
            (scale_and_zero_size, out_dim / pack_factor!(bits)),
            "qzeros",
            Default::default(),
            DType::I32,
        )?;

        let (g_idx, perm) = if is_awq {
            (None, None)
        } else {
            let g_idx =
                vb.get_with_hints_dtype((in_dim,), "g_idx", Default::default(), DType::I32)?;
            let perm = g_idx
                .to_device(&Device::Cpu)?
                .arg_sort_last_dim(true)?
                .to_device(g_idx.device())?;
            (Some(g_idx), Some(perm))
        };

        // Repack to marlin format
        let qweight = if marlin_compatible {
            marlin_weight_repack(&qweight, &perm, in_dim, *bits as i32, is_awq)?
        } else {
            qweight
        };

        let scales = if marlin_compatible {
            marlin_permute_scales(
                &scales,
                in_dim / pack_factor!(bits),
                out_dim,
                *group_size as i32,
                *bits as u32,
            )?
        } else {
            scales
        };
        let workspace = if marlin_compatible {
            Some(workspace)
        } else {
            None
        };

        QuantMethodConfig::GptqAwq {
            bits: *bits as i32,
            use_exllama: false,
            q_weight: qweight,
            qzeros: Some(qzeros),
            scales,
            g_idx,
            bias,
            workspace,
            is_marlin: marlin_compatible,
            is_awq,
        }
    };
    Ok(Arc::new(GptqLayer::new(config)?))
}
