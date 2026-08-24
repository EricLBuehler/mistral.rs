use std::sync::Arc;

use candle_core::{shape::Dim, DType, Result, Tensor, D};

#[cfg(feature = "cuda")]
use crate::cuda::ffi;
use crate::layers::Activation;
#[cfg(feature = "cuda")]
use candle_core::Shape;

#[cfg(feature = "cuda")]
const CUDA_TOPK_CHUNK_SIZE: usize = 2048;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOPK_MAX_EXACT_PACKED_VOCAB: usize = (1 << 24) + 1;
#[cfg(feature = "cuda")]
const CUDA_TOPK_MAX_GRID_Y: usize = 65_535;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOPK_MAX_K: usize = 128;
#[cfg(feature = "cuda")]
const CUDA_TOPK_MAX_STAGE2_CANDIDATES: usize = 47 * 1024;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_CATEGORICAL_PACKED_WIDTH: usize = 2;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOP1_PACKED_WIDTH: usize = 2;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOP1_INVALID_TOKEN: u32 = u32::MAX;
#[cfg(feature = "cuda")]
const CUDA_TOP1_RING_SLOTS: usize = 2;
#[cfg(feature = "cuda")]
pub(crate) const CUDA_DFLASH_SELECTOR_MAX_K: usize = 128;
#[cfg(all(feature = "cuda", test))]
const CUDA_DFLASH_SELECTOR_INVALID_TOKEN: u32 = u32::MAX;
#[cfg(feature = "cuda")]
const CUDA_DFLASH_SELECTOR_F32: i32 = 0;
#[cfg(feature = "cuda")]
const CUDA_DFLASH_SELECTOR_BF16: i32 = 1;

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_max_k(vocab: usize) -> Option<usize> {
    if vocab == 0 || vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        return None;
    }
    let chunks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_bound = CUDA_TOPK_MAX_STAGE2_CANDIDATES.checked_div(chunks)?;
    let max_k = vocab.min(CUDA_TOPK_MAX_K).min(workspace_bound);
    (max_k > 0).then_some(max_k)
}

// ============================================================================
// Optimized parallel topk for CUDA
// Uses a dedicated kernel that's much faster than full sort for small k
// Single kernel call writes both values and indices - no post-processing needed
// ============================================================================

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
fn cuda_topk(input: &Tensor, k: usize) -> Result<TopKOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    let input = final_logits_row(input)?;
    let dims = input.dims();
    let ncols = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".to_string()))?;
    let nrows = (input.elem_count() / ncols) as i32;
    let ncols_i32 = ncols as i32;
    let k_i32 = k as i32;

    // Output shapes
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = k;
    let out_elem_count = nrows as usize * k;

    let (storage, _layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::BF16(inp) => inp.device_ptr(&stream),
        CudaStorageSlice::F16(inp) => inp.device_ptr(&stream),
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk only supports BF16/F16/F32"),
    };
    let src_ptr = src_ptr as *const c_void;

    // Allocate both output buffers
    let mut indices_dst = unsafe { dev.alloc::<u32>(out_elem_count) }?;
    let (indices_ptr, indices_guard) = indices_dst.device_ptr_mut(&stream);

    let (values_tensor, indices_tensor) = match input.dtype() {
        DType::BF16 => {
            let mut values_dst = unsafe { dev.alloc::<half::bf16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_bf16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::BF16(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        DType::F16 => {
            let mut values_dst = unsafe { dev.alloc::<half::f16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_f16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F16(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        DType::F32 => {
            let mut values_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_f32(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        dt => candle_core::bail!("cuda_topk unsupported dtype: {:?}", dt),
    };

    Ok(TopKOutput {
        values: values_tensor,
        indices: indices_tensor,
    })
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum MoeRouterScoreFunction {
    Raw,
    Softmax,
    Sigmoid,
}

#[cfg(feature = "cuda")]
impl MoeRouterScoreFunction {
    const fn as_i32(self) -> i32 {
        match self {
            Self::Raw => 0,
            Self::Softmax => 1,
            Self::Sigmoid => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum MoeRouterSelectedWeight {
    Score,
    Softmax,
    Sigmoid,
}

#[cfg(feature = "cuda")]
impl MoeRouterSelectedWeight {
    const fn as_i32(self) -> i32 {
        match self {
            Self::Score => 0,
            Self::Softmax => 1,
            Self::Sigmoid => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MoeRouterTopKConfig {
    pub top_k: usize,
    pub score_function: MoeRouterScoreFunction,
    pub selected_weight: MoeRouterSelectedWeight,
    pub renormalize: bool,
    pub norm_min: f32,
    pub output_scale: f32,
    pub logit_clip: Option<(f32, f32)>,
}

pub fn moe_router_topk(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    #[cfg(feature = "cuda")]
    if let Some(topk) =
        cuda_moe_router_topk_if_supported(logits, config, selection_bias, expert_scale)?
    {
        return Ok(topk);
    }

    let logits = logits.to_dtype(DType::F32)?;
    let logits = match config.logit_clip {
        Some((min, max)) => logits.clamp(min as f64, max as f64)?,
        None => logits,
    };
    let scores = match config.score_function {
        MoeRouterScoreFunction::Raw => logits.clone(),
        MoeRouterScoreFunction::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
        MoeRouterScoreFunction::Sigmoid => candle_nn::ops::sigmoid(&logits)?,
    };
    let selection_scores = if let Some(selection_bias) = selection_bias {
        scores.broadcast_add(&selection_bias.to_dtype(DType::F32)?)?
    } else {
        scores.clone()
    };
    let indices = selection_scores
        .topk(config.top_k)?
        .indices
        .to_dtype(DType::U32)?;
    let selected_logits = match config.selected_weight {
        MoeRouterSelectedWeight::Score => None,
        MoeRouterSelectedWeight::Softmax | MoeRouterSelectedWeight::Sigmoid => {
            Some(logits.gather(&indices, D::Minus1)?)
        }
    };
    let mut values = match config.selected_weight {
        MoeRouterSelectedWeight::Score => scores.gather(&indices, D::Minus1)?,
        MoeRouterSelectedWeight::Softmax => {
            candle_nn::ops::softmax_last_dim(selected_logits.as_ref().unwrap())?
        }
        MoeRouterSelectedWeight::Sigmoid => {
            candle_nn::ops::sigmoid(selected_logits.as_ref().unwrap())?
        }
    };

    if config.renormalize {
        let denominator = values.sum_keepdim(D::Minus1)?;
        let denominator = if config.norm_min > 0.0 {
            let min = Tensor::full(config.norm_min, denominator.shape(), denominator.device())?;
            denominator.broadcast_maximum(&min)?
        } else {
            denominator
        };
        values = values.broadcast_div(&denominator)?;
    }
    if config.output_scale != 1.0 {
        values = (values * config.output_scale as f64)?;
    }
    if let Some(expert_scale) = expert_scale {
        let scales = expert_scale
            .to_dtype(DType::F32)?
            .index_select(&indices.flatten_all()?, 0)?
            .reshape(indices.shape())?;
        values = (values * scales)?;
    }

    Ok(TopKOutput { values, indices })
}

#[cfg(feature = "cuda")]
const MOE_ROUTER_MAX_POWER_OF_TWO_EXPERTS: usize = 512;

#[cfg(feature = "cuda")]
const MOE_ROUTER_EXTRA_EXPERT_COUNTS: &[usize] = &[576];

#[cfg(feature = "cuda")]
pub fn cuda_moe_router_topk_supports_experts(n_experts: usize) -> bool {
    (n_experts.is_power_of_two() && n_experts <= MOE_ROUTER_MAX_POWER_OF_TWO_EXPERTS)
        || MOE_ROUTER_EXTRA_EXPERT_COUNTS.contains(&n_experts)
}

#[cfg(feature = "cuda")]
pub fn cuda_moe_router_topk_if_supported(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<Option<TopKOutput>> {
    if !logits.device().is_cuda() {
        return Ok(None);
    }
    let n_experts = match logits.dims().last() {
        Some(n_experts) => *n_experts,
        None => return Ok(None),
    };
    if !cuda_moe_router_topk_supports_experts(n_experts) {
        return Ok(None);
    }
    cuda_moe_router_topk(logits, config, selection_bias, expert_scale).map(Some)
}

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_moe_router_topk(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    let logits = logits.contiguous()?;
    let dims = logits.dims();
    let n_experts = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".to_string()))?;
    if config.top_k == 0 || config.top_k > n_experts {
        candle_core::bail!(
            "cuda_moe_router_topk top_k={} must be in [1, {}]",
            config.top_k,
            n_experts
        );
    }
    if !cuda_moe_router_topk_supports_experts(n_experts) {
        candle_core::bail!("cuda_moe_router_topk unsupported expert count {n_experts}");
    }

    let selection_bias = selection_bias.map(Tensor::contiguous).transpose()?;
    if let Some(selection_bias) = &selection_bias {
        if selection_bias.dtype() != DType::F32 || selection_bias.elem_count() != n_experts {
            candle_core::bail!("cuda_moe_router_topk selection_bias must be F32 [n_experts]");
        }
    }

    let expert_scale = expert_scale.map(Tensor::contiguous).transpose()?;
    if let Some(expert_scale) = &expert_scale {
        if expert_scale.dtype() != DType::F32 || expert_scale.elem_count() != n_experts {
            candle_core::bail!("cuda_moe_router_topk expert_scale must be F32 [n_experts]");
        }
    }
    let selection_bias_storage_and_layout = selection_bias.as_ref().map(|t| t.storage_and_layout());
    let expert_scale_storage_and_layout = expert_scale.as_ref().map(|t| t.storage_and_layout());

    let nrows = logits.elem_count() / n_experts;
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = config.top_k;
    let out_elem_count = nrows * config.top_k;

    let (logits_storage, _logits_layout) = logits.storage_and_layout();
    let logits_storage = match &*logits_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_moe_router_topk requires CUDA logits"),
    };

    let dev = logits_storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let mut weights_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
    let mut ids_dst = unsafe { dev.alloc::<u32>(out_elem_count) }?;
    let (weights_ptr, weights_guard) = weights_dst.device_ptr_mut(&stream);
    let (ids_ptr, ids_guard) = ids_dst.device_ptr_mut(&stream);

    let (clip_min, clip_max, clamp_logits) = match config.logit_clip {
        Some((min, max)) => (min, max, true),
        None => (0.0, 0.0, false),
    };

    macro_rules! launch {
        ($variant:ident, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(logits_src) = &logits_storage.slice else {
                candle_core::bail!("cuda_moe_router_topk logits dtype mismatch");
            };
            let (logits_ptr, _logits_guard) = logits_src.device_ptr(&stream);

            let (selection_bias_ptr, _selection_bias_guard) = if let Some((storage, _layout)) =
                &selection_bias_storage_and_layout
            {
                let storage = match &**storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!("cuda_moe_router_topk requires CUDA selection_bias"),
                };
                let CudaStorageSlice::F32(src) = &storage.slice else {
                    candle_core::bail!("cuda_moe_router_topk selection_bias dtype mismatch");
                };
                let (ptr, guard) = src.device_ptr(&stream);
                (ptr as *const c_void, Some(guard))
            } else {
                (std::ptr::null(), None)
            };

            let (expert_scale_ptr, _expert_scale_guard) =
                if let Some((storage, _layout)) = &expert_scale_storage_and_layout {
                    let storage = match &**storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => candle_core::bail!("cuda_moe_router_topk requires CUDA expert_scale"),
                    };
                    let CudaStorageSlice::F32(src) = &storage.slice else {
                        candle_core::bail!("cuda_moe_router_topk expert_scale dtype mismatch");
                    };
                    let (ptr, guard) = src.device_ptr(&stream);
                    (ptr as *const c_void, Some(guard))
                } else {
                    (std::ptr::null(), None)
                };

            unsafe {
                ffi::$ffi_fn(
                    logits_ptr as *const c_void,
                    weights_ptr as *mut c_void,
                    ids_ptr as *mut c_void,
                    selection_bias_ptr,
                    expert_scale_ptr,
                    nrows as i32,
                    n_experts as i32,
                    config.top_k as i32,
                    config.score_function.as_i32(),
                    config.selected_weight.as_i32(),
                    config.renormalize,
                    clamp_logits,
                    clip_min,
                    clip_max,
                    config.norm_min,
                    config.output_scale,
                    stream_raw,
                );
            }
        }};
    }

    match logits.dtype() {
        DType::BF16 => launch!(BF16, moe_router_topk_bf16),
        DType::F16 => launch!(F16, moe_router_topk_f16),
        DType::F32 => launch!(F32, moe_router_topk_f32),
        dt => candle_core::bail!("cuda_moe_router_topk unsupported dtype: {:?}", dt),
    }

    drop(weights_guard);
    drop(ids_guard);

    let weights_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(weights_dst),
        device: dev.clone(),
    };
    let ids_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::U32(ids_dst),
        device: dev.clone(),
    };

    Ok(TopKOutput {
        values: Tensor::from((
            candle_core::Storage::Cuda(weights_storage),
            Shape::from_dims(&out_dims),
        )),
        indices: Tensor::from((
            candle_core::Storage::Cuda(ids_storage),
            Shape::from_dims(&out_dims),
        )),
    })
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_topk_logits_f32(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("cuda_topk_logits_f32 requires a positive finite temperature");
    }

    let input = input.contiguous()?;
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32 requires F32 logits");
    }

    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("cuda_topk_logits_f32 got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!(
            "cuda_topk_logits_f32 k={} must be in [1, {}]",
            k,
            CUDA_TOPK_MAX_K
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let stage2_candidates = nblocks * k;
    if stage2_candidates > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!(
            "cuda_topk_logits_f32 workspace too large: {} candidates",
            stage2_candidates
        );
    }

    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_logits_f32 requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk_logits_f32 only supports F32"),
    };
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };

    let workspace_elems = nblocks * k;
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut block_maxes = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut values_dst = unsafe { dev.alloc::<f32>(k) }?;
    let mut indices_dst = unsafe { dev.alloc::<u32>(k) }?;
    let mut softmax_info_dst = unsafe { dev.alloc::<f32>(2) }?;

    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);
    let (indices_ptr, indices_guard) = indices_dst.device_ptr_mut(&stream);
    let (softmax_info_ptr, softmax_info_guard) = softmax_info_dst.device_ptr_mut(&stream);

    unsafe {
        ffi::topk_large_f32(
            src_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            values_ptr as *mut f32,
            indices_ptr as *mut u32,
            softmax_info_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CUDA_TOPK_CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream_raw,
        );
    }

    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(values_guard);
    drop(indices_guard);
    drop(softmax_info_guard);

    let values_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(values_dst),
        device: dev.clone(),
    };
    let indices_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::U32(indices_dst),
        device: dev.clone(),
    };
    let softmax_info_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(softmax_info_dst),
        device: dev.clone(),
    };
    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_maxes),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
    ];

    Ok(TopKLogitsOutput {
        values: Tensor::from((
            candle_core::Storage::Cuda(values_storage),
            Shape::from_dims(&[k]),
        )),
        indices: Tensor::from((
            candle_core::Storage::Cuda(indices_storage),
            Shape::from_dims(&[k]),
        )),
        softmax_info: Tensor::from((
            candle_core::Storage::Cuda(softmax_info_storage),
            Shape::from_dims(&[2]),
        )),
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_topk_logits_f32_packed(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("cuda_topk_logits_f32_packed requires a positive finite temperature");
    }

    let input = input.contiguous()?;
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32_packed requires F32 logits");
    }

    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("cuda_topk_logits_f32_packed got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed k={} must be in [1, {}]",
            k,
            CUDA_TOPK_MAX_K
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let stage2_candidates = nblocks * k;
    if stage2_candidates > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed workspace too large: {} candidates",
            stage2_candidates
        );
    }

    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_logits_f32_packed requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk_logits_f32_packed only supports F32"),
    };
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };

    let workspace_elems = nblocks * k;
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut block_maxes = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(2 * k + 2) }?;

    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);

    unsafe {
        ffi::topk_large_f32_packed(
            src_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CUDA_TOPK_CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream_raw,
        );
    }

    drop(src_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(packed_guard);

    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };
    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_maxes),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
    ];

    Ok(TopKLogitsPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[2 * k + 2]),
        )),
        k,
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTopKLogitsPackedWorkspace {
    location: candle_core::DeviceLocation,
    stream: Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    capacity_rows: usize,
    vocab: usize,
    capacity_k: usize,
    nblocks: usize,
    #[cfg(test)]
    id: u64,
    block_values: Tensor,
    block_indices: Tensor,
    block_maxes: Tensor,
    block_sums: Tensor,
    packed: Tensor,
}

#[cfg(all(feature = "cuda", test))]
fn cuda_topk_logits_packed_workspace_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
impl CudaTopKLogitsPackedWorkspace {
    fn new(
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> Result<Self> {
        use candle_core::backend::BackendDevice;

        let capacity_rows = rows
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k row capacity overflow"))?;
        let capacity_k = k
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k width capacity overflow"))?;
        let workspace_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k workspace overflow"))?;
        let block_elems = capacity_rows
            .checked_mul(nblocks)
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k block workspace overflow"))?;
        let packed_width = capacity_k
            .checked_mul(2)
            .and_then(|width| width.checked_add(2))
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k packed width overflow"))?;
        let packed_elems = capacity_rows
            .checked_mul(packed_width)
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k packed workspace overflow"))?;
        let device = candle_core::Device::Cuda(dev.clone());
        Ok(Self {
            location: dev.location(),
            stream: dev.cuda_stream(),
            capacity_rows,
            vocab,
            capacity_k,
            nblocks,
            #[cfg(test)]
            id: cuda_topk_logits_packed_workspace_id(),
            block_values: Tensor::zeros(workspace_elems, DType::F32, &device)?,
            block_indices: Tensor::zeros(workspace_elems, DType::U32, &device)?,
            block_maxes: Tensor::zeros(block_elems, DType::F32, &device)?,
            block_sums: Tensor::zeros(block_elems, DType::F32, &device)?,
            packed: Tensor::zeros(packed_elems, DType::F32, &device)?,
        })
    }

    fn can_hold(
        &self,
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> bool {
        use candle_core::backend::BackendDevice;

        let stream = dev.cuda_stream();
        self.location == dev.location()
            && Arc::ptr_eq(self.stream.context(), stream.context())
            && self.stream.cu_stream() == stream.cu_stream()
            && self.capacity_rows >= rows
            && self.vocab == vocab
            && self.capacity_k >= k
            && self.nblocks == nblocks
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_packed_batched(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
) -> Result<TopKLogitsPackedOutput> {
    let mut workspace = None;
    cuda_topk_logits_packed_batched_with_workspace(input, k, inverse_temperatures, &mut workspace)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_packed_batched_with_workspace(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
    cache: &mut Option<CudaTopKLogitsPackedWorkspace>,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_topk_logits_packed_batched";

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{OP} requires BF16, F16, or F32 logits");
    }
    if inverse_temperatures.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 inverse temperatures");
    }
    if !input.is_contiguous() || !inverse_temperatures.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    if !input.device().same_device(inverse_temperatures.device()) {
        candle_core::bail!("{OP} tensors must be on the same CUDA device");
    }

    let vocab =
        input.dims().last().copied().ok_or_else(|| {
            candle_core::Error::Msg(format!("{OP} requires logits with rank >= 1"))
        })?;
    if vocab == 0 {
        candle_core::bail!("{OP} got an empty vocabulary");
    }
    let batch = input.elem_count() / vocab;
    if batch == 0 {
        candle_core::bail!("{OP} got an empty batch");
    }
    if inverse_temperatures.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected inverse temperatures with shape [{batch}], got {:?}",
            inverse_temperatures.dims()
        );
    }
    let k = k.min(vocab);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(vocab));
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if vocab > i32::MAX as usize {
        candle_core::bail!("{OP} vocabulary is too large: {vocab}");
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let candidates_per_row = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate count overflow")))?;
    if candidates_per_row > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {candidates_per_row} candidates per row");
    }
    let workspace_elems = batch
        .checked_mul(candidates_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate workspace overflow")))?;
    let block_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} block workspace overflow")))?;
    let packed_width = k
        .checked_mul(2)
        .and_then(|width| width.checked_add(2))
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let packed_elems = batch
        .checked_mul(packed_width)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed output overflow")))?;

    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA inverse temperatures"),
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} only supports F32 inverse temperatures");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let needs_alloc = cache
        .as_ref()
        .is_none_or(|workspace| !workspace.can_hold(dev, batch, vocab, k, nblocks));
    if needs_alloc {
        *cache = Some(CudaTopKLogitsPackedWorkspace::new(
            dev, batch, vocab, k, nblocks,
        )?);
    }
    let workspace = cache
        .as_ref()
        .expect("CUDA top-k workspace was allocated above");
    let block_values = workspace.block_values.narrow(0, 0, workspace_elems)?;
    let block_indices = workspace.block_indices.narrow(0, 0, workspace_elems)?;
    let block_maxes = workspace.block_maxes.narrow(0, 0, block_elems)?;
    let block_sums = workspace.block_sums.narrow(0, 0, block_elems)?;
    let packed_dst = workspace.packed.narrow(0, 0, packed_elems)?;

    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr =
                unsafe { (ptr as *const $ty).add(input_layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &input_storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => candle_core::bail!("{OP} logits dtype mismatch"),
    };
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let (block_values_storage, block_values_layout) = block_values.storage_and_layout();
    let candle_core::Storage::Cuda(block_values_storage) = &*block_values_storage else {
        unreachable!("CUDA top-k workspace values are CUDA")
    };
    let CudaStorageSlice::F32(block_values_slice) = &block_values_storage.slice else {
        unreachable!("CUDA top-k workspace values are F32")
    };
    let (block_values_ptr, block_values_guard) = block_values_slice.device_ptr(&stream);
    let block_values_ptr =
        unsafe { (block_values_ptr as *mut f32).add(block_values_layout.start_offset()) };
    let (block_indices_storage, block_indices_layout) = block_indices.storage_and_layout();
    let candle_core::Storage::Cuda(block_indices_storage) = &*block_indices_storage else {
        unreachable!("CUDA top-k workspace indices are CUDA")
    };
    let CudaStorageSlice::U32(block_indices_slice) = &block_indices_storage.slice else {
        unreachable!("CUDA top-k workspace indices are U32")
    };
    let (block_indices_ptr, block_indices_guard) = block_indices_slice.device_ptr(&stream);
    let block_indices_ptr =
        unsafe { (block_indices_ptr as *mut u32).add(block_indices_layout.start_offset()) };
    let (block_maxes_storage, block_maxes_layout) = block_maxes.storage_and_layout();
    let candle_core::Storage::Cuda(block_maxes_storage) = &*block_maxes_storage else {
        unreachable!("CUDA top-k workspace maxima are CUDA")
    };
    let CudaStorageSlice::F32(block_maxes_slice) = &block_maxes_storage.slice else {
        unreachable!("CUDA top-k workspace maxima are F32")
    };
    let (block_maxes_ptr, block_maxes_guard) = block_maxes_slice.device_ptr(&stream);
    let block_maxes_ptr =
        unsafe { (block_maxes_ptr as *mut f32).add(block_maxes_layout.start_offset()) };
    let (block_sums_storage, block_sums_layout) = block_sums.storage_and_layout();
    let candle_core::Storage::Cuda(block_sums_storage) = &*block_sums_storage else {
        unreachable!("CUDA top-k workspace sums are CUDA")
    };
    let CudaStorageSlice::F32(block_sums_slice) = &block_sums_storage.slice else {
        unreachable!("CUDA top-k workspace sums are F32")
    };
    let (block_sums_ptr, block_sums_guard) = block_sums_slice.device_ptr(&stream);
    let block_sums_ptr =
        unsafe { (block_sums_ptr as *mut f32).add(block_sums_layout.start_offset()) };
    let (packed_storage, packed_layout) = packed_dst.storage_and_layout();
    let candle_core::Storage::Cuda(packed_storage) = &*packed_storage else {
        unreachable!("CUDA top-k packed workspace is CUDA")
    };
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        unreachable!("CUDA top-k packed workspace is F32")
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *mut f32).add(packed_layout.start_offset()) };
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };

    macro_rules! launch {
        ($kernel:path, $input:expr) => {{
            unsafe {
                $kernel(
                    $input,
                    temperature_ptr,
                    block_values_ptr,
                    block_indices_ptr,
                    block_maxes_ptr,
                    block_sums_ptr,
                    packed_ptr,
                    nrows_i32,
                    ncols_i32,
                    k_i32,
                    chunk_size_i32,
                    nblocks_i32,
                    stream.cu_stream() as i64,
                );
            }
        }};
    }
    match input.dtype() {
        DType::F32 => launch!(ffi::topk_large_f32_packed_batched, input_ptr.cast::<f32>()),
        DType::BF16 => launch!(ffi::topk_large_bf16_packed_batched, input_ptr),
        DType::F16 => launch!(ffi::topk_large_f16_packed_batched, input_ptr),
        _ => unreachable!(),
    }

    drop(input_guard);
    drop(temperature_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(packed_guard);
    Ok(TopKLogitsPackedOutput {
        packed: packed_dst.reshape((batch, packed_width))?,
        k,
        _workspace: vec![
            block_values.clone(),
            block_indices.clone(),
            block_maxes.clone(),
            block_sums.clone(),
        ],
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_batched(
    input: &Tensor,
    k: usize,
) -> Result<RankedTopKPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_topk_ranked_packed_batched";

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{OP} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    let vocab =
        input.dims().last().copied().ok_or_else(|| {
            candle_core::Error::Msg(format!("{OP} requires logits with rank >= 1"))
        })?;
    if vocab == 0 {
        candle_core::bail!("{OP} got an empty vocabulary");
    }
    let batch = input.elem_count() / vocab;
    if batch == 0 {
        candle_core::bail!("{OP} got an empty batch");
    }
    let k = k.min(vocab);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(vocab));
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if vocab > i32::MAX as usize {
        candle_core::bail!("{OP} vocabulary is too large: {vocab}");
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let candidates_per_row = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate count overflow")))?;
    if candidates_per_row > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {candidates_per_row} candidates per row");
    }
    let workspace_elems = batch
        .checked_mul(candidates_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate workspace overflow")))?;
    let radix_state_words_per_row = unsafe { ffi::topk_large_ranked_state_words_per_row() };
    let radix_state_elems = batch
        .checked_mul(radix_state_words_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} radix workspace overflow")))?;
    let value_workspace_elems = workspace_elems.max(radix_state_elems);
    let packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let packed_elems = batch
        .checked_mul(packed_width)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed output overflow")))?;

    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut block_values = dev.alloc_zeros::<f32>(value_workspace_elems)?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(packed_elems) }?;

    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr =
                unsafe { (ptr as *const $ty).add(input_layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &input_storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => candle_core::bail!("{OP} logits dtype mismatch"),
    };
    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);

    macro_rules! launch {
        ($kernel:path, $input:expr) => {{
            unsafe {
                $kernel(
                    $input,
                    block_values_ptr as *mut f32,
                    block_indices_ptr as *mut u32,
                    packed_ptr as *mut f32,
                    nrows_i32,
                    ncols_i32,
                    k_i32,
                    chunk_size_i32,
                    nblocks_i32,
                    stream.cu_stream() as i64,
                )
            }
        }};
    }
    let status = match input.dtype() {
        DType::F32 => launch!(
            ffi::topk_large_ranked_f32_packed_batched,
            input_ptr.cast::<f32>()
        ),
        DType::BF16 => launch!(ffi::topk_large_ranked_bf16_packed_batched, input_ptr),
        DType::F16 => launch!(ffi::topk_large_ranked_f16_packed_batched, input_ptr),
        _ => unreachable!(),
    };

    drop(input_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(packed_guard);
    if status != 0 {
        candle_core::bail!("{OP} CUDA launch failed with status {status}");
    }

    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[value_workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks, k]),
        )),
    ];
    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };

    Ok(RankedTopKPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[batch, packed_width]),
        )),
        k,
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_f32_packed_batched(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
) -> Result<TopKLogitsPackedOutput> {
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32_packed_batched requires F32 logits");
    }
    cuda_topk_logits_packed_batched(input, k, inverse_temperatures)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_dflash_greedy_select(
    topk: &RankedTopKPackedOutput,
    projected_hidden: &Tensor,
    predecessor_codebook: &Tensor,
    successor_codebook: &Tensor,
    anchors: &Tensor,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_dflash_greedy_select";
    let packed_topk = &topk.packed;
    let k = topk.k;

    let [rows, packed_width] = packed_topk.dims() else {
        candle_core::bail!("{OP} expected packed top-k with shape [batch * positions, 2 * k]");
    };
    let [hidden_rows, rank] = projected_hidden.dims() else {
        candle_core::bail!(
            "{OP} expected projected hidden states with shape [batch * positions, rank]"
        );
    };
    let [predecessor_vocab, predecessor_rank] = predecessor_codebook.dims() else {
        candle_core::bail!("{OP} expected predecessor codebook with shape [vocab, rank]");
    };
    let [successor_vocab, successor_rank] = successor_codebook.dims() else {
        candle_core::bail!("{OP} expected successor codebook with shape [vocab, rank]");
    };
    let [batch] = anchors.dims() else {
        candle_core::bail!("{OP} expected anchors with shape [batch]");
    };
    let (rows, packed_width, hidden_rows, rank) = (*rows, *packed_width, *hidden_rows, *rank);
    let (predecessor_vocab, predecessor_rank) = (*predecessor_vocab, *predecessor_rank);
    let (successor_vocab, successor_rank, batch) = (*successor_vocab, *successor_rank, *batch);

    if rows == 0 || batch == 0 || rank == 0 || predecessor_vocab == 0 {
        candle_core::bail!("{OP} does not support empty inputs");
    }
    if rows % batch != 0 {
        candle_core::bail!("{OP} row count {rows} is not divisible by batch size {batch}");
    }
    if hidden_rows != rows {
        candle_core::bail!("{OP} expected {rows} projected hidden rows, got {hidden_rows}");
    }
    if predecessor_vocab != successor_vocab || predecessor_rank != rank || successor_rank != rank {
        candle_core::bail!(
            "{OP} codebook shapes {:?} and {:?} do not match hidden rank {rank}",
            predecessor_codebook.dims(),
            successor_codebook.dims()
        );
    }
    let expected_packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    if packed_width != expected_packed_width {
        candle_core::bail!(
            "{OP} expected rank-only packed top-k width {expected_packed_width}, got {packed_width}"
        );
    }
    if k == 0 || k > CUDA_DFLASH_SELECTOR_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {CUDA_DFLASH_SELECTOR_MAX_K}]");
    }
    if predecessor_vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {predecessor_vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if packed_topk.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 packed top-k values");
    }
    if anchors.dtype() != DType::U32 {
        candle_core::bail!("{OP} requires U32 anchors");
    }
    for (name, tensor) in [
        ("projected hidden states", projected_hidden),
        ("predecessor codebook", predecessor_codebook),
        ("successor codebook", successor_codebook),
    ] {
        if !matches!(tensor.dtype(), DType::BF16 | DType::F32) {
            candle_core::bail!("{OP} requires BF16 or F32 {name}");
        }
    }
    for tensor in [
        packed_topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
    ] {
        if !tensor.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: OP });
        }
        if !packed_topk.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} tensors must be on the same CUDA device");
        }
    }

    let positions = rows / batch;
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let positions_i32 = i32::try_from(positions).map_err(candle_core::Error::wrap)?;
    let rank_i32 = i32::try_from(rank).map_err(candle_core::Error::wrap)?;
    let vocab_i32 = i32::try_from(predecessor_vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let packed_width_i32 = i32::try_from(packed_width).map_err(candle_core::Error::wrap)?;

    let (packed_storage, packed_layout) = packed_topk.storage_and_layout();
    let packed_storage = match &*packed_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (hidden_storage, hidden_layout) = projected_hidden.storage_and_layout();
    let hidden_storage = match &*hidden_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (predecessor_storage, predecessor_layout) = predecessor_codebook.storage_and_layout();
    let predecessor_storage = match &*predecessor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (successor_storage, successor_layout) = successor_codebook.storage_and_layout();
    let successor_storage = match &*successor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (anchor_storage, anchor_layout) = anchors.storage_and_layout();
    let anchor_storage = match &*anchor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };

    let dev = packed_storage.device();
    let stream = dev.cuda_stream();
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        candle_core::bail!("{OP} packed top-k dtype mismatch");
    };
    let CudaStorageSlice::U32(anchor_slice) = &anchor_storage.slice else {
        candle_core::bail!("{OP} anchor dtype mismatch");
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
    let (anchor_ptr, anchor_guard) = anchor_slice.device_ptr(&stream);
    let anchor_ptr = unsafe { (anchor_ptr as *const u32).add(anchor_layout.start_offset()) };

    macro_rules! data_ptr {
        ($storage:expr, $layout:expr, $name:expr) => {{
            match &$storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr =
                        unsafe { (ptr as *const f32).add($layout.start_offset()) as *const c_void };
                    (ptr, CUDA_DFLASH_SELECTOR_F32, guard)
                }
                CudaStorageSlice::BF16(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr = unsafe {
                        (ptr as *const half::bf16).add($layout.start_offset()) as *const c_void
                    };
                    (ptr, CUDA_DFLASH_SELECTOR_BF16, guard)
                }
                _ => candle_core::bail!("{OP} {} dtype mismatch", $name),
            }
        }};
    }

    let (hidden_ptr, hidden_dtype, hidden_guard) =
        data_ptr!(hidden_storage, hidden_layout, "projected hidden states");
    let (predecessor_ptr, predecessor_dtype, predecessor_guard) = data_ptr!(
        predecessor_storage,
        predecessor_layout,
        "predecessor codebook"
    );
    let (successor_ptr, successor_dtype, successor_guard) =
        data_ptr!(successor_storage, successor_layout, "successor codebook");

    let mut selected = unsafe { dev.alloc::<u32>(rows) }?;
    let (selected_ptr, selected_guard) = selected.device_ptr_mut(&stream);
    unsafe {
        ffi::dflash_greedy_select(
            packed_ptr,
            hidden_ptr,
            predecessor_ptr,
            successor_ptr,
            anchor_ptr,
            selected_ptr as *mut u32,
            batch_i32,
            positions_i32,
            rank_i32,
            vocab_i32,
            k_i32,
            packed_width_i32,
            hidden_dtype,
            predecessor_dtype,
            successor_dtype,
            stream.cu_stream() as i64,
        );
    }

    drop(packed_guard);
    drop(hidden_guard);
    drop(predecessor_guard);
    drop(successor_guard);
    drop(anchor_guard);
    drop(selected_guard);

    Ok(Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(selected),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions]),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_dflash_sample_select(
    input: DFlashSelectorSampleInput<'_>,
) -> Result<DFlashSelectorSampleOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_dflash_sample_select";
    let DFlashSelectorSampleInput {
        topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
        inverse_temperatures,
        uniforms,
    } = input;
    let packed_topk = &topk.packed;
    let k = topk.k;

    let [rows, packed_width] = packed_topk.dims() else {
        candle_core::bail!("{OP} expected packed top-k with shape [batch * positions, 2 * k]");
    };
    let [hidden_rows, rank] = projected_hidden.dims() else {
        candle_core::bail!(
            "{OP} expected projected hidden states with shape [batch * positions, rank]"
        );
    };
    let [predecessor_vocab, predecessor_rank] = predecessor_codebook.dims() else {
        candle_core::bail!("{OP} expected predecessor codebook with shape [vocab, rank]");
    };
    let [successor_vocab, successor_rank] = successor_codebook.dims() else {
        candle_core::bail!("{OP} expected successor codebook with shape [vocab, rank]");
    };
    let [batch] = anchors.dims() else {
        candle_core::bail!("{OP} expected anchors with shape [batch]");
    };
    let (rows, packed_width, hidden_rows, rank) = (*rows, *packed_width, *hidden_rows, *rank);
    let (predecessor_vocab, predecessor_rank) = (*predecessor_vocab, *predecessor_rank);
    let (successor_vocab, successor_rank, batch) = (*successor_vocab, *successor_rank, *batch);

    if rows == 0 || batch == 0 || rank == 0 || predecessor_vocab == 0 {
        candle_core::bail!("{OP} does not support empty inputs");
    }
    if rows % batch != 0 {
        candle_core::bail!("{OP} row count {rows} is not divisible by batch size {batch}");
    }
    if hidden_rows != rows {
        candle_core::bail!("{OP} expected {rows} projected hidden rows, got {hidden_rows}");
    }
    if predecessor_vocab != successor_vocab || predecessor_rank != rank || successor_rank != rank {
        candle_core::bail!(
            "{OP} codebook shapes {:?} and {:?} do not match hidden rank {rank}",
            predecessor_codebook.dims(),
            successor_codebook.dims()
        );
    }
    let expected_packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    if packed_width != expected_packed_width {
        candle_core::bail!(
            "{OP} expected rank-only packed top-k width {expected_packed_width}, got {packed_width}"
        );
    }
    if k == 0 || k > CUDA_DFLASH_SELECTOR_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {CUDA_DFLASH_SELECTOR_MAX_K}]");
    }
    if predecessor_vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {predecessor_vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    let positions = rows / batch;
    if inverse_temperatures.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected inverse temperatures with shape [{batch}], got {:?}",
            inverse_temperatures.dims()
        );
    }
    if uniforms.dims() != [batch, positions] {
        candle_core::bail!(
            "{OP} expected uniforms with shape [{batch}, {positions}], got {:?}",
            uniforms.dims()
        );
    }
    if packed_topk.dtype() != DType::F32
        || inverse_temperatures.dtype() != DType::F32
        || uniforms.dtype() != DType::F32
    {
        candle_core::bail!("{OP} requires F32 packed top-k, inverse temperatures, and uniforms");
    }
    if anchors.dtype() != DType::U32 {
        candle_core::bail!("{OP} requires U32 anchors");
    }
    for (name, tensor) in [
        ("projected hidden states", projected_hidden),
        ("predecessor codebook", predecessor_codebook),
        ("successor codebook", successor_codebook),
    ] {
        if !matches!(tensor.dtype(), DType::BF16 | DType::F32) {
            candle_core::bail!("{OP} requires BF16 or F32 {name}");
        }
    }
    for tensor in [
        packed_topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
        inverse_temperatures,
        uniforms,
    ] {
        if !tensor.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: OP });
        }
        if !packed_topk.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} tensors must be on the same CUDA device");
        }
    }

    let sparse_elems = rows
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let positions_i32 = i32::try_from(positions).map_err(candle_core::Error::wrap)?;
    let rank_i32 = i32::try_from(rank).map_err(candle_core::Error::wrap)?;
    let vocab_i32 = i32::try_from(predecessor_vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let packed_width_i32 = i32::try_from(packed_width).map_err(candle_core::Error::wrap)?;

    let (packed_storage, packed_layout) = packed_topk.storage_and_layout();
    let packed_storage = match &*packed_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (hidden_storage, hidden_layout) = projected_hidden.storage_and_layout();
    let hidden_storage = match &*hidden_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (predecessor_storage, predecessor_layout) = predecessor_codebook.storage_and_layout();
    let predecessor_storage = match &*predecessor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (successor_storage, successor_layout) = successor_codebook.storage_and_layout();
    let successor_storage = match &*successor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (anchor_storage, anchor_layout) = anchors.storage_and_layout();
    let anchor_storage = match &*anchor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (uniform_storage, uniform_layout) = uniforms.storage_and_layout();
    let uniform_storage = match &*uniform_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };

    let dev = packed_storage.device();
    let stream = dev.cuda_stream();
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        candle_core::bail!("{OP} packed top-k dtype mismatch");
    };
    let CudaStorageSlice::U32(anchor_slice) = &anchor_storage.slice else {
        candle_core::bail!("{OP} anchor dtype mismatch");
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} inverse temperature dtype mismatch");
    };
    let CudaStorageSlice::F32(uniform_slice) = &uniform_storage.slice else {
        candle_core::bail!("{OP} uniform dtype mismatch");
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
    let (anchor_ptr, anchor_guard) = anchor_slice.device_ptr(&stream);
    let anchor_ptr = unsafe { (anchor_ptr as *const u32).add(anchor_layout.start_offset()) };
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };
    let (uniform_ptr, uniform_guard) = uniform_slice.device_ptr(&stream);
    let uniform_ptr = unsafe { (uniform_ptr as *const f32).add(uniform_layout.start_offset()) };

    macro_rules! data_ptr {
        ($storage:expr, $layout:expr, $name:expr) => {{
            match &$storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr =
                        unsafe { (ptr as *const f32).add($layout.start_offset()) as *const c_void };
                    (ptr, CUDA_DFLASH_SELECTOR_F32, guard)
                }
                CudaStorageSlice::BF16(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr = unsafe {
                        (ptr as *const half::bf16).add($layout.start_offset()) as *const c_void
                    };
                    (ptr, CUDA_DFLASH_SELECTOR_BF16, guard)
                }
                _ => candle_core::bail!("{OP} {} dtype mismatch", $name),
            }
        }};
    }

    let (hidden_ptr, hidden_dtype, hidden_guard) =
        data_ptr!(hidden_storage, hidden_layout, "projected hidden states");
    let (predecessor_ptr, predecessor_dtype, predecessor_guard) = data_ptr!(
        predecessor_storage,
        predecessor_layout,
        "predecessor codebook"
    );
    let (successor_ptr, successor_dtype, successor_guard) =
        data_ptr!(successor_storage, successor_layout, "successor codebook");

    let mut selected = unsafe { dev.alloc::<u32>(rows) }?;
    let mut candidate_ids = unsafe { dev.alloc::<u32>(sparse_elems) }?;
    let mut candidate_probs = unsafe { dev.alloc::<f32>(sparse_elems) }?;
    let (selected_ptr, selected_guard) = selected.device_ptr_mut(&stream);
    let (candidate_ids_ptr, candidate_ids_guard) = candidate_ids.device_ptr_mut(&stream);
    let (candidate_probs_ptr, candidate_probs_guard) = candidate_probs.device_ptr_mut(&stream);
    unsafe {
        ffi::dflash_sample_select(
            packed_ptr,
            hidden_ptr,
            predecessor_ptr,
            successor_ptr,
            anchor_ptr,
            temperature_ptr,
            uniform_ptr,
            selected_ptr as *mut u32,
            candidate_ids_ptr as *mut u32,
            candidate_probs_ptr as *mut f32,
            batch_i32,
            positions_i32,
            rank_i32,
            vocab_i32,
            k_i32,
            packed_width_i32,
            hidden_dtype,
            predecessor_dtype,
            successor_dtype,
            stream.cu_stream() as i64,
        );
    }

    drop(packed_guard);
    drop(hidden_guard);
    drop(predecessor_guard);
    drop(successor_guard);
    drop(anchor_guard);
    drop(temperature_guard);
    drop(uniform_guard);
    drop(selected_guard);
    drop(candidate_ids_guard);
    drop(candidate_probs_guard);

    let tokens = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(selected),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions]),
    ));
    let candidate_ids = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(candidate_ids),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions, k]),
    ));
    let candidate_probs = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::F32(candidate_probs),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions, k]),
    ));

    Ok(DFlashSelectorSampleOutput {
        tokens,
        candidate_ids,
        candidate_probs,
    })
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) fn cuda_top1_logits_f32_packed_batched(
    input: &Tensor,
) -> Result<Top1LogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    const OP: &str = "cuda_top1_logits_f32_packed_batched";
    if input.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }

    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    let (batch, vocab) = (*batch, *vocab);
    if batch == 0 || vocab == 0 {
        candle_core::bail!("{OP} requires a non-empty batch and vocabulary");
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} workspace overflow")))?;
    let packed_elems = batch
        .checked_mul(CUDA_TOP1_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        candle_core::bail!("{OP} only supports F32 logits");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(packed_elems) }?;

    let (input_ptr, input_guard) = input_slice.device_ptr(&stream);
    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);
    let input_ptr = unsafe { (input_ptr as *const f32).add(input_layout.start_offset()) };

    unsafe {
        ffi::top1_large_f32_packed_batched(
            input_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            packed_ptr as *mut f32,
            std::ptr::null_mut(),
            nrows_i32,
            ncols_i32,
            chunk_size_i32,
            nblocks_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(input_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(packed_guard);

    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
    ];
    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };

    Ok(Top1LogitsPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[batch, CUDA_TOP1_PACKED_WIDTH]),
        )),
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_categorical_logits_f32_packed_batched(
    input: &Tensor,
    inverse_temperatures: &Tensor,
    uniforms: &Tensor,
) -> Result<CategoricalLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    const OP: &str = "cuda_categorical_logits_f32_packed_batched";
    if input.dtype() != DType::F32
        || inverse_temperatures.dtype() != DType::F32
        || uniforms.dtype() != DType::F32
    {
        candle_core::bail!("{OP} requires F32 tensors");
    }
    if !input.is_contiguous() || !inverse_temperatures.is_contiguous() || !uniforms.is_contiguous()
    {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    if !input.device().same_device(inverse_temperatures.device())
        || !input.device().same_device(uniforms.device())
    {
        candle_core::bail!("{OP} tensors must be on the same CUDA device");
    }

    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    let (batch, vocab) = (*batch, *vocab);
    if batch == 0 || vocab == 0 {
        candle_core::bail!("{OP} requires a non-empty batch and vocabulary");
    }
    if inverse_temperatures.dims() != [batch] || uniforms.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected sampling tensors with shape [{batch}], got {:?} and {:?}",
            inverse_temperatures.dims(),
            uniforms.dims()
        );
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} workspace overflow")))?;
    let packed_elems = batch
        .checked_mul(CUDA_CATEGORICAL_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        candle_core::bail!("{OP} only supports F32 logits");
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA inverse temperatures"),
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} only supports F32 inverse temperatures");
    };
    let (uniform_storage, uniform_layout) = uniforms.storage_and_layout();
    let uniform_storage = match &*uniform_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA uniforms"),
    };
    let CudaStorageSlice::F32(uniform_slice) = &uniform_storage.slice else {
        candle_core::bail!("{OP} only supports F32 uniforms");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(packed_elems) }?;

    let (input_ptr, input_guard) = input_slice.device_ptr(&stream);
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let (uniform_ptr, uniform_guard) = uniform_slice.device_ptr(&stream);
    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);
    let input_ptr = unsafe { (input_ptr as *const f32).add(input_layout.start_offset()) };
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };
    let uniform_ptr = unsafe { (uniform_ptr as *const f32).add(uniform_layout.start_offset()) };

    unsafe {
        ffi::categorical_large_f32_packed_batched(
            input_ptr,
            temperature_ptr,
            uniform_ptr,
            block_values_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            nrows_i32,
            ncols_i32,
            chunk_size_i32,
            nblocks_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(input_guard);
    drop(temperature_guard);
    drop(uniform_guard);
    drop(block_values_guard);
    drop(block_sums_guard);
    drop(packed_guard);

    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
    ];
    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };

    Ok(CategoricalLogitsPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[batch, CUDA_CATEGORICAL_PACKED_WIDTH]),
        )),
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub struct CudaTop1LogitsWorkspace {
    nrows: usize,
    capacity_rows: usize,
    ncols: usize,
    nblocks: usize,
    location: candle_core::DeviceLocation,
    id: u64,
    next_slot: usize,
    next_generation: u64,
    slots: Vec<CudaTop1LogitsSlot>,
}

#[cfg(feature = "cuda")]
struct CudaTop1LogitsSlot {
    block_values: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    block_indices: candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
    packed: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    packed_host: candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<f32>,
    owned_token_ids: Tensor,
    token_ids_host: candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<u32>,
    device_ready: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    host_complete: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    consumer_complete: candle_core::cuda_backend::cudarc::driver::CudaEvent,
    reuse_ready: candle_core::cuda_backend::cudarc::driver::CudaEvent,
    reuse_pending: bool,
    pending: Option<CudaTop1Pending>,
}

#[cfg(feature = "cuda")]
struct CudaTop1Pending {
    generation: u64,
    nrows: usize,
    copy_packed: bool,
    producer_stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    consumer_stream: Option<std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>>,
    token_ptr: u64,
    token_end_ptr: u64,
    token_released: bool,
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTop1Submission {
    workspace_id: u64,
    slot: usize,
    generation: u64,
    nrows: usize,
    _device_tokens: Tensor,
    device_ready: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    host_complete: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
}

#[cfg(feature = "cuda")]
impl CudaTop1Submission {
    #[cfg(test)]
    pub(crate) fn device_tokens(&self) -> &Tensor {
        &self._device_tokens
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.nrows
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.host_complete
            .synchronize()
            .map_err(candle_core::Error::wrap)
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTop1Completion<'a> {
    token_ids: &'a [u32],
    packed: Option<&'a [f32]>,
}

#[cfg(feature = "cuda")]
impl<'a> CudaTop1Completion<'a> {
    pub(crate) fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }

    pub(crate) fn packed(&self) -> Option<&'a [f32]> {
        self.packed
    }
}

#[cfg(feature = "cuda")]
struct CudaTop1SubmitOptions<'a> {
    nrows: usize,
    ncols: usize,
    token_ids_dst: Option<&'a Tensor>,
    copy_packed: bool,
    op: &'static str,
}

#[cfg(feature = "cuda")]
fn final_logits_row(input: &Tensor) -> Result<Tensor> {
    let dims = input.dims();
    if dims.len() <= 1 {
        return input.contiguous();
    }
    let vocab = *dims.last().expect("rank checked above");
    if vocab == 0 {
        candle_core::bail!("logits last dimension is empty");
    }
    let rows = input.elem_count() / vocab;
    if rows == 0 {
        candle_core::bail!("logits tensor is empty");
    }
    input
        .reshape((rows, vocab))?
        .narrow(0, rows - 1, 1)?
        .reshape(vocab)?
        .contiguous()
}

#[cfg(feature = "cuda")]
fn cuda_top1_workspace_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
fn same_cuda_stream(
    left: &candle_core::cuda_backend::cudarc::driver::CudaStream,
    right: &candle_core::cuda_backend::cudarc::driver::CudaStream,
) -> bool {
    std::sync::Arc::ptr_eq(left.context(), right.context()) && left.cu_stream() == right.cu_stream()
}

#[cfg(feature = "cuda")]
fn new_cuda_top1_slot(
    dev: &candle_core::CudaDevice,
    nrows: usize,
    workspace_elems: usize,
    packed_elems: usize,
) -> Result<CudaTop1LogitsSlot> {
    use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

    let stream = dev.cuda_stream();
    let context = stream.context();
    let mut token_ids = unsafe { dev.alloc::<u32>(nrows) }?;
    let (_, token_ids_guard) = token_ids.device_ptr_mut(&stream);
    drop(token_ids_guard);
    let owned_token_ids = Tensor::from((
        candle_core::Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::U32(token_ids),
            device: dev.clone(),
        }),
        Shape::from_dims(&[nrows, 1]),
    ));
    let event_flags = Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC);

    Ok(CudaTop1LogitsSlot {
        block_values: unsafe { dev.alloc::<f32>(workspace_elems) }?,
        block_indices: unsafe { dev.alloc::<u32>(workspace_elems) }?,
        packed: unsafe { dev.alloc::<f32>(packed_elems) }?,
        packed_host: unsafe { context.alloc_pinned::<f32>(packed_elems) }
            .map_err(candle_core::Error::wrap)?,
        owned_token_ids,
        token_ids_host: unsafe { context.alloc_pinned::<u32>(nrows) }
            .map_err(candle_core::Error::wrap)?,
        device_ready: std::sync::Arc::new(
            context
                .new_event(event_flags)
                .map_err(candle_core::Error::wrap)?,
        ),
        host_complete: std::sync::Arc::new(
            context
                .new_event(event_flags)
                .map_err(candle_core::Error::wrap)?,
        ),
        consumer_complete: context
            .new_event(event_flags)
            .map_err(candle_core::Error::wrap)?,
        reuse_ready: context
            .new_event(event_flags)
            .map_err(candle_core::Error::wrap)?,
        reuse_pending: false,
        pending: None,
    })
}

#[cfg(feature = "cuda")]
fn new_cuda_top1_workspace(
    dev: &candle_core::CudaDevice,
    nrows: usize,
    ncols: usize,
    nblocks: usize,
) -> Result<CudaTop1LogitsWorkspace> {
    use candle_core::backend::BackendDevice;

    let workspace_elems = nrows
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg("CUDA top-1 workspace overflow".to_string()))?;
    let packed_elems = nrows
        .checked_mul(CUDA_TOP1_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg("CUDA top-1 output overflow".to_string()))?;
    let mut slots = Vec::with_capacity(CUDA_TOP1_RING_SLOTS);
    for _ in 0..CUDA_TOP1_RING_SLOTS {
        slots.push(new_cuda_top1_slot(
            dev,
            nrows,
            workspace_elems,
            packed_elems,
        )?);
    }
    Ok(CudaTop1LogitsWorkspace {
        nrows,
        capacity_rows: nrows,
        ncols,
        nblocks,
        location: dev.location(),
        id: cuda_top1_workspace_id(),
        next_slot: 0,
        next_generation: 1,
        slots,
    })
}

#[cfg(feature = "cuda")]
fn validate_cuda_top1_submission<'a>(
    workspace: &'a CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    op: &'static str,
) -> Result<&'a CudaTop1LogitsSlot> {
    if submission.workspace_id != workspace.id {
        candle_core::bail!("{op} received a submission from a different workspace");
    }
    let slot = workspace
        .slots
        .get(submission.slot)
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} received an invalid ring slot")))?;
    let Some(pending) = &slot.pending else {
        candle_core::bail!("{op} received an inactive submission");
    };
    if pending.generation != submission.generation {
        candle_core::bail!("{op} received a stale submission");
    }
    Ok(slot)
}

#[cfg(feature = "cuda")]
fn cuda_top1_logits_submit_inner(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
    options: CudaTop1SubmitOptions<'_>,
) -> Result<CudaTop1Submission> {
    use candle_core::backend::{BackendDevice, BackendStorage};
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    let CudaTop1SubmitOptions {
        nrows,
        ncols,
        token_ids_dst,
        copy_packed,
        op,
    } = options;

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{op} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op });
    }
    if nrows == 0 || ncols == 0 {
        candle_core::bail!("{op} requires non-empty logits");
    }
    if ncols > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{op} vocabulary size {ncols} cannot be represented exactly by packed F32 indices"
        );
    }
    if nrows > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{op} batch is too large for a 2D CUDA launch: {nrows}");
    }
    let expected_elems = nrows
        .checked_mul(ncols)
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} input size overflow")))?;
    if input.elem_count() != expected_elems {
        candle_core::bail!(
            "{op} expected {nrows} rows of {ncols} logits, got {} elements",
            input.elem_count()
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("{op} requires CUDA logits"),
    };
    let dev = storage.device();
    let location = dev.location();
    let needs_alloc = cache.as_ref().is_none_or(|workspace| {
        workspace.capacity_rows < nrows
            || workspace.ncols != ncols
            || workspace.nblocks != nblocks
            || workspace.location != location
    });
    if needs_alloc {
        if cache
            .as_ref()
            .is_some_and(|workspace| workspace.slots.iter().any(|slot| slot.pending.is_some()))
        {
            candle_core::bail!("{op} cannot resize while submissions are pending");
        }
        *cache = Some(new_cuda_top1_workspace(dev, nrows, ncols, nblocks)?);
    }

    let stream = dev.cuda_stream();
    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr = unsafe { (ptr as *const $ty).add(layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => unreachable!("logits dtype was validated above"),
    };
    let workspace = cache
        .as_mut()
        .expect("CUDA top-1 workspace was allocated above");
    workspace.nrows = nrows;
    let slot_index = (0..CUDA_TOP1_RING_SLOTS)
        .map(|offset| (workspace.next_slot + offset) % CUDA_TOP1_RING_SLOTS)
        .find(|&index| workspace.slots[index].pending.is_none())
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} submission ring is full")))?;
    workspace.next_slot = (slot_index + 1) % CUDA_TOP1_RING_SLOTS;
    let generation = workspace.next_generation;
    workspace.next_generation = workspace.next_generation.wrapping_add(1).max(1);
    let device_tokens = token_ids_dst.cloned().map(Ok).unwrap_or_else(|| {
        workspace.slots[slot_index]
            .owned_token_ids
            .narrow(0, 0, nrows)
    })?;
    let destination_capacity = match device_tokens.dims() {
        [capacity, 1] => *capacity,
        _ => 0,
    };
    if device_tokens.dtype() != DType::U32
        || destination_capacity < nrows
        || !device_tokens.is_contiguous()
    {
        candle_core::bail!(
            "{op} token destination must be contiguous U32 with shape [capacity, 1], capacity >= {nrows}"
        );
    }
    if !device_tokens.device().same_device(input.device()) {
        candle_core::bail!("{op} token destination and logits must be on the same CUDA device");
    }

    let device_tokens_storage = device_tokens.clone();
    let (token_storage, token_layout) = device_tokens_storage.storage_and_layout();
    let token_storage = match &*token_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{op} token destination must be on CUDA"),
    };
    let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
        candle_core::bail!("{op} token destination must use U32 storage");
    };
    let (token_ptr, token_guard) = token_slice.device_ptr(&stream);
    let token_ptr = unsafe { (token_ptr as *mut u32).add(token_layout.start_offset()) };
    let token_end_ptr = token_ptr as u64 + (nrows * std::mem::size_of::<u32>()) as u64;
    for other in &workspace.slots {
        let Some(pending) = &other.pending else {
            continue;
        };
        if pending.token_ptr >= token_end_ptr || token_ptr as u64 >= pending.token_end_ptr {
            continue;
        }
        if !pending.token_released {
            candle_core::bail!("{op} token destination is already leased by another submission");
        }
        if other.reuse_pending {
            stream
                .wait(&other.reuse_ready)
                .map_err(candle_core::Error::wrap)?;
        }
    }
    let slot = &mut workspace.slots[slot_index];
    if slot.reuse_pending {
        stream
            .wait(&slot.reuse_ready)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_pending = false;
    }
    slot.pending = Some(CudaTop1Pending {
        generation,
        nrows,
        copy_packed,
        producer_stream: stream.clone(),
        consumer_stream: None,
        token_ptr: token_ptr as u64,
        token_end_ptr,
        token_released: false,
    });

    let result = (|| {
        let (block_values_ptr, block_values_guard) = slot.block_values.device_ptr_mut(&stream);
        let (block_indices_ptr, block_indices_guard) = slot.block_indices.device_ptr_mut(&stream);
        let (packed_ptr, packed_guard) = if copy_packed {
            let (packed_ptr, packed_guard) = slot.packed.device_ptr_mut(&stream);
            (packed_ptr as *mut f32, Some(packed_guard))
        } else {
            (std::ptr::null_mut(), None)
        };

        let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
        let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;
        let chunk_size_i32 =
            i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
        let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;
        macro_rules! launch {
            ($single:path, $batched:path, $ptr:expr) => {{
                unsafe {
                    if nrows == 1 {
                        $single(
                            $ptr,
                            block_values_ptr as *mut f32,
                            block_indices_ptr as *mut u32,
                            packed_ptr,
                            token_ptr,
                            ncols_i32,
                            chunk_size_i32,
                            nblocks_i32,
                            stream.cu_stream() as i64,
                        );
                    } else {
                        $batched(
                            $ptr,
                            block_values_ptr as *mut f32,
                            block_indices_ptr as *mut u32,
                            packed_ptr,
                            token_ptr,
                            nrows_i32,
                            ncols_i32,
                            chunk_size_i32,
                            nblocks_i32,
                            stream.cu_stream() as i64,
                        );
                    }
                }
            }};
        }
        match input.dtype() {
            DType::F32 => launch!(
                ffi::top1_large_f32_packed,
                ffi::top1_large_f32_packed_batched,
                input_ptr.cast::<f32>()
            ),
            DType::BF16 => launch!(
                ffi::top1_large_bf16_packed,
                ffi::top1_large_bf16_packed_batched,
                input_ptr
            ),
            DType::F16 => launch!(
                ffi::top1_large_f16_packed,
                ffi::top1_large_f16_packed_batched,
                input_ptr
            ),
            _ => unreachable!("logits dtype was validated above"),
        }

        drop(input_guard);
        drop(block_values_guard);
        drop(block_indices_guard);
        drop(packed_guard);
        drop(token_guard);

        slot.device_ready
            .record(&stream)
            .map_err(candle_core::Error::wrap)?;
        let token_copy = token_slice
            .slice(token_layout.start_offset()..token_layout.start_offset().saturating_add(nrows));
        dev.memcpy_dtoh(&token_copy, &mut slot.token_ids_host)?;
        if copy_packed {
            dev.memcpy_dtoh(&slot.packed, &mut slot.packed_host)?;
        }
        slot.host_complete
            .record(&stream)
            .map_err(candle_core::Error::wrap)?;
        Result::<()>::Ok(())
    })();
    if let Err(error) = result {
        let _ = stream.synchronize();
        slot.pending = None;
        return Err(error);
    }
    Ok(CudaTop1Submission {
        workspace_id: workspace.id,
        slot: slot_index,
        generation,
        nrows,
        _device_tokens: device_tokens,
        device_ready: slot.device_ready.clone(),
        host_complete: slot.host_complete.clone(),
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_logits_submit_batched(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: None,
            copy_packed: false,
            op: OP,
        },
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_logits_submit_batched_packed(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched_packed";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: None,
            copy_packed: true,
            op: OP,
        },
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_logits_submit_batched_into(
    input: &Tensor,
    token_ids_dst: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched_into";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: Some(token_ids_dst),
            copy_packed: false,
            op: OP,
        },
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_device_tokens_wait_on(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    consumer_stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_top1_device_tokens_wait_on";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    let slot = &mut workspace.slots[submission.slot];
    let pending = slot.pending.as_mut().expect("submission validated above");
    if same_cuda_stream(&pending.producer_stream, consumer_stream) {
        slot.reuse_ready
            .record(&pending.producer_stream)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_pending = true;
        pending.token_released = true;
        return Ok(());
    }
    if let Some(current) = &pending.consumer_stream {
        if same_cuda_stream(current, consumer_stream) {
            return Ok(());
        }
        candle_core::bail!("{OP} only supports one cross-stream consumer per submission");
    }
    consumer_stream
        .wait(&submission.device_ready)
        .map_err(candle_core::Error::wrap)?;
    pending.consumer_stream = Some(consumer_stream.clone());
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_device_tokens_release_after(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    consumer_stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_top1_device_tokens_release_after";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    let slot = &mut workspace.slots[submission.slot];
    let pending = slot.pending.as_mut().expect("submission validated above");
    if same_cuda_stream(&pending.producer_stream, consumer_stream) {
        return Ok(());
    }
    let Some(current) = &pending.consumer_stream else {
        candle_core::bail!("{OP} requires wait_on before release_after");
    };
    if !same_cuda_stream(current, consumer_stream) {
        candle_core::bail!("{OP} consumer stream does not match wait_on");
    }
    slot.consumer_complete
        .record(consumer_stream)
        .map_err(candle_core::Error::wrap)?;
    pending
        .producer_stream
        .wait(&slot.consumer_complete)
        .map_err(candle_core::Error::wrap)?;
    slot.reuse_ready
        .record(&pending.producer_stream)
        .map_err(candle_core::Error::wrap)?;
    slot.reuse_pending = true;
    pending.consumer_stream = None;
    pending.token_released = true;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_submission_complete<'a>(
    workspace: &'a mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
) -> Result<CudaTop1Completion<'a>> {
    const OP: &str = "cuda_top1_submission_complete";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    let slot = &mut workspace.slots[submission.slot];
    let pending = slot.pending.as_ref().expect("submission validated above");
    if pending.consumer_stream.is_some() {
        candle_core::bail!("{OP} requires release_after for the cross-stream consumer");
    }
    slot.host_complete
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    let copy_packed = pending.copy_packed;
    let nrows = pending.nrows;
    slot.pending = None;
    let token_ids = &slot
        .token_ids_host
        .as_slice()
        .map_err(candle_core::Error::wrap)?[..nrows];
    let packed = if copy_packed {
        Some(
            &slot
                .packed_host
                .as_slice()
                .map_err(candle_core::Error::wrap)?[..nrows * CUDA_TOP1_PACKED_WIDTH],
        )
    } else {
        None
    };
    Ok(CudaTop1Completion { token_ids, packed })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_top1_submission_cancel(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
) -> Result<()> {
    const OP: &str = "cuda_top1_submission_cancel";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    let slot = &mut workspace.slots[submission.slot];
    let pending = slot.pending.as_mut().expect("submission validated above");
    if let Some(consumer_stream) = pending.consumer_stream.take() {
        slot.consumer_complete
            .record(&consumer_stream)
            .map_err(candle_core::Error::wrap)?;
        pending
            .producer_stream
            .wait(&slot.consumer_complete)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_ready
            .record(&pending.producer_stream)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_pending = true;
    }
    slot.host_complete
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    slot.pending = None;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_top1_logits_f32_packed_cached_inner<'a>(
    input: &Tensor,
    nrows: usize,
    ncols: usize,
    cache: &'a mut Option<CudaTop1LogitsWorkspace>,
    op: &'static str,
) -> Result<&'a [f32]> {
    let submission = cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows,
            ncols,
            token_ids_dst: None,
            copy_packed: true,
            op,
        },
    )?;
    let completion = cuda_top1_submission_complete(
        cache
            .as_mut()
            .expect("CUDA top-1 workspace was allocated during submission"),
        &submission,
    )?;
    Ok(completion
        .packed()
        .expect("packed output was requested during submission"))
}

#[cfg(feature = "cuda")]
pub fn cuda_top1_logits_f32_cached(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<[f32; CUDA_TOP1_PACKED_WIDTH]> {
    const OP: &str = "cuda_top1_logits_f32_cached";
    let input = final_logits_row(input)?;
    let ncols = input.elem_count();
    let packed = cuda_top1_logits_f32_packed_cached_inner(&input, 1, ncols, cache, OP)?;
    Ok([packed[0], packed[1]])
}

#[cfg(feature = "cuda")]
#[cfg(test)]
pub(crate) fn cuda_top1_logits_f32_packed_batched_cached(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<Vec<[f32; CUDA_TOP1_PACKED_WIDTH]>> {
    const OP: &str = "cuda_top1_logits_f32_packed_batched_cached";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    let (batch, vocab) = (*batch, *vocab);
    let packed = cuda_top1_logits_f32_packed_cached_inner(input, batch, vocab, cache, OP)?;
    Ok(packed
        .chunks_exact(CUDA_TOP1_PACKED_WIDTH)
        .map(|row| [row[0], row[1]])
        .collect())
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
    inplace: bool,
}

impl candle_core::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        panic!("not implemented!")
    }

    #[allow(clippy::cast_possible_truncation)]
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;

        let dev = storage.device();
        let elem_count = layout.shape().elem_count();
        let ncols = self.last_dim as i32;
        let nrows = elem_count as i32 / ncols;
        let dst = unsafe { dev.alloc::<u32>(elem_count) }?;

        use std::ffi::c_void;

        let (src, _src_guard) = match &storage.slice {
            CudaStorageSlice::U8(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::U32(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::I64(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::BF16(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::F16(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::F32(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::F64(inp) => inp.device_ptr(inp.stream()),
            _ => candle_core::bail!("Unexpected dtype in asort"),
        };
        let src_ptr = src as *const c_void;
        let (dst_ptr, dst_guard) = dst.device_ptr(dst.stream());
        let dst_ptr = dst_ptr as *mut c_void;
        let stream = dev.cuda_stream().cu_stream() as i64;
        unsafe {
            if self.asc {
                match storage.dtype() {
                    candle_core::DType::U8 => {
                        ffi::asort_asc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::U32 => {
                        ffi::asort_asc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::I64 => {
                        ffi::asort_asc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::BF16 => {
                        ffi::asort_asc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F16 => {
                        ffi::asort_asc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F32 => {
                        ffi::asort_asc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F64 => {
                        ffi::asort_asc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    _ => candle_core::bail!("Unexpected dtype in asort"),
                }
            } else {
                match storage.dtype() {
                    candle_core::DType::U8 => {
                        ffi::asort_desc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::U32 => {
                        ffi::asort_desc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::I64 => {
                        ffi::asort_desc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::BF16 => {
                        ffi::asort_desc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F16 => {
                        ffi::asort_desc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F32 => {
                        ffi::asort_desc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F64 => {
                        ffi::asort_desc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    _ => candle_core::bail!("Unexpected dtype in asort"),
                }
            }
        }
        drop(dst_guard);
        let dst_ret = candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(dst),
            device: dev.clone(),
        };
        Ok((dst_ret, layout.shape().clone()))
    }
}

#[allow(dead_code)]
pub trait ArgSortOp {
    fn arg_sort(&self, asc: bool) -> Result<Tensor>;
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)>;
}

impl ArgSortOp for Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    fn arg_sort(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: false,
        })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        let sorted = self.copy()?;

        let asort = sorted.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: true,
        })?;

        Ok((sorted, asort))
    }
}

#[allow(dead_code)]
pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

#[allow(dead_code)]
pub struct TopKLogitsOutput {
    pub values: Tensor,
    pub indices: Tensor,
    /// `[softmax_denominator, global_max]` for the full-vocabulary softmax at
    /// the temperature used for top-k selection.
    pub softmax_info: Tensor,
    _workspace: Vec<Tensor>,
}

#[allow(dead_code)]
pub struct TopKLogitsPackedOutput {
    /// Each row is packed as `[values; indices_as_f32; softmax_denominator; global_max]`.
    pub packed: Tensor,
    pub k: usize,
    _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
pub(crate) struct RankedTopKPackedOutput {
    /// Each row is packed as `[values; indices_as_f32]`.
    pub(crate) packed: Tensor,
    pub(crate) k: usize,
    _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
pub(crate) struct CategoricalLogitsPackedOutput {
    /// Each row is packed as `[token_index_as_f32, full_softmax_logprob]`.
    pub(crate) packed: Tensor,
    _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) struct Top1LogitsPackedOutput {
    pub(crate) packed: Tensor,
    _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DFlashSelectorSampleInput<'a> {
    pub(crate) topk: &'a RankedTopKPackedOutput,
    pub(crate) projected_hidden: &'a Tensor,
    pub(crate) predecessor_codebook: &'a Tensor,
    pub(crate) successor_codebook: &'a Tensor,
    pub(crate) anchors: &'a Tensor,
    pub(crate) inverse_temperatures: &'a Tensor,
    pub(crate) uniforms: &'a Tensor,
}

#[cfg(feature = "cuda")]
pub(crate) struct DFlashSelectorSampleOutput {
    pub(crate) tokens: Tensor,
    pub(crate) candidate_ids: Tensor,
    pub(crate) candidate_probs: Tensor,
}

#[cfg(feature = "cuda")]
pub fn cuda_apply_sparse_penalties_f32(
    input: &Tensor,
    token_ids: &Tensor,
    counts: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires F32 logits");
    }
    if token_ids.dtype() != DType::U32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires U32 token ids");
    }
    if counts.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires F32 counts");
    }
    if token_ids.elem_count() != counts.elem_count() {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 token ids/counts length mismatch: {} vs {}",
            token_ids.elem_count(),
            counts.elem_count()
        );
    }
    if !token_ids.device().same_device(input.device())
        || !counts.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 tensors must be on the same CUDA device"
        );
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let counts = counts.contiguous()?;

    let elem_count = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 got empty logits");
    }
    if elem_count > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 input is too large: {elem_count} elements"
        );
    }
    if n_tokens > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 token list is too large: {n_tokens} elements"
        );
    }
    let elem_count_i32 = i32::try_from(elem_count).map_err(candle_core::Error::wrap)?;
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA logits"),
    };
    let CudaStorageSlice::F32(src) = &input_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports F32 logits");
    };

    let (token_storage, token_layout) = token_ids.storage_and_layout();
    let token_storage = match &*token_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA token ids"),
    };
    let CudaStorageSlice::U32(token_src) = &token_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports U32 token ids");
    };

    let (count_storage, count_layout) = counts.storage_and_layout();
    let count_storage = match &*count_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA counts"),
    };
    let CudaStorageSlice::F32(count_src) = &count_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports F32 counts");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, src_guard) = src.device_ptr(&stream);
    let (token_ptr, token_guard) = token_src.device_ptr(&stream);
    let (count_ptr, count_guard) = count_src.device_ptr(&stream);
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

    let src_ptr = unsafe { (src_ptr as *const f32).add(input_layout.start_offset()) };
    let token_ptr = unsafe { (token_ptr as *const u32).add(token_layout.start_offset()) };
    let count_ptr = unsafe { (count_ptr as *const f32).add(count_layout.start_offset()) };

    unsafe {
        ffi::apply_sparse_penalties_f32(
            src_ptr as *const c_void,
            out_ptr as *mut c_void,
            token_ptr,
            count_ptr,
            elem_count_i32,
            n_tokens_i32,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            stream.cu_stream() as i64,
        );
    }

    drop(src_guard);
    drop(token_guard);
    drop(count_guard);
    drop(out_guard);

    let out_storage = CudaStorage {
        slice: CudaStorageSlice::F32(out),
        device: dev.clone(),
    };
    Ok(Tensor::from((
        candle_core::Storage::Cuda(out_storage),
        input.shape().clone(),
    )))
}

#[cfg(feature = "cuda")]
pub fn cuda_apply_sparse_logits_bias_f32(
    input: &Tensor,
    token_ids: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires F32 logits");
    }
    if token_ids.dtype() != DType::U32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires U32 token ids");
    }
    if biases.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires F32 biases");
    }
    if token_ids.elem_count() != biases.elem_count() {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 token ids/biases length mismatch: {} vs {}",
            token_ids.elem_count(),
            biases.elem_count()
        );
    }
    if !token_ids.device().same_device(input.device())
        || !biases.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 tensors must be on the same CUDA device"
        );
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let biases = biases.contiguous()?;

    let elem_count = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 got empty logits");
    }
    if elem_count > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 input is too large: {elem_count} elements"
        );
    }
    if n_tokens > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 token list is too large: {n_tokens} elements"
        );
    }
    let elem_count_i32 = i32::try_from(elem_count).map_err(candle_core::Error::wrap)?;
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA logits"),
    };
    let CudaStorageSlice::F32(src) = &input_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports F32 logits");
    };

    let (token_storage, token_layout) = token_ids.storage_and_layout();
    let token_storage = match &*token_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA token ids"),
    };
    let CudaStorageSlice::U32(token_src) = &token_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports U32 token ids");
    };

    let (bias_storage, bias_layout) = biases.storage_and_layout();
    let bias_storage = match &*bias_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA biases"),
    };
    let CudaStorageSlice::F32(bias_src) = &bias_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports F32 biases");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, src_guard) = src.device_ptr(&stream);
    let (token_ptr, token_guard) = token_src.device_ptr(&stream);
    let (bias_ptr, bias_guard) = bias_src.device_ptr(&stream);
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

    let src_ptr = unsafe { (src_ptr as *const f32).add(input_layout.start_offset()) };
    let token_ptr = unsafe { (token_ptr as *const u32).add(token_layout.start_offset()) };
    let bias_ptr = unsafe { (bias_ptr as *const f32).add(bias_layout.start_offset()) };

    unsafe {
        ffi::apply_sparse_logits_bias_f32(
            src_ptr as *const c_void,
            out_ptr as *mut c_void,
            token_ptr,
            bias_ptr,
            elem_count_i32,
            n_tokens_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(src_guard);
    drop(token_guard);
    drop(bias_guard);
    drop(out_guard);

    let out_storage = CudaStorage {
        slice: CudaStorageSlice::F32(out),
        device: dev.clone(),
    };
    Ok(Tensor::from((
        candle_core::Storage::Cuda(out_storage),
        input.shape().clone(),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_apply_causal_mask_f32(
    scores: &Tensor,
    q_offset: usize,
    prefix_len: usize,
) -> Result<()> {
    struct CausalMaskF32 {
        q_offset: usize,
        prefix_len: usize,
    }

    impl candle_core::InplaceOp1 for CausalMaskF32 {
        fn name(&self) -> &'static str {
            "causal-mask-f32"
        }

        fn cpu_fwd(
            &self,
            _storage: &mut candle_core::CpuStorage,
            _layout: &candle_core::Layout,
        ) -> Result<()> {
            candle_core::bail!("causal-mask-f32 requires CUDA storage")
        }

        fn cuda_fwd(
            &self,
            storage: &mut candle_core::CudaStorage,
            layout: &candle_core::Layout,
        ) -> Result<()> {
            use candle_core::backend::BackendStorage;
            use candle_core::cuda_backend::cudarc::driver::DevicePtrMut;
            use candle_core::cuda_backend::CudaStorageSlice;
            use std::ffi::c_void;

            let (batch_heads, q_len, kv_len) = layout.shape().dims3()?;
            let batch_heads = i32::try_from(batch_heads).map_err(candle_core::Error::wrap)?;
            let q_len = i32::try_from(q_len).map_err(candle_core::Error::wrap)?;
            let kv_len = i32::try_from(kv_len).map_err(candle_core::Error::wrap)?;
            let q_offset = i32::try_from(self.q_offset).map_err(candle_core::Error::wrap)?;
            let prefix_len = i32::try_from(self.prefix_len).map_err(candle_core::Error::wrap)?;
            if !layout.is_contiguous() {
                candle_core::bail!("causal-mask-f32 requires contiguous scores")
            }
            let dev = storage.device();
            let stream = dev.cuda_stream();
            let CudaStorageSlice::F32(scores) = &mut storage.slice else {
                candle_core::bail!("causal-mask-f32 requires F32 scores")
            };
            let (scores_ptr, scores_guard) = scores.device_ptr_mut(&stream);
            let scores_ptr =
                unsafe { (scores_ptr as *mut f32).add(layout.start_offset()) as *mut c_void };
            unsafe {
                ffi::apply_causal_mask_f32(
                    scores_ptr,
                    batch_heads,
                    q_len,
                    kv_len,
                    q_offset,
                    prefix_len,
                    stream.cu_stream() as i64,
                );
            }
            drop(scores_guard);
            Ok(())
        }
    }

    scores.inplace_op1(&CausalMaskF32 {
        q_offset,
        prefix_len,
    })
}

#[cfg(feature = "metal")]
pub fn metal_apply_sparse_penalties(
    input: &Tensor,
    token_ids: &Tensor,
    counts: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<Tensor> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("metal_apply_sparse_penalties requires F32/F16/BF16 logits");
    }
    if token_ids.dtype() != DType::U32 || counts.dtype() != DType::F32 {
        candle_core::bail!("metal_apply_sparse_penalties token_ids must be u32, counts f32");
    }
    let dtype = input.dtype();
    let n = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if counts.elem_count() != n_tokens {
        candle_core::bail!("token_ids and counts length mismatch");
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let counts = counts.contiguous()?;

    let (input_s, input_l) = input.storage_and_layout();
    let (tok_s, tok_l) = token_ids.storage_and_layout();
    let (cnt_s, cnt_l) = counts.storage_and_layout();
    let (Storage::Metal(input_s), Storage::Metal(tok_s), Storage::Metal(cnt_s)) =
        (&*input_s, &*tok_s, &*cnt_s)
    else {
        candle_core::bail!("metal_apply_sparse_penalties requires Metal tensors");
    };
    let device = input_s.device().clone();

    let out_buf = device.new_buffer(n, dtype, "penalties-out")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("penalties-copy");
    {
        use mistralrs_quant::metal_kernels::Kernels;
        mistralrs_quant::metal_kernels::call_copy_logits(
            device.device(),
            &encoder,
            &Kernels::new(),
            dtype,
            input_s.buffer(),
            input_l.start_offset() * input.dtype().size_in_bytes(),
            &out_buf,
            n,
        )
        .map_err(|e| candle_core::Error::Msg(format!("metal copy: {e}")))?;
    }
    encoder.set_label("penalties-apply");
    mistralrs_quant::metal_kernels::call_apply_sparse_penalties(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        dtype,
        &out_buf,
        tok_s.buffer(),
        cnt_s.buffer(),
        n,
        n_tokens,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
    )
    .map_err(|e| candle_core::Error::Msg(format!("metal penalties: {e}")))?;
    let _ = (tok_l, cnt_l);
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(out_buf, device.clone(), n, dtype)),
        Shape::from(input.dims()),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn try_cuda_rms_norm_strided_4d(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !input.device().is_cuda() || input.rank() != 4 {
        return Ok(None);
    }

    let dtype = input.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) || weight.dtype() != dtype {
        return Ok(None);
    }
    if !weight.device().same_device(input.device()) {
        return Ok(None);
    }

    let (batch, heads, seq_len, head_dim) = input.dims4()?;
    if weight.dims1()? != head_dim {
        candle_core::bail!(
            "cuda_rms_norm_strided_4d weight size {} does not match head dim {head_dim}",
            weight.dims1()?
        );
    }
    if input.elem_count() == 0 {
        return Ok(None);
    }
    for (name, value) in [
        ("batch", batch),
        ("heads", heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("cuda_rms_norm_strided_4d {name} is too large: {value}");
        }
    }

    let (input_storage, input_layout) = input.storage_and_layout();
    if input_layout.is_contiguous() {
        return Ok(None);
    }
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let weight = weight.contiguous()?;
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();
    let elem_count = input.elem_count();
    let stride = input_layout.stride();
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let heads_i32 = i32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_strided_4d input dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_rms_norm_strided_4d weight dtype mismatch");
            };
            let mut out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    stride[0] as i64,
                    stride[1] as i64,
                    stride[2] as i64,
                    stride[3] as i64,
                    batch_i32,
                    heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(weight_guard);
            drop(out_guard);

            let out_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(out),
                device: dev.clone(),
            };
            Ok(Some(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                shape,
            ))))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_strided_4d_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_strided_4d_f16),
        DType::F32 => launch!(F32, f32, rms_norm_strided_4d_f32),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_rms_norm_residual(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    scale: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_rms_norm_residual input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        candle_core::bail!(
            "cuda_rms_norm_residual dtype mismatch: input {:?}, residual {:?}, weight {:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_rms_norm_residual only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        candle_core::bail!("cuda_rms_norm_residual tensors must be on the same CUDA device");
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            candle_core::bail!(
                "cuda_rms_norm_residual scale must have one element, got {}",
                scale.elem_count()
            );
        }
        if scale.dtype() != input.dtype() {
            candle_core::bail!(
                "cuda_rms_norm_residual scale dtype mismatch: input {:?}, scale {:?}",
                input.dtype(),
                scale.dtype()
            );
        }
        if !scale.device().same_device(input.device()) {
            candle_core::bail!("cuda_rms_norm_residual scale must be on the same CUDA device");
        }
    }

    let ncols = input.dim(D::Minus1)?;
    if weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual weight size {} does not match last dim {ncols}",
            weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_rms_norm_residual got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!(
            "cuda_rms_norm_residual input is too large: nrows={nrows}, ncols={ncols}"
        );
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let scale = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA residual"),
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA weight"),
    };
    let scale_storage_and_layout = scale.as_ref().map(|scale| scale.storage_and_layout());

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual weight dtype mismatch");
            };
            let (scale_ptr, scale_guard) =
                if let Some((scale_storage, scale_layout)) = &scale_storage_and_layout {
                    let scale_storage = match &**scale_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA scale"),
                    };
                    let CudaStorageSlice::$variant(scale_src) = &scale_storage.slice else {
                        candle_core::bail!("cuda_rms_norm_residual scale dtype mismatch");
                    };
                    let (scale_ptr, scale_guard) = scale_src.device_ptr(&stream);
                    (
                        unsafe { (scale_ptr as *const $ty).add(scale_layout.start_offset()) }
                            as *const c_void,
                        Some(scale_guard),
                    )
                } else {
                    (std::ptr::null(), None)
                };

            let mut out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    scale_ptr,
                    out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(weight_guard);
            drop(scale_guard);
            drop(out_guard);

            let out_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(out),
                device: dev.clone(),
            };
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                shape,
            )))
        }};
    }

    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_residual_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_residual_f16),
        DType::F32 => launch!(F32, f32, rms_norm_residual_f32),
        dtype => candle_core::bail!("cuda_rms_norm_residual unsupported dtype {dtype:?}"),
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_add_rms_norm input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        candle_core::bail!(
            "cuda_add_rms_norm dtype mismatch: input {:?}, residual {:?}, weight {:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_add_rms_norm only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        candle_core::bail!("cuda_add_rms_norm tensors must be on the same CUDA device");
    }

    let ncols = input.dim(D::Minus1)?;
    if weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_add_rms_norm weight size {} does not match last dim {ncols}",
            weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if ncols == 0 || elem_count == 0 {
        candle_core::bail!("cuda_add_rms_norm got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!("cuda_add_rms_norm input is too large: nrows={nrows}, ncols={ncols}");
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA residual"),
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA weight"),
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm weight dtype mismatch");
            };

            let mut residual_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let mut norm_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (residual_out_ptr, residual_out_guard) = residual_out.device_ptr_mut(&stream);
            let (norm_out_ptr, norm_out_guard) = norm_out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    residual_out_ptr as *mut c_void,
                    norm_out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(weight_guard);
            drop(residual_out_guard);
            drop(norm_out_guard);

            let residual_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(residual_out),
                device: dev.clone(),
            };
            let norm_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(norm_out),
                device: dev.clone(),
            };
            Ok((
                Tensor::from((candle_core::Storage::Cuda(residual_storage), shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(norm_storage), shape)),
            ))
        }};
    }

    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, add_rms_norm_bf16),
        DType::F16 => launch!(F16, half::f16, add_rms_norm_f16),
        DType::F32 => launch!(F32, f32, add_rms_norm_f32),
        dtype => candle_core::bail!("cuda_add_rms_norm unsupported dtype {dtype:?}"),
    }
}

#[cfg(feature = "metal")]
pub fn metal_rms_norm_residual(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    scale: Option<&Tensor>,
    eps: f32,
) -> Result<Option<Tensor>> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if input.shape() != residual.shape() {
        return Ok(None);
    }
    let n_cols = input.dim(D::Minus1)?;
    if weight.dims1()? != n_cols {
        return Ok(None);
    }
    let n_rows = input.elem_count() / n_cols;
    if n_rows == 0 {
        return Ok(None);
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            return Ok(None);
        }
    }

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let scale_t = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Metal(input_storage) = &*input_storage else {
        return Ok(None);
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let Storage::Metal(residual_storage) = &*residual_storage else {
        return Ok(None);
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Metal(weight_storage) = &*weight_storage else {
        return Ok(None);
    };
    let scale_storage_and_layout = scale_t.as_ref().map(|s| s.storage_and_layout());
    let scale_metal = match scale_storage_and_layout.as_ref() {
        Some((s, l)) => {
            let Storage::Metal(s) = &**s else {
                return Ok(None);
            };
            Some((s, l))
        }
        None => None,
    };

    let device = input_storage.device().clone();
    let dtype = input.dtype();
    let out_buf = device.new_buffer(input.elem_count(), dtype, "rmsnorm-residual-out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("rmsnorm-residual");

    let x_offset = input_layout.start_offset() * dtype.size_in_bytes();
    let res_offset = residual_layout.start_offset() * dtype.size_in_bytes();
    let w_offset = weight_layout.start_offset() * dtype.size_in_bytes();
    let scale_arg = scale_metal
        .as_ref()
        .map(|(s, l)| (s.buffer(), l.start_offset() * dtype.size_in_bytes()));

    mistralrs_quant::metal_kernels::call_rmsnorm_residual(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        dtype,
        (input_storage.buffer(), x_offset),
        (residual_storage.buffer(), res_offset),
        (weight_storage.buffer(), w_offset),
        scale_arg,
        &out_buf,
        n_cols,
        n_rows,
        eps,
    )
    .map_err(candle_core::Error::wrap)?;

    let out = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            device.clone(),
            input.elem_count(),
            dtype,
        )),
        Shape::from(input.dims()),
    ));
    Ok(Some(out))
}

#[cfg(feature = "metal")]
#[allow(clippy::cast_possible_truncation)]
pub fn metal_topk_logits_packed(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    const MAX_K: usize = 128;
    const CHUNK_SIZE: usize = 2048;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("metal_topk_logits_packed requires a positive finite temperature");
    }
    let input = input.contiguous()?;
    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("metal_topk_logits_packed requires F32/F16/BF16 logits");
    }
    let dtype = input.dtype();
    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("metal_topk_logits_packed got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > MAX_K {
        candle_core::bail!("metal_topk_logits_packed k={k} must be in [1, {MAX_K}]");
    }
    let nblocks = ncols.div_ceil(CHUNK_SIZE);

    let (input_s, input_l) = input.storage_and_layout();
    let Storage::Metal(input_s) = &*input_s else {
        candle_core::bail!("metal_topk_logits_packed requires Metal tensor");
    };
    let device = input_s.device().clone();

    let block_values_buf = device.new_buffer(nblocks * k, DType::F32, "topk-block-values")?;
    let block_indices_buf = device.new_buffer(nblocks * k, DType::U32, "topk-block-indices")?;
    let block_maxes_buf = device.new_buffer(nblocks, DType::F32, "topk-block-maxes")?;
    let block_sums_buf = device.new_buffer(nblocks, DType::F32, "topk-block-sums")?;
    let packed_buf = device.new_buffer(2 * k + 2, DType::F32, "topk-packed")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("topk-logits-packed");

    let inv_temp = (1.0_f64 / temperature) as f32;
    let input_offset = input_l.start_offset() * input.dtype().size_in_bytes();

    mistralrs_quant::metal_kernels::call_topk_logits_packed(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        dtype,
        input_s.buffer(),
        &block_values_buf,
        &block_indices_buf,
        &block_maxes_buf,
        &block_sums_buf,
        &packed_buf,
        ncols,
        k,
        CHUNK_SIZE,
        inv_temp,
    )
    .map_err(|e| candle_core::Error::Msg(format!("metal_topk_logits_packed kernel error: {e}")))?;
    let _ = (
        input_offset,
        &block_values_buf,
        &block_indices_buf,
        &block_maxes_buf,
        &block_sums_buf,
    );

    let packed = Tensor::from((
        Storage::Metal(MetalStorage::new(
            packed_buf,
            device.clone(),
            2 * k + 2,
            DType::F32,
        )),
        Shape::from(vec![2 * k + 2]),
    ));
    Ok(TopKLogitsPackedOutput {
        packed,
        k,
        _workspace: vec![],
    })
}

#[cfg(feature = "cuda")]
pub fn cuda_rms_norm_residual_then_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    residual_weight: &Tensor,
    scale: Option<&Tensor>,
    norm_weight: &Tensor,
    residual_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype()
        || input.dtype() != residual_weight.dtype()
        || input.dtype() != norm_weight.dtype()
    {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm dtype mismatch: input {:?}, residual {:?}, residual_weight {:?}, norm_weight {:?}",
            input.dtype(),
            residual.dtype(),
            residual_weight.dtype(),
            norm_weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !residual_weight.device().same_device(input.device())
        || !norm_weight.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm tensors must be on the same CUDA device"
        );
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale must have one element, got {}",
                scale.elem_count()
            );
        }
        if scale.dtype() != input.dtype() {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale dtype mismatch: input {:?}, scale {:?}",
                input.dtype(),
                scale.dtype()
            );
        }
        if !scale.device().same_device(input.device()) {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale must be on the same CUDA device"
            );
        }
    }

    let ncols = input.dim(D::Minus1)?;
    if residual_weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm residual weight size {} does not match last dim {ncols}",
            residual_weight.dims1()?
        );
    }
    if norm_weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm norm weight size {} does not match last dim {ncols}",
            norm_weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_rms_norm_residual_then_rms_norm got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm input is too large: nrows={nrows}, ncols={ncols}"
        );
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let residual_weight = residual_weight.contiguous()?;
    let norm_weight = norm_weight.contiguous()?;
    let scale = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA residual"),
    };
    let (residual_weight_storage, residual_weight_layout) = residual_weight.storage_and_layout();
    let residual_weight_storage = match &*residual_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => {
            candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA residual weight")
        }
    };
    let (norm_weight_storage, norm_weight_layout) = norm_weight.storage_and_layout();
    let norm_weight_storage = match &*norm_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA norm weight"),
    };
    let scale_storage_and_layout = scale.as_ref().map(|scale| scale.storage_and_layout());

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual_then_rms_norm input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual_then_rms_norm residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_weight_src) = &residual_weight_storage.slice
            else {
                candle_core::bail!(
                    "cuda_rms_norm_residual_then_rms_norm residual weight dtype mismatch"
                );
            };
            let CudaStorageSlice::$variant(norm_weight_src) = &norm_weight_storage.slice else {
                candle_core::bail!(
                    "cuda_rms_norm_residual_then_rms_norm norm weight dtype mismatch"
                );
            };
            let (scale_ptr, scale_guard) = if let Some((scale_storage, scale_layout)) =
                &scale_storage_and_layout
            {
                let scale_storage = match &**scale_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!(
                        "cuda_rms_norm_residual_then_rms_norm requires CUDA scale"
                    ),
                };
                let CudaStorageSlice::$variant(scale_src) = &scale_storage.slice else {
                    candle_core::bail!("cuda_rms_norm_residual_then_rms_norm scale dtype mismatch");
                };
                let (scale_ptr, scale_guard) = scale_src.device_ptr(&stream);
                (
                    unsafe { (scale_ptr as *const $ty).add(scale_layout.start_offset()) }
                        as *const c_void,
                    Some(scale_guard),
                )
            } else {
                (std::ptr::null(), None)
            };

            let mut residual_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let mut norm_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (residual_weight_ptr, residual_weight_guard) =
                residual_weight_src.device_ptr(&stream);
            let (norm_weight_ptr, norm_weight_guard) = norm_weight_src.device_ptr(&stream);
            let (residual_out_ptr, residual_out_guard) = residual_out.device_ptr_mut(&stream);
            let (norm_out_ptr, norm_out_guard) = norm_out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let residual_weight_ptr = unsafe {
                (residual_weight_ptr as *const $ty).add(residual_weight_layout.start_offset())
            };
            let norm_weight_ptr =
                unsafe { (norm_weight_ptr as *const $ty).add(norm_weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    residual_weight_ptr as *const c_void,
                    scale_ptr,
                    norm_weight_ptr as *const c_void,
                    residual_out_ptr as *mut c_void,
                    norm_out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    residual_eps,
                    norm_eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(residual_weight_guard);
            drop(norm_weight_guard);
            drop(scale_guard);
            drop(residual_out_guard);
            drop(norm_out_guard);

            let residual_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(residual_out),
                device: dev.clone(),
            };
            let norm_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(norm_out),
                device: dev.clone(),
            };
            Ok((
                Tensor::from((candle_core::Storage::Cuda(residual_storage), shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(norm_storage), shape)),
            ))
        }};
    }

    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_residual_then_rms_norm_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_residual_then_rms_norm_f16),
        DType::F32 => launch!(F32, f32, rms_norm_residual_then_rms_norm_f32),
        dtype => {
            candle_core::bail!("cuda_rms_norm_residual_then_rms_norm unsupported dtype {dtype:?}")
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QkRopeOutputLayout {
    HeadsFirst,
    TokensFirst,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cuda_qk_rms_norm_rope(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    q_eps: f32,
    k_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
    output_layout: QkRopeOutputLayout,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || q_weight.dtype() != dtype
        || k_weight.is_some_and(|weight| weight.dtype() != dtype)
        || cos.dtype() != dtype
        || sin.dtype() != dtype
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || k.is_some_and(|k| !k.device().same_device(q.device()) || k.dtype() != dtype)
        || k_weight.is_some_and(|weight| !weight.device().same_device(q.device()))
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    if seq_len == 1 && q.is_contiguous() && k.is_none_or(Tensor::is_contiguous) {
        return Ok(None);
    }

    let (k_heads, k_elem_count) = if let Some(k) = k {
        let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
        if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
            candle_core::bail!(
                "q/k shape mismatch for fused qk norm rope: {:?} vs {:?}",
                q.shape(),
                k.shape()
            );
        }
        let Some(k_weight) = k_weight else {
            candle_core::bail!("missing k norm weight for fused qk norm rope");
        };
        if k_weight.dims1()? != head_dim {
            candle_core::bail!(
                "k norm weight size {} does not match head dim {head_dim}",
                k_weight.dims1()?
            );
        }
        (k_heads, k.elem_count())
    } else {
        (0, 0)
    };

    if q_weight.dims1()? != head_dim {
        candle_core::bail!(
            "q norm weight size {} does not match head dim {head_dim}",
            q_weight.dims1()?
        );
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qk norm rope: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    let cos_batch_stride = if cos_rows == seq_len {
        0
    } else if cos_rows == batch * seq_len {
        seq_len
    } else {
        candle_core::bail!(
            "cos/sin rows {cos_rows} do not match seq_len {seq_len} or batch*seq_len {}",
            batch * seq_len
        );
    };

    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
        ("cos_batch_stride", cos_batch_stride),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qk norm rope {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;
    let cos_batch_stride_i32 = i32::try_from(cos_batch_stride).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.map(Tensor::contiguous).transpose()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_storage_and_layout = k.map(Tensor::storage_and_layout);
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_weight_storage_and_layout = k_weight.as_ref().map(Tensor::storage_and_layout);
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let q_shape = match output_layout {
        QkRopeOutputLayout::HeadsFirst => Shape::from_dims(&[batch, q_heads, seq_len, head_dim]),
        QkRopeOutputLayout::TokensFirst => Shape::from_dims(&[batch, seq_len, q_heads, head_dim]),
    };
    let k_shape = match output_layout {
        QkRopeOutputLayout::HeadsFirst => Shape::from_dims(&[batch, k_heads, seq_len, head_dim]),
        QkRopeOutputLayout::TokensFirst => Shape::from_dims(&[batch, seq_len, k_heads, head_dim]),
    };
    let q_elem_count = q.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_storage_and_layout
        .as_ref()
        .map(|(_, layout)| layout.stride())
        .unwrap_or(&[0, 0, 0, 0]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qk norm rope q dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qk norm rope q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qk norm rope cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qk norm rope sin dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = if k_elem_count == 0 {
                None
            } else {
                Some(unsafe { dev.alloc::<$ty>(k_elem_count) }?)
            };

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };

            let mut k_guard = None;
            let k_ptr = if let Some((k_storage, k_layout)) = &k_storage_and_layout {
                let k_storage = match &**k_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => return Ok(None),
                };
                let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                    candle_core::bail!("fused qk norm rope k dtype mismatch");
                };
                let (ptr, guard) = k_src.device_ptr(&stream);
                k_guard = Some(guard);
                unsafe { (ptr as *const $ty).add(k_layout.start_offset()) }
            } else {
                std::ptr::null()
            };

            let mut k_weight_guard = None;
            let k_weight_ptr =
                if let Some((k_weight_storage, k_weight_layout)) = &k_weight_storage_and_layout {
                    let k_weight_storage = match &**k_weight_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => return Ok(None),
                    };
                    let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                        candle_core::bail!("fused qk norm rope k weight dtype mismatch");
                    };
                    let (ptr, guard) = k_weight_src.device_ptr(&stream);
                    k_weight_guard = Some(guard);
                    unsafe { (ptr as *const $ty).add(k_weight_layout.start_offset()) }
                } else {
                    q_weight_ptr
                };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let mut k_out_guard = None;
            let k_out_ptr = if let Some(k_out_buf) = &mut k_out_buf {
                let (ptr, guard) = k_out_buf.device_ptr_mut(&stream);
                k_out_guard = Some(guard);
                ptr as *mut $ty
            } else {
                std::ptr::null_mut()
            };

            unsafe {
                ffi::qk_rms_norm_rope(
                    q_ptr as *const c_void,
                    k_ptr as *const c_void,
                    q_weight_ptr as *const c_void,
                    k_weight_ptr as *const c_void,
                    cos_ptr as *const c_void,
                    sin_ptr as *const c_void,
                    q_out_ptr as *mut c_void,
                    k_out_ptr as *mut c_void,
                    q_stride[0] as i64,
                    q_stride[1] as i64,
                    q_stride[2] as i64,
                    q_stride[3] as i64,
                    k_stride[0] as i64,
                    k_stride[1] as i64,
                    k_stride[2] as i64,
                    k_stride[3] as i64,
                    batch_i32,
                    q_heads_i32,
                    k_heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    rot_dim_i32,
                    cos_batch_stride_i32,
                    q_eps,
                    k_eps,
                    i32::from(is_neox),
                    $dtype_id,
                    i32::from(output_layout == QkRopeOutputLayout::TokensFirst),
                    stream_ptr,
                );
            }

            drop(q_guard);
            drop(q_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(k_guard);
            drop(k_weight_guard);
            drop(q_out_guard);
            drop(k_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let q_tensor = Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape));

            let k_tensor = if let Some(k_out_buf) = k_out_buf {
                let k_storage = CudaStorage {
                    slice: CudaStorageSlice::$variant(k_out_buf),
                    device: dev.clone(),
                };
                Some(Tensor::from((
                    candle_core::Storage::Cuda(k_storage),
                    k_shape,
                )))
            } else {
                None
            };

            Ok(Some((q_tensor, k_tensor)))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn try_cuda_rope_sincos_positions(
    positions: &Tensor,
    inv_freq: &Tensor,
    dtype: DType,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !positions.device().is_cuda()
        || positions.dtype() != DType::U32
        || inv_freq.dtype() != DType::F32
        || !inv_freq.device().same_device(positions.device())
        || !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
    {
        return Ok(None);
    }

    let rows = positions.dims1()?;
    let width = inv_freq.dims1()?;
    if rows == 0 || width == 0 {
        return Ok(None);
    }
    let rows_i32 = i32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let width_i32 = i32::try_from(width).map_err(candle_core::Error::wrap)?;
    let elements = rows
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::msg("RoPE sincos output size overflow"))?;

    let positions = positions.contiguous()?;
    let inv_freq = inv_freq.contiguous()?;
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let (inv_freq_storage, inv_freq_layout) = inv_freq.storage_and_layout();
    let inv_freq_storage = match &*inv_freq_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
        candle_core::bail!("RoPE sincos positions dtype mismatch");
    };
    let CudaStorageSlice::F32(inv_freq_src) = &inv_freq_storage.slice else {
        candle_core::bail!("RoPE sincos inverse frequency dtype mismatch");
    };

    let dev = positions_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
    let positions_ptr =
        unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };
    let (inv_freq_ptr, inv_freq_guard) = inv_freq_src.device_ptr(&stream);
    let inv_freq_ptr = unsafe { (inv_freq_ptr as *const f32).add(inv_freq_layout.start_offset()) };
    let output_shape = Shape::from_dims(&[rows, width]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let mut cos_buf = unsafe { dev.alloc::<$ty>(elements) }?;
            let mut sin_buf = unsafe { dev.alloc::<$ty>(elements) }?;
            let (cos_ptr, cos_guard) = cos_buf.device_ptr_mut(&stream);
            let (sin_ptr, sin_guard) = sin_buf.device_ptr_mut(&stream);
            unsafe {
                ffi::rope_sincos_positions(
                    positions_ptr as *const c_void,
                    inv_freq_ptr as *const c_void,
                    cos_ptr as *mut c_void,
                    sin_ptr as *mut c_void,
                    rows_i32,
                    width_i32,
                    $dtype_id,
                    stream_ptr,
                );
            }
            drop(cos_guard);
            drop(sin_guard);

            let cos_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(cos_buf),
                device: dev.clone(),
            };
            let sin_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(sin_buf),
                device: dev.clone(),
            };
            let cos = Tensor::from((
                candle_core::Storage::Cuda(cos_storage),
                output_shape.clone(),
            ));
            let sin = Tensor::from((
                candle_core::Storage::Cuda(sin_storage),
                output_shape.clone(),
            ));
            Ok(Some((cos, sin)))
        }};
    }

    let result = match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => unreachable!(),
    };
    drop(positions_guard);
    drop(inv_freq_guard);
    result
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cuda_qk_rms_norm_rope_positions(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    q_eps: f32,
    k_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || q_weight.dtype() != dtype
        || k_weight.is_some_and(|weight| weight.dtype() != dtype)
        || cos.dtype() != dtype
        || sin.dtype() != dtype
        || positions.dtype() != DType::U32
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || !positions.device().same_device(q.device())
        || k.is_some_and(|k| !k.device().same_device(q.device()) || k.dtype() != dtype)
        || k_weight.is_some_and(|weight| !weight.device().same_device(q.device()))
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let expected_positions = batch * seq_len;
    if positions.dims1()? != expected_positions {
        candle_core::bail!(
            "positions length {} does not match token count {expected_positions}",
            positions.dims1()?
        );
    }

    let (k_heads, k_elem_count) = if let Some(k) = k {
        let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
        if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
            candle_core::bail!(
                "q/k shape mismatch for fused qk norm rope positions: {:?} vs {:?}",
                q.shape(),
                k.shape()
            );
        }
        let Some(k_weight) = k_weight else {
            candle_core::bail!("missing k norm weight for fused qk norm rope positions");
        };
        if k_weight.dims1()? != head_dim {
            candle_core::bail!(
                "k norm weight size {} does not match head dim {head_dim}",
                k_weight.dims1()?
            );
        }
        (k_heads, k.elem_count())
    } else {
        (0, 0)
    };

    if q_weight.dims1()? != head_dim {
        candle_core::bail!(
            "q norm weight size {} does not match head dim {head_dim}",
            q_weight.dims1()?
        );
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qk norm rope positions: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qk norm rope positions {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.map(Tensor::contiguous).transpose()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_storage_and_layout = k.map(Tensor::storage_and_layout);
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_weight_storage_and_layout = k_weight.as_ref().map(Tensor::storage_and_layout);
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let q_shape = Shape::from_dims(&[batch, q_heads, seq_len, head_dim]);
    let k_shape = Shape::from_dims(&[batch, k_heads, seq_len, head_dim]);
    let q_elem_count = q.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_storage_and_layout
        .as_ref()
        .map(|(_, layout)| layout.stride())
        .unwrap_or(&[0, 0, 0, 0]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qk norm rope positions q dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qk norm rope positions q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qk norm rope positions cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qk norm rope positions sin dtype mismatch");
            };
            let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
                candle_core::bail!("fused qk norm rope positions dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = if k_elem_count == 0 {
                None
            } else {
                Some(unsafe { dev.alloc::<$ty>(k_elem_count) }?)
            };

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };
            let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
            let positions_ptr =
                unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };

            let mut k_guard = None;
            let k_ptr = if let Some((k_storage, k_layout)) = &k_storage_and_layout {
                let k_storage = match &**k_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => return Ok(None),
                };
                let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                    candle_core::bail!("fused qk norm rope positions k dtype mismatch");
                };
                let (ptr, guard) = k_src.device_ptr(&stream);
                k_guard = Some(guard);
                unsafe { (ptr as *const $ty).add(k_layout.start_offset()) }
            } else {
                std::ptr::null()
            };

            let mut k_weight_guard = None;
            let k_weight_ptr =
                if let Some((k_weight_storage, k_weight_layout)) = &k_weight_storage_and_layout {
                    let k_weight_storage = match &**k_weight_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => return Ok(None),
                    };
                    let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                        candle_core::bail!("fused qk norm rope positions k weight dtype mismatch");
                    };
                    let (ptr, guard) = k_weight_src.device_ptr(&stream);
                    k_weight_guard = Some(guard);
                    unsafe { (ptr as *const $ty).add(k_weight_layout.start_offset()) }
                } else {
                    q_weight_ptr
                };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let mut k_out_guard = None;
            let k_out_ptr = if let Some(k_out_buf) = &mut k_out_buf {
                let (ptr, guard) = k_out_buf.device_ptr_mut(&stream);
                k_out_guard = Some(guard);
                ptr as *mut $ty
            } else {
                std::ptr::null_mut()
            };

            unsafe {
                ffi::qk_rms_norm_rope_positions(
                    q_ptr as *const c_void,
                    k_ptr as *const c_void,
                    q_weight_ptr as *const c_void,
                    k_weight_ptr as *const c_void,
                    cos_ptr as *const c_void,
                    sin_ptr as *const c_void,
                    positions_ptr as *const c_void,
                    q_out_ptr as *mut c_void,
                    k_out_ptr as *mut c_void,
                    q_stride[0] as i64,
                    q_stride[1] as i64,
                    q_stride[2] as i64,
                    q_stride[3] as i64,
                    k_stride[0] as i64,
                    k_stride[1] as i64,
                    k_stride[2] as i64,
                    k_stride[3] as i64,
                    batch_i32,
                    q_heads_i32,
                    k_heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    rot_dim_i32,
                    q_eps,
                    k_eps,
                    i32::from(is_neox),
                    $dtype_id,
                    stream_ptr,
                );
            }

            drop(q_guard);
            drop(q_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(positions_guard);
            drop(k_guard);
            drop(k_weight_guard);
            drop(q_out_guard);
            drop(k_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let q_tensor = Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape));

            let k_tensor = if let Some(k_out_buf) = k_out_buf {
                let k_storage = CudaStorage {
                    slice: CudaStorageSlice::$variant(k_out_buf),
                    device: dev.clone(),
                };
                Some(Tensor::from((
                    candle_core::Storage::Cuda(k_storage),
                    k_shape,
                )))
            } else {
                None
            };

            Ok(Some((q_tensor, k_tensor)))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cuda_qkv_rms_norm_rope_positions(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    q_eps: f32,
    k_eps: f32,
    v_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || k.dtype() != dtype
        || v.dtype() != dtype
        || q_weight.dtype() != dtype
        || k_weight.dtype() != dtype
        || v_weight.dtype() != dtype
        || cos.dtype() != dtype
        || sin.dtype() != dtype
        || positions.dtype() != DType::U32
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !k_weight.device().same_device(q.device())
        || !v_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || !positions.device().same_device(q.device())
        || !k.device().same_device(q.device())
        || !v.device().same_device(q.device())
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    let (v_batch, v_heads, v_seq_len, v_head_dim) = v.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim)
        || (v_batch, v_heads, v_seq_len, v_head_dim) != (batch, k_heads, seq_len, head_dim)
    {
        candle_core::bail!(
            "q/k/v shape mismatch for fused qkv norm rope positions: {:?}, {:?}, {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    let expected_positions = batch * seq_len;
    if positions.dims1()? != expected_positions {
        candle_core::bail!(
            "positions length {} does not match token count {expected_positions}",
            positions.dims1()?
        );
    }
    if q_weight.dims1()? != head_dim
        || k_weight.dims1()? != head_dim
        || v_weight.dims1()? != head_dim
    {
        candle_core::bail!("qkv norm weight size does not match head dim {head_dim}");
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qkv norm rope positions: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qkv norm rope positions {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.contiguous()?;
    let v_weight = v_weight.contiguous()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (k_storage, k_layout) = k.storage_and_layout();
    let k_storage = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (v_storage, v_layout) = v.storage_and_layout();
    let v_storage = match &*v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (k_weight_storage, k_weight_layout) = k_weight.storage_and_layout();
    let k_weight_storage = match &*k_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (v_weight_storage, v_weight_layout) = v_weight.storage_and_layout();
    let v_weight_storage = match &*v_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let q_shape = Shape::from_dims(&[batch, q_heads, seq_len, head_dim]);
    let kv_shape = Shape::from_dims(&[batch, k_heads, seq_len, head_dim]);
    let q_elem_count = q.elem_count();
    let kv_elem_count = k.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_layout.stride();
    let v_stride = v_layout.stride();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions q dtype mismatch");
            };
            let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions k dtype mismatch");
            };
            let CudaStorageSlice::$variant(v_src) = &v_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions v dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions k weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(v_weight_src) = &v_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions v weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions sin dtype mismatch");
            };
            let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = unsafe { dev.alloc::<$ty>(kv_elem_count) }?;
            let mut v_out_buf = unsafe { dev.alloc::<$ty>(kv_elem_count) }?;

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (k_ptr, k_guard) = k_src.device_ptr(&stream);
            let k_ptr = unsafe { (k_ptr as *const $ty).add(k_layout.start_offset()) };
            let (v_ptr, v_guard) = v_src.device_ptr(&stream);
            let v_ptr = unsafe { (v_ptr as *const $ty).add(v_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (k_weight_ptr, k_weight_guard) = k_weight_src.device_ptr(&stream);
            let k_weight_ptr =
                unsafe { (k_weight_ptr as *const $ty).add(k_weight_layout.start_offset()) };
            let (v_weight_ptr, v_weight_guard) = v_weight_src.device_ptr(&stream);
            let v_weight_ptr =
                unsafe { (v_weight_ptr as *const $ty).add(v_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };
            let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
            let positions_ptr =
                unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let (k_out_ptr, k_out_guard) = k_out_buf.device_ptr_mut(&stream);
            let (v_out_ptr, v_out_guard) = v_out_buf.device_ptr_mut(&stream);

            unsafe {
                ffi::qkv_rms_norm_rope_positions(
                    q_ptr as *const c_void,
                    k_ptr as *const c_void,
                    v_ptr as *const c_void,
                    q_weight_ptr as *const c_void,
                    k_weight_ptr as *const c_void,
                    v_weight_ptr as *const c_void,
                    cos_ptr as *const c_void,
                    sin_ptr as *const c_void,
                    positions_ptr as *const c_void,
                    q_out_ptr as *mut c_void,
                    k_out_ptr as *mut c_void,
                    v_out_ptr as *mut c_void,
                    q_stride[0] as i64,
                    q_stride[1] as i64,
                    q_stride[2] as i64,
                    q_stride[3] as i64,
                    k_stride[0] as i64,
                    k_stride[1] as i64,
                    k_stride[2] as i64,
                    k_stride[3] as i64,
                    v_stride[0] as i64,
                    v_stride[1] as i64,
                    v_stride[2] as i64,
                    v_stride[3] as i64,
                    batch_i32,
                    q_heads_i32,
                    k_heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    rot_dim_i32,
                    q_eps,
                    k_eps,
                    v_eps,
                    i32::from(is_neox),
                    $dtype_id,
                    stream_ptr,
                );
            }

            drop(q_guard);
            drop(k_guard);
            drop(v_guard);
            drop(q_weight_guard);
            drop(k_weight_guard);
            drop(v_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(positions_guard);
            drop(q_out_guard);
            drop(k_out_guard);
            drop(v_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let k_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(k_out_buf),
                device: dev.clone(),
            };
            let v_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(v_out_buf),
                device: dev.clone(),
            };
            Ok(Some((
                Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape)),
                Tensor::from((candle_core::Storage::Cuda(k_storage), kv_shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(v_storage), kv_shape)),
            )))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

pub trait TopKLastDimOp {
    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=True.
    fn topk(&self, topk: usize) -> Result<TopKOutput>;

    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=False.
    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Use optimized parallel topk kernel on CUDA
        // Single kernel call, no post-processing overhead
        #[cfg(feature = "cuda")]
        if self.device().is_cuda() {
            return cuda_topk(self, topk);
        }

        // Fallback: full sort (CPU or non-CUDA)
        let (values, sorted_indices) = self.sort_last_dim(false)?;

        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        let topk_values = values.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: topk_values,
            indices: topk_indices,
        })
    }

    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let TopKOutput { values, indices } = self.topk(topk)?;
        // Reorder the indices ascending
        #[cfg(feature = "cuda")]
        let reorder_indices = indices.arg_sort(true)?;
        #[cfg(not(feature = "cuda"))]
        let reorder_indices = indices.arg_sort_last_dim(true)?;
        let topk_indices_unsorted = indices
            .to_dtype(DType::F32)?
            .gather(&reorder_indices, D::Minus1)?
            .to_dtype(DType::U32)?;
        let topk_values_unsorted = values.gather(&reorder_indices, D::Minus1)?;
        Ok(TopKOutput {
            values: topk_values_unsorted,
            indices: topk_indices_unsorted,
        })
    }
}

pub trait RepeatInterleaveOp {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor>;
    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor>;
}

impl RepeatInterleaveOp for Tensor {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor> {
        let dim = dim.to_index(self.shape(), "repeat_interleave")?;
        let dim_elements = self.dim(dim)?;
        // For metal
        assert!(self.dtype().is_float());
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..dim_elements)
                .flat_map(|i| vec![i as u32; repeats])
                .collect::<Vec<_>>(),
            self.device(),
        )?;
        self.index_select(&indices, dim)
    }

    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor> {
        let xs = self.flatten_all()?;
        if repeats.len() != xs.dim(0)? {
            candle_core::bail!(
                "repeats ({}) must match flattened self length ({})",
                repeats.len(),
                xs.dim(0)?
            );
        }
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..xs.dim(0)?)
                .flat_map(|i| vec![i as u32; repeats[i] as usize])
                .collect::<Vec<_>>(),
            xs.device(),
        )?;
        xs.index_select(&indices, 0)
    }
}

pub trait SplitOp {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>>;
}

impl SplitOp for Tensor {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let mut split_res = Vec::new();
        let mut index = 0;
        for split in splits {
            split_res.push(self.narrow(dim, index, *split)?);
            index += *split;
        }
        Ok(split_res)
    }
}

#[allow(dead_code)]
pub trait BincountOp {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>>;
}

#[allow(dead_code)]
fn bincount(values: &[u32], minlength: u32) -> Vec<u32> {
    // let max_val = values.iter().max().copied().unwrap_or(0);
    // let result_len = (max_val + 1).max(minlength);
    // values.iter().fold(
    //     // Start with a histogram vector of zeros.
    //     vec![0u32; result_len as usize],
    //     // For each value, update the histogram.
    //     |mut histogram, &value| {
    //         histogram[value as usize] += 1;
    //         histogram
    //     },
    // )

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    // Early return if there are no values.
    if values.is_empty() {
        return vec![0u32; minlength as usize];
    }

    // Compute the maximum value in parallel.
    // SAFETY: We just checked that values is nonempty above, so max() will return Some.
    // Using expect() for clearer error message if this invariant is somehow violated.
    let max_val = *values
        .par_iter()
        .max()
        .expect("values should be non-empty after empty check");

    // The histogram length must cover all observed values as well as `minlength`.
    let result_len = (max_val + 1).max(minlength) as usize;

    // Build per-thread histograms in parallel.
    // We use unsafe indexing to eliminate bounds checks in the inner loop.
    values
        .par_iter()
        .fold(
            || vec![0u32; result_len],
            |mut local_hist, &v| {
                // SAFETY: v is guaranteed to be <= max_val, so it is in bounds.
                unsafe {
                    *local_hist.get_unchecked_mut(v as usize) += 1;
                }
                local_hist
            },
        )
        // Merge the per-thread histograms in parallel.
        .reduce(
            || vec![0u32; result_len],
            |mut global_hist, local_hist| {
                for i in 0..result_len {
                    // SAFETY: we know local histogram is at least result_len, as is global_hist
                    unsafe {
                        *global_hist.get_unchecked_mut(i) += local_hist.get_unchecked(i);
                    }
                }
                global_hist
            },
        )
}

#[allow(dead_code)]
impl BincountOp for Tensor {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>> {
        let values = self.to_vec1::<u32>()?;

        Ok(bincount(&values, minlength))
    }
}

// https://github.com/mokeyish/candle-ext/blob/ca4547c803469bd51c00ce5eda2f18dd249c8f10/src/triangular.rs#L21
pub fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> Result<Tensor> {
    let device = xs.device();
    let (l, s) = xs.dims2()?;
    let mut xs_tri = vec![];
    for i in 0..l as isize {
        for j in 0..s as isize {
            let cond = if upper {
                i + diagonal > j
            } else {
                i + diagonal < j
            };
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
}

/// Elementwise multiply and activation. The following activations are supported:
/// - `gelu`
/// - `silu`
/// - `relu`
///
/// This is equivalent to:
/// `act(a) * b`
///
/// With supported dtypes (F16, BF16, F32) and fused activations,
/// this uses a fused kernel for better performance by eliminating intermediate
/// memory allocation. Optimized implementations are available for:
/// - CUDA: Custom CUDA kernel with vec4 optimization
/// - Metal: Native Metal kernel
/// - CPU: Rayon-parallelized implementation
fn glu_activation_type(act: Activation) -> Option<mistralrs_quant::GluActivationType> {
    match act {
        Activation::Silu | Activation::Swish => Some(mistralrs_quant::GluActivationType::Silu),
        Activation::NewGelu | Activation::GeluPytorchTanh => {
            Some(mistralrs_quant::GluActivationType::Gelu)
        }
        Activation::Gelu => Some(mistralrs_quant::GluActivationType::GeluErf),
        Activation::Relu => Some(mistralrs_quant::GluActivationType::Relu),
        Activation::Sigmoid => Some(mistralrs_quant::GluActivationType::Sigmoid),
        _ => None,
    }
}

fn candle_glu_activation_type(
    act: candle_nn::Activation,
) -> Option<mistralrs_quant::GluActivationType> {
    match act {
        candle_nn::Activation::Silu | candle_nn::Activation::Swish => {
            Some(mistralrs_quant::GluActivationType::Silu)
        }
        candle_nn::Activation::NewGelu | candle_nn::Activation::GeluPytorchTanh => {
            Some(mistralrs_quant::GluActivationType::Gelu)
        }
        candle_nn::Activation::Gelu => Some(mistralrs_quant::GluActivationType::GeluErf),
        candle_nn::Activation::Relu => Some(mistralrs_quant::GluActivationType::Relu),
        candle_nn::Activation::Sigmoid => Some(mistralrs_quant::GluActivationType::Sigmoid),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatedActivationOrder {
    GateUp,
    UpGate,
}

pub fn mul_and_act(a: &Tensor, b: &Tensor, act: Activation) -> Result<Tensor> {
    // Check if we can use the fused kernel (works on CUDA, Metal, and CPU)
    if matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32) && a.dtype() == b.dtype() {
        if let Some(activation_type) = glu_activation_type(act) {
            return mistralrs_quant::fused_glu(a, b, activation_type);
        }
    }

    a.apply(&act)? * b
}

pub fn mul_and_candle_act(a: &Tensor, b: &Tensor, act: candle_nn::Activation) -> Result<Tensor> {
    // Check if we can use the fused kernel (works on CUDA, Metal, and CPU)
    if matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32) && a.dtype() == b.dtype() {
        if let Some(activation_type) = candle_glu_activation_type(act) {
            return mistralrs_quant::fused_glu(a, b, activation_type);
        }
    }

    a.apply(&act)? * b
}

pub fn split_mul_and_act(xs: &Tensor, split_size: usize, act: Activation) -> Result<Tensor> {
    split_mul_and_act_order(xs, split_size, act, GatedActivationOrder::GateUp)
}

pub fn split_mul_and_act_order(
    xs: &Tensor,
    split_size: usize,
    act: Activation,
    order: GatedActivationOrder,
) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let Some(expected_last_dim) = split_size.checked_mul(2) else {
        candle_core::bail!("split_mul_and_act split size overflow: {split_size}");
    };
    if last_dim != expected_last_dim {
        candle_core::bail!(
            "split_mul_and_act expected last dim {expected_last_dim}, got {last_dim}"
        );
    }
    if order == GatedActivationOrder::GateUp
        && matches!(xs.dtype(), DType::F16 | DType::BF16 | DType::F32)
    {
        if let Some(activation_type) = glu_activation_type(act) {
            return mistralrs_quant::fused_split_glu(xs, split_size, activation_type);
        }
    }

    let first = xs.narrow(D::Minus1, 0, split_size)?;
    let second = xs.narrow(D::Minus1, split_size, split_size)?;
    match order {
        GatedActivationOrder::GateUp => mul_and_act(&first, &second, act),
        GatedActivationOrder::UpGate => mul_and_act(&second, &first, act),
    }
}

#[derive(Clone)]
pub(crate) struct MergedDenseProjection {
    proj: Arc<dyn mistralrs_quant::QuantMethod>,
    originals: Vec<Arc<dyn mistralrs_quant::QuantMethod>>,
    output_dims: Vec<usize>,
}

impl MergedDenseProjection {
    /// Wrap a packed projection group: `packed` owns the fused weight, `constituents` are its
    /// view-backed layers used for the dynamic-LoRA fallback path.
    pub(crate) fn from_packed(group: &mistralrs_quant::PackedColumnParallel) -> Self {
        Self {
            proj: group.packed.clone(),
            originals: group.constituents.clone(),
            output_dims: group.rows_per_rank.clone(),
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let Some(ys) = self.forward_packed(xs)? else {
            return self.originals.iter().map(|proj| proj.forward(xs)).collect();
        };
        let mut parts = Vec::with_capacity(self.output_dims.len());
        let mut offset = 0;
        for &dim in &self.output_dims {
            parts.push(ys.narrow(D::Minus1, offset, dim)?);
            offset += dim;
        }
        Ok(parts)
    }

    pub(crate) fn forward_packed(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        if self
            .originals
            .iter()
            .any(|proj| proj.is_dynamic_lora_active())
        {
            Ok(None)
        } else {
            self.proj.forward(xs).map(Some)
        }
    }
}

/// Feed-forward path for quantized gate/up/down projections.
pub(crate) fn quantized_ffn(
    xs: &Tensor,
    gate: &dyn mistralrs_quant::QuantMethod,
    up: &dyn mistralrs_quant::QuantMethod,
    down: &dyn mistralrs_quant::QuantMethod,
    act: Activation,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(activation_type) = glu_activation_type(act) {
        if let Some(out) =
            mistralrs_quant::try_fused_quantized_ffn(xs, gate, up, down, activation_type)?
        {
            return Ok(out);
        }
        if let Some(inter) =
            mistralrs_quant::try_fused_quantized_gate_up(xs, gate, up, activation_type)?
        {
            return down.forward(&inter);
        }
    }

    #[cfg(feature = "metal")]
    if let Some(activation_type) = glu_activation_type(act) {
        if let Some(inter) =
            mistralrs_quant::try_fused_gate_up_metal(xs, gate, up, activation_type)?
        {
            return down.forward(&inter);
        }
    }

    if xs.device().is_cpu() {
        if let Some(mut out) = mistralrs_quant::try_fused_gemv_shared_lhs_cpu(xs, &[gate, up])? {
            let rhs = out.pop().unwrap();
            let lhs = out.pop().unwrap();
            let inter = mul_and_act(&lhs, &rhs, act)?;
            return down.forward(&inter);
        }
    }

    let lhs = gate.forward(xs)?;
    let rhs = up.forward(xs)?;
    let inter = mul_and_act(&lhs, &rhs, act)?;
    down.forward(&inter)
}

pub(crate) fn qkv_projections(
    xs: &Tensor,
    q_proj: &dyn mistralrs_quant::QuantMethod,
    k_proj: &dyn mistralrs_quant::QuantMethod,
    v_proj: &dyn mistralrs_quant::QuantMethod,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if let Some(qkv) = mistralrs_quant::try_fused_quantized_qkv(xs, q_proj, k_proj, v_proj)? {
        return Ok(qkv);
    }

    #[cfg(feature = "metal")]
    if let Some(qkv) = mistralrs_quant::try_fused_qkv_metal(xs, q_proj, k_proj, v_proj)? {
        return Ok(qkv);
    }

    if xs.device().is_cpu() {
        if let Some(mut out) =
            mistralrs_quant::try_fused_gemv_shared_lhs_cpu(xs, &[q_proj, k_proj, v_proj])?
        {
            let v = out.pop().unwrap();
            let k = out.pop().unwrap();
            let q = out.pop().unwrap();
            return Ok((q, k, v));
        }
    }

    Ok((
        q_proj.forward(xs)?,
        k_proj.forward(xs)?,
        v_proj.forward(xs)?,
    ))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;
    use mistralrs_quant::{
        maybe_wrap_dynamic_lora, with_lora_execution, LoraExecution, LoraLayerRegistry,
        LoraLinearSpec, LoraWeights, QuantMethod, QuantMethodConfig, ShardedSafeTensors,
        UnquantLinear,
    };

    use super::MergedDenseProjection;

    #[cfg(feature = "cuda")]
    const CUDA_F32_REL_TOLERANCE: f32 = 1e-5;
    #[cfg(feature = "cuda")]
    const CUDA_BF16_ABS_TOLERANCE: f32 = 2e-2;
    #[cfg(feature = "cuda")]
    const CUDA_LOGPROB_REL_TOLERANCE: f32 = 1e-4;

    #[cfg(feature = "cuda")]
    fn assert_close(actual: f32, expected: f32, relative_tolerance: f32) {
        let tolerance = relative_tolerance * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[cfg(feature = "cuda")]
    fn ranked_topk(packed: Tensor, k: usize) -> super::RankedTopKPackedOutput {
        super::RankedTopKPackedOutput {
            packed,
            k,
            _workspace: Vec::new(),
        }
    }

    #[cfg(feature = "cuda")]
    struct DFlashSelectorReference<'a> {
        packed_topk: &'a [f32],
        hidden: &'a [f32],
        predecessor_codebook: &'a [f32],
        successor_codebook: &'a [f32],
        anchors: &'a [u32],
        positions: usize,
        rank: usize,
        vocab: usize,
        k: usize,
    }

    #[cfg(feature = "cuda")]
    fn dflash_selector_reference(input: DFlashSelectorReference<'_>) -> Vec<u32> {
        let packed_width = 2 * input.k;
        let mut selected = Vec::with_capacity(input.anchors.len() * input.positions);
        for (batch, anchor) in input.anchors.iter().enumerate() {
            let mut predecessor = *anchor as usize;
            for position in 0..input.positions {
                let row = batch * input.positions + position;
                let packed = &input.packed_topk[row * packed_width..(row + 1) * packed_width];
                let hidden = &input.hidden[row * input.rank..(row + 1) * input.rank];
                let pred = &input.predecessor_codebook
                    [predecessor * input.rank..(predecessor + 1) * input.rank];
                let mut best_score = f32::NEG_INFINITY;
                let mut best_token = packed[input.k] as u32;
                for candidate_slot in 0..input.k {
                    let candidate = packed[input.k + candidate_slot] as usize;
                    assert!(candidate < input.vocab);
                    let succ = &input.successor_codebook
                        [candidate * input.rank..(candidate + 1) * input.rank];
                    let dot = pred
                        .iter()
                        .zip(hidden)
                        .zip(succ)
                        .map(|((pred, hidden), succ)| pred * hidden * succ)
                        .sum::<f32>();
                    let score = packed[candidate_slot] + dot;
                    if score > best_score {
                        best_score = score;
                        best_token = candidate as u32;
                    }
                }
                selected.push(best_token);
                predecessor = best_token as usize;
            }
        }
        selected
    }

    #[cfg(feature = "cuda")]
    fn dflash_sample_selector_reference(
        input: DFlashSelectorReference<'_>,
        inverse_temperatures: &[f32],
        uniforms: &[f32],
    ) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let packed_width = 2 * input.k;
        let mut selected = Vec::with_capacity(input.anchors.len() * input.positions);
        let mut candidate_ids = Vec::with_capacity(selected.capacity() * input.k);
        let mut candidate_probs = Vec::with_capacity(candidate_ids.capacity());
        for (batch, anchor) in input.anchors.iter().enumerate() {
            let mut predecessor = *anchor as usize;
            for position in 0..input.positions {
                let row = batch * input.positions + position;
                let packed = &input.packed_topk[row * packed_width..(row + 1) * packed_width];
                let hidden = &input.hidden[row * input.rank..(row + 1) * input.rank];
                let pred = &input.predecessor_codebook
                    [predecessor * input.rank..(predecessor + 1) * input.rank];
                let mut scores = Vec::with_capacity(input.k);
                for candidate_slot in 0..input.k {
                    let candidate = packed[input.k + candidate_slot] as usize;
                    let succ = &input.successor_codebook
                        [candidate * input.rank..(candidate + 1) * input.rank];
                    let dot = pred
                        .iter()
                        .zip(hidden)
                        .zip(succ)
                        .map(|((pred, hidden), succ)| pred * hidden * succ)
                        .sum::<f32>();
                    candidate_ids.push(candidate as u32);
                    scores.push(packed[candidate_slot] + dot);
                }

                let inverse_temperature = inverse_temperatures[batch];
                let selected_slot = if inverse_temperature <= 0.0 {
                    let mut selected_slot = 0;
                    let mut best_score = f32::NEG_INFINITY;
                    for (candidate_slot, score) in scores.iter().enumerate() {
                        if *score > best_score {
                            best_score = *score;
                            selected_slot = candidate_slot;
                        }
                    }
                    candidate_probs.extend((0..input.k).map(|candidate_slot| {
                        if candidate_slot == selected_slot {
                            1.0
                        } else {
                            0.0
                        }
                    }));
                    selected_slot
                } else {
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights = scores
                        .iter()
                        .map(|score| ((score - max_score) * inverse_temperature).exp())
                        .collect::<Vec<_>>();
                    let denominator = weights.iter().sum::<f32>();
                    candidate_probs.extend(weights.iter().map(|weight| weight / denominator));
                    let target = uniforms[row] * denominator;
                    let mut cumulative = 0.0f32;
                    weights
                        .iter()
                        .position(|weight| {
                            cumulative += weight;
                            target < cumulative
                        })
                        .unwrap()
                };
                predecessor = packed[input.k + selected_slot] as usize;
                selected.push(predecessor as u32);
            }
        }
        (selected, candidate_ids, candidate_probs)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_add_rms_norm_matches_separate_ops() -> candle_core::Result<()> {
        const ROWS: usize = 2;
        const COLS: usize = 16;
        const EPS: f32 = 1e-6;

        let device = Device::new_cuda(0)?;
        let input = Tensor::from_vec(
            (0..ROWS * COLS)
                .map(|index| index as f32 * 0.03125 - 0.4)
                .collect::<Vec<_>>(),
            (ROWS, COLS),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let residual = Tensor::from_vec(
            (0..ROWS * COLS)
                .map(|index| 0.25 - index as f32 * 0.015625)
                .collect::<Vec<_>>(),
            (ROWS, COLS),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let weight = Tensor::from_vec(
            (0..COLS)
                .map(|index| 0.75 + index as f32 * 0.01)
                .collect::<Vec<_>>(),
            COLS,
            &device,
        )?
        .to_dtype(DType::BF16)?;

        let expected_sum = (&input + &residual)?;
        let expected_norm = candle_nn::ops::rms_norm(&expected_sum.contiguous()?, &weight, EPS)?;
        let (actual_sum, actual_norm) = super::cuda_add_rms_norm(&input, &residual, &weight, EPS)?;

        let expected_sum = expected_sum
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual_sum = actual_sum
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual_sum, expected_sum);

        let expected_norm = expected_norm
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual_norm = actual_norm
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (actual, expected) in actual_norm.into_iter().zip(expected_norm) {
            assert!((actual - expected).abs() <= CUDA_BF16_ABS_TOLERANCE);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn cuda_qk_norm_rope_writes_token_major_from_packed_projection() -> candle_core::Result<()> {
        const BATCH: usize = 2;
        const SEQ_LEN: usize = 3;
        const Q_HEADS: usize = 2;
        const K_HEADS: usize = 1;
        const HEAD_DIM: usize = 8;
        const EPS: f32 = 1e-6;

        let device = Device::new_cuda(0)?;
        let packed_width = (Q_HEADS + K_HEADS + K_HEADS) * HEAD_DIM;
        let packed = Tensor::arange(0f32, (BATCH * SEQ_LEN * packed_width) as f32, &device)?
            .affine(0.003, -0.4)?
            .to_dtype(DType::BF16)?
            .reshape((BATCH, SEQ_LEN, packed_width))?;
        let q = packed
            .narrow(2, 0, Q_HEADS * HEAD_DIM)?
            .reshape((BATCH, SEQ_LEN, Q_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let k = packed
            .narrow(2, Q_HEADS * HEAD_DIM, K_HEADS * HEAD_DIM)?
            .reshape((BATCH, SEQ_LEN, K_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let q_weight = Tensor::arange(0f32, HEAD_DIM as f32, &device)?
            .affine(0.02, 0.8)?
            .to_dtype(DType::BF16)?;
        let k_weight = Tensor::arange(0f32, HEAD_DIM as f32, &device)?
            .affine(-0.015, 1.1)?
            .to_dtype(DType::BF16)?;
        let angles = Tensor::arange(0f32, (BATCH * SEQ_LEN * HEAD_DIM / 2) as f32, &device)?
            .affine(0.01, 0.0)?
            .reshape((BATCH, SEQ_LEN, HEAD_DIM / 2))?;
        let cos = angles.cos()?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.to_dtype(DType::BF16)?;

        let expected_q = candle_nn::rotary_emb::rope(
            &candle_nn::ops::rms_norm(&q.contiguous()?, &q_weight, EPS)?,
            &cos,
            &sin,
        )?
        .transpose(1, 2)?
        .contiguous()?;
        let expected_k = candle_nn::rotary_emb::rope(
            &candle_nn::ops::rms_norm(&k.contiguous()?, &k_weight, EPS)?,
            &cos,
            &sin,
        )?
        .transpose(1, 2)?
        .contiguous()?;
        let (actual_q, actual_k) = super::try_cuda_qk_rms_norm_rope(
            &q,
            Some(&k),
            &q_weight,
            Some(&k_weight),
            EPS,
            EPS,
            &cos.reshape((BATCH * SEQ_LEN, HEAD_DIM / 2))?,
            &sin.reshape((BATCH * SEQ_LEN, HEAD_DIM / 2))?,
            true,
            super::QkRopeOutputLayout::TokensFirst,
        )?
        .expect("supported CUDA Q/K fusion");
        let actual_k = actual_k.expect("K output");
        assert_eq!(actual_q.dims4()?, (BATCH, SEQ_LEN, Q_HEADS, HEAD_DIM));
        assert_eq!(actual_k.dims4()?, (BATCH, SEQ_LEN, K_HEADS, HEAD_DIM));

        for (actual, expected) in actual_q
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .zip(
                expected_q
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            )
            .chain(
                actual_k
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .zip(
                        expected_k
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    ),
            )
        {
            assert!((actual - expected).abs() <= CUDA_BF16_ABS_TOLERANCE);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn packed_reference(logits: &[f32], k: usize, inverse_temperature: f32) -> Vec<f32> {
        let mut indices = (0..logits.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&lhs, &rhs| logits[rhs].total_cmp(&logits[lhs]));
        indices.truncate(k.min(logits.len()));

        let global_max =
            logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inverse_temperature;
        let denominator = logits
            .iter()
            .map(|value| (value * inverse_temperature - global_max).exp())
            .sum::<f32>();
        let mut packed = indices
            .iter()
            .map(|&index| logits[index])
            .collect::<Vec<_>>();
        packed.extend(
            indices
                .into_iter()
                .map(|index| f32::from(u16::try_from(index).expect("test vocabulary fits u16"))),
        );
        packed.extend([denominator, global_max]);
        packed
    }

    #[cfg(feature = "cuda")]
    fn categorical_reference(logits: &[f32], inverse_temperature: f32, uniform: f32) -> [f32; 2] {
        let global_max = logits
            .iter()
            .map(|value| value * inverse_temperature)
            .fold(f32::NEG_INFINITY, f32::max);
        let weights = logits
            .iter()
            .map(|value| (value * inverse_temperature - global_max).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        let target = uniform * denominator;
        let mut cumulative = 0.0f32;
        let token = weights
            .iter()
            .position(|weight| {
                cumulative += weight;
                target < cumulative
            })
            .expect("valid categorical distribution");
        [
            f32::from(u16::try_from(token).expect("test vocabulary fits u16")),
            logits[token] * inverse_temperature - global_max - denominator.ln(),
        ]
    }

    #[test]
    fn merged_projection_uses_dynamic_lora_constituents_when_active() -> candle_core::Result<()> {
        let registry = Arc::new(LoraLayerRegistry::new());
        let vb =
            ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
                .with_lora_registry(registry.clone());
        let packed_weight =
            Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.], [1., -1.]], &Device::Cpu)?;
        let packed = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(packed_weight.clone(), None),
        ))?) as Arc<dyn QuantMethod>;
        let gate_view = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(packed_weight.narrow(0, 0, 2)?, None),
        ))?) as Arc<dyn QuantMethod>;
        let up_view = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(packed_weight.narrow(0, 2, 2)?, None),
        ))?) as Arc<dyn QuantMethod>;
        let gate =
            maybe_wrap_dynamic_lora(&vb.pp("gate"), gate_view, LoraLinearSpec::replicated(2, 2))?;
        let up = maybe_wrap_dynamic_lora(&vb.pp("up"), up_view, LoraLinearSpec::replicated(2, 2))?;
        registry.finalize()?;
        let merged = MergedDenseProjection::from_packed(&mistralrs_quant::PackedColumnParallel {
            packed,
            constituents: vec![gate, up],
            rows_per_rank: vec![2, 2],
        });
        let input = Tensor::new(&[[2f32, 3.]], &Device::Cpu)?;
        let base = merged.forward(&input)?;
        assert_eq!(base[0].to_vec2::<f32>()?, vec![vec![2., 3.]]);
        assert_eq!(base[1].to_vec2::<f32>()?, vec![vec![5., -1.]]);

        let gate_site = registry
            .sites()
            .into_iter()
            .find(|site| site.key().path() == "gate")
            .expect("gate site");
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        execution.insert(
            &gate_site,
            0,
            LoraWeights::new(
                Tensor::new(&[[1f32, 0.]], &Device::Cpu)?,
                Tensor::new(&[[1f32], [0.]], &Device::Cpu)?,
                2.0,
            )?,
        )?;
        let active = with_lora_execution(Some(Arc::new(execution)), || merged.forward(&input))?;
        assert_eq!(active[0].to_vec2::<f32>()?, vec![vec![6., 3.]]);
        assert_eq!(active[1].to_vec2::<f32>()?, vec![vec![5., -1.]]);
        Ok(())
    }

    #[test]
    fn merged_projection_keeps_multirow_gate_up_packed() -> candle_core::Result<()> {
        let packed_weight =
            Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.], [1., -1.]], &Device::Cpu)?;
        let packed = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(packed_weight, None),
        ))?) as Arc<dyn QuantMethod>;
        let dummy = || {
            Arc::new(mistralrs_quant::DummyLayer::placeholder(
                mistralrs_quant::DummyLayerInfo::unknown(),
            )) as Arc<dyn QuantMethod>
        };
        let merged = MergedDenseProjection::from_packed(&mistralrs_quant::PackedLinear {
            packed,
            constituents: vec![dummy(), dummy()],
            rows_per_rank: vec![2, 2],
        });
        let input = Tensor::new(&[[2f32, 3.], [4., 5.]], &Device::Cpu)?;
        let packed_output = merged
            .forward_packed(&input)?
            .expect("inactive constituents keep the packed path");
        assert_eq!(packed_output.dims(), &[2, 4]);
        let actual = super::split_mul_and_act(&packed_output, 2, crate::layers::Activation::Silu)?;
        let gate = packed_output
            .narrow(candle_core::D::Minus1, 0, 2)?
            .contiguous()?;
        let up = packed_output
            .narrow(candle_core::D::Minus1, 2, 2)?
            .contiguous()?;
        let expected = super::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        assert_eq!(actual.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_topk() {
        use crate::ops::{TopKLastDimOp, TopKOutput};
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        //  [[1, 3, 5],
        //   [2, 4, 6]]
        let x = Tensor::arange(1f32, 7f32, &device)
            .unwrap()
            .reshape((3, 2))
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let TopKOutput { values, indices } = x.topk(2).unwrap();
        assert_eq!(
            x.to_vec2::<f32>().unwrap(),
            vec![vec![1f32, 3f32, 5f32], vec![2f32, 4f32, 6f32]]
        );
        assert_eq!(
            values.to_vec2::<f32>().unwrap(),
            vec![vec![5f32, 3f32], vec![6f32, 4f32]]
        );
        assert_eq!(
            indices.to_vec2::<u32>().unwrap(),
            vec![vec![2u32, 1u32], vec![2u32, 1u32]]
        );
    }

    #[test]
    fn test_repeat_interleave() -> candle_core::Result<()> {
        use crate::ops::RepeatInterleaveOp;
        use candle_core::{Device, Tensor};

        let input = Tensor::new(
            vec![vec![vec![1f32, 2., 3.], vec![4f32, 5., 6.]]],
            &Device::Cpu,
        )?;

        let repeat_interleaved = input.repeat_interleave(2, 2)?;
        assert_eq!(
            repeat_interleaved.to_vec3::<f32>()?,
            vec![vec![
                vec![1., 1., 2., 2., 3., 3.],
                vec![4., 4., 5., 5., 6., 6.]
            ]]
        );

        Ok(())
    }

    #[test]
    fn test_repeat_interleave_flat() -> candle_core::Result<()> {
        use crate::ops::RepeatInterleaveOp;
        use candle_core::{Device, Tensor};

        let input = Tensor::new(vec![1., 2., 3., 4.], &Device::Cpu)?;

        let repeat_interleaved = input.repeat_interleave_flat(vec![1u32, 2u32, 3u32, 4u32])?;
        assert_eq!(
            repeat_interleaved.to_vec1::<f64>()?,
            vec![1., 2., 2., 3., 3., 3., 4., 4., 4., 4.]
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batched_topk_matches_cpu_with_offsets_and_mixed_temperatures() -> candle_core::Result<()>
    {
        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(
            &[
                [90.0f32, 91.0, 92.0, 93.0],
                [-1.0, 4.0, 0.0, 2.0],
                [3.0, 1.0, 5.0, -2.0],
            ],
            &device,
        )?
        .narrow(0, 1, 2)?;
        let inverse_temperatures = Tensor::new(&[99.0f32, 1.0, 0.5], &device)?.narrow(0, 1, 2)?;

        let output = super::cuda_topk_logits_f32_packed_batched(&logits, 8, &inverse_temperatures)?;
        assert_eq!(output.k, 4);
        let actual = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let expected = [
            packed_reference(&[-1.0, 4.0, 0.0, 2.0], 4, 1.0),
            packed_reference(&[3.0, 1.0, 5.0, -2.0], 4, 0.5),
        ];

        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            for (&actual, &expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_close(actual, expected, CUDA_F32_REL_TOLERANCE);
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batched_topk_rejects_nan_distribution() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(&[[1.0f32, f32::NAN, 3.0, 2.0]], &device)?;
        let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
        let output = super::cuda_topk_logits_f32_packed_batched(&logits, 2, &inverse_temperatures)?;
        let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        assert!(packed[0][2 * output.k].is_nan());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cached_batched_top1_tracks_batch_shape() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let mut workspace = None;
        let first = Tensor::new(&[[1.0f32, 4.0, 3.0], [8.0, 2.0, 5.0]], &device)?;
        let actual = super::cuda_top1_logits_f32_packed_batched_cached(&first, &mut workspace)?;
        assert_eq!(actual, vec![[4.0, 1.0], [8.0, 0.0]]);
        assert_eq!(workspace.as_ref().unwrap().nrows, 2);

        let second = Tensor::new(&[[0.0f32, -2.0, 7.0]], &device)?;
        let actual = super::cuda_top1_logits_f32_packed_batched_cached(&second, &mut workspace)?;
        assert_eq!(actual, vec![[7.0, 2.0]]);
        assert_eq!(workspace.as_ref().unwrap().nrows, 1);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_async_top1_device_and_host_tokens_match() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(&[[1.0f32, 7.0, 3.0], [9.0, 2.0, 5.0]], &device)?;
        let resident_input = Tensor::zeros((4, 1), DType::U32, &device)?;
        let mut workspace = None;
        let submission =
            super::cuda_top1_logits_submit_batched_into(&logits, &resident_input, &mut workspace)?;

        assert_eq!(submission.batch_size(), 2);
        assert_eq!(submission.device_tokens().dims(), &[4, 1]);
        let device_tokens = submission
            .device_tokens()
            .narrow(0, 0, 2)?
            .to_vec2::<u32>()?;
        let completion =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;

        assert_eq!(device_tokens, [[1], [0]]);
        assert_eq!(completion.token_ids(), &[1, 0]);
        assert!(completion.packed().is_none());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_async_top1_queues_two_submissions_and_reuses_slots() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let first = Tensor::new(&[[1.0f32, 4.0, 3.0], [8.0, 2.0, 5.0]], &device)?;
        let second = Tensor::new(&[[6.0f32, 4.0, 3.0], [1.0, 2.0, 9.0]], &device)?;
        let mut workspace = None;
        let first = super::cuda_top1_logits_submit_batched(&first, &mut workspace)?;
        let first_slot = first.slot;
        let second = super::cuda_top1_logits_submit_batched(&second, &mut workspace)?;

        assert_ne!(first_slot, second.slot);
        assert!(super::cuda_top1_logits_submit_batched(
            &Tensor::zeros((2, 3), DType::F32, &device)?,
            &mut workspace,
        )
        .is_err());
        let first_tokens =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &first)?
                .token_ids()
                .to_vec();
        let second_tokens =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &second)?
                .token_ids()
                .to_vec();
        assert_eq!(first_tokens, [1, 0]);
        assert_eq!(second_tokens, [0, 2]);

        let third = Tensor::new(&[[1.0f32, 2.0, 8.0], [3.0, 7.0, 4.0]], &device)?;
        let third = super::cuda_top1_logits_submit_batched(&third, &mut workspace)?;
        assert_eq!(third.slot, first_slot);
        let third_tokens =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &third)?
                .token_ids()
                .to_vec();
        assert_eq!(third_tokens, [2, 1]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_async_top1_releases_resident_target_before_host_completion() -> candle_core::Result<()>
    {
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let resident_input = Tensor::zeros((2, 1), DType::U32, &device)?;
        let first_logits = Tensor::new(&[[1.0f32, 7.0], [9.0, 2.0]], &device)?;
        let second_logits = Tensor::new(&[[8.0f32, 1.0], [3.0, 6.0]], &device)?;
        let mut workspace = None;
        let first = super::cuda_top1_logits_submit_batched_into(
            &first_logits,
            &resident_input,
            &mut workspace,
        )?;
        super::cuda_top1_device_tokens_wait_on(workspace.as_mut().unwrap(), &first, &stream)?;
        super::cuda_top1_device_tokens_release_after(workspace.as_mut().unwrap(), &first, &stream)?;
        let second = super::cuda_top1_logits_submit_batched_into(
            &second_logits,
            &resident_input,
            &mut workspace,
        )?;
        super::cuda_top1_device_tokens_release_after(
            workspace.as_mut().unwrap(),
            &second,
            &stream,
        )?;

        let first_tokens =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &first)?
                .token_ids()
                .to_vec();
        let second_tokens =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &second)?
                .token_ids()
                .to_vec();
        assert_eq!(first_tokens, [1, 0]);
        assert_eq!(second_tokens, [0, 1]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_async_top1_resizes_after_completion() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let mut workspace = None;
        let first = Tensor::new(&[[1.0f32, 4.0], [8.0, 2.0]], &device)?;
        let submission = super::cuda_top1_logits_submit_batched(&first, &mut workspace)?;
        super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;
        let first_workspace_id = workspace.as_ref().unwrap().id;

        let second = Tensor::new(&[[1.0f32, 9.0]], &device)?;
        let submission = super::cuda_top1_logits_submit_batched(&second, &mut workspace)?;
        assert_eq!(workspace.as_ref().unwrap().id, first_workspace_id);
        assert_eq!(workspace.as_ref().unwrap().nrows, 1);
        let completion =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;
        assert_eq!(completion.token_ids(), &[1]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_async_top1_marks_nan_token_invalid() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(&[[1.0f32, f32::NAN, 3.0]], &device)?;
        let mut workspace = None;
        let submission = super::cuda_top1_logits_submit_batched(&logits, &mut workspace)?;
        let completion =
            super::cuda_top1_submission_complete(workspace.as_mut().unwrap(), &submission)?;

        assert_eq!(completion.token_ids(), &[super::CUDA_TOP1_INVALID_TOKEN]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_low_precision_top1_matches_f32_across_ties_and_nonfinite_values(
    ) -> candle_core::Result<()> {
        const BACKING_ROWS: usize = 6;
        const ROWS: usize = 5;
        const VOCAB: usize = 4097;

        let device = Device::new_cuda(0)?;
        let mut values = vec![-32.0f32; BACKING_ROWS * VOCAB];
        let row = |row: usize, column: usize| row * VOCAB + column;
        for column in [1, 256, 2048, 3000] {
            values[row(1, column)] = 7.0;
        }
        values[row(2, 2047)] = -2.0;
        values[row(2, 2048)] = 3.5;
        values[row(3, 17)] = f32::NAN;
        values[row(4, 9)] = f32::INFINITY;
        for value in &mut values[row(5, 0)..row(5, VOCAB)] {
            *value = f32::NEG_INFINITY;
        }
        let logits = Tensor::from_vec(values, (BACKING_ROWS, VOCAB), &device)?;

        for dtype in [DType::BF16, DType::F16] {
            let native = logits.to_dtype(dtype)?.narrow(0, 1, ROWS)?;
            let reference = native.to_dtype(DType::F32)?.contiguous()?;
            let mut workspace = None;

            let native_submission =
                super::cuda_top1_logits_submit_batched_packed(&native, &mut workspace)?;
            let (native_tokens, native_packed) = {
                let completion = super::cuda_top1_submission_complete(
                    workspace.as_mut().unwrap(),
                    &native_submission,
                )?;
                (
                    completion.token_ids().to_vec(),
                    completion.packed().unwrap().to_vec(),
                )
            };
            let reference_submission =
                super::cuda_top1_logits_submit_batched_packed(&reference, &mut workspace)?;
            let (reference_tokens, reference_packed) = {
                let completion = super::cuda_top1_submission_complete(
                    workspace.as_mut().unwrap(),
                    &reference_submission,
                )?;
                (
                    completion.token_ids().to_vec(),
                    completion.packed().unwrap().to_vec(),
                )
            };

            assert_eq!(native_tokens, reference_tokens);
            assert_eq!(native_tokens[0], 1);
            assert_eq!(native_tokens[1], 2048);
            assert_eq!(native_tokens[2], super::CUDA_TOP1_INVALID_TOKEN);
            assert_eq!(native_tokens[3], 9);
            assert_eq!(native_tokens[4], 0);
            for (native, reference) in native_packed.iter().zip(reference_packed) {
                assert!(native == &reference || (native.is_nan() && reference.is_nan()));
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batched_topk_orders_ties_by_lowest_index() -> candle_core::Result<()> {
        const VOCAB: usize = 4097;

        let device = Device::new_cuda(0)?;
        let mut row = vec![-10.0f32; VOCAB];
        for index in [1, 256, 300, 2048] {
            row[index] = 5.0;
        }
        let logits = Tensor::from_vec(row, (1, VOCAB), &device)?;
        let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
        let output = super::cuda_topk_logits_f32_packed_batched(&logits, 4, &inverse_temperatures)?;
        let packed = output.packed.to_vec2::<f32>()?;

        assert_eq!(
            &packed[0][output.k..2 * output.k],
            &[1.0, 256.0, 300.0, 2048.0]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batched_topk_low_precision_inputs_match_f32() -> candle_core::Result<()> {
        const ROWS: usize = 3;
        const VOCAB: usize = 4097;
        const K: usize = 17;

        let device = Device::new_cuda(0)?;
        let values = (0..ROWS * VOCAB)
            .map(|index| (((index * 37) % 257) as f32 - 128.0) / 8.0)
            .collect::<Vec<_>>();
        let logits = Tensor::from_vec(values, (ROWS, VOCAB), &device)?;
        let inverse_temperatures = Tensor::new(&[2.0f32, 0.75, 0.125], &device)?.narrow(0, 1, 2)?;

        for dtype in [DType::BF16, DType::F16] {
            let low_precision = logits.to_dtype(dtype)?.narrow(0, 1, 2)?;
            let reference = low_precision.to_dtype(DType::F32)?.contiguous()?;
            let actual =
                super::cuda_topk_logits_packed_batched(&low_precision, K, &inverse_temperatures)?;
            let expected =
                super::cuda_topk_logits_f32_packed_batched(&reference, K, &inverse_temperatures)?;

            assert_eq!(actual.k, expected.k);
            assert_eq!(
                actual.packed.to_vec2::<f32>()?,
                expected.packed.to_vec2::<f32>()?
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn cuda_batched_topk_workspace_reuses_and_grows() -> candle_core::Result<()> {
        const ROWS: usize = 4;
        const VOCAB: usize = 4097;

        let device = Device::new_cuda(0)?;
        let values = (0..ROWS * VOCAB)
            .map(|index| (((index * 37) % 257) as f32 - 128.0) / 8.0)
            .collect::<Vec<_>>();
        let logits = Tensor::from_vec(values, (ROWS, VOCAB), &device)?;
        let inverse_temperatures = Tensor::new(&[1.0f32, 0.75, 0.5, 0.25], &device)?;
        let mut workspace = None;

        let first_logits = logits.narrow(0, 0, 2)?;
        let first_temperatures = inverse_temperatures.narrow(0, 0, 2)?;
        let first = super::cuda_topk_logits_packed_batched_with_workspace(
            &first_logits,
            17,
            &first_temperatures,
            &mut workspace,
        )?;
        let expected =
            super::cuda_topk_logits_packed_batched(&first_logits, 17, &first_temperatures)?;
        assert_eq!(
            first.packed.to_vec2::<f32>()?,
            expected.packed.to_vec2::<f32>()?
        );
        drop(first);
        drop(expected);
        let first_workspace = workspace.as_ref().expect("workspace was allocated");
        let first_id = first_workspace.id;
        assert_eq!(first_workspace.capacity_rows, 2);
        assert_eq!(first_workspace.capacity_k, 32);

        let smaller_logits = logits.narrow(0, 1, 1)?;
        let smaller_temperatures = inverse_temperatures.narrow(0, 1, 1)?;
        super::cuda_topk_logits_packed_batched_with_workspace(
            &smaller_logits,
            8,
            &smaller_temperatures,
            &mut workspace,
        )?;
        assert_eq!(
            workspace.as_ref().expect("workspace was reused").id,
            first_id
        );

        let grown = super::cuda_topk_logits_packed_batched_with_workspace(
            &logits,
            33,
            &inverse_temperatures,
            &mut workspace,
        )?;
        let expected = super::cuda_topk_logits_packed_batched(&logits, 33, &inverse_temperatures)?;
        assert_eq!(
            grown.packed.to_vec2::<f32>()?,
            expected.packed.to_vec2::<f32>()?
        );
        let grown_workspace = workspace.as_ref().expect("workspace was grown");
        assert_ne!(grown_workspace.id, first_id);
        assert_eq!(grown_workspace.capacity_rows, 4);
        assert_eq!(grown_workspace.capacity_k, 64);

        let changed_vocab = Tensor::zeros((1, 2049), DType::F32, &device)?;
        let one_temperature = inverse_temperatures.narrow(0, 0, 1)?;
        let grown_id = grown_workspace.id;
        super::cuda_topk_logits_packed_batched_with_workspace(
            &changed_vocab,
            20,
            &one_temperature,
            &mut workspace,
        )?;
        assert_ne!(
            workspace.as_ref().expect("shape change was applied").id,
            grown_id
        );

        let wrong_temperatures = inverse_temperatures.narrow(0, 0, 2)?;
        let error = match super::cuda_topk_logits_packed_batched_with_workspace(
            &changed_vocab,
            20,
            &wrong_temperatures,
            &mut workspace,
        ) {
            Ok(_) => candle_core::bail!("row temperature shape mismatch must fail"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("inverse temperatures with shape"));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_ranked_topk_matches_cpu_across_dtypes_ties_and_offsets() -> candle_core::Result<()> {
        const BACKING_ROWS: usize = 4;
        const ROWS: usize = 2;
        const VOCAB: usize = 4097;
        const K: usize = 8;

        let device = Device::new_cuda(0)?;
        let mut values = (0..BACKING_ROWS * VOCAB)
            .map(|index| ((index % VOCAB) % 127) as f32 / 8.0)
            .collect::<Vec<_>>();
        for (row, peaks) in [
            &[
                (1, 50.0),
                (256, 50.0),
                (300, 50.0),
                (2048, 50.0),
                (4096, 50.0),
            ][..],
            &[(0, 60.0), (255, 60.0), (1023, 60.0), (3000, 60.0)][..],
        ]
        .into_iter()
        .enumerate()
        {
            let row = row + 1;
            for &(index, value) in peaks {
                values[row * VOCAB + index] = value;
            }
        }
        let logits = Tensor::from_vec(values, (BACKING_ROWS, VOCAB), &device)?;

        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let input = logits.to_dtype(dtype)?.narrow(0, 1, ROWS)?;
            let (_storage, layout) = input.storage_and_layout();
            assert!(layout.start_offset() > 0);
            let reference = input
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            let output = super::cuda_topk_ranked_packed_batched(&input, K)?;
            assert_eq!(output.k, K);
            assert_eq!(output.packed.dims(), &[ROWS, 2 * K]);
            let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

            for (packed_row, reference_row) in packed.iter().zip(&reference) {
                let mut indices = (0..VOCAB).collect::<Vec<_>>();
                indices.sort_unstable_by(|&left, &right| {
                    reference_row[right]
                        .total_cmp(&reference_row[left])
                        .then_with(|| left.cmp(&right))
                });
                for (slot, &index) in indices.iter().take(K).enumerate() {
                    assert_eq!(packed_row[slot], reference_row[index]);
                    assert_eq!(packed_row[K + slot], index as f32);
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_ranked_topk_radix_matches_realistic_vocab_and_cross_chunk_ties(
    ) -> candle_core::Result<()> {
        const BACKING_ROWS: usize = 3;
        const ROWS: usize = 2;
        const VOCAB: usize = 248_320;
        const K: usize = 16;

        let device = Device::new_cuda(0)?;
        let mut values = vec![-32.0f32; BACKING_ROWS * VOCAB];
        for index in 0..VOCAB {
            values[VOCAB + index] = ((index * 37) % 4096) as f32 / 32.0 - 64.0;
            values[2 * VOCAB + index] = 3.0;
        }
        for index in [
            1, 82_775, 82_776, 120_001, 165_551, 165_552, 220_003, 248_319,
        ] {
            values[VOCAB + index] = 256.0;
        }
        let logits = Tensor::from_vec(values, (BACKING_ROWS, VOCAB), &device)?;

        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let input = logits.to_dtype(dtype)?.narrow(0, 1, ROWS)?;
            let reference = input
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            let output = super::cuda_topk_ranked_packed_batched(&input, K)?;
            let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

            for (packed_row, reference_row) in packed.iter().zip(&reference) {
                let mut indices = (0..VOCAB).collect::<Vec<_>>();
                indices.sort_unstable_by(|&left, &right| {
                    reference_row[right]
                        .total_cmp(&reference_row[left])
                        .then_with(|| left.cmp(&right))
                });
                for (slot, &index) in indices.iter().take(K).enumerate() {
                    assert_eq!(packed_row[slot], reference_row[index]);
                    assert_eq!(packed_row[K + slot], index as f32);
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_ranked_topk_radix_preserves_special_value_contract() -> candle_core::Result<()> {
        const VOCAB: usize = 4097;
        const K: usize = 16;

        let device = Device::new_cuda(0)?;
        let mut row = vec![f32::NEG_INFINITY; VOCAB];
        row[1] = f32::NAN;
        row[2] = f32::INFINITY;
        row[3] = -0.0;
        row[4] = 0.0;
        row[5] = -1.0;
        row[6] = 1.0;
        row[1024] = 7.0;
        row[2048] = 7.0;
        row[4096] = 7.0;
        let logits = Tensor::from_vec(row, (1, VOCAB), &device)?;

        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let input = logits.to_dtype(dtype)?;
            let output = super::cuda_topk_ranked_packed_batched(&input, K)?;
            let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let values = &packed[0][..K];
            let indices = &packed[0][K..];

            assert_eq!(
                &indices[..8],
                &[2.0, 1024.0, 2048.0, 4096.0, 6.0, 4.0, 3.0, 5.0]
            );
            assert!(values[0].is_infinite() && values[0].is_sign_positive());
            assert_eq!(&indices[8..], &[0.0; K - 8]);
            assert!(values[8..]
                .iter()
                .all(|value| value.is_infinite() && value.is_sign_negative()));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_ranked_topk_cooperative_boundaries_and_fallback_match_cpu() -> candle_core::Result<()> {
        const VOCAB: usize = 248_320;

        let device = Device::new_cuda(0)?;
        let values = (0..VOCAB)
            .map(|index| ((index * 104_729) % VOCAB) as f32 / 64.0)
            .collect::<Vec<_>>();
        let logits = Tensor::from_vec(values.clone(), (1, VOCAB), &device)?;
        let mut expected_indices = (0..VOCAB).collect::<Vec<_>>();
        expected_indices.sort_unstable_by(|&left, &right| {
            values[right]
                .total_cmp(&values[left])
                .then_with(|| left.cmp(&right))
        });

        for k in [7, 8, 16, 17, 20, 128] {
            let output = super::cuda_topk_ranked_packed_batched(&logits, k)?;
            let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            for (slot, &index) in expected_indices.iter().take(k).enumerate() {
                assert_eq!(packed[0][slot], values[index]);
                assert_eq!(packed[0][k + slot], index as f32);
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn cuda_dflash_selector_matches_reference_with_bf16_codebooks() -> candle_core::Result<()> {
        const BATCH: usize = 2;
        const POSITIONS: usize = 3;
        const K: usize = 4;
        const RANK: usize = 7;
        const VOCAB: usize = 13;

        let rows = BATCH * POSITIONS;
        let packed_width = 2 * K;
        let mut packed = vec![0.0f32; rows * packed_width];
        for row in 0..rows {
            for candidate_slot in 0..K {
                packed[row * packed_width + candidate_slot] =
                    ((row * 3 + candidate_slot * 5) % 7) as f32 * 0.25 - 0.75;
                packed[row * packed_width + K + candidate_slot] =
                    ((row * 3 + candidate_slot * 2 + 1) % VOCAB) as f32;
            }
        }
        let hidden = (0..rows * RANK)
            .map(|index| ((index * 7) % 9) as f32 * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let predecessor = (0..VOCAB * RANK)
            .map(|index| ((index * 5) % 11) as f32 * 0.125 - 0.625)
            .collect::<Vec<_>>();
        let successor = (0..VOCAB * RANK)
            .map(|index| ((index * 3) % 13) as f32 * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let anchors = [2u32, 7];
        let expected = dflash_selector_reference(DFlashSelectorReference {
            packed_topk: &packed,
            hidden: &hidden,
            predecessor_codebook: &predecessor,
            successor_codebook: &successor,
            anchors: &anchors,
            positions: POSITIONS,
            rank: RANK,
            vocab: VOCAB,
            k: K,
        });

        let device = Device::new_cuda(0)?;
        let topk = ranked_topk(Tensor::from_vec(packed, (rows, packed_width), &device)?, K);
        let actual = super::cuda_dflash_greedy_select(
            &topk,
            &Tensor::from_vec(hidden, (rows, RANK), &device)?,
            &Tensor::from_vec(predecessor, (VOCAB, RANK), &device)?.to_dtype(DType::BF16)?,
            &Tensor::from_vec(successor, (VOCAB, RANK), &device)?.to_dtype(DType::BF16)?,
            &Tensor::new(&anchors, &device)?,
        )?
        .to_vec2::<u32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn cuda_dflash_selector_supports_max_k_and_stable_ties() -> candle_core::Result<()> {
        const POSITIONS: usize = 2;
        const K: usize = super::CUDA_DFLASH_SELECTOR_MAX_K;
        const RANK: usize = 3;
        const VOCAB: usize = K;

        let packed_width = 2 * K;
        let mut packed = vec![0.0f32; POSITIONS * packed_width];
        for position in 0..POSITIONS {
            for candidate_slot in 0..K {
                packed[position * packed_width + candidate_slot] = 1.0;
                packed[position * packed_width + K + candidate_slot] =
                    (K - candidate_slot - 1) as f32;
            }
        }

        let device = Device::new_cuda(0)?;
        let topk = ranked_topk(
            Tensor::from_vec(packed, (POSITIONS, packed_width), &device)?,
            K,
        );
        let actual = super::cuda_dflash_greedy_select(
            &topk,
            &Tensor::zeros((POSITIONS, RANK), DType::BF16, &device)?,
            &Tensor::zeros((VOCAB, RANK), DType::F32, &device)?,
            &Tensor::zeros((VOCAB, RANK), DType::F32, &device)?,
            &Tensor::new(&[0u32], &device)?,
        )?
        .to_vec2::<u32>()?;

        assert_eq!(actual, [vec![(K - 1) as u32; POSITIONS]]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn cuda_dflash_sample_selector_matches_sequential_reference() -> candle_core::Result<()> {
        const BATCH: usize = 2;
        const POSITIONS: usize = 3;
        const K: usize = 3;
        const RANK: usize = 2;
        const VOCAB: usize = 7;

        let rows = BATCH * POSITIONS;
        let packed_width = 2 * K;
        let mut packed = vec![0.0f32; rows * packed_width];
        for row in 0..rows {
            for candidate_slot in 0..K {
                packed[row * packed_width + candidate_slot] =
                    ((row * 5 + candidate_slot * 3) % 11) as f32 * 0.2 - 0.8;
                packed[row * packed_width + K + candidate_slot] =
                    ((row + candidate_slot * 2 + 1) % VOCAB) as f32;
            }
        }
        let hidden = (0..rows * RANK)
            .map(|index| ((index * 3) % 7) as f32 * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let predecessor = (0..VOCAB * RANK)
            .map(|index| ((index * 5) % 9) as f32 * 0.125 - 0.375)
            .collect::<Vec<_>>();
        let successor = (0..VOCAB * RANK)
            .map(|index| ((index * 7) % 11) as f32 * 0.1 - 0.4)
            .collect::<Vec<_>>();
        let anchors = [2u32, 5];
        let inverse_temperatures = [0.0f32, 0.75];
        let uniforms = [f32::NAN, f32::NAN, f32::NAN, 0.1, 0.7, 0.4];
        let (expected_tokens, expected_ids, expected_probs) = dflash_sample_selector_reference(
            DFlashSelectorReference {
                packed_topk: &packed,
                hidden: &hidden,
                predecessor_codebook: &predecessor,
                successor_codebook: &successor,
                anchors: &anchors,
                positions: POSITIONS,
                rank: RANK,
                vocab: VOCAB,
                k: K,
            },
            &inverse_temperatures,
            &uniforms,
        );

        let device = Device::new_cuda(0)?;
        let topk = ranked_topk(Tensor::from_vec(packed, (rows, packed_width), &device)?, K);
        let output = super::cuda_dflash_sample_select(super::DFlashSelectorSampleInput {
            topk: &topk,
            projected_hidden: &Tensor::from_vec(hidden, (rows, RANK), &device)?,
            predecessor_codebook: &Tensor::from_vec(predecessor, (VOCAB, RANK), &device)?,
            successor_codebook: &Tensor::from_vec(successor, (VOCAB, RANK), &device)?,
            anchors: &Tensor::new(&anchors, &device)?,
            inverse_temperatures: &Tensor::new(&inverse_temperatures, &device)?,
            uniforms: &Tensor::from_vec(uniforms.to_vec(), (BATCH, POSITIONS), &device)?,
        })?;
        let actual_tokens = output
            .tokens
            .to_vec2::<u32>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let actual_ids = output
            .candidate_ids
            .to_vec3::<u32>()?
            .into_iter()
            .flatten()
            .flatten()
            .collect::<Vec<_>>();
        let actual_probs = output
            .candidate_probs
            .to_vec3::<f32>()?
            .into_iter()
            .flatten()
            .flatten()
            .collect::<Vec<_>>();

        assert_eq!(actual_tokens, expected_tokens);
        assert_eq!(actual_ids, expected_ids);
        for (actual, expected) in actual_probs.into_iter().zip(expected_probs) {
            assert_close(actual, expected, CUDA_LOGPROB_REL_TOLERANCE);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn cuda_dflash_sample_selector_marks_invalid_sampling_params() -> candle_core::Result<()> {
        const K: usize = 2;
        const VOCAB: usize = 2;
        const PACKED_WIDTH: usize = 2 * K;

        let device = Device::new_cuda(0)?;
        let topk = ranked_topk(
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, PACKED_WIDTH), &device)?,
            K,
        );
        let output = super::cuda_dflash_sample_select(super::DFlashSelectorSampleInput {
            topk: &topk,
            projected_hidden: &Tensor::zeros((1, 1), DType::F32, &device)?,
            predecessor_codebook: &Tensor::zeros((VOCAB, 1), DType::F32, &device)?,
            successor_codebook: &Tensor::zeros((VOCAB, 1), DType::F32, &device)?,
            anchors: &Tensor::new(&[0u32], &device)?,
            inverse_temperatures: &Tensor::new(&[f32::INFINITY], &device)?,
            uniforms: &Tensor::new(&[[0.5f32]], &device)?,
        })?;

        assert_eq!(
            output.tokens.to_vec2::<u32>()?,
            [vec![super::CUDA_DFLASH_SELECTOR_INVALID_TOKEN]]
        );
        assert!(output.candidate_probs.to_vec3::<f32>()?[0][0]
            .iter()
            .all(|probability| probability.is_nan()));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batched_categorical_matches_reference_across_chunks() -> candle_core::Result<()> {
        const VOCAB: usize = 2051;

        let device = Device::new_cuda(0)?;
        let mut backing = vec![90.0f32; 3 * VOCAB];
        let mut first = vec![-10.0f32; VOCAB];
        first[..3].copy_from_slice(&[0.0, 1.0, 2.0]);
        let mut second = vec![-20.0f32; VOCAB];
        second[2049] = 4.0;
        backing[VOCAB..2 * VOCAB].copy_from_slice(&first);
        backing[2 * VOCAB..].copy_from_slice(&second);
        let logits = Tensor::from_vec(backing, (3, VOCAB), &device)?.narrow(0, 1, 2)?;
        let inverse_temperatures = Tensor::new(&[99.0f32, 1.0, 0.5], &device)?.narrow(0, 1, 2)?;
        let uniforms = Tensor::new(&[0.99f32, 0.2, 0.5], &device)?.narrow(0, 1, 2)?;

        let output = super::cuda_categorical_logits_f32_packed_batched(
            &logits,
            &inverse_temperatures,
            &uniforms,
        )?;
        let actual = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let expected = [
            categorical_reference(&first, 1.0, 0.2),
            categorical_reference(&second, 0.5, 0.5),
        ];

        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual[0], expected[0]);
            assert_close(actual[1], expected[1], CUDA_LOGPROB_REL_TOLERANCE);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batched_categorical_marks_invalid_distribution() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(&[[1.0f32, f32::NAN, 3.0]], &device)?;
        let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
        let uniforms = Tensor::new(&[0.5f32], &device)?;
        let output = super::cuda_categorical_logits_f32_packed_batched(
            &logits,
            &inverse_temperatures,
            &uniforms,
        )?;
        let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        assert!(packed[0].iter().all(|value| value.is_nan()));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batched_categorical_selects_at_upper_boundary() -> candle_core::Result<()> {
        const VOCAB: usize = 2048;

        let device = Device::new_cuda(0)?;
        let logits = Tensor::zeros((1, VOCAB), DType::F32, &device)?;
        let inverse_temperatures = Tensor::new(&[1.0f32], &device)?;
        let upper = f32::from_bits(1.0f32.to_bits() - 1);
        let uniforms = Tensor::new(&[upper], &device)?;
        let output = super::cuda_categorical_logits_f32_packed_batched(
            &logits,
            &inverse_temperatures,
            &uniforms,
        )?;
        let packed = output.packed.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        assert_eq!(packed[0][0], 2047.0);
        assert_close(packed[0][1], -(2048.0f32).ln(), CUDA_LOGPROB_REL_TOLERANCE);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cached_top1_honors_view_offset() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(
            &[[90.0f32, 91.0, 92.0, 93.0], [-1.0, 4.0, 0.0, 2.0]],
            &device,
        )?
        .narrow(0, 1, 1)?;
        let mut workspace = None;
        let actual = super::cuda_top1_logits_f32_cached(&logits, &mut workspace)?;

        assert_eq!(actual, [4.0, 1.0]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_top1_uses_first_maximum_across_lanes_and_chunks() -> candle_core::Result<()> {
        const VOCAB: usize = 4097;

        let device = Device::new_cuda(0)?;
        let mut first = vec![-10.0f32; VOCAB];
        first[1] = 5.0;
        first[256] = 5.0;
        first[300] = 5.0;
        let mut second = vec![-10.0f32; VOCAB];
        second[2047] = 7.0;
        second[2048] = 7.0;
        second[3000] = 7.0;

        let mut workspace = None;
        let single = Tensor::from_vec(first.clone(), VOCAB, &device)?;
        let single = super::cuda_top1_logits_f32_cached(&single, &mut workspace)?;
        assert_eq!(single, [5.0, 1.0]);

        first.extend(second);
        let batched = Tensor::from_vec(first, (2, VOCAB), &device)?;
        let packed = super::cuda_top1_logits_f32_packed_batched(&batched)?
            .packed
            .to_vec2::<f32>()?;
        assert_eq!(packed, [[5.0, 1.0], [7.0, 2047.0]]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cached_top1_marks_nan_distribution() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let logits = Tensor::new(&[1.0f32, f32::NAN, 3.0], &device)?;
        let mut workspace = None;
        let actual = super::cuda_top1_logits_f32_cached(&logits, &mut workspace)?;

        assert!(actual.iter().all(|value| value.is_nan()));
        Ok(())
    }
}
