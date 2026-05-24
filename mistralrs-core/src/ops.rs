use candle_core::{shape::Dim, DType, Result, Tensor, D};

#[cfg(feature = "cuda")]
use crate::cuda::ffi;
use crate::layers::Activation;
#[cfg(feature = "cuda")]
use candle_core::Shape;

// ============================================================================
// Optimized parallel topk for CUDA
// Uses a dedicated kernel that's much faster than full sort for small k
// Single kernel call writes both values and indices - no post-processing needed
// ============================================================================

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
fn cuda_topk(input: &Tensor, k: usize) -> Result<TopKOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    let input = input.contiguous()?;
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
    let stream = dev.cuda_stream().cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::BF16(inp) => inp.device_ptr(inp.stream()),
        CudaStorageSlice::F16(inp) => inp.device_ptr(inp.stream()),
        CudaStorageSlice::F32(inp) => inp.device_ptr(inp.stream()),
        _ => candle_core::bail!("cuda_topk only supports BF16/F16/F32"),
    };
    let src_ptr = src_ptr as *const c_void;

    // Allocate both output buffers
    let indices_dst = unsafe { dev.alloc::<u32>(out_elem_count) }?;
    let (indices_ptr, indices_guard) = indices_dst.device_ptr(indices_dst.stream());

    let (values_tensor, indices_tensor) = match input.dtype() {
        DType::BF16 => {
            let values_dst = unsafe { dev.alloc::<half::bf16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr(values_dst.stream());

            unsafe {
                ffi::topk_bf16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream,
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
            let values_dst = unsafe { dev.alloc::<half::f16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr(values_dst.stream());

            unsafe {
                ffi::topk_f16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream,
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
            let values_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr(values_dst.stream());

            unsafe {
                ffi::topk_f32(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream,
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

#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_topk_logits_f32(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;

    const MAX_K: usize = 128;
    const CHUNK_SIZE: usize = 2048;
    const MAX_STAGE2_CANDIDATES: usize = 48 * 1024;

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
    if k == 0 || k > MAX_K {
        candle_core::bail!("cuda_topk_logits_f32 k={} must be in [1, {}]", k, MAX_K);
    }

    let nblocks = ncols.div_ceil(CHUNK_SIZE);
    let stage2_candidates = nblocks * k;
    if stage2_candidates > MAX_STAGE2_CANDIDATES {
        candle_core::bail!(
            "cuda_topk_logits_f32 workspace too large: {} candidates",
            stage2_candidates
        );
    }

    let (storage, _layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_logits_f32 requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream().cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(inp.stream()),
        _ => candle_core::bail!("cuda_topk_logits_f32 only supports F32"),
    };

    let workspace_elems = nblocks * k;
    let block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let block_maxes = unsafe { dev.alloc::<f32>(nblocks) }?;
    let block_sums = unsafe { dev.alloc::<f32>(nblocks) }?;
    let values_dst = unsafe { dev.alloc::<f32>(k) }?;
    let indices_dst = unsafe { dev.alloc::<u32>(k) }?;
    let softmax_info_dst = unsafe { dev.alloc::<f32>(2) }?;

    let (block_values_ptr, block_values_guard) = block_values.device_ptr(block_values.stream());
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr(block_indices.stream());
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr(block_maxes.stream());
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr(block_sums.stream());
    let (values_ptr, values_guard) = values_dst.device_ptr(values_dst.stream());
    let (indices_ptr, indices_guard) = indices_dst.device_ptr(indices_dst.stream());
    let (softmax_info_ptr, softmax_info_guard) =
        softmax_info_dst.device_ptr(softmax_info_dst.stream());

    unsafe {
        ffi::topk_large_f32(
            src_ptr as *const f32,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            values_ptr as *mut f32,
            indices_ptr as *mut u32,
            softmax_info_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream,
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;

    const MAX_K: usize = 128;
    const CHUNK_SIZE: usize = 2048;
    const MAX_STAGE2_CANDIDATES: usize = 48 * 1024;

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
    if k == 0 || k > MAX_K {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed k={} must be in [1, {}]",
            k,
            MAX_K
        );
    }

    let nblocks = ncols.div_ceil(CHUNK_SIZE);
    let stage2_candidates = nblocks * k;
    if stage2_candidates > MAX_STAGE2_CANDIDATES {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed workspace too large: {} candidates",
            stage2_candidates
        );
    }

    let (storage, _layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_logits_f32_packed requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream().cu_stream() as i64;

    let (src_ptr, src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(inp.stream()),
        _ => candle_core::bail!("cuda_topk_logits_f32_packed only supports F32"),
    };

    let workspace_elems = nblocks * k;
    let block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let block_maxes = unsafe { dev.alloc::<f32>(nblocks) }?;
    let block_sums = unsafe { dev.alloc::<f32>(nblocks) }?;
    let packed_dst = unsafe { dev.alloc::<f32>(2 * k + 2) }?;

    let (block_values_ptr, block_values_guard) = block_values.device_ptr(block_values.stream());
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr(block_indices.stream());
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr(block_maxes.stream());
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr(block_sums.stream());
    let (packed_ptr, packed_guard) = packed_dst.device_ptr(packed_dst.stream());

    unsafe {
        ffi::topk_large_f32_packed(
            src_ptr as *const f32,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream,
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

/// Fused topk + softmax for MoE routing
/// Returns softmax weights (not raw logits) and indices in a single kernel call
/// This eliminates intermediate tensor allocations and the separate softmax kernel
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_topk_softmax(input: &Tensor, k: usize) -> Result<TopKOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    // Validate k to prevent shared memory issues in the CUDA kernel
    const MAX_K: usize = 256;
    if k == 0 || k > MAX_K {
        candle_core::bail!("cuda_topk_softmax: k={} must be in range [1, {}]", k, MAX_K);
    }

    let input = input.contiguous()?;
    let dims = input.dims();
    let ncols = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".to_string()))?;
    let nrows = (input.elem_count() / ncols) as i32;
    let ncols_i32 = ncols as i32;
    let k_i32 = k as i32;

    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = k;
    let out_elem_count = nrows as usize * k;

    let (storage, _layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_softmax requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream().cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::BF16(inp) => inp.device_ptr(inp.stream()),
        CudaStorageSlice::F16(inp) => inp.device_ptr(inp.stream()),
        CudaStorageSlice::F32(inp) => inp.device_ptr(inp.stream()),
        _ => candle_core::bail!("cuda_topk_softmax only supports BF16/F16/F32"),
    };
    let src_ptr = src_ptr as *const c_void;

    let indices_dst = unsafe { dev.alloc::<u32>(out_elem_count) }?;
    let (indices_ptr, indices_guard) = indices_dst.device_ptr(indices_dst.stream());

    let (weights_tensor, indices_tensor) = match input.dtype() {
        DType::BF16 => {
            let weights_dst = unsafe { dev.alloc::<half::bf16>(out_elem_count) }?;
            let (weights_ptr, weights_guard) = weights_dst.device_ptr(weights_dst.stream());

            unsafe {
                ffi::topk_softmax_bf16(
                    src_ptr,
                    weights_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream,
                );
            }

            drop(weights_guard);
            drop(indices_guard);

            let weights_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::BF16(weights_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            (
                Tensor::from((
                    candle_core::Storage::Cuda(weights_storage),
                    Shape::from_dims(&out_dims),
                )),
                Tensor::from((
                    candle_core::Storage::Cuda(indices_storage),
                    Shape::from_dims(&out_dims),
                )),
            )
        }
        DType::F16 => {
            let weights_dst = unsafe { dev.alloc::<half::f16>(out_elem_count) }?;
            let (weights_ptr, weights_guard) = weights_dst.device_ptr(weights_dst.stream());

            unsafe {
                ffi::topk_softmax_f16(
                    src_ptr,
                    weights_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream,
                );
            }

            drop(weights_guard);
            drop(indices_guard);

            let weights_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F16(weights_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            (
                Tensor::from((
                    candle_core::Storage::Cuda(weights_storage),
                    Shape::from_dims(&out_dims),
                )),
                Tensor::from((
                    candle_core::Storage::Cuda(indices_storage),
                    Shape::from_dims(&out_dims),
                )),
            )
        }
        DType::F32 => {
            let weights_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
            let (weights_ptr, weights_guard) = weights_dst.device_ptr(weights_dst.stream());

            unsafe {
                ffi::topk_softmax_f32(
                    src_ptr,
                    weights_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream,
                );
            }

            drop(weights_guard);
            drop(indices_guard);

            let weights_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(weights_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            (
                Tensor::from((
                    candle_core::Storage::Cuda(weights_storage),
                    Shape::from_dims(&out_dims),
                )),
                Tensor::from((
                    candle_core::Storage::Cuda(indices_storage),
                    Shape::from_dims(&out_dims),
                )),
            )
        }
        dt => candle_core::bail!("cuda_topk_softmax unsupported dtype: {:?}", dt),
    };

    // Note: "values" here are actually softmax weights, not raw logits
    Ok(TopKOutput {
        values: weights_tensor,
        indices: indices_tensor,
    })
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
    /// Packed as `[values; indices_as_f32; softmax_denominator; global_max]`.
    pub packed: Tensor,
    pub k: usize,
    _workspace: Vec<Tensor>,
}

#[cfg(feature = "cuda")]
pub fn cuda_softcap_f32(input: &Tensor, cap: f32) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_softcap_f32 requires F32 input");
    }
    if !cap.is_finite() || cap <= 0.0 {
        candle_core::bail!("cuda_softcap_f32 requires a positive finite cap");
    }

    let input = input.contiguous()?;
    let elem_count = input.elem_count();
    if elem_count > i32::MAX as usize {
        candle_core::bail!("cuda_softcap_f32 input is too large: {elem_count} elements");
    }
    let elem_count_i32 = i32::try_from(elem_count).map_err(candle_core::Error::wrap)?;

    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_softcap_f32 requires CUDA tensor"),
    };
    let CudaStorageSlice::F32(src) = &storage.slice else {
        candle_core::bail!("cuda_softcap_f32 only supports F32");
    };
    let dev = storage.device();
    let out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, _src_guard) = src.device_ptr(src.stream());
    let (out_ptr, _out_guard) = out.device_ptr(out.stream());
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };
    unsafe {
        ffi::softcap_f32(
            src_ptr as *const c_void,
            out_ptr as *mut c_void,
            elem_count_i32,
            cap,
            dev.cuda_stream().cu_stream() as i64,
        );
    }
    drop(_src_guard);
    drop(_out_guard);

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
pub fn cuda_apply_sparse_penalties_f32(
    input: &Tensor,
    token_ids: &Tensor,
    counts: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, src_guard) = src.device_ptr(src.stream());
    let (token_ptr, token_guard) = token_src.device_ptr(token_src.stream());
    let (count_ptr, count_guard) = count_src.device_ptr(count_src.stream());
    let (out_ptr, out_guard) = out.device_ptr(out.stream());

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
            dev.cuda_stream().cu_stream() as i64,
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
pub fn cuda_rms_norm_residual(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    scale: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let stream = dev.cuda_stream().cu_stream() as i64;
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
                    let (scale_ptr, scale_guard) = scale_src.device_ptr(scale_src.stream());
                    (
                        unsafe { (scale_ptr as *const $ty).add(scale_layout.start_offset()) }
                            as *const c_void,
                        Some(scale_guard),
                    )
                } else {
                    (std::ptr::null(), None)
                };

            let out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(src.stream());
            let (residual_ptr, residual_guard) = residual_src.device_ptr(residual_src.stream());
            let (weight_ptr, weight_guard) = weight_src.device_ptr(weight_src.stream());
            let (out_ptr, out_guard) = out.device_ptr(out.stream());

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
                    stream,
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
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    // The fused kernel is intended for prompt/multi-token attention prep.
    // Decode rows are already cheap in the existing kernels, and this row-wise
    // reduction launch is slower for seq_len == 1 on current CUDA targets.
    if seq_len == 1 {
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
    let stream = dev.cuda_stream().cu_stream() as i64;
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

            let q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let k_out_buf = if k_elem_count == 0 {
                None
            } else {
                Some(unsafe { dev.alloc::<$ty>(k_elem_count) }?)
            };

            let (q_ptr, q_guard) = q_src.device_ptr(q_src.stream());
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(q_weight_src.stream());
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(cos_src.stream());
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(sin_src.stream());
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
                let (ptr, guard) = k_src.device_ptr(k_src.stream());
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
                    let (ptr, guard) = k_weight_src.device_ptr(k_weight_src.stream());
                    k_weight_guard = Some(guard);
                    unsafe { (ptr as *const $ty).add(k_weight_layout.start_offset()) }
                } else {
                    q_weight_ptr
                };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr(q_out_buf.stream());
            let mut k_out_guard = None;
            let k_out_ptr = if let Some(k_out_buf) = &k_out_buf {
                let (ptr, guard) = k_out_buf.device_ptr(k_out_buf.stream());
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
                    stream,
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
/// With supported dtypes (F16, BF16, F32) and activations (SiLU, GELU, ReLU),
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

    let first = xs.narrow(D::Minus1, 0, split_size)?;
    let second = xs.narrow(D::Minus1, split_size, split_size)?;
    match order {
        GatedActivationOrder::GateUp => mul_and_act(&first, &second, act),
        GatedActivationOrder::UpGate => mul_and_act(&second, &first, act),
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

    Ok((
        q_proj.forward(xs)?,
        k_proj.forward(xs)?,
        v_proj.forward(xs)?,
    ))
}

mod tests {
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
}
