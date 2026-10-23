use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::cuda_backend::cudarc::driver::{
    sys, CudaEvent, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, PinnedHostSlice,
    ValidAsZeroBits,
};
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::{
    CpuStorage, CudaStorage, DType, DeviceLocation, InplaceOp1, Layout, Result, Shape, Storage,
    Tensor,
};

use crate::ops::{
    cuda_topk_ranked_packed_batched, cuda_topk_ranked_packed_batched_with_workspace,
    CudaRankedTopKPackedWorkspace, RankedTopKPackedOutput, CUDA_TOPK_MAX_K,
};

use super::ffi;

pub(crate) const SPARSE_REJECTION_STATUS_OK: u32 = 0;
pub(crate) const SPARSE_REJECTION_STATUS_NEEDS_CPU: u32 = 1;
pub(crate) const SPARSE_REJECTION_STATUS_INVALID_Q: u32 = 2;
pub(crate) const SPARSE_REJECTION_STATUS_INVALID_TARGET: u32 = 3;
pub(crate) const SPARSE_REJECTION_STATUS_INVALID_RNG: u32 = 4;
pub(crate) const SPARSE_REJECTION_INVALID_VALUE: u32 = u32::MAX;
pub(crate) const SPARSE_REJECTION_OUTCOME_WIDTH: usize = 3;
pub(crate) const SPARSE_REJECTION_MAX_Q_WIDTH: usize = 128;

const OP: &str = "sparse_rejection_cuda";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SparseRejectionMode {
    Categorical,
    BoundedTopK { max_top_k: usize },
}

#[cfg(test)]
pub(crate) struct SparseRejectionInput<'a> {
    pub(crate) target_logits: &'a Tensor,
    pub(crate) draft_tokens: &'a Tensor,
    pub(crate) q_token_ids: &'a Tensor,
    pub(crate) q_probs: &'a Tensor,
    pub(crate) inverse_temperatures: &'a Tensor,
    pub(crate) target_top_k: &'a Tensor,
    pub(crate) top_p: &'a Tensor,
    pub(crate) min_p: &'a Tensor,
    pub(crate) accept_uniforms: &'a Tensor,
    pub(crate) sample_uniforms: &'a Tensor,
    pub(crate) mode: SparseRejectionMode,
}

pub(crate) struct SparseRejectionWorkspaceInput<'a> {
    pub(crate) target_logits: &'a Tensor,
    pub(crate) proposal: SparseRejectionProposalInput<'a>,
    pub(crate) draft_tokens: SparseRejectionDraftInput<'a>,
    pub(crate) inverse_temperatures: &'a [f32],
    pub(crate) target_top_k: &'a [u32],
    pub(crate) top_p: &'a [f32],
    pub(crate) min_p: &'a [f32],
    pub(crate) accept_uniforms: &'a [f32],
    pub(crate) sample_uniforms: &'a [f32],
    pub(crate) mode: SparseRejectionMode,
}

#[derive(Clone, Copy)]
pub(crate) enum SparseRejectionDraftInput<'a> {
    Host(&'a [u32]),
    #[cfg(test)]
    Device(&'a Tensor),
    DeviceRows(&'a [Tensor]),
}

#[derive(Clone, Copy)]
pub(crate) enum SparseRejectionProposalInput<'a> {
    Deterministic,
    #[cfg(test)]
    Sparse {
        token_ids: &'a Tensor,
        probs: &'a Tensor,
    },
    SparseRows {
        token_ids: &'a [Tensor],
        probs: &'a [Tensor],
    },
}

#[derive(Clone)]
struct DenseCudaRows {
    rows: Vec<Tensor>,
    shape: Shape,
}

#[derive(Clone)]
enum SparseRejectionDeviceTensor {
    Tensor(Tensor),
    DenseRows(DenseCudaRows),
}

impl SparseRejectionDeviceTensor {
    fn from_tensor(tensor: &Tensor) -> Self {
        Self::Tensor(tensor.clone())
    }

    fn from_rows(
        rows: &[Tensor],
        dtype: DType,
        row_shape: &[usize],
        device: &candle_core::Device,
    ) -> Result<Self> {
        if let Some(rows) = dense_cuda_rows(rows, dtype, row_shape, device)? {
            return Ok(Self::DenseRows(rows));
        }
        let rows = rows.iter().collect::<Vec<_>>();
        let tensor = match rows.as_slice() {
            [row] => row.unsqueeze(0)?.contiguous()?,
            [] => candle_core::bail!("{OP} cannot materialize an empty row batch"),
            _ => Tensor::stack(&rows, 0)?.contiguous()?,
        };
        Ok(Self::Tensor(tensor))
    }

    fn anchor(&self) -> &Tensor {
        match self {
            Self::Tensor(tensor) => tensor,
            Self::DenseRows(rows) => rows.rows.first().expect("dense CUDA rows are non-empty"),
        }
    }

    fn dims(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.dims(),
            Self::DenseRows(rows) => rows.shape.dims(),
        }
    }

    fn dtype(&self) -> DType {
        self.anchor().dtype()
    }

    fn device(&self) -> &candle_core::Device {
        self.anchor().device()
    }

    fn elem_count(&self) -> usize {
        match self {
            Self::Tensor(tensor) => tensor.elem_count(),
            Self::DenseRows(rows) => rows.shape.elem_count(),
        }
    }

    fn is_contiguous(&self) -> bool {
        match self {
            Self::Tensor(tensor) => tensor.is_contiguous(),
            Self::DenseRows(_) => true,
        }
    }

    fn reshape(&self, shape: Shape) -> Result<Self> {
        if shape.elem_count() != self.elem_count() {
            candle_core::bail!(
                "{OP} cannot reshape {:?} to {:?}",
                self.dims(),
                shape.dims()
            );
        }
        match self {
            Self::Tensor(tensor) => Ok(Self::Tensor(tensor.reshape(shape)?)),
            Self::DenseRows(rows) => Ok(Self::DenseRows(DenseCudaRows {
                rows: rows.rows.clone(),
                shape,
            })),
        }
    }

    fn extend_inputs(&self, inputs: &mut Vec<Tensor>) {
        match self {
            Self::Tensor(tensor) => inputs.push(tensor.clone()),
            Self::DenseRows(rows) => inputs.extend(rows.rows.iter().cloned()),
        }
    }
}

fn dense_cuda_rows(
    rows: &[Tensor],
    dtype: DType,
    row_shape: &[usize],
    device: &candle_core::Device,
) -> Result<Option<DenseCudaRows>> {
    if rows.is_empty() || !device.is_cuda() {
        return Ok(None);
    }
    let row_elems = row_shape.iter().try_fold(1usize, |elements, dim| {
        elements
            .checked_mul(*dim)
            .ok_or_else(|| candle_core::Error::msg("sparse rejection row size overflow"))
    })?;
    let mut storage_id = None;
    let mut first_offset = None;
    for (index, row) in rows.iter().enumerate() {
        if row.dtype() != dtype
            || row.dims() != row_shape
            || !row.is_contiguous()
            || !row.device().same_device(device)
        {
            return Ok(None);
        }
        let (storage, layout) = row.storage_and_layout();
        if !matches!(&*storage, Storage::Cuda(_)) {
            return Ok(None);
        }
        let row_storage_id = std::ptr::from_ref::<Storage>(&*storage) as usize;
        let row_offset = layout.start_offset();
        let first_storage_id = *storage_id.get_or_insert(row_storage_id);
        let first_row_offset = *first_offset.get_or_insert(row_offset);
        let expected_offset = index
            .checked_mul(row_elems)
            .and_then(|offset| first_row_offset.checked_add(offset))
            .ok_or_else(|| candle_core::Error::msg("sparse rejection row offset overflow"))?;
        if row_storage_id != first_storage_id || row_offset != expected_offset {
            return Ok(None);
        }
    }
    let mut shape = Vec::with_capacity(row_shape.len() + 1);
    shape.push(rows.len());
    shape.extend_from_slice(row_shape);
    Ok(Some(DenseCudaRows {
        rows: rows.to_vec(),
        shape: Shape::from_dims(&shape),
    }))
}

struct SparseRejectionDeviceInput<'a> {
    target_logits: &'a Tensor,
    draft_tokens: &'a SparseRejectionDeviceTensor,
    q_token_ids: &'a SparseRejectionDeviceTensor,
    q_probs: &'a SparseRejectionDeviceTensor,
    inverse_temperatures: &'a Tensor,
    target_top_k: &'a Tensor,
    top_p: &'a Tensor,
    min_p: &'a Tensor,
    accept_uniforms: &'a Tensor,
    sample_uniforms: &'a Tensor,
    mode: SparseRejectionMode,
}

struct SparseRejectionPinned<T> {
    allocation: PinnedHostSlice<T>,
    ptr: NonNull<T>,
}

unsafe impl<T: Send> Send for SparseRejectionPinned<T> {}

impl<T: DeviceRepr + ValidAsZeroBits> SparseRejectionPinned<T> {
    fn new(stream: &Arc<CudaStream>, len: usize) -> Result<Self> {
        let mut allocation =
            unsafe { stream.context().alloc_pinned::<T>(len) }.map_err(candle_core::Error::wrap)?;
        let ptr = NonNull::new(allocation.as_mut_ptr().map_err(candle_core::Error::wrap)?)
            .ok_or_else(|| candle_core::Error::msg("CUDA returned a null pinned pointer"))?;
        Ok(Self { allocation, ptr })
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.allocation.len()) }
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.allocation.len()) }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

struct CudaSparseRejectionPending {
    generation: u64,
    batch: usize,
    drafts: usize,
    _output: SparseRejectionOutput,
}

pub struct CudaSparseRejectionWorkspace {
    location: DeviceLocation,
    stream: Arc<CudaStream>,
    capacity_batch: usize,
    capacity_drafts: usize,
    id: u64,
    next_generation: u64,
    draft_tokens: Tensor,
    inverse_temperatures: Tensor,
    row_inverse_temperatures: Tensor,
    target_top_k: Tensor,
    top_p: Tensor,
    min_p: Tensor,
    accept_uniforms: Tensor,
    sample_uniforms: Tensor,
    deterministic_q_probs: Tensor,
    outcomes: Tensor,
    draft_tokens_host: SparseRejectionPinned<u32>,
    inverse_temperatures_host: SparseRejectionPinned<f32>,
    row_inverse_temperatures_host: SparseRejectionPinned<f32>,
    target_top_k_host: SparseRejectionPinned<u32>,
    top_p_host: SparseRejectionPinned<f32>,
    min_p_host: SparseRejectionPinned<f32>,
    accept_uniforms_host: SparseRejectionPinned<f32>,
    sample_uniforms_host: SparseRejectionPinned<f32>,
    outcomes_host: SparseRejectionPinned<u32>,
    topk: Option<CudaRankedTopKPackedWorkspace>,
    completion: Arc<CudaEvent>,
    pending: Option<CudaSparseRejectionPending>,
}

impl Drop for CudaSparseRejectionWorkspace {
    fn drop(&mut self) {
        if self.pending.is_some() {
            let _ = self.completion.synchronize();
        }
    }
}

pub(crate) struct CudaSparseRejectionSubmission {
    workspace_id: u64,
    generation: u64,
    batch: usize,
    drafts: usize,
    completion: Arc<CudaEvent>,
}

impl CudaSparseRejectionSubmission {
    fn wait(&self) -> Result<()> {
        self.completion
            .synchronize()
            .map_err(candle_core::Error::wrap)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseRejectionRow {
    pub(crate) accepted_count: u32,
    pub(crate) continuation: u32,
    pub(crate) status: u32,
}

#[derive(Debug)]
pub(crate) struct SparseRejectionCompletion {
    pub(crate) rows: Vec<SparseRejectionRow>,
    pub(crate) draft_tokens: Vec<Vec<u32>>,
}

pub(crate) struct SparseRejectionOutput {
    _outcomes: Tensor,
    _inputs: Vec<Tensor>,
    _packed_target: Option<RankedTopKPackedOutput>,
    _row_inverse_temperatures: Option<Tensor>,
}

impl SparseRejectionOutput {
    #[cfg(test)]
    pub(crate) fn to_rows(&self) -> Result<Vec<SparseRejectionRow>> {
        self._outcomes
            .to_vec2::<u32>()?
            .into_iter()
            .map(|row| {
                let [accepted_count, continuation, status] = row.as_slice() else {
                    candle_core::bail!("{OP} produced a malformed outcome row")
                };
                Ok(SparseRejectionRow {
                    accepted_count: *accepted_count,
                    continuation: *continuation,
                    status: *status,
                })
            })
            .collect()
    }
}

struct SparseRejectionShape {
    batch: usize,
    rows: usize,
    vocab: usize,
    batch_i32: i32,
    drafts_i32: i32,
    vocab_i32: i32,
    q_width_i32: i32,
}

struct SparseRejectionKernelLaunch {
    mode: SparseRejectionMode,
    target_ptr: usize,
    draft_ptr: usize,
    q_ids_ptr: usize,
    q_probs_ptr: usize,
    temperature_ptr: usize,
    top_k_ptr: usize,
    top_p_ptr: usize,
    min_p_ptr: usize,
    accept_ptr: usize,
    sample_ptr: usize,
    batch: i32,
    drafts: i32,
    vocab: i32,
    q_width: i32,
    packed_k: Option<i32>,
}

impl InplaceOp1 for SparseRejectionKernelLaunch {
    fn name(&self) -> &'static str {
        OP
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        candle_core::bail!("{OP} requires CUDA storage")
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let stream = storage.device().cuda_stream();
        let outcomes = storage.as_cuda_slice_mut::<u32>()?;
        let (outcomes_ptr, outcomes_guard) = outcomes.device_ptr_mut(&stream);
        let outcomes_ptr = unsafe { (outcomes_ptr as *mut u32).add(layout.start_offset()) };
        unsafe {
            match self.mode {
                SparseRejectionMode::Categorical => ffi::sparse_rejection_categorical_f32(
                    self.target_ptr as *const f32,
                    self.draft_ptr as *const u32,
                    self.q_ids_ptr as *const u32,
                    self.q_probs_ptr as *const f32,
                    self.temperature_ptr as *const f32,
                    self.top_k_ptr as *const u32,
                    self.top_p_ptr as *const f32,
                    self.min_p_ptr as *const f32,
                    self.accept_ptr as *const f32,
                    self.sample_ptr as *const f32,
                    outcomes_ptr,
                    self.batch,
                    self.drafts,
                    self.vocab,
                    self.q_width,
                    stream.cu_stream() as i64,
                ),
                SparseRejectionMode::BoundedTopK { .. } => ffi::sparse_rejection_topk_f32(
                    self.target_ptr as *const f32,
                    self.draft_ptr as *const u32,
                    self.q_ids_ptr as *const u32,
                    self.q_probs_ptr as *const f32,
                    self.temperature_ptr as *const f32,
                    self.top_k_ptr as *const u32,
                    self.top_p_ptr as *const f32,
                    self.min_p_ptr as *const f32,
                    self.accept_ptr as *const f32,
                    self.sample_ptr as *const f32,
                    outcomes_ptr,
                    self.batch,
                    self.drafts,
                    self.vocab,
                    self.q_width,
                    self.packed_k
                        .expect("bounded top-k launch has a packed width"),
                    stream.cu_stream() as i64,
                ),
            }
        }
        drop(outcomes_guard);
        Ok(())
    }
}

fn validate_input(input: &SparseRejectionDeviceInput<'_>) -> Result<SparseRejectionShape> {
    let target_dims = input.target_logits.dims();
    if target_dims.len() != 3 {
        candle_core::bail!(
            "{OP} expected target logits with shape [batch, drafts + 1, vocab], got {target_dims:?}"
        );
    }
    let batch = target_dims[0];
    let rows = target_dims[1];
    let vocab = target_dims[2];
    if batch == 0 || rows < 2 || vocab == 0 {
        candle_core::bail!(
            "{OP} requires a non-empty batch, at least one draft, and a non-empty vocabulary"
        );
    }
    let drafts = rows - 1;
    let q_dims = input.q_token_ids.dims();
    if q_dims.len() != 3 || q_dims[0] != batch || q_dims[1] != drafts {
        candle_core::bail!(
            "{OP} expected q token ids with shape [{batch}, {drafts}, q_width], got {q_dims:?}"
        );
    }
    let q_width = q_dims[2];
    if q_width == 0 || q_width > SPARSE_REJECTION_MAX_Q_WIDTH {
        candle_core::bail!("{OP} q_width={q_width} must be in [1, {SPARSE_REJECTION_MAX_Q_WIDTH}]");
    }

    let draft_shape = [batch, drafts];
    let q_shape = [batch, drafts, q_width];
    let batch_shape = [batch];
    match input.mode {
        SparseRejectionMode::Categorical if input.target_logits.dtype() != DType::F32 => {
            candle_core::bail!("{OP} categorical mode requires F32 target logits");
        }
        SparseRejectionMode::BoundedTopK { .. }
            if !matches!(
                input.target_logits.dtype(),
                DType::BF16 | DType::F16 | DType::F32
            ) =>
        {
            candle_core::bail!("{OP} bounded top-k mode requires BF16, F16, or F32 target logits");
        }
        _ => {}
    }
    if !input.target_logits.is_contiguous() {
        candle_core::bail!("{OP} requires contiguous target logits");
    }
    let device_specs = [
        (
            input.draft_tokens,
            DType::U32,
            draft_shape.as_slice(),
            "draft tokens",
        ),
        (
            input.q_token_ids,
            DType::U32,
            q_shape.as_slice(),
            "q token ids",
        ),
        (
            input.q_probs,
            DType::F32,
            q_shape.as_slice(),
            "q probabilities",
        ),
    ];
    for (tensor, dtype, shape, name) in device_specs {
        if tensor.dtype() != dtype {
            candle_core::bail!(
                "{OP} expected {name} to have dtype {dtype:?}, got {:?}",
                tensor.dtype()
            );
        }
        if tensor.dims() != shape {
            candle_core::bail!(
                "{OP} expected {name} with shape {shape:?}, got {:?}",
                tensor.dims()
            );
        }
        if !tensor.is_contiguous() {
            candle_core::bail!("{OP} requires contiguous {name}");
        }
        if !input.target_logits.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} requires every tensor on one CUDA device");
        }
    }
    let tensor_specs = [
        (
            input.inverse_temperatures,
            DType::F32,
            batch_shape.as_slice(),
            "inverse temperatures",
        ),
        (
            input.target_top_k,
            DType::U32,
            batch_shape.as_slice(),
            "target top-k",
        ),
        (input.top_p, DType::F32, batch_shape.as_slice(), "top-p"),
        (input.min_p, DType::F32, batch_shape.as_slice(), "min-p"),
        (
            input.accept_uniforms,
            DType::F32,
            draft_shape.as_slice(),
            "accept uniforms",
        ),
        (
            input.sample_uniforms,
            DType::F32,
            batch_shape.as_slice(),
            "sample uniforms",
        ),
    ];
    for (tensor, dtype, shape, name) in tensor_specs {
        if tensor.dtype() != dtype {
            candle_core::bail!(
                "{OP} expected {name} to have dtype {dtype:?}, got {:?}",
                tensor.dtype()
            );
        }
        if tensor.dims() != shape {
            candle_core::bail!(
                "{OP} expected {name} with shape {shape:?}, got {:?}",
                tensor.dims()
            );
        }
        if !tensor.is_contiguous() {
            candle_core::bail!("{OP} requires contiguous {name}");
        }
        if !input.target_logits.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} requires every tensor on one CUDA device");
        }
    }
    if !input.target_logits.device().is_cuda() {
        candle_core::bail!("{OP} requires CUDA tensors");
    }

    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let drafts_i32 = i32::try_from(drafts).map_err(candle_core::Error::wrap)?;
    let vocab_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let q_width_i32 = i32::try_from(q_width).map_err(candle_core::Error::wrap)?;
    batch
        .checked_mul(SPARSE_REJECTION_OUTCOME_WIDTH)
        .ok_or_else(|| candle_core::Error::msg("sparse rejection outcome size overflow"))?;

    Ok(SparseRejectionShape {
        batch,
        rows,
        vocab,
        batch_i32,
        drafts_i32,
        vocab_i32,
        q_width_i32,
    })
}

fn sparse_rejection_workspace_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn workspace_capacity(required: usize, name: &str) -> Result<usize> {
    required.checked_next_power_of_two().ok_or_else(|| {
        candle_core::Error::msg(format!("sparse rejection {name} capacity overflow"))
    })
}

impl CudaSparseRejectionWorkspace {
    fn new(device: &candle_core::Device, batch: usize, drafts: usize) -> Result<Self> {
        let cuda_device = device.as_cuda_device()?;
        let stream = cuda_device.cuda_stream();
        let capacity_batch = workspace_capacity(batch, "batch")?;
        let capacity_drafts = workspace_capacity(drafts, "draft")?;
        let draft_elems = capacity_batch
            .checked_mul(capacity_drafts)
            .ok_or_else(|| candle_core::Error::msg("sparse rejection draft workspace overflow"))?;
        let outcome_elems = capacity_batch
            .checked_mul(SPARSE_REJECTION_OUTCOME_WIDTH)
            .ok_or_else(|| {
                candle_core::Error::msg("sparse rejection outcome workspace overflow")
            })?;
        let row_elems = capacity_drafts
            .checked_add(1)
            .and_then(|rows| capacity_batch.checked_mul(rows))
            .ok_or_else(|| {
                candle_core::Error::msg("sparse rejection row temperature workspace overflow")
            })?;
        let event_flags = Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC);
        Ok(Self {
            location: cuda_device.location(),
            stream: stream.clone(),
            capacity_batch,
            capacity_drafts,
            id: sparse_rejection_workspace_id(),
            next_generation: 1,
            draft_tokens: Tensor::zeros(draft_elems, DType::U32, device)?,
            inverse_temperatures: Tensor::zeros(capacity_batch, DType::F32, device)?,
            row_inverse_temperatures: Tensor::zeros(row_elems, DType::F32, device)?,
            target_top_k: Tensor::zeros(capacity_batch, DType::U32, device)?,
            top_p: Tensor::zeros(capacity_batch, DType::F32, device)?,
            min_p: Tensor::zeros(capacity_batch, DType::F32, device)?,
            accept_uniforms: Tensor::zeros(draft_elems, DType::F32, device)?,
            sample_uniforms: Tensor::zeros(capacity_batch, DType::F32, device)?,
            deterministic_q_probs: Tensor::ones(draft_elems, DType::F32, device)?,
            outcomes: Tensor::zeros(outcome_elems, DType::U32, device)?,
            draft_tokens_host: SparseRejectionPinned::new(&stream, draft_elems)?,
            inverse_temperatures_host: SparseRejectionPinned::new(&stream, capacity_batch)?,
            row_inverse_temperatures_host: SparseRejectionPinned::new(&stream, row_elems)?,
            target_top_k_host: SparseRejectionPinned::new(&stream, capacity_batch)?,
            top_p_host: SparseRejectionPinned::new(&stream, capacity_batch)?,
            min_p_host: SparseRejectionPinned::new(&stream, capacity_batch)?,
            accept_uniforms_host: SparseRejectionPinned::new(&stream, draft_elems)?,
            sample_uniforms_host: SparseRejectionPinned::new(&stream, capacity_batch)?,
            outcomes_host: SparseRejectionPinned::new(&stream, outcome_elems)?,
            topk: None,
            completion: Arc::new(
                stream
                    .context()
                    .new_event(event_flags)
                    .map_err(candle_core::Error::wrap)?,
            ),
            pending: None,
        })
    }

    fn can_hold(
        &self,
        location: DeviceLocation,
        stream: &Arc<CudaStream>,
        batch: usize,
        drafts: usize,
    ) -> bool {
        self.location == location
            && Arc::ptr_eq(self.stream.context(), stream.context())
            && self.stream.cu_stream() == stream.cu_stream()
            && self.capacity_batch >= batch
            && self.capacity_drafts >= drafts
    }
}

struct PinnedU32HtoD<'a> {
    host: &'a SparseRejectionPinned<u32>,
    len: usize,
    stream: &'a Arc<CudaStream>,
}

impl InplaceOp1 for PinnedU32HtoD<'_> {
    fn name(&self) -> &'static str {
        "sparse-rejection-u32-htod"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        candle_core::bail!("{} requires CUDA storage", self.name())
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let dst = storage.as_cuda_slice_mut::<u32>()?;
        let start = layout.start_offset();
        let mut dst = dst.slice_mut(start..start + self.len);
        let (dst_ptr, dst_guard) = dst.device_ptr_mut(self.stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                self.host.as_ptr().cast(),
                self.len * std::mem::size_of::<u32>(),
                self.stream.cu_stream(),
            )
        };
        drop(dst_guard);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("sparse rejection U32 H2D copy failed"));
        }
        Ok(())
    }
}

struct PinnedF32HtoD<'a> {
    host: &'a SparseRejectionPinned<f32>,
    len: usize,
    stream: &'a Arc<CudaStream>,
}

impl InplaceOp1 for PinnedF32HtoD<'_> {
    fn name(&self) -> &'static str {
        "sparse-rejection-f32-htod"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        candle_core::bail!("{} requires CUDA storage", self.name())
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let dst = storage.as_cuda_slice_mut::<f32>()?;
        let start = layout.start_offset();
        let mut dst = dst.slice_mut(start..start + self.len);
        let (dst_ptr, dst_guard) = dst.device_ptr_mut(self.stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                self.host.as_ptr().cast(),
                self.len * std::mem::size_of::<f32>(),
                self.stream.cu_stream(),
            )
        };
        drop(dst_guard);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("sparse rejection F32 H2D copy failed"));
        }
        Ok(())
    }
}

fn enqueue_u32_htod(
    host: &SparseRejectionPinned<u32>,
    dst: &Tensor,
    len: usize,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    if !dst.device().is_cuda() {
        candle_core::bail!("{OP} workspace destination must be CUDA");
    }
    if dst.dtype() != DType::U32 {
        candle_core::bail!("{OP} workspace destination must be U32");
    }
    if !dst.is_contiguous() || len > dst.elem_count() || len > host.allocation.len() {
        candle_core::bail!("{OP} workspace U32 copy exceeds its capacity");
    }
    dst.inplace_op1(&PinnedU32HtoD { host, len, stream })
}

fn enqueue_f32_htod(
    host: &SparseRejectionPinned<f32>,
    dst: &Tensor,
    len: usize,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    if !dst.device().is_cuda() {
        candle_core::bail!("{OP} workspace destination must be CUDA");
    }
    if dst.dtype() != DType::F32 {
        candle_core::bail!("{OP} workspace destination must be F32");
    }
    if !dst.is_contiguous() || len > dst.elem_count() || len > host.allocation.len() {
        candle_core::bail!("{OP} workspace F32 copy exceeds its capacity");
    }
    dst.inplace_op1(&PinnedF32HtoD { host, len, stream })
}

fn enqueue_u32_dtoh(
    src: &Tensor,
    host: &mut SparseRejectionPinned<u32>,
    len: usize,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let (storage, layout) = src.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("{OP} workspace source must be CUDA");
    };
    let CudaStorageSlice::U32(slice) = &storage.slice else {
        candle_core::bail!("{OP} workspace source must be U32");
    };
    if !layout.is_contiguous() || len > src.elem_count() || len > host.allocation.len() {
        candle_core::bail!("{OP} workspace D2H copy exceeds its capacity");
    }
    let start = layout.start_offset();
    let slice = slice.slice(start..start + len);
    let (src_ptr, src_guard) = slice.device_ptr(stream);
    let result = unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            host.as_mut_ptr().cast(),
            src_ptr,
            len * std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
    };
    drop(src_guard);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("sparse rejection U32 D2H copy failed"));
    }
    Ok(())
}

fn enqueue_device_u32_dtoh(
    src: &SparseRejectionDeviceTensor,
    host: &mut SparseRejectionPinned<u32>,
    len: usize,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let anchor = src.anchor();
    let (storage, layout) = anchor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("{OP} workspace source must be CUDA");
    };
    let CudaStorageSlice::U32(slice) = &storage.slice else {
        candle_core::bail!("{OP} workspace source must be U32");
    };
    if !src.is_contiguous() || len > src.elem_count() || len > host.allocation.len() {
        candle_core::bail!("{OP} workspace D2H copy exceeds its capacity");
    }
    let start = layout.start_offset();
    let slice = slice.slice(start..start + len);
    let (src_ptr, src_guard) = slice.device_ptr(stream);
    let result = unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            host.as_mut_ptr().cast(),
            src_ptr,
            len * std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
    };
    drop(src_guard);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("sparse rejection U32 D2H copy failed"));
    }
    Ok(())
}

fn validate_host_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        candle_core::bail!("{OP} expected {expected} {name} values, got {actual}");
    }
    Ok(())
}

pub(crate) fn sparse_rejection_cuda_submit(
    input: SparseRejectionWorkspaceInput<'_>,
    cache: &mut Option<CudaSparseRejectionWorkspace>,
) -> Result<CudaSparseRejectionSubmission> {
    let [batch, rows, _] = input.target_logits.dims() else {
        candle_core::bail!(
            "{OP} expected target logits with shape [batch, drafts + 1, vocab], got {:?}",
            input.target_logits.dims()
        );
    };
    if *batch == 0 || *rows < 2 {
        candle_core::bail!("{OP} requires a non-empty batch and at least one draft");
    }
    let batch = *batch;
    let rows = *rows;
    let drafts = rows - 1;
    let draft_elems = batch
        .checked_mul(drafts)
        .ok_or_else(|| candle_core::Error::msg("sparse rejection draft input overflow"))?;
    match input.draft_tokens {
        SparseRejectionDraftInput::Host(tokens) => {
            validate_host_len("draft token", tokens.len(), draft_elems)?;
        }
        #[cfg(test)]
        SparseRejectionDraftInput::Device(tokens) => {
            if tokens.dtype() != DType::U32
                || tokens.dims() != [batch, drafts]
                || !tokens.is_contiguous()
                || !tokens.device().same_device(input.target_logits.device())
            {
                candle_core::bail!(
                    "{OP} device draft tokens must be contiguous CUDA U32 with shape [{batch}, {drafts}] on the target device"
                );
            }
        }
        SparseRejectionDraftInput::DeviceRows(tokens) => {
            if tokens.len() != batch {
                candle_core::bail!(
                    "{OP} expected {batch} device draft token rows, got {}",
                    tokens.len()
                );
            }
        }
    }
    if let SparseRejectionProposalInput::SparseRows { token_ids, probs } = input.proposal {
        if token_ids.len() != batch || probs.len() != batch {
            candle_core::bail!(
                "{OP} expected {batch} sparse proposal rows, got {} token-id and {} probability rows",
                token_ids.len(),
                probs.len()
            );
        }
    }
    validate_host_len(
        "inverse temperature",
        input.inverse_temperatures.len(),
        batch,
    )?;
    validate_host_len("target top-k", input.target_top_k.len(), batch)?;
    validate_host_len("top-p", input.top_p.len(), batch)?;
    validate_host_len("min-p", input.min_p.len(), batch)?;
    validate_host_len("accept uniform", input.accept_uniforms.len(), draft_elems)?;
    validate_host_len("sample uniform", input.sample_uniforms.len(), batch)?;

    let cuda_device = input.target_logits.device().as_cuda_device()?;
    let location = cuda_device.location();
    let stream = cuda_device.cuda_stream();
    let needs_alloc = cache
        .as_ref()
        .is_none_or(|workspace| !workspace.can_hold(location, &stream, batch, drafts));
    if needs_alloc {
        if cache
            .as_ref()
            .is_some_and(|workspace| workspace.pending.is_some())
        {
            candle_core::bail!("{OP} cannot resize while a submission is pending");
        }
        *cache = Some(CudaSparseRejectionWorkspace::new(
            input.target_logits.device(),
            batch,
            drafts,
        )?);
    }
    let workspace = cache
        .as_mut()
        .expect("sparse rejection workspace was allocated above");
    if workspace.pending.is_some() {
        candle_core::bail!("{OP} workspace already has a pending submission");
    }
    let generation = workspace.next_generation;

    let result = (|| {
        if let SparseRejectionDraftInput::Host(tokens) = input.draft_tokens {
            workspace.draft_tokens_host.as_mut_slice()[..draft_elems].copy_from_slice(tokens);
        }
        workspace.inverse_temperatures_host.as_mut_slice()[..batch]
            .copy_from_slice(input.inverse_temperatures);
        workspace.target_top_k_host.as_mut_slice()[..batch].copy_from_slice(input.target_top_k);
        workspace.top_p_host.as_mut_slice()[..batch].copy_from_slice(input.top_p);
        workspace.min_p_host.as_mut_slice()[..batch].copy_from_slice(input.min_p);
        workspace.accept_uniforms_host.as_mut_slice()[..draft_elems]
            .copy_from_slice(input.accept_uniforms);
        workspace.sample_uniforms_host.as_mut_slice()[..batch]
            .copy_from_slice(input.sample_uniforms);
        let row_elems = if matches!(input.mode, SparseRejectionMode::BoundedTopK { .. }) {
            let row_elems = batch.checked_mul(rows).ok_or_else(|| {
                candle_core::Error::msg("sparse rejection row temperature input overflow")
            })?;
            for (row_temperatures, &inverse_temperature) in
                workspace.row_inverse_temperatures_host.as_mut_slice()[..row_elems]
                    .chunks_exact_mut(rows)
                    .zip(input.inverse_temperatures)
            {
                row_temperatures.fill(inverse_temperature);
            }
            Some(row_elems)
        } else {
            None
        };

        if matches!(input.draft_tokens, SparseRejectionDraftInput::Host(_)) {
            enqueue_u32_htod(
                &workspace.draft_tokens_host,
                &workspace.draft_tokens,
                draft_elems,
                &stream,
            )?;
        }
        enqueue_f32_htod(
            &workspace.inverse_temperatures_host,
            &workspace.inverse_temperatures,
            batch,
            &stream,
        )?;
        enqueue_u32_htod(
            &workspace.target_top_k_host,
            &workspace.target_top_k,
            batch,
            &stream,
        )?;
        enqueue_f32_htod(&workspace.top_p_host, &workspace.top_p, batch, &stream)?;
        enqueue_f32_htod(&workspace.min_p_host, &workspace.min_p, batch, &stream)?;
        enqueue_f32_htod(
            &workspace.accept_uniforms_host,
            &workspace.accept_uniforms,
            draft_elems,
            &stream,
        )?;
        enqueue_f32_htod(
            &workspace.sample_uniforms_host,
            &workspace.sample_uniforms,
            batch,
            &stream,
        )?;
        if let Some(row_elems) = row_elems {
            enqueue_f32_htod(
                &workspace.row_inverse_temperatures_host,
                &workspace.row_inverse_temperatures,
                row_elems,
                &stream,
            )?;
        }

        let draft_tokens = match input.draft_tokens {
            SparseRejectionDraftInput::Host(_) => {
                let tokens = workspace
                    .draft_tokens
                    .narrow(0, 0, draft_elems)?
                    .reshape((batch, drafts))?;
                SparseRejectionDeviceTensor::from_tensor(&tokens)
            }
            #[cfg(test)]
            SparseRejectionDraftInput::Device(tokens) => {
                SparseRejectionDeviceTensor::from_tensor(tokens)
            }
            SparseRejectionDraftInput::DeviceRows(tokens) => {
                SparseRejectionDeviceTensor::from_rows(
                    tokens,
                    DType::U32,
                    &[drafts],
                    input.target_logits.device(),
                )?
            }
        };
        let (q_token_ids, q_probs) = match input.proposal {
            SparseRejectionProposalInput::Deterministic => (
                draft_tokens.reshape(Shape::from_dims(&[batch, drafts, 1]))?,
                SparseRejectionDeviceTensor::from_tensor(
                    &workspace
                        .deterministic_q_probs
                        .narrow(0, 0, draft_elems)?
                        .reshape((batch, drafts, 1))?,
                ),
            ),
            #[cfg(test)]
            SparseRejectionProposalInput::Sparse { token_ids, probs } => (
                SparseRejectionDeviceTensor::from_tensor(token_ids),
                SparseRejectionDeviceTensor::from_tensor(probs),
            ),
            SparseRejectionProposalInput::SparseRows { token_ids, probs } => {
                let Some(first) = token_ids.first() else {
                    candle_core::bail!("{OP} requires sparse proposal rows");
                };
                let [row_drafts, q_width] = first.dims() else {
                    candle_core::bail!(
                        "{OP} expected sparse proposal rows with shape [drafts, q_width], got {:?}",
                        first.dims()
                    );
                };
                if *row_drafts != drafts {
                    candle_core::bail!(
                        "{OP} expected sparse proposal rows with {drafts} drafts, got {row_drafts}"
                    );
                }
                let row_shape = [drafts, *q_width];
                (
                    SparseRejectionDeviceTensor::from_rows(
                        token_ids,
                        DType::U32,
                        &row_shape,
                        input.target_logits.device(),
                    )?,
                    SparseRejectionDeviceTensor::from_rows(
                        probs,
                        DType::F32,
                        &row_shape,
                        input.target_logits.device(),
                    )?,
                )
            }
        };
        let inverse_temperatures = workspace.inverse_temperatures.narrow(0, 0, batch)?;
        let target_top_k = workspace.target_top_k.narrow(0, 0, batch)?;
        let top_p = workspace.top_p.narrow(0, 0, batch)?;
        let min_p = workspace.min_p.narrow(0, 0, batch)?;
        let accept_uniforms = workspace
            .accept_uniforms
            .narrow(0, 0, draft_elems)?
            .reshape((batch, drafts))?;
        let sample_uniforms = workspace.sample_uniforms.narrow(0, 0, batch)?;
        let row_inverse_temperatures = row_elems
            .map(|row_elems| workspace.row_inverse_temperatures.narrow(0, 0, row_elems))
            .transpose()?;
        let outcome_elems = batch * SPARSE_REJECTION_OUTCOME_WIDTH;
        let outcomes = workspace
            .outcomes
            .narrow(0, 0, outcome_elems)?
            .reshape((batch, SPARSE_REJECTION_OUTCOME_WIDTH))?;
        let output = sparse_rejection_cuda_device_with_outcomes(
            SparseRejectionDeviceInput {
                target_logits: input.target_logits,
                draft_tokens: &draft_tokens,
                q_token_ids: &q_token_ids,
                q_probs: &q_probs,
                inverse_temperatures: &inverse_temperatures,
                target_top_k: &target_top_k,
                top_p: &top_p,
                min_p: &min_p,
                accept_uniforms: &accept_uniforms,
                sample_uniforms: &sample_uniforms,
                mode: input.mode,
            },
            Some(&outcomes),
            row_inverse_temperatures
                .as_ref()
                .map(|_| SparseRejectionTopKContext {
                    workspace: &mut workspace.topk,
                }),
        )?;
        enqueue_u32_dtoh(
            &outcomes,
            &mut workspace.outcomes_host,
            outcome_elems,
            &stream,
        )?;
        let device_drafts = match input.draft_tokens {
            #[cfg(test)]
            SparseRejectionDraftInput::Device(_) => true,
            SparseRejectionDraftInput::DeviceRows(_) => true,
            SparseRejectionDraftInput::Host(_) => false,
        };
        if device_drafts {
            enqueue_device_u32_dtoh(
                &draft_tokens,
                &mut workspace.draft_tokens_host,
                draft_elems,
                &stream,
            )?;
        }
        workspace
            .completion
            .record(&stream)
            .map_err(candle_core::Error::wrap)?;
        Ok(output)
    })();
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let _ = stream.synchronize();
            return Err(error);
        }
    };

    workspace.next_generation = generation.wrapping_add(1).max(1);
    workspace.pending = Some(CudaSparseRejectionPending {
        generation,
        batch,
        drafts,
        _output: output,
    });
    Ok(CudaSparseRejectionSubmission {
        workspace_id: workspace.id,
        generation,
        batch,
        drafts,
        completion: workspace.completion.clone(),
    })
}

pub(crate) fn sparse_rejection_cuda_complete(
    cache: &mut Option<CudaSparseRejectionWorkspace>,
    submission: &CudaSparseRejectionSubmission,
) -> Result<SparseRejectionCompletion> {
    let workspace = cache
        .as_mut()
        .ok_or_else(|| candle_core::Error::msg("sparse rejection workspace is missing"))?;
    if submission.workspace_id != workspace.id {
        candle_core::bail!("{OP} received a submission from a different workspace");
    }
    let Some(pending) = workspace.pending.as_ref() else {
        candle_core::bail!("{OP} received an inactive submission");
    };
    if pending.generation != submission.generation
        || pending.batch != submission.batch
        || pending.drafts != submission.drafts
    {
        candle_core::bail!("{OP} received a stale submission");
    }
    let wait_result = submission.wait();
    if let Err(error) = wait_result {
        workspace.pending = None;
        return Err(error);
    }
    let outcome_elems = submission.batch * SPARSE_REJECTION_OUTCOME_WIDTH;
    let rows = workspace.outcomes_host.as_slice()[..outcome_elems]
        .as_chunks::<SPARSE_REJECTION_OUTCOME_WIDTH>()
        .0
        .iter()
        .map(|row| SparseRejectionRow {
            accepted_count: row[0],
            continuation: row[1],
            status: row[2],
        })
        .collect();
    let draft_tokens = workspace.draft_tokens_host.as_slice()
        [..submission.batch * submission.drafts]
        .chunks_exact(submission.drafts)
        .map(ToOwned::to_owned)
        .collect();
    workspace.pending = None;
    Ok(SparseRejectionCompletion { rows, draft_tokens })
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
pub(crate) fn sparse_rejection_cuda(
    input: SparseRejectionInput<'_>,
) -> Result<SparseRejectionOutput> {
    let draft_tokens = SparseRejectionDeviceTensor::from_tensor(input.draft_tokens);
    let q_token_ids = SparseRejectionDeviceTensor::from_tensor(input.q_token_ids);
    let q_probs = SparseRejectionDeviceTensor::from_tensor(input.q_probs);
    sparse_rejection_cuda_device_with_outcomes(
        SparseRejectionDeviceInput {
            target_logits: input.target_logits,
            draft_tokens: &draft_tokens,
            q_token_ids: &q_token_ids,
            q_probs: &q_probs,
            inverse_temperatures: input.inverse_temperatures,
            target_top_k: input.target_top_k,
            top_p: input.top_p,
            min_p: input.min_p,
            accept_uniforms: input.accept_uniforms,
            sample_uniforms: input.sample_uniforms,
            mode: input.mode,
        },
        None,
        None,
    )
}

struct SparseRejectionTopKContext<'a> {
    workspace: &'a mut Option<CudaRankedTopKPackedWorkspace>,
}

#[allow(clippy::too_many_lines)]
fn sparse_rejection_cuda_device_with_outcomes(
    input: SparseRejectionDeviceInput<'_>,
    outcomes: Option<&Tensor>,
    mut topk_context: Option<SparseRejectionTopKContext<'_>>,
) -> Result<SparseRejectionOutput> {
    let shape = validate_input(&input)?;
    let packed_target = match input.mode {
        SparseRejectionMode::Categorical => None,
        SparseRejectionMode::BoundedTopK { max_top_k } => {
            if max_top_k == 0 || max_top_k > CUDA_TOPK_MAX_K {
                candle_core::bail!("{OP} max_top_k={max_top_k} must be in [1, {CUDA_TOPK_MAX_K}]");
            }
            let flattened = input
                .target_logits
                .reshape((shape.batch * shape.rows, shape.vocab))?;
            let packed = if let Some(context) = topk_context.as_mut() {
                cuda_topk_ranked_packed_batched_with_workspace(
                    &flattened,
                    max_top_k,
                    &mut *context.workspace,
                )?
            } else {
                cuda_topk_ranked_packed_batched(&flattened, max_top_k)?
            };
            Some(packed)
        }
    };
    let kernel_target = packed_target.as_ref().map_or_else(
        || input.target_logits.clone(),
        |packed| packed.packed.clone(),
    );

    let (target_storage, target_layout) = kernel_target.storage_and_layout();
    let target_storage = match &*target_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA target logits"),
    };
    let draft_tokens = input.draft_tokens.anchor();
    let (draft_storage, draft_layout) = draft_tokens.storage_and_layout();
    let draft_storage = match &*draft_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA draft tokens"),
    };
    let q_token_ids = input.q_token_ids.anchor();
    let (q_ids_storage, q_ids_layout) = q_token_ids.storage_and_layout();
    let q_ids_storage = match &*q_ids_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA q token ids"),
    };
    let q_probs = input.q_probs.anchor();
    let (q_probs_storage, q_probs_layout) = q_probs.storage_and_layout();
    let q_probs_storage = match &*q_probs_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA q probabilities"),
    };
    let (temperature_storage, temperature_layout) = input.inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA inverse temperatures"),
    };
    let (top_k_storage, top_k_layout) = input.target_top_k.storage_and_layout();
    let top_k_storage = match &*top_k_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA target top-k"),
    };
    let (top_p_storage, top_p_layout) = input.top_p.storage_and_layout();
    let top_p_storage = match &*top_p_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA top-p"),
    };
    let (min_p_storage, min_p_layout) = input.min_p.storage_and_layout();
    let min_p_storage = match &*min_p_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA min-p"),
    };
    let (accept_storage, accept_layout) = input.accept_uniforms.storage_and_layout();
    let accept_storage = match &*accept_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA accept uniforms"),
    };
    let (sample_storage, sample_layout) = input.sample_uniforms.storage_and_layout();
    let sample_storage = match &*sample_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA sample uniforms"),
    };

    let CudaStorageSlice::F32(target_slice) = &target_storage.slice else {
        candle_core::bail!("{OP} target storage dtype mismatch");
    };
    let CudaStorageSlice::U32(draft_slice) = &draft_storage.slice else {
        candle_core::bail!("{OP} draft storage dtype mismatch");
    };
    let CudaStorageSlice::U32(q_ids_slice) = &q_ids_storage.slice else {
        candle_core::bail!("{OP} q token id storage dtype mismatch");
    };
    let CudaStorageSlice::F32(q_probs_slice) = &q_probs_storage.slice else {
        candle_core::bail!("{OP} q probability storage dtype mismatch");
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} inverse temperature storage dtype mismatch");
    };
    let CudaStorageSlice::U32(top_k_slice) = &top_k_storage.slice else {
        candle_core::bail!("{OP} target top-k storage dtype mismatch");
    };
    let CudaStorageSlice::F32(top_p_slice) = &top_p_storage.slice else {
        candle_core::bail!("{OP} top-p storage dtype mismatch");
    };
    let CudaStorageSlice::F32(min_p_slice) = &min_p_storage.slice else {
        candle_core::bail!("{OP} min-p storage dtype mismatch");
    };
    let CudaStorageSlice::F32(accept_slice) = &accept_storage.slice else {
        candle_core::bail!("{OP} accept uniform storage dtype mismatch");
    };
    let CudaStorageSlice::F32(sample_slice) = &sample_storage.slice else {
        candle_core::bail!("{OP} sample uniform storage dtype mismatch");
    };

    let dev = target_storage.device();
    let stream = dev.cuda_stream();
    let (target_ptr, target_guard) = target_slice.device_ptr(&stream);
    let target_ptr = unsafe { (target_ptr as *const f32).add(target_layout.start_offset()) };
    let (draft_ptr, draft_guard) = draft_slice.device_ptr(&stream);
    let draft_ptr = unsafe { (draft_ptr as *const u32).add(draft_layout.start_offset()) };
    let (q_ids_ptr, q_ids_guard) = q_ids_slice.device_ptr(&stream);
    let q_ids_ptr = unsafe { (q_ids_ptr as *const u32).add(q_ids_layout.start_offset()) };
    let (q_probs_ptr, q_probs_guard) = q_probs_slice.device_ptr(&stream);
    let q_probs_ptr = unsafe { (q_probs_ptr as *const f32).add(q_probs_layout.start_offset()) };
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };
    let (top_k_ptr, top_k_guard) = top_k_slice.device_ptr(&stream);
    let top_k_ptr = unsafe { (top_k_ptr as *const u32).add(top_k_layout.start_offset()) };
    let (top_p_ptr, top_p_guard) = top_p_slice.device_ptr(&stream);
    let top_p_ptr = unsafe { (top_p_ptr as *const f32).add(top_p_layout.start_offset()) };
    let (min_p_ptr, min_p_guard) = min_p_slice.device_ptr(&stream);
    let min_p_ptr = unsafe { (min_p_ptr as *const f32).add(min_p_layout.start_offset()) };
    let (accept_ptr, accept_guard) = accept_slice.device_ptr(&stream);
    let accept_ptr = unsafe { (accept_ptr as *const f32).add(accept_layout.start_offset()) };
    let (sample_ptr, sample_guard) = sample_slice.device_ptr(&stream);
    let sample_ptr = unsafe { (sample_ptr as *const f32).add(sample_layout.start_offset()) };

    let outcomes = match outcomes {
        Some(outcomes) => {
            if outcomes.dtype() != DType::U32
                || outcomes.dims() != [shape.batch, SPARSE_REJECTION_OUTCOME_WIDTH]
                || !outcomes.is_contiguous()
                || !outcomes.device().same_device(input.target_logits.device())
            {
                candle_core::bail!(
                    "{OP} workspace outcomes must be contiguous CUDA U32 with shape [{}, {}]",
                    shape.batch,
                    SPARSE_REJECTION_OUTCOME_WIDTH
                );
            }
            outcomes.clone()
        }
        None => {
            let outcome_elems = shape.batch * SPARSE_REJECTION_OUTCOME_WIDTH;
            let outcomes = unsafe { dev.alloc::<u32>(outcome_elems) }?;
            Tensor::from((
                candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                    slice: CudaStorageSlice::U32(outcomes),
                    device: dev.clone(),
                }),
                Shape::from_dims(&[shape.batch, SPARSE_REJECTION_OUTCOME_WIDTH]),
            ))
        }
    };
    let packed_k = packed_target
        .as_ref()
        .map(|packed| i32::try_from(packed.k).map_err(candle_core::Error::wrap))
        .transpose()?;
    outcomes.inplace_op1(&SparseRejectionKernelLaunch {
        mode: input.mode,
        target_ptr: target_ptr as usize,
        draft_ptr: draft_ptr as usize,
        q_ids_ptr: q_ids_ptr as usize,
        q_probs_ptr: q_probs_ptr as usize,
        temperature_ptr: temperature_ptr as usize,
        top_k_ptr: top_k_ptr as usize,
        top_p_ptr: top_p_ptr as usize,
        min_p_ptr: min_p_ptr as usize,
        accept_ptr: accept_ptr as usize,
        sample_ptr: sample_ptr as usize,
        batch: shape.batch_i32,
        drafts: shape.drafts_i32,
        vocab: shape.vocab_i32,
        q_width: shape.q_width_i32,
        packed_k,
    })?;

    drop(target_guard);
    drop(draft_guard);
    drop(q_ids_guard);
    drop(q_probs_guard);
    drop(temperature_guard);
    drop(top_k_guard);
    drop(top_p_guard);
    drop(min_p_guard);
    drop(accept_guard);
    drop(sample_guard);

    let mut inputs = vec![
        input.target_logits.clone(),
        input.inverse_temperatures.clone(),
        input.target_top_k.clone(),
        input.top_p.clone(),
        input.min_p.clone(),
        input.accept_uniforms.clone(),
        input.sample_uniforms.clone(),
    ];
    input.draft_tokens.extend_inputs(&mut inputs);
    input.q_token_ids.extend_inputs(&mut inputs);
    input.q_probs.extend_inputs(&mut inputs);
    Ok(SparseRejectionOutput {
        _outcomes: outcomes.clone(),
        _inputs: inputs,
        _packed_target: packed_target,
        _row_inverse_temperatures: None,
    })
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use candle_core::Device;

    struct HostCase {
        batch: usize,
        drafts: usize,
        vocab: usize,
        q_width: usize,
        target_logits: Vec<f32>,
        draft_tokens: Vec<u32>,
        q_token_ids: Vec<u32>,
        q_probs: Vec<f32>,
        inverse_temperatures: Vec<f32>,
        target_top_k: Vec<u32>,
        top_p: Vec<f32>,
        min_p: Vec<f32>,
        accept_uniforms: Vec<f32>,
        sample_uniforms: Vec<f32>,
        mode: SparseRejectionMode,
    }

    struct DeviceCase {
        target_logits: Tensor,
        draft_tokens: Tensor,
        q_token_ids: Tensor,
        q_probs: Tensor,
        inverse_temperatures: Tensor,
        target_top_k: Tensor,
        top_p: Tensor,
        min_p: Tensor,
        accept_uniforms: Tensor,
        sample_uniforms: Tensor,
    }

    impl DeviceCase {
        fn run_with_target(
            &self,
            target_logits: &Tensor,
            mode: SparseRejectionMode,
        ) -> Result<Vec<SparseRejectionRow>> {
            sparse_rejection_cuda(SparseRejectionInput {
                target_logits,
                draft_tokens: &self.draft_tokens,
                q_token_ids: &self.q_token_ids,
                q_probs: &self.q_probs,
                inverse_temperatures: &self.inverse_temperatures,
                target_top_k: &self.target_top_k,
                top_p: &self.top_p,
                min_p: &self.min_p,
                accept_uniforms: &self.accept_uniforms,
                sample_uniforms: &self.sample_uniforms,
                mode,
            })?
            .to_rows()
        }
    }

    impl HostCase {
        fn to_device(&self, device: &Device) -> Result<DeviceCase> {
            let rows = self.drafts + 1;
            Ok(DeviceCase {
                target_logits: Tensor::from_vec(
                    self.target_logits.clone(),
                    (self.batch, rows, self.vocab),
                    device,
                )?,
                draft_tokens: Tensor::from_vec(
                    self.draft_tokens.clone(),
                    (self.batch, self.drafts),
                    device,
                )?,
                q_token_ids: Tensor::from_vec(
                    self.q_token_ids.clone(),
                    (self.batch, self.drafts, self.q_width),
                    device,
                )?,
                q_probs: Tensor::from_vec(
                    self.q_probs.clone(),
                    (self.batch, self.drafts, self.q_width),
                    device,
                )?,
                inverse_temperatures: Tensor::from_vec(
                    self.inverse_temperatures.clone(),
                    self.batch,
                    device,
                )?,
                target_top_k: Tensor::from_vec(self.target_top_k.clone(), self.batch, device)?,
                top_p: Tensor::from_vec(self.top_p.clone(), self.batch, device)?,
                min_p: Tensor::from_vec(self.min_p.clone(), self.batch, device)?,
                accept_uniforms: Tensor::from_vec(
                    self.accept_uniforms.clone(),
                    (self.batch, self.drafts),
                    device,
                )?,
                sample_uniforms: Tensor::from_vec(
                    self.sample_uniforms.clone(),
                    self.batch,
                    device,
                )?,
            })
        }

        fn run(&self, device: &Device) -> Result<Vec<SparseRejectionRow>> {
            let tensors = self.to_device(device)?;
            tensors.run_with_target(&tensors.target_logits, self.mode)
        }

        fn run_workspace(
            &self,
            device: &Device,
            workspace: &mut Option<CudaSparseRejectionWorkspace>,
            deterministic: bool,
        ) -> Result<Vec<SparseRejectionRow>> {
            let tensors = self.to_device(device)?;
            let proposal = if deterministic {
                SparseRejectionProposalInput::Deterministic
            } else {
                SparseRejectionProposalInput::Sparse {
                    token_ids: &tensors.q_token_ids,
                    probs: &tensors.q_probs,
                }
            };
            let submission = sparse_rejection_cuda_submit(
                SparseRejectionWorkspaceInput {
                    target_logits: &tensors.target_logits,
                    proposal,
                    draft_tokens: SparseRejectionDraftInput::Host(&self.draft_tokens),
                    inverse_temperatures: &self.inverse_temperatures,
                    target_top_k: &self.target_top_k,
                    top_p: &self.top_p,
                    min_p: &self.min_p,
                    accept_uniforms: &self.accept_uniforms,
                    sample_uniforms: &self.sample_uniforms,
                    mode: self.mode,
                },
                workspace,
            )?;
            Ok(sparse_rejection_cuda_complete(workspace, &submission)?.rows)
        }
    }

    fn active_filter(value: f32) -> bool {
        value > 0.0 && value < 1.0
    }

    fn target_distribution(case: &HostCase, sequence: usize, row: usize) -> Vec<f32> {
        let offset = (sequence * (case.drafts + 1) + row) * case.vocab;
        let logits = &case.target_logits[offset..offset + case.vocab];
        let inverse_temperature = case.inverse_temperatures[sequence];
        let scaled_max = logits
            .iter()
            .map(|value| value * inverse_temperature)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights = logits
            .iter()
            .map(|value| (value * inverse_temperature - scaled_max).exp())
            .collect::<Vec<_>>();

        if matches!(case.mode, SparseRejectionMode::BoundedTopK { .. }) {
            let mut sorted = (0..case.vocab).collect::<Vec<_>>();
            sorted.sort_by(|left, right| {
                logits[*right]
                    .total_cmp(&logits[*left])
                    .then_with(|| left.cmp(right))
            });
            let count = (case.target_top_k[sequence] as usize).min(case.vocab);
            for token in sorted.iter().skip(count) {
                weights[*token] = 0.0;
            }
            let kept = &sorted[..count];
            if active_filter(case.top_p[sequence]) {
                let cutoff =
                    case.top_p[sequence] * kept.iter().map(|token| weights[*token]).sum::<f32>();
                let mut cumulative = 0.0f32;
                for token in kept {
                    if cumulative >= cutoff {
                        weights[*token] = 0.0;
                    } else {
                        cumulative += weights[*token];
                    }
                }
            }
            if active_filter(case.min_p[sequence]) {
                let threshold = case.min_p[sequence];
                for token in kept {
                    if threshold >= weights[*token] {
                        weights[*token] = 0.0;
                    }
                }
            }
        }

        let denominator = weights.iter().sum::<f32>();
        for weight in &mut weights {
            *weight /= denominator;
        }
        weights
    }

    fn q_distribution(case: &HostCase, sequence: usize, row: usize) -> Vec<f32> {
        let offset = (sequence * case.drafts + row) * case.q_width;
        let ids = &case.q_token_ids[offset..offset + case.q_width];
        let probabilities = &case.q_probs[offset..offset + case.q_width];
        let denominator = probabilities
            .iter()
            .map(|probability| *probability as f64)
            .sum::<f64>();
        let mut normalized = vec![0.0f32; case.vocab];
        for (&id, &probability) in ids.iter().zip(probabilities) {
            normalized[id as usize] += (probability as f64 / denominator) as f32;
        }
        normalized
    }

    fn sample_distribution(probabilities: &[f32], uniform: f32) -> u32 {
        let total = probabilities.iter().sum::<f32>();
        let target = uniform * total;
        let mut cumulative = 0.0f32;
        for (token, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if target < cumulative {
                return token as u32;
            }
        }
        probabilities
            .iter()
            .rposition(|probability| *probability > 0.0)
            .expect("reference distribution has positive mass") as u32
    }

    fn reference(case: &HostCase) -> Vec<SparseRejectionRow> {
        (0..case.batch)
            .map(|sequence| {
                let mut accepted_count = 0u32;
                for row in 0..case.drafts {
                    let target = target_distribution(case, sequence, row);
                    let q = q_distribution(case, sequence, row);
                    let draft = case.draft_tokens[sequence * case.drafts + row] as usize;
                    let accept_probability = (target[draft] / q[draft]).min(1.0);
                    let uniform = case.accept_uniforms[sequence * case.drafts + row];
                    if uniform < accept_probability {
                        accepted_count += 1;
                        continue;
                    }
                    let residual = target
                        .iter()
                        .zip(q)
                        .map(|(target, q)| (target - q).max(0.0))
                        .collect::<Vec<_>>();
                    let continuation = if residual.iter().sum::<f32>() > 0.0 {
                        sample_distribution(&residual, case.sample_uniforms[sequence])
                    } else {
                        sample_distribution(&target, case.sample_uniforms[sequence])
                    };
                    return SparseRejectionRow {
                        accepted_count,
                        continuation,
                        status: SPARSE_REJECTION_STATUS_OK,
                    };
                }
                let target = target_distribution(case, sequence, case.drafts);
                SparseRejectionRow {
                    accepted_count,
                    continuation: sample_distribution(&target, case.sample_uniforms[sequence]),
                    status: SPARSE_REJECTION_STATUS_OK,
                }
            })
            .collect()
    }

    fn logits(probability_rows: &[[f32; 5]]) -> Vec<f32> {
        probability_rows
            .iter()
            .flat_map(|row| row.iter().map(|probability| probability.ln()))
            .collect()
    }

    fn deterministic_workspace_case(batch: usize, drafts: usize) -> HostCase {
        let vocab = 5;
        let draft_tokens = (0..batch * drafts)
            .map(|idx| (idx % vocab) as u32)
            .collect::<Vec<_>>();
        let target_logits = (0..batch * (drafts + 1))
            .flat_map(|row| {
                (0..vocab).map(move |token| ((row * 3 + token * 2) % 11) as f32 * 0.25 - 1.0)
            })
            .collect::<Vec<_>>();
        HostCase {
            batch,
            drafts,
            vocab,
            q_width: 1,
            target_logits,
            q_token_ids: draft_tokens.clone(),
            q_probs: vec![1.0; batch * drafts],
            draft_tokens,
            inverse_temperatures: vec![1.0; batch],
            target_top_k: vec![0; batch],
            top_p: vec![0.0; batch],
            min_p: vec![0.0; batch],
            accept_uniforms: (0..batch * drafts)
                .map(|idx| 0.1 + 0.8 * (idx % 7) as f32 / 7.0)
                .collect(),
            sample_uniforms: (0..batch)
                .map(|idx| 0.2 + 0.6 * (idx % 5) as f32 / 5.0)
                .collect(),
            mode: SparseRejectionMode::Categorical,
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn dense_rows_require_shared_ordered_tight_backing() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let packed = Tensor::from_vec((0..24u32).collect(), (4, 3, 2), &device)?;
        let rows = (0..4)
            .map(|row| packed.get(row))
            .collect::<Result<Vec<_>>>()?;
        let dense = dense_cuda_rows(&rows, DType::U32, &[3, 2], &device)?
            .expect("packed row views must remain dense");
        assert_eq!(dense.shape.dims(), [4, 3, 2]);

        let reordered = vec![
            rows[1].clone(),
            rows[0].clone(),
            rows[2].clone(),
            rows[3].clone(),
        ];
        assert!(dense_cuda_rows(&reordered, DType::U32, &[3, 2], &device)?.is_none());
        assert!(matches!(
            SparseRejectionDeviceTensor::from_rows(&reordered, DType::U32, &[3, 2], &device)?,
            SparseRejectionDeviceTensor::Tensor(_)
        ));

        let gapped = vec![rows[0].clone(), rows[2].clone()];
        assert!(dense_cuda_rows(&gapped, DType::U32, &[3, 2], &device)?.is_none());

        let separate = (0..4)
            .map(|row| Tensor::from_vec(vec![row as u32; 6], (3, 2), &device))
            .collect::<Result<Vec<_>>>()?;
        assert!(dense_cuda_rows(&separate, DType::U32, &[3, 2], &device)?.is_none());

        let transposed = rows
            .iter()
            .map(|row| row.transpose(0, 1))
            .collect::<Result<Vec<_>>>()?;
        assert!(dense_cuda_rows(&transposed, DType::U32, &[2, 3], &device)?.is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn dense_rows_match_packed_inputs_and_survive_owner_drop() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for bounded_topk in [false, true] {
            let mut case = deterministic_workspace_case(4, 3);
            if bounded_topk {
                case.target_top_k.fill(case.vocab as u32);
                case.mode = SparseRejectionMode::BoundedTopK {
                    max_top_k: case.vocab,
                };
            }
            let tensors = case.to_device(&device)?;
            let mut packed_workspace = None;
            let packed_submission = sparse_rejection_cuda_submit(
                SparseRejectionWorkspaceInput {
                    target_logits: &tensors.target_logits,
                    proposal: SparseRejectionProposalInput::Sparse {
                        token_ids: &tensors.q_token_ids,
                        probs: &tensors.q_probs,
                    },
                    draft_tokens: SparseRejectionDraftInput::Device(&tensors.draft_tokens),
                    inverse_temperatures: &case.inverse_temperatures,
                    target_top_k: &case.target_top_k,
                    top_p: &case.top_p,
                    min_p: &case.min_p,
                    accept_uniforms: &case.accept_uniforms,
                    sample_uniforms: &case.sample_uniforms,
                    mode: case.mode,
                },
                &mut packed_workspace,
            )?;
            let packed = sparse_rejection_cuda_complete(&mut packed_workspace, &packed_submission)?;

            let draft_rows = (0..case.batch)
                .map(|row| tensors.draft_tokens.get(row))
                .collect::<Result<Vec<_>>>()?;
            let q_token_rows = (0..case.batch)
                .map(|row| tensors.q_token_ids.get(row))
                .collect::<Result<Vec<_>>>()?;
            let q_prob_rows = (0..case.batch)
                .map(|row| tensors.q_probs.get(row))
                .collect::<Result<Vec<_>>>()?;
            assert!(dense_cuda_rows(&draft_rows, DType::U32, &[case.drafts], &device)?.is_some());
            assert!(dense_cuda_rows(
                &q_token_rows,
                DType::U32,
                &[case.drafts, case.q_width],
                &device
            )?
            .is_some());
            assert!(dense_cuda_rows(
                &q_prob_rows,
                DType::F32,
                &[case.drafts, case.q_width],
                &device
            )?
            .is_some());

            let mut row_workspace = None;
            let row_submission = sparse_rejection_cuda_submit(
                SparseRejectionWorkspaceInput {
                    target_logits: &tensors.target_logits,
                    proposal: SparseRejectionProposalInput::SparseRows {
                        token_ids: &q_token_rows,
                        probs: &q_prob_rows,
                    },
                    draft_tokens: SparseRejectionDraftInput::DeviceRows(&draft_rows),
                    inverse_temperatures: &case.inverse_temperatures,
                    target_top_k: &case.target_top_k,
                    top_p: &case.top_p,
                    min_p: &case.min_p,
                    accept_uniforms: &case.accept_uniforms,
                    sample_uniforms: &case.sample_uniforms,
                    mode: case.mode,
                },
                &mut row_workspace,
            )?;
            drop(draft_rows);
            drop(q_token_rows);
            drop(q_prob_rows);
            drop(tensors);
            let rows = sparse_rejection_cuda_complete(&mut row_workspace, &row_submission)?;
            assert_eq!(rows.rows, packed.rows);
            assert_eq!(rows.draft_tokens, packed.draft_tokens);
            assert_eq!(rows.rows, reference(&case));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn workspace_reuses_grows_and_supports_deterministic_proposals() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let mut workspace = None;
        let first = deterministic_workspace_case(2, 2);
        let expected = reference(&first);
        assert_eq!(
            first.run_workspace(&device, &mut workspace, true)?,
            expected
        );
        let first_workspace = workspace.as_ref().expect("workspace was allocated");
        let first_id = first_workspace.id;
        let first_batch_capacity = first_workspace.capacity_batch;
        let first_draft_capacity = first_workspace.capacity_drafts;

        assert_eq!(
            first.run_workspace(&device, &mut workspace, false)?,
            expected
        );
        let reused = workspace.as_ref().expect("workspace was retained");
        assert_eq!(reused.id, first_id);
        assert_eq!(reused.capacity_batch, first_batch_capacity);
        assert_eq!(reused.capacity_drafts, first_draft_capacity);

        let larger = deterministic_workspace_case(3, 3);
        assert_eq!(
            larger.run_workspace(&device, &mut workspace, true)?,
            reference(&larger)
        );
        let grown = workspace.as_ref().expect("workspace was grown");
        assert_ne!(grown.id, first_id);
        assert!(grown.capacity_batch >= larger.batch);
        assert!(grown.capacity_drafts >= larger.drafts);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn workspace_accepts_device_drafts_and_materializes_with_outcomes() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let case = deterministic_workspace_case(3, 4);
        let tensors = case.to_device(&device)?;
        let mut workspace = None;
        let submission = sparse_rejection_cuda_submit(
            SparseRejectionWorkspaceInput {
                target_logits: &tensors.target_logits,
                proposal: SparseRejectionProposalInput::Deterministic,
                draft_tokens: SparseRejectionDraftInput::Device(&tensors.draft_tokens),
                inverse_temperatures: &case.inverse_temperatures,
                target_top_k: &case.target_top_k,
                top_p: &case.top_p,
                min_p: &case.min_p,
                accept_uniforms: &case.accept_uniforms,
                sample_uniforms: &case.sample_uniforms,
                mode: case.mode,
            },
            &mut workspace,
        )?;
        let completion = sparse_rejection_cuda_complete(&mut workspace, &submission)?;
        assert_eq!(completion.rows, reference(&case));
        assert_eq!(
            completion.draft_tokens,
            case.draft_tokens
                .chunks_exact(case.drafts)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn workspace_reuses_bounded_topk_scratch_and_row_temperatures() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let mut workspace = None;
        let mut case = deterministic_workspace_case(2, 2);
        case.inverse_temperatures = vec![0.5, 1.25];
        case.target_top_k = vec![5; case.batch];
        case.mode = SparseRejectionMode::BoundedTopK { max_top_k: 5 };
        let expected = reference(&case);

        assert_eq!(case.run_workspace(&device, &mut workspace, true)?, expected);
        let topk = workspace
            .as_ref()
            .and_then(|workspace| workspace.topk.as_ref())
            .expect("bounded top-k workspace was allocated") as *const _;

        assert_eq!(
            case.run_workspace(&device, &mut workspace, false)?,
            expected
        );
        let reused = workspace
            .as_ref()
            .and_then(|workspace| workspace.topk.as_ref())
            .expect("bounded top-k workspace was retained") as *const _;
        assert_eq!(topk, reused);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn workspace_rejects_overlap_and_stale_completion() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let case = deterministic_workspace_case(2, 2);
        let tensors = case.to_device(&device)?;
        let input = || SparseRejectionWorkspaceInput {
            target_logits: &tensors.target_logits,
            proposal: SparseRejectionProposalInput::Deterministic,
            draft_tokens: SparseRejectionDraftInput::Host(&case.draft_tokens),
            inverse_temperatures: &case.inverse_temperatures,
            target_top_k: &case.target_top_k,
            top_p: &case.top_p,
            min_p: &case.min_p,
            accept_uniforms: &case.accept_uniforms,
            sample_uniforms: &case.sample_uniforms,
            mode: case.mode,
        };
        let mut workspace = None;
        let submission = sparse_rejection_cuda_submit(input(), &mut workspace)?;
        let overlap = match sparse_rejection_cuda_submit(input(), &mut workspace) {
            Ok(_) => candle_core::bail!("overlapping submissions must fail"),
            Err(error) => error,
        };
        assert!(overlap.to_string().contains("pending submission"));
        sparse_rejection_cuda_complete(&mut workspace, &submission)?;
        let next = sparse_rejection_cuda_submit(input(), &mut workspace)?;
        let stale = sparse_rejection_cuda_complete(&mut workspace, &submission)
            .expect_err("an old generation must not complete a new submission");
        assert!(stale.to_string().contains("stale submission"));
        sparse_rejection_cuda_complete(&mut workspace, &next)?;
        let inactive = sparse_rejection_cuda_complete(&mut workspace, &next)
            .expect_err("a submission must complete once");
        assert!(inactive.to_string().contains("inactive submission"));
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn categorical_matches_cpu_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let target_logits = logits(&[
            [0.05, 0.20, 0.20, 0.30, 0.25],
            [0.10, 0.20, 0.30, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20, 0.20],
            [0.55, 0.10, 0.10, 0.15, 0.10],
            [0.10, 0.03, 0.40, 0.25, 0.22],
            [0.10, 0.20, 0.30, 0.25, 0.15],
            [0.10, 0.10, 0.60, 0.10, 0.10],
            [0.10, 0.10, 0.10, 0.60, 0.10],
            [0.05, 0.10, 0.15, 0.20, 0.50],
        ]);
        let case = HostCase {
            batch: 3,
            drafts: 2,
            vocab: 5,
            q_width: 3,
            target_logits,
            draft_tokens: vec![0, 4, 0, 1, 2, 3],
            q_token_ids: vec![0, 1, 2, 4, 1, 2, 0, 1, 2, 1, 2, 3, 2, 0, 4, 3, 1, 4],
            q_probs: vec![
                0.90, 0.05, 0.05, 0.50, 0.25, 0.25, 0.20, 0.40, 0.40, 0.80, 0.10, 0.10, 0.40, 0.30,
                0.30, 0.50, 0.25, 0.25,
            ],
            inverse_temperatures: vec![1.0, 1.0, 1.0],
            target_top_k: vec![0, 0, 0],
            top_p: vec![0.0, 0.0, 0.0],
            min_p: vec![0.0, 0.0, 0.0],
            accept_uniforms: vec![0.80, 0.20, 0.90, 0.70, 0.99, 0.99],
            sample_uniforms: vec![0.67, 0.45, 0.76],
            mode: SparseRejectionMode::Categorical,
        };
        let expected = reference(&case);
        assert_eq!(
            expected
                .iter()
                .map(|row| row.accepted_count)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(case.run(&device)?, expected);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn bounded_topk_matches_cpu_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let case = HostCase {
            batch: 2,
            drafts: 2,
            vocab: 7,
            q_width: 4,
            target_logits: [
                [0.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 0.0, 7.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 5.0],
                [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
            .concat(),
            draft_tokens: vec![2, 2, 0, 1],
            q_token_ids: vec![2, 0, 1, 3, 2, 4, 5, 6, 0, 1, 2, 3, 1, 6, 0, 2],
            q_probs: vec![
                0.10, 0.30, 0.30, 0.30, 0.10, 0.30, 0.30, 0.30, 0.20, 0.30, 0.30, 0.20, 0.70, 0.10,
                0.10, 0.10,
            ],
            inverse_temperatures: vec![1.0, 1.0],
            target_top_k: vec![1, 4],
            top_p: vec![0.0, 0.70],
            min_p: vec![0.0, 0.15],
            accept_uniforms: vec![0.99, 0.99, 0.90, 0.50],
            sample_uniforms: vec![0.30, 0.80],
            mode: SparseRejectionMode::BoundedTopK { max_top_k: 4 },
        };
        let expected = reference(&case);
        assert_eq!(
            expected
                .iter()
                .map(|row| row.accepted_count)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert_eq!(case.run(&device)?, expected);

        let tensors = case.to_device(&device)?;
        for dtype in [DType::BF16, DType::F16] {
            let low_precision = tensors.target_logits.to_dtype(dtype)?.contiguous()?;
            let reference = low_precision.to_dtype(DType::F32)?.contiguous()?;
            let expected = tensors.run_with_target(&reference, case.mode)?;
            let actual = tensors.run_with_target(&low_precision, case.mode)?;
            assert_eq!(actual, expected);
        }

        let bf16 = tensors.target_logits.to_dtype(DType::BF16)?.contiguous()?;
        let error = match tensors.run_with_target(&bf16, SparseRejectionMode::Categorical) {
            Ok(_) => candle_core::bail!("categorical sparse rejection accepted BF16 logits"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("categorical mode requires F32 target logits"));
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn ranked_topk_20_32_matches_cpu_reference_with_ties_and_top_p() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for max_top_k in [20, 32] {
            let batch = 2;
            let drafts = 2;
            let vocab = 64;
            let draft_tokens = vec![0, 7, 13, 31];
            let case = HostCase {
                batch,
                drafts,
                vocab,
                q_width: 1,
                target_logits: (0..batch * (drafts + 1))
                    .flat_map(|row| {
                        (0..vocab).map(move |token| ((token * 7 + row * 3) % 13) as f32 * 0.2 - 1.0)
                    })
                    .collect(),
                draft_tokens: draft_tokens.clone(),
                q_token_ids: draft_tokens,
                q_probs: vec![1.0; batch * drafts],
                inverse_temperatures: vec![0.8, 1.2],
                target_top_k: vec![max_top_k as u32; batch],
                top_p: vec![0.55, 0.72],
                min_p: vec![0.05, 0.10],
                accept_uniforms: vec![0.99; batch * drafts],
                sample_uniforms: vec![0.27, 0.83],
                mode: SparseRejectionMode::BoundedTopK { max_top_k },
            };
            let expected = reference(&case);
            assert_eq!(case.run(&device)?, expected);

            let mut workspace = None;
            assert_eq!(case.run_workspace(&device, &mut workspace, true)?, expected);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn reports_per_row_fallback_and_validation_status() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let mut q_probs = vec![0.5f32; 8];
        q_probs[2] = -1.0;
        q_probs[3] = 2.0;
        let case = HostCase {
            batch: 4,
            drafts: 1,
            vocab: 3,
            q_width: 2,
            target_logits: vec![0.0; 24],
            draft_tokens: vec![0; 4],
            q_token_ids: vec![0, 1, 0, 1, 0, 1, 0, 1],
            q_probs,
            inverse_temperatures: vec![1.0, 1.0, 0.0, 1.0],
            target_top_k: vec![1, 0, 0, 0],
            top_p: vec![0.0; 4],
            min_p: vec![0.0; 4],
            accept_uniforms: vec![0.0; 4],
            sample_uniforms: vec![0.5, 0.5, 0.5, 1.0],
            mode: SparseRejectionMode::Categorical,
        };
        let rows = case.run(&device)?;
        assert_eq!(
            rows.iter().map(|row| row.status).collect::<Vec<_>>(),
            vec![
                SPARSE_REJECTION_STATUS_NEEDS_CPU,
                SPARSE_REJECTION_STATUS_INVALID_Q,
                SPARSE_REJECTION_STATUS_INVALID_TARGET,
                SPARSE_REJECTION_STATUS_INVALID_RNG,
            ]
        );
        assert!(rows.iter().all(|row| {
            row.accepted_count == SPARSE_REJECTION_INVALID_VALUE
                && row.continuation == SPARSE_REJECTION_INVALID_VALUE
        }));
        Ok(())
    }
}
