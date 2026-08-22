use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::{DType, Result, Shape, Tensor};

use crate::ops::{cuda_topk_logits_packed_batched, TopKLogitsPackedOutput, CUDA_TOPK_MAX_K};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseRejectionRow {
    pub(crate) accepted_count: u32,
    pub(crate) continuation: u32,
    pub(crate) status: u32,
}

pub(crate) struct SparseRejectionOutput {
    outcomes: Tensor,
    _inputs: Vec<Tensor>,
    _packed_target: Option<TopKLogitsPackedOutput>,
    _row_inverse_temperatures: Option<Tensor>,
}

impl SparseRejectionOutput {
    pub(crate) fn to_rows(&self) -> Result<Vec<SparseRejectionRow>> {
        self.outcomes
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

fn validate_input(input: &SparseRejectionInput<'_>) -> Result<SparseRejectionShape> {
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
    let specs = [
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
    for (tensor, dtype, shape, name) in specs {
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

#[allow(clippy::too_many_lines)]
pub(crate) fn sparse_rejection_cuda(
    input: SparseRejectionInput<'_>,
) -> Result<SparseRejectionOutput> {
    let shape = validate_input(&input)?;
    let mut row_inverse_temperatures = None;
    let packed_target = match input.mode {
        SparseRejectionMode::Categorical => None,
        SparseRejectionMode::BoundedTopK { max_top_k } => {
            if max_top_k == 0 || max_top_k > CUDA_TOPK_MAX_K {
                candle_core::bail!("{OP} max_top_k={max_top_k} must be in [1, {CUDA_TOPK_MAX_K}]");
            }
            let repeated = input
                .inverse_temperatures
                .unsqueeze(1)?
                .broadcast_as((shape.batch, shape.rows))?
                .flatten_all()?
                .contiguous()?;
            let flattened = input
                .target_logits
                .reshape((shape.batch * shape.rows, shape.vocab))?;
            let packed = cuda_topk_logits_packed_batched(&flattened, max_top_k, &repeated)?;
            row_inverse_temperatures = Some(repeated);
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
    let (draft_storage, draft_layout) = input.draft_tokens.storage_and_layout();
    let draft_storage = match &*draft_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA draft tokens"),
    };
    let (q_ids_storage, q_ids_layout) = input.q_token_ids.storage_and_layout();
    let q_ids_storage = match &*q_ids_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA q token ids"),
    };
    let (q_probs_storage, q_probs_layout) = input.q_probs.storage_and_layout();
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

    let outcome_elems = shape.batch * SPARSE_REJECTION_OUTCOME_WIDTH;
    let mut outcomes = unsafe { dev.alloc::<u32>(outcome_elems) }?;
    let (outcomes_ptr, outcomes_guard) = outcomes.device_ptr_mut(&stream);
    unsafe {
        match input.mode {
            SparseRejectionMode::Categorical => ffi::sparse_rejection_categorical_f32(
                target_ptr,
                draft_ptr,
                q_ids_ptr,
                q_probs_ptr,
                temperature_ptr,
                top_k_ptr,
                top_p_ptr,
                min_p_ptr,
                accept_ptr,
                sample_ptr,
                outcomes_ptr as *mut u32,
                shape.batch_i32,
                shape.drafts_i32,
                shape.vocab_i32,
                shape.q_width_i32,
                stream.cu_stream() as i64,
            ),
            SparseRejectionMode::BoundedTopK { .. } => ffi::sparse_rejection_topk_f32(
                target_ptr,
                draft_ptr,
                q_ids_ptr,
                q_probs_ptr,
                temperature_ptr,
                top_k_ptr,
                top_p_ptr,
                min_p_ptr,
                accept_ptr,
                sample_ptr,
                outcomes_ptr as *mut u32,
                shape.batch_i32,
                shape.drafts_i32,
                shape.vocab_i32,
                shape.q_width_i32,
                i32::try_from(
                    packed_target
                        .as_ref()
                        .expect("bounded top-k has packed targets")
                        .k,
                )
                .map_err(candle_core::Error::wrap)?,
                stream.cu_stream() as i64,
            ),
        }
    }

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
    drop(outcomes_guard);

    let outcomes = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(outcomes),
            device: dev.clone(),
        }),
        Shape::from_dims(&[shape.batch, SPARSE_REJECTION_OUTCOME_WIDTH]),
    ));
    Ok(SparseRejectionOutput {
        outcomes,
        _inputs: vec![
            input.target_logits.clone(),
            input.draft_tokens.clone(),
            input.q_token_ids.clone(),
            input.q_probs.clone(),
            input.inverse_temperatures.clone(),
            input.target_top_k.clone(),
            input.top_p.clone(),
            input.min_p.clone(),
            input.accept_uniforms.clone(),
            input.sample_uniforms.clone(),
        ],
        _packed_target: packed_target,
        _row_inverse_temperatures: row_inverse_temperatures,
    })
}

#[cfg(test)]
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
