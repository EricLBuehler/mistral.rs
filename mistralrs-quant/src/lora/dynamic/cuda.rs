use candle_core::DType;

#[cfg(feature = "cuda")]
use super::{LoraExecution, LoraSiteHandle};

pub(crate) const MAX_CUDA_LORA_RANK: usize = 128;
const CUDA_EXPAND_WARPS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CudaPlan {
    dtype: DType,
    rows: usize,
    input_features: usize,
    output_features: usize,
}

struct CudaLayout<'a> {
    input_dtype: DType,
    output_dtype: DType,
    input_dims: &'a [usize],
    output_dims: &'a [usize],
    route_rows: usize,
    input_contiguous: bool,
    output_contiguous: bool,
    input_offset: usize,
}

fn plan_cuda(layout: CudaLayout<'_>) -> Option<CudaPlan> {
    if !matches!(layout.input_dtype, DType::F16 | DType::BF16)
        || layout.output_dtype != layout.input_dtype
        || !layout.input_contiguous
        || !layout.output_contiguous
        || layout.input_dims.len() < 2
        || layout.input_dims.len() != layout.output_dims.len()
        || !layout.input_offset.is_multiple_of(2)
    {
        return None;
    }

    let rank = layout.input_dims.len();
    if layout.input_dims[..rank - 1] != layout.output_dims[..rank - 1] {
        return None;
    }

    let input_features = layout.input_dims[rank - 1];
    let output_features = layout.output_dims[rank - 1];
    let rows = layout.input_dims[..rank - 1]
        .iter()
        .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))?;
    if rows == 0
        || input_features == 0
        || output_features == 0
        || !input_features.is_multiple_of(2)
        || rows != layout.route_rows
        || input_features > i32::MAX as usize
        || output_features > i32::MAX as usize
    {
        return None;
    }

    Some(CudaPlan {
        dtype: layout.input_dtype,
        rows,
        input_features,
        output_features,
    })
}

struct AdapterLayout<'a> {
    a_dtype: DType,
    b_dtype: DType,
    a_dims: &'a [usize],
    b_dims: &'a [usize],
    a_contiguous: bool,
    b_contiguous: bool,
    a_offset: usize,
    scale: f64,
}

fn adapter_supported(layout: AdapterLayout<'_>, plan: CudaPlan) -> bool {
    if layout.a_dtype != plan.dtype || layout.b_dtype != plan.dtype {
        return false;
    }
    if layout.a_dims.len() != 2
        || layout.b_dims.len() != 2
        || !layout.a_contiguous
        || !layout.b_contiguous
        || !layout.a_offset.is_multiple_of(2)
    {
        return false;
    }
    let rank = layout.a_dims[0];
    if rank == 0
        || rank > MAX_CUDA_LORA_RANK
        || layout.a_dims[1] != plan.input_features
        || layout.b_dims != [plan.output_features, rank]
    {
        return false;
    }
    let scale = layout.scale as f32;
    if !scale.is_finite() {
        return false;
    }

    let output_blocks = plan.output_features.div_ceil(CUDA_EXPAND_WARPS);
    plan.rows
        .checked_mul(rank)
        .is_some_and(|v| v <= i32::MAX as usize)
        && plan
            .rows
            .checked_mul(output_blocks)
            .is_some_and(|v| v <= i32::MAX as usize)
}

#[cfg(feature = "cuda")]
use candle_core::{
    cuda::{
        cudarc::driver::{DevicePtrMut, DeviceRepr},
        CudaDType,
    },
    CudaDevice, CudaStorage, Result, Shape, Storage, Tensor, WithDType,
};
#[cfg(feature = "cuda")]
use half::{bf16, f16};

#[cfg(feature = "cuda")]
use crate::utils::slice_ptr_on_stream;

#[cfg(feature = "cuda")]
struct ActiveAdapter<'a> {
    weights: &'a super::LoraWeights,
    row_indices: Tensor,
    rank: usize,
    scale: f64,
}

#[cfg(feature = "cuda")]
struct CudaLaunch {
    input: u64,
    a: u64,
    b: u64,
    row_indices: u64,
    hidden: u64,
    output: u64,
    input_features: i32,
    output_features: i32,
    rank: i32,
    active_rows: i32,
    scale: f32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[cfg(feature = "cuda")]
trait CudaLoraElement: CudaDType + DeviceRepr + WithDType {
    unsafe fn launch(args: CudaLaunch) -> i32;
    fn rounded_scale(scale: f64) -> f32;
}

#[cfg(feature = "cuda")]
impl CudaLoraElement for f16 {
    fn rounded_scale(scale: f64) -> f32 {
        <f16 as WithDType>::from_f64(scale).to_f32()
    }

    unsafe fn launch(args: CudaLaunch) -> i32 {
        super::cuda_ffi::launch_dynamic_lora_f16(
            args.input as *const f16,
            args.a as *const f16,
            args.b as *const f16,
            args.row_indices as *const u32,
            args.hidden as *mut f16,
            args.output as *mut f16,
            args.input_features,
            args.output_features,
            args.rank,
            args.active_rows,
            args.scale,
            args.stream,
        )
    }
}

#[cfg(feature = "cuda")]
impl CudaLoraElement for bf16 {
    fn rounded_scale(scale: f64) -> f32 {
        <bf16 as WithDType>::from_f64(scale).to_f32()
    }

    unsafe fn launch(args: CudaLaunch) -> i32 {
        super::cuda_ffi::launch_dynamic_lora_bf16(
            args.input as *const bf16,
            args.a as *const bf16,
            args.b as *const bf16,
            args.row_indices as *const u32,
            args.hidden as *mut bf16,
            args.output as *mut bf16,
            args.input_features,
            args.output_features,
            args.rank,
            args.active_rows,
            args.scale,
            args.stream,
        )
    }
}

#[cfg(feature = "cuda")]
fn run_cuda<T: CudaLoraElement>(
    device: &CudaDevice,
    input: &Tensor,
    base_output: &Tensor,
    plan: CudaPlan,
    adapters: &[ActiveAdapter<'_>],
) -> Result<Tensor> {
    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Cuda(input_storage) = &*input_storage else {
        candle_core::bail!("dynamic LoRA CUDA input storage is not CUDA");
    };
    let input_slice = input_storage.as_cuda_slice::<T>()?;

    let (base_storage, base_layout) = base_output.storage_and_layout();
    let Storage::Cuda(base_storage) = &*base_storage else {
        candle_core::bail!("dynamic LoRA CUDA output storage is not CUDA");
    };
    let base_slice = base_storage.as_cuda_slice::<T>()?;
    let output_len = base_output.elem_count();
    let base_view =
        base_slice.slice(base_layout.start_offset()..base_layout.start_offset() + output_len);
    let mut output = unsafe { device.alloc::<T>(output_len)? };
    device.memcpy_dtod(&base_view, &mut output)?;

    let stream = device.cuda_stream();
    let (input_ptr, _input_guard) =
        slice_ptr_on_stream(input_slice, input_layout.start_offset(), &stream);
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);

    for adapter in adapters {
        let (a_storage, a_layout) = adapter.weights.a.storage_and_layout();
        let Storage::Cuda(a_storage) = &*a_storage else {
            candle_core::bail!("dynamic LoRA CUDA A storage is not CUDA");
        };
        let a_slice = a_storage.as_cuda_slice::<T>()?;
        let (a_ptr, _a_guard) = slice_ptr_on_stream(a_slice, a_layout.start_offset(), &stream);

        let (b_storage, b_layout) = adapter.weights.b.storage_and_layout();
        let Storage::Cuda(b_storage) = &*b_storage else {
            candle_core::bail!("dynamic LoRA CUDA B storage is not CUDA");
        };
        let b_slice = b_storage.as_cuda_slice::<T>()?;
        let (b_ptr, _b_guard) = slice_ptr_on_stream(b_slice, b_layout.start_offset(), &stream);

        let (rows_storage, rows_layout) = adapter.row_indices.storage_and_layout();
        let Storage::Cuda(rows_storage) = &*rows_storage else {
            candle_core::bail!("dynamic LoRA CUDA row indices are not CUDA");
        };
        let rows_slice = rows_storage.as_cuda_slice::<u32>()?;
        let (rows_ptr, _rows_guard) =
            slice_ptr_on_stream(rows_slice, rows_layout.start_offset(), &stream);

        let active_rows = adapter.row_indices.elem_count();
        let mut hidden = unsafe { device.alloc::<T>(active_rows * adapter.rank)? };
        let (hidden_ptr, hidden_guard) = hidden.device_ptr_mut(&stream);
        let status = unsafe {
            T::launch(CudaLaunch {
                input: input_ptr,
                a: a_ptr,
                b: b_ptr,
                row_indices: rows_ptr,
                hidden: hidden_ptr,
                output: output_ptr,
                input_features: plan.input_features as i32,
                output_features: plan.output_features as i32,
                rank: adapter.rank as i32,
                active_rows: active_rows as i32,
                scale: T::rounded_scale(adapter.scale),
                stream: stream.cu_stream(),
            })
        };
        drop(hidden_guard);
        if status != 0 {
            candle_core::bail!("dynamic LoRA CUDA kernel launch failed with CUDA error {status}");
        }
    }
    drop(output_guard);

    let storage = CudaStorage::wrap_cuda_slice(output, device.clone());
    Ok(Tensor::from((
        Storage::Cuda(storage),
        Shape::from(base_output.dims()),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn try_add_delta_cuda(
    execution: &LoraExecution,
    site: &LoraSiteHandle,
    input: &Tensor,
    base_output: &Tensor,
) -> Result<Option<Tensor>> {
    if !input.device().is_cuda() || base_output.device().location() != input.device().location() {
        return Ok(None);
    }
    let Some(plan) = plan_cuda(CudaLayout {
        input_dtype: input.dtype(),
        output_dtype: base_output.dtype(),
        input_dims: input.dims(),
        output_dims: base_output.dims(),
        route_rows: execution.row_slots().len(),
        input_contiguous: input.is_contiguous(),
        output_contiguous: base_output.is_contiguous(),
        input_offset: input.layout().start_offset(),
    }) else {
        return Ok(None);
    };

    let mut adapters = Vec::new();
    for slot in execution.rows_by_slot().keys() {
        let Some(weights) = execution.weights(site, *slot)? else {
            continue;
        };
        if weights.a.device().location() != input.device().location()
            || weights.b.device().location() != input.device().location()
            || weights.a.dtype() != input.dtype()
            || weights.b.dtype() != input.dtype()
            || !adapter_supported(
                AdapterLayout {
                    a_dtype: weights.a.dtype(),
                    b_dtype: weights.b.dtype(),
                    a_dims: weights.a.dims(),
                    b_dims: weights.b.dims(),
                    a_contiguous: weights.a.is_contiguous(),
                    b_contiguous: weights.b.is_contiguous(),
                    a_offset: weights.a.layout().start_offset(),
                    scale: weights.scale,
                },
                plan,
            )
        {
            return Ok(None);
        }
        let row_indices = execution
            .row_indices(*slot, input.device())?
            .expect("active LoRA slot has cached rows");
        adapters.push(ActiveAdapter {
            weights,
            row_indices,
            rank: weights.a.dim(0)?,
            scale: weights.scale,
        });
    }
    if adapters.is_empty() {
        return Ok(Some(base_output.clone()));
    }

    let device = input.device().as_cuda_device()?;
    match input.dtype() {
        DType::F16 => run_cuda::<f16>(device, input, base_output, plan, &adapters).map(Some),
        DType::BF16 => run_cuda::<bf16>(device, input, base_output, plan, &adapters).map(Some),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_layout<'a>(
        dtype: DType,
        input_dims: &'a [usize],
        output_dims: &'a [usize],
        route_rows: usize,
    ) -> CudaLayout<'a> {
        CudaLayout {
            input_dtype: dtype,
            output_dtype: dtype,
            input_dims,
            output_dims,
            route_rows,
            input_contiguous: true,
            output_contiguous: true,
            input_offset: 0,
        }
    }

    #[test]
    fn cuda_dispatch_accepts_decode_and_prefill_inputs() {
        let rank_two = plan_cuda(cuda_layout(DType::F16, &[4, 4096], &[4, 11008], 4));
        assert_eq!(
            rank_two,
            Some(CudaPlan {
                dtype: DType::F16,
                rows: 4,
                input_features: 4096,
                output_features: 11008,
            })
        );
        assert!(plan_cuda(cuda_layout(DType::BF16, &[4, 1, 4096], &[4, 1, 11008], 4,)).is_some());
        assert!(plan_cuda(cuda_layout(DType::F16, &[4, 2, 4096], &[4, 2, 11008], 8,)).is_some());
    }

    #[test]
    fn cuda_dispatch_rejects_dtype_and_layout_mismatches() {
        assert!(plan_cuda(cuda_layout(DType::F32, &[4, 4096], &[4, 4096], 4,)).is_none());
        assert!(plan_cuda(cuda_layout(DType::F16, &[4, 4095], &[4, 4096], 4)).is_none());

        let mut non_contiguous = cuda_layout(DType::F16, &[4, 4096], &[4, 4096], 4);
        non_contiguous.input_contiguous = false;
        assert!(plan_cuda(non_contiguous).is_none());

        let mut unaligned = cuda_layout(DType::F16, &[4, 4096], &[4, 4096], 4);
        unaligned.input_offset = 1;
        assert!(plan_cuda(unaligned).is_none());
        assert!(plan_cuda(cuda_layout(DType::F16, &[4, 4096], &[4, 4096], 3,)).is_none());
    }

    #[test]
    fn adapter_dispatch_enforces_shapes_rank_and_scale() {
        let plan = plan_cuda(cuda_layout(DType::F16, &[4, 4096], &[4, 11008], 4))
            .expect("valid CUDA layout");
        let valid = AdapterLayout {
            a_dtype: DType::F16,
            b_dtype: DType::F16,
            a_dims: &[128, 4096],
            b_dims: &[11008, 128],
            a_contiguous: true,
            b_contiguous: true,
            a_offset: 0,
            scale: 0.5,
        };
        assert!(adapter_supported(valid, plan));

        assert!(!adapter_supported(
            AdapterLayout {
                a_dtype: DType::F16,
                b_dtype: DType::BF16,
                a_dims: &[16, 4096],
                b_dims: &[11008, 16],
                a_contiguous: true,
                b_contiguous: true,
                a_offset: 0,
                scale: 1.0,
            },
            plan,
        ));

        assert!(!adapter_supported(
            AdapterLayout {
                a_dtype: DType::F16,
                b_dtype: DType::F16,
                a_dims: &[MAX_CUDA_LORA_RANK + 1, 4096],
                b_dims: &[11008, MAX_CUDA_LORA_RANK + 1],
                a_contiguous: true,
                b_contiguous: true,
                a_offset: 0,
                scale: 1.0,
            },
            plan,
        ));
        assert!(!adapter_supported(
            AdapterLayout {
                a_dtype: DType::F16,
                b_dtype: DType::F16,
                a_dims: &[16, 4096],
                b_dims: &[11008, 8],
                a_contiguous: true,
                b_contiguous: true,
                a_offset: 0,
                scale: 1.0,
            },
            plan,
        ));
        assert!(!adapter_supported(
            AdapterLayout {
                a_dtype: DType::F16,
                b_dtype: DType::F16,
                a_dims: &[16, 4096],
                b_dims: &[11008, 16],
                a_contiguous: true,
                b_contiguous: true,
                a_offset: 0,
                scale: f64::MAX,
            },
            plan,
        ));
    }

    #[cfg(feature = "cuda")]
    fn cuda_tensor(
        data: &[f32],
        shape: &[usize],
        dtype: DType,
        device: &candle_core::Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::from_vec(data.to_vec(), shape.to_vec(), device)?.to_dtype(dtype)
    }

    #[cfg(feature = "cuda")]
    fn check_cuda_dtype(dtype: DType, device: &candle_core::Device) -> candle_core::Result<()> {
        use super::super::{
            reference::add_delta_reference, LoraLayerRegistry, LoraLinearSpec, LoraSiteKey,
            LoraWeights,
        };

        let registry = LoraLayerRegistry::new();
        let site = registry.register(
            LoraSiteKey::new("proj"),
            LoraLinearSpec::replicated(4, 3),
            dtype,
            device.clone(),
        )?;
        registry.finalize()?;
        for sequence_length in [1, 2] {
            let mut execution = LoraExecution::from_sequence_slots(
                registry.runtime_id(),
                &[Some(0), None, Some(1)],
                sequence_length,
            );
            execution.insert(
                &site,
                0,
                LoraWeights::new(
                    cuda_tensor(&[1., 0., 0., 0., 0., 1., 0., 0.], &[2, 4], dtype, device)?,
                    cuda_tensor(&[1., 0., 0., 1., 1., 1.], &[3, 2], dtype, device)?,
                    1.0,
                )?,
            )?;
            execution.insert(
                &site,
                1,
                LoraWeights::new(
                    cuda_tensor(&[1., 1., 0., 0.], &[1, 4], dtype, device)?,
                    cuda_tensor(&[2., 3., 4.], &[3, 1], dtype, device)?,
                    0.5,
                )?,
            )?;

            let rows = 3 * sequence_length;
            let input_data = (0..rows * 4)
                .map(|index| (index % 9 + 1) as f32)
                .collect::<Vec<_>>();
            let input = cuda_tensor(&input_data, &[3, sequence_length, 4], dtype, device)?;
            let base = cuda_tensor(
                &vec![0.5; rows * 3],
                &[3, sequence_length, 3],
                dtype,
                device,
            )?;
            let fast = try_add_delta_cuda(&execution, &site, &input, &base)?
                .expect("supported CUDA layout");
            let reference = add_delta_reference(&execution, &site, &input, base)?;
            let fast = fast.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let reference = reference
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let tolerance = if dtype == DType::BF16 { 0.08 } else { 0.02 };
            for (fast, reference) in fast.into_iter().zip(reference) {
                assert!(
                    (fast - reference).abs() <= tolerance,
                    "{fast} != {reference}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_prefill_and_decode_match_reference_for_mixed_slots() -> candle_core::Result<()> {
        let device = candle_core::Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            return Ok(());
        }
        check_cuda_dtype(DType::F16, &device)?;
        check_cuda_dtype(DType::BF16, &device)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_and_fallback_match_at_a_bf16_rounding_boundary() -> candle_core::Result<()> {
        use super::super::{LoraLayerRegistry, LoraLinearSpec, LoraSiteKey, LoraWeights};

        let device = candle_core::Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            return Ok(());
        }
        let registry = LoraLayerRegistry::new();
        let site = registry.register(
            LoraSiteKey::new("proj"),
            LoraLinearSpec::replicated(2, 1),
            DType::BF16,
            device.clone(),
        )?;
        registry.finalize()?;
        let input = cuda_tensor(&[1., 1.], &[1, 2], DType::BF16, &device)?;
        let base = cuda_tensor(&[0.], &[1, 1], DType::BF16, &device)?;
        let aligned_a = cuda_tensor(&[1., 1. / 256.], &[1, 2], DType::BF16, &device)?;
        let offset_a =
            cuda_tensor(&[0., 1., 1. / 256.], &[1, 3], DType::BF16, &device)?.narrow(1, 1, 2)?;
        assert!(offset_a.is_contiguous());
        assert_eq!(offset_a.layout().start_offset(), 1);
        let b = cuda_tensor(&[192.], &[1, 1], DType::BF16, &device)?;

        let mut aligned = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        aligned.insert(&site, 0, LoraWeights::new(aligned_a, b.clone(), 1.0)?)?;
        let mut offset = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        offset.insert(&site, 0, LoraWeights::new(offset_a, b, 1.0)?)?;

        let fast = try_add_delta_cuda(&aligned, &site, &input, &base)?
            .expect("aligned BF16 input uses the CUDA path");
        assert!(try_add_delta_cuda(&offset, &site, &input, &base)?.is_none());
        let fallback = super::super::reference::add_delta(&offset, &site, &input, base)?;
        let fast = fast.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let fallback = fallback.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(fast, fallback);
        assert_eq!(fast, vec![vec![192.]]);
        Ok(())
    }
}
