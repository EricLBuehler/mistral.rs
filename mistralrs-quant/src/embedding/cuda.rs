use std::ffi::c_void;

use candle_core::{
    cuda::cudarc::driver::{sys::CUstream, DevicePtrMut},
    CudaStorage, DType, Result, Shape, Storage, Tensor,
};

use super::{
    EmbeddingOutput, EmbeddingTable, RowCodec, RowScale, ScaleType, TableBytes, TableStorage, UNIT,
};

use crate::utils::CudaReadView;

const VECTOR_ALIGN: u64 = 16;
const ROW_SCALE_NONE: i32 = 0;
const ROW_SCALE_SCALAR: i32 = 1;
const ROW_SCALE_PER_ROW: i32 = 2;

// Mirrors `GatherArgs` in kernels/embedding/gather_rows.cu.
#[repr(C)]
struct GatherArgs {
    ids: *const c_void,
    data: *const u8,
    scales: *const u8,
    aux: *const u8,
    shards: *const u64,
    row_scales: *const c_void,
    out: *mut c_void,
    lookups: u64,
    rows: u64,
    columns: u64,
    data_stride: u64,
    scale_stride: u64,
    scale_row_div: u64,
    group: u64,
    out_scale: f32,
    ids_i64: i32,
    shard_count: i32,
    codec: i32,
    scale_type: i32,
    row_scale_mode: i32,
    row_scale_type: i32,
    bits: i32,
    out_dtype: i32,
}

unsafe extern "C" {
    fn mistralrs_gather_rows(args: *const GatherArgs, scalar: i32, stream: CUstream);
}

pub(super) fn gather(
    table: &EmbeddingTable<'_>,
    ids: &Tensor,
    out: EmbeddingOutput,
) -> Result<Tensor> {
    let out_dtype = match out.dtype {
        DType::BF16 => 0,
        DType::F16 => 1,
        DType::F32 => 2,
        other => {
            candle_core::bail!("embedding gather output must be BF16, F16 or F32, got {other:?}")
        }
    };
    table.codec.validate(table.columns)?;
    let device = ids.device().as_cuda_device()?.clone();
    let stream = device.cuda_stream();
    let ids = ids.flatten_all()?;
    let ids = match ids.dtype() {
        DType::U32 | DType::I64 => ids.contiguous()?,
        _ => ids.to_dtype(DType::U32)?.contiguous()?,
    };
    let lookups = ids.elem_count();
    let columns = table.columns;
    let codec = table.codec;
    let _mapped_use = match table.storage {
        TableStorage::Mapped(rows) => Some(rows.cuda_read(&stream)?),
        _ => None,
    };

    let (data_tensor, scales, aux, index, qtensor, shard_count) = match table.storage {
        TableStorage::Device { data, scales, aux } => {
            let (data, qtensor) = match data {
                TableBytes::Tensor(t) => (Some(t), None),
                TableBytes::QTensor(q) => (None, Some(q)),
            };
            (data, scales, aux, None, qtensor, 0)
        }
        TableStorage::Mapped(rows) => {
            let index = rows.index.as_ref().ok_or_else(|| {
                candle_core::Error::msg("mapped embedding rows have no device index")
            })?;
            (
                None,
                None,
                None,
                Some(index),
                None,
                rows.shards.len() as i32,
            )
        }
    };
    let data_view = CudaReadView::optional(data_tensor)?;
    let scale_view = CudaReadView::optional(scales)?;
    let aux_view = CudaReadView::optional(aux)?;
    let index_view = CudaReadView::optional(index)?;
    let ids_view = CudaReadView::new(&ids)?;
    let (data, _data_guard) = match qtensor {
        Some(q) => {
            let (pointer, guard) = q.device_ptr_with_guard(&stream)?;
            (pointer as u64, guard)
        }
        None => CudaReadView::read_optional(data_view.as_ref(), &stream)?,
    };
    let (scales, _scale_guard) = CudaReadView::read_optional(scale_view.as_ref(), &stream)?;
    let (aux, _aux_guard) = CudaReadView::read_optional(aux_view.as_ref(), &stream)?;
    let (shards, _index_guard) = CudaReadView::read_optional(index_view.as_ref(), &stream)?;
    let (ids_ptr, _ids_guard) = ids_view.read(&stream)?;

    let (row_scale_mode, row_scale) = match table.row_scale {
        RowScale::None | RowScale::Value(_) => (ROW_SCALE_NONE, None),
        RowScale::Scalar(t) => (ROW_SCALE_SCALAR, Some(t)),
        RowScale::PerRow(t) => (ROW_SCALE_PER_ROW, Some(t)),
    };
    let row_scale_type = row_scale.map_or(Ok(0), |t| {
        Ok::<_, candle_core::Error>(ScaleType::from_dtype(t.dtype())? as i32)
    })?;
    let row_scale_view = CudaReadView::optional(row_scale)?;
    let (row_scales, _row_scale_guard) =
        CudaReadView::read_optional(row_scale_view.as_ref(), &stream)?;

    let data_stride = codec.row_bytes(columns) as u64;
    let scalar = match codec {
        RowCodec::Dense(_) => {
            !columns.is_multiple_of(UNIT)
                || !(data | data_stride).is_multiple_of(VECTOR_ALIGN)
                || shard_count > 0
        }
        RowCodec::Fp8(grid) => {
            !columns.is_multiple_of(UNIT) || grid.is_some_and(|g| !g.group.is_multiple_of(UNIT))
        }
        _ => false,
    };
    let bits = match codec {
        RowCodec::Afq { bits, .. } | RowCodec::SymInt { bits, .. } => bits as i32,
        _ => 0,
    };
    let (scale_type, group, scale_stride, scale_row_div) = codec.grid().map_or((0, 1, 0, 1), |g| {
        (g.ty as i32, g.group, g.row_bytes(columns), g.rows_per_scale)
    });

    macro_rules! launch {
        ($ty:ty) => {{
            let mut output = unsafe { device.alloc::<$ty>(lookups * columns)? };
            let (out_ptr, out_guard) = output.device_ptr_mut(&stream);
            let args = GatherArgs {
                ids: ids_ptr as *const c_void,
                data: data as *const u8,
                scales: scales as *const u8,
                aux: aux as *const u8,
                shards: shards as *const u64,
                row_scales: row_scales as *const c_void,
                out: out_ptr as *mut c_void,
                lookups: lookups as u64,
                rows: table.rows as u64,
                columns: columns as u64,
                data_stride,
                scale_stride: scale_stride as u64,
                scale_row_div: scale_row_div as u64,
                group: group as u64,
                out_scale: table.host_multiplier(out),
                ids_i64: i32::from(ids.dtype() == DType::I64),
                shard_count,
                codec: codec.id(),
                scale_type,
                row_scale_mode,
                row_scale_type,
                bits,
                out_dtype,
            };
            unsafe { mistralrs_gather_rows(&args, i32::from(scalar), stream.cu_stream()) };
            drop(out_guard);
            let storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            Ok(Tensor::from((
                Storage::Cuda(storage),
                Shape::from((lookups, columns)),
            )))
        }};
    }
    match out.dtype {
        DType::BF16 => launch!(half::bf16),
        DType::F16 => launch!(half::f16),
        _ => launch!(f32),
    }
}
