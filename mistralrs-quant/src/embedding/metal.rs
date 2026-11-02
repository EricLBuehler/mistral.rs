use candle_core::{DType, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;

use super::{
    EmbeddingOutput, EmbeddingTable, RowCodec, RowScale, ScaleType, TableBytes, TableStorage,
};
use crate::metal_kernels::{
    call_embedding_gather, EmbeddingGather, EmbeddingGatherConfig, Kernels,
};

const ROW_SCALE_NONE: u32 = 0;
const ROW_SCALE_SCALAR: u32 = 1;
const ROW_SCALE_PER_ROW: u32 = 2;

fn buffer(tensor: &Tensor) -> Result<(Buffer, usize)> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        candle_core::bail!("embedding gather expects Metal tensors");
    };
    if !layout.is_contiguous() {
        candle_core::bail!("embedding gather expects contiguous tensors");
    }
    Ok((
        storage.buffer().clone(),
        layout.start_offset() * tensor.dtype().size_in_bytes(),
    ))
}

fn optional_buffer<'a>(
    value: &'a Option<(Buffer, usize)>,
    fallback: &'a (Buffer, usize),
) -> (&'a Buffer, usize) {
    let (buffer, offset) = value.as_ref().unwrap_or(fallback);
    (buffer, *offset)
}

pub(super) fn gather(
    table: &EmbeddingTable<'_>,
    ids: &Tensor,
    out: EmbeddingOutput,
) -> Result<Tensor> {
    table.codec.validate(table.columns)?;
    let ids = ids.flatten_all()?;
    let ids = match ids.dtype() {
        DType::U32 | DType::I64 => ids.contiguous()?,
        _ => ids.to_dtype(DType::U32)?.contiguous()?,
    };
    if ids.elem_count() == 0 {
        return Tensor::zeros((0, table.columns), out.dtype, ids.device());
    }
    let TableStorage::Device { data, scales, aux } = table.storage else {
        candle_core::bail!("Metal embedding gather requires device-backed rows");
    };
    let data = match data {
        TableBytes::Tensor(data) => data,
        TableBytes::QTensor(qtensor) => {
            // Candle already has quantized Metal get_rows, and keeps its Metal weight buffer private.
            let mut rows = qtensor.embedding(&ids)?;
            match table.row_scale {
                RowScale::Scalar(scale) => {
                    rows = rows.broadcast_mul(&scale.flatten_all()?.to_dtype(DType::F32)?)?;
                }
                RowScale::PerRow(scales) => {
                    let scales = scales
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .index_select(&ids.to_dtype(DType::U32)?, 0)?
                        .unsqueeze(1)?;
                    rows = rows.broadcast_mul(&scales)?;
                }
                RowScale::None | RowScale::Value(_) => {}
            }
            return super::scale_rows(rows, table.host_multiplier(out), out.dtype);
        }
    };
    let data_buffer = buffer(data)?;
    let scales_buffer = scales.map(buffer).transpose()?;
    let aux_buffer = aux.map(buffer).transpose()?;
    let (row_scale_mode, row_scales) = match table.row_scale {
        RowScale::None | RowScale::Value(_) => (ROW_SCALE_NONE, None),
        RowScale::Scalar(scale) => (ROW_SCALE_SCALAR, Some(scale)),
        RowScale::PerRow(scales) => (ROW_SCALE_PER_ROW, Some(scales)),
    };
    let row_scale_buffer = row_scales.map(buffer).transpose()?;
    let row_scale_type = row_scales.map_or(Ok(0), |scales| {
        Ok::<_, candle_core::Error>(ScaleType::from_dtype(scales.dtype())? as u32)
    })?;
    let ids_buffer = buffer(&ids)?;
    let device = ids.device().as_metal_device()?;
    let elements = ids.elem_count() * table.columns;
    let output = device.new_buffer(elements, out.dtype, "embedding-gather")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("embedding-gather");
    let (scale_type, group, scale_stride, scale_row_div) =
        table.codec.grid().map_or((0, 1, 0, 1), |grid| {
            (
                grid.ty as u32,
                grid.group,
                grid.row_bytes(table.columns),
                grid.rows_per_scale,
            )
        });
    let bits = match table.codec {
        RowCodec::Afq { bits, .. } | RowCodec::SymInt { bits, .. } => bits as u32,
        _ => 0,
    };
    call_embedding_gather(
        device.device(),
        &encoder,
        Kernels::global(),
        &EmbeddingGather {
            data: (&data_buffer.0, data_buffer.1),
            scales: optional_buffer(&scales_buffer, &data_buffer),
            aux: optional_buffer(&aux_buffer, &data_buffer),
            row_scales: optional_buffer(&row_scale_buffer, &data_buffer),
            ids: (&ids_buffer.0, ids_buffer.1),
            output: &output,
            codec: table.codec.id(),
            dtype: out.dtype,
            config: EmbeddingGatherConfig {
                lookups: ids.elem_count() as u64,
                rows: table.rows as u64,
                columns: table.columns as u64,
                data_stride: table.codec.row_bytes(table.columns) as u64,
                scale_stride: scale_stride as u64,
                scale_row_div: scale_row_div as u64,
                group: group as u64,
                out_scale: table.host_multiplier(out),
                scale_type,
                row_scale_mode,
                row_scale_type,
                bits,
                has_scales: u32::from(scales.is_some()),
                ids_i64: u32::from(ids.dtype() == DType::I64),
                padding: 0,
            },
        },
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            elements,
            out.dtype,
        )),
        (ids.elem_count(), table.columns),
    )))
}
