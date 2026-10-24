use std::sync::Arc;

use candle_core::{DType, Result, Tensor};

const METAL_TOP1_MAX_EXACT_PACKED_VOCAB: usize = (1 << 24) + 1;
const METAL_TOP1_PACKED_WIDTH: usize = 2;

pub(crate) struct MetalTop1LogitsWorkspace {
    device: candle_core::Device,
    nblocks: usize,
    block_values: Arc<candle_metal_kernels::metal::Buffer>,
    block_indices: Arc<candle_metal_kernels::metal::Buffer>,
    packed: Tensor,
}

pub(crate) fn metal_top1_logits_cached(
    input: &Tensor,
    cache: &mut Option<MetalTop1LogitsWorkspace>,
) -> Result<[f32; METAL_TOP1_PACKED_WIDTH]> {
    use candle_core::{backend::BackendStorage, Device, MetalStorage, Shape, Storage};
    use mistralrs_quant::metal_kernels::{
        call_top1_logits_packed, Kernels, Top1LogitsBuffers, TOP1_LOGITS_CHUNK_SIZE,
    };

    let input = input.contiguous()?;
    let ncols = input.elem_count();
    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("metal_top1_logits_cached requires F32/F16/BF16 logits");
    }
    if ncols == 0 || ncols > METAL_TOP1_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!("metal_top1_logits_cached has unsupported vocab size {ncols}");
    }
    let nblocks = ncols.div_ceil(TOP1_LOGITS_CHUNK_SIZE);
    let (storage, layout) = input.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        candle_core::bail!("metal_top1_logits_cached requires Metal logits");
    };
    let device = storage.device();
    if cache.as_ref().is_none_or(|workspace| {
        workspace.nblocks != nblocks || !workspace.device.same_device(input.device())
    }) {
        let packed_buffer =
            device.new_buffer(METAL_TOP1_PACKED_WIDTH, DType::F32, "top1-packed")?;
        *cache = Some(MetalTop1LogitsWorkspace {
            device: Device::Metal(device.clone()),
            nblocks,
            block_values: device.new_buffer(nblocks, DType::F32, "top1-block-values")?,
            block_indices: device.new_buffer(nblocks, DType::U32, "top1-block-indices")?,
            packed: Tensor::from((
                Storage::Metal(MetalStorage::new(
                    packed_buffer,
                    device.clone(),
                    METAL_TOP1_PACKED_WIDTH,
                    DType::F32,
                )),
                Shape::from(vec![METAL_TOP1_PACKED_WIDTH]),
            )),
        });
    }
    let workspace = cache.as_ref().unwrap();
    {
        let (packed_storage, _) = workspace.packed.storage_and_layout();
        let Storage::Metal(packed_storage) = &*packed_storage else {
            unreachable!()
        };
        let encoder = device.command_encoder()?;
        encoder.set_label("top1-logits-packed");
        call_top1_logits_packed(
            device.device(),
            &encoder,
            Kernels::global(),
            input.dtype(),
            Top1LogitsBuffers {
                input: (
                    storage.buffer(),
                    layout.start_offset() * input.dtype().size_in_bytes(),
                ),
                block_values: &workspace.block_values,
                block_indices: &workspace.block_indices,
                packed_out: packed_storage.buffer(),
            },
            ncols,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    let packed = workspace.packed.to_vec1::<f32>()?;
    Ok([packed[0], packed[1]])
}

#[cfg(test)]
mod tests {
    use super::metal_top1_logits_cached;
    use candle_core::{DType, Device, Result, Tensor};
    use mistralrs_quant::metal_kernels::TOP1_LOGITS_CHUNK_SIZE;

    #[test]
    fn metal_top1_reduces_offset_rows_across_chunks() -> Result<()> {
        const VOCAB_SIZES: [usize; 2] = [33, TOP1_LOGITS_CHUNK_SIZE + 1];

        let device = Device::new_metal(0)?;
        let mut workspace = None;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for vocab in VOCAB_SIZES {
                let mut values = vec![f32::NEG_INFINITY; vocab + 1];
                values[0] = 90.0;
                values[vocab] = -2.0;
                for first_max in [vocab - 1, 1] {
                    values[first_max + 1] = -2.0;
                    let logits = Tensor::from_vec(values.clone(), vocab + 1, &device)?
                        .to_dtype(dtype)?
                        .narrow(0, 1, vocab)?;
                    assert_eq!(
                        metal_top1_logits_cached(&logits, &mut workspace)?,
                        [-2.0, first_max as f32],
                        "{dtype:?}, {vocab}"
                    );
                }
            }
        }
        Ok(())
    }
}
