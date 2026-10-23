use candle_core::{Result, Tensor};

pub(crate) fn contiguous_fp8(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda()
        && matches!(
            x.dtype(),
            candle_core::DType::F8E4M3 | candle_core::DType::F8E8M0
        )
        && !x.is_contiguous()
    {
        return x.apply_op1_no_bwd(&CudaFp8Contiguous);
    }
    x.contiguous()
}

#[cfg(feature = "cuda")]
struct CudaFp8Contiguous;

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for CudaFp8Contiguous {
    fn name(&self) -> &'static str {
        "fp8-contiguous"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("FP8 byte-copy operation requires CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::{
            cudarc::driver::{LaunchConfig, PushKernelArg},
            kernels, CudaStorageSlice, SlicePtrOrNull, WrapErr,
        };

        let dev = &storage.device;
        let count = layout.shape().elem_count();
        let rank = layout.dims().len();
        let shape = layout.shape().clone();
        let launch_count = i32::try_from(count)? as u32;
        if count != 0 {
            let max_offset = layout
                .dims()
                .iter()
                .zip(layout.stride())
                .map(|(&dim, &stride)| (dim - 1) * stride)
                .sum::<usize>();
            // Candle's byte-copy kernel uses 32-bit element and source indices.
            u32::try_from(max_offset)?;
        }
        macro_rules! copy {
            ($source:expr, $dtype:ty, $variant:ident) => {{
                let source = $source;
                let mut output = unsafe { dev.alloc::<$dtype>(count)? };
                if count != 0 {
                    let params = SlicePtrOrNull::params_from_layout(dev, layout)?;
                    let function = dev.get_or_load_func("ucopy_u8", &kernels::UNARY)?;
                    // Both FP8 formats occupy one byte, so these views preserve every bit.
                    let source = unsafe { source.transmute::<u8>(source.len()).unwrap() }
                        .slice(layout.start_offset()..);
                    let mut destination = unsafe { output.transmute_mut::<u8>(count).unwrap() };
                    let mut builder = function.builder();
                    builder.arg(&count);
                    builder.arg(&rank);
                    params.builder_arg(&mut builder);
                    builder.arg(&source);
                    builder.arg(&mut destination);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(launch_count)) }.w()?;
                }
                CudaStorageSlice::$variant(output)
            }};
        }
        let slice = match &storage.slice {
            CudaStorageSlice::F8E4M3(source) => copy!(source, float8::F8E4M3, F8E4M3),
            CudaStorageSlice::F8E8M0(source) => copy!(source, u8, F8E8M0),
            _ => candle_core::bail!("FP8 byte-copy operation requires an FP8 tensor"),
        };
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            shape,
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::contiguous_fp8;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::{CudaStorage, DType, Device, Result, Shape, Storage, Tensor};
    use float8::F8E4M3;

    #[test]
    fn cuda_fp8_contiguous_preserves_strides_offsets_and_bits() -> Result<()> {
        const BATCH: usize = 3;
        const ROWS: usize = 5;
        const COLUMNS: usize = 257;
        const SELECTED_ROWS: usize = 3;
        const BYTE_VALUES: usize = 256;

        let device = Device::new_cuda(0)?;
        let dev = device.as_cuda_device()?;
        let bytes = (0..BATCH * ROWS * COLUMNS)
            .map(|index| (index % BYTE_VALUES) as u8)
            .collect::<Vec<_>>();
        for dtype in [DType::F8E4M3, DType::F8E8M0] {
            let slice = match dtype {
                DType::F8E4M3 => CudaStorageSlice::F8E4M3(
                    dev.clone_htod(
                        &bytes
                            .iter()
                            .copied()
                            .map(F8E4M3::from_bits)
                            .collect::<Vec<_>>(),
                    )?,
                ),
                _ => CudaStorageSlice::F8E8M0(dev.clone_htod(&bytes)?),
            };
            let source = Tensor::from((
                Storage::Cuda(CudaStorage {
                    slice,
                    device: dev.clone(),
                }),
                Shape::from_dims(&[BATCH, ROWS, COLUMNS]),
            ));
            let view = source.narrow(1, 1, SELECTED_ROWS)?.transpose(0, 2)?;
            assert!(!view.is_contiguous());
            assert_eq!(view.storage_and_layout().1.start_offset(), COLUMNS);
            let output = contiguous_fp8(&view)?;
            assert_eq!(output.shape(), view.shape());
            assert_eq!(output.dtype(), dtype);
            assert!(output.is_contiguous());
            let (storage, layout) = output.storage_and_layout();
            assert_eq!(layout.start_offset(), 0);
            let Storage::Cuda(storage) = &*storage else {
                panic!("unexpected non-CUDA FP8 output");
            };
            let actual = match &storage.slice {
                CudaStorageSlice::F8E4M3(values) => dev
                    .clone_dtoh(values)?
                    .iter()
                    .map(F8E4M3::to_bits)
                    .collect::<Vec<_>>(),
                CudaStorageSlice::F8E8M0(values) => dev.clone_dtoh(values)?,
                _ => panic!("unexpected FP8 output storage"),
            };
            let mut expected = Vec::with_capacity(view.elem_count());
            for column in 0..COLUMNS {
                for row in 1..=SELECTED_ROWS {
                    for batch in 0..BATCH {
                        expected.push(bytes[(batch * ROWS + row) * COLUMNS + column]);
                    }
                }
            }
            assert_eq!(actual, expected, "{dtype:?}");
        }
        Ok(())
    }
}
