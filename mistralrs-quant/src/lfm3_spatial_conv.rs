use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor, WithDType};
use rayon::prelude::*;

#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_core::{backend::BackendStorage, DType};

#[derive(Clone, Copy)]
struct SpatialConv {
    height: usize,
    width: usize,
    merge: usize,
    channels: usize,
    kernel: usize,
}

impl SpatialConv {
    fn cpu<T: WithDType>(&self, projected: &[T], weights: &[T], rows: usize) -> Vec<T> {
        let mut output = vec![T::from_f64(0.); rows * self.channels];
        output
            .par_chunks_mut(self.channels)
            .enumerate()
            .for_each_init(
                || vec![0f32; self.channels],
                |sums, (row, out)| {
                    sums.fill(0.);
                    let frame = row / (self.height * self.width);
                    let patch = row % (self.height * self.width);
                    let block = patch / self.merge.pow(2);
                    let y = block / (self.width / self.merge) * self.merge
                        + patch / self.merge % self.merge;
                    let x = block % (self.width / self.merge) * self.merge + patch % self.merge;
                    for ky in 0..self.kernel {
                        let iy = y as isize + ky as isize - (self.kernel / 2) as isize;
                        if !(0..self.height as isize).contains(&iy) {
                            continue;
                        }
                        for kx in 0..self.kernel {
                            let ix = x as isize + kx as isize - (self.kernel / 2) as isize;
                            if !(0..self.width as isize).contains(&ix) {
                                continue;
                            }
                            let (iy, ix) = (iy as usize, ix as usize);
                            let token = frame * self.height * self.width
                                + (iy / self.merge * (self.width / self.merge) + ix / self.merge)
                                    * self.merge.pow(2)
                                + iy % self.merge * self.merge
                                + ix % self.merge;
                            let base = token * 3 * self.channels;
                            for (channel, sum) in sums.iter_mut().enumerate() {
                                let gated = T::from_f64(
                                    (projected[base + channel].to_f64() as f32
                                        * projected[base + 2 * self.channels + channel].to_f64()
                                            as f32) as f64,
                                );
                                let weight = weights
                                    [channel * self.kernel.pow(2) + ky * self.kernel + kx]
                                    .to_f64() as f32;
                                *sum += gated.to_f64() as f32 * weight;
                            }
                        }
                    }
                    for (channel, value) in out.iter_mut().enumerate() {
                        let convolved = T::from_f64(sums[channel] as f64);
                        *value = T::from_f64(
                            (convolved.to_f64() as f32
                                * projected[row * 3 * self.channels + self.channels + channel]
                                    .to_f64() as f32) as f64,
                        );
                    }
                },
            );
        output
    }
}

impl CustomOp2 for SpatialConv {
    fn name(&self) -> &'static str {
        "lfm3-spatial-conv"
    }

    fn cpu_fwd(
        &self,
        projected: &CpuStorage,
        pl: &Layout,
        weights: &CpuStorage,
        wl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let rows = pl.dims()[0];
        let p = pl.start_offset();
        let w = wl.start_offset();
        let output = match (projected, weights) {
            (CpuStorage::F32(a), CpuStorage::F32(b)) => {
                CpuStorage::F32(self.cpu(&a[p..], &b[w..], rows))
            }
            (CpuStorage::F16(a), CpuStorage::F16(b)) => {
                CpuStorage::F16(self.cpu(&a[p..], &b[w..], rows))
            }
            (CpuStorage::BF16(a), CpuStorage::BF16(b)) => {
                CpuStorage::BF16(self.cpu(&a[p..], &b[w..], rows))
            }
            _ => candle_core::bail!(
                "LFM3 spatial convolution expects matching floating point dtypes"
            ),
        };
        Ok((output, Shape::from((rows, self.channels))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        projected: &candle_core::MetalStorage,
        pl: &Layout,
        weights: &candle_core::MetalStorage,
        wl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use objc2_metal::MTLSize;
        let device = projected.device();
        let dtype = projected.dtype();
        let suffix = match dtype {
            DType::F32 => "float",
            DType::F16 => "half",
            DType::BF16 => "bfloat16_t",
            _ => candle_core::bail!("Unsupported LFM3 spatial convolution dtype {dtype:?}"),
        };
        let pipeline = crate::metal_kernels::Kernels::global()
            .load_pipeline(device.device(), format!("lfm3_spatial_conv_{suffix}"))
            .map_err(candle_core::Error::wrap)?;
        let rows = pl.dims()[0];
        let elements = rows * self.channels;
        let output = device.new_buffer(elements, dtype, "lfm3-spatial-conv")?;
        let encoder = device.command_encoder()?;
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(projected.buffer()),
            pl.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(weights.buffer()),
            wl.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        for (index, value) in [
            elements,
            self.height,
            self.width,
            self.merge,
            self.channels,
            self.kernel,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_bytes(3 + index, &(*value as u32));
        }
        const THREADS: usize = 256;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: elements.div_ceil(THREADS),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elements, dtype),
            Shape::from((rows, self.channels)),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        projected: &candle_core::CudaStorage,
        pl: &Layout,
        weights: &candle_core::CudaStorage,
        wl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use std::ffi::c_void;
        unsafe extern "C" {
            fn lfm3_spatial_conv_f32(
                p: *const c_void,
                w: *const c_void,
                out: *mut c_void,
                elements: usize,
                height: u32,
                width: u32,
                merge: u32,
                channels: u32,
                kernel: u32,
                stream: *mut c_void,
            ) -> i32;
            fn lfm3_spatial_conv_f16(
                p: *const c_void,
                w: *const c_void,
                out: *mut c_void,
                elements: usize,
                height: u32,
                width: u32,
                merge: u32,
                channels: u32,
                kernel: u32,
                stream: *mut c_void,
            ) -> i32;
            fn lfm3_spatial_conv_bf16(
                p: *const c_void,
                w: *const c_void,
                out: *mut c_void,
                elements: usize,
                height: u32,
                width: u32,
                merge: u32,
                channels: u32,
                kernel: u32,
                stream: *mut c_void,
            ) -> i32;
        }
        let device = projected.device();
        let stream = device.cuda_stream();
        let rows = pl.dims()[0];
        let elements = rows * self.channels;
        macro_rules! launch {
            ($ty:ty, $kernel:ident) => {{
                let p = projected.as_cuda_slice::<$ty>()?.slice(pl.start_offset()..);
                let w = weights.as_cuda_slice::<$ty>()?.slice(wl.start_offset()..);
                let mut out = unsafe { device.alloc::<$ty>(elements)? };
                let (pp, pg) = p.device_ptr(&stream);
                let (wp, wg) = w.device_ptr(&stream);
                let (op, og) = out.device_ptr_mut(&stream);
                let status = unsafe {
                    $kernel(
                        pp as *const c_void,
                        wp as *const c_void,
                        op as *mut c_void,
                        elements,
                        self.height as u32,
                        self.width as u32,
                        self.merge as u32,
                        self.channels as u32,
                        self.kernel as u32,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                drop((pg, wg, og));
                if status != 0 {
                    candle_core::bail!("LFM3 spatial convolution CUDA launch failed: {status}");
                }
                candle_core::CudaStorage::wrap_cuda_slice(out, device.clone())
            }};
        }
        let output = match projected.dtype() {
            DType::F32 => launch!(f32, lfm3_spatial_conv_f32),
            DType::F16 => launch!(half::f16, lfm3_spatial_conv_f16),
            DType::BF16 => launch!(half::bf16, lfm3_spatial_conv_bf16),
            dtype => candle_core::bail!("Unsupported LFM3 spatial convolution dtype {dtype:?}"),
        };
        Ok((output, Shape::from((rows, self.channels))))
    }
}

pub fn gated_spatial_conv(
    projected: &Tensor,
    weights: &Tensor,
    height: usize,
    width: usize,
    merge: usize,
) -> Result<Tensor> {
    let (channels, _, kernel, _) = weights.dims4()?;
    let op = SpatialConv {
        height,
        width,
        merge,
        channels,
        kernel,
    };
    projected
        .contiguous()?
        .apply_op2_no_bwd(&weights.contiguous()?, &op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const BATCH: usize = 2;
    const HEIGHT: usize = 4;
    const WIDTH: usize = 6;
    const CHANNELS: usize = 5;
    const MERGE: usize = 2;
    const KERNEL: usize = 3;

    fn reference(projected: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let dtype = projected.dtype();
        let b = projected.narrow(1, 0, CHANNELS)?;
        let c = projected.narrow(1, CHANNELS, CHANNELS)?;
        let x = projected.narrow(1, CHANNELS * 2, CHANNELS)?;
        let raster = (b * x)?
            .reshape((BATCH, HEIGHT / MERGE, WIDTH / MERGE, MERGE, MERGE, CHANNELS))?
            .permute((0, 5, 1, 3, 2, 4))?
            .reshape((BATCH, CHANNELS, HEIGHT, WIDTH))?
            .contiguous()?
            .to_dtype(DType::F32)?;
        let output = raster.conv2d(&weights.to_dtype(DType::F32)?, KERNEL / 2, 1, 1, CHANNELS)?;
        let output = output
            .reshape((BATCH, CHANNELS, HEIGHT / MERGE, MERGE, WIDTH / MERGE, MERGE))?
            .permute((0, 2, 4, 3, 5, 1))?
            .reshape((BATCH * HEIGHT * WIDTH, CHANNELS))?
            .to_dtype(dtype)?;
        output * c
    }

    fn check(device: Device) -> Result<()> {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let values = (0..(BATCH + 1) * HEIGHT * WIDTH * 3 * CHANNELS)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 32.)
                .collect::<Vec<_>>();
            let weights = (0..CHANNELS * KERNEL * KERNEL)
                .map(|i| ((i * 5 % 17) as f32 - 8.) / 16.)
                .collect::<Vec<_>>();
            let projected = Tensor::from_vec(
                values,
                ((BATCH + 1) * HEIGHT * WIDTH, 3 * CHANNELS),
                &Device::Cpu,
            )?
            .to_dtype(dtype)?;
            let weights = Tensor::from_vec(weights, (CHANNELS, 1, KERNEL, KERNEL), &Device::Cpu)?
                .to_dtype(dtype)?;
            let expected = reference(
                &projected.narrow(0, HEIGHT * WIDTH, BATCH * HEIGHT * WIDTH)?,
                &weights,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let actual = gated_spatial_conv(
                &projected
                    .to_device(&device)?
                    .narrow(0, HEIGHT * WIDTH, BATCH * HEIGHT * WIDTH)?,
                &weights.to_device(&device)?,
                HEIGHT,
                WIDTH,
                MERGE,
            )?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let tolerance = match dtype {
                DType::F32 => 1e-6,
                DType::F16 => 5e-4,
                _ => 4e-3,
            };
            for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() < tolerance,
                    "{dtype:?} element {i}: {actual} != {expected}"
                );
            }
        }
        check_output_rounding(&device)
    }

    fn check_output_rounding(device: &Device) -> Result<()> {
        let mut projected = [[1f32, 0., 0.]; KERNEL * KERNEL];
        for (row, value) in projected
            .iter_mut()
            .zip([0.028533936, 0.016647339, 0.007133484, 1.])
        {
            row[2] = value;
        }
        projected[KERNEL * KERNEL / 2][1] = 0.37939453;
        let projected = Tensor::new(&projected, device)?.to_dtype(DType::F16)?;
        let weights = Tensor::new(
            &[
                -0.17651367f32,
                0.23535156,
                -0.35302734,
                0.3635254,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
            device,
        )?
        .reshape((1, 1, KERNEL, KERNEL))?
        .to_dtype(DType::F16)?;
        let actual = gated_spatial_conv(&projected, &weights, KERNEL, KERNEL, 1)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert_eq!(actual[KERNEL * KERNEL / 2][0], 0.13647461);
        let projected =
            Tensor::new(&[[0.20300293f32, 0.3190918, 0.20300293]], device)?.to_dtype(DType::F16)?;
        let weights = Tensor::new(&[[[[0.37304688f32]]]], device)?.to_dtype(DType::F16)?;
        let actual = gated_spatial_conv(&projected, &weights, 1, 1, 1)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert_eq!(actual[0][0], 0.004901886);
        Ok(())
    }

    #[test]
    fn cpu_spatial_conv_matches_grouped_convolution() -> Result<()> {
        check(Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires Metal"]
    fn metal_spatial_conv_matches_grouped_convolution() -> Result<()> {
        check(Device::new_metal(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn cuda_spatial_conv_matches_grouped_convolution() -> Result<()> {
        check(Device::new_cuda(0)?)
    }
}
