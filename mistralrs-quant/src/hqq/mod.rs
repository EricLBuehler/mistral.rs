use candle_core::{DType, Device, Result, Shape, Tensor};

#[cfg(feature = "cuda")]
use candle_core::{
    cuda::{cudarc::driver::DevicePtr, CudaStorageSlice},
    CudaStorage, Storage,
};

#[cfg(feature = "metal")]
use candle_core::Storage;

use candle_nn::Linear;
#[cfg(feature = "cuda")]
use half::{bf16, f16};
use safetensors::tensor::Dtype;
use std::{
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::{
    utils::{BitWiseOp, LeftshiftOp},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
    Shard, UnquantLinear, UqffReader, UqffTensor,
};

#[cfg(feature = "cuda")]
use crate::utils::get_cuda_device;

#[cfg(feature = "cuda")]
use ffi::{eight_bit, four_bit, one_bit, three_bit, two_bit};

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod bitpack_ffi;

#[cfg(not(feature = "cuda"))]
mod hqq_op;

mod optimize;
mod quantize;

pub(crate) const ISQ_HQQ_GROUP_SIZE: usize = 64;
pub(crate) const ISQ_HQQ_DEFAULT_OPT_STEPS: Option<usize> = Some(10);
pub(crate) const OPTIMIZER_HQQ_DEFAULT_STEPS: usize = 20;

#[cfg(feature = "cuda")]
macro_rules! dequant_for_dtype {
    ($this:expr, w=$wq_t:ty, sz=$scale_t:ty, $dtype:ident, pack=$pack:expr, $dev:expr, $bit_thing:ident, $postfix:tt) => {{
        paste::paste! {
            let (wq, _) = $this.w_q.storage_and_layout();
            let wq = match &*wq {
                candle_core::Storage::Cuda(s) => s,
                _ => candle_core::bail!("wq must be a cuda tensor"),
            };
            let (w_slice, _w_guard) = crate::utils::slice_ptr(wq.as_cuda_slice::<$wq_t>()?, $this.w_q.layout().start_offset());

            let (scale, _) = $this.scales.storage_and_layout();
            let scale = match &*scale {
                candle_core::Storage::Cuda(s) => s,
                _ => candle_core::bail!("scale must be a cuda tensor"),
            };
            let (scale_slice, _scale_guard) = crate::utils::slice_ptr(scale.as_cuda_slice::<$scale_t>()?, $this.scales.layout().start_offset());

            let (zero, _) = $this.zeros.storage_and_layout();
            let zero = match &*zero {
                candle_core::Storage::Cuda(s) => s,
                _ => candle_core::bail!("zero must be a cuda tensor"),
            };
            let (zero_slice, _zero_guard) = crate::utils::slice_ptr(zero.as_cuda_slice::<$scale_t>()?, $this.zeros.layout().start_offset());

            let (h, w) = $this.w_q.dims2()?;
            let num_packed_elems = $pack;
            let out_shape = Shape::from_dims(&[num_packed_elems * h, w]);

            let out = unsafe { $dev.alloc::<$scale_t>(out_shape.elem_count())? };
            let (out_ptr, out_guard) = out.device_ptr(out.stream());
            unsafe {
                $bit_thing::[< dequantize_ $postfix >](
                    w_slice as *const $wq_t,
                    scale_slice as *const $scale_t,
                    zero_slice as *const $scale_t,
                    out_ptr as *mut $scale_t,
                    h as i32,
                    w as i32,
                );
            }
            drop(out_guard);

            let storage = CudaStorage {
                slice: CudaStorageSlice::$dtype(out),
                device: $dev.clone(),
            };
            let storage = Storage::Cuda(storage);

            Tensor::from((storage, out_shape))
        }
    }};
}

#[derive(Debug, Clone, Copy)]
pub enum HqqAxis {
    Zero = 0,
    One = 1,
}

impl TryFrom<usize> for HqqAxis {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            other => candle_core::bail!("Unexpected value for HQQ axis {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum HqqBits {
    Eight = 8,
    Four = 4,
    Three = 3,
    Two = 2,
    One = 1,
}

impl TryFrom<usize> for HqqBits {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::Eight),
            4 => Ok(Self::Four),
            3 => Ok(Self::Three),
            2 => Ok(Self::Two),
            1 => Ok(Self::One),
            other => candle_core::bail!("Unexpected value for HQQ bits {other}"),
        }
    }
}

impl HqqBits {
    // https://github.com/mobiusml/hqq/blob/306e30d9400629523c8e0af70101d8d7073cb3d5/hqq/core/bitpack.py#L10
    pub(crate) fn bitpack_type(&self) -> impl Fn(Tensor) -> Result<Tensor> {
        match self {
            Self::Eight => |wq: Tensor| -> Result<Tensor> {
                #[allow(unused_variables)]
                let device = wq.device();

                #[cfg(feature = "cuda")]
                if device.is_cuda() {
                    // Use CUDA kernel for 8-bit (which is essentially a copy)
                    let dev = get_cuda_device(&wq)?;
                    let wq = wq.to_dtype(DType::U8)?;
                    let (wq_storage, _) = wq.storage_and_layout();
                    let wq_storage = match &*wq_storage {
                        Storage::Cuda(s) => s,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };

                    let output_shape = wq.shape().clone();
                    let output = unsafe { dev.alloc::<u8>(output_shape.elem_count())? };

                    unsafe {
                        let (output_ptr, output_guard) = output.device_ptr(output.stream());
                        let (input_ptr, _input_guard) = crate::utils::slice_ptr(
                            wq_storage.as_cuda_slice::<u8>()?,
                            wq.layout().start_offset(),
                        );

                        bitpack_ffi::launch_pack_8bit_kernel(
                            input_ptr as *const u8,
                            output_ptr as *mut u8,
                            output_shape.elem_count(),
                            dev.cuda_stream().cu_stream(),
                        );
                        drop(output_guard);
                    }

                    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
                    let storage = Storage::Cuda(storage);
                    return Ok(Tensor::from((storage, output_shape)));
                }

                #[cfg(feature = "metal")]
                if device.is_metal() {
                    use candle_core::MetalStorage;

                    let dev = device.as_metal_device()?;
                    let encoder = dev.command_encoder()?;
                    encoder.set_label("hqq_pack_8bit");

                    let (wq_storage, _wq_layout) = wq.storage_and_layout();
                    let wq_storage = match &*wq_storage {
                        Storage::Metal(s) => s,
                        _ => candle_core::bail!("Expected Metal storage"),
                    };

                    let output_shape = wq.shape().clone();
                    let output = dev.new_buffer(
                        output_shape.elem_count(),
                        DType::U8,
                        "hqq_pack_8bit_output",
                    )?;

                    crate::metal_kernels::call_hqq_pack_8bit(
                        dev.device(),
                        &encoder,
                        &crate::metal_kernels::Kernels::new(),
                        wq_storage.buffer(),
                        &output,
                        output_shape.elem_count(),
                    )
                    .map_err(candle_core::Error::wrap)?;

                    let storage = MetalStorage::new(
                        output,
                        dev.clone(),
                        output_shape.elem_count(),
                        DType::U8,
                    );
                    let storage = Storage::Metal(storage);

                    return Ok(Tensor::from((storage, output_shape)));
                }

                wq.to_dtype(DType::U8)
            },
            Self::Four => |wq_in: Tensor| -> Result<Tensor> {
                #[allow(unused_variables)]
                let device = wq_in.device();

                #[cfg(feature = "cuda")]
                if device.is_cuda() {
                    // Use CUDA kernel for 4-bit packing
                    let dev = get_cuda_device(&wq_in)?;
                    let wq = wq_in.to_dtype(DType::U8)?;
                    let (wq_storage, _) = wq.storage_and_layout();
                    let wq_storage = match &*wq_storage {
                        Storage::Cuda(s) => s,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };

                    let output_height = wq.dims()[0] / 2;
                    let output_shape = Shape::from_dims(&[output_height, wq.dims()[1]]);
                    let output = unsafe { dev.alloc::<u8>(output_shape.elem_count())? };

                    unsafe {
                        let (output_ptr, output_guard) = output.device_ptr(output.stream());
                        let (input_ptr, _input_guard) = crate::utils::slice_ptr(
                            wq_storage.as_cuda_slice::<u8>()?,
                            wq.layout().start_offset(),
                        );

                        bitpack_ffi::launch_pack_4bit_kernel(
                            input_ptr as *const u8,
                            output_ptr as *mut u8,
                            wq.dims()[0],
                            wq.dims()[1],
                            dev.cuda_stream().cu_stream(),
                        );
                        drop(output_guard);
                    }

                    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
                    let storage = Storage::Cuda(storage);
                    return Ok(Tensor::from((storage, output_shape)));
                }

                #[cfg(feature = "metal")]
                if device.is_metal() {
                    use candle_core::MetalStorage;

                    let dev = device.as_metal_device()?;
                    let encoder = dev.command_encoder()?;
                    encoder.set_label("hqq_pack_4bit");

                    let wq = wq_in.to_dtype(DType::U8)?;
                    let (wq_storage, _wq_layout) = wq.storage_and_layout();
                    let wq_storage = match &*wq_storage {
                        Storage::Metal(s) => s,
                        _ => candle_core::bail!("Expected Metal storage"),
                    };

                    let output_height = wq.dims()[0] / 2;
                    let output_shape = Shape::from_dims(&[output_height, wq.dims()[1]]);
                    let output = dev.new_buffer(
                        output_shape.elem_count(),
                        DType::U8,
                        "hqq_pack_4bit_output",
                    )?;

                    crate::metal_kernels::call_hqq_pack_4bit(
                        dev.device(),
                        &encoder,
                        &crate::metal_kernels::Kernels::new(),
                        wq_storage.buffer(),
                        &output,
                        wq.dims()[0],
                        wq.dims()[1],
                    )
                    .map_err(candle_core::Error::wrap)?;

                    let storage = MetalStorage::new(
                        output,
                        dev.clone(),
                        output_shape.elem_count(),
                        DType::U8,
                    );
                    let storage = Storage::Metal(storage);

                    return Ok(Tensor::from((storage, output_shape)));
                }

                // CPU fallback
                let wq = wq_in.to_dtype(DType::U8)?;
                let step = (wq.dims()[0] as f64 / 2.) as usize;

                let a = wq.narrow(0, 0, step)?;
                let b = wq.narrow(0, step, step)?;
                a.leftshift(4)?.bitwise_or(&b)
            },
            Self::Two => |wq_in: Tensor| -> Result<Tensor> {
                #[allow(unused_variables)]
                let device = wq_in.device();

                #[cfg(feature = "cuda")]
                if device.is_cuda() {
                    // Use CUDA kernel for 2-bit packing
                    let dev = get_cuda_device(&wq_in)?;
                    let wq = wq_in.to_dtype(DType::U8)?;
                    let (wq_storage, _) = wq.storage_and_layout();
                    let wq_storage = match &*wq_storage {
                        Storage::Cuda(s) => s,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };

                    let output_height = wq.dims()[0] / 4;
                    let output_shape = Shape::from_dims(&[output_height, wq.dims()[1]]);
                    let output = unsafe { dev.alloc::<u8>(output_shape.elem_count())? };

                    unsafe {
                        let (output_ptr, output_guard) = output.device_ptr(output.stream());
                        let (input_ptr, _input_guard) = crate::utils::slice_ptr(
                            wq_storage.as_cuda_slice::<u8>()?,
                            wq.layout().start_offset(),
                        );

                        bitpack_ffi::launch_pack_2bit_kernel(
                            input_ptr as *const u8,
                            output_ptr as *mut u8,
                            wq.dims()[0],
                            wq.dims()[1],
                            dev.cuda_stream().cu_stream(),
                        );
                        drop(output_guard);
                    }

                    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
                    let storage = Storage::Cuda(storage);
                    Ok(Tensor::from((storage, output_shape)))
                } else {
                    // CPU fallback
                    let wq = wq_in.to_dtype(DType::U8)?;
                    let step = (wq.dims()[0] as f64 / 4.) as usize;

                    let a = wq.narrow(0, 0, step)?;
                    let b = wq.narrow(0, step, step)?;
                    let c = wq.narrow(0, step * 2, step)?;
                    let d = wq.narrow(0, step * 3, step)?;

                    a.leftshift(6)?
                        .bitwise_or(&b.leftshift(4)?)?
                        .bitwise_or(&c.leftshift(2)?)?
                        .bitwise_or(&d)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let wq = wq_in.to_dtype(DType::U8)?;
                    let step = (wq.dims()[0] as f64 / 4.) as usize;

                    let a = wq.narrow(0, 0, step)?;
                    let b = wq.narrow(0, step, step)?;
                    let c = wq.narrow(0, step * 2, step)?;
                    let d = wq.narrow(0, step * 3, step)?;

                    a.leftshift(6)?
                        .bitwise_or(&b.leftshift(4)?)?
                        .bitwise_or(&c.leftshift(2)?)?
                        .bitwise_or(&d)
                }
            },
            Self::Three => |wq_in: Tensor| -> Result<Tensor> {
                let device = wq_in.device();

                // Pad input to multiple of 10
                let padded_height = (10. * (wq_in.dims()[0] as f64 / 10.).ceil()) as usize;
                let wq = Tensor::zeros((padded_height, wq_in.dims()[1]), DType::U32, device)?;
                let wq = wq.slice_assign(
                    &[0..wq_in.dims()[0], 0..wq.dims()[1]],
                    &wq_in.to_dtype(DType::U32)?,
                )?;

                #[cfg(feature = "cuda")]
                if device.is_cuda() {
                    // Use CUDA kernel for efficient 3-bit packing
                    let dev = get_cuda_device(&wq)?;
                    let (wq_storage, _) = wq.storage_and_layout();
                    let wq_storage = match &*wq_storage {
                        Storage::Cuda(s) => s,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };

                    let output_height = padded_height / 10;
                    let output_shape = Shape::from_dims(&[output_height, wq_in.dims()[1]]);
                    let output = unsafe { dev.alloc::<i32>(output_shape.elem_count())? };

                    unsafe {
                        let (output_ptr, output_guard) = output.device_ptr(output.stream());
                        let (input_ptr, _input_guard) = crate::utils::slice_ptr(
                            wq_storage.as_cuda_slice::<u32>()?,
                            wq.layout().start_offset(),
                        );

                        bitpack_ffi::launch_pack_3bit_kernel(
                            input_ptr as *const u32,
                            output_ptr as *mut i32,
                            padded_height,
                            wq_in.dims()[1],
                            dev.cuda_stream().cu_stream(),
                        );
                        drop(output_guard);
                    }

                    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
                    let storage = Storage::Cuda(storage);
                    return Ok(Tensor::from((storage, output_shape)));
                }

                // CPU fallback implementation
                let wq = if wq.device().is_metal() {
                    // Metal doesn't support direct U32 to I32 conversion, use CPU as intermediate
                    let cpu_wq = wq.to_device(&Device::Cpu)?;
                    cpu_wq.to_dtype(DType::I32)?.to_device(wq.device())?
                } else {
                    wq.to_dtype(DType::I32)?
                };
                let step = (wq.dims()[0] as f64 / 10.) as usize;

                let a = wq.narrow(0, 0, step)?;
                let b = wq.narrow(0, step, step)?;
                let c = wq.narrow(0, step * 2, step)?;
                let d = wq.narrow(0, step * 3, step)?;
                let e = wq.narrow(0, step * 4, step)?;
                let f = wq.narrow(0, step * 5, step)?;
                let g = wq.narrow(0, step * 6, step)?;
                let h = wq.narrow(0, step * 7, step)?;
                let i = wq.narrow(0, step * 8, step)?;
                let j = wq.narrow(0, step * 9, step)?;

                a.leftshift(27)?
                    .bitwise_or(&b.leftshift(24)?)?
                    .bitwise_or(&c.leftshift(21)?)?
                    .bitwise_or(&d.leftshift(18)?)?
                    .bitwise_or(&e.leftshift(15)?)?
                    .bitwise_or(&f.leftshift(12)?)?
                    .bitwise_or(&g.leftshift(9)?)?
                    .bitwise_or(&h.leftshift(6)?)?
                    .bitwise_or(&i.leftshift(3)?)?
                    .bitwise_or(&j)
            },
            Self::One => |wq_in: Tensor| -> Result<Tensor> {
                #[allow(unused_variables)]
                let device = wq_in.device();

                #[cfg(feature = "cuda")]
                if device.is_cuda() {
                    // Use CUDA kernel for 1-bit packing
                    let dev = get_cuda_device(&wq_in)?;
                    let wq = wq_in.to_dtype(DType::U8)?;
                    let (wq_storage, _) = wq.storage_and_layout();
                    let wq_storage = match &*wq_storage {
                        Storage::Cuda(s) => s,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };

                    let output_height = wq.dims()[0] / 8;
                    let output_shape = Shape::from_dims(&[output_height, wq.dims()[1]]);
                    let output = unsafe { dev.alloc::<u8>(output_shape.elem_count())? };

                    unsafe {
                        let (output_ptr, output_guard) = output.device_ptr(output.stream());
                        let (input_ptr, _input_guard) = crate::utils::slice_ptr(
                            wq_storage.as_cuda_slice::<u8>()?,
                            wq.layout().start_offset(),
                        );

                        bitpack_ffi::launch_pack_1bit_kernel(
                            input_ptr as *const u8,
                            output_ptr as *mut u8,
                            wq.dims()[0],
                            wq.dims()[1],
                            dev.cuda_stream().cu_stream(),
                        );
                        drop(output_guard);
                    }

                    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
                    let storage = Storage::Cuda(storage);
                    Ok(Tensor::from((storage, output_shape)))
                } else {
                    // CPU fallback
                    let wq = wq_in.to_dtype(DType::U8)?;
                    let step = (wq.dims()[0] as f64 / 8.) as usize;

                    let a = wq.narrow(0, 0, step)?;
                    let b = wq.narrow(0, step, step)?;
                    let c = wq.narrow(0, step * 2, step)?;
                    let d = wq.narrow(0, step * 3, step)?;
                    let e = wq.narrow(0, step * 4, step)?;
                    let f = wq.narrow(0, step * 5, step)?;
                    let g = wq.narrow(0, step * 6, step)?;
                    let h = wq.narrow(0, step * 7, step)?;

                    a.leftshift(7)?
                        .bitwise_or(&b.leftshift(6)?)?
                        .bitwise_or(&c.leftshift(5)?)?
                        .bitwise_or(&d.leftshift(4)?)?
                        .bitwise_or(&e.leftshift(3)?)?
                        .bitwise_or(&f.leftshift(2)?)?
                        .bitwise_or(&g.leftshift(1)?)?
                        .bitwise_or(&h)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let wq = wq_in.to_dtype(DType::U8)?;
                    let step = (wq.dims()[0] as f64 / 8.) as usize;

                    let a = wq.narrow(0, 0, step)?;
                    let b = wq.narrow(0, step, step)?;
                    let c = wq.narrow(0, step * 2, step)?;
                    let d = wq.narrow(0, step * 3, step)?;
                    let e = wq.narrow(0, step * 4, step)?;
                    let f = wq.narrow(0, step * 5, step)?;
                    let g = wq.narrow(0, step * 6, step)?;
                    let h = wq.narrow(0, step * 7, step)?;

                    a.leftshift(7)?
                        .bitwise_or(&b.leftshift(6)?)?
                        .bitwise_or(&c.leftshift(5)?)?
                        .bitwise_or(&d.leftshift(4)?)?
                        .bitwise_or(&e.leftshift(3)?)?
                        .bitwise_or(&f.leftshift(2)?)?
                        .bitwise_or(&g.leftshift(1)?)?
                        .bitwise_or(&h)
                }
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HqqConfig {
    pub bits: HqqBits,
    pub group_size: NonZeroUsize,
    pub axis: HqqAxis,
    pub optimization_steps: Option<usize>,
    pub round_zeros: bool,  // default false
    pub channel_wise: bool, // default true
}

#[derive(Debug)]
pub struct HqqLayer {
    pub(crate) w_q: Tensor,
    pub(crate) zeros: Tensor,
    pub(crate) scales: Tensor,
    pub(crate) bias: Option<Tensor>,
    pub(crate) w_shape: Shape,
    pub(crate) cfg: HqqConfig,
}

impl HqqLayer {
    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] = &[
            "weight",
            "weight.format",
            "weight.scales",
            "weight.zeros",
            "weight.shape",
            "weight.bits",
            "weight.group_size",
            "weight.axis",
            "weight.optimization_steps",
            "weight.round_zeros",
            "weight.channel_wise",
        ];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES)
            && layer.scalar("weight.format", Dtype::U8)
            && layer.scalar("weight.bits", Dtype::U8)
            && layer.scalar("weight.group_size", Dtype::U32)
            && layer.scalar("weight.axis", Dtype::U8)
            && layer.scalar("weight.optimization_steps", Dtype::U32)
            && layer.scalar("weight.round_zeros", Dtype::U8)
            && layer.scalar("weight.channel_wise", Dtype::U8)
            && layer.u32_vector("weight.shape")
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::Hqq,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        tensors: &[UqffTensor],
        prefix: &str,
    ) -> Result<String> {
        let bits = crate::uqff::u8_scalar_with_suffix(tensors, prefix, "weight.bits")?;
        Ok(format!("hqq{bits}"))
    }

    pub fn from_parts(
        w_q: Tensor,
        scales: Tensor,
        zeros: Tensor,
        bias: Option<Tensor>,
        w_shape: Shape,
        cfg: HqqConfig,
    ) -> Self {
        Self {
            w_q,
            zeros,
            scales,
            bias,
            w_shape,
            cfg,
        }
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        if !matches!(shard, Shard::Simple { world_size: 1, .. }) {
            candle_core::bail!("HQQ UQFF artifacts do not support sharded loading.");
        }
        let w_q = reader.load_tensor(&format!("{key}.weight"), device)?;
        let scales = reader.load_tensor(&format!("{key}.weight.scales"), device)?;
        let zeros = reader.load_tensor(&format!("{key}.weight.zeros"), device)?;
        let bias = reader.load_optional_tensor(&format!("{key}.bias"), device)?;
        let w_shape = Shape::from_dims(&reader.load_u32_vec(&format!("{key}.weight.shape"))?);
        let optimization_steps =
            match reader.load_u32_scalar(&format!("{key}.weight.optimization_steps"))? as usize {
                0 => None,
                steps => Some(steps),
            };
        let cfg = HqqConfig {
            bits: HqqBits::try_from(reader.load_u8_scalar(&format!("{key}.weight.bits"))? as usize)?,
            group_size: NonZeroUsize::try_from(
                reader.load_u32_scalar(&format!("{key}.weight.group_size"))? as usize,
            )?,
            axis: HqqAxis::try_from(reader.load_u8_scalar(&format!("{key}.weight.axis"))? as usize)?,
            optimization_steps,
            round_zeros: reader.load_u8_scalar(&format!("{key}.weight.round_zeros"))? != 0,
            channel_wise: reader.load_u8_scalar(&format!("{key}.weight.channel_wise"))? != 0,
        };
        Ok(Self::from_parts(w_q, scales, zeros, bias, w_shape, cfg))
    }

    /// Dequantize `self` into a tensor of shape `scales` or `zeros`.
    #[cfg(not(feature = "cuda"))]
    fn dequantize(&self) -> Result<Tensor> {
        use crate::hqq::hqq_op::{Dequant1Bit, Dequant2Bit, Dequant3Bit, Dequant4Bit, Dequant8Bit};

        match (self.scales.dtype(), self.zeros.dtype()) {
            (DType::F16, DType::F16) | (DType::BF16, DType::BF16) | (DType::F32, DType::F32) => (),
            (a, b) => {
                candle_core::bail!("Expected all dtypes to be the same, got ({a:?}, {b:?}).")
            }
        }
        if !(self.w_q.is_contiguous() && self.scales.is_contiguous() && self.zeros.is_contiguous())
        {
            candle_core::bail!("All tensors must be contiguous!");
        }
        if self.cfg.axis as usize != 0 {
            candle_core::bail!(
                "CPU HQQ dequantization requires axis == 0, got {}.",
                self.cfg.axis as usize
            );
        }
        let (h, w) = self.w_q.dims2()?;

        match self.cfg.bits as usize {
            8 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant8Bit { h, w })?
                .reshape(&self.w_shape),
            4 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant4Bit { h, w })?
                .reshape(&self.w_shape),
            3 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant3Bit { h, w })?
                .reshape(&self.w_shape),
            2 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant2Bit { h, w })?
                .reshape(&self.w_shape),
            1 => self
                .w_q
                .apply_op3_no_bwd(&self.scales, &self.zeros, &Dequant1Bit { h, w })?
                .reshape(&self.w_shape),
            b => candle_core::bail!("Unreachable bits {b}"),
        }
    }

    /// Dequantize `self` into a tensor of shape `scales` or `zeros`.
    #[cfg(feature = "cuda")]
    fn dequantize(&self) -> Result<Tensor> {
        match (self.scales.dtype(), self.zeros.dtype()) {
            (DType::F16, DType::F16) | (DType::BF16, DType::BF16) | (DType::F32, DType::F32) => (),
            (a, b) => {
                candle_core::bail!("Expected all dtypes to be the same, got ({a:?}, {b:?}).")
            }
        }
        if !(self.w_q.is_contiguous() && self.scales.is_contiguous() && self.zeros.is_contiguous())
        {
            candle_core::bail!("All tensors must be contiguous!");
        }
        if self.cfg.axis as usize != 0 {
            candle_core::bail!(
                "CUDA HQQ dequantization requires axis == 0, got {}.",
                self.cfg.axis as usize
            );
        }
        let dev = get_cuda_device(&self.w_q)?;

        let inner = match (self.cfg.bits as usize, self.scales.dtype()) {
            // 8 bits
            (8, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 1,
                    dev,
                    eight_bit,
                    8bit_u8_kernel_f32
                )
            }
            (8, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 1,
                    dev,
                    eight_bit,
                    8bit_u8_kernel_f16
                )
            }
            (8, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 1,
                    dev,
                    eight_bit,
                    8bit_u8_kernel_bf16
                )
            }

            // 4 bits
            (4, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 2,
                    dev,
                    four_bit,
                    4bit_u8_kernel_f32
                )
            }
            (4, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 2,
                    dev,
                    four_bit,
                    4bit_u8_kernel_f16
                )
            }
            (4, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 2,
                    dev,
                    four_bit,
                    4bit_u8_kernel_bf16
                )
            }

            // 3 bits
            // https://github.com/mobiusml/hqq/blob/306e30d9400629523c8e0af70101d8d7073cb3d5/hqq/kernels/hqq_aten_cuda.cpp#L42-L45
            (3, DType::F32) => {
                let res = dequant_for_dtype!(
                    self,
                    w = i32,
                    sz = f32,
                    F32,
                    pack = 10,
                    dev,
                    three_bit,
                    3bit_32_kernel_f32
                );
                res.narrow(self.cfg.axis as usize, 0, self.cfg.group_size.into())?
            }
            (3, DType::F16) => {
                let res = dequant_for_dtype!(
                    self,
                    w = i32,
                    sz = f16,
                    F16,
                    pack = 10,
                    dev,
                    three_bit,
                    3bit_32_kernel_f16
                );
                res.narrow(self.cfg.axis as usize, 0, self.cfg.group_size.into())?
            }
            (3, DType::BF16) => {
                let res = dequant_for_dtype!(
                    self,
                    w = i32,
                    sz = bf16,
                    BF16,
                    pack = 10,
                    dev,
                    three_bit,
                    3bit_32_kernel_bf16
                );
                res.narrow(self.cfg.axis as usize, 0, self.cfg.group_size.into())?
            }

            // 2 bits
            (2, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 4,
                    dev,
                    two_bit,
                    2bit_u8_kernel_f32
                )
            }
            (2, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 4,
                    dev,
                    two_bit,
                    2bit_u8_kernel_f16
                )
            }
            (2, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 4,
                    dev,
                    two_bit,
                    2bit_u8_kernel_bf16
                )
            }

            // 1 bit
            (1, DType::F32) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f32,
                    F32,
                    pack = 8,
                    dev,
                    one_bit,
                    1bit_u8_kernel_f32
                )
            }
            (1, DType::F16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = f16,
                    F16,
                    pack = 8,
                    dev,
                    one_bit,
                    1bit_u8_kernel_f16
                )
            }
            (1, DType::BF16) => {
                dequant_for_dtype!(
                    self,
                    w = u8,
                    sz = bf16,
                    BF16,
                    pack = 8,
                    dev,
                    one_bit,
                    1bit_u8_kernel_bf16
                )
            }
            (bits, dtype) => candle_core::bail!("Unsupported bit width {bits} and dtype {dtype:?}"),
        };
        inner.reshape(&self.w_shape)
    }

    fn dequantize_matmul(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize()?;
        // Dispatch to unquant. This uses some cublaslt for bias & on cuda always, so it is better
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            w,
            self.bias.clone(),
        )))?;
        unquant.forward(xs)
    }

    pub fn with_bias(mut self, bias: Tensor) -> Self {
        self.bias = Some(bias);
        self
    }
}

impl QuantMethod for HqqLayer {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => {
                unreachable!()
            }
            QuantMethodConfig::Hqq {
                tensor,
                bits,
                group_size,
                axis,
                optimization_steps,
                round_zeros,
                channel_wise,
                bias,
            } => {
                let cfg = HqqConfig {
                    bits,
                    group_size,
                    axis,
                    optimization_steps,
                    round_zeros: round_zeros.unwrap_or(false),
                    channel_wise: channel_wise.unwrap_or(true),
                };

                let this = Self::quantize(&tensor, tensor.device(), cfg)?;
                if let Some(bias) = bias {
                    Ok(this.with_bias(bias))
                } else {
                    Ok(this)
                }
            }
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.dequantize()
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        /*
        if self.cfg.force_dequantize {
            self.dequantize_matmul(a)
        } else {
            todo!()
        } */
        self.dequantize_matmul(a)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(self.scales.dtype())
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("HQQ quantization does not support adding weight delta.")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.scales.dtype(), self.scales.device().clone())
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.scales.dtype(),
            self.scales.device().clone(),
            self.w_shape.dims().to_vec(),
            request,
            true,
        ))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let _acquired_quantize_guard = guard.acquire(&device);
        if imatrix_weight.is_some() {
            // TODO just warn?
            candle_core::bail!("HQQ does not support imatrix.");
        }

        n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bits = match dtype {
            Some(IsqType::HQQ8) => HqqBits::Eight,
            Some(IsqType::HQQ4) => HqqBits::Four,
            // Some(IsqType::HQQ3) => HqqBits::Three,
            // Some(IsqType::HQQ2) => HqqBits::Two,
            // Some(IsqType::HQQ1) => HqqBits::One,
            _ => candle_core::bail!("Expected a HQQ ISQ type."),
        };
        let cfg = HqqConfig {
            bits,
            group_size: ISQ_HQQ_GROUP_SIZE.try_into()?,
            axis: HqqAxis::Zero,
            optimization_steps: ISQ_HQQ_DEFAULT_OPT_STEPS,
            round_zeros: false,
            channel_wise: true,
        };
        let dequant = self.dequantize()?;
        let res = Self::quantize(&dequant, &device, cfg)?;
        if let Some(ref bias) = self.bias {
            let bias = bias
                .to_device(&device)?
                .to_dtype(res.dtype_and_device().0)?;
            Ok(Arc::new(res.with_bias(bias)))
        } else {
            Ok(Arc::new(res))
        }
    }
}

impl QuantizedSerde for HqqLayer {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "hqq"
    }
    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        let actual_ty = match self.cfg.bits {
            HqqBits::Eight => IsqType::HQQ8,
            HqqBits::Four => IsqType::HQQ4,
            HqqBits::One | HqqBits::Two | HqqBits::Three => {
                candle_core::bail!("Cannot serialize unsupported HQQ bit width as UQFF.")
            }
        };
        if ty != actual_ty {
            candle_core::bail!("Cannot serialize HQQ layer as {ty}; actual type is {actual_ty}.");
        }

        let mut data = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Hqq as u8,
            ),
            UqffTensor::from_tensor(format!("{prefix}.weight"), &self.w_q)?,
            UqffTensor::from_tensor(format!("{prefix}.weight.scales"), &self.scales)?,
            UqffTensor::from_tensor(format!("{prefix}.weight.zeros"), &self.zeros)?,
            UqffTensor::from_u32_vec(
                format!("{prefix}.weight.shape"),
                self.w_shape.dims().iter().map(|dim| *dim as u32).collect(),
                vec![self.w_shape.dims().len()],
            ),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), self.cfg.bits as u8),
            UqffTensor::from_u32_scalar(
                format!("{prefix}.weight.group_size"),
                self.cfg.group_size.get() as u32,
            ),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.axis"), self.cfg.axis as u8),
            UqffTensor::from_u32_scalar(
                format!("{prefix}.weight.optimization_steps"),
                self.cfg.optimization_steps.unwrap_or(0) as u32,
            ),
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.round_zeros"),
                self.cfg.round_zeros as u8,
            ),
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.channel_wise"),
                self.cfg.channel_wise as u8,
            ),
        ];
        if let Some(bias) = &self.bias {
            data.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(data)
    }
    fn deserialize_uqff(
        reader: &UqffReader,
        prefix: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::from_uqff(reader, prefix, device, shard)?))
    }
    fn isq_type_from_uqff(reader: &UqffReader, prefix: &str) -> Result<IsqType> {
        match HqqBits::try_from(reader.load_u8_scalar(&format!("{prefix}.weight.bits"))? as usize)?
        {
            HqqBits::Eight => Ok(IsqType::HQQ8),
            HqqBits::Four => Ok(IsqType::HQQ4),
            HqqBits::One | HqqBits::Two | HqqBits::Three => {
                candle_core::bail!("Cannot convert HQQ bit width to an ISQ type.")
            }
        }
    }
}
