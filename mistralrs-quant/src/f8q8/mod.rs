use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::{Linear, Module};
use float8::F8E4M3;
use half::f16;

use crate::{
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
};

#[cfg(target_feature = "avx")]
mod avx;
#[cfg(target_feature = "neon")]
mod neon;
#[cfg(target_feature = "simd128")]
mod simd128;

pub(crate) const QK8_0: usize = 32;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockF8Q8 {
    d: F8E4M3,
    pub(crate) qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockF8Q8>() == 33);

impl BlockF8Q8 {
    pub fn dq_d(&self) -> f32 {
        self.d.to_f32() / F8E4M3::MAX.to_f32()
    }

    fn zeros() -> Self {
        BlockF8Q8 {
            d: F8E4M3::ZERO,
            qs: [0i8; QK8_0],
        }
    }
}

// Our own BlockQ8_0 with accessible fields for vec_dot kernels.
// candle_core's BlockQ8_0 has pub(crate) fields we can't access.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

// ---- GgmlType-like functions ----

fn to_float(xs: &[BlockF8Q8], ys: &mut [f32]) -> Result<()> {
    let k = ys.len();
    if !k.is_multiple_of(QK8_0) {
        candle_core::bail!("dequantize_row_f8q8: {k} is not divisible by {QK8_0}");
    }

    let nb = k / QK8_0;

    for i in 0..nb {
        let d = xs[i].dq_d();

        for j in 0..QK8_0 {
            ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
        }
    }
    Ok(())
}

fn from_float(xs: &[f32], ys: &mut [BlockF8Q8]) -> Result<()> {
    let k = xs.len();
    if !k.is_multiple_of(QK8_0) {
        candle_core::bail!("{k} is not divisible by {QK8_0}");
    }
    let nb = k / QK8_0;
    if ys.len() != nb {
        candle_core::bail!("size mismatch {} {} {}", xs.len(), ys.len(), QK8_0)
    }
    for (i, ys) in ys.iter_mut().enumerate() {
        let mut amax = 0f32;
        let xs = &xs[i * QK8_0..(i + 1) * QK8_0];
        for &x in xs.iter() {
            amax = amax.max(x.abs())
        }
        let d = amax / ((1 << 7) - 1) as f32;
        let id = if d != 0f32 { 1. / d } else { 0. };
        ys.d = F8E4M3::from_f32(d * F8E4M3::MAX.to_f32());
        for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
            *y = f32::round(x * id) as i8
        }
    }
    Ok(())
}

#[allow(dead_code)]
#[allow(unreachable_code)]
fn vec_dot(n: usize, xs: &[BlockF8Q8], ys: &[BlockQ8_0]) -> Result<f32> {
    #[cfg(target_feature = "avx")]
    return avx::vec_dot_f8q8_q8_0(n, xs, ys);

    #[cfg(target_feature = "neon")]
    return neon::vec_dot_f8q8_q8_0(n, xs, ys);

    #[cfg(target_feature = "simd128")]
    return simd128::vec_dot_f8q8_q8_0(n, xs, ys);

    vec_dot_unopt(n, xs, ys)
}

#[allow(dead_code)]
fn vec_dot_unopt(n: usize, xs: &[BlockF8Q8], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    if !n.is_multiple_of(QK8_0) {
        candle_core::bail!("vec_dot_f8q8_q8_0: {n} is not divisible by {qk}")
    }

    let mut sumf = 0f32;
    for (xs, ys) in xs.iter().zip(ys.iter()) {
        let sum_i = xs
            .qs
            .iter()
            .zip(ys.qs.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum::<i32>();
        sumf += sum_i as f32 * xs.dq_d() * f16::to_f32(ys.d)
    }
    Ok(sumf)
}

#[allow(dead_code)]
#[allow(unreachable_code)]
#[allow(unused)]
#[cfg(feature = "arm-nightly-feat")]
fn matmul_i8mm(
    n: usize,
    xs_0: &[BlockF8Q8],
    xs_1: &[BlockF8Q8],
    ys_0: &[BlockQ8_0],
    ys_1: &[BlockQ8_0],
) -> Result<[f32; 4]> {
    #[cfg(target_feature = "neon")]
    return neon::i8mm_f8q8_q8_0(n, xs_0, xs_1, ys_0, ys_1);

    candle_core::bail!("Unsupported block type for i8mm");
}

// ---- F8Q8Linear ----

#[derive(Debug)]
pub struct F8Q8Linear {
    data: Vec<BlockF8Q8>,
    shape: Shape,
    bias: Option<Tensor>,
}

impl F8Q8Linear {
    pub fn from_weight(weight: &Tensor, bias: Option<Tensor>) -> Result<Self> {
        let shape = weight.shape().clone();
        let weight_f32 = weight.to_dtype(DType::F32)?.flatten_all()?;
        let mut weight_data: Vec<f32> = weight_f32.to_vec1()?;

        // Pad to multiple of QK8_0
        let elem_count = weight_data.len();
        let padded_count = elem_count.div_ceil(QK8_0) * QK8_0;
        weight_data.resize(padded_count, 0.0);

        let num_blocks = padded_count / QK8_0;
        let mut blocks = vec![BlockF8Q8::zeros(); num_blocks];
        from_float(&weight_data, &mut blocks)?;

        Ok(Self {
            data: blocks,
            shape,
            bias,
        })
    }

    fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        let num_blocks = self.data.len();
        let total_floats = num_blocks * QK8_0;
        let mut output = vec![0f32; total_floats];
        to_float(&self.data, &mut output)?;

        // Trim padding and reshape
        let n = self.shape.elem_count();
        let output = &output[..n];
        Tensor::from_slice(output, &self.shape, &Device::Cpu)?.to_dtype(dtype)
    }
}

impl QuantMethod for F8Q8Linear {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        let _ = method;
        candle_core::bail!("F8Q8Linear should be constructed via from_weight")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.dequantize(DType::F32)
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let dequant_w = self.dequantize(a.dtype())?;
        let lin = Linear::new(dequant_w, self.bias.clone());
        lin.forward(a)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, Device::Cpu)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dequant = self.dequantize(delta.dtype())?;
        let new_w = (dequant + delta)?;
        Ok(Arc::new(Self::from_weight(&new_w, self.bias.clone())?))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            Some(IsqType::F8Q8) | None => {
                // Already F8Q8 or no-op — just return self
                Ok(self)
            }
            Some(other) => {
                // Dequantize and re-quantize to requested type
                let w = self.dequantize(DType::F32)?;
                let b = self.bias.clone();
                let unquant =
                    crate::UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(w, b)))?;
                Arc::new(unquant).apply_isq(Some(other), device, n_quantized, None, guard)
            }
        }
    }
}

// ---- Serialization ----
//
// Layout:
// | UQFF_VERSION (u32) | type=5 (u8) | has_bias (u8) | num_blocks (u32) |
// | shape_ndims (u32) | shape_dims[] (u32 each) |
// | raw BlockF8Q8 data (33 * num_blocks bytes) |
// | [optional bias via serialize_tensor] |

impl QuantizedSerde for F8Q8Linear {
    fn name(&self) -> &'static str {
        "f8q8-linear"
    }

    fn isq_serde_supported(&self) -> bool {
        true
    }

    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }

    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();

        // Version
        buffer.extend(&UQFF_VERSION.to_le_bytes());

        // ISQ type
        buffer.push(QuantizedSerdeType::F8Q8 as u8);

        // Has bias
        buffer.push(bias.is_some() as u8);

        // Num blocks
        buffer.extend(&(self.data.len() as u32).to_le_bytes());

        // Shape
        let dims = self.shape.dims();
        buffer.extend(&(dims.len() as u32).to_le_bytes());
        for &dim in dims {
            buffer.extend(&(dim as u32).to_le_bytes());
        }

        // Raw block data
        let block_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<BlockF8Q8>(),
            )
        };
        buffer.extend(block_bytes);

        // Optional bias
        if let Some(ref b) = bias {
            serialize_tensor(&mut buffer, b)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data.to_vec());

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::F8Q8 as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::F8Q8 as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let num_blocks = buffer.read_u32::<LittleEndian>()? as usize;

        // Shape
        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize);
        }
        let shape = Shape::from_dims(&dims);

        // Raw block data
        let block_byte_count = num_blocks * std::mem::size_of::<BlockF8Q8>();
        let mut raw_data = vec![0u8; block_byte_count];
        std::io::Read::read_exact(&mut buffer, &mut raw_data)?;

        // Safety: BlockF8Q8 is #[repr(C)] and 33 bytes
        let blocks: Vec<BlockF8Q8> = unsafe {
            let mut blocks = Vec::with_capacity(num_blocks);
            std::ptr::copy_nonoverlapping(
                raw_data.as_ptr(),
                blocks.as_mut_ptr() as *mut u8,
                block_byte_count,
            );
            blocks.set_len(num_blocks);
            blocks
        };

        let _acquired_load_guard = guard.acquire(device);

        let bias = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok(Arc::new(F8Q8Linear {
            data: blocks,
            shape,
            bias,
        }))
    }

    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data.to_vec());

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::F8Q8 as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::F8Q8 as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let num_blocks = buffer.read_u32::<LittleEndian>()? as usize;

        // Shape
        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize);
        }
        let shape = Shape::from_dims(&dims);

        // Raw block data
        let block_byte_count = num_blocks * std::mem::size_of::<BlockF8Q8>();
        let mut raw_data = vec![0u8; block_byte_count];
        std::io::Read::read_exact(&mut buffer, &mut raw_data)?;

        let blocks: Vec<BlockF8Q8> = unsafe {
            let mut blocks = Vec::with_capacity(num_blocks);
            std::ptr::copy_nonoverlapping(
                raw_data.as_ptr(),
                blocks.as_mut_ptr() as *mut u8,
                block_byte_count,
            );
            blocks.set_len(num_blocks);
            blocks
        };

        let _acquired_load_guard = guard.acquire(device);

        let bias = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok((
            Arc::new(F8Q8Linear {
                data: blocks,
                shape,
                bias: None,
            }),
            bias,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f8q8_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let weight = Tensor::from_slice(&data, (16, 16), &Device::Cpu).unwrap();
        let linear = F8Q8Linear::from_weight(&weight, None).unwrap();
        let dequant = linear.dequantize(DType::F32).unwrap();
        let dequant_data: Vec<f32> = dequant.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_err = 0f32;
        for (a, b) in data.iter().zip(dequant_data.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 0.1,
            "F8Q8 roundtrip max error {max_err} exceeds threshold"
        );
    }

    #[test]
    fn test_f8q8_non_divisible_shape() {
        let data: Vec<f32> = (0..10000).map(|i| (i as f32 - 5000.0) / 5000.0).collect();
        let weight = Tensor::from_slice(&data, (100, 100), &Device::Cpu).unwrap();
        let linear = F8Q8Linear::from_weight(&weight, None).unwrap();
        let dequant = linear.dequantize(DType::F32).unwrap();
        assert_eq!(dequant.dims(), &[100, 100]);

        let dequant_data: Vec<f32> = dequant.flatten_all().unwrap().to_vec1().unwrap();
        let mut max_err = 0f32;
        for (a, b) in data.iter().zip(dequant_data.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 0.1,
            "F8Q8 non-divisible shape roundtrip max error {max_err} exceeds threshold"
        );
    }

    #[test]
    fn test_f8q8_block_size() {
        assert_eq!(std::mem::size_of::<BlockF8Q8>(), 33);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }
}
