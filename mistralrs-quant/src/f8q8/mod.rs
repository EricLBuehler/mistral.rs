use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::{Linear, Module};
use float8::F8E4M3;
use half::f16;
use safetensors::tensor::Dtype;

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
    Shard, UqffReader, UqffTensor,
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
    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] = &[
            "weight",
            "weight.format",
            "weight.num_blocks",
            "weight.shape",
        ];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES)
            && layer.tensor_dtype("weight", Dtype::U8)
            && layer.scalar("weight.format", Dtype::U8)
            && layer.scalar("weight.num_blocks", Dtype::U32)
            && layer.u32_vector("weight.shape")
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::F8Q8,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        _tensors: &[UqffTensor],
        _prefix: &str,
    ) -> Result<String> {
        Ok("f8q8".to_string())
    }

    pub fn from_raw_parts(
        raw_data: Vec<u8>,
        dims: Vec<usize>,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let block_size = std::mem::size_of::<BlockF8Q8>();
        if !raw_data.len().is_multiple_of(block_size) {
            candle_core::bail!(
                "F8Q8 raw data length {} is not divisible by block size {block_size}.",
                raw_data.len()
            );
        }
        let num_blocks = raw_data.len() / block_size;
        let data = unsafe {
            let mut blocks = Vec::with_capacity(num_blocks);
            std::ptr::copy_nonoverlapping(
                raw_data.as_ptr(),
                blocks.as_mut_ptr() as *mut u8,
                raw_data.len(),
            );
            blocks.set_len(num_blocks);
            blocks
        };
        Ok(Self {
            data,
            shape: Shape::from_dims(&dims),
            bias,
        })
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        let mut weight = reader.load_raw_u8(&format!("{key}.weight"))?;
        let mut shape = reader.load_u32_vec(&format!("{key}.weight.shape"))?;
        let range = crate::uqff::shard_range(shard, &shape)?;
        if let Some((dim, start, len)) = range {
            weight = crate::uqff::slice_blocked_data(
                &weight,
                &shape,
                QK8_0,
                std::mem::size_of::<BlockF8Q8>(),
                dim,
                start,
                len,
            )?;
            shape[dim] = len;
        }
        let bias = reader.load_bias(key, device, range, shape.len())?;
        Self::from_raw_parts(weight, shape, bias)
    }

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

    fn dequantize_rows(&self, ids: &Tensor) -> Result<Tensor> {
        let (row_count, row_size) = self.shape.dims2()?;
        let output_shape = [ids.dims(), &[row_size]].concat();
        let ids = ids
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let mut output = Vec::with_capacity(ids.len() * row_size);

        for id in ids {
            let id = id as usize;
            if id >= row_count {
                candle_core::bail!("embedding index {id} is out of bounds for {row_count} rows");
            }
            let mut offset = id * row_size;
            let end = offset + row_size;
            while offset < end {
                let block = &self.data[offset / QK8_0];
                let block_offset = offset % QK8_0;
                let len = (QK8_0 - block_offset).min(end - offset);
                let scale = block.dq_d();
                output.extend(
                    block.qs[block_offset..block_offset + len]
                        .iter()
                        .map(|value| *value as f32 * scale),
                );
                offset += len;
            }
        }

        Tensor::from_vec(output, output_shape, &Device::Cpu)
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

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        self.dequantize_rows(ids)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
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

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            DType::F32,
            Device::Cpu,
            self.shape.dims().to_vec(),
            request,
            true,
        ))
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
                // Already F8Q8 or no-op, just return self
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

impl QuantizedSerde for F8Q8Linear {
    fn name(&self) -> &'static str {
        "f8q8-linear"
    }

    fn isq_serde_supported(&self) -> bool {
        true
    }

    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        if ty != IsqType::F8Q8 {
            candle_core::bail!("Cannot serialize F8Q8 layer as {ty}; actual type is F8Q8.");
        }

        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<BlockF8Q8>(),
            )
            .to_vec()
        };
        let block_bytes_len = block_bytes.len();
        let mut data = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::F8Q8 as u8,
            ),
            UqffTensor::from_raw_u8(
                format!("{prefix}.weight"),
                block_bytes,
                vec![block_bytes_len],
            ),
            UqffTensor::from_u32_scalar(
                format!("{prefix}.weight.num_blocks"),
                self.data.len() as u32,
            ),
            UqffTensor::from_u32_vec(
                format!("{prefix}.weight.shape"),
                self.shape.dims().iter().map(|dim| *dim as u32).collect(),
                vec![self.shape.dims().len()],
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
    fn isq_type_from_uqff(_reader: &UqffReader, _prefix: &str) -> Result<IsqType> {
        Ok(IsqType::F8Q8)
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
    fn test_f8q8_embedding_gathers_before_dequantizing() {
        let data = (0..185)
            .map(|i| (i as f32 - 92.0) / 37.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_slice(&data, (5, 37), &Device::Cpu).unwrap();
        let linear = F8Q8Linear::from_weight(&weight, None).unwrap();
        let ids = Tensor::new(&[[4u32, 1], [3, 1]], &Device::Cpu).unwrap();
        let actual = linear.embedding_forward_raw(&ids).unwrap();
        let expected = linear
            .dequantize(DType::F32)
            .unwrap()
            .index_select(&ids.flatten_all().unwrap(), 0)
            .unwrap()
            .reshape((2, 2, 37))
            .unwrap();

        assert_eq!(actual.dims(), &[2, 2, 37]);
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn test_f8q8_block_size() {
        assert_eq!(std::mem::size_of::<BlockF8Q8>(), 33);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }
}
