//! Candle ↔ MLX tensor bridge.
//!
//! Converts tensors between Candle (`candle_core::Tensor`) and MLX (`mlx_rs::Array`)
//! representations. On Apple Silicon the copy stays within unified memory — there is
//! no PCIe transfer, just a memcpy within the same physical RAM pool.
//!
//! ## Zero-copy upgrade path (future)
//!
//! When `mlx-rs` exposes `Array::from_metal_buffer` (or equivalent stable Metal buffer
//! interop), replace the body of `candle_to_mlx` with a pointer share guarded by a
//! `PhantomData` lifetime tie to the source Candle tensor. This would eliminate the
//! memcpy entirely since both frameworks operate on the same unified memory.
//!
//! ```ignore
//! // FUTURE: zero-copy path (not yet available in mlx-rs 0.25)
//! // pub fn candle_to_mlx_zero_copy<'a>(tensor: &'a Tensor) -> Array {
//! //     let metal_buf = tensor.metal_buffer_ptr();
//! //     unsafe { Array::from_metal_ptr(metal_buf, shape, dtype) }
//! // }
//! ```

use candle_core::{DType, Device, Tensor};
use mlx_rs::Array;

/// Convert a Candle tensor to an MLX array.
///
/// Moves the tensor to CPU storage first (stays in unified memory on Apple Silicon),
/// then constructs an MLX array from the raw data. Supports F32, F16, and BF16.
pub fn candle_to_mlx(tensor: &Tensor) -> candle_core::Result<Array> {
    let cpu = tensor.to_device(&Device::Cpu)?;
    let shape: Vec<i32> = cpu.dims().iter().map(|&d| d as i32).collect();

    match cpu.dtype() {
        DType::F32 => {
            let data = cpu.flatten_all()?.to_vec1::<f32>()?;
            Ok(Array::from_slice(&data, &shape))
        }
        DType::F16 => {
            // mlx-rs handles half::f16 via the half crate
            let data = cpu.flatten_all()?.to_vec1::<half::f16>()?;
            // Convert to f32 for MLX (mlx-rs Array::from_slice supports f32 natively)
            let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
            Ok(Array::from_slice(&f32_data, &shape))
        }
        DType::BF16 => {
            let data = cpu.flatten_all()?.to_vec1::<half::bf16>()?;
            let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
            Ok(Array::from_slice(&f32_data, &shape))
        }
        dt => candle_core::bail!("candle_to_mlx: unsupported dtype {:?}", dt),
    }
}

/// Convert an MLX array back to a Candle tensor on the specified device.
///
/// Forces MLX lazy evaluation via `eval()` before reading data, then constructs
/// a Candle tensor with the requested dtype and device placement.
pub fn mlx_to_candle(
    array: &Array,
    device: &Device,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    // Force computation of any pending lazy MLX ops
    array
        .eval()
        .map_err(|e| candle_core::Error::Msg(format!("mlx eval failed: {e}")))?;

    let shape: Vec<usize> = array.shape().iter().map(|&d| d as usize).collect();
    let numel: usize = shape.iter().product();

    // Read as f32 from MLX (the internal working dtype)
    let f32_data: Vec<f32> = array
        .as_slice::<f32>()
        .map_err(|e| candle_core::Error::Msg(format!("mlx as_slice failed: {e}")))?
        .to_vec();

    if f32_data.len() != numel {
        candle_core::bail!(
            "mlx_to_candle: expected {} elements, got {}",
            numel,
            f32_data.len()
        );
    }

    // Build tensor in requested dtype
    let tensor = match dtype {
        DType::F32 => Tensor::from_vec(f32_data, shape, &Device::Cpu)?,
        DType::F16 => {
            let f16_data: Vec<half::f16> = f32_data.iter().map(|&x| half::f16::from_f32(x)).collect();
            Tensor::from_vec(f16_data, shape, &Device::Cpu)?
        }
        DType::BF16 => {
            let bf16_data: Vec<half::bf16> =
                f32_data.iter().map(|&x| half::bf16::from_f32(x)).collect();
            Tensor::from_vec(bf16_data, shape, &Device::Cpu)?
        }
        dt => candle_core::bail!("mlx_to_candle: unsupported dtype {:?}", dt),
    };

    tensor.to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_roundtrip_f32() {
        let original = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let original = original.reshape((2, 2)).unwrap();
        let mlx_arr = candle_to_mlx(&original).unwrap();
        let restored = mlx_to_candle(&mlx_arr, &Device::Cpu, DType::F32).unwrap();
        let orig_data = original.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rest_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(orig_data, rest_data);
        assert_eq!(original.dims(), restored.dims());
    }

    #[test]
    fn test_roundtrip_f16() {
        let data: Vec<half::f16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect();
        let original = Tensor::from_vec(data, (2, 2), &Device::Cpu).unwrap();
        let mlx_arr = candle_to_mlx(&original).unwrap();
        let restored = mlx_to_candle(&mlx_arr, &Device::Cpu, DType::F16).unwrap();
        // F16 round-trip through F32 should preserve these simple values
        let orig_data = original.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        let rest_data = restored.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        assert_eq!(orig_data, rest_data);
        assert_eq!(original.dims(), restored.dims());
    }

    #[test]
    fn test_unsupported_dtype() {
        let t = Tensor::new(&[1u8, 2, 3, 4], &Device::Cpu).unwrap();
        assert!(candle_to_mlx(&t).is_err());
    }

    #[test]
    fn test_shape_preservation() {
        let original = Tensor::zeros((1, 8, 5, 128), DType::F32, &Device::Cpu).unwrap();
        let mlx_arr = candle_to_mlx(&original).unwrap();
        let restored = mlx_to_candle(&mlx_arr, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(original.dims(), restored.dims());
    }
}
