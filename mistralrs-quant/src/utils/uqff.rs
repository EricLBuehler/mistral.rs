use byteorder::{LittleEndian, ReadBytesExt};

use candle_core::{DType, Device, Result, Tensor, WithDType};
use float8::F8E4M3;
use half::{bf16, f16};

// v0.1.0: initial release
// v0.1.1: add i16 dtype
// v0.1.2: add F8E4M3

const UQFF_VERSION_MAJOR: u32 = 0;
const UQFF_VERSION_MINOR: u32 = 1;
const UQFF_VERSION_PATCH: u32 = 2;

/// Format 4 bytes, little endian: [ UNSPECIFIED ] [ MAJOR ] [ MINOR ] [ PATCH ]
pub(crate) const UQFF_VERSION: u32 =
    (UQFF_VERSION_MAJOR << (8 * 2)) | (UQFF_VERSION_MINOR << 8) | UQFF_VERSION_PATCH;
/// Offset for the quant type. UQFF always serializes the version first.
pub const UQFF_QUANT_TYPE_OFFSET: usize = std::mem::size_of::<u32>();

/// Check if major version matches: is backwards compatible
pub(crate) fn version_is_compatible(version: u32) -> Result<()> {
    let major = version >> (8 * 2);
    let _minor = version >> 8;
    let _patch = version;

    if major != UQFF_VERSION_MAJOR {
        candle_core::bail!("Major version of ISQ artifact file ({major}) does not match the implementation in this build ({UQFF_VERSION_MAJOR})");
    }

    Ok(())
}

// -----------------------
// Tensor dtype, u32, little endian
// -----------------------
pub(crate) fn write_dtype(dtype: DType, buffer: &mut Vec<u8>) {
    let dtype: u32 = match dtype {
        DType::U8 => 0,
        DType::U32 => 1,
        DType::I32 => 2,
        DType::I64 => 3,
        DType::F16 => 4,
        DType::BF16 => 5,
        DType::F32 => 6,
        DType::F64 => 7,
        DType::I16 => 8,
        DType::F8E4M3 => 9,
    };
    buffer.extend(&dtype.to_le_bytes());
}

pub(crate) fn read_dtype<R: std::io::Read>(buffer: &mut R) -> Result<DType> {
    let dtype = buffer.read_u32::<LittleEndian>()?;
    let dtype = match dtype {
        0 => DType::U8,
        1 => DType::U32,
        2 => DType::I32,
        3 => DType::I64,
        4 => DType::F16,
        5 => DType::BF16,
        6 => DType::F32,
        7 => DType::F64,
        8 => DType::I16,
        9 => DType::F8E4M3,
        _ => candle_core::bail!("unknown dtype for quantized tensor {dtype}"),
    };
    Ok(dtype)
}

// -----------------------
// Tensor data length, u32, little endian
// -----------------------
// Tensor dtype, u32, little endian
// -----------------------
// Num shape dims, u32, little endian
// -----------------------
// ...
// Array (in original order): shape dims, u32, little endian
// ...
// -----------------------
// ...
// Array: tensor data, u8s
// ...
// -----------------------

pub(crate) fn serialize_tensor(buffer: &mut Vec<u8>, tensor: &Tensor) -> Result<()> {
    let b_shape = tensor.dims();
    let tensor = tensor.flatten_all()?;

    let bias = match tensor.dtype() {
        DType::U8 => data_to_bytes::<u8>(tensor.to_vec1()?),
        DType::U32 => data_to_bytes::<u32>(tensor.to_vec1()?),
        DType::I16 => data_to_bytes::<i16>(tensor.to_vec1()?),
        DType::I32 => data_to_bytes::<i32>(tensor.to_vec1()?),
        DType::I64 => data_to_bytes::<i64>(tensor.to_vec1()?),
        DType::F16 => data_to_bytes::<half::f16>(tensor.to_vec1()?),
        DType::BF16 => data_to_bytes::<half::bf16>(tensor.to_vec1()?),
        DType::F32 => data_to_bytes::<f32>(tensor.to_vec1()?),
        DType::F64 => data_to_bytes::<f64>(tensor.to_vec1()?),
        DType::F8E4M3 => data_to_bytes::<F8E4M3>(tensor.to_vec1()?),
    };
    buffer.extend(&(bias.len() as u32).to_le_bytes());

    // DType
    write_dtype(tensor.dtype(), buffer);

    // Shape
    buffer.extend((b_shape.len() as u32).to_le_bytes());
    for dim in b_shape {
        buffer.extend((*dim as u32).to_le_bytes());
    }

    buffer.extend(bias);

    Ok(())
}

pub(crate) fn deserialize_tensor<R: std::io::Read>(
    buffer: &mut R,
    device: &Device,
) -> Result<Tensor> {
    let data_len = buffer.read_u32::<LittleEndian>()? as usize;

    // DType
    let dtype = read_dtype(buffer)?;

    let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

    let mut dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        dims.push(buffer.read_u32::<LittleEndian>()? as usize)
    }

    let mut tensor_data = vec![0; data_len];
    buffer.read_exact(&mut tensor_data)?;

    match dtype {
        DType::F16 => bytes_to_data::<f16>(&tensor_data, &dims, device),
        DType::BF16 => bytes_to_data::<bf16>(&tensor_data, &dims, device),
        DType::F32 => bytes_to_data::<f32>(&tensor_data, &dims, device),
        DType::F64 => bytes_to_data::<f64>(&tensor_data, &dims, device),
        DType::I32 => bytes_to_data::<i32>(&tensor_data, &dims, device),
        DType::I64 => bytes_to_data::<i64>(&tensor_data, &dims, device),
        DType::I16 => bytes_to_data::<i16>(&tensor_data, &dims, device),
        DType::U32 => bytes_to_data::<u32>(&tensor_data, &dims, device),
        DType::U8 => bytes_to_data::<u8>(&tensor_data, &dims, device),
        DType::F8E4M3 => bytes_to_data::<F8E4M3>(&tensor_data, &dims, device),
    }
}

/// Just seek the reader ahead.
pub(crate) fn fake_deserialize_tensor<R: std::io::Read + std::io::Seek>(
    buffer: &mut R,
) -> Result<()> {
    let data_len = buffer.read_u32::<LittleEndian>()? as usize;

    // DType
    let _dtype = read_dtype(buffer)?;

    let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

    let mut dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        dims.push(buffer.read_u32::<LittleEndian>()? as usize)
    }

    // Fake read the data in bytes
    buffer.seek_relative(data_len as i64)?;

    Ok(())
}

fn data_to_bytes<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let length = vs.len() * size_in_bytes;
    let capacity = vs.capacity() * size_in_bytes;
    let ptr = vs.as_mut_ptr() as *mut u8;
    // Don't run the destructor for Vec<T>
    std::mem::forget(vs);
    // SAFETY:
    //
    // Every T is larger than u8, so there is no issue regarding alignment.
    // This re-interpret the Vec<T> as a Vec<u8>.
    unsafe { Vec::from_raw_parts(ptr, length, capacity) }
}

fn bytes_to_data<T: WithDType>(
    data: &[u8],
    shape: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, shape, device)
    }
}
