use std::sync::Arc;

use candle_core::{quantized::k_quants::*, Device, Result, Tensor};

use super::archive::{GgufArchive, GgufDType, GgufEndian};

#[cfg(feature = "cuda")]
const CUDA_GATHER_THREADS: usize = 256;
const IQ4_NL: u32 = 20;
const IQ4_BLOCK: usize = 32;
const IQ4_BLOCK_BYTES: usize = 18;
const IQ4_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub struct GgufEmbedding {
    archive: Arc<GgufArchive>,
    name: String,
    rows: usize,
    columns: usize,
    row_bytes: usize,
    dtype: GgufDType,
}

impl GgufEmbedding {
    pub fn new(archive: Arc<GgufArchive>, name: &str) -> Result<Self> {
        let info = archive.tensor_info(name)?;
        if archive.shards()[info.shard_index()].endian() != GgufEndian::Little {
            candle_core::bail!("big-endian GGUF tensor loading is not supported");
        }
        let &[rows, columns] = info.shape() else {
            candle_core::bail!("GGUF embedding {name} must be a matrix");
        };
        if columns == 0 {
            candle_core::bail!("GGUF embedding {name} must have a nonzero row width");
        }
        let dtype = info.dtype();
        if dtype.raw() != IQ4_NL {
            dtype.candle_dtype()?;
        }
        let block_size = dtype.block_size().unwrap();
        if !columns.is_multiple_of(block_size) {
            candle_core::bail!(
                "GGUF embedding {name} row width must be a multiple of {block_size}"
            );
        }
        let row_bytes = columns / block_size * dtype.type_size().unwrap();
        Ok(Self {
            archive,
            name: name.to_owned(),
            rows,
            columns,
            row_bytes,
            dtype,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn bytes(&self) -> Result<&[u8]> {
        Ok(self.archive.tensor_data(&self.name)?.bytes())
    }

    pub fn gather(&self, rows: &[u64], device: &Device) -> Result<Tensor> {
        let bytes = self.bytes()?;
        let mut out = vec![0f32; rows.len() * self.columns];
        for (dst, &row) in out.chunks_exact_mut(self.columns).zip(rows) {
            if row >= self.rows as u64 {
                candle_core::bail!("embedding row {row} is outside {} rows", self.rows);
            }
            let offset = row as usize * self.row_bytes;
            dequantize_row(self.dtype, &bytes[offset..offset + self.row_bytes], dst)?;
        }
        Tensor::from_vec(out, (rows.len(), self.columns), device)
    }

    /// # Safety
    /// IDs must be in bounds and fit i32; data must contain aligned, contiguous bytes on the IDs' device.
    /// For K formats, each ID multiplied by (columns / 256) must also fit i32.
    /// The storage must outlive GPU work and graph replay.
    #[cfg(feature = "cuda")]
    pub unsafe fn gather_resident(&self, rows: &Tensor, data: &Tensor) -> Result<Tensor> {
        use candle_core::{cuda::cudarc::driver::DevicePtr, Storage};
        let (storage, layout) = data.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("resident GGUF embedding requires CUDA storage");
        };
        let slice = storage.as_cuda_slice::<u8>()?;
        let stream = rows.device().as_cuda_device()?.cuda_stream();
        let (ptr, _guard) = slice.device_ptr(&stream);
        self.gather_cuda(rows, ptr + layout.start_offset() as u64)
    }

    /// # Safety
    /// IDs must be in bounds and fit i32; the device must support pageable access and the table be aligned.
    /// For K formats, each ID multiplied by (columns / 256) must also fit i32.
    /// The mapping must outlive GPU work and graph replay.
    #[cfg(feature = "cuda")]
    pub unsafe fn gather_mapped(&self, rows: &Tensor) -> Result<Tensor> {
        self.gather_cuda(rows, self.bytes()?.as_ptr() as u64)
    }

    #[cfg(feature = "cuda")]
    pub fn mapped_gather_aligned(&self) -> Result<bool> {
        let alignment = match self.dtype.raw() {
            0 | 3 | 7 | 10 | 12 | 13 | 15 => std::mem::align_of::<f32>(),
            _ => std::mem::align_of::<half::f16>(),
        };
        Ok(self.bytes()?.as_ptr().align_offset(alignment) == 0)
    }

    #[cfg(feature = "cuda")]
    fn gather_cuda(&self, rows: &Tensor, pointer: u64) -> Result<Tensor> {
        use candle_core::{
            cuda::cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
            CudaStorage, DType, Storage,
        };
        let dev = rows.device().as_cuda_device()?;
        if rows.elem_count() == 0 {
            return Tensor::zeros((0, self.columns), DType::F32, rows.device());
        }
        let ids = rows.flatten_all()?.to_dtype(DType::U32)?.contiguous()?;
        let (storage, layout) = ids.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            unreachable!()
        };
        let ids = storage
            .as_cuda_slice::<u32>()?
            .slice(layout.start_offset()..layout.start_offset() + rows.elem_count());
        let mut output = unsafe { dev.alloc::<f32>(rows.elem_count() * self.columns)? };
        let (kernel, threads, tiles) = match self.dtype.raw() {
            0 => (
                "get_rows_f32",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(CUDA_GATHER_THREADS),
            ),
            1 => (
                "get_rows_f16",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(CUDA_GATHER_THREADS),
            ),
            30 => (
                "get_rows_bf16",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(CUDA_GATHER_THREADS),
            ),
            2 => (
                "get_rows_q4_0",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(2 * CUDA_GATHER_THREADS),
            ),
            3 => (
                "get_rows_q4_1",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(2 * CUDA_GATHER_THREADS),
            ),
            6 => (
                "get_rows_q5_0",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(2 * CUDA_GATHER_THREADS),
            ),
            7 => (
                "get_rows_q5_1",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(2 * CUDA_GATHER_THREADS),
            ),
            8 => (
                "get_rows_q8_0",
                CUDA_GATHER_THREADS,
                self.columns.div_ceil(2 * CUDA_GATHER_THREADS),
            ),
            10 => ("get_rows_q2_K", 64, self.columns / 256),
            11 => ("get_rows_q3_K", 64, self.columns / 256),
            12 => ("get_rows_q4_K", 32, self.columns / 256),
            13 => ("get_rows_q5_K", 64, self.columns / 256),
            14 => ("get_rows_q6_K", 64, self.columns / 256),
            9 | 15 | IQ4_NL => {
                unsafe extern "C" {
                    fn gguf_embedding_gather_extra(
                        data: *const u8,
                        ids: *const u32,
                        out: *mut f32,
                        count: usize,
                        columns: usize,
                        format: u32,
                        stream: *mut std::ffi::c_void,
                    );
                }
                let stream = dev.cuda_stream();
                let (id_ptr, _id_guard) = ids.device_ptr(&stream);
                let (out_ptr, _out_guard) = output.device_ptr_mut(&stream);
                unsafe {
                    gguf_embedding_gather_extra(
                        pointer as *const u8,
                        id_ptr as *const u32,
                        out_ptr as *mut f32,
                        rows.elem_count(),
                        self.columns,
                        self.dtype.raw(),
                        stream.cu_stream() as *mut _,
                    );
                }
                drop(_out_guard);
                return Ok(Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                    candle_core::Shape::from((rows.elem_count(), self.columns)),
                )));
            }
            _ => unreachable!("validated GGUF embedding dtype"),
        };
        let func = dev.get_or_load_func(kernel, &candle_core::cuda_backend::kernels::QUANTIZED)?;
        let mut builder = func.builder();
        let columns = self.columns as i64;
        builder
            .arg(&pointer)
            .arg(&ids)
            .arg(&mut output)
            .arg(&columns)
            .arg(&self.row_bytes);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows.elem_count() as u32, tiles as u32, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle_core::Error::wrap)?;
        Ok(Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
            candle_core::Shape::from((rows.elem_count(), self.columns)),
        )))
    }
}

fn dequantize_blocks<T: GgmlType>(src: &[u8], out: &mut [f32]) {
    for (bytes, dst) in src
        .chunks_exact(std::mem::size_of::<T>())
        .zip(out.chunks_exact_mut(T::BLCK_SIZE))
    {
        // GGUF rows can start at an unaligned shard offset.
        let block = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) };
        T::to_float(std::slice::from_ref(&block), dst);
    }
}

fn dequantize_row(dtype: GgufDType, src: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype.raw() {
        0 => dequantize_blocks::<f32>(src, out),
        1 => dequantize_blocks::<half::f16>(src, out),
        30 => dequantize_blocks::<half::bf16>(src, out),
        2 => dequantize_blocks::<BlockQ4_0>(src, out),
        3 => dequantize_blocks::<BlockQ4_1>(src, out),
        6 => dequantize_blocks::<BlockQ5_0>(src, out),
        7 => dequantize_blocks::<BlockQ5_1>(src, out),
        8 => dequantize_blocks::<BlockQ8_0>(src, out),
        9 => dequantize_blocks::<BlockQ8_1>(src, out),
        10 => dequantize_blocks::<BlockQ2K>(src, out),
        11 => dequantize_blocks::<BlockQ3K>(src, out),
        12 => dequantize_blocks::<BlockQ4K>(src, out),
        13 => dequantize_blocks::<BlockQ5K>(src, out),
        14 => dequantize_blocks::<BlockQ6K>(src, out),
        15 => dequantize_blocks::<BlockQ8K>(src, out),
        IQ4_NL => {
            for (block, dst) in src
                .as_chunks::<IQ4_BLOCK_BYTES>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<IQ4_BLOCK>().0)
            {
                let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                for (j, &byte) in block[2..].iter().enumerate() {
                    dst[j] = scale * f32::from(IQ4_VALUES[usize::from(byte & 15)]);
                    dst[j + IQ4_BLOCK / 2] = scale * f32::from(IQ4_VALUES[usize::from(byte >> 4)]);
                }
            }
        }
        _ => candle_core::bail!("unsupported GGUF embedding format {}", dtype.name()),
    }
    Ok(())
}
