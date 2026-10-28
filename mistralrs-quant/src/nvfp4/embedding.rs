use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;

use super::{FP4_VALUES, VALUES_PER_BYTE};
use crate::{safetensors::MmapedSafetensors, NVFP4_BLOCK_SIZE};

pub struct Nvfp4Embedding {
    raw: Arc<MmapedSafetensors>,
    shards: Vec<(String, String)>,
    starts: Vec<usize>,
    rows: usize,
    columns: usize,
    global_scale: f32,
}

pub struct Nvfp4EmbeddingShard<'a> {
    pub data: &'a [u8],
    pub scales: &'a [u8],
    pub first_row: usize,
    pub rows: usize,
}

impl Nvfp4Embedding {
    pub fn new(
        raw: Arc<MmapedSafetensors>,
        shards: Vec<(String, String)>,
        columns: usize,
        global_scale_name: &str,
    ) -> Result<Self> {
        if shards.is_empty() || columns == 0 || !columns.is_multiple_of(NVFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "NVFP4 embeddings require shards and a width divisible by {NVFP4_BLOCK_SIZE}"
            );
        }
        let global = raw.get(global_scale_name)?;
        if DType::try_from(global.dtype())? != DType::F32 || global.data().len() != size_of::<f32>()
        {
            candle_core::bail!(
                "NVFP4 embedding global scale {global_scale_name} must contain one F32 value"
            );
        }
        let global_scale = f32::from_le_bytes(global.data().try_into().unwrap());
        if !global_scale.is_finite() || global_scale <= 0.0 {
            candle_core::bail!("NVFP4 embedding global scale must be finite and positive");
        }
        let mut starts = Vec::with_capacity(shards.len());
        let mut rows = 0;
        for (weight, scale) in &shards {
            let data = raw.get(weight)?;
            let scales = raw.get(scale)?;
            let shape = data.shape();
            if DType::try_from(data.dtype())? != DType::U8
                || shape.len() != 2
                || shape[0] == 0
                || shape[1] != columns / VALUES_PER_BYTE
            {
                candle_core::bail!(
                    "NVFP4 embedding {weight} must be packed U8 [rows, {}]",
                    columns / VALUES_PER_BYTE
                );
            }
            if DType::try_from(scales.dtype())? != DType::F8E4M3
                || scales.shape() != [shape[0], columns / NVFP4_BLOCK_SIZE]
            {
                candle_core::bail!(
                    "NVFP4 embedding {scale} must be E4M3 [{}, {}]",
                    shape[0],
                    columns / NVFP4_BLOCK_SIZE
                );
            }
            starts.push(rows);
            rows += shape[0];
        }
        Ok(Self {
            raw,
            shards,
            starts,
            rows,
            columns,
            global_scale,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_bytes(&self) -> usize {
        self.columns / VALUES_PER_BYTE + self.columns / NVFP4_BLOCK_SIZE
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard(&self, index: usize) -> Result<Nvfp4EmbeddingShard<'_>> {
        let (weight, scale) = &self.shards[index];
        let data = self.raw.get(weight)?;
        let scales = self.raw.get(scale)?;
        Ok(Nvfp4EmbeddingShard {
            data: data.data(),
            scales: scales.data(),
            first_row: self.starts[index],
            rows: data.shape()[0],
        })
    }

    pub fn gather(&self, rows: &[u64], device: &Device) -> Result<Tensor> {
        let mut output = vec![0f32; rows.len() * self.columns];
        let row_bytes = self.columns / VALUES_PER_BYTE;
        let scale_bytes = self.columns / NVFP4_BLOCK_SIZE;
        let shards = (0..self.shard_count())
            .map(|index| self.shard(index))
            .collect::<Result<Vec<_>>>()?;
        for (dst, &row) in output.chunks_exact_mut(self.columns).zip(rows) {
            if row >= self.rows as u64 {
                candle_core::bail!("NVFP4 embedding row {row} is outside {} rows", self.rows);
            }
            let index = self.starts.partition_point(|&start| start <= row as usize) - 1;
            let shard = &shards[index];
            let local = row as usize - shard.first_row;
            let data = &shard.data[local * row_bytes..(local + 1) * row_bytes];
            let scales = &shard.scales[local * scale_bytes..(local + 1) * scale_bytes];
            for ((dst, packed), &scale) in dst
                .as_chunks_mut::<NVFP4_BLOCK_SIZE>()
                .0
                .iter_mut()
                .zip(data.as_chunks::<{ NVFP4_BLOCK_SIZE / VALUES_PER_BYTE }>().0)
                .zip(scales)
            {
                let scale = F8E4M3::from_bits(scale).to_f32();
                for (pair, &packed) in dst
                    .as_chunks_mut::<VALUES_PER_BYTE>()
                    .0
                    .iter_mut()
                    .zip(packed)
                {
                    for (value, code) in pair.iter_mut().zip([packed & 15, packed >> 4]) {
                        let magnitude = FP4_VALUES[usize::from(code & 7)];
                        let q = if code & 8 == 0 { magnitude } else { -magnitude };
                        *value = (q * scale) * self.global_scale;
                    }
                }
            }
        }
        Tensor::from_vec(output, (rows.len(), self.columns), device)
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::{
        cuda::cudarc::driver::{sys::CUstream, DevicePtrMut},
        CudaStorage, Storage,
    };

    use super::*;
    use crate::utils::slice_ptr_on_stream;

    const INDEX_FIELDS: usize = 3;

    #[repr(C)]
    struct Launch {
        rows: *const i64,
        shards: *const u64,
        output: *mut std::ffi::c_void,
        lookups: usize,
        columns: usize,
        shard_count: i32,
        global_scale: f32,
        dtype: i32,
    }

    unsafe extern "C" {
        fn mistralrs_nvfp4_embedding_gather(args: *const Launch, stream: CUstream);
    }

    pub struct CudaNvfp4Embedding {
        index: Tensor,
        columns: usize,
        global_scale: f32,
        _storage: StorageOwner,
    }

    enum StorageOwner {
        Mapped { _raw: Arc<MmapedSafetensors> },
        Resident { data: Tensor, scales: Tensor },
    }

    impl Nvfp4Embedding {
        /// # Safety
        /// The device must support pageable memory access; retain the index through GPU work and graph replay.
        pub unsafe fn cuda_mapped(&self, device: &Device) -> Result<CudaNvfp4Embedding> {
            let mut index = Vec::with_capacity(self.shards.len() * INDEX_FIELDS);
            for i in 0..self.shards.len() {
                let shard = self.shard(i)?;
                index.extend([
                    shard.data.as_ptr() as i64,
                    shard.scales.as_ptr() as i64,
                    shard.first_row as i64,
                ]);
            }
            Ok(CudaNvfp4Embedding {
                index: Tensor::from_vec(index, (self.shards.len(), INDEX_FIELDS), device)?,
                columns: self.columns,
                global_scale: self.global_scale,
                _storage: StorageOwner::Mapped {
                    _raw: self.raw.clone(),
                },
            })
        }

        pub fn cuda_resident(&self, data: Tensor, scales: Tensor) -> Result<CudaNvfp4Embedding> {
            if data.dtype() != DType::U8
                || scales.dtype() != DType::U8
                || data.dims() != [self.rows, self.columns / VALUES_PER_BYTE]
                || scales.dims() != [self.rows, self.columns / NVFP4_BLOCK_SIZE]
                || !data.is_contiguous()
                || !scales.is_contiguous()
                || !data.device().same_device(scales.device())
            {
                candle_core::bail!("NVFP4 embedding resident buffers must be contiguous U8 weight and scale matrices");
            }
            CudaNvfp4Embedding::resident(data, scales, self.columns, self.global_scale)
        }

        pub fn gather_staged(&self, rows: &[u64], dtype: DType, device: &Device) -> Result<Tensor> {
            let row_bytes = self.columns / VALUES_PER_BYTE;
            let scale_bytes = self.columns / NVFP4_BLOCK_SIZE;
            let shards = (0..self.shard_count())
                .map(|index| self.shard(index))
                .collect::<Result<Vec<_>>>()?;
            let mut data = vec![0u8; rows.len() * row_bytes];
            let mut scales = vec![0u8; rows.len() * scale_bytes];
            for ((data, scales), &row) in data
                .chunks_exact_mut(row_bytes)
                .zip(scales.chunks_exact_mut(scale_bytes))
                .zip(rows)
            {
                if row >= self.rows as u64 {
                    candle_core::bail!("NVFP4 embedding row {row} is outside {} rows", self.rows);
                }
                let index = self.starts.partition_point(|&start| start <= row as usize) - 1;
                let shard = &shards[index];
                let local = row as usize - shard.first_row;
                data.copy_from_slice(&shard.data[local * row_bytes..(local + 1) * row_bytes]);
                scales
                    .copy_from_slice(&shard.scales[local * scale_bytes..(local + 1) * scale_bytes]);
            }
            let data = Tensor::from_vec(data, (rows.len(), row_bytes), device)?;
            let scales = Tensor::from_vec(scales, (rows.len(), scale_bytes), device)?;
            let table =
                CudaNvfp4Embedding::resident(data, scales, self.columns, self.global_scale)?;
            let ids = Tensor::arange(0i64, rows.len() as i64, device)?;
            unsafe { table.gather(&ids, dtype) }
        }
    }

    impl CudaNvfp4Embedding {
        fn resident(
            data: Tensor,
            scales: Tensor,
            columns: usize,
            global_scale: f32,
        ) -> Result<Self> {
            let stream = data.device().as_cuda_device()?.cuda_stream();
            let pointer = |tensor: &Tensor| -> Result<i64> {
                let (storage, layout) = tensor.storage_and_layout();
                let Storage::Cuda(storage) = &*storage else {
                    unreachable!()
                };
                let (pointer, _guard) = slice_ptr_on_stream(
                    storage.as_cuda_slice::<u8>()?,
                    layout.start_offset(),
                    &stream,
                );
                Ok(pointer as i64)
            };
            let index = Tensor::from_slice(
                &[pointer(&data)?, pointer(&scales)?, 0],
                (1, INDEX_FIELDS),
                data.device(),
            )?;
            Ok(Self {
                index,
                columns,
                global_scale,
                _storage: StorageOwner::Resident { data, scales },
            })
        }

        /// # Safety
        /// IDs must be in bounds; retain mapped storage through GPU work and all embeddings through graph replay.
        pub unsafe fn gather(&self, ids: &Tensor, dtype: DType) -> Result<Tensor> {
            let dtype_code = match dtype {
                DType::BF16 => 0,
                DType::F16 => 1,
                DType::F32 => 2,
                _ => candle_core::bail!("NVFP4 embedding output must be BF16, F16, or F32"),
            };
            let ids = ids.to_dtype(DType::I64)?.contiguous()?;
            let device = ids.device().as_cuda_device()?;
            if !ids.device().same_device(self.index.device()) {
                candle_core::bail!("NVFP4 embedding IDs and index must be on the same device");
            }
            let count = ids.elem_count();
            if count == 0 {
                return Tensor::zeros((0, self.columns), dtype, ids.device());
            }
            let stream = device.cuda_stream();
            let resident_storage = match &self._storage {
                StorageOwner::Resident { data, scales } => {
                    Some((data.storage_and_layout().0, scales.storage_and_layout().0))
                }
                StorageOwner::Mapped { .. } => None,
            };
            let _resident_guards = if let Some((data, scales)) = resident_storage.as_ref() {
                let (Storage::Cuda(data), Storage::Cuda(scales)) = (&**data, &**scales) else {
                    unreachable!()
                };
                Some((
                    slice_ptr_on_stream(data.as_cuda_slice::<u8>()?, 0, &stream).1,
                    slice_ptr_on_stream(scales.as_cuda_slice::<u8>()?, 0, &stream).1,
                ))
            } else {
                None
            };
            let (ids_storage, ids_layout) = ids.storage_and_layout();
            let (index_storage, index_layout) = self.index.storage_and_layout();
            let (Storage::Cuda(ids_storage), Storage::Cuda(index_storage)) =
                (&*ids_storage, &*index_storage)
            else {
                unreachable!()
            };
            let (ids_ptr, _ids_guard) = slice_ptr_on_stream(
                ids_storage.as_cuda_slice::<i64>()?,
                ids_layout.start_offset(),
                &stream,
            );
            let (index_ptr, _index_guard) = slice_ptr_on_stream(
                index_storage.as_cuda_slice::<i64>()?,
                index_layout.start_offset(),
                &stream,
            );
            macro_rules! launch {
                ($ty:ty) => {{
                    let mut output = unsafe { device.alloc::<$ty>(count * self.columns)? };
                    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
                    let args = Launch {
                        rows: ids_ptr as *const i64,
                        shards: index_ptr as *const u64,
                        output: output_ptr as *mut _,
                        lookups: count,
                        columns: self.columns,
                        shard_count: self.index.dim(0)? as i32,
                        global_scale: self.global_scale,
                        dtype: dtype_code,
                    };
                    unsafe {
                        mistralrs_nvfp4_embedding_gather(&args, stream.cu_stream());
                    }
                    drop(output_guard);
                    let storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                    Ok(Tensor::from((
                        Storage::Cuda(storage),
                        candle_core::Shape::from((count, self.columns)),
                    )))
                }};
            }
            match dtype {
                DType::BF16 => launch!(half::bf16),
                DType::F16 => launch!(half::f16),
                DType::F32 => launch!(f32),
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::CudaNvfp4Embedding;

#[cfg(test)]
#[path = "embedding_tests.rs"]
mod tests;
