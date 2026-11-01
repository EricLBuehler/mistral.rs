use candle_core::{
    cuda::cudarc::driver::{CudaStream, SyncOnDrop},
    DType, Result, Storage, StorageRef, Tensor,
};

use super::slice_ptr_on_stream;

pub(crate) struct CudaReadView<'a> {
    storage: StorageRef<'a>,
    offset: usize,
}

impl<'a> CudaReadView<'a> {
    pub(crate) fn new(tensor: &'a Tensor) -> Result<Self> {
        let (storage, layout) = tensor.storage_and_layout();
        if !matches!(&*storage, Storage::Cuda(_)) || !layout.is_contiguous() {
            candle_core::bail!("CUDA read view requires a contiguous CUDA tensor");
        }
        Ok(Self {
            storage,
            offset: layout.start_offset(),
        })
    }

    pub(crate) fn optional(tensor: Option<&'a Tensor>) -> Result<Option<Self>> {
        tensor.map(Self::new).transpose()
    }

    pub(crate) fn read<'b>(&'b self, stream: &'b CudaStream) -> Result<(u64, SyncOnDrop<'b>)> {
        let Storage::Cuda(storage) = &*self.storage else {
            unreachable!()
        };
        macro_rules! read {
            ($ty:ty) => {
                slice_ptr_on_stream(storage.as_cuda_slice::<$ty>()?, self.offset, stream)
            };
        }
        Ok(match self.storage.dtype() {
            DType::U8 => read!(u8),
            DType::U32 => read!(u32),
            DType::I32 => read!(i32),
            DType::I64 => read!(i64),
            DType::BF16 => read!(half::bf16),
            DType::F16 => read!(half::f16),
            DType::F32 => read!(f32),
            DType::F8E4M3 => read!(float8::F8E4M3),
            dtype => candle_core::bail!("unsupported CUDA read dtype {dtype:?}"),
        })
    }

    pub(crate) fn read_optional<'b>(
        view: Option<&'b Self>,
        stream: &'b CudaStream,
    ) -> Result<(u64, SyncOnDrop<'b>)> {
        view.map_or_else(
            || Ok((0, SyncOnDrop::Record(None))),
            |view| view.read(stream),
        )
    }
}
