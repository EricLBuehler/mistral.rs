use super::*;

pub(super) struct CudaGraphPinnedAllocation<T> {
    pub(super) allocation: PinnedHostSlice<T>,
    pub(super) ptr: NonNull<T>,
}

// The allocation owns the pointer and all accesses require an exclusive borrow.
unsafe impl<T: Send> Send for CudaGraphPinnedAllocation<T> {}

impl<T> CudaGraphPinnedAllocation<T> {
    pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.allocation.len()) }
    }

    pub(super) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

pub(super) enum CudaGraphPinnedData {
    U8(CudaGraphPinnedAllocation<u8>),
    U32(CudaGraphPinnedAllocation<u32>),
    I32(CudaGraphPinnedAllocation<i32>),
    I64(CudaGraphPinnedAllocation<i64>),
    F32(CudaGraphPinnedAllocation<f32>),
}

pub(super) struct CudaGraphPinnedBuffer {
    pub(super) data: CudaGraphPinnedData,
    pub(super) initialized: bool,
}

pub(super) struct CudaGraphCopyCompletion {
    pub(super) event: CudaEvent,
    pub(super) stream: Arc<CudaStream>,
    pub(super) pending: bool,
    pub(super) active: bool,
    pub(super) ordered_after_graph: bool,
}

pub(crate) struct CudaGraphHostStaging {
    pub(super) buffers: HashMap<(&'static str, DeviceLocation), CudaGraphPinnedBuffer>,
    pub(super) completions: HashMap<DeviceLocation, CudaGraphCopyCompletion>,
    pub(super) graph_complete: CudaEvent,
    pub(super) graph_stream: Arc<CudaStream>,
    pub(super) graph_pending: bool,
}

pub(super) fn same_cuda_stream(left: &CudaStream, right: &CudaStream) -> bool {
    Arc::ptr_eq(left.context(), right.context()) && left.cu_stream() == right.cu_stream()
}

impl Drop for CudaGraphHostStaging {
    fn drop(&mut self) {
        for completion in self.completions.values() {
            if completion.pending {
                let _ = completion.event.synchronize();
            } else if completion.active {
                let _ = completion.stream.synchronize();
            }
        }
    }
}

impl CudaGraphPinnedBuffer {
    pub(super) fn new(dst: &Var) -> candle_core::Result<Self> {
        let Device::Cuda(device) = dst.device() else {
            candle_core::bail!("CUDA graph host staging requires a CUDA destination");
        };
        let stream = device.cuda_stream();
        let context = stream.context();
        let len = dst.elem_count();
        macro_rules! allocate {
            ($variant:ident, $ty:ty) => {{
                let mut allocation = unsafe {
                    context
                        .alloc_pinned::<$ty>(len)
                        .map_err(candle_core::Error::wrap)?
                };
                let ptr = NonNull::new(allocation.as_mut_ptr().map_err(candle_core::Error::wrap)?)
                    .ok_or_else(|| {
                        candle_core::Error::msg("CUDA returned a null pinned pointer")
                    })?;
                CudaGraphPinnedData::$variant(CudaGraphPinnedAllocation { allocation, ptr })
            }};
        }
        let data = match dst.dtype() {
            DType::U8 => allocate!(U8, u8),
            DType::U32 => allocate!(U32, u32),
            DType::I32 => allocate!(I32, i32),
            DType::I64 => allocate!(I64, i64),
            DType::F32 => allocate!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        };
        Ok(Self {
            data,
            initialized: false,
        })
    }

    pub(super) fn copy_from(
        &mut self,
        src: &Tensor,
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
            candle_core::bail!("CUDA graph host staging expected matching tensors");
        }
        let (src_storage, src_layout) = src.storage_and_layout();
        let Storage::Cpu(src_storage) = &*src_storage else {
            candle_core::bail!("CUDA graph host staging expected CPU source metadata");
        };
        if !src_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous source metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA destination metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous destination metadata");
        }
        let len = src.elem_count();
        let src_offset = src_layout.start_offset();
        let dst_offset = dst_layout.start_offset();

        macro_rules! stage_and_copy {
            ($variant:ident, $ty:ty) => {{
                let CudaGraphPinnedData::$variant(host) = &mut self.data else {
                    candle_core::bail!("CUDA graph host staging dtype changed");
                };
                let src = src_storage.as_slice::<$ty>()?;
                let src = &src[src_offset..src_offset + len];
                let host_slice = host.as_mut_slice();
                if self.initialized && host_slice == src {
                    return Ok(());
                }
                host_slice.copy_from_slice(src);
                self.initialized = true;
                let dst = dst_storage.as_cuda_slice::<$ty>()?;
                let dst = dst.slice(dst_offset..dst_offset + len);
                let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
                let result = unsafe {
                    sys::cuMemcpyHtoDAsync_v2(
                        dst_ptr,
                        host.as_ptr().cast(),
                        len * std::mem::size_of::<$ty>(),
                        stream.cu_stream(),
                    )
                };
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(candle_core::Error::msg(format!("{result:?}"))
                        .context("CUDA graph metadata H2D copy failed"));
                }
            }};
        }

        match src.dtype() {
            DType::U8 => stage_and_copy!(U8, u8),
            DType::U32 => stage_and_copy!(U32, u32),
            DType::I32 => stage_and_copy!(I32, i32),
            DType::I64 => stage_and_copy!(I64, i64),
            DType::F32 => stage_and_copy!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        }
        Ok(())
    }

    pub(super) fn copy_from_u32_slice(
        &mut self,
        src: &[u32],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::U32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching u32 state indices");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA state indices");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous state indices");
        }
        let CudaGraphPinnedData::U32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging state index dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<u32>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph state index H2D copy failed"));
        }
        Ok(())
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub(super) fn copy_from_f32_slice(
        &mut self,
        src: &[f32],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::F32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching f32 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA f32 metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous f32 metadata");
        }
        let CudaGraphPinnedData::F32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging f32 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<f32>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph f32 metadata H2D copy failed"));
        }
        Ok(())
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub(super) fn copy_from_i64_slice(
        &mut self,
        src: &[i64],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::I64 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching i64 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA i64 metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous i64 metadata");
        }
        let CudaGraphPinnedData::I64(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging i64 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<i64>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph i64 metadata H2D copy failed"));
        }
        Ok(())
    }
}

impl CudaGraphHostStaging {
    pub(crate) fn new(graph_stream: Arc<CudaStream>) -> candle_core::Result<Self> {
        let graph_complete = graph_stream
            .context()
            .new_event(None)
            .map_err(candle_core::Error::wrap)?;
        Ok(Self {
            buffers: HashMap::new(),
            completions: HashMap::new(),
            graph_complete,
            graph_stream,
            graph_pending: false,
        })
    }

    pub(crate) fn update(
        &mut self,
        copy: impl FnOnce(&mut Self) -> candle_core::Result<()>,
    ) -> candle_core::Result<()> {
        self.begin_update()?;
        let copy_result = copy(self);
        let finish_result = self.finish_update();
        copy_result.and(finish_result)
    }

    pub(super) fn begin_update(&mut self) -> candle_core::Result<()> {
        for completion in self.completions.values_mut() {
            completion.ordered_after_graph = false;
            if completion.active {
                completion
                    .stream
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.active = false;
            }
            if completion.pending {
                completion
                    .event
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.pending = false;
            }
        }
        Ok(())
    }

    pub(super) fn finish_update(&mut self) -> candle_core::Result<()> {
        let mut result = Ok(());
        for completion in self.completions.values_mut() {
            if !completion.active {
                continue;
            }
            match completion.event.record(&completion.stream) {
                Ok(()) => {
                    completion.pending = true;
                    completion.active = false;
                }
                Err(err) => {
                    completion.pending = false;
                    if completion.stream.synchronize().is_ok() {
                        completion.active = false;
                    }
                    if result.is_ok() {
                        result = Err(candle_core::Error::wrap(err));
                    }
                }
            }
        }
        result
    }

    pub(crate) fn order_before_graph(&self) -> candle_core::Result<()> {
        for completion in self.completions.values() {
            if completion.pending && !same_cuda_stream(&completion.stream, &self.graph_stream) {
                self.graph_stream
                    .wait(&completion.event)
                    .map_err(candle_core::Error::wrap)?;
            }
        }
        Ok(())
    }

    pub(crate) fn record_graph_complete(&mut self) -> candle_core::Result<()> {
        self.graph_complete
            .record(&self.graph_stream)
            .map_err(candle_core::Error::wrap)?;
        self.graph_pending = true;
        Ok(())
    }

    pub(super) fn prepare_copy(
        &mut self,
        location: DeviceLocation,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        let completion = match self.completions.entry(location) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let event = stream
                    .context()
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                    .map_err(candle_core::Error::wrap)?;
                entry.insert(CudaGraphCopyCompletion {
                    event,
                    stream: stream.clone(),
                    pending: false,
                    active: false,
                    ordered_after_graph: false,
                })
            }
        };
        if !same_cuda_stream(&completion.stream, stream) {
            candle_core::bail!("CUDA graph metadata stream changed during replay");
        }
        if self.graph_pending && !completion.ordered_after_graph {
            if !same_cuda_stream(&completion.stream, &self.graph_stream) {
                completion
                    .stream
                    .wait(&self.graph_complete)
                    .map_err(candle_core::Error::wrap)?;
            }
            completion.ordered_after_graph = true;
        }
        completion.active = true;
        Ok(())
    }

    pub(super) fn copy_from(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &Tensor,
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from(src, dst, &stream)
    }

    pub(crate) fn copy_from_u32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[u32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_u32_slice(src, dst, &stream)
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn copy_from_f32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[f32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_f32_slice(src, dst, &stream)
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn copy_from_i64_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[i64],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_i64_slice(src, dst, &stream)
    }
}
