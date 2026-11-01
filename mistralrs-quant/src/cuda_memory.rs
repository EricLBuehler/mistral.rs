use candle_core::{
    cuda::cudarc::driver::{result, sys},
    Device, Result,
};

const PREFAULT_STRIDE_BYTES: usize = 4096;

/// Synchronously prepare mapped bytes for GPU reads; returns false without shared host-page-table support.
pub fn prefault_mapped_memory(bytes: &[u8], device: &Device) -> Result<bool> {
    let Device::Cuda(device) = device else {
        return Ok(false);
    };
    let stream = device.cuda_stream();
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let ordinal = stream.context().ordinal();
    let dev = result::device::get(ordinal as i32).map_err(candle_core::Error::wrap)?;
    for attribute in [
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
    ] {
        if unsafe { result::device::get_attribute(dev, attribute) }
            .map_err(candle_core::Error::wrap)?
            != 1
        {
            return Ok(false);
        }
    }
    if bytes.is_empty() {
        return Ok(true);
    }
    if unsafe { result::stream::is_capturing(stream.cu_stream()) }
        .map_err(candle_core::Error::wrap)?
        != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        candle_core::bail!("mapped memory prefault must run outside CUDA graph capture");
    }
    // CPU page faults are cheaper than faulting every page from a GPU warp.
    for byte in bytes.iter().step_by(PREFAULT_STRIDE_BYTES) {
        std::hint::black_box(*byte);
    }
    std::hint::black_box(bytes[bytes.len() - 1]);
    unsafe extern "C" {
        fn mapped_memory_prefault(
            data: *const u8,
            bytes: usize,
            stream: *mut std::ffi::c_void,
        ) -> i32;
    }
    let status = unsafe {
        mapped_memory_prefault(bytes.as_ptr(), bytes.len(), stream.cu_stream() as *mut _)
    };
    if status != 0 {
        candle_core::bail!("mapped memory prefault failed with CUDA status {status}");
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATTERN: u8 = 137;

    #[test]
    #[ignore = "requires CUDA with shared host page tables"]
    fn mapped_prefault_rejects_capture_of_borrowed_memory() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;
        if !prefault_mapped_memory(&[], &device)? {
            return Ok(());
        }
        let bytes = vec![PATTERN; PREFAULT_STRIDE_BYTES + 1];
        let stream = device.as_cuda_device()?.cuda_stream();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let result = prefault_mapped_memory(&bytes[1..], &device);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        assert!(result.is_err());
        drop(graph.map_err(candle_core::Error::wrap)?);
        assert!(prefault_mapped_memory(&bytes[1..], &device)?);
        assert!(bytes.iter().all(|byte| *byte == PATTERN));
        std::thread::spawn(move || -> Result<()> {
            assert!(prefault_mapped_memory(&bytes[1..], &device)?);
            assert!(bytes.iter().all(|byte| *byte == PATTERN));
            Ok(())
        })
        .join()
        .unwrap()?;
        Ok(())
    }
}
