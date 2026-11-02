use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use candle_core::{
    cuda::{
        cudarc::driver::{result, sys::CUdevice_attribute},
        DeviceId,
    },
    CudaDevice,
};

const TURING: (i32, i32) = (7, 5);
const AMPERE: (i32, i32) = (8, 0);
const ADA: (i32, i32) = (8, 9);
const HOPPER: (i32, i32) = (9, 0);
const BLACKWELL: (i32, i32) = (10, 0);
const DEFAULT_WARP_SIZE: usize = 32;
const DEFAULT_SMEM_OPTIN: usize = 48 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub major: i32,
    pub minor: i32,
    pub sms: usize,
    pub threads_per_sm: usize,
    pub l2_bytes: usize,
    pub smem_optin: usize,
    pub warp_size: usize,
    pub integrated: bool,
}

impl CudaDeviceInfo {
    pub fn capability(&self) -> (i32, i32) {
        (self.major, self.minor)
    }

    /// llama.cpp-style `major * 100 + minor * 10`.
    pub fn cc_code(&self) -> i32 {
        self.major * 100 + self.minor * 10
    }

    pub fn at_least(&self, capability: (i32, i32)) -> bool {
        self.capability() >= capability
    }

    pub fn has_int8_mma(&self) -> bool {
        self.at_least(TURING)
    }

    pub fn has_cp_async(&self) -> bool {
        self.at_least(AMPERE)
    }

    pub fn has_fp8_mma(&self) -> bool {
        self.at_least(ADA)
    }

    pub fn is_hopper_or_newer(&self) -> bool {
        self.at_least(HOPPER)
    }

    pub fn is_blackwell_or_newer(&self) -> bool {
        self.at_least(BLACKWELL)
    }

    pub fn resident_threads(&self) -> usize {
        self.sms * self.threads_per_sm
    }

    pub fn has_bf16_mma(&self) -> bool {
        self.at_least(AMPERE)
    }
}

fn attribute(dev: &CudaDevice, attr: CUdevice_attribute) -> Option<i32> {
    let cu_device = dev.cuda_stream().context().cu_device();
    unsafe { result::device::get_attribute(cu_device, attr) }.ok()
}

fn probe(dev: &CudaDevice) -> CudaDeviceInfo {
    let get = |attr, default: i32| attribute(dev, attr).unwrap_or(default);
    let positive = |value: i32, default: usize| {
        usize::try_from(value)
            .ok()
            .filter(|&v| v > 0)
            .unwrap_or(default)
    };
    CudaDeviceInfo {
        major: get(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            0,
        ),
        minor: get(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            0,
        ),
        sms: positive(
            get(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                1,
            ),
            1,
        ),
        threads_per_sm: positive(
            get(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                0,
            ),
            1024,
        ),
        l2_bytes: positive(
            get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0),
            0,
        ),
        smem_optin: positive(
            get(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                0,
            ),
            DEFAULT_SMEM_OPTIN,
        ),
        warp_size: positive(
            get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0),
            DEFAULT_WARP_SIZE,
        ),
        integrated: get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED, 0) != 0,
    }
}

pub fn cuda_device_info(dev: &CudaDevice) -> CudaDeviceInfo {
    static INFO: OnceLock<Mutex<HashMap<DeviceId, CudaDeviceInfo>>> = OnceLock::new();
    *INFO
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .entry(dev.id())
        .or_insert_with(|| probe(dev))
}
