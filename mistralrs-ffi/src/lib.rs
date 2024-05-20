use std::{
    collections::HashMap,
    ffi::CStr,
    str::FromStr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use candle_core::{quantized::GgmlDType, Device, Result};
use mistralrs::{
    DeviceMapMetadata, MistralRs, MistralRsBuilder, NormalLoaderBuilder, NormalLoaderType,
    NormalSpecificConfig, SchedulerMethod, TokenSource,
};
use once_cell::sync::Lazy;

static HANDLE_N: AtomicUsize = AtomicUsize::new(0);
static mut TABLE: Lazy<HashMap<usize, Arc<MistralRs>>> = Lazy::new(HashMap::new);

#[cfg(not(feature = "metal"))]
static CUDA_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);
#[cfg(feature = "metal")]
static METAL_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);

#[cfg(not(feature = "metal"))]
fn get_device() -> Result<Device> {
    let mut device = CUDA_DEVICE.lock().unwrap();
    if let Some(device) = device.as_ref() {
        return Ok(device.clone());
    };
    let res = Device::cuda_if_available(0)?;
    *device = Some(res.clone());
    Ok(res)
}
#[cfg(feature = "metal")]
fn get_device() -> Result<Device> {
    let mut device = METAL_DEVICE.lock().unwrap();
    if let Some(device) = device.as_ref() {
        return Ok(device.clone());
    };
    let res = Device::new_metal(0)?;
    *device = Some(res.clone());
    Ok(res)
}

fn parse_isq(s: &str) -> std::result::Result<GgmlDType, String> {
    match s {
        "Q4_0" => Ok(GgmlDType::Q4_0),
        "Q4_1" => Ok(GgmlDType::Q4_1),
        "Q5_0" => Ok(GgmlDType::Q5_0),
        "Q5_1" => Ok(GgmlDType::Q5_1),
        "Q8_0" => Ok(GgmlDType::Q8_0),
        "Q8_1" => Ok(GgmlDType::Q8_1),
        "Q2K" => Ok(GgmlDType::Q2K),
        "Q3K" => Ok(GgmlDType::Q3K),
        "Q4K" => Ok(GgmlDType::Q4K),
        "Q5K" => Ok(GgmlDType::Q5K),
        "Q6K" => Ok(GgmlDType::Q6K),
        "Q8K" => Ok(GgmlDType::Q8K),
        _ => Err(format!("GGML type {s} unknown")),
    }
}

/// Return None if `ptr.is_null()`
/// # Safety
/// Expects a null terminated buffer
/// # Panics
/// On invalid UTF-8.
unsafe fn from_const_ptr(ptr: *const i8) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .expect("Invalid UTF-8")
            .to_string(),
    )
}

/// Return None if `ptr.is_null()`
/// # Safety
/// Expects a valid pointer.
/// # Panics
/// On invalid UTF-8.
unsafe fn from_const_ptr_usize(ptr: *const usize) -> Option<usize> {
    if ptr.is_null() {
        return None;
    }
    Some(*ptr)
}

#[repr(C)]
pub struct MistralRsHandle(usize);

/// # Safety
/// If the user is calling from C (which is inherently unsafe), and they provide
/// corrupt, or corrupting information, there is nothing we can do
#[no_mangle]
pub unsafe extern "C" fn create_mistralrs_plain_model(
    // Metadata
    log_file: *const i8,
    truncate_sequence: bool,
    no_kv_cache: bool,
    no_prefix_cache: bool,
    prefix_cache_n: usize,
    disable_eos_stop: bool,
    gemm_full_precision_f16: bool,
    max_seqs: usize,
    chat_template: *const i8,        // may be null
    num_device_layers: *const usize, // may be null
    token_source: *const i8,         // may be null
    // Model loading things
    model_id: *const i8,
    tokenizer_json: *const i8, // may be null
    repeat_last_n: usize,      // may be null
    arch: *const i8,
    isq: *const i8, // may be null
    use_flash_attn: bool,
) -> MistralRsHandle {
    let log_file = from_const_ptr(log_file).expect("log_file");
    let model_id = from_const_ptr(model_id).expect("model_id");
    let arch = from_const_ptr(arch).expect("arch");
    let tokenizer_json = from_const_ptr(tokenizer_json);
    let isq = from_const_ptr(isq);
    let chat_template = from_const_ptr(chat_template);
    let token_source = from_const_ptr(token_source).unwrap_or("none".to_string());
    let num_device_layers = from_const_ptr_usize(num_device_layers);

    let arch = NormalLoaderType::from_str(&arch).expect("Invalid architecture");
    let device = get_device().expect("Failed to get device");
    let isq = isq.map(|isq| parse_isq(&isq).expect("Failed to parse ISQ"));
    let loader = NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn,
            repeat_last_n,
        },
        chat_template,
        tokenizer_json,
        Some(model_id),
    )
    .build(arch);
    let pipeline = loader
        .load_model_from_hf(
            None,
            TokenSource::from_str(&token_source).expect("Failed to parse token source."),
            None,
            &device,
            true, // Silent for jupyter
            num_device_layers
                .map(DeviceMapMetadata::from_num_device_layers)
                .unwrap_or(DeviceMapMetadata::dummy()),
            isq,
        )
        .expect("Failed to load model.");
    let mistralrs = MistralRsBuilder::new(
        pipeline,
        SchedulerMethod::Fixed(max_seqs.try_into().expect("Failed to parse max seqs")),
    )
    .with_no_kv_cache(no_kv_cache)
    .with_prefix_cache_n(prefix_cache_n)
    .with_log(log_file)
    .with_truncate_sequence(truncate_sequence)
    .with_disable_eos_stop(disable_eos_stop)
    .with_gemm_full_precision_f16(gemm_full_precision_f16)
    .with_no_kv_cache(no_kv_cache)
    .with_no_prefix_cache(no_prefix_cache)
    .build();
    let handle = HANDLE_N.fetch_add(0, Ordering::Relaxed);
    TABLE.insert(handle, mistralrs);
    MistralRsHandle(handle)
}
