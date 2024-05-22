use std::{
    collections::HashMap,
    ffi::CStr,
    str::FromStr,
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
};

use candle_core::{quantized::GgmlDType, Device, Result};
use indexmap::IndexMap;
use mistralrs::{
    Constraint, DeviceMapMetadata, MistralRs, MistralRsBuilder, NormalLoaderBuilder,
    NormalLoaderType, NormalRequest, NormalSpecificConfig, Request, RequestMessage, SamplingParams,
    SchedulerMethod, TokenSource,
};
use once_cell::sync::Lazy;
use tokio::sync::mpsc::channel;

static HANDLE_N: AtomicU32 = AtomicU32::new(0);
static REQUEST_N: AtomicUsize = AtomicUsize::new(0);
static mut TABLE: Lazy<HashMap<u32, Arc<MistralRs>>> = Lazy::new(HashMap::new);

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
unsafe fn from_const_ptr_scalar<I: Copy, O: From<I>>(ptr: *const I) -> Option<O> {
    if ptr.is_null() {
        return None;
    }
    Some((*ptr).into())
}

#[repr(C)]
pub struct MistralRsHandle(u32);

#[repr(C)]
pub struct Message {
    pub content: *const i8,
    pub role: *const i8,
    pub name: *const i8, // may be null
}

#[repr(C)]
pub struct ChatCompletionRequest {
    pub messages: *const Message,
    pub n_messages: u32,
    pub max_tokens: *const u32, // may be null
    pub n: u32,
    pub temperature: *const f32,      // may be null
    pub top_p: *const f32,            // may be null
    pub top_k: *const u32,            // may be null
    pub freq_penalty: *const f32,     // may be null
    pub presence_penalty: *const f32, // may be null
}

/// # Safety
/// C is unsafe.
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
    max_seqs: u32,
    chat_template: *const i8,      // may be null
    num_device_layers: *const u32, // may be null
    token_source: *const i8,       // may be null
    // Model loading things
    model_id: *const i8,
    tokenizer_json: *const i8, // may be null
    repeat_last_n: u32,        // may be null
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
    let num_device_layers =
        from_const_ptr_scalar::<u32, u32>(num_device_layers).map(|x| x as usize);

    let arch = NormalLoaderType::from_str(&arch).expect("Invalid architecture");
    let device = get_device().expect("Failed to get device");
    let isq = isq.map(|isq| parse_isq(&isq).expect("Failed to parse ISQ"));
    let loader = NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn,
            repeat_last_n: repeat_last_n as usize,
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
        SchedulerMethod::Fixed(
            (max_seqs as usize)
                .try_into()
                .expect("Failed to parse max seqs"),
        ),
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
    let handle = HANDLE_N.fetch_add(1, Ordering::Relaxed);
    TABLE.insert(handle, mistralrs);
    MistralRsHandle(handle)
}

/// # Safety
/// C is unsafe.
#[no_mangle]
pub unsafe extern "C" fn send_chat_completion(
    chat_completion: ChatCompletionRequest,
    handle: MistralRsHandle,
) {
    if !TABLE.contains_key(&handle.0) {
        panic!("Unknown handle {}", handle.0);
    }
    let mut messages = Vec::new();
    for i in 0..chat_completion.n_messages {
        let message_obj = chat_completion.messages.add(i as usize);
        let role = from_const_ptr((*message_obj).role).expect("Message role");
        let content = from_const_ptr((*message_obj).content).expect("Message content");
        let mut message_map = IndexMap::new();
        message_map.insert("role".to_string(), role);
        message_map.insert("content".to_string(), content);
        messages.push(message_map);
    }

    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        id: REQUEST_N.fetch_add(1, Ordering::Relaxed),
        messages: RequestMessage::Chat(messages),
        sampling_params: SamplingParams {
            temperature: from_const_ptr_scalar(chat_completion.temperature),
            top_k: from_const_ptr_scalar::<u32, u32>(chat_completion.top_k).map(|x| x as usize),
            top_p: from_const_ptr_scalar(chat_completion.top_p),
            top_n_logprobs: 1,
            frequency_penalty: from_const_ptr_scalar(chat_completion.freq_penalty),
            presence_penalty: from_const_ptr_scalar(chat_completion.presence_penalty),
            max_len: from_const_ptr_scalar::<u32, u32>(chat_completion.max_tokens)
                .map(|x| x as usize),
            stop_toks: None,
            logits_bias: None,
            n_choices: chat_completion.n as usize,
        },
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        adapters: None,
    });

    let state = TABLE.get(&handle.0).unwrap().clone();
    let sender = state.get_sender();

    if let Err(e) = sender.blocking_send(request) {
        let e = anyhow::Error::msg(e.to_string());
        MistralRs::maybe_log_error(state, &*e);
        panic!("Failed with error: {e}");
    }

    let _response = match rx.blocking_recv() {
        Some(response) => response,
        None => {
            let e = anyhow::Error::msg("No response received from the model.");
            MistralRs::maybe_log_error(state, &*e);
            panic!("Failed with error: {e}");
        }
    };
}
