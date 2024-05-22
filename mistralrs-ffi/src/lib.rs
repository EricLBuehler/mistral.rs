use std::{
    collections::HashMap,
    ffi::{CStr, CString},
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
    NormalLoaderType, NormalRequest, NormalSpecificConfig, Request, RequestMessage, Response,
    SamplingParams, SchedulerMethod, TokenSource,
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

/// An object representing an input message.
/// The following arguments are optional and may be null:
/// - name
#[repr(C)]
pub struct Message {
    pub content: *const i8,
    pub role: *const i8,
    pub name: *const i8, // may be null
}

/// An object representing a chat completion request
/// The following arguments are optional and may be null:
/// - max_tokens
/// - temperature
/// - top_p
/// - top_k
/// - freq_penalty
/// - presence_penalty
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

#[repr(C)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    pub avg_tok_per_sec: f32,
    pub avg_prompt_tok_per_sec: f32,
    pub avg_compl_tok_per_sec: f32,
    pub total_time_sec: f32,
    pub total_prompt_time_sec: f32,
    pub total_completion_time_sec: f32,
}

#[repr(C)]
pub struct ResponseMessage {
    pub content: *const i8,
    pub role: *const i8,
}

#[repr(C)]
pub struct Choice {
    pub finish_reason: *const i8,
    pub index: u32,
    pub message: ResponseMessage,
}

#[repr(C)]
pub struct ChatCompletionResponse {
    pub id: *const i8,
    pub choices: *const Choice,
    pub n_choices: u32,
    pub created: u64,
    pub model: *const i8,
    pub system_fingerprint: *const i8,
    pub object: *const i8,
    pub usage: Usage,
}

/// Construct a MistralRS instance. See the other API docs.
/// The following arguments are optional and may be null:
/// - chat_template
/// - num_device_layers
/// - token_source
/// - tokenizer_json
/// - repeat_last_n
///
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

/// Make a chat completion request.
///
/// The user *must* call `drop_chat_completion_response` or memory will leak.
/// Do not deallocate returned pointers with C `free`, that is unsound.
/// # Safety
/// C is unsafe.
#[no_mangle]
pub unsafe extern "C" fn send_chat_completion(
    chat_completion: ChatCompletionRequest,
    handle: MistralRsHandle,
) -> ChatCompletionResponse {
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

    let response = match rx.blocking_recv() {
        Some(response) => response,
        None => {
            let e = anyhow::Error::msg("No response received from the model.");
            MistralRs::maybe_log_error(state, &*e);
            panic!("Failed with error: {e}");
        }
    };
    match response {
        Response::Done(resp) => {
            let id = CString::new(resp.id.as_bytes())
                .expect("Failed to convert to cstring")
                .into_raw() as *const i8;
            let mut choices = Vec::with_capacity(resp.choices.len());
            for choice in resp.choices {
                choices.push(Choice {
                    finish_reason: CString::new(choice.finish_reason.as_bytes())
                        .expect("Failed to convert to cstring")
                        .into_raw() as *const i8,
                    index: choice.index as u32,
                    message: ResponseMessage {
                        content: CString::new(choice.message.content.as_bytes())
                            .expect("Failed to convert to cstring")
                            .into_raw() as *const i8,
                        role: CString::new(choice.message.role.as_bytes())
                            .expect("Failed to convert to cstring")
                            .into_raw() as *const i8,
                    },
                })
            }
            let choices_ptr = choices.as_ptr();
            let model = CString::new(resp.model.as_bytes())
                .expect("Failed to convert to cstring")
                .into_raw() as *const i8;
            let system_fingerprint = CString::new(resp.system_fingerprint.as_bytes())
                .expect("Failed to convert to cstring")
                .into_raw() as *const i8;
            let object = CString::new(resp.object.as_bytes())
                .expect("Failed to convert to cstring")
                .into_raw() as *const i8;

            ChatCompletionResponse {
                id,
                choices: choices_ptr,
                n_choices: choices.len() as u32,
                created: resp.created,
                model,
                system_fingerprint,
                object,
                usage: Usage {
                    completion_tokens: resp.usage.completion_tokens as u32,
                    prompt_tokens: resp.usage.prompt_tokens as u32,
                    total_tokens: resp.usage.total_tokens as u32,
                    avg_tok_per_sec: resp.usage.avg_tok_per_sec,
                    avg_prompt_tok_per_sec: resp.usage.avg_prompt_tok_per_sec,
                    avg_compl_tok_per_sec: resp.usage.avg_compl_tok_per_sec,
                    total_time_sec: resp.usage.total_time_sec,
                    total_prompt_time_sec: resp.usage.total_prompt_time_sec,
                    total_completion_time_sec: resp.usage.total_completion_time_sec,
                },
            }
        }
        Response::InternalError(e) => {
            panic!("Internal error: {e}");
        }
        Response::ModelError(e, resp) => {
            panic!("Model error: {e}, response: {resp:?}");
        }
        Response::ValidationError(e) => {
            panic!("Validation error: {e}");
        }
        _ => unreachable!(),
    }
}

/// Free the chat completion response object properly.
///
/// This function *must* be called on all chat completion response objects before they go out of scope.
/// Otherwise, memory will leak. Do not deallocate pointers with C `free`, that is unsound.
/// # Safety
/// C is unsafe.
#[no_mangle]
pub unsafe extern "C" fn drop_chat_completion_response(object: ChatCompletionResponse) {
    let _ = unsafe { CString::from_raw(object.id as *mut i8) };
    let _ = unsafe { CString::from_raw(object.model as *mut i8) };
    let _ = unsafe { CString::from_raw(object.object as *mut i8) };
    let _ = unsafe { CString::from_raw(object.system_fingerprint as *mut i8) };
    let choices = Vec::from_raw_parts(
        object.choices as *mut Choice,
        object.n_choices as usize,
        object.n_choices as usize,
    );
    for choice in choices {
        let _ = unsafe { CString::from_raw(choice.finish_reason as *mut i8) };
        let _ = unsafe { CString::from_raw(choice.message.content as *mut i8) };
        let _ = unsafe { CString::from_raw(choice.message.role as *mut i8) };
    }
}
