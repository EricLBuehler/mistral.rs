use candle_core::Result;
use either::Either;
use indexmap::IndexMap;
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::Debug,
    sync::{mpsc::channel, Arc, Mutex},
};

use ::mistralrs::{MistralRs, Request as _Request, Response, SamplingParams, StopTokens};
use candle_core::Device;
use loaders::{
    gemma::GemmaLoader, llama::LlamaLoader, mistral::MistralLoader, mixtral::MixtralLoader,
    NormalLoader, QuantizedLoader, XLoraLoader, XLoraQuantizedLoader,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyString},
};
mod loaders;

#[pyclass]
enum ModelKind {
    Normal,
    XLoraNormal,
    XLoraGGUF,
    XLoraGGML,
    QuantizedGGUF,
    QuantizedGGML,
}

#[pyclass]
#[derive(Clone)]
enum DType {
    // Unsigned 8 bits integer.
    U8,
    // Unsigned 32 bits integer.
    U32,
    // Signed 64 bits integer.
    I64,
    // Brain floating-point using half precision (16 bits).
    BF16,
    // Floating-point using half precision (16 bits).
    F16,
    // Floating-point using single precision (32 bits).
    F32,
    // Floating-point using double precision (64 bits).
    F64,
}

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

#[pyclass]
/// An object wrapping the underlying Rust system to handle requests and process conversations.
struct Runner {
    runner: Arc<MistralRs>,
}

static NEXT_REQUEST_ID: Mutex<RefCell<usize>> = Mutex::new(RefCell::new(0));

#[pymethods]
impl Runner {
    /// Send an OpenAI API compatible request, returning raw JSON.
    fn send_chat_completion_request(
        &mut self,
        request: Py<ChatCompletionRequest>,
    ) -> PyResult<String> {
        let (tx, rx) = channel();
        Python::with_gil(|py| {
            let request = request.bind(py).borrow();
            let stop_toks = request
                .stop_token_ids
                .as_ref()
                .map(|x| StopTokens::Ids(x.to_vec()));
            let model_request = _Request {
                id: {
                    let l = NEXT_REQUEST_ID.lock().unwrap();
                    let last = &mut *l.borrow_mut();
                    let last_v = *last;
                    *last += 1;
                    last_v
                },
                messages: request.messages.clone(),
                sampling_params: SamplingParams {
                    temperature: request.temperature,
                    top_k: request.top_k,
                    top_p: request.top_p,
                    top_n_logprobs: request.top_logprobs.unwrap_or(1),
                    repeat_penalty: request.repetition_penalty,
                    presence_penalty: request.presence_penalty,
                    max_len: request.max_tokens,
                    stop_toks,
                    logits_bias: request.logit_bias.clone(),
                    n_choices: request.n_choices,
                },
                response: tx,
                return_logprobs: request.logprobs,
                is_streaming: request.stream,
            };

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender();
            sender.send(model_request).unwrap();
            let response = rx.recv().unwrap();

            match response {
                Response::ValidationError(e) | Response::InternalError(e) => {
                    Err(PyValueError::new_err(e.to_string()))
                }
                Response::Done(response) => {
                    MistralRs::maybe_log_response(self.runner.clone(), &response);
                    Ok(serde_json::to_string(&response).unwrap())
                }
                Response::Chunk(_) => unreachable!(),
                Response::ModelError(msg, _) => Err(PyValueError::new_err(msg.to_string())),
            }
        })
    }
}

#[pyclass]
#[derive(Debug)]
/// An OpenAI API compatible chat completion request.
struct ChatCompletionRequest {
    messages: Either<Vec<IndexMap<String, String>>, String>,
    _model: String,
    logit_bias: Option<HashMap<u32, f32>>,
    logprobs: bool,
    top_logprobs: Option<usize>,
    max_tokens: Option<usize>,
    n_choices: usize,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    stop_token_ids: Option<Vec<u32>>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    stream: bool,
    top_k: Option<usize>,
}

#[pymethods]
impl ChatCompletionRequest {
    #[new]
    #[pyo3(signature = (messages, model, logprobs = false, n_choices = 1, logit_bias = None, top_logprobs = None, max_tokens = None, presence_penalty = None, repetition_penalty = None, stop_token_ids = None, temperature = None, top_p = None, top_k = None, stream=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        messages: Py<PyAny>,
        model: String,
        logprobs: bool,
        n_choices: usize,
        logit_bias: Option<HashMap<u32, f32>>,
        top_logprobs: Option<usize>,
        max_tokens: Option<usize>,
        presence_penalty: Option<f32>,
        repetition_penalty: Option<f32>,
        stop_token_ids: Option<Vec<u32>>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        stream: Option<bool>,
    ) -> PyResult<Self> {
        let messages = Python::with_gil(|py| {
            if let Ok(messages) = messages.bind(py).downcast_exact::<PyList>() {
                let mut messages_vec = Vec::new();
                for message in messages {
                    let mapping = message.downcast::<PyDict>()?.as_mapping();
                    let mut messages_map = IndexMap::new();
                    for i in 0..mapping.len()? {
                        let k = mapping
                            .keys()?
                            .get_item(i)?
                            .downcast::<PyString>()?
                            .extract::<String>()?;
                        let v = mapping
                            .values()?
                            .get_item(i)?
                            .downcast::<PyString>()?
                            .extract::<String>()?;
                        messages_map.insert(k, v);
                    }
                    messages_vec.push(messages_map);
                }
                Ok::<Either<Vec<IndexMap<String, String>>, String>, PyErr>(Either::Left(
                    messages_vec,
                ))
            } else if let Ok(messages) = messages.bind(py).downcast_exact::<PyString>() {
                let prompt = messages.extract::<String>()?;
                Ok::<Either<Vec<IndexMap<String, String>>, String>, PyErr>(Either::Right(prompt))
            } else {
                return Err(PyTypeError::new_err("Expected a string or list of dicts."));
            }
        })?;
        Ok(Self {
            messages,
            _model: model,
            logit_bias,
            logprobs,
            top_logprobs,
            max_tokens,
            n_choices,
            presence_penalty,
            repetition_penalty,
            stop_token_ids,
            temperature,
            top_p,
            top_k,
            stream: stream.unwrap_or(false),
        })
    }
}

#[pymodule]
fn mistralrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Runner>()?;
    m.add_class::<MistralLoader>()?;
    m.add_class::<MixtralLoader>()?;
    m.add_class::<GemmaLoader>()?;
    m.add_class::<LlamaLoader>()?;
    m.add_class::<ModelKind>()?;
    m.add_class::<ChatCompletionRequest>()?;
    m.add_class::<NormalLoader>()?;
    m.add_class::<XLoraLoader>()?;
    m.add_class::<QuantizedLoader>()?;
    m.add_class::<XLoraQuantizedLoader>()?;
    m.add_class::<DType>()?;
    Ok(())
}
