use candle_core::Result;
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{mpsc::channel, Arc},
};

use ::mistralrs::{
    Conversation, MistralRs, Request as _Request, Response, SamplingParams, StopTokens,
};
use candle_core::Device;
use loaders::mistral::MistralLoader;
use pyo3::{exceptions::PyValueError, prelude::*};
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
struct MistralRunner {
    runner: Arc<MistralRs>,
    conversation: Arc<dyn Conversation + Send + Sync>,
}

#[pymethods]
impl MistralRunner {
    /// Send an OpenAI API compatible request, returning raw JSON.
    fn send_chat_completion_request(&mut self, request: Py<Request>) -> PyResult<String> {
        let (tx, rx) = channel();
        Python::with_gil(|py| {
            let request = request.as_ref(py).borrow();
            let stop_toks = request
                .stop_token_ids
                .as_ref()
                .map(|x| StopTokens::Ids(x.to_vec()));
            let prompt = match self.conversation.get_prompt(request.messages.clone(), true) {
                Err(e) => return Err(PyValueError::new_err(e.to_string())),
                Ok(p) => p,
            };
            let model_request = _Request {
                prompt,
                sampling_params: SamplingParams {
                    temperature: request.temperature,
                    top_k: request.top_k,
                    top_p: request.top_p,
                    top_n_logprobs: request.top_logprobs.unwrap_or(1),
                    repeat_penalty: request.repetition_penalty,
                    presence_penalty: request.presence_penalty,
                    max_len: request.max_tokens,
                    stop_toks,
                },
                response: tx,
                return_logprobs: request.logprobs,
            };

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender();
            sender.send(model_request).unwrap();
            let response = rx.recv().unwrap();

            match response {
                Response::Error(e) => Err(PyValueError::new_err(e.to_string())),
                Response::Done(response) => {
                    MistralRs::maybe_log_response(self.runner.clone(), &response);
                    Ok(serde_json::to_string(&response).unwrap())
                }
            }
        })
    }
}

#[pyclass]
#[derive(Debug)]
/// An OpenAI API compatible chat completion request.
struct Request {
    messages: Vec<HashMap<String, String>>,
    _model: String,
    _logit_bias: Option<HashMap<u32, f64>>,
    logprobs: bool,
    top_logprobs: Option<usize>,
    max_tokens: Option<usize>,
    _n_choices: usize,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    stop_token_ids: Option<Vec<u32>>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
}

#[pymethods]
impl Request {
    #[new]
    #[pyo3(signature = (messages, model, logprobs = false, n_choices = 1, logit_bias = None, top_logprobs = None, max_tokens = None, presence_penalty = None, repetition_penalty = None, stop_token_ids = None, temperature = None, top_p = None, top_k = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        messages: Vec<HashMap<String, String>>,
        model: String,
        logprobs: bool,
        n_choices: usize,
        logit_bias: Option<HashMap<u32, f64>>,
        top_logprobs: Option<usize>,
        max_tokens: Option<usize>,
        presence_penalty: Option<f32>,
        repetition_penalty: Option<f32>,
        stop_token_ids: Option<Vec<u32>>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> Self {
        Self {
            messages,
            _model: model,
            _logit_bias: logit_bias,
            logprobs,
            top_logprobs,
            max_tokens,
            _n_choices: n_choices,
            presence_penalty,
            repetition_penalty,
            stop_token_ids,
            temperature,
            top_p,
            top_k,
        }
    }
}

#[pymodule]
fn mistralrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MistralRunner>()?;
    m.add_class::<MistralLoader>()?;
    m.add_class::<ModelKind>()?;
    m.add_class::<Request>()?;
    Ok(())
}
