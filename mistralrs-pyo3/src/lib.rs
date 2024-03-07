use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{mpsc::channel, Arc},
};

use ::mistralrs::{
    Conversation, MistralRs, Request as _Request, Response, SamplingParams, StopTokens,
};
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

#[pyclass]
struct MistralRunner {
    runner: Arc<MistralRs>,
    conversation: Arc<dyn Conversation + Send + Sync>,
}

#[pyclass]
#[derive(Debug)]
pub struct Request {
    pub messages: Vec<HashMap<String, String>>,
    pub model: String,
    pub logit_bias: Option<HashMap<u32, f64>>,
    pub logprobs: bool,
    pub top_logprobs: Option<usize>,
    pub max_tokens: Option<usize>,
    pub n_choices: usize,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
}

#[pymethods]
impl MistralRunner {
    fn add_request(&mut self, request: Py<Request>) -> PyResult<String> {
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

#[pymodule]
fn mistralrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MistralRunner>()?;
    m.add_class::<MistralLoader>()?;
    m.add_class::<ModelKind>()?;
    Ok(())
}
