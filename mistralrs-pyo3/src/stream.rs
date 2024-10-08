use tokio::sync::mpsc::Receiver;

use crate::response::{ChatCompletionChunkResponse, ChunkChoice};
use mistralrs_core::Response;
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyRef, PyRefMut, PyResult};

#[pyclass]
pub struct ChatCompletionStreamer {
    rx: Receiver<Response>,
    is_done: bool,
}

impl ChatCompletionStreamer {
    pub fn from_rx(rx: Receiver<Response>) -> Self {
        Self { rx, is_done: false }
    }
}

#[pymethods]
impl ChatCompletionStreamer {
    fn __iter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }
    fn __next__(mut this: PyRefMut<'_, Self>) -> Option<PyResult<ChatCompletionChunkResponse>> {
        if this.is_done {
            return None;
        }
        match this.rx.blocking_recv() {
            Some(resp) => match resp {
                Response::ModelError(msg, _) => Some(Err(PyValueError::new_err(msg.to_string()))),
                Response::ValidationError(e) => Some(Err(PyValueError::new_err(e.to_string()))),
                Response::InternalError(e) => Some(Err(PyValueError::new_err(e.to_string()))),
                Response::Chunk(response) => {
                    if response.choices.iter().all(|x| x.finish_reason.is_some()) {
                        this.is_done = true;
                    }
                    Some(Ok(ChatCompletionChunkResponse {
                        id: response.id,
                        created: response.created,
                        model: response.model,
                        system_fingerprint: response.system_fingerprint,
                        object: response.object,
                        choices: response
                            .choices
                            .into_iter()
                            .map(|choice| ChunkChoice {
                                finish_reason: choice.finish_reason.map(|r| r.to_string()),
                                index: choice.index,
                                delta: choice.delta,
                                logprobs: choice.logprobs,
                            })
                            .collect::<Vec<_>>(),
                    }))
                }
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::CompletionModelError(_, _) => unreachable!(),
                Response::CompletionChunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
            },
            None => Some(Err(PyValueError::new_err(
                "Received none in ChatCompletionStreamer".to_string(),
            ))),
        }
    }
}
