use std::sync::mpsc::Receiver;

use mistralrs::Response;
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
    fn __next__(mut this: PyRefMut<'_, Self>) -> Option<PyResult<String>> {
        if this.is_done {
            return None;
        }
        match this.rx.recv() {
            Ok(resp) => match resp {
                Response::ModelError(msg, _) => Some(Err(PyValueError::new_err(msg.to_string()))),
                Response::ValidationError(e) => Some(Err(PyValueError::new_err(e.to_string()))),
                Response::InternalError(e) => Some(Err(PyValueError::new_err(e.to_string()))),
                Response::Chunk(response) => {
                    if response.choices.iter().all(|x| x.stopreason.is_some()) {
                        this.is_done = true;
                    }
                    Some(Ok(serde_json::to_string(&response).unwrap()))
                }
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::CompletionModelError(_, _) => unreachable!(),
            },
            Err(e) => Some(Err(PyValueError::new_err(e.to_string()))),
        }
    }
}
