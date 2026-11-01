use std::collections::HashSet;

use tokio::sync::mpsc::Receiver;

use mistralrs_core::{ChatCompletionChunkResponse, Response};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyRef, PyRefMut, PyResult};

#[pyclass]
pub struct ChatCompletionStreamer {
    rx: Option<Receiver<Response>>,
    is_done: bool,
    n_choices: usize,
    finished_choices: HashSet<usize>,
}

impl ChatCompletionStreamer {
    pub fn from_rx(rx: Receiver<Response>, n_choices: usize) -> Self {
        Self {
            rx: Some(rx),
            is_done: false,
            n_choices,
            finished_choices: HashSet::new(),
        }
    }

    fn observe_chunk(&mut self, response: &ChatCompletionChunkResponse) {
        self.finished_choices.extend(
            response
                .choices
                .iter()
                .filter(|choice| choice.finish_reason.is_some())
                .map(|choice| choice.index),
        );
        if self.finished_choices.len() == self.n_choices {
            self.finish();
        }
    }

    fn finish(&mut self) {
        self.is_done = true;
        self.rx.as_mut().expect("stream receiver").close();
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
        loop {
            let py = this.py();
            // `&mut Receiver` through PyRefMut is not Ungil, so move the receiver out while waiting.
            let mut rx = this.rx.take().expect("stream receiver");
            let recv_result = py.allow_threads(|| rx.blocking_recv());
            this.rx = Some(rx);

            match recv_result {
                Some(resp) => match resp {
                    Response::AgenticToolCallProgress { .. } => continue,
                    Response::AgenticToolApprovalRequired { .. } => continue,
                    Response::BlockDenoisingProgress(_) => continue,
                    Response::File(_) => continue,
                    Response::ModelError(msg, _) => {
                        this.finish();
                        return Some(Err(PyValueError::new_err(msg.to_string())));
                    }
                    Response::ValidationError(e) => {
                        this.finish();
                        return Some(Err(PyValueError::new_err(e.to_string())));
                    }
                    Response::InternalError(e) => {
                        this.finish();
                        return Some(Err(PyValueError::new_err(e.to_string())));
                    }
                    Response::Chunk(response) => {
                        this.observe_chunk(&response);
                        return Some(Ok(response));
                    }
                    Response::Done(_) => unreachable!(),
                    Response::CompletionDone(_) => unreachable!(),
                    Response::CompletionModelError(_, _) => unreachable!(),
                    Response::CompletionChunk(_) => unreachable!(),
                    Response::ImageGeneration(_) => unreachable!(),
                    Response::Speech { .. } => unreachable!(),
                    Response::Raw { .. } => unreachable!(),
                    Response::Embeddings { .. } => unreachable!(),
                },
                None => {
                    this.finish();
                    return Some(Err(PyValueError::new_err(
                        "Received none in ChatCompletionStreamer".to_string(),
                    )));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs_core::{ChatAudioResponse, ChunkChoice, Delta};
    use tokio::sync::mpsc::channel;

    fn chunk(choices: &[(usize, bool)]) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: "response".to_string(),
            choices: choices
                .iter()
                .map(|&(index, finished)| ChunkChoice {
                    finish_reason: finished.then(|| "stop".to_string()),
                    stop_sequence: None,
                    index,
                    delta: Delta {
                        audio: Some(ChatAudioResponse {
                            id: Some(format!("audio_{index}")),
                            data: Some("AQI=".to_string()),
                            expires_at: None,
                            transcript: None,
                        }),
                        content: None,
                        role: "assistant".to_string(),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                })
                .collect(),
            created: 0,
            model: "test".to_string(),
            system_fingerprint: String::new(),
            object: "chat.completion.chunk".to_string(),
            usage: None,
            adapter_generation: None,
            session_id: None,
        }
    }

    #[test]
    fn waits_for_every_choice_without_counting_repeated_finishes() {
        let (tx, rx) = channel(1);
        let mut streamer = ChatCompletionStreamer::from_rx(rx, 2);
        streamer.observe_chunk(&chunk(&[]));
        assert!(!streamer.is_done);
        streamer.observe_chunk(&chunk(&[(0, true)]));
        streamer.observe_chunk(&chunk(&[(0, true), (1, false)]));
        assert!(!streamer.is_done);
        assert!(!tx.is_closed());

        let final_chunk = chunk(&[(1, true)]);
        streamer.observe_chunk(&final_chunk);
        assert!(streamer.is_done);
        assert!(tx.is_closed());
        assert_eq!(
            final_chunk.choices[0]
                .delta
                .audio
                .as_ref()
                .unwrap()
                .data
                .as_deref(),
            Some("AQI=")
        );
    }

    #[test]
    fn accepts_all_choices_finishing_in_one_chunk() {
        let (_, rx) = channel(1);
        let mut streamer = ChatCompletionStreamer::from_rx(rx, 2);
        streamer.observe_chunk(&chunk(&[(1, true), (0, true)]));
        assert!(streamer.is_done);
    }
}
