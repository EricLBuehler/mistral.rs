use std::{
    cell::RefCell,
    fs::{self, File},
    io::Read,
    sync::{Arc, Mutex},
};

use either::Either;
use image::DynamicImage;
use mistralrs_core::{
    AudioInput, ChatCompletionResponse, CompletionResponse, MistralRs, Request, Response,
    ResponseErr,
};
use pyo3::{exceptions::PyValueError, PyErr};
use tokio::sync::mpsc::Receiver;

static NEXT_REQUEST_ID: Mutex<RefCell<usize>> = Mutex::new(RefCell::new(0));

pub(crate) struct PyApiErr(pub(crate) PyErr);
pub(crate) type PyApiResult<T> = Result<T, PyApiErr>;

impl std::fmt::Debug for PyApiErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::fmt::Display for PyApiErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for PyApiErr {}

impl From<reqwest::Error> for PyApiErr {
    fn from(value: reqwest::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<std::io::Error> for PyApiErr {
    fn from(value: std::io::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<anyhow::Error> for PyApiErr {
    fn from(value: anyhow::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<&candle_core::Error> for PyApiErr {
    fn from(value: &candle_core::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<serde_json::Error> for PyApiErr {
    fn from(value: serde_json::Error) -> Self {
        Self::from(value.to_string())
    }
}

impl From<mistralrs_core::MistralRsError> for PyApiErr {
    fn from(value: mistralrs_core::MistralRsError) -> Self {
        Self::from(value.to_string())
    }
}

impl From<String> for PyApiErr {
    fn from(value: String) -> Self {
        Self(PyValueError::new_err(value.to_string()))
    }
}

impl From<&str> for PyApiErr {
    fn from(value: &str) -> Self {
        Self(PyValueError::new_err(value.to_string()))
    }
}

impl From<PyApiErr> for PyErr {
    fn from(value: PyApiErr) -> Self {
        value.0
    }
}

impl From<Box<ResponseErr>> for PyApiErr {
    fn from(value: Box<ResponseErr>) -> Self {
        Self(PyValueError::new_err(value.to_string()))
    }
}

pub(crate) fn next_request_id() -> usize {
    let next_id = NEXT_REQUEST_ID.lock().unwrap();
    let last = &mut *next_id.borrow_mut();
    let id = *last;
    *last += 1;
    id
}

pub(crate) fn send_request_with_optional_stream(
    runner: Arc<MistralRs>,
    model_id: Option<String>,
    request: Request,
    mut rx: Receiver<Response>,
    debug_repr: String,
    is_streaming: bool,
) -> Result<Either<Response, Receiver<Response>>, String> {
    MistralRs::maybe_log_request(runner.clone(), debug_repr);
    let sender = runner
        .get_sender(model_id.as_deref())
        .map_err(|e| e.to_string())?;
    sender.blocking_send(request).map_err(|e| e.to_string())?;

    if is_streaming {
        Ok(Either::Right(rx))
    } else {
        rx.blocking_recv()
            .ok_or_else(|| "Response channel closed unexpectedly".to_string())
            .map(Either::Left)
    }
}

pub(crate) fn send_request_and_wait(
    runner: Arc<MistralRs>,
    model_id: Option<String>,
    request: Request,
    rx: Receiver<Response>,
    debug_repr: String,
) -> Result<Response, String> {
    match send_request_with_optional_stream(runner, model_id, request, rx, debug_repr, false)? {
        Either::Left(response) => Ok(response),
        Either::Right(_) => unreachable!("non-streaming requests must return a single response"),
    }
}

pub(crate) fn parse_chat_response(response: Response) -> PyApiResult<ChatCompletionResponse> {
    match response {
        Response::ValidationError(e) | Response::InternalError(e) => {
            Err(PyApiErr::from(e.to_string()))
        }
        Response::Done(response) => Ok(response),
        Response::ModelError(msg, _) => Err(PyApiErr::from(msg.to_string())),
        Response::Chunk(_) => unreachable!(),
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionModelError(_, _) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::ImageGeneration(_) => unreachable!(),
        Response::Speech { .. } => unreachable!(),
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
    }
}

pub(crate) fn parse_completion_response(response: Response) -> PyApiResult<CompletionResponse> {
    match response {
        Response::ValidationError(e) | Response::InternalError(e) => {
            Err(PyApiErr::from(e.to_string()))
        }
        Response::CompletionDone(response) => Ok(response),
        Response::CompletionModelError(msg, _) => Err(PyApiErr::from(msg.to_string())),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::ImageGeneration(_) => unreachable!(),
        Response::Speech { .. } => unreachable!(),
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
    }
}

pub(crate) fn parse_embedding_response(response: Response) -> Result<Vec<f32>, String> {
    match response {
        Response::Embeddings { embeddings, .. } => Ok(embeddings),
        Response::ValidationError(e) | Response::InternalError(e) => Err(e.to_string()),
        Response::ModelError(msg, _) => Err(msg.to_string()),
        _ => Err("Received unexpected response type from embeddings request.".to_string()),
    }
}

pub(crate) fn parse_image_url(url_unparsed: &str) -> PyApiResult<DynamicImage> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| format!("Could not parse file path: {url_unparsed}"))?
    } else {
        url::Url::parse(url_unparsed).map_err(|_| {
            format!(
                "Invalid source '{}': not a valid URL (http/https/data) and file not found. \
                 Use a full URL, a data URL, or a file path that exists.",
                url_unparsed
            )
        })?
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        // Read from http
        match reqwest::blocking::get(url.clone()) {
            Ok(http_resp) => http_resp.bytes()?.to_vec(),
            Err(e) => return Err(PyApiErr::from(format!("{e}"))),
        }
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| format!("Could not parse file path: {url}"))?;

        if let Ok(mut f) = File::open(&path) {
            // Read from local file
            let metadata = fs::metadata(&path)?;
            let mut buffer = vec![0; metadata.len() as usize];
            f.read_exact(&mut buffer)?;
            buffer
        } else {
            return Err(PyApiErr::from(format!(
                "Could not open file at path: {url}"
            )));
        }
    } else if url.scheme() == "data" {
        // Decode with base64
        let data_url = data_url::DataUrl::process(url.as_str()).map_err(|e| format!("{e}"))?;
        data_url.decode_to_vec().map_err(|e| format!("{e}"))?.0
    } else {
        return Err(PyApiErr::from(format!(
            "Unsupported URL scheme: {}",
            url.scheme()
        )));
    };

    image::load_from_memory(&bytes).map_err(|e| PyApiErr::from(format!("{e}")))
}

/// Parses and loads an audio file from a URL, file path, or data URL.
/// Mirrors `parse_image_url` but returns an `AudioInput`.
pub(crate) fn parse_audio_url(url_unparsed: &str) -> PyApiResult<AudioInput> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| format!("Could not parse file path: {url_unparsed}"))?
    } else {
        url::Url::parse(url_unparsed).map_err(|_| {
            format!(
                "Invalid source '{}': not a valid URL (http/https/data) and file not found. \
                 Use a full URL, a data URL, or a file path that exists.",
                url_unparsed
            )
        })?
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        match reqwest::blocking::get(url.clone()) {
            Ok(http_resp) => http_resp
                .bytes()
                .map_err(|e| PyApiErr::from(format!("{e}")))?
                .to_vec(),
            Err(e) => return Err(PyApiErr::from(format!("{e}"))),
        }
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| format!("Could not parse file path: {url}"))?;

        if let Ok(mut f) = File::open(&path) {
            let metadata = fs::metadata(&path)?;
            let mut buffer = vec![0; metadata.len() as usize];
            f.read_exact(&mut buffer)?;
            buffer
        } else {
            return Err(PyApiErr::from(format!(
                "Could not open file at path: {url}"
            )));
        }
    } else if url.scheme() == "data" {
        let data_url = data_url::DataUrl::process(url.as_str()).map_err(|e| format!("{e}"))?;
        data_url.decode_to_vec().map_err(|e| format!("{e}"))?.0
    } else {
        return Err(PyApiErr::from(format!(
            "Unsupported URL scheme: {}",
            url.scheme()
        )));
    };

    AudioInput::from_bytes(&bytes).map_err(|e| PyApiErr::from(format!("{e}")))
}
