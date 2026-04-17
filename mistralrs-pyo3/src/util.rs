use std::{
    cell::RefCell,
    fs::{self, File},
    io::{Cursor, Read},
    sync::{Arc, Mutex},
};

use either::Either;
use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, DynamicImage};
use mistralrs_core::{
    sample_frame_indices, AudioInput, ChatCompletionResponse, CompletionResponse, MistralRs,
    Request, Response, ResponseErr, VideoInput,
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

/// Default number of frames to sample from a video.
const DEFAULT_NUM_FRAMES: usize = 32;

/// Default FPS assumed when metadata is unavailable.
const DEFAULT_FPS: f64 = 24.0;

/// Parses and loads a video from a URL, file path, or data URL.
/// Mirrors `parse_image_url`/`parse_audio_url` but returns a [`VideoInput`].
///
/// GIF files are decoded with the `image` crate. All other formats require
/// FFmpeg on `$PATH`.
pub(crate) fn parse_video_url(url_unparsed: &str) -> PyApiResult<VideoInput> {
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

    let lower = url_unparsed.to_lowercase();
    let is_gif = lower.ends_with(".gif")
        || lower.contains("image/gif")
        || (bytes.len() >= 6 && &bytes[..6] == b"GIF89a")
        || (bytes.len() >= 6 && &bytes[..6] == b"GIF87a");

    if is_gif {
        decode_gif_frames(&bytes).map_err(|e| PyApiErr::from(format!("{e}")))
    } else {
        decode_video_ffmpeg(&bytes, url_unparsed).map_err(|e| PyApiErr::from(format!("{e}")))
    }
}

fn decode_gif_frames(bytes: &[u8]) -> anyhow::Result<VideoInput> {
    let decoder = GifDecoder::new(Cursor::new(bytes))
        .map_err(|e| anyhow::anyhow!("Failed to decode GIF: {e}"))?;

    let raw_frames: Vec<_> = decoder
        .into_frames()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let total = raw_frames.len();
    if total == 0 {
        anyhow::bail!("GIF contains no frames");
    }

    let total_delay_ms: u32 = raw_frames
        .iter()
        .map(|f| {
            let (num, den) = f.delay().numer_denom_ms();
            (num * 1000).checked_div(den).unwrap_or(100)
        })
        .sum();
    let fps = if total_delay_ms > 0 {
        (total as f64 * 1000.0) / total_delay_ms as f64
    } else {
        DEFAULT_FPS
    };

    let indices = sample_frame_indices(total, DEFAULT_NUM_FRAMES);
    let frames: Vec<DynamicImage> = indices
        .iter()
        .map(|&i| DynamicImage::ImageRgba8(raw_frames[i].buffer().clone()))
        .collect();

    Ok(VideoInput {
        frames,
        fps,
        total_num_frames: total,
        sampled_indices: indices,
    })
}

fn decode_video_ffmpeg(bytes: &[u8], source_hint: &str) -> anyhow::Result<VideoInput> {
    let ffmpeg_ok = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok();

    if !ffmpeg_ok {
        anyhow::bail!(
            "Cannot decode video '{}': FFmpeg not found.\n\
             FFmpeg is required for video input (non-GIF formats). Install it:\n\
             - Linux:   apt install ffmpeg  /  dnf install ffmpeg\n\
             - macOS:   brew install ffmpeg\n\
             - Windows: https://ffmpeg.org/download.html",
            source_hint,
        );
    }

    let tmp_dir = std::env::temp_dir().join("mistralrs_video");
    fs::create_dir_all(&tmp_dir)?;
    let video_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let input_path = tmp_dir.join(format!("{video_id}.video"));
    fs::write(&input_path, bytes)?;

    let (fps, total_frames) = probe_video_blocking(&input_path).unwrap_or((DEFAULT_FPS, 0));

    let effective_total = if total_frames > 0 {
        total_frames
    } else {
        DEFAULT_NUM_FRAMES
    };

    let indices = sample_frame_indices(effective_total, DEFAULT_NUM_FRAMES);
    let n = indices.len();

    let out_dir = tmp_dir.join(format!("{video_id}_frames"));
    fs::create_dir_all(&out_dir)?;

    let select_expr = indices
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-i",
            input_path.to_str().unwrap(),
            "-vf",
            &format!("select='{select_expr}'"),
            "-vsync",
            "vfr",
            "-frames:v",
            &n.to_string(),
            &format!("{}/frame_%04d.png", out_dir.display()),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;

    if !status.success() {
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_dir_all(&out_dir);
        anyhow::bail!(
            "FFmpeg failed to extract frames from '{}' (exit code: {:?})",
            source_hint,
            status.code()
        );
    }

    let mut frames = Vec::with_capacity(n);
    for i in 1..=n {
        let frame_path = out_dir.join(format!("frame_{i:04}.png"));
        if frame_path.exists() {
            let frame_bytes = fs::read(&frame_path)?;
            let img = image::load_from_memory(&frame_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to load extracted frame {i}: {e}"))?;
            frames.push(img);
        }
    }

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_dir_all(&out_dir);

    if frames.is_empty() {
        anyhow::bail!(
            "FFmpeg extracted 0 frames from '{}'. The file may be corrupt or empty.",
            source_hint
        );
    }

    let actual_indices = if frames.len() < indices.len() {
        sample_frame_indices(effective_total, frames.len())
    } else {
        indices
    };

    Ok(VideoInput {
        frames,
        fps,
        total_num_frames: effective_total,
        sampled_indices: actual_indices,
    })
}

fn probe_video_blocking(path: &std::path::Path) -> anyhow::Result<(f64, usize)> {
    let fps_output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_str().unwrap(),
        ])
        .output()?;

    let fps_str = String::from_utf8_lossy(&fps_output.stdout);
    let fps = parse_fps_fraction(fps_str.trim()).unwrap_or(DEFAULT_FPS);

    let count_output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_str().unwrap(),
        ])
        .output()?;

    let count_str = String::from_utf8_lossy(&count_output.stdout);
    let total_frames: usize = count_str.trim().parse().unwrap_or(0);

    Ok((fps, total_frames))
}

fn parse_fps_fraction(s: &str) -> Option<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.parse().ok()?;
        let d: f64 = den.parse().ok()?;
        if d > 0.0 {
            Some(n / d)
        } else {
            None
        }
    } else {
        s.parse().ok()
    }
}
