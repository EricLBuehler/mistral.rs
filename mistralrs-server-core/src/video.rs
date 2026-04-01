//! Video frame extraction: FFmpeg for general formats, `image` crate for GIFs.
//!
//! Both the HTTP server and the CLI use this module to decode video files
//! into frames suitable for multimodal model input.
//!
//! ## FFmpeg requirement
//!
//! For non-GIF formats (mp4, avi, mov, mkv, webm, etc.) the `ffmpeg` binary
//! must be installed and available on `$PATH`. If it is not found the module
//! falls back to GIF-only support via the `image` crate.
//!
//! Install FFmpeg:
//! - **Linux**: `apt install ffmpeg` / `dnf install ffmpeg`
//! - **macOS**: `brew install ffmpeg`
//! - **Windows**: <https://ffmpeg.org/download.html>
//!
//! See <https://ericlbuehler.github.io/mistral.rs/VIDEO/> for full details.

use anyhow::{bail, Context, Result};
use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, DynamicImage};
use mistralrs_core::{sample_frame_indices, VideoInput};
use std::io::Cursor;
use std::path::Path;
use tokio::{
    fs::{self, File},
    io::AsyncReadExt,
};

/// Default number of frames to sample from a video.
const DEFAULT_NUM_FRAMES: usize = 32;

/// Default frames-per-second assumed when metadata is unavailable (e.g. GIF).
const DEFAULT_FPS: f64 = 24.0;

const FFMPEG_INSTALL_HELP: &str = "\
FFmpeg is required for video input (non-GIF formats). Install it:
  - Linux:   apt install ffmpeg  /  dnf install ffmpeg
  - macOS:   brew install ffmpeg
  - Windows: https://ffmpeg.org/download.html
See https://ericlbuehler.github.io/mistral.rs/VIDEO/ for details.";

/// Fetch video bytes from a URL, file path, or data URL, then decode into
/// a [`VideoInput`] (sampled frames + metadata).
///
/// Supports:
/// - HTTP/HTTPS URLs
/// - Local file paths (absolute or relative)
/// - `file://` URLs
/// - `data:video/...;base64,...` data URLs
///
/// GIF files are decoded with the `image` crate. All other formats require
/// FFmpeg.
pub async fn parse_video_url(
    url_unparsed: &str,
    num_frames: Option<usize>,
) -> Result<VideoInput> {
    let num_frames = num_frames.unwrap_or(DEFAULT_NUM_FRAMES);

    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).await.is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url_unparsed))?
    } else {
        bail!(
            "Invalid video source '{}': not a valid URL (http/https/data) and file not found. \
             Use a full URL, a data URL, or an absolute file path.",
            url_unparsed
        )
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        let resp = reqwest::get(url.clone())
            .await
            .context(format!("Failed to fetch video: {url}"))?;
        resp.bytes().await?.to_vec()
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file path: {}", url))?;
        let mut f = File::open(&path)
            .await
            .context(format!("Could not open video file: {}", path.display()))?;
        let metadata = fs::metadata(&path).await?;
        let mut buffer = vec![0; metadata.len() as usize];
        f.read_exact(&mut buffer).await?;
        buffer
    } else if url.scheme() == "data" {
        let data_url = data_url::DataUrl::process(url.as_str())?;
        data_url.decode_to_vec()?.0
    } else {
        bail!("Unsupported URL scheme for video: {}", url.scheme());
    };

    // Detect format
    let lower = url_unparsed.to_lowercase();
    let is_gif = lower.ends_with(".gif")
        || lower.contains("image/gif")
        || (bytes.len() >= 6 && &bytes[..6] == b"GIF89a")
        || (bytes.len() >= 6 && &bytes[..6] == b"GIF87a");

    if is_gif {
        decode_gif_frames(&bytes, num_frames)
    } else {
        decode_video_ffmpeg(&bytes, num_frames, url_unparsed).await
    }
}

/// Decode a GIF into sampled frames using the `image` crate.
fn decode_gif_frames(bytes: &[u8], num_frames: usize) -> Result<VideoInput> {
    let decoder =
        GifDecoder::new(Cursor::new(bytes)).context("Failed to decode GIF")?;

    let raw_frames: Vec<_> = decoder.into_frames().collect::<Result<Vec<_>, _>>()?;
    let total = raw_frames.len();
    if total == 0 {
        bail!("GIF contains no frames");
    }

    // Estimate FPS from average frame delay
    let total_delay_ms: u32 = raw_frames
        .iter()
        .map(|f| {
            let (num, den) = f.delay().numer_denom_ms();
            if den == 0 { 100 } else { num * 1000 / den }
        })
        .sum();
    let fps = if total_delay_ms > 0 {
        (total as f64 * 1000.0) / total_delay_ms as f64
    } else {
        DEFAULT_FPS
    };

    let indices = sample_frame_indices(total, num_frames);
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

/// Decode a video file using FFmpeg subprocess.
///
/// 1. Probe with `ffprobe` for FPS and total frame count.
/// 2. Extract sampled frames with `ffmpeg`.
/// 3. Load frames as images.
async fn decode_video_ffmpeg(
    bytes: &[u8],
    num_frames: usize,
    source_hint: &str,
) -> Result<VideoInput> {
    // Check ffmpeg availability
    let ffmpeg_ok = tokio::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .is_ok();

    if !ffmpeg_ok {
        bail!(
            "Cannot decode video '{}': FFmpeg not found.\n{}",
            source_hint,
            FFMPEG_INSTALL_HELP
        );
    }

    // Write to temp file
    let tmp_dir = std::env::temp_dir().join("mistralrs_video");
    fs::create_dir_all(&tmp_dir).await?;
    let video_id = uuid::Uuid::new_v4();
    let input_path = tmp_dir.join(format!("{video_id}.video"));
    fs::write(&input_path, bytes).await?;

    // Probe video metadata with ffprobe
    let (fps, total_frames) = probe_video(&input_path).await.unwrap_or((DEFAULT_FPS, 0));

    // Determine how many frames to actually sample
    let effective_total = if total_frames > 0 {
        total_frames
    } else {
        // Fallback: just request num_frames frames
        num_frames
    };

    let indices = sample_frame_indices(effective_total, num_frames);
    let n = indices.len();

    // Create output directory
    let out_dir = tmp_dir.join(format!("{video_id}_frames"));
    fs::create_dir_all(&out_dir).await?;

    // Build ffmpeg select filter: select specific frames
    // select='eq(n\,0)+eq(n\,3)+eq(n\,6)+...'
    let select_expr = indices
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");

    let status = tokio::process::Command::new("ffmpeg")
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
        .status()
        .await
        .context("Failed to run ffmpeg")?;

    if !status.success() {
        // Cleanup
        let _ = fs::remove_file(&input_path).await;
        let _ = fs::remove_dir_all(&out_dir).await;
        bail!(
            "FFmpeg failed to extract frames from '{}' (exit code: {:?})",
            source_hint,
            status.code()
        );
    }

    // Load extracted frame images
    let mut frames = Vec::with_capacity(n);
    for i in 1..=n {
        let frame_path = out_dir.join(format!("frame_{i:04}.png"));
        if frame_path.exists() {
            let frame_bytes = fs::read(&frame_path).await?;
            let img = image::load_from_memory(&frame_bytes)
                .context(format!("Failed to load extracted frame {i}"))?;
            frames.push(img);
        }
    }

    // Cleanup temp files
    let _ = fs::remove_file(&input_path).await;
    let _ = fs::remove_dir_all(&out_dir).await;

    if frames.is_empty() {
        bail!(
            "FFmpeg extracted 0 frames from '{}'. The file may be corrupt or empty.",
            source_hint
        );
    }

    // If we got fewer frames than expected (e.g. video shorter than expected),
    // rebuild indices to match actual frame count
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

/// Use `ffprobe` to get FPS and total frame count for a video file.
async fn probe_video(path: &Path) -> Result<(f64, usize)> {
    // Get FPS
    let fps_output = tokio::process::Command::new("ffprobe")
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
        .output()
        .await
        .context("Failed to run ffprobe for FPS")?;

    let fps_str = String::from_utf8_lossy(&fps_output.stdout);
    let fps = parse_fps_fraction(fps_str.trim()).unwrap_or(DEFAULT_FPS);

    // Get total frame count
    let count_output = tokio::process::Command::new("ffprobe")
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
        .output()
        .await
        .context("Failed to run ffprobe for frame count")?;

    let count_str = String::from_utf8_lossy(&count_output.stdout);
    let total_frames: usize = count_str.trim().parse().unwrap_or(0);

    Ok((fps, total_frames))
}

/// Parse a fractional FPS string like "30000/1001" or "30" into f64.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fps_fraction() {
        assert!((parse_fps_fraction("30000/1001").unwrap() - 29.97).abs() < 0.01);
        assert!((parse_fps_fraction("30").unwrap() - 30.0).abs() < 0.01);
        assert!((parse_fps_fraction("24/1").unwrap() - 24.0).abs() < 0.01);
        assert!(parse_fps_fraction("").is_none());
        assert!(parse_fps_fraction("abc").is_none());
    }
}
