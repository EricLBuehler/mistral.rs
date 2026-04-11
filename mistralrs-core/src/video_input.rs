use image::DynamicImage;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Decoded video input: a sequence of frames with metadata for timestamp generation.
///
/// Create from pre-decoded frames with [`VideoInput::from_frames`], or use the
/// server-core `parse_video_url` helper to decode from a video file (requires FFmpeg
/// for non-GIF formats).
#[derive(Clone, PartialEq)]
pub struct VideoInput {
    /// Decoded video frames (RGB images).
    pub frames: Vec<DynamicImage>,
    /// Frames per second of the *original* video. Used to compute per-frame
    /// timestamps for the prompt (e.g. `"00:05"`). Defaults to 24.0.
    pub fps: f64,
    /// Total number of frames in the original video before sampling.
    pub total_num_frames: usize,
    /// Indices of the frames that were sampled from the original video.
    /// Length must equal `frames.len()`.
    pub sampled_indices: Vec<usize>,
}

impl std::fmt::Debug for VideoInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoInput")
            .field("num_frames", &self.frames.len())
            .field("fps", &self.fps)
            .field("total_num_frames", &self.total_num_frames)
            .finish()
    }
}

impl VideoInput {
    /// Create a `VideoInput` from pre-decoded frames.
    ///
    /// `fps` is the original video frame rate (used for timestamp generation).
    /// If the frames were not sampled (i.e. all frames are provided), pass `None`
    /// for `sampled_indices` and they will default to `0..frames.len()`.
    pub fn from_frames(
        frames: Vec<DynamicImage>,
        fps: f64,
        sampled_indices: Option<Vec<usize>>,
    ) -> Self {
        let n = frames.len();
        let sampled_indices = sampled_indices.unwrap_or_else(|| (0..n).collect());
        Self {
            frames,
            fps,
            total_num_frames: *sampled_indices.last().unwrap_or(&0) + 1,
            sampled_indices,
        }
    }

    /// Compute per-frame timestamps in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn timestamps_secs(&self) -> Vec<f64> {
        self.sampled_indices
            .iter()
            .map(|&idx| idx as f64 / self.fps)
            .collect()
    }

    /// Format timestamps as `"mm:ss"` strings.
    #[allow(clippy::cast_possible_truncation)]
    pub fn timestamp_strings(&self) -> Vec<String> {
        self.timestamps_secs()
            .iter()
            .map(|&secs| {
                let minutes = (secs / 60.0) as u32;
                let seconds = (secs % 60.0) as u32;
                format!("{minutes:02}:{seconds:02}")
            })
            .collect()
    }

    /// Compute a content hash for each frame (for prefix caching).
    pub fn frame_hashes(&self) -> Vec<u64> {
        self.frames
            .iter()
            .map(|img| {
                let mut hasher = DefaultHasher::new();
                img.as_bytes().hash(&mut hasher);
                hasher.finish()
            })
            .collect()
    }

    /// Compute a single hash representing the entire video (for prefix caching).
    pub fn video_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for frame in &self.frames {
            frame.as_bytes().hash(&mut hasher);
        }
        self.fps.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

/// Sample `num_frames` frame indices uniformly from a video with `total_frames` frames.
///
/// Matches the HF reference: `torch.arange(0, total, total / num_frames).int()`
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn sample_frame_indices(total_frames: usize, num_frames: usize) -> Vec<usize> {
    if num_frames == 0 || total_frames == 0 {
        return Vec::new();
    }
    let n = num_frames.min(total_frames);
    (0..n)
        .map(|i| ((i as f64) * (total_frames as f64) / (n as f64)) as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_frame_indices() {
        let indices = sample_frame_indices(96, 32);
        assert_eq!(indices.len(), 32);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 3);
        assert_eq!(indices[31], 93);
    }

    #[test]
    fn test_sample_frame_indices_equal() {
        let indices = sample_frame_indices(5, 5);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sample_frame_indices_more_than_total() {
        let indices = sample_frame_indices(3, 10);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_timestamp_strings() {
        let vi = VideoInput {
            frames: Vec::new(),
            fps: 24.0,
            total_num_frames: 2880,
            sampled_indices: vec![0, 720, 1440, 2160],
        };
        let ts = vi.timestamp_strings();
        assert_eq!(ts, vec!["00:00", "00:30", "01:00", "01:30"]);
    }
}
