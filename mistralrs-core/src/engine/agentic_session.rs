use std::collections::HashMap;
use std::io::Cursor;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use either::Either;
use image::{DynamicImage, ImageFormat};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{MessageContent, NormalRequest, RequestMessage, VideoInput};

const MAX_SESSIONS: usize = 128;
const SESSION_TTL: Duration = Duration::from_secs(30 * 60); // 30 minutes

/// A stored agentic conversation including tool call/response messages.
#[derive(Clone)]
pub struct AgenticSessionEntry {
    /// Full message history including tool call/response messages.
    pub messages: Vec<IndexMap<String, MessageContent>>,
    /// Images from multimodal tool responses (positional with the messages).
    pub images: Vec<DynamicImage>,
    /// Videos from multimodal tool responses.
    pub videos: Vec<VideoInput>,
    /// Last access time for eviction.
    last_accessed: Instant,
}

/// Server-side store for agentic conversation state.
///
/// Keyed by session ID. Supports both explicit session ID lookup and
/// implicit content-based matching for clients that don't pass a session ID.
pub struct AgenticSessionStore {
    sessions: HashMap<String, AgenticSessionEntry>,
}

impl AgenticSessionStore {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Look up a session by explicit ID. Updates `last_accessed`.
    pub fn get(&mut self, session_id: &str) -> Option<AgenticSessionEntry> {
        let entry = self.sessions.get_mut(session_id)?;
        entry.last_accessed = Instant::now();
        Some(entry.clone())
    }

    /// Find a session by matching user-visible messages as a prefix.
    ///
    /// Extracts user-visible messages (skipping `role: "tool"` and messages
    /// with `tool_calls`) from each stored session. If the incoming messages
    /// match as a prefix, returns the session ID and entry.
    pub fn find_by_messages(
        &mut self,
        incoming: &[IndexMap<String, MessageContent>],
    ) -> Option<(String, AgenticSessionEntry)> {
        // Need at least 2 messages (system/user + assistant) to match meaningfully.
        if incoming.len() < 2 {
            return None;
        }

        for (id, entry) in &mut self.sessions {
            let stored_visible = user_visible_messages(&entry.messages);

            // The incoming messages should be a prefix of (or equal to + new)
            // the stored user-visible messages. We check that all stored
            // user-visible messages match the beginning of incoming.
            if stored_visible.len() > incoming.len() {
                continue;
            }

            let matches = stored_visible
                .iter()
                .zip(incoming.iter())
                .all(|(stored, inc)| messages_match(stored, inc));

            if matches && !stored_visible.is_empty() {
                entry.last_accessed = Instant::now();
                return Some((id.clone(), entry.clone()));
            }
        }

        None
    }

    /// Save or update a session. Evicts stale entries if needed.
    pub fn save(&mut self, session_id: String, entry: AgenticSessionEntry) {
        self.evict();
        self.sessions.insert(session_id, entry);
    }

    /// Delete a session. Returns whether the session existed.
    pub fn delete(&mut self, session_id: &str) -> bool {
        self.sessions.remove(session_id).is_some()
    }

    /// Snapshot of all session IDs currently stored.
    pub fn list_ids(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    /// Export a session to its serializable representation.
    pub fn export(&mut self, session_id: &str) -> Result<Option<SerializedSession>> {
        let Some(entry) = self.get(session_id) else {
            return Ok(None);
        };
        Ok(Some(SerializedSession::from_entry(&entry)?))
    }

    /// Import a serialized session, replacing any existing entry with the same ID.
    pub fn import(&mut self, session_id: String, serialized: SerializedSession) -> Result<()> {
        let entry = serialized.into_entry()?;
        self.save(session_id, entry);
        Ok(())
    }

    /// Remove sessions that are expired or over the limit.
    fn evict(&mut self) {
        let now = Instant::now();

        // Remove expired sessions.
        self.sessions
            .retain(|_, entry| now.duration_since(entry.last_accessed) < SESSION_TTL);

        // If still over limit, remove oldest.
        while self.sessions.len() >= MAX_SESSIONS {
            let oldest = self
                .sessions
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest {
                self.sessions.remove(&key);
            } else {
                break;
            }
        }
    }
}

impl AgenticSessionEntry {
    pub fn new(
        messages: Vec<IndexMap<String, MessageContent>>,
        images: Vec<DynamicImage>,
        videos: Vec<VideoInput>,
    ) -> Self {
        Self {
            messages,
            images,
            videos,
            last_accessed: Instant::now(),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract user-visible messages from a full conversation, skipping tool
/// call/response messages that the client never sees.
fn user_visible_messages(
    messages: &[IndexMap<String, MessageContent>],
) -> Vec<&IndexMap<String, MessageContent>> {
    messages
        .iter()
        .filter(|msg| !is_tool_message(msg))
        .collect()
}

/// Check if a message is a tool-related message (invisible to clients).
fn is_tool_message(msg: &IndexMap<String, MessageContent>) -> bool {
    let role = msg
        .get("role")
        .and_then(|r| match r {
            Either::Left(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("");

    // Tool response messages.
    if role == "tool" {
        return true;
    }

    // Assistant messages with tool_calls (the "I want to call a tool" message).
    if msg.contains_key("tool_calls") {
        return true;
    }

    false
}

/// Check if two messages match by role and content.
fn messages_match(
    a: &IndexMap<String, MessageContent>,
    b: &IndexMap<String, MessageContent>,
) -> bool {
    a.get("role") == b.get("role") && a.get("content") == b.get("content")
}

// ── Splice ───────────────────────────────────────────────────────────────────

/// Splice stored tool messages back into an incoming request.
///
/// The client sends user-visible messages (user, assistant, system). The
/// stored session has those same messages PLUS interleaved tool call/response
/// messages. This function merges them: for each matched user-visible
/// message, it includes any stored tool messages that follow it.
pub fn splice_session_into_request(request: &mut NormalRequest, entry: &AgenticSessionEntry) {
    let incoming = match &mut request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::MultimodalChat { messages, .. } => {
            messages
        }
        _ => return,
    };

    let stored = &entry.messages;

    // Walk stored messages, matching user-visible ones against incoming.
    let mut result: Vec<IndexMap<String, MessageContent>> = Vec::new();
    let mut incoming_idx = 0;
    let mut stored_idx = 0;

    while stored_idx < stored.len() && incoming_idx < incoming.len() {
        let stored_msg = &stored[stored_idx];

        if is_tool_message(stored_msg) {
            // Tool message from stored history — splice it in.
            result.push(stored_msg.clone());
            stored_idx += 1;
        } else {
            // User-visible message — check if it matches the incoming one.
            let incoming_msg = &incoming[incoming_idx];
            if messages_match(stored_msg, incoming_msg) {
                result.push(stored_msg.clone());
                stored_idx += 1;
                incoming_idx += 1;
            } else {
                // Mismatch — conversation diverged. Stop splicing and
                // append remaining incoming messages as-is.
                break;
            }
        }
    }

    // If we consumed all stored messages, there may be trailing tool
    // messages after the last matched user-visible message.
    while stored_idx < stored.len() && is_tool_message(&stored[stored_idx]) {
        result.push(stored[stored_idx].clone());
        stored_idx += 1;
    }

    // Append any new incoming messages beyond what was matched.
    while incoming_idx < incoming.len() {
        result.push(incoming[incoming_idx].clone());
        incoming_idx += 1;
    }

    *incoming = result;

    // Restore images and videos for multimodal tool responses.
    if !entry.images.is_empty() || !entry.videos.is_empty() {
        super::agentic_loop::upgrade_to_multimodal(request);
        if !entry.images.is_empty() {
            let req_images = super::agentic_loop::get_images_mut(request);
            *req_images = entry.images.clone();
        }
        if !entry.videos.is_empty() {
            let req_videos = super::agentic_loop::get_videos_mut(request);
            *req_videos = entry.videos.clone();
        }
    }
}

// ── Serialization ────────────────────────────────────────────────────────────

/// Wire format for an agentic session. Images are encoded as base64 PNG.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SerializedSession {
    #[cfg_attr(feature = "utoipa", schema(value_type = Vec<serde_json::Value>))]
    pub messages: Vec<IndexMap<String, MessageContent>>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub videos: Vec<SerializedVideo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SerializedVideo {
    pub fps: f64,
    /// One base64 PNG per sampled frame.
    pub frames: Vec<String>,
    pub total_num_frames: usize,
    pub sampled_indices: Vec<usize>,
}

impl SerializedSession {
    /// Build a serializable view of an in-memory session.
    pub fn from_entry(entry: &AgenticSessionEntry) -> Result<Self> {
        let images = entry
            .images
            .iter()
            .map(encode_png_base64)
            .collect::<Result<Vec<_>>>()?;

        let videos = entry
            .videos
            .iter()
            .map(SerializedVideo::from_video)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            messages: entry.messages.clone(),
            images,
            videos,
        })
    }

    /// Reconstruct an in-memory session entry.
    pub fn into_entry(self) -> Result<AgenticSessionEntry> {
        let images = self
            .images
            .iter()
            .map(|s| decode_png_base64(s))
            .collect::<Result<Vec<_>>>()?;

        let videos = self
            .videos
            .into_iter()
            .map(SerializedVideo::into_video)
            .collect::<Result<Vec<_>>>()?;

        Ok(AgenticSessionEntry {
            messages: self.messages,
            images,
            videos,
            last_accessed: Instant::now(),
        })
    }
}

impl SerializedVideo {
    fn from_video(video: &VideoInput) -> Result<Self> {
        let frames = video
            .frames
            .iter()
            .map(encode_png_base64)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fps: video.fps,
            frames,
            total_num_frames: video.total_num_frames,
            sampled_indices: video.sampled_indices.clone(),
        })
    }

    fn into_video(self) -> Result<VideoInput> {
        let frames = self
            .frames
            .iter()
            .map(|s| decode_png_base64(s))
            .collect::<Result<Vec<_>>>()?;
        Ok(VideoInput {
            frames,
            fps: self.fps,
            total_num_frames: self.total_num_frames,
            sampled_indices: self.sampled_indices,
        })
    }
}

fn encode_png_base64(img: &DynamicImage) -> Result<String> {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .context("encoding image as PNG")?;
    Ok(BASE64.encode(&buf))
}

fn decode_png_base64(s: &str) -> Result<DynamicImage> {
    let bytes = BASE64.decode(s).context("base64 decoding image")?;
    image::load_from_memory(&bytes).context("loading image bytes")
}
