//! In-process file store keyed by id, with per-entry TTL.
//!
//! Holds [`File`] bodies for the lifetime of:
//! - the HTTP `GET /v1/files/{id}` fetch endpoint;
//! - `read_file` / `list_files` model-tool dispatch;
//! - any SDK consumer that needs to resolve a truncated-on-the-wire
//!   file to its bytes.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use super::File;

/// Default per-entry TTL. Matches the agentic session default.
pub const DEFAULT_FILE_TTL: Duration = Duration::from_secs(30 * 60);

struct StoredFile {
    file: Arc<File>,
    expires_at: Instant,
    /// Optional session id this file belongs to; `None` for runs without
    /// a session. `list_files` filters by session id.
    session_id: Option<String>,
    /// Insertion order, used by `list_for_session` to return oldest-first.
    seq: u64,
}

/// Shared, thread-safe file store.
#[derive(Clone)]
pub struct FileStore {
    inner: Arc<RwLock<Inner>>,
    ttl: Duration,
}

struct Inner {
    by_id: HashMap<String, StoredFile>,
    next_seq: u64,
}

impl FileStore {
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_FILE_TTL)
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                by_id: HashMap::new(),
                next_seq: 0,
            })),
            ttl,
        }
    }

    /// Insert a file. Replaces any entry with the same id.
    pub fn insert(&self, file: File, session_id: Option<String>) {
        let id = file.id.clone();
        let mut guard = self.inner.write().unwrap();
        let seq = guard.next_seq;
        guard.next_seq += 1;
        guard.by_id.insert(
            id,
            StoredFile {
                file: Arc::new(file),
                expires_at: Instant::now() + self.ttl,
                session_id,
                seq,
            },
        );
    }

    /// Get a file by id if present and not expired.
    pub fn get(&self, id: &str) -> Option<Arc<File>> {
        let now = Instant::now();
        let guard = self.inner.read().unwrap();
        let entry = guard.by_id.get(id)?;
        if entry.expires_at < now {
            None
        } else {
            Some(Arc::clone(&entry.file))
        }
    }

    /// Remove a file by id. Returns true if an entry existed.
    pub fn remove(&self, id: &str) -> bool {
        self.inner.write().unwrap().by_id.remove(id).is_some()
    }

    /// Refresh the TTL on every file tagged with the given session id.
    /// Call when the session is touched (start of a new turn, save) so
    /// long-running sessions don't lose their files to expiry.
    pub fn touch_session(&self, session_id: &str) {
        let new_expiry = std::time::Instant::now() + self.ttl;
        let mut guard = self.inner.write().unwrap();
        for entry in guard.by_id.values_mut() {
            if entry.session_id.as_deref() == Some(session_id) {
                entry.expires_at = new_expiry;
            }
        }
    }

    /// All non-expired files associated with a session, oldest first.
    pub fn list_for_session(&self, session_id: &str) -> Vec<Arc<File>> {
        let now = Instant::now();
        let guard = self.inner.read().unwrap();
        let mut hits: Vec<&StoredFile> = guard
            .by_id
            .values()
            .filter(|s| s.expires_at >= now && s.session_id.as_deref() == Some(session_id))
            .collect();
        hits.sort_by_key(|s| s.seq);
        hits.into_iter().map(|s| Arc::clone(&s.file)).collect()
    }

    /// Drop all expired entries. Safe to call periodically.
    pub fn cleanup_expired(&self) -> usize {
        let now = Instant::now();
        let mut guard = self.inner.write().unwrap();
        let before = guard.by_id.len();
        guard.by_id.retain(|_, entry| entry.expires_at >= now);
        before - guard.by_id.len()
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().by_id.is_empty()
    }
}

impl Default for FileStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{FileContent, FileSource};

    fn make(id: &str) -> File {
        File {
            id: id.into(),
            name: format!("{id}.txt"),
            format: Some("txt".into()),
            mime_type: Some("text/plain".into()),
            bytes: 2,
            source: FileSource {
                tool: "execute_python".into(),
                round: 0,
                turn: 0,
            },
            content: FileContent::Text {
                text: Some("hi".into()),
                preview: None,
            },
        }
    }

    #[test]
    fn insert_and_get() {
        let s = FileStore::new();
        s.insert(make("file_a"), None);
        assert_eq!(s.get("file_a").unwrap().as_text(), Some("hi"));
        assert!(s.get("missing").is_none());
    }

    #[test]
    fn list_by_session_oldest_first() {
        let s = FileStore::new();
        s.insert(make("file_a"), Some("sess1".into()));
        s.insert(make("file_b"), Some("sess1".into()));
        s.insert(make("file_c"), Some("sess2".into()));
        let list: Vec<_> = s
            .list_for_session("sess1")
            .iter()
            .map(|f| f.id.clone())
            .collect();
        assert_eq!(list, vec!["file_a".to_string(), "file_b".to_string()]);
        let list2: Vec<_> = s
            .list_for_session("sess2")
            .iter()
            .map(|f| f.id.clone())
            .collect();
        assert_eq!(list2, vec!["file_c".to_string()]);
    }

    #[test]
    fn ttl_eviction() {
        let s = FileStore::with_ttl(Duration::from_millis(1));
        s.insert(make("file_a"), None);
        std::thread::sleep(Duration::from_millis(5));
        assert!(s.get("file_a").is_none());
        let swept = s.cleanup_expired();
        assert_eq!(swept, 1);
        assert!(s.is_empty());
    }
}
