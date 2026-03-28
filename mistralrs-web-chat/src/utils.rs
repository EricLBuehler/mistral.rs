use std::path::PathBuf;

/// Determine the base cache directory for the application.
/// Uses XDG_CACHE_HOME or falls back to ~/.cache/mistralrs-web-chat.
pub fn get_cache_dir() -> PathBuf {
    // XDG_CACHE_HOME or default to ~/.cache
    let cache_home = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| PathBuf::from(".cache"))
        });
    cache_home.join("mistralrs-web-chat")
}

/// Check if a chat ID is safe to use in a filename.
pub fn is_chat_id_safe(id: &str) -> bool {
    if id.is_empty() {
        return false;
    }
    id.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_chat_id_safe() {
        assert!(is_chat_id_safe("chat_1"));
        assert!(is_chat_id_safe("chat-2"));
        assert!(is_chat_id_safe("chat_3-4"));
        assert!(!is_chat_id_safe("../etc/passwd"));
        assert!(!is_chat_id_safe("..\\windows\\system32"));
        assert!(!is_chat_id_safe("chat.json"));
        assert!(!is_chat_id_safe("chat 1"));
        assert!(!is_chat_id_safe("chat/1"));
        assert!(!is_chat_id_safe(""));
    }
}
