

fn validate_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

#[test]
fn test_path_traversal() {
    let chats_dir = "/tmp/chats";
    let chat_id = "../../etc/passwd";

    if !validate_id(chat_id) {
        println!("Access denied for invalid chat ID: {}", chat_id);
        return;
    }

    let path = format!("{}/{}.json", chats_dir, chat_id);
    println!("Resulting path: {}", path);
    panic!("Vulnerability still present! Resulting path: {}", path);
}

#[test]
fn test_safe_id() {
    let chat_id = "chat_123";
    assert!(validate_id(chat_id));
}

#[test]
fn test_unsafe_ids() {
    assert!(!validate_id("../../etc/passwd"));
    assert!(!validate_id("chat_123; rm -rf /"));
    assert!(!validate_id("chat_123.json"));
    assert!(!validate_id(""));
}
