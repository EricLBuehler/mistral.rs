use super::Content;

// Get chat template from GGUF metadata if it exists
pub fn get_gguf_chat_template<R: std::io::Seek + std::io::Read>(
    content: &Content<'_, R>,
) -> Option<String> {
    content
        .get_metadata("tokenizer.chat_template")
        .ok()
        .map(|template| {
            template
                .to_string()
                .expect("Chat template must be a string")
                .clone()
        })
}
