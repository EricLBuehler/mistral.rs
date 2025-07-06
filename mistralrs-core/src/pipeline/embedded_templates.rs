use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Embedded chat templates for supported models
/// These are used as fallbacks when no chat template is found
static EMBEDDED_CHAT_TEMPLATES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut templates = HashMap::new();

    // Embed Gemma3n chat template
    templates.insert(
        "gemma3n",
        include_str!("../../../chat_templates/gemma3n.jinja"),
    );

    templates
});

/// Get an embedded chat template for a model type string
/// Returns None if no embedded template exists for the model type
pub fn get_embedded_chat_template_for_model_type(model_type: &str) -> Option<&'static str> {
    // Check for exact match first
    if let Some(template) = EMBEDDED_CHAT_TEMPLATES.get(model_type) {
        return Some(template);
    }

    // Check for partial matches (e.g., "gemma3n_text" matches "gemma3n")
    let normalized = model_type.to_lowercase().replace("-", "").replace("_", "");
    for (key, value) in EMBEDDED_CHAT_TEMPLATES.iter() {
        if normalized.contains(key) || key.contains(&normalized) {
            return Some(value);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_gemma3n_template() {
        // Test exact match
        assert!(get_embedded_chat_template_for_model_type("gemma3n").is_some());

        // Test partial matches
        assert!(get_embedded_chat_template_for_model_type("gemma3n_text").is_some());
        assert!(get_embedded_chat_template_for_model_type("gemma3n_vision").is_some());
        assert!(get_embedded_chat_template_for_model_type("gemma3n_audio").is_some());

        // Test case insensitive
        assert!(get_embedded_chat_template_for_model_type("Gemma3n").is_some());
        assert!(get_embedded_chat_template_for_model_type("GEMMA3N").is_some());

        // Test with dashes/underscores
        assert!(get_embedded_chat_template_for_model_type("gemma-3n").is_some());
        assert!(get_embedded_chat_template_for_model_type("gemma_3n").is_some());

        // Test non-existent model
        assert!(get_embedded_chat_template_for_model_type("nonexistent").is_none());
    }

    #[test]
    fn test_embedded_template_content() {
        let template = get_embedded_chat_template_for_model_type("gemma3n");
        assert!(template.is_some());

        let template_content = template.unwrap();
        // Verify key parts of the template
        assert!(template_content.contains("<start_of_turn>"));
        assert!(template_content.contains("<end_of_turn>"));
        assert!(template_content.contains("<audio_soft_token>"));
        assert!(template_content.contains("<image_soft_token>"));
    }
}
