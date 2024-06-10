use anyhow::Result;
use candle_core::quantized::gguf_file::Content;
use tracing::info;

use crate::utils::gguf_metadata::ContentMetadata;

struct PropsGGUFTemplate {
    chat_template: Option<String>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUFTemplate {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> Result<Self, Self::Error> {
        // No required keys

        let props = Self {
            chat_template: c.get_option_value("chat_template")?,
        };

        Ok(props)
    }
}

// Get chat template from GGUF metadata if it exists
pub fn get_gguf_chat_template(content: &Content) -> Result<Option<String>> {
    let metadata = ContentMetadata {
        path_prefix: "tokenizer",
        metadata: &content.metadata,
    };
    let props = PropsGGUFTemplate::try_from(metadata)?;
    if let Some(ref chat_template) = props.chat_template {
        info!(
            "Discovered and using GGUF chat template: `{}`",
            chat_template.replace('\n', "\\n")
        );
    }
    Ok(props.chat_template)
}
