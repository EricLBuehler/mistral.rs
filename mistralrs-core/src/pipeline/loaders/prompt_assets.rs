use std::fs;

use either::Either;
use tracing::warn;

use crate::pipeline::chat_template::{BeginEndUnkPadTok, ChatTemplate, GenerationConfig};
use crate::pipeline::{get_chat_template, ModelPaths};

/// Template data a prepared (already parsed) source carries instead of files.
pub(crate) struct PreparedPromptAssets<'a> {
    pub generation_config: Option<&'a GenerationConfig>,
    pub chat_template: Option<&'a String>,
    pub bos_token: Option<&'a String>,
    pub eos_token: Option<&'a String>,
    pub unk_token: Option<&'a String>,
}

pub(crate) struct PromptAssetsRequest<'a> {
    pub paths: &'a dyn ModelPaths,
    pub config: &'a str,
    pub prepared: Option<PreparedPromptAssets<'a>>,
    pub jinja_explicit: Option<&'a String>,
    pub chat_template: Option<&'a String>,
}

/// Loads the generation config and chat template, preferring the prepared source.
pub(crate) fn load_prompt_assets(
    request: PromptAssetsRequest<'_>,
) -> (Option<GenerationConfig>, ChatTemplate) {
    let PromptAssetsRequest {
        paths,
        config,
        prepared,
        jinja_explicit,
        chat_template: chat_template_override,
    } = request;
    let gen_conf: Option<GenerationConfig> = match prepared.as_ref() {
        Some(source) => source.generation_config.cloned(),
        None => paths.get_gen_conf_filename().and_then(|f| {
            match serde_json::from_str::<GenerationConfig>(&fs::read_to_string(f).unwrap()) {
                Ok(conf) => Some(conf),
                Err(e) => {
                    warn!("Failed to parse generation_config.json: {}", e);
                    None
                }
            }
        }),
    };
    let gen_conf = gen_conf.or_else(|| GenerationConfig::from_model_config(config));

    let chat_template_explicit = paths
        .get_chat_template_explicit()
        .as_ref()
        .map(|x| x.to_string_lossy().to_string());
    let mut chat_template = get_chat_template(
        paths,
        jinja_explicit,
        chat_template_explicit.as_ref(),
        chat_template_override,
        prepared
            .as_ref()
            .and_then(|source| source.chat_template.cloned()),
    );
    if let Some(source) = prepared.as_ref() {
        let token = |token: Option<&String>| {
            token
                .cloned()
                .map(|token| BeginEndUnkPadTok(Either::Left(token)))
        };
        if chat_template.bos_token.is_none() {
            chat_template.bos_token = token(source.bos_token);
        }
        if chat_template.eos_token.is_none() {
            chat_template.eos_token = token(source.eos_token);
        }
        if chat_template.unk_token.is_none() {
            chat_template.unk_token = token(source.unk_token);
        }
    }
    (gen_conf, chat_template)
}
