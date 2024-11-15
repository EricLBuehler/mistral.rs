use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use either::Either;
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use regex_automata::meta::{BuildError, Regex};
use serde_json::Value;
use tracing::{info, warn};

use crate::{
    lora::LoraConfig,
    pipeline::{
        chat_template::{ChatTemplate, ChatTemplateValue},
        isq::UQFF_RESIDUAL_SAFETENSORS,
    },
    utils::tokens::get_token,
    xlora_models::XLoraConfig,
    ModelPaths, Ordering, TokenSource,
};

use super::hf::{api_dir_list, api_get_file, HFError};

// Match files against these, avoids situations like `consolidated.safetensors`
const SAFETENSOR_MATCH: &str = r"model-\d{5}-of-\d{5}.safetensors\b";
const QUANT_SAFETENSOR_MATCH: &str = r"model.safetensors\b";
const PICKLE_MATCH: &str = r"pytorch_model-\d{5}-of-\d{5}.((pth)|(pt)|(bin))\b";

#[derive(Debug, Default)]
pub(crate) struct XLoraPaths {
    pub adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
    pub adapter_safetensors: Option<Vec<(String, PathBuf)>>,
    pub classifier_path: Option<PathBuf>,
    pub xlora_order: Option<Ordering>,
    pub xlora_config: Option<XLoraConfig>,
    pub lora_preload_adapter_info: Option<HashMap<String, (PathBuf, LoraConfig)>>,
}

pub fn get_xlora_paths(
    base_model_id: String,
    xlora_model_id: Option<&str>,
    token_source: &TokenSource,
    revision: String,
    xlora_order: Option<&Ordering>,
) -> Result<XLoraPaths, HFError> {
    let Some(xlora_id) = xlora_model_id else {
        return Ok(XLoraPaths::default());
    };
    let api = ApiBuilder::new()
        .with_progress(true)
        .with_token(get_token(token_source)?)
        .build()?
        .repo(Repo::with_revision(
            xlora_id.to_string(),
            RepoType::Model,
            revision,
        ));

    let model_id = Path::new(&xlora_id);
    let api_dir = api_dir_list(&api, model_id)?;

    // Get the path for the xlora classifier
    let xlora_classifier = api_dir
        .iter()
        .filter(|x| x.contains("xlora_classifier.safetensors"))
        .collect::<Vec<_>>();
    if xlora_classifier.len() > 1 {
        warn!("Detected multiple X-LoRA classifiers: {xlora_classifier:?}");
        warn!("Selected classifier: `{}`", &xlora_classifier[0]);
    }
    let xlora_classifier = xlora_classifier.first();

    let classifier_path = xlora_classifier
        .map(|xlora_classifier| api_get_file(&api, xlora_classifier, model_id))
        .transpose()?;

    // Get the path for the xlora config by checking all for valid versions.
    // NOTE(EricLBuehler): Remove this functionality because all configs should be deserializable
    let xlora_configs = api_dir
        .iter()
        .filter(|x| x.contains("xlora_config.json"))
        .collect::<Vec<_>>();
    if xlora_configs.len() > 1 {
        warn!("Detected multiple X-LoRA configs: {xlora_configs:?}");
    }

    let mut xlora_config: Option<XLoraConfig> = None;
    let mut last_err: Option<serde_json::Error> = None;
    for (i, config_path) in xlora_configs.iter().enumerate() {
        if xlora_configs.len() != 1 {
            warn!("Selecting config: `{}`", config_path);
        }
        let config_path = api_get_file(&api, config_path, model_id)?;
        let conf = fs::read_to_string(config_path)?;
        let deser: Result<XLoraConfig, serde_json::Error> = serde_json::from_str(&conf);
        match deser {
            Ok(conf) => {
                xlora_config = Some(conf);
                break;
            }
            Err(e) => {
                if i != xlora_configs.len() - 1 {
                    warn!("Config is broken with error `{e}`");
                }
                last_err = Some(e);
            }
        }
    }
    let xlora_config = xlora_config.map(Some).unwrap_or_else(|| {
        if let Some(last_err) = last_err {
            panic!(
                "Unable to derserialize any configs. Last error: {}",
                last_err
            )
        } else {
            None
        }
    });

    // If there are adapters in the ordering file, get their names and remote paths
    let adapter_files = api_dir
        .iter()
        .filter_map(|name| {
            let Some(order) = xlora_order else {
                return None;
            };
            let Some(ref adapters) = order.adapters else {
                return None;
            };
            for adapter_name in adapters {
                if name.contains(adapter_name) {
                    return Some((name, adapter_name.clone()));
                }
            }
            None
        })
        .collect::<Vec<_>>();

    if adapter_files.is_empty() && xlora_order.is_some_and(|x| x.adapters.is_some()) {
        return Err(
            HFError::InvalidRepoStructure(
                "Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?".to_string()
            )
        );
    }

    // Get the local paths for each adapter
    let mut adapters_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for (file, name) in adapter_files {
        if let Some(paths) = adapters_paths.get_mut(&name) {
            paths.push(api_get_file(&api, &file, model_id)?);
        } else {
            adapters_paths.insert(name, vec![api_get_file(&api, &file, model_id)?]);
        }
    }

    // Sort local paths for the adapter configs and safetensors files
    let (adapters_configs, adapters_safetensors) =
        xlora_adapters_config_and_safetensors(xlora_order, adapters_paths)?;

    // Make sure they all match
    if xlora_order.as_ref().is_some_and(|order| {
        &order.base_model_id
            != xlora_config
                .as_ref()
                .map(|cfg| &cfg.base_model_id)
                .unwrap_or(&base_model_id)
    }) || xlora_config
        .as_ref()
        .map(|cfg| &cfg.base_model_id)
        .unwrap_or(&base_model_id)
        != &base_model_id
    {
        return Err(HFError::InvalidRepoStructure(format!("Adapter ordering file, adapter model config, and base model ID do not match: {}, {}, and {} respectively.",
            xlora_order.as_ref().unwrap().base_model_id,
            xlora_config.map(|cfg| cfg.base_model_id).unwrap_or(base_model_id.clone()),
            base_model_id
        )));
    };

    let lora_preload_adapter_info = lora_preload_adapter_info(&api, model_id, xlora_order)?;
    Ok(XLoraPaths {
        adapter_configs: Some(adapters_configs),
        adapter_safetensors: Some(adapters_safetensors),
        classifier_path,
        xlora_order: xlora_order.cloned(),
        xlora_config,
        lora_preload_adapter_info,
    })
}

// Sort local paths for the adapter configs and safetensors files
//
// Returns a tuple of the adapter configs and safetensors files
pub fn xlora_adapters_config_and_safetensors(
    xlora_order: Option<&Ordering>,
    adapter_paths: HashMap<String, Vec<PathBuf>>,
) -> Result<(Vec<((String, String), LoraConfig)>, Vec<(String, PathBuf)>), HFError> {
    let Some(xlora_order) = xlora_order else {
        return Ok((Vec::new(), Vec::new()));
    };
    let Some(ref adapters) = xlora_order.adapters else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut adapters_configs = Vec::new();
    let mut adapters_safetensors = Vec::new();
    for (i, name) in adapters.iter().enumerate() {
        let paths = adapter_paths
            .get(name)
            .unwrap_or_else(|| panic!("Adapter {name} not found."));
        for path in paths {
            if path.extension().unwrap() == "safetensors" {
                adapters_safetensors.push((name.clone(), path.to_owned()));
            } else {
                let conf = fs::read_to_string(path).map_err(|e| HFError::IoError(e))?;
                let lora_config: LoraConfig = serde_json::from_str(&conf).map_err(|e| {
                    HFError::JsonError(e, "failed to parse LORA config file".to_string())
                })?;
                adapters_configs.push((((i + 1).to_string(), name.clone()), lora_config));
            }
        }
    }
    Ok((adapters_configs, adapters_safetensors))
}

// If preload adapters are specified
pub fn lora_preload_adapter_info(
    api: &ApiRepo,
    model_id: &Path,
    xlora_order: Option<&Ordering>,
) -> Result<Option<HashMap<String, (PathBuf, LoraConfig)>>, HFError> {
    let Some(xlora_order) = xlora_order else {
        return Ok(None);
    };
    let Some(ref preload_adapters) = xlora_order.preload_adapters else {
        return Ok(None);
    };
    let mut output = HashMap::new();
    for adapter in preload_adapters {
        // Get the names and remote paths of the files associated with this adapter
        let files = api_dir_list(api, &adapter.adapter_model_id)?;
        let adapter_files = files
            .iter()
            .filter_map(|f| {
                if f.contains(&adapter.name) {
                    Some((f, adapter.name.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if adapter_files.is_empty() {
            return Err(HFError::InvalidRepoStructure("Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?".to_string()));
        }
        // Get local paths for this adapter
        let mut adapters_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
        for (file, name) in adapter_files {
            let path = api_get_file(api, file, model_id)?;
            if let Some(paths) = adapters_paths.get_mut(&name) {
                paths.push(path);
            } else {
                adapters_paths.insert(name, vec![path]);
            }
        }

        let mut config = None;
        let mut safetensor = None;

        // Sort local paths for the adapter configs and safetensors files
        let paths = adapters_paths
            .get(&adapter.name)
            .unwrap_or_else(|| panic!("Adapter {} not found.", adapter.name));
        for path in paths {
            if path.extension().unwrap() == "safetensors" {
                safetensor = Some(path.to_owned());
            } else {
                let conf = fs::read_to_string(path)?;
                let lora_config: LoraConfig = serde_json::from_str(&conf).map_err(|e| {
                    HFError::JsonError(e, "failed to parse LORA config file".to_string())
                })?;
                config = Some(lora_config);
            }
        }

        let (Some(config), Some(safetensor)) = (config, safetensor) else {
            return Err(HFError::InvalidRepoStructure("Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?".to_string()));
        };
        output.insert(adapter.name.clone(), (safetensor, config));
    }
    Ok(Some(output))
}

fn compile_patterns() -> Result<(Regex, Regex, Regex), BuildError> {
    let safetensor_match = Regex::new(SAFETENSOR_MATCH)?;
    let quant_safetensor_match = Regex::new(QUANT_SAFETENSOR_MATCH)?;
    let pickle_match = Regex::new(PICKLE_MATCH)?;
    Ok((safetensor_match, quant_safetensor_match, pickle_match))
}

pub fn get_model_paths(
    revision: String,
    token_source: &TokenSource,
    quantized_model_id: Option<&str>,
    quantized_filename: Option<Vec<String>>,
    api: &ApiRepo,
    model_id: &Path,
    loading_from_uqff: bool,
) -> Result<Vec<PathBuf>, HFError> {
    match &quantized_filename {
        Some(names) => {
            let id = quantized_model_id.as_ref().unwrap();
            let mut files = Vec::new();

            for name in names {
                let qapi = ApiBuilder::new()
                    .with_progress(true)
                    .with_token(get_token(token_source)?)
                    .build()?;
                let qapi = qapi.repo(Repo::with_revision(
                    id.to_string(),
                    RepoType::Model,
                    revision.clone(),
                ));
                let model_id = Path::new(&id);
                files.push(api_get_file(&qapi, name, model_id)?);
            }
            Ok(files)
        }
        None => {
            // We only match these patterns for model names
            let (safetensor_match, quant_safetensor_match, pickle_match) = compile_patterns()
                .map_err(|_| {
                    HFError::InvalidRepoStructure("Invalid model pattern(s)".to_string())
                })?;

            let mut filenames = vec![];
            let all_listing = api_dir_list(api, model_id)?;
            let listing = all_listing.iter().filter(|&x| {
                safetensor_match.is_match(x)
                    || pickle_match.is_match(x)
                    || quant_safetensor_match.is_match(x)
                    || x == UQFF_RESIDUAL_SAFETENSORS
            });
            let safetensors = listing
                .clone()
                .filter(|x| x.ends_with(".safetensors"))
                .collect::<Vec<_>>();
            let pickles = listing
                .clone()
                .filter(|x| x.ends_with(".pth") || x.ends_with(".pt") || x.ends_with(".bin"))
                .collect::<Vec<_>>();
            let uqff_residual = listing
                .clone()
                .filter(|&x| x == UQFF_RESIDUAL_SAFETENSORS)
                .collect::<Vec<_>>();
            let files = if !safetensors.is_empty() {
                // Always prefer safetensors
                safetensors
            } else if !pickles.is_empty() {
                // Fall back to pickle
                pickles
            } else if !uqff_residual.is_empty() && loading_from_uqff {
                uqff_residual
            } else {
                return Err(HFError::InvalidRepoStructure(
                    "Expected file with extension one of .safetensors, .pth, .pt, .bin."
                        .to_string(),
                ));
            };
            info!(
                "Found model weight filenames {:?}",
                files
                    .iter()
                    .map(|x| x.split('/').last().unwrap())
                    .collect::<Vec<_>>()
            );
            for rfilename in files {
                filenames.push(api_get_file(&api, &rfilename, model_id)?);
            }
            Ok(filenames)
        }
    }
}

/// Find and parse the appropriate [`ChatTemplate`], and ensure is has a valid [`ChatTemplate.chat_template`].
/// If the provided `tokenizer_config.json` from [`ModelPaths.get_template_filename`] does not
/// have a `chat_template`, use the provided one.
///
/// - Uses `chat_template_fallback` if `paths` does not contain a chat template file. This may be a literal or .json file.
/// - `chat_template_ovrd` (GGUF chat template content) causes the usage of that string chat template initially.
///   Falls back to `chat_template_file` if it is invalid. *The user must add the bos/unk/eos tokens manually if this
///   is used.*
#[allow(clippy::borrowed_box)]
pub(crate) fn get_chat_template(
    paths: &Box<dyn ModelPaths>,
    chat_template_fallback: &Option<String>,
    chat_template_ovrd: Option<String>,
) -> ChatTemplate {
    // Get template content, this may be overridden.
    let template_content = if let Some(template_filename) = paths.get_template_filename() {
        if template_filename
            .extension()
            .expect("Template filename must be a file")
            .to_string_lossy()
            != "json"
        {
            panic!("Template filename {template_filename:?} must end with `.json`.");
        }
        Some(fs::read_to_string(template_filename).expect("Loading chat template failed."))
    } else if chat_template_fallback
        .as_ref()
        .is_some_and(|f| f.ends_with(".json"))
    {
        // User specified a file
        let template_filename = chat_template_fallback
            .as_ref()
            .expect("A tokenizer config or chat template file path must be specified.");
        Some(fs::read_to_string(template_filename).expect("Loading chat template failed."))
    } else if chat_template_ovrd.is_some() {
        None
    } else {
        panic!("Expected chat template file to end with .json, or you can specify a tokenizer model ID to load the chat template there. If you are running a GGUF model, it probably does not contain a chat template.");
    };

    let mut template: ChatTemplate = match chat_template_ovrd {
        Some(chat_template) => {
            // In this case the override chat template is being used. The user must add the bos/eos/unk toks themselves.
            info!("Using literal chat template.");
            let mut template = ChatTemplate::default();
            template.chat_template = Some(ChatTemplateValue(Either::Left(chat_template)));
            template
        }
        None => serde_json::from_str(&template_content.as_ref().unwrap().clone()).unwrap(),
    };

    let processor_conf: Option<crate::vision_models::processor_config::ProcessorConfig> = paths
        .get_processor_config()
        .as_ref()
        .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
    if let Some(processor_conf) = processor_conf {
        if processor_conf.chat_template.is_some() {
            template.chat_template = processor_conf
                .chat_template
                .map(|x| ChatTemplateValue(Either::Left(x)));
        }
    }

    #[derive(Debug, serde::Deserialize)]
    struct SpecifiedTemplate {
        chat_template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
        unk_token: Option<String>,
    }

    if template.chat_template.is_some() {
        return template;
    };

    match &template.chat_template {
        Some(_) => template,
        None => {
            info!("`tokenizer_config.json` does not contain a chat template, attempting to use specified JINJA chat template.");
            let mut deser: HashMap<String, Value> =
                serde_json::from_str(&template_content.unwrap()).unwrap();

            match chat_template_fallback.clone() {
                Some(t) => {
                    info!("Loading specified loading chat template file at `{t}`.");
                    let templ: SpecifiedTemplate =
                        serde_json::from_str(&fs::read_to_string(t.clone()).unwrap()).unwrap();
                    deser.insert(
                        "chat_template".to_string(),
                        Value::String(templ.chat_template),
                    );
                    if templ.bos_token.is_some() {
                        deser.insert(
                            "bos_token".to_string(),
                            Value::String(templ.bos_token.unwrap()),
                        );
                    }
                    if templ.eos_token.is_some() {
                        deser.insert(
                            "eos_token".to_string(),
                            Value::String(templ.eos_token.unwrap()),
                        );
                    }
                    if templ.unk_token.is_some() {
                        deser.insert(
                            "unk_token".to_string(),
                            Value::String(templ.unk_token.unwrap()),
                        );
                    }
                }
                None => {
                    info!("No specified chat template. No chat template will be used. Only prompts will be accepted, not messages.");
                    deser.insert("chat_template".to_string(), Value::Null);
                }
            }

            let ser = serde_json::to_string_pretty(&deser)
                .expect("Serialization of modified chat template failed.");
            serde_json::from_str(&ser).unwrap()
        }
    }
}

mod tests {
    #[test]
    fn match_safetensors() -> anyhow::Result<()> {
        use regex_automata::meta::Regex;

        use super::SAFETENSOR_MATCH;
        let safetensor_match = Regex::new(SAFETENSOR_MATCH)?;

        let positive_ids = [
            "model-00001-of-00001.safetensors",
            "model-00002-of-00002.safetensors",
            "model-00003-of-00003.safetensors",
            "model-00004-of-00004.safetensors",
            "model-00005-of-00005.safetensors",
            "model-00006-of-00006.safetensors",
        ];
        let negative_ids = [
            "model-000001-of-00001.safetensors",
            "model-0000a-of-00002.safetensors",
            "model-000-of-00003.safetensors",
            "consolidated.safetensors",
        ];
        for id in positive_ids {
            assert!(safetensor_match.is_match(id));
        }
        for id in negative_ids {
            assert!(!safetensor_match.is_match(id));
        }
        Ok(())
    }

    #[test]
    fn match_pickle() -> anyhow::Result<()> {
        use regex_automata::meta::Regex;

        use super::PICKLE_MATCH;
        let pickle_match = Regex::new(PICKLE_MATCH)?;

        let positive_ids = [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ];
        let negative_ids = [
            "pytorch_model-000001-of-00001.bin",
            "pytorch_model-0000a-of-00002.bin",
            "pytorch_model-000-of-00003.bin",
            "pytorch_consolidated.bin",
        ];
        for id in positive_ids {
            assert!(pickle_match.is_match(id));
        }
        for id in negative_ids {
            assert!(!pickle_match.is_match(id));
        }
        Ok(())
    }
}
