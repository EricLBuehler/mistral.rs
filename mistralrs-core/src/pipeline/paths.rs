use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use crate::pipeline::hf::HubRepo;
use anyhow::Result;
use either::Either;
use regex_automata::meta::Regex;
use serde_json::Value;
use tracing::{debug, info, trace, warn};

use crate::{
    api_dir_list, api_get_file,
    lora::LoraConfig,
    pipeline::{
        chat_template::{BeginEndUnkPadTok, ChatTemplate, ChatTemplateValue},
        hf::build_api,
        isq::UQFF_RESIDUAL_SAFETENSORS,
    },
    xlora_models::XLoraConfig,
    LoraAdapterSpec, ModelPaths, Ordering, TokenSource,
};

// Match files against these
const SAFETENSOR_MATCH: &str = r"model-\d+-of-\d+\.safetensors\b";
const QUANT_SAFETENSOR_MATCH: &str = r"model\.safetensors\b";
const CONSOLIDATED_SAFETENSOR_MATCH: &str = r"consolidated\.safetensors\b";
const SAFETENSOR_INDEX: &str = "model.safetensors.index.json";
const AUDIO_TOKENIZER_DIRECTORY: &str = "audio_tokenizer";
const AUDIO_TOKENIZER_WEIGHTS: &str = "audio_tokenizer/model.safetensors";
const AUDIO_TOKENIZER_INDEX: &str = "audio_tokenizer/model.safetensors.index.json";
const PICKLE_MATCH: &str = r"pytorch_model-\d{5}-of-\d{5}.((pth)|(pt)|(bin))\b";

#[derive(Clone, Debug)]
pub struct ResolvedLoraAdapter {
    pub alias: String,
    pub source: String,
    pub revision: Option<String>,
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum AdapterPaths {
    XLora {
        metadata_paths: Vec<PathBuf>,
        adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
        adapter_safetensors: Option<Vec<(String, PathBuf)>>,
        classifier_path: Option<PathBuf>,
        xlora_order: Option<Ordering>,
        xlora_config: Option<XLoraConfig>,
        lora_preload_adapter_info: Option<HashMap<String, (PathBuf, LoraConfig)>>,
    },
    Lora(Vec<ResolvedLoraAdapter>),
    None,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum XLoraPreload {
    Skip,
    Load,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AdapterPathOptions<'a> {
    pub(crate) xlora_model_id: Option<&'a String>,
    pub(crate) lora_adapters: Option<&'a [LoraAdapterSpec]>,
    pub(crate) xlora_order: Option<&'a Ordering>,
    pub(crate) xlora_preload: XLoraPreload,
}

pub(crate) fn get_adapter_paths(
    base_model_id: String,
    options: AdapterPathOptions<'_>,
    token_source: &TokenSource,
    base_revision: String,
    revision_model_id: &str,
) -> Result<AdapterPaths> {
    let AdapterPathOptions {
        xlora_model_id,
        lora_adapters,
        xlora_order,
        xlora_preload,
    } = options;
    match (lora_adapters, xlora_model_id, xlora_order) {
        (None, Some(xlora_id), Some(xlora_order)) => {
            let revision =
                super::hf::model_source_revision(revision_model_id, &base_revision, xlora_id);
            let mut metadata_paths = Vec::new();
            let api = build_api(token_source, true).map_err(candle_core::Error::msg)?;
            let api = api.repo(xlora_id.clone(), revision.to_string());
            let model_id = Path::new(&xlora_id);
            let dir_list = api_dir_list!(api, model_id, true, revision).collect::<Vec<_>>();
            // Get the path for the xlora classifier
            let xlora_classifier = &dir_list
                .clone()
                .into_iter()
                .filter(|x| x.contains("xlora_classifier.safetensors"))
                .collect::<Vec<_>>();
            if xlora_classifier.len() > 1 {
                warn!("Detected multiple X-LoRA classifiers: {xlora_classifier:?}");
                warn!("Selected classifier: `{}`", &xlora_classifier[0]);
            }
            let xlora_classifier = xlora_classifier.first();

            let classifier_path = xlora_classifier
                .map(|xlora_classifier| -> candle_core::Result<_> {
                    Ok(api_get_file!(api, xlora_classifier, model_id, revision))
                })
                .transpose()?;

            // Get the path for the xlora config by checking all for valid versions.
            // NOTE(EricLBuehler): Remove this functionality because all configs should be deserializable
            let xlora_configs = &dir_list
                .clone()
                .into_iter()
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
                let config_path = api_get_file!(api, config_path, model_id, revision);
                let conf = fs::read_to_string(&config_path)?;
                let deser: Result<XLoraConfig, serde_json::Error> = serde_json::from_str(&conf);
                match deser {
                    Ok(conf) => {
                        metadata_paths.push(config_path);
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
                    panic!("Unable to derserialize any configs. Last error: {last_err}")
                } else {
                    None
                }
            });

            // If there are adapters in the ordering file, get their names and remote paths
            let adapter_files = dir_list
                .into_iter()
                .filter_map(|name| {
                    if let Some(ref adapters) = xlora_order.adapters {
                        for adapter_name in adapters {
                            if name.contains(adapter_name) {
                                return Some((name, adapter_name.clone()));
                            }
                        }
                    }
                    None
                })
                .collect::<Vec<_>>();
            if adapter_files.is_empty() && xlora_order.adapters.is_some() {
                anyhow::bail!("Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?")
            }

            // Get the local paths for each adapter
            let mut adapters_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
            for (file, name) in adapter_files {
                if let Some(paths) = adapters_paths.get_mut(&name) {
                    paths.push(api_get_file!(api, &file, model_id, revision));
                } else {
                    adapters_paths
                        .insert(name, vec![api_get_file!(api, &file, model_id, revision)]);
                }
            }

            // Sort local paths for the adapter configs and safetensors files
            let mut adapters_configs = Vec::new();
            let mut adapters_safetensors = Vec::new();
            if let Some(ref adapters) = xlora_order.adapters {
                for (i, name) in adapters.iter().enumerate() {
                    let paths = adapters_paths
                        .get(name)
                        .unwrap_or_else(|| panic!("Adapter {name} not found."));
                    for path in paths {
                        if path.extension().unwrap() == "safetensors" {
                            adapters_safetensors.push((name.clone(), path.to_owned()));
                        } else {
                            let conf = fs::read_to_string(path)?;
                            let lora_config: LoraConfig = serde_json::from_str(&conf)?;
                            metadata_paths.push(path.clone());
                            adapters_configs
                                .push((((i + 1).to_string(), name.clone()), lora_config));
                        }
                    }
                }
            }

            // Make sure they all match
            if xlora_order.base_model_id
                != *xlora_config
                    .as_ref()
                    .map(|cfg| &cfg.base_model_id)
                    .unwrap_or(&base_model_id)
                || xlora_config
                    .as_ref()
                    .map(|cfg| &cfg.base_model_id)
                    .unwrap_or(&base_model_id)
                    != &base_model_id
            {
                anyhow::bail!(
                    "Adapter ordering file, adapter model config, and base model ID do not match: {}, {}, and {} respectively.",
                    xlora_order.base_model_id,
                    xlora_config.map(|cfg| cfg.base_model_id).unwrap_or(base_model_id.clone()),
                    base_model_id
                );
            }

            let lora_preload_adapter_info = if matches!(xlora_preload, XLoraPreload::Load) {
                if let Some(preload_adapters) = &xlora_order.preload_adapters {
                    let mut output = HashMap::new();
                    for adapter in preload_adapters {
                        let revision = super::hf::model_source_revision(
                            revision_model_id,
                            &base_revision,
                            &adapter.adapter_model_id,
                        );
                        let api = build_api(token_source, true)?
                            .repo(adapter.adapter_model_id.clone(), revision);
                        let model_id = Path::new(&adapter.adapter_model_id);
                        // Get the names and remote paths of the files associated with this adapter
                        let adapter_files =
                            api_dir_list!(api, &adapter.adapter_model_id, true, revision)
                                .filter_map(|f| {
                                    if f.contains(&adapter.name) {
                                        Some((f, adapter.name.clone()))
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>();
                        if adapter_files.is_empty() {
                            anyhow::bail!("Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?")
                        }
                        // Get local paths for this adapter
                        let mut adapters_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
                        for (file, name) in adapter_files {
                            if let Some(paths) = adapters_paths.get_mut(&name) {
                                paths.push(api_get_file!(api, &file, model_id, revision));
                            } else {
                                adapters_paths.insert(
                                    name,
                                    vec![api_get_file!(api, &file, model_id, revision)],
                                );
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
                                let lora_config: LoraConfig = serde_json::from_str(&conf)?;
                                metadata_paths.push(path.clone());
                                config = Some(lora_config);
                            }
                        }

                        let (config, safetensor) = (config.unwrap(), safetensor.unwrap());
                        output.insert(adapter.name.clone(), (safetensor, config));
                    }
                    Some(output)
                } else {
                    None
                }
            } else {
                None
            };

            Ok(AdapterPaths::XLora {
                metadata_paths,
                adapter_configs: Some(adapters_configs),
                adapter_safetensors: Some(adapters_safetensors),
                classifier_path,
                xlora_order: Some(xlora_order.clone()),
                xlora_config,
                lora_preload_adapter_info,
            })
        }
        (Some(adapters), None, None) => {
            let mut lora_adapter_paths = Vec::new();
            let mut aliases = HashSet::new();
            for adapter in adapters {
                let alias = adapter.alias.trim();
                let source = adapter.source.trim();
                let revision = adapter
                    .revision
                    .as_deref()
                    .map(str::trim)
                    .unwrap_or_else(|| {
                        super::hf::model_source_revision(revision_model_id, &base_revision, source)
                    });
                if alias.is_empty() {
                    anyhow::bail!("LoRA adapter alias must not be empty");
                }
                if source.is_empty() {
                    anyhow::bail!(
                        "LoRA adapter source for alias `{}` must not be empty",
                        adapter.alias
                    );
                }
                if revision.is_empty() {
                    anyhow::bail!(
                        "LoRA adapter revision for alias `{}` must not be empty",
                        adapter.alias
                    );
                }
                if let Some(expected) = adapter.base_model_name.as_deref().map(str::trim) {
                    if expected.is_empty() {
                        anyhow::bail!(
                            "LoRA adapter `{}` has an empty base_model_name",
                            adapter.alias
                        );
                    }
                }
                if !aliases.insert(alias) {
                    anyhow::bail!(
                        "LoRA adapter alias `{}` is specified more than once",
                        adapter.alias
                    );
                }
                info!(
                    "Loading LoRA adapter `{}` from `{}` at revision `{}`",
                    alias, source, revision
                );

                let api = build_api(token_source, true).map_err(candle_core::Error::msg)?;
                let api = api.repo(source.to_string(), revision.to_string());

                let adapter_path_buf = std::path::Path::new(source);
                let config_path = crate::pipeline::hf::get_file(
                    &api,
                    adapter_path_buf,
                    "adapter_config.json",
                    revision,
                )?;
                let weights_path = crate::pipeline::hf::get_file(
                    &api,
                    adapter_path_buf,
                    "adapter_model.safetensors",
                    revision,
                )?;
                lora_adapter_paths.push(ResolvedLoraAdapter {
                    alias: alias.to_string(),
                    source: source.to_string(),
                    revision: (!adapter_path_buf.exists()).then(|| revision.to_string()),
                    config_path,
                    weights_path,
                });
            }

            Ok(AdapterPaths::Lora(lora_adapter_paths))
        }
        (None, None, None) => Ok(AdapterPaths::None),
        _ => anyhow::bail!(
            "Incorrect configuration for an adapter model. Lora and XLora are mutually exclusive."
        ),
    }
}

pub fn get_model_paths(
    revision: String,
    token_source: &TokenSource,
    quantized_model_id: Option<&String>,
    quantized_filename: Option<&Vec<String>>,
    api: &HubRepo,
    model_id: &Path,
    loading_from_uqff: bool,
) -> Result<Vec<PathBuf>> {
    match quantized_filename {
        Some(names) => {
            let id = quantized_model_id.unwrap();
            let mut files = Vec::new();

            for name in names {
                let qapi = build_api(token_source, true).map_err(candle_core::Error::msg)?;
                let qapi = qapi.repo(id.to_string(), revision.clone());
                let model_id = Path::new(&id);
                files.push(api_get_file!(qapi, name, model_id, &revision));
            }
            Ok(files)
        }
        None => {
            let safetensor_match = Regex::new(SAFETENSOR_MATCH)?;
            let quant_safetensor_match = Regex::new(QUANT_SAFETENSOR_MATCH)?;
            let consolidated_safetensor_match = Regex::new(CONSOLIDATED_SAFETENSOR_MATCH)?;
            let pickle_match = Regex::new(PICKLE_MATCH)?;

            let mut filenames = vec![];
            let repo_files = api_dir_list!(api, model_id, true, &revision).collect::<Vec<_>>();
            let safetensors = if loading_from_uqff
                && repo_files
                    .iter()
                    .any(|file| file == UQFF_RESIDUAL_SAFETENSORS)
            {
                vec![UQFF_RESIDUAL_SAFETENSORS.to_string()]
            } else if repo_files.iter().any(|file| file == SAFETENSOR_INDEX) {
                let index_path = api_get_file!(api, SAFETENSOR_INDEX, model_id, &revision);
                parse_safetensor_index(&fs::read_to_string(index_path)?)?
            } else {
                let files = repo_files
                    .iter()
                    .filter(|file| file.split('/').next() != Some(AUDIO_TOKENIZER_DIRECTORY))
                    .filter(|file| {
                        safetensor_match.is_match(file)
                            || quant_safetensor_match.is_match(file)
                            || consolidated_safetensor_match.is_match(file)
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                validate_safetensor_shards(&files)?;
                files
            };
            let pickles = repo_files
                .iter()
                .filter(|file| pickle_match.is_match(file))
                .filter(|x| x.ends_with(".pth") || x.ends_with(".pt") || x.ends_with(".bin"))
                .cloned()
                .collect::<Vec<_>>();
            let mut files = if !safetensors.is_empty() {
                // Always prefer safetensors
                safetensors
            } else if !pickles.is_empty() {
                // Fall back to pickle
                pickles
            } else {
                anyhow::bail!("Expected file with extension one of .safetensors, .pth, .pt, .bin.");
            };
            if !files.iter().any(|file| file == UQFF_RESIDUAL_SAFETENSORS) {
                let available = |name: &str| {
                    repo_files.iter().any(|file| file == name) || model_id.join(name).is_file()
                };
                let codec_files = if available(AUDIO_TOKENIZER_INDEX) {
                    let index_path = api_get_file!(api, AUDIO_TOKENIZER_INDEX, model_id, &revision);
                    parse_safetensor_index(&fs::read_to_string(index_path)?)?
                        .into_iter()
                        .map(|file| format!("{AUDIO_TOKENIZER_DIRECTORY}/{file}"))
                        .collect()
                } else if available(AUDIO_TOKENIZER_WEIGHTS) {
                    vec![AUDIO_TOKENIZER_WEIGHTS.to_string()]
                } else {
                    Vec::new()
                };
                append_unique_paths(&mut files, codec_files);
            }
            trace!(
                "Found model weight filenames {:?}",
                files
                    .iter()
                    .map(|x| x.split('/').next_back().unwrap())
                    .collect::<Vec<_>>()
            );
            for rfilename in files {
                filenames.push(api_get_file!(api, &rfilename, model_id, &revision));
            }
            Ok(filenames)
        }
    }
}

fn append_unique_paths(files: &mut Vec<String>, auxiliary_files: Vec<String>) {
    for file in auxiliary_files {
        if !files.contains(&file) {
            files.push(file);
        }
    }
}

fn validate_safetensor_shards(files: &[String]) -> Result<()> {
    let mut checkpoints: HashMap<&str, (usize, HashSet<usize>)> = HashMap::new();
    for file in files {
        let (directory, filename) = file.rsplit_once('/').unwrap_or((".", file));
        let Some((shard, total)) = filename
            .strip_prefix("model-")
            .and_then(|name| name.strip_suffix(".safetensors"))
            .and_then(|name| name.split_once("-of-"))
        else {
            continue;
        };
        let shard = shard.parse::<usize>()?;
        let total = total.parse::<usize>()?;
        anyhow::ensure!(
            shard > 0 && shard <= total,
            "Invalid safetensors shard numbering in `{file}`"
        );
        let (expected, present) = checkpoints
            .entry(directory)
            .or_insert((total, HashSet::new()));
        anyhow::ensure!(
            *expected == total,
            "Inconsistent safetensors shard counts in `{directory}`"
        );
        present.insert(shard);
    }
    for (directory, (expected, present)) in checkpoints {
        anyhow::ensure!(
            present.len() == expected,
            "Incomplete safetensors checkpoint in `{directory}`: found {} of {expected} shards. Download all checkpoint shards before loading.",
            present.len()
        );
    }
    Ok(())
}

fn parse_safetensor_index(contents: &str) -> Result<Vec<String>> {
    let index: Value = serde_json::from_str(contents)?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("Safetensors index has no object `weight_map`"))?;
    let mut files = HashSet::new();
    for filename in weight_map.values() {
        let filename = filename
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Safetensors index contains a non-string filename"))?;
        if !filename.ends_with(".safetensors") {
            anyhow::bail!("Safetensors index references non-safetensors file `{filename}`")
        }
        files.insert(filename.to_string());
    }
    if files.is_empty() {
        anyhow::bail!("Safetensors index has an empty `weight_map`")
    }
    let mut files = files.into_iter().collect::<Vec<_>>();
    files.sort_unstable();
    Ok(files)
}

/// Find and parse the appropriate [`ChatTemplate`], and ensure is has a valid [`ChatTemplate.chat_template`].
/// If the provided `tokenizer_config.json` from [`ModelPaths.get_template_filename`] does not
/// have a `chat_template`, use the provided one.
///
/// - Uses `chat_template_fallback` if `paths` does not contain a chat template file. This may be a literal or .json file.
/// - `chat_template_ovrd` (GGUF chat template content) causes the usage of that string chat template initially.
///   Falls back to `chat_template_file` if it is invalid. *The user must add the bos/unk/eos tokens manually if this
///   is used.*
///
/// THE FOLLOWING IS IGNORED:
/// After this, if the `chat_template_explicit` filename is specified (a json with one field: "chat_template" OR a jinja file),
///  the chat template is overwritten with this chat template.
#[allow(clippy::borrowed_box)]
pub(crate) fn get_chat_template(
    paths: &dyn ModelPaths,
    jinja_explicit: Option<&String>,
    chat_template_explicit: Option<&String>,
    chat_template_fallback: Option<&String>,
    chat_template_ovrd: Option<String>,
) -> ChatTemplate {
    // Get template content, this may be overridden.
    let template_content = if let Some(template_filename) = paths.get_template_filename() {
        if !["jinja", "json"].contains(
            &template_filename
                .extension()
                .expect("Template filename must be a file")
                .to_string_lossy()
                .to_string()
                .as_str(),
        ) {
            panic!("Template filename {template_filename:?} must end with `.json` or `.jinja`.");
        }
        Some(fs::read_to_string(template_filename).expect("Loading chat template failed."))
    } else if chat_template_fallback.is_some_and(|f| f.ends_with(".json")) {
        // User specified a file
        let template_filename = chat_template_fallback
            .expect("A tokenizer config or chat template file path must be specified.");
        Some(fs::read_to_string(template_filename).expect("Loading chat template failed."))
    } else if chat_template_ovrd.is_some() {
        None
    } else {
        debug!("No chat template file found. Chat template may be set via `chat_template.json` or processor config.");
        None
    };
    let mut template: ChatTemplate = match chat_template_ovrd {
        Some(chat_template) => {
            // In this case the override chat template is being used. The user must add the bos/eos/unk toks themselves.
            debug!("Using literal chat template.");
            let mut template = ChatTemplate::default();
            template.chat_template = Some(ChatTemplateValue(Either::Left(chat_template)));
            template
        }
        None => {
            if let Some(ref content) = template_content {
                // Check if template_filename is a .jinja file
                if let Some(template_filename) = paths.get_template_filename() {
                    if template_filename.extension().map(|e| e.to_str()) == Some(Some("jinja")) {
                        debug!("Using chat template from .jinja file.");
                        // Load special tokens (bos/eos/unk) from tokenizer_config.json
                        // in the same directory, matching HF's behavior where
                        // apply_chat_template passes self.special_tokens_map to the template.
                        let mut template = template_filename
                            .parent()
                            .map(|dir| dir.join("tokenizer_config.json"))
                            .filter(|p| p.exists())
                            .and_then(|p| fs::read_to_string(p).ok())
                            .and_then(|s| serde_json::from_str::<ChatTemplate>(&s).ok())
                            .unwrap_or_else(|| {
                                // Fallback: older UQFF repos may not have tokenizer_config.json.
                                // Try to extract bos/eos tokens from the tokenizer.json's
                                // added_tokens list to avoid rendering "none" in the template.
                                let mut ct = ChatTemplate::default();
                                if let Some(tok_path) = paths
                                    .get_tokenizer_filename()
                                    .parent()
                                    .map(|d| d.join("tokenizer.json"))
                                    .filter(|p| p.exists())
                                    .or_else(|| {
                                        template_filename
                                            .parent()
                                            .map(|d| d.join("tokenizer.json"))
                                            .filter(|p| p.exists())
                                    })
                                {
                                    if let Some(tok_json) =
                                        fs::read_to_string(&tok_path).ok().and_then(|s| {
                                            serde_json::from_str::<serde_json::Value>(&s).ok()
                                        })
                                    {
                                        let added = tok_json
                                            .get("added_tokens")
                                            .and_then(serde_json::Value::as_array);
                                        for token in added.into_iter().flatten() {
                                            let content = token
                                                .get("content")
                                                .and_then(serde_json::Value::as_str)
                                                .unwrap_or("");
                                            let special = token
                                                .get("special")
                                                .and_then(serde_json::Value::as_bool)
                                                .unwrap_or(false);
                                            if special {
                                                if content == "<bos>" {
                                                    ct.bos_token = Some(BeginEndUnkPadTok(
                                                        Either::Left(content.to_string()),
                                                    ));
                                                } else if content == "<eos>" {
                                                    ct.eos_token = Some(BeginEndUnkPadTok(
                                                        Either::Left(content.to_string()),
                                                    ));
                                                } else if content == "<unk>" {
                                                    ct.unk_token = Some(BeginEndUnkPadTok(
                                                        Either::Left(content.to_string()),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                                ct
                            });
                        template.chat_template =
                            Some(ChatTemplateValue(Either::Left(content.clone())));
                        template
                    } else {
                        serde_json::from_str(content).unwrap()
                    }
                } else {
                    serde_json::from_str(content).unwrap()
                }
            } else {
                // No template content available; downstream code may fill in from
                // chat_template.json, processor_config, or jinja_explicit.
                ChatTemplate::default()
            }
        }
    };
    // Overwrite to use any present `chat_template.json`, only if there is not one present already.
    if template.chat_template.is_none() {
        if let Some(chat_template_explicit) = chat_template_explicit {
            let ct =
                fs::read_to_string(chat_template_explicit).expect("Loading chat template failed.");

            let new_chat_template = if chat_template_explicit.ends_with(".jinja") {
                ct
            } else {
                #[derive(Debug, serde::Deserialize)]
                struct AutomaticTemplate {
                    chat_template: String,
                }
                let deser: AutomaticTemplate = serde_json::from_str(&ct).unwrap();
                deser.chat_template
            };

            template.chat_template = Some(ChatTemplateValue(Either::Left(new_chat_template)));
        }
    }

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

    if let Some(jinja_explicit) = jinja_explicit {
        if !jinja_explicit.ends_with(".jinja") {
            panic!("jinja_explicit must end with .jinja!");
        }

        let ct = fs::read_to_string(jinja_explicit).expect("Loading chat template failed.");
        template.chat_template = Some(ChatTemplateValue(Either::Left(ct)));
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
            if let Some(template_content) = template_content {
                info!("`tokenizer_config.json` does not contain a chat template, attempting to use specified JINJA chat template.");
                let mut deser: HashMap<String, Value> =
                    serde_json::from_str(&template_content).unwrap();

                match chat_template_fallback.cloned() {
                    Some(t) => {
                        info!("Loading specified loading chat template file at `{t}`.");
                        let templ: SpecifiedTemplate =
                            serde_json::from_str(&fs::read_to_string(t.clone()).unwrap()).unwrap();
                        deser.insert(
                            "chat_template".to_string(),
                            Value::String(templ.chat_template),
                        );
                        if let Some(bos_token) = templ.bos_token {
                            deser.insert("bos_token".to_string(), Value::String(bos_token));
                        }
                        if let Some(eos_token) = templ.eos_token {
                            deser.insert("eos_token".to_string(), Value::String(eos_token));
                        }
                        if let Some(unk_token) = templ.unk_token {
                            deser.insert("unk_token".to_string(), Value::String(unk_token));
                        }
                    }
                    None => {
                        warn!("No specified chat template. No chat template will be used. Only prompts will be accepted, not messages.");
                        deser.insert("chat_template".to_string(), Value::Null);
                    }
                }

                let ser = serde_json::to_string_pretty(&deser)
                    .expect("Serialization of modified chat template failed.");
                serde_json::from_str(&ser).unwrap()
            } else {
                warn!("No chat template source found. No chat template will be used. Only prompts will be accepted, not messages.");
                template
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline::loaders::LocalModelPaths;

    use super::{get_chat_template, parse_safetensor_index, AdapterPaths};

    #[test]
    fn download_rejects_incomplete_numbered_safetensor_shards() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        for filename in [
            "config.json",
            "tokenizer.json",
            "model-00001-of-00002.safetensors",
        ] {
            std::fs::write(dir.path().join(filename), b"invalid model data")?;
        }
        let api = super::build_api(&crate::TokenSource::None, false)?.repo("local/test", "main");
        let discover = || {
            super::get_model_paths(
                "main".to_string(),
                &crate::TokenSource::None,
                None,
                None,
                &api,
                dir.path(),
                false,
            )
        };
        let error = discover().unwrap_err();
        assert!(error.to_string().contains("found 1 of 2 shards"));
        std::fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            b"invalid model data",
        )?;
        assert_eq!(discover()?.len(), 2);
        Ok(())
    }

    #[test]
    fn local_sharded_model_includes_nested_audio_codec() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(super::AUDIO_TOKENIZER_DIRECTORY))?;
        std::fs::write(dir.path().join("text-shard.safetensors"), [])?;
        std::fs::write(dir.path().join(super::AUDIO_TOKENIZER_WEIGHTS), [])?;
        std::fs::write(
            dir.path().join(super::SAFETENSOR_INDEX),
            r#"{"weight_map":{"model.weight":"text-shard.safetensors"}}"#,
        )?;
        let api = super::build_api(&crate::TokenSource::None, false)?.repo("local/test", "main");
        let paths = super::get_model_paths(
            "main".to_string(),
            &crate::TokenSource::None,
            None,
            None,
            &api,
            dir.path(),
            false,
        )?;
        assert_eq!(
            paths,
            [
                dir.path().join("text-shard.safetensors"),
                dir.path().join(super::AUDIO_TOKENIZER_WEIGHTS)
            ]
        );

        std::fs::remove_file(dir.path().join(super::SAFETENSOR_INDEX))?;
        std::fs::remove_file(dir.path().join("text-shard.safetensors"))?;
        std::fs::write(dir.path().join(super::UQFF_RESIDUAL_SAFETENSORS), [])?;
        let paths = super::get_model_paths(
            "main".to_string(),
            &crate::TokenSource::None,
            None,
            None,
            &api,
            dir.path(),
            true,
        )?;
        assert_eq!(paths, [dir.path().join(super::UQFF_RESIDUAL_SAFETENSORS)]);
        Ok(())
    }

    #[test]
    fn explicit_jinja_overrides_processor_template() {
        let dir = tempfile::tempdir().unwrap();
        let processor_path = dir.path().join("processor_config.json");
        let jinja_path = dir.path().join("explicit.jinja");
        std::fs::write(&processor_path, r#"{"chat_template":"processor"}"#).unwrap();
        std::fs::write(&jinja_path, "explicit").unwrap();
        let paths = LocalModelPaths {
            tokenizer_filename: dir.path().join("tokenizer.json"),
            config_filename: dir.path().join("config.json"),
            template_filename: None,
            filenames: Vec::new(),
            adapter_paths: AdapterPaths::None,
            gen_conf: None,
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: Some(processor_path),
            chat_template_json_filename: None,
        };
        let jinja_path = jinja_path.to_string_lossy().into_owned();

        let template = get_chat_template(&paths, Some(&jinja_path), None, None, None);

        assert_eq!(template.get_template_contents(), ["explicit"]);
    }

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
            "model-0000a-of-00002.safetensors",
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
    fn safetensor_index_drives_arbitrary_shard_names() -> anyhow::Result<()> {
        let files = parse_safetensor_index(
            r#"{
                "metadata": {"total_size": 42},
                "weight_map": {
                    "model.embed_tokens.weight": "embeddings.safetensors",
                    "model.layers.0.weight": "layers-0.safetensors",
                    "model.layers.0.weight_scale_inv": "layers-0.safetensors",
                    "model.visual.weight": "vision.safetensors"
                }
            }"#,
        )?;
        assert_eq!(
            files,
            [
                "embeddings.safetensors",
                "layers-0.safetensors",
                "vision.safetensors"
            ]
        );
        Ok(())
    }

    #[test]
    fn safetensor_index_rejects_non_safetensor_shards() {
        let err = parse_safetensor_index(r#"{"weight_map":{"model.weight":"pytorch_model.bin"}}"#)
            .unwrap_err();
        assert!(err.to_string().contains("non-safetensors"));
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
