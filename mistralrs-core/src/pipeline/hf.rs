use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use hf_hub::{
    api::sync::{ApiBuilder, ApiError as HFHubApiError, ApiRepo},
    Repo, RepoType,
};
use thiserror::Error;
use tracing::info;

use crate::{
    pipeline::{get_model_paths, get_xlora_paths, XLoraPaths},
    utils::tokens::{get_token, TokenRetrievalError},
    LocalModelPaths, Ordering,
};

use super::{ModelPaths, TokenSource};

#[derive(Error, Debug)]
pub enum HFError {
    #[error("Unable to load {path} file from {model_id} repo.")]
    FileNotFound { model_id: String, path: String },

    #[error("Not authorized to access {model_id} repo.")]
    AuthorizationError { model_id: String },

    #[error("HF API error occurred: {0:?}")]
    HFHubApiError(#[from] HFHubApiError),

    #[error("Could not retrieve HF API token: {0:?}")]
    HFTokenError(#[from] TokenRetrievalError),

    #[error("IoError: {0:?}")]
    IoError(#[from] std::io::Error),

    #[error("Json Error. Reason {1:?}. Error: {0:?}")]
    JsonError(serde_json::Error, String),

    #[error("HF repo has invalid structure: {0:?}")]
    InvalidRepoStructure(String),
}

/// Attempts to retrieve a file from a HF repo. Will check if the file exists locally first.
///
/// # Returns
/// * `Result<PathBuf, String>` - The path to the file (if found, or downloaded), error message if not.
pub(crate) fn api_get_file(
    api: &ApiRepo,
    file: &str,
    model_id: impl AsRef<Path>,
) -> Result<PathBuf, HFError> {
    let model_id = model_id.as_ref();

    if model_id.exists() {
        let path = model_id.join(file);
        if !path.exists() {
            return Err(HFError::FileNotFound {
                model_id: model_id.display().to_string(),
                path: file.to_string(),
            });
        }
        info!("Loading `{file}` locally at `{}`", path.display());
        Ok(path)
    } else {
        api.get(file).map_err(|e| match e {
            HFHubApiError::RequestError(err) if matches!(*err, ureq::Error::Status(403, _)) => {
                HFError::AuthorizationError {
                    model_id: model_id.display().to_string(),
                }
            }
            ee => HFError::HFHubApiError(ee),
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_paths(
    model_id: String,
    tokenizer_json: Option<&str>,
    xlora_model_id: Option<&str>,
    xlora_order: Option<&Ordering>,
    chat_template: Option<&str>,
    token_source: &TokenSource,
    revision: Option<String>,
    quantized_model_id: Option<&str>,
    quantized_filenames: Option<Vec<String>>,
    silent: bool,
    loading_uqff: bool,
) -> Result<Box<dyn ModelPaths>, HFError> {
    let token = get_token(token_source).map_err(HFError::HFTokenError)?;
    let api = ApiBuilder::new()
        .with_progress(!silent)
        .with_token(token)
        .build()?;

    let revision = revision.unwrap_or_else(|| "main".to_string());
    let api = api.repo(Repo::with_revision(
        model_id.clone(),
        RepoType::Model,
        revision.clone(),
    ));

    // Get tokenizer path
    let tokenizer_filename = if let Some(p) = tokenizer_json {
        info!("Using tokenizer.json at `{p}`");
        PathBuf::from_str(p).map_err(|_| HFError::FileNotFound {
            model_id: model_id.clone(),
            path: p.to_string(),
        })?
    } else {
        info!("Loading `tokenizer.json` at `{:?}`", model_id);
        api_get_file(&api, "tokenizer.json", &model_id)?
    };

    // Get config path
    info!("Loading `config.json` at `{:?}`", model_id);
    let config_filename = api_get_file(&api, "config.json", &model_id)?;

    // Get model paths
    let filenames = get_model_paths(
        revision.clone(),
        token_source,
        quantized_model_id,
        quantized_filenames,
        &api,
        Path::new(&model_id),
        loading_uqff,
    )?;

    // Get XLora paths
    let XLoraPaths {
        adapter_configs: xlora_adapter_configs,
        adapter_safetensors: xlora_adapter_safetensors,
        classifier_path,
        xlora_order,
        xlora_config,
        lora_preload_adapter_info,
    } = get_xlora_paths(
        model_id.clone(),
        xlora_model_id,
        token_source,
        revision.clone(),
        xlora_order,
    )?;

    // Get optional configs by checking directory contents
    let dir_contents: Vec<String> = api_dir_list(&api, Path::new(&model_id))?;

    let gen_conf = if dir_contents.contains(&"generation_config.json".to_string()) {
        info!("Loading `generation_config.json` at `{}`", model_id);
        Some(api_get_file(&api, "generation_config.json", &model_id)?)
    } else {
        None
    };

    let preprocessor_config = if dir_contents.contains(&"preprocessor_config.json".to_string()) {
        info!("Loading `preprocessor_config.json` at `{}`", model_id);
        Some(api_get_file(&api, "preprocessor_config.json", &model_id)?)
    } else {
        None
    };

    let processor_config = if dir_contents.contains(&"processor_config.json".to_string()) {
        info!("Loading `processor_config.json` at `{}`", model_id);
        Some(api_get_file(&api, "processor_config.json", &model_id)?)
    } else {
        None
    };

    let template_filename = if let Some(ref p) = chat_template {
        info!("Using chat template file at `{p}`");
        Some(PathBuf::from_str(p).map_err(|_| HFError::FileNotFound {
            model_id: model_id.clone(),
            path: p.to_string(),
        })?)
    } else {
        info!("Loading `tokenizer_config.json` at `{}`", model_id);
        Some(api_get_file(&api, "tokenizer_config.json", &model_id)?)
    };

    Ok(Box::new(LocalModelPaths::new(
        tokenizer_filename,
        config_filename,
        template_filename,
        filenames,
        xlora_adapter_safetensors,
        xlora_adapter_configs,
        classifier_path,
        xlora_config,
        xlora_order,
        gen_conf,
        lora_preload_adapter_info,
        preprocessor_config,
        processor_config,
        None,
    )))
}

/// List contents of a directory, either from local filesystem or API
///
/// # Arguments
/// * `api` - The API instance to use when model isn't found locally
/// * `model_id` - Path to check locally before falling back to API
///
/// # Returns
/// * `Result<Vec<String>>` - List of filenames in the directory
pub fn api_dir_list(api: &ApiRepo, model_id: impl AsRef<Path>) -> Result<Vec<String>, HFError> {
    let model_id = model_id.as_ref();

    if model_id.exists() {
        std::fs::read_dir(model_id)
            .map_err(HFError::IoError)?
            .map(|entry| {
                let entry = entry.map_err(HFError::IoError)?;

                let filename = entry
                    .path()
                    .file_name()
                    .ok_or_else(|| HFError::FileNotFound {
                        model_id: model_id.display().to_string(),
                        path: entry.path().display().to_string(),
                    })?
                    .to_str()
                    .ok_or_else(|| HFError::FileNotFound {
                        model_id: model_id.display().to_string(),
                        path: entry.path().display().to_string(),
                    })?
                    .to_string();

                Ok(filename)
            })
            .collect()
    } else {
        // Get listing from API
        let repo = api.info()?;
        Ok(repo.siblings.iter().map(|x| x.rfilename.clone()).collect())
    }
}
