use super::hf::{hf_access_error, remote_issue_from_api_error, RemoteAccessIssue};
use super::{
    DiffusionLoaderBuilder, DiffusionLoaderType, EmbeddingLoaderBuilder, EmbeddingLoaderType,
    EmbeddingSpecificConfig, Loader, ModelKind, ModelPaths, NormalLoaderBuilder, NormalLoaderType,
    NormalSpecificConfig, SpeechLoader, TokenSource, VisionLoaderBuilder, VisionLoaderType,
    VisionSpecificConfig,
};
use crate::utils::{progress::ProgressScopeGuard, tokens::get_token};
use crate::Ordering;
use crate::{DeviceMapSetting, IsqType, PagedAttentionConfig, Pipeline, TryIntoDType};
use anyhow::Result;
use candle_core::Device;
use hf_hub::{
    api::sync::{ApiBuilder, ApiError, ApiRepo},
    Cache, Repo, RepoType,
};
use serde::Deserialize;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// Automatically selects the appropriate loader based on repository/config metadata.
pub struct AutoLoader {
    model_id: String,
    normal_builder: Mutex<Option<NormalLoaderBuilder>>,
    vision_builder: Mutex<Option<VisionLoaderBuilder>>,
    embedding_builder: Mutex<Option<EmbeddingLoaderBuilder>>,
    loader: Mutex<Option<Box<dyn Loader>>>,
    hf_cache_path: Option<PathBuf>,
}

pub struct AutoLoaderBuilder {
    normal_cfg: NormalSpecificConfig,
    vision_cfg: VisionSpecificConfig,
    embedding_cfg: EmbeddingSpecificConfig,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    model_id: String,
    jinja_explicit: Option<String>,
    no_kv_cache: bool,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    tgt_non_granular_index: Option<usize>,
    lora_adapter_ids: Option<Vec<String>>,
    hf_cache_path: Option<PathBuf>,
}

impl AutoLoaderBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        normal_cfg: NormalSpecificConfig,
        vision_cfg: VisionSpecificConfig,
        embedding_cfg: EmbeddingSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: String,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        Self {
            normal_cfg,
            vision_cfg,
            embedding_cfg,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            no_kv_cache,
            xlora_model_id: None,
            xlora_order: None,
            tgt_non_granular_index: None,
            lora_adapter_ids: None,
            hf_cache_path: None,
        }
    }

    pub fn with_xlora(
        mut self,
        model_id: String,
        order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.xlora_model_id = Some(model_id);
        self.xlora_order = Some(order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self
    }

    pub fn with_lora(mut self, adapters: Vec<String>) -> Self {
        self.lora_adapter_ids = Some(adapters);
        self
    }

    pub fn hf_cache_path(mut self, path: PathBuf) -> Self {
        self.hf_cache_path = Some(path);
        self
    }

    pub fn build(self) -> Box<dyn Loader> {
        let Self {
            normal_cfg,
            vision_cfg,
            embedding_cfg,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            no_kv_cache,
            xlora_model_id,
            xlora_order,
            tgt_non_granular_index,
            lora_adapter_ids,
            hf_cache_path,
        } = self;

        let mut normal_builder = NormalLoaderBuilder::new(
            normal_cfg,
            chat_template.clone(),
            tokenizer_json.clone(),
            Some(model_id.clone()),
            no_kv_cache,
            jinja_explicit.clone(),
        );
        if let (Some(id), Some(ord)) = (xlora_model_id.clone(), xlora_order.clone()) {
            normal_builder =
                normal_builder.with_xlora(id, ord, no_kv_cache, tgt_non_granular_index);
        }
        if let Some(ref adapters) = lora_adapter_ids {
            normal_builder = normal_builder.with_lora(adapters.clone());
        }
        if let Some(ref path) = hf_cache_path {
            normal_builder = normal_builder.hf_cache_path(path.clone());
        }

        let mut vision_builder = VisionLoaderBuilder::new(
            vision_cfg,
            chat_template,
            tokenizer_json.clone(),
            Some(model_id.clone()),
            jinja_explicit,
        );
        if let Some(ref adapters) = lora_adapter_ids {
            vision_builder = vision_builder.with_lora(adapters.clone());
        }
        if let Some(ref path) = hf_cache_path {
            vision_builder = vision_builder.hf_cache_path(path.clone());
        }

        let mut embedding_builder =
            EmbeddingLoaderBuilder::new(embedding_cfg, tokenizer_json, Some(model_id.clone()));
        if let Some(ref adapters) = lora_adapter_ids {
            embedding_builder = embedding_builder.with_lora(adapters.clone());
        }
        if let Some(ref path) = hf_cache_path {
            embedding_builder = embedding_builder.hf_cache_path(path.clone());
        }

        Box::new(AutoLoader {
            model_id,
            normal_builder: Mutex::new(Some(normal_builder)),
            vision_builder: Mutex::new(Some(vision_builder)),
            embedding_builder: Mutex::new(Some(embedding_builder)),
            loader: Mutex::new(None),
            hf_cache_path,
        })
    }
}

#[derive(Deserialize)]
struct AutoConfig {
    #[serde(default)]
    architectures: Vec<String>,
}

struct ConfigArtifacts {
    contents: Option<String>,
    sentence_transformers_present: bool,
    repo_files: Vec<String>,
    remote_access_issue: Option<RemoteAccessIssue>,
}

enum Detected {
    Normal(NormalLoaderType),
    Vision(VisionLoaderType),
    Embedding(Option<EmbeddingLoaderType>),
    Diffusion(DiffusionLoaderType),
    Speech(crate::speech_models::SpeechLoaderType),
}

impl AutoLoader {
    fn try_get_file(
        api: &ApiRepo,
        model_id: &Path,
        file: &str,
    ) -> std::result::Result<Option<PathBuf>, ApiError> {
        if model_id.exists() {
            let path = model_id.join(file);
            if path.exists() {
                info!("Loading `{}` locally at `{}`", file, path.display());
                Ok(Some(path))
            } else {
                Ok(None)
            }
        } else {
            api.get(file).map(Some)
        }
    }

    fn list_local_repo_files(model_root: &Path) -> Vec<String> {
        fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    collect_files(root, &path, out)?;
                } else if let Ok(rel) = path.strip_prefix(root) {
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
            Ok(())
        }

        if !model_root.is_dir() {
            return Vec::new();
        }

        let mut files = Vec::new();
        if collect_files(model_root, model_root, &mut files).is_err() {
            return Vec::new();
        }
        files
    }

    fn read_config_from_path(&self, paths: &dyn ModelPaths) -> Result<ConfigArtifacts> {
        let config_path = paths.get_config_filename();
        let contents = match std::fs::read_to_string(config_path) {
            Ok(contents) => Some(contents),
            Err(err) if err.kind() == io::ErrorKind::NotFound => None,
            Err(err) => return Err(err.into()),
        };
        let model_root = Path::new(&self.model_id);
        let repo_files = if model_root.exists() {
            Self::list_local_repo_files(model_root)
        } else {
            Vec::new()
        };
        let sentence_transformers_present = Self::has_sentence_transformers_sibling(config_path)
            || repo_files
                .iter()
                .any(|f| f == "config_sentence_transformers.json");
        Ok(ConfigArtifacts {
            contents,
            sentence_transformers_present,
            repo_files,
            remote_access_issue: None,
        })
    }

    fn read_config_from_hf(
        &self,
        revision: Option<String>,
        token_source: &TokenSource,
        silent: bool,
    ) -> Result<ConfigArtifacts> {
        let cache = self
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        let mut api = ApiBuilder::from_cache(cache)
            .with_progress(!silent)
            .with_token(get_token(token_source)?);
        if let Some(cache_dir) = crate::hf_hub_cache_dir() {
            api = api.with_cache_dir(cache_dir);
        }
        let api = api.build()?;
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision,
        ));
        let model_id = Path::new(&self.model_id);
        let mut remote_access_issue = None;
        let contents = match Self::try_get_file(&api, model_id, "config.json") {
            Ok(Some(path)) => Some(std::fs::read_to_string(&path)?),
            Ok(None) => None,
            Err(err) => {
                let issue = remote_issue_from_api_error(model_id, Some("config.json"), &err);
                warn!(
                    "Auto loader could not fetch `config.json` for `{}`: {}",
                    self.model_id, issue.message
                );
                remote_access_issue = Some(issue);
                None
            }
        };
        let sentence_transformers_present =
            model_id.join("config_sentence_transformers.json").exists()
                || Self::fetch_sentence_transformers_config(&api, model_id);
        let repo_files = if model_id.exists() {
            Self::list_local_repo_files(model_id)
        } else {
            crate::api_dir_list!(api, model_id, false).collect::<Vec<_>>()
        };
        Ok(ConfigArtifacts {
            contents,
            sentence_transformers_present,
            repo_files,
            remote_access_issue,
        })
    }

    fn has_sentence_transformers_sibling(config_path: &Path) -> bool {
        config_path
            .parent()
            .map(|parent| parent.join("config_sentence_transformers.json").exists())
            .unwrap_or(false)
    }

    fn fetch_sentence_transformers_config(api: &ApiRepo, model_id: &Path) -> bool {
        if model_id.exists() {
            return false;
        }
        match api.get("config_sentence_transformers.json") {
            Ok(_) => true,
            Err(err) => {
                debug!(
                    "No `config_sentence_transformers.json` found for `{}`: {err}",
                    model_id.display()
                );
                false
            }
        }
    }

    fn detect(&self, artifacts: &ConfigArtifacts) -> Result<Detected> {
        if let Some(tp) = DiffusionLoaderType::auto_detect_from_files(&artifacts.repo_files) {
            return Ok(Detected::Diffusion(tp));
        }

        if let Some(ref config) = artifacts.contents {
            if let Some(tp) =
                crate::speech_models::SpeechLoaderType::auto_detect_from_config(config)
            {
                return Ok(Detected::Speech(tp));
            }
        }

        if artifacts.sentence_transformers_present {
            if let Some(ref config) = artifacts.contents {
                let cfg: AutoConfig = serde_json::from_str(config)?;
                if let Some(name) = cfg.architectures.first() {
                    if let Ok(tp) = EmbeddingLoaderType::from_causal_lm_name(name) {
                        info!(
                            "Detected `config_sentence_transformers.json`; using embedding loader `{tp}`."
                        );
                        return Ok(Detected::Embedding(Some(tp)));
                    }
                }
            }
            if artifacts.contents.is_none() {
                if let Some(issue) = artifacts.remote_access_issue.as_ref() {
                    return Err(hf_access_error(Path::new(&self.model_id), issue));
                }
            }
            info!(
                "Detected `config_sentence_transformers.json`; routing via auto embedding loader."
            );
            return Ok(Detected::Embedding(None));
        }

        // Detect Mistral-native models that use params.json instead of config.json
        if artifacts.contents.is_none() && artifacts.repo_files.iter().any(|f| f == "params.json") {
            // Voxtral uses params.json with a "multimodal" key containing "whisper_model_args"
            info!("Detected `params.json` in repo; routing as Voxtral.");
            return Ok(Detected::Vision(VisionLoaderType::Voxtral));
        }

        let config = artifacts.contents.as_ref().ok_or_else(|| {
            if let Some(issue) = artifacts.remote_access_issue.as_ref() {
                hf_access_error(Path::new(&self.model_id), issue)
            } else {
                anyhow::anyhow!(
                    "Auto loader could not determine model type: missing `config.json` and no diffusion/speech markers found."
                )
            }
        })?;
        let cfg: AutoConfig = serde_json::from_str(config)?;
        if cfg.architectures.len() != 1 {
            anyhow::bail!("Expected exactly one architecture in config");
        }
        let name = &cfg.architectures[0];
        if let Ok(tp) = VisionLoaderType::from_causal_lm_name(name) {
            return Ok(Detected::Vision(tp));
        }
        let tp = NormalLoaderType::from_causal_lm_name(name)?;
        Ok(Detected::Normal(tp))
    }

    fn ensure_loader(&self, artifacts: &ConfigArtifacts) -> Result<()> {
        let mut guard = self.loader.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        match self.detect(artifacts)? {
            Detected::Normal(tp) => {
                let builder = self
                    .normal_builder
                    .lock()
                    .unwrap()
                    .take()
                    .expect("builder taken");
                let loader = builder.build(Some(tp)).expect("build normal");
                *guard = Some(loader);
            }
            Detected::Vision(tp) => {
                let builder = self
                    .vision_builder
                    .lock()
                    .unwrap()
                    .take()
                    .expect("builder taken");
                let loader = builder.build(Some(tp));
                *guard = Some(loader);
            }
            Detected::Embedding(tp) => {
                let builder = self
                    .embedding_builder
                    .lock()
                    .unwrap()
                    .take()
                    .expect("builder taken");
                let loader = builder.build(tp);
                *guard = Some(loader);
            }
            Detected::Diffusion(tp) => {
                let loader = DiffusionLoaderBuilder::new(Some(self.model_id.clone())).build(tp);
                *guard = Some(loader);
            }
            Detected::Speech(tp) => {
                let loader: Box<dyn Loader> = Box::new(SpeechLoader {
                    model_id: self.model_id.clone(),
                    dac_model_id: None,
                    arch: tp,
                    cfg: None,
                });
                *guard = Some(loader);
            }
        }
        Ok(())
    }
}

impl Loader for AutoLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = self.read_config_from_hf(revision.clone(), &token_source, silent)?;
        self.ensure_loader(&config)?;
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .load_model_from_hf(
                revision,
                token_source,
                dtype,
                device,
                silent,
                mapper,
                in_situ_quant,
                paged_attn_config,
            )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = self.read_config_from_path(paths.as_ref())?;
        self.ensure_loader(&config)?;
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .load_model_from_path(
                paths,
                dtype,
                device,
                silent,
                mapper,
                in_situ_quant,
                paged_attn_config,
            )
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .map(|l| l.get_kind())
            .unwrap_or(ModelKind::Normal)
    }
}
