use super::{
    EmbeddingLoaderBuilder, EmbeddingLoaderType, EmbeddingSpecificConfig, Loader, ModelKind,
    ModelPaths, NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig, TokenSource,
    VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
};
use crate::api_get_file;
use crate::utils::tokens::get_token;
use crate::Ordering;
use crate::{DeviceMapSetting, IsqType, PagedAttentionConfig, Pipeline, TryIntoDType};
use anyhow::Result;
use candle_core::Device;
use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};
use serde::Deserialize;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::info;

/// Automatically selects between a normal or vision loader based on the `architectures` field.
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
    architectures: Vec<String>,
}

enum Detected {
    Normal(NormalLoaderType),
    Vision(VisionLoaderType),
    Embedding(EmbeddingLoaderType),
}

impl AutoLoader {
    fn read_config_from_path(&self, paths: &dyn ModelPaths) -> Result<String> {
        Ok(std::fs::read_to_string(paths.get_config_filename())?)
    }

    fn read_config_from_hf(
        &self,
        revision: Option<String>,
        token_source: &TokenSource,
        silent: bool,
    ) -> Result<String> {
        let cache = self
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        let mut api = ApiBuilder::from_cache(cache)
            .with_progress(!silent)
            .with_token(get_token(token_source)?);
        if let Ok(x) = std::env::var("HF_HUB_CACHE") {
            api = api.with_cache_dir(x.into());
        }
        let api = api.build()?;
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision,
        ));
        let model_id = Path::new(&self.model_id);
        let config_filename = api_get_file!(api, "config.json", model_id);
        Ok(std::fs::read_to_string(config_filename)?)
    }

    fn detect(&self, config: &str) -> Result<Detected> {
        let cfg: AutoConfig = serde_json::from_str(config)?;
        if cfg.architectures.len() != 1 {
            anyhow::bail!("Expected exactly one architecture in config");
        }
        let name = &cfg.architectures[0];
        if let Ok(tp) = VisionLoaderType::from_causal_lm_name(name) {
            return Ok(Detected::Vision(tp));
        }
        if let Ok(tp) = EmbeddingLoaderType::from_causal_lm_name(name) {
            return Ok(Detected::Embedding(tp));
        }
        let tp = NormalLoaderType::from_causal_lm_name(name)?;
        Ok(Detected::Normal(tp))
    }

    fn ensure_loader(&self, config: &str) -> Result<()> {
        let mut guard = self.loader.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        match self.detect(config)? {
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
                let loader = builder.build(Some(tp));
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
