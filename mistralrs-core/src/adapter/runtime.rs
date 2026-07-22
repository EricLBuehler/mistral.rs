use std::{
    collections::HashSet,
    fs::File,
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use candle_core::Tensor;
use mistralrs_quant::{
    LoraConfig, LoraExecution, LoraExecutionArena, LoraLayerRegistry, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{
    registry::{
        AdapterBudgetError, AdapterLease, AdapterRegistry, AdapterRegistryError,
        ResidentAdapterGeneration, ResidentBudget,
    },
    AdapterGenerationId, LoraAdapterInfo, LoraResidentGenerationInfo,
};

/// Default maximum number of loaded aliases and resident adapter generations.
pub const DEFAULT_LORA_MAX_ADAPTERS: usize = 16;
/// Default maximum accepted LoRA rank.
pub const DEFAULT_LORA_MAX_RANK: usize = 256;
/// Default maximum resident adapter tensor bytes.
pub const DEFAULT_LORA_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Maximum UTF-8 byte length accepted for an adapter alias.
pub const MAX_LORA_ALIAS_BYTES: usize = 256;

const LORA_CONFIG_MAX_BYTES: u64 = 1024 * 1024;
const LORA_SOURCE_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const LORA_SAFETENSORS_HEADER_ALLOWANCE: u64 = 16 * 1024 * 1024;
const LORA_SNAPSHOT_BUFFER_SIZE: usize = 1024 * 1024;
static LORA_STAGING_LOCK: Mutex<()> = Mutex::new(());
static LORA_ASYNC_LOAD_GATE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
/// Admission limits for a dynamic LoRA runtime.
pub struct LoraRuntimeConfig {
    /// Maximum loaded aliases and resident generations, including retired generations in use.
    pub max_adapters: usize,
    /// Maximum rank accepted from an adapter configuration.
    pub max_rank: usize,
    /// Maximum bytes held by resident adapter tensors.
    pub max_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
/// Current dynamic LoRA residency and alias state.
pub struct LoraRuntimeStatus {
    /// Loaded request-facing aliases.
    pub adapters: Vec<LoraAdapterInfo>,
    /// Every resident generation, including retired generations held by active leases.
    pub generations: Vec<LoraResidentGenerationInfo>,
    /// Generations currently occupying resident capacity.
    pub resident_generations: usize,
    /// Resident generations kept alive only by admitted requests.
    pub retired_generations: usize,
    /// Bytes currently occupied by resident adapter tensors.
    pub resident_bytes: u64,
    /// Admission limits for this runtime.
    pub limits: LoraRuntimeConfig,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Atomic publication rule for a LoRA adapter load.
pub enum LoraAdapterLoadPolicy {
    /// Publish only when the alias is not already loaded.
    #[default]
    Create,
    /// Create the alias or atomically replace its current generation.
    Upsert,
    /// Replace only when the alias still points at the expected generation.
    CompareAndSwap(AdapterGenerationId),
}

#[derive(Debug)]
/// Already-open adapter files used by trusted lifecycle integrations.
pub struct LoraAdapterFiles {
    source: String,
    revision: Option<String>,
    config_path: PathBuf,
    weights_path: PathBuf,
    config: File,
    weights: File,
}

impl LoraAdapterFiles {
    /// Creates an adapter source from already-open configuration and safetensors files.
    pub fn new(
        source: impl Into<String>,
        config_path: impl Into<PathBuf>,
        config: File,
        weights_path: impl Into<PathBuf>,
        weights: File,
    ) -> Self {
        Self {
            source: source.into(),
            revision: None,
            config_path: config_path.into(),
            weights_path: weights_path.into(),
            config,
            weights,
        }
    }

    /// Records the requested remote revision for adapter provenance.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }
}

impl Default for LoraRuntimeConfig {
    fn default() -> Self {
        Self {
            max_adapters: DEFAULT_LORA_MAX_ADAPTERS,
            max_rank: DEFAULT_LORA_MAX_RANK,
            max_bytes: DEFAULT_LORA_MAX_BYTES,
        }
    }
}

impl LoraRuntimeConfig {
    /// Rejects zero-valued runtime limits.
    pub fn validate(self) -> Result<Self, LoraAdapterError> {
        if self.max_adapters == 0 {
            return Err(LoraAdapterError::InvalidRuntimeConfig(
                "max_adapters must be greater than zero",
            ));
        }
        if self.max_rank == 0 {
            return Err(LoraAdapterError::InvalidRuntimeConfig(
                "max_rank must be greater than zero",
            ));
        }
        if self.max_bytes == 0 {
            return Err(LoraAdapterError::InvalidRuntimeConfig(
                "max_bytes must be greater than zero",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// Error returned by dynamic LoRA lifecycle operations.
pub enum LoraAdapterError {
    #[error("model `{model_id}` has no dynamic LoRA runtime")]
    RuntimeUnavailable { model_id: String },
    #[error("live LoRA adapter updates are unavailable for tensor-parallel model `{model_id}`")]
    TensorParallelUnsupported { model_id: String },
    #[error("dynamic LoRA runtime for model `{model_id}` changed during the operation")]
    RuntimeChanged { model_id: String },
    #[error("adapter alias must not be empty")]
    InvalidAlias,
    #[error("adapter alias is {bytes} bytes, exceeding the {max}-byte limit")]
    AliasTooLong { bytes: usize, max: usize },
    #[error("loaded LoRA adapter alias limit {max} reached")]
    AliasLimit { max: usize },
    #[error("another LoRA adapter load is already in progress")]
    LoadBusy,
    #[error("adapter alias `{alias}` is already loaded at generation `{generation}`")]
    AlreadyLoaded {
        alias: String,
        generation: AdapterGenerationId,
    },
    #[error("adapter alias `{alias}` changed: expected generation `{expected}`, found `{actual}`")]
    GenerationMismatch {
        alias: String,
        expected: AdapterGenerationId,
        actual: AdapterGenerationId,
    },
    #[error("adapter alias `{alias}` is not registered")]
    NotFound { alias: String },
    #[error("adapter generation `{generation}` is not resident")]
    GenerationNotFound { generation: AdapterGenerationId },
    #[error("adapter generation `{generation}` conflicts with resident data")]
    GenerationConflict { generation: AdapterGenerationId },
    #[error("adapter rank {rank} exceeds the configured maximum {max}")]
    RankLimit { rank: usize, max: usize },
    #[error("resident LoRA adapter limit {max} reached")]
    AdapterLimit { max: usize },
    #[error(
        "LoRA adapter requires {requested} bytes with {resident} already resident, exceeding the {max} byte limit"
    )]
    ByteLimit {
        requested: u64,
        resident: u64,
        max: u64,
    },
    #[error("LoRA slot space is exhausted")]
    SlotExhausted,
    #[error("invalid LoRA runtime configuration: {0}")]
    InvalidRuntimeConfig(&'static str),
    #[error("LoRA adapter tensor size overflow")]
    SizeOverflow,
    #[error("LoRA adapter file `{path}` is {bytes} bytes, exceeding the {max}-byte input limit")]
    FileTooLarge { path: PathBuf, bytes: u64, max: u64 },
    #[error("failed to read LoRA adapter file `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse LoRA adapter config `{path}`: {source}")]
    Config {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid LoRA adapter tensors: {0}")]
    Format(#[source] candle_core::Error),
    #[error("failed to load LoRA adapter tensors onto model devices: {0}")]
    Load(#[source] candle_core::Error),
    #[error("LoRA adapter blocking task failed: {0}")]
    Task(#[source] tokio::task::JoinError),
}

impl From<AdapterRegistryError> for LoraAdapterError {
    fn from(value: AdapterRegistryError) -> Self {
        match value {
            AdapterRegistryError::EmptyAlias => Self::InvalidAlias,
            AdapterRegistryError::AliasTooLong { bytes, max } => Self::AliasTooLong { bytes, max },
            AdapterRegistryError::AliasLimit { max } => Self::AliasLimit { max },
            AdapterRegistryError::AliasNotFound(alias) => Self::NotFound { alias },
            AdapterRegistryError::GenerationNotFound(generation) => {
                Self::GenerationNotFound { generation }
            }
            AdapterRegistryError::GenerationConflict(generation) => {
                Self::GenerationConflict { generation }
            }
            AdapterRegistryError::SlotExhausted => Self::SlotExhausted,
        }
    }
}

impl From<AdapterBudgetError> for LoraAdapterError {
    fn from(value: AdapterBudgetError) -> Self {
        match value {
            AdapterBudgetError::AdapterLimit { max } => Self::AdapterLimit { max },
            AdapterBudgetError::ByteLimit {
                requested,
                resident,
                max,
            } => Self::ByteLimit {
                requested,
                resident,
                max,
            },
        }
    }
}

pub struct DynamicLoraRuntime {
    layers: Arc<LoraLayerRegistry>,
    adapters: AdapterRegistry,
    budget: Arc<ResidentBudget>,
    limits: LoraRuntimeConfig,
    live_updates: bool,
    mutation_lock: Mutex<()>,
    execution_arena: Arc<LoraExecutionArena>,
}

impl std::fmt::Debug for DynamicLoraRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicLoraRuntime")
            .field("layers", &self.layers)
            .field("adapters", &self.adapters)
            .field("limits", &self.limits)
            .field("live_updates", &self.live_updates)
            .finish()
    }
}

impl DynamicLoraRuntime {
    pub(crate) fn try_acquire_load_permit(
    ) -> Result<tokio::sync::OwnedSemaphorePermit, LoraAdapterError> {
        LORA_ASYNC_LOAD_GATE
            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(1)))
            .clone()
            .try_acquire_owned()
            .map_err(|_| LoraAdapterError::LoadBusy)
    }

    pub(crate) fn new(
        layers: Arc<LoraLayerRegistry>,
        limits: LoraRuntimeConfig,
        live_updates: bool,
    ) -> Result<Self, LoraAdapterError> {
        let limits = limits.validate()?;
        Ok(Self {
            layers,
            adapters: AdapterRegistry::new(limits.max_adapters, MAX_LORA_ALIAS_BYTES),
            budget: Arc::new(ResidentBudget::new(limits)),
            limits,
            live_updates,
            mutation_lock: Mutex::new(()),
            execution_arena: Arc::new(LoraExecutionArena::new()),
        })
    }

    pub(crate) fn supports_live_updates(&self) -> bool {
        self.live_updates
    }

    fn install_inner(
        &self,
        alias: String,
        source: String,
        revision: Option<String>,
        generation: AdapterGenerationId,
        config: &LoraConfig,
        weights: &ShardedVarBuilder,
    ) -> Result<LoraAdapterInfo, LoraAdapterError> {
        let alias = alias.trim().to_string();
        if alias.is_empty() {
            return Err(LoraAdapterError::InvalidAlias);
        }
        if alias.len() > MAX_LORA_ALIAS_BYTES {
            return Err(LoraAdapterError::AliasTooLong {
                bytes: alias.len(),
                max: MAX_LORA_ALIAS_BYTES,
            });
        }
        if let Ok(existing) = self.adapters.resolve_generation(generation) {
            self.adapters
                .register_lease(alias.clone(), source, revision, &existing)?;
            return self.adapters.info(&alias).map_err(Into::into);
        }

        let rank = self.validate_adapter_config(config)?;
        let plan = mistralrs_quant::plan_dynamic_lora_weights(&self.layers, config, weights)
            .map_err(LoraAdapterError::Format)?;
        let bytes = plan.bytes();
        let reservation = self.budget.reserve(bytes)?;
        let loaded = mistralrs_quant::load_dynamic_lora_weights(&self.layers, config, weights)
            .map_err(LoraAdapterError::Load)?;
        if adapter_bytes(&loaded)? != bytes {
            return Err(LoraAdapterError::Load(candle_core::Error::msg(
                "LoRA preflight size did not match loaded tensors",
            )));
        }
        let weights = Arc::new(
            mistralrs_quant::LoraAdapterWeights::new(self.layers.runtime_id(), loaded)
                .map_err(LoraAdapterError::Load)?,
        );
        let slot = self.adapters.allocate_slot()?;
        let resident = Arc::new(ResidentAdapterGeneration::new(
            self.layers.runtime_id(),
            generation,
            slot,
            rank,
            bytes,
            weights,
            reservation,
        ));
        self.adapters
            .register(alias.clone(), source, revision, resident)?;
        self.adapters.info(&alias).map_err(Into::into)
    }

    fn validate_adapter_config(&self, config: &LoraConfig) -> Result<usize, LoraAdapterError> {
        config
            .validate_dynamic()
            .map_err(LoraAdapterError::Format)?;
        let rank = config
            .rank_pattern
            .values()
            .copied()
            .max()
            .unwrap_or(config.rank)
            .max(config.rank);
        if rank > self.limits.max_rank {
            return Err(LoraAdapterError::RankLimit {
                rank,
                max: self.limits.max_rank,
            });
        }
        Ok(rank)
    }

    #[cfg(test)]
    pub(crate) fn load_from_directory(
        &self,
        alias: impl Into<String>,
        adapter_dir: impl AsRef<Path>,
    ) -> Result<LoraAdapterInfo, LoraAdapterError> {
        self.load_from_directory_with_policy(alias, adapter_dir, LoraAdapterLoadPolicy::Upsert)
    }

    pub(crate) fn load_from_directory_with_policy(
        &self,
        alias: impl Into<String>,
        adapter_dir: impl AsRef<Path>,
        policy: LoraAdapterLoadPolicy,
    ) -> Result<LoraAdapterInfo, LoraAdapterError> {
        let adapter_dir = adapter_dir.as_ref();
        let config_path = adapter_dir.join("adapter_config.json");
        let weights_path = adapter_dir.join("adapter_model.safetensors");
        let config = open_file(&config_path)?;
        let weights = open_file(&weights_path)?;
        self.load_from_files_with_policy(
            alias,
            LoraAdapterFiles::new(
                adapter_dir.display().to_string(),
                config_path,
                config,
                weights_path,
                weights,
            ),
            policy,
        )
    }

    pub(crate) fn load_from_safetensors(
        &self,
        alias: impl Into<String>,
        source: impl Into<String>,
        revision: Option<String>,
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
    ) -> Result<LoraAdapterInfo, LoraAdapterError> {
        let config_path = config_path.as_ref();
        let weights_path = weights_path.as_ref();
        let config = open_file(config_path)?;
        let weights = open_file(weights_path)?;
        let mut files = LoraAdapterFiles::new(source, config_path, config, weights_path, weights);
        if let Some(revision) = revision {
            files = files.with_revision(revision);
        }
        self.load_from_files(alias, files)
    }

    pub(crate) fn load_from_files(
        &self,
        alias: impl Into<String>,
        files: LoraAdapterFiles,
    ) -> Result<LoraAdapterInfo, LoraAdapterError> {
        self.load_from_files_with_policy(alias, files, LoraAdapterLoadPolicy::Upsert)
    }

    pub(crate) fn load_from_files_with_policy(
        &self,
        alias: impl Into<String>,
        mut files: LoraAdapterFiles,
        policy: LoraAdapterLoadPolicy,
    ) -> Result<LoraAdapterInfo, LoraAdapterError> {
        let alias = alias.into().trim().to_string();
        self.adapters.validate_registration(&alias)?;
        self.validate_load_policy(&alias, policy)?;
        let _staging_guard = LORA_STAGING_LOCK
            .lock()
            .expect("LoRA staging lock poisoned");
        let config_bytes =
            read_limited_file(&mut files.config, &files.config_path, LORA_CONFIG_MAX_BYTES)?;
        let config = serde_json::from_slice::<LoraConfig>(&config_bytes).map_err(|source| {
            LoraAdapterError::Config {
                path: files.config_path.clone(),
                source,
            }
        })?;
        self.validate_adapter_config(&config)?;
        let resident_scaled_limit = self.limits.max_bytes.saturating_mul(8);
        let weights_limit = resident_scaled_limit
            .min(LORA_SOURCE_MAX_BYTES)
            .checked_add(LORA_SAFETENSORS_HEADER_ALLOWANCE)
            .ok_or(LoraAdapterError::SizeOverflow)?;
        let (snapshot, weights_digest) =
            snapshot_file(&mut files.weights, &files.weights_path, weights_limit)?;
        let config_digest = Sha256::digest(&config_bytes).into();
        let generation = AdapterGenerationId::from_adapter_digests(config_digest, weights_digest);
        let weights = crate::utils::varbuilder_utils::from_mmaped_safetensors(
            vec![snapshot.path().to_path_buf()],
            Vec::new(),
            Some(candle_core::DType::F32),
            &candle_core::Device::Cpu,
            Vec::new(),
            true,
            None,
            |_| true,
            Arc::new(|_| crate::utils::varbuilder_utils::DeviceForLoadTensor::Base),
        )
        .map_err(LoraAdapterError::Format)?;
        let _guard = self
            .mutation_lock
            .lock()
            .expect("adapter mutation lock poisoned");
        self.validate_load_policy(&alias, policy)?;
        self.install_inner(
            alias,
            files.source,
            files.revision,
            generation,
            &config,
            &weights,
        )
    }

    fn validate_load_policy(
        &self,
        alias: &str,
        policy: LoraAdapterLoadPolicy,
    ) -> Result<(), LoraAdapterError> {
        let current = match self.adapters.info(alias) {
            Ok(info) => Some(info),
            Err(AdapterRegistryError::AliasNotFound(_)) => None,
            Err(error) => return Err(error.into()),
        };
        match (policy, current) {
            (LoraAdapterLoadPolicy::Create, Some(info)) => Err(LoraAdapterError::AlreadyLoaded {
                alias: alias.to_string(),
                generation: info.generation,
            }),
            (LoraAdapterLoadPolicy::CompareAndSwap(expected), Some(info))
                if info.generation != expected =>
            {
                Err(LoraAdapterError::GenerationMismatch {
                    alias: alias.to_string(),
                    expected,
                    actual: info.generation,
                })
            }
            (LoraAdapterLoadPolicy::CompareAndSwap(_), None) => Err(LoraAdapterError::NotFound {
                alias: alias.to_string(),
            }),
            _ => Ok(()),
        }
    }

    #[cfg(test)]
    pub(crate) fn unload(&self, alias: &str) -> Result<LoraAdapterInfo, LoraAdapterError> {
        self.unload_if_generation(alias, None)
    }

    pub(crate) fn unload_if_generation(
        &self,
        alias: &str,
        expected_generation: Option<AdapterGenerationId>,
    ) -> Result<LoraAdapterInfo, LoraAdapterError> {
        let alias = alias.trim();
        if alias.is_empty() {
            return Err(LoraAdapterError::InvalidAlias);
        }
        let _guard = self
            .mutation_lock
            .lock()
            .expect("adapter mutation lock poisoned");
        if let Some(expected) = expected_generation {
            let current = self.adapters.info(alias)?;
            if current.generation != expected {
                return Err(LoraAdapterError::GenerationMismatch {
                    alias: alias.to_string(),
                    expected,
                    actual: current.generation,
                });
            }
        }
        self.adapters.unload(alias).map_err(Into::into)
    }

    pub(crate) fn list(&self) -> Vec<LoraAdapterInfo> {
        self.adapters.list()
    }

    pub(crate) fn status(&self) -> LoraRuntimeStatus {
        let snapshot = self.adapters.snapshot();
        LoraRuntimeStatus {
            adapters: snapshot.adapters,
            generations: snapshot.generations,
            resident_generations: snapshot.resident_generations,
            retired_generations: snapshot.retired_generations,
            resident_bytes: snapshot.resident_bytes,
            limits: self.limits,
        }
    }

    pub(crate) fn resolve_alias(&self, alias: &str) -> Result<AdapterLease, LoraAdapterError> {
        let alias = alias.trim();
        if alias.is_empty() {
            return Err(LoraAdapterError::InvalidAlias);
        }
        self.adapters.resolve_alias(alias).map_err(Into::into)
    }

    pub(crate) fn resolve_generation(
        &self,
        generation: AdapterGenerationId,
    ) -> Result<AdapterLease, LoraAdapterError> {
        self.adapters
            .resolve_generation(generation)
            .map_err(Into::into)
    }

    pub(crate) fn execution(
        &self,
        adapter_leases: &[Option<AdapterLease>],
        sequence_length: usize,
    ) -> candle_core::Result<Arc<LoraExecution>> {
        let sequence_slots = adapter_leases
            .iter()
            .map(|lease| lease.as_ref().map(AdapterLease::slot))
            .collect::<Vec<_>>();
        let mut execution = LoraExecution::from_sequence_slots_with_arena(
            self.layers.runtime_id(),
            &sequence_slots,
            sequence_length,
            self.execution_arena.clone(),
        );
        let mut installed = HashSet::new();
        for lease in adapter_leases.iter().flatten() {
            let slot = lease.slot();
            if !installed.insert(slot) {
                continue;
            }
            let resident = lease.resident();
            if resident.runtime_id() != self.layers.runtime_id() {
                candle_core::bail!(
                    "adapter generation `{}` belongs to a different runtime",
                    lease.generation()
                );
            }
            execution.insert_adapter(slot, resident.weights().clone())?;
        }
        Ok(Arc::new(execution))
    }
}

fn open_file(path: &Path) -> Result<File, LoraAdapterError> {
    File::open(path).map_err(|source| LoraAdapterError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_limited_file(file: &mut File, path: &Path, max: u64) -> Result<Vec<u8>, LoraAdapterError> {
    file.rewind().map_err(|source| LoraAdapterError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let bytes = file
        .metadata()
        .map_err(|source| LoraAdapterError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if bytes > max {
        return Err(LoraAdapterError::FileTooLarge {
            path: path.to_path_buf(),
            bytes,
            max,
        });
    }
    let capacity = usize::try_from(bytes).map_err(|_| LoraAdapterError::SizeOverflow)?;
    let mut output = Vec::with_capacity(capacity);
    file.take(max + 1)
        .read_to_end(&mut output)
        .map_err(|source| LoraAdapterError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if output.len() as u64 > max {
        return Err(LoraAdapterError::FileTooLarge {
            path: path.to_path_buf(),
            bytes: output.len() as u64,
            max,
        });
    }
    Ok(output)
}

fn snapshot_file(
    source: &mut File,
    path: &Path,
    max: u64,
) -> Result<(tempfile::NamedTempFile, [u8; 32]), LoraAdapterError> {
    source.rewind().map_err(|source| LoraAdapterError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let bytes = source
        .metadata()
        .map_err(|source| LoraAdapterError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if bytes > max {
        return Err(LoraAdapterError::FileTooLarge {
            path: path.to_path_buf(),
            bytes,
            max,
        });
    }

    let mut snapshot = tempfile::NamedTempFile::new().map_err(|source| LoraAdapterError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut remaining = bytes;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; LORA_SNAPSHOT_BUFFER_SIZE];
    while remaining != 0 {
        let limit = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| LoraAdapterError::SizeOverflow)?;
        let read = source
            .read(&mut buffer[..limit])
            .map_err(|source| LoraAdapterError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            return Err(LoraAdapterError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "adapter file changed while it was being snapshotted",
                ),
            });
        }
        hasher.update(&buffer[..read]);
        snapshot
            .write_all(&buffer[..read])
            .map_err(|source| LoraAdapterError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        remaining -= read as u64;
    }
    snapshot.flush().map_err(|source| LoraAdapterError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok((snapshot, hasher.finalize().into()))
}

fn adapter_bytes(weights: &mistralrs_quant::DynamicLoraWeights) -> Result<u64, LoraAdapterError> {
    fn add_tensor(
        total: u64,
        tensor: &Tensor,
        counted: &mut HashSet<candle_core::TensorId>,
    ) -> Result<u64, LoraAdapterError> {
        if !counted.insert(tensor.id()) {
            return Ok(total);
        }
        let elements =
            u64::try_from(tensor.elem_count()).map_err(|_| LoraAdapterError::SizeOverflow)?;
        let element_bytes = u64::try_from(tensor.dtype().size_in_bytes())
            .map_err(|_| LoraAdapterError::SizeOverflow)?;
        total
            .checked_add(
                elements
                    .checked_mul(element_bytes)
                    .ok_or(LoraAdapterError::SizeOverflow)?,
            )
            .ok_or(LoraAdapterError::SizeOverflow)
    }

    let mut total = 0u64;
    let mut counted = HashSet::new();
    for (_, weights) in &weights.linear {
        total = add_tensor(total, weights.a(), &mut counted)?;
        total = add_tensor(total, weights.b(), &mut counted)?;
    }
    for (_, weights) in &weights.experts {
        for projection in [weights.gate(), weights.up(), weights.down()]
            .into_iter()
            .flatten()
        {
            total = add_tensor(total, projection.a(), &mut counted)?;
            total = add_tensor(total, projection.b(), &mut counted)?;
            total = add_tensor(total, projection.scales(), &mut counted)?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;
    use mistralrs_quant::{
        maybe_wrap_dynamic_lora, with_lora_execution, LoraExpertInputMode, LoraExpertProjection,
        LoraExpertProjectionNames, LoraExpertSiteSpec, LoraLinearSpec, LoraSiteKey, QuantMethod,
        QuantMethodConfig, Shard, ShardedSafeTensors, UnquantLinear,
    };

    use super::*;
    use crate::AdapterSelection;

    fn write_adapter(dir: &Path, scale: f32) {
        std::fs::write(
            dir.join("adapter_config.json"),
            r#"{"r":1,"lora_alpha":1,"target_modules":["proj"]}"#,
        )
        .unwrap();
        let tensors = HashMap::from([
            (
                "base_model.model.proj.lora_A.weight".to_string(),
                Tensor::new(&[[1f32, 0.]], &Device::Cpu).unwrap(),
            ),
            (
                "base_model.model.proj.lora_B.weight".to_string(),
                Tensor::new(&[[scale], [0.]], &Device::Cpu).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors")).unwrap();
    }

    fn write_expert_adapter(dir: &Path) {
        std::fs::write(
            dir.join("adapter_config.json"),
            r#"{"r":1,"lora_alpha":1,"target_modules":["gate_proj"]}"#,
        )
        .unwrap();
        let root = "base_model.model.model.layers.0.mlp.experts";
        let tensors = HashMap::from([
            (
                format!("{root}.0.gate_proj.lora_A.weight"),
                Tensor::new(&[[1f32, 0.]], &Device::Cpu).unwrap(),
            ),
            (
                format!("{root}.0.gate_proj.lora_B.weight"),
                Tensor::new(&[[1f32], [2.]], &Device::Cpu).unwrap(),
            ),
            (
                format!("{root}.1.gate_proj.lora_A.weight"),
                Tensor::new(&[[0f32, 1.]], &Device::Cpu).unwrap(),
            ),
            (
                format!("{root}.1.gate_proj.lora_B.weight"),
                Tensor::new(&[[3f32], [4.]], &Device::Cpu).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors")).unwrap();
    }

    fn write_fused_expert_adapter(dir: &Path) {
        std::fs::write(
            dir.join("adapter_config.json"),
            r#"{"r":1,"lora_alpha":1,"target_parameters":["mlp.experts.gate_up_proj","mlp.experts.down_proj"]}"#,
        )
        .unwrap();
        let root = "base_model.model.model.layers.0.mlp.experts";
        let tensors = HashMap::from([
            (
                format!("{root}.base_layer.lora_A.weight"),
                Tensor::ones((2, 2), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                format!("{root}.base_layer.lora_B.weight"),
                Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                format!("{root}.lora_A.weight"),
                Tensor::ones((2, 2), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                format!("{root}.lora_B.weight"),
                Tensor::ones((2, 2), DType::F32, &Device::Cpu).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors")).unwrap();
    }

    fn expert_registry() -> (
        Arc<LoraLayerRegistry>,
        Arc<mistralrs_quant::LoraExpertSiteHandle>,
    ) {
        let registry = Arc::new(LoraLayerRegistry::new());
        let site = registry
            .register_expert(
                LoraSiteKey::new("model.layers.0.mlp.experts"),
                LoraExpertSiteSpec::new(
                    2,
                    2,
                    2,
                    LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
                    Shard::default(),
                    Shard::default(),
                )
                .unwrap(),
                DType::F32,
                Device::Cpu,
            )
            .unwrap();
        registry.finalize().unwrap();
        (registry, site)
    }

    fn runtime_and_layer(
        limits: LoraRuntimeConfig,
    ) -> (Arc<DynamicLoraRuntime>, Arc<dyn QuantMethod>) {
        let layers = Arc::new(LoraLayerRegistry::new());
        let vb = ShardedSafeTensors::wrap_with_dummy_regexes(
            HashMap::new(),
            DType::F32,
            Device::Cpu,
            None,
        )
        .with_lora_registry(layers.clone())
        .pp("proj");
        let base: Arc<dyn QuantMethod> = Arc::new(
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
                Tensor::new(&[[1f32, 0.], [0., 1.]], &Device::Cpu).unwrap(),
                None,
            )))
            .unwrap(),
        );
        let layer = maybe_wrap_dynamic_lora(&vb, base, LoraLinearSpec::replicated(2, 2)).unwrap();
        layers.finalize().unwrap();
        let runtime = Arc::new(DynamicLoraRuntime::new(layers, limits, true).unwrap());
        (runtime, layer)
    }

    fn forward(
        runtime: &DynamicLoraRuntime,
        layer: &dyn QuantMethod,
        lease: AdapterLease,
    ) -> Vec<Vec<f32>> {
        let execution = runtime.execution(&[Some(lease)], 1).unwrap();
        with_lora_execution(Some(execution), || {
            layer
                .forward(&Tensor::new(&[[2f32, 3.]], &Device::Cpu).unwrap())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
        })
    }

    #[test]
    fn directory_load_and_alias_swap_pin_exact_generations() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        write_adapter(first_dir.path(), 1.0);
        write_adapter(second_dir.path(), 2.0);
        let (runtime, layer) = runtime_and_layer(LoraRuntimeConfig::default());

        let first = runtime
            .load_from_directory("production", first_dir.path())
            .unwrap();
        let old_lease = runtime.resolve_alias("production").unwrap();
        assert_eq!(
            forward(&runtime, &*layer, old_lease.clone()),
            vec![vec![4., 3.]]
        );

        let second = runtime
            .load_from_directory("production", second_dir.path())
            .unwrap();
        let new_lease = runtime.resolve_alias("production").unwrap();
        assert_ne!(first.generation, second.generation);
        assert_eq!(forward(&runtime, &*layer, old_lease), vec![vec![4., 3.]]);
        assert_eq!(forward(&runtime, &*layer, new_lease), vec![vec![6., 3.]]);
        assert_eq!(runtime.list(), vec![second]);
    }

    #[test]
    fn expert_directory_load_runs_through_runtime_execution() {
        let adapter_dir = tempfile::tempdir().unwrap();
        write_expert_adapter(adapter_dir.path());
        let (registry, site) = expert_registry();
        let runtime = DynamicLoraRuntime::new(
            registry,
            LoraRuntimeConfig {
                max_bytes: 40,
                ..LoraRuntimeConfig::default()
            },
            true,
        )
        .unwrap();
        let info = runtime
            .load_from_directory("domain", adapter_dir.path())
            .unwrap();
        assert_eq!(info.bytes, 40);
        assert_eq!(runtime.status().resident_bytes, 40);

        let lease = runtime.resolve_alias("domain").unwrap();
        let execution = runtime.execution(&[Some(lease)], 1).unwrap();
        let output = with_lora_execution(Some(execution), || {
            let lora = mistralrs_quant::LoraExpertExecution::current(&site)
                .unwrap()
                .expect("active expert adapter");
            lora.add_delta(
                LoraExpertProjection::Gate,
                &Tensor::new(&[[2f32, 3.]], &Device::Cpu).unwrap(),
                Tensor::zeros((1, 2, 2), DType::F32, &Device::Cpu).unwrap(),
                &Tensor::new(&[[0u32, 1]], &Device::Cpu).unwrap(),
                None,
                LoraExpertInputMode::TokenRows,
            )
            .unwrap()
            .to_vec3::<f32>()
            .unwrap()
        });
        assert_eq!(output, vec![vec![vec![2., 4.], vec![9., 12.]]]);
    }

    #[test]
    fn fused_expert_budget_counts_shared_tensors_once() {
        let adapter_dir = tempfile::tempdir().unwrap();
        write_fused_expert_adapter(adapter_dir.path());
        let (registry, _) = expert_registry();
        let runtime = DynamicLoraRuntime::new(
            registry,
            LoraRuntimeConfig {
                max_bytes: 96,
                ..LoraRuntimeConfig::default()
            },
            true,
        )
        .unwrap();

        let info = runtime
            .load_from_directory("domain", adapter_dir.path())
            .unwrap();
        assert_eq!(info.bytes, 96);
        assert_eq!(runtime.status().resident_bytes, 96);
    }

    #[test]
    fn publication_policies_and_conditional_unload_are_atomic() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        write_adapter(first_dir.path(), 1.0);
        write_adapter(second_dir.path(), 2.0);
        let (runtime, _) = runtime_and_layer(LoraRuntimeConfig::default());

        let first = runtime
            .load_from_directory_with_policy(
                "production",
                first_dir.path(),
                LoraAdapterLoadPolicy::Create,
            )
            .unwrap();
        assert!(matches!(
            runtime.load_from_directory_with_policy(
                "production",
                second_dir.path(),
                LoraAdapterLoadPolicy::Create,
            ),
            Err(LoraAdapterError::AlreadyLoaded { .. })
        ));
        assert_eq!(runtime.list(), vec![first.clone()]);

        let stale = AdapterGenerationId::from_bytes([9; 32]);
        assert!(matches!(
            runtime.load_from_directory_with_policy(
                "production",
                second_dir.path(),
                LoraAdapterLoadPolicy::CompareAndSwap(stale),
            ),
            Err(LoraAdapterError::GenerationMismatch { .. })
        ));
        let second = runtime
            .load_from_directory_with_policy(
                "production",
                second_dir.path(),
                LoraAdapterLoadPolicy::CompareAndSwap(first.generation),
            )
            .unwrap();
        assert!(matches!(
            runtime.unload_if_generation("production", Some(first.generation)),
            Err(LoraAdapterError::GenerationMismatch { .. })
        ));
        assert_eq!(runtime.list(), vec![second.clone()]);
        assert_eq!(
            runtime
                .unload_if_generation("production", Some(second.generation))
                .unwrap(),
            second
        );
    }

    #[test]
    fn default_load_policy_does_not_replace_an_alias() {
        assert_eq!(
            LoraAdapterLoadPolicy::default(),
            LoraAdapterLoadPolicy::Create
        );
    }

    #[test]
    fn pinned_selection_serializes_as_its_generation() {
        let adapter_dir = tempfile::tempdir().unwrap();
        write_adapter(adapter_dir.path(), 1.0);
        let (runtime, _) = runtime_and_layer(LoraRuntimeConfig::default());
        let info = runtime
            .load_from_directory("production", adapter_dir.path())
            .unwrap();
        let mut selection = AdapterSelection::alias("production");
        selection.pin(&runtime).unwrap();

        let serialized = serde_json::to_string(&selection).unwrap();
        let replicated: AdapterSelection = serde_json::from_str(&serialized).unwrap();
        assert_eq!(replicated.resolved_generation(), Some(info.generation));
        assert!(serialized.contains("generation"));
        assert!(!serialized.contains("production"));
    }

    #[test]
    fn resident_budget_counts_in_flight_retired_generations() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        write_adapter(first_dir.path(), 1.0);
        write_adapter(second_dir.path(), 2.0);
        let limits = LoraRuntimeConfig {
            max_adapters: 1,
            ..LoraRuntimeConfig::default()
        };
        let (runtime, _) = runtime_and_layer(limits);
        runtime
            .load_from_directory("production", first_dir.path())
            .unwrap();
        let lease = runtime.resolve_alias("production").unwrap();
        runtime.unload("production").unwrap();

        let status = runtime.status();
        assert!(status.adapters.is_empty());
        assert_eq!(status.resident_generations, 1);
        assert_eq!(status.retired_generations, 1);
        assert_eq!(status.resident_bytes, 16);
        assert_eq!(status.generations.len(), 1);
        assert!(status.generations[0].retired);
        assert_eq!(status.generations[0].active_leases, 1);

        assert!(matches!(
            runtime.load_from_directory("production", second_dir.path()),
            Err(LoraAdapterError::AdapterLimit { max: 1 })
        ));
        drop(lease);
        assert_eq!(runtime.status().resident_generations, 0);
        runtime
            .load_from_directory("production", second_dir.path())
            .unwrap();
    }

    #[test]
    fn pending_reservations_are_not_reported_as_resident() {
        let (runtime, _) = runtime_and_layer(LoraRuntimeConfig::default());
        let reservation = runtime.budget.reserve(16).unwrap();

        let status = runtime.status();
        assert!(status.adapters.is_empty());
        assert_eq!(status.resident_generations, 0);
        assert_eq!(status.retired_generations, 0);
        assert_eq!(status.resident_bytes, 0);

        drop(reservation);
    }

    #[test]
    fn atomic_swap_requires_generation_and_byte_headroom() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        write_adapter(first_dir.path(), 1.0);
        write_adapter(second_dir.path(), 2.0);
        let limits = LoraRuntimeConfig {
            max_adapters: 1,
            max_bytes: 16,
            ..LoraRuntimeConfig::default()
        };
        let (runtime, layer) = runtime_and_layer(limits);
        let first = runtime
            .load_from_directory("production", first_dir.path())
            .unwrap();

        assert!(matches!(
            runtime.load_from_directory("production", second_dir.path()),
            Err(LoraAdapterError::AdapterLimit { max: 1 })
        ));
        assert_eq!(runtime.list(), vec![first]);

        runtime.unload("production").unwrap();
        let second = runtime
            .load_from_directory("production", second_dir.path())
            .unwrap();
        let lease = runtime.resolve_alias("production").unwrap();
        assert_eq!(second.bytes, 16);
        assert_eq!(forward(&runtime, &*layer, lease), vec![vec![6., 3.]]);
    }

    #[test]
    fn byte_limit_is_checked_from_metadata_before_installation() {
        let adapter_dir = tempfile::tempdir().unwrap();
        write_adapter(adapter_dir.path(), 1.0);
        let limits = LoraRuntimeConfig {
            max_bytes: 15,
            ..LoraRuntimeConfig::default()
        };
        let (runtime, _) = runtime_and_layer(limits);
        assert!(matches!(
            runtime.load_from_directory("production", adapter_dir.path()),
            Err(LoraAdapterError::ByteLimit {
                requested: 16,
                resident: 0,
                max: 15,
            })
        ));
        assert!(runtime.list().is_empty());
    }

    #[test]
    fn pinned_swap_generation_remains_visible_and_consumes_capacity() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let third_dir = tempfile::tempdir().unwrap();
        write_adapter(first_dir.path(), 1.0);
        write_adapter(second_dir.path(), 2.0);
        write_adapter(third_dir.path(), 3.0);
        let limits = LoraRuntimeConfig {
            max_adapters: 2,
            ..LoraRuntimeConfig::default()
        };
        let (runtime, layer) = runtime_and_layer(limits);
        let first = runtime
            .load_from_directory("production", first_dir.path())
            .unwrap();
        let lease = runtime.resolve_alias("production").unwrap();
        let second = runtime
            .load_from_directory("production", second_dir.path())
            .unwrap();

        let status = runtime.status();
        assert_eq!(status.adapters, vec![second.clone()]);
        assert_eq!(status.resident_generations, 2);
        assert_eq!(status.retired_generations, 1);
        assert_eq!(status.resident_bytes, 32);

        let exact = runtime.resolve_generation(first.generation).unwrap();
        assert_eq!(
            forward(&runtime, &*layer, exact.clone()),
            vec![vec![4., 3.]]
        );

        assert!(matches!(
            runtime.load_from_directory("production", third_dir.path()),
            Err(LoraAdapterError::AdapterLimit { max: 2 })
        ));
        assert_eq!(runtime.list(), vec![second]);

        drop(lease);
        assert_eq!(runtime.status().resident_generations, 2);
        drop(exact);
        assert_eq!(runtime.status().resident_generations, 1);
        assert!(matches!(
            runtime.resolve_generation(first.generation),
            Err(LoraAdapterError::GenerationNotFound { .. })
        ));
        runtime
            .load_from_directory("production", third_dir.path())
            .unwrap();
    }

    #[test]
    fn identical_generations_share_resident_budget() {
        let adapter_dir = tempfile::tempdir().unwrap();
        write_adapter(adapter_dir.path(), 1.0);
        let limits = LoraRuntimeConfig {
            max_adapters: 2,
            ..LoraRuntimeConfig::default()
        };
        let (runtime, _) = runtime_and_layer(limits);
        let first = runtime
            .load_from_directory("first", adapter_dir.path())
            .unwrap();
        let second = runtime
            .load_from_directory("second", adapter_dir.path())
            .unwrap();
        assert_eq!(first.generation, second.generation);
        assert_eq!(runtime.list().len(), 2);
        assert_eq!(runtime.status().resident_generations, 1);
    }

    #[test]
    fn alias_count_is_bounded_even_when_generations_are_shared() {
        let adapter_dir = tempfile::tempdir().unwrap();
        write_adapter(adapter_dir.path(), 1.0);
        let limits = LoraRuntimeConfig {
            max_adapters: 1,
            ..LoraRuntimeConfig::default()
        };
        let (runtime, _) = runtime_and_layer(limits);
        runtime
            .load_from_directory("first", adapter_dir.path())
            .unwrap();

        assert!(matches!(
            runtime.load_from_directory("second", adapter_dir.path()),
            Err(LoraAdapterError::AliasLimit { max: 1 })
        ));
        assert_eq!(runtime.list().len(), 1);
    }

    #[test]
    fn alias_length_is_bounded_before_loading() {
        let adapter_dir = tempfile::tempdir().unwrap();
        write_adapter(adapter_dir.path(), 1.0);
        let (runtime, _) = runtime_and_layer(LoraRuntimeConfig::default());
        let alias = "a".repeat(MAX_LORA_ALIAS_BYTES + 1);

        assert!(matches!(
            runtime.load_from_directory(alias, adapter_dir.path()),
            Err(LoraAdapterError::AliasTooLong {
                bytes,
                max: MAX_LORA_ALIAS_BYTES,
            }) if bytes == MAX_LORA_ALIAS_BYTES + 1
        ));
    }

    #[test]
    fn async_load_admission_is_non_queueing() {
        let permit = DynamicLoraRuntime::try_acquire_load_permit().unwrap();
        assert!(matches!(
            DynamicLoraRuntime::try_acquire_load_permit(),
            Err(LoraAdapterError::LoadBusy)
        ));
        drop(permit);
        assert!(DynamicLoraRuntime::try_acquire_load_permit().is_ok());
    }

    #[test]
    fn rank_limit_is_checked_before_weight_snapshotting() {
        let adapter_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            adapter_dir.path().join("adapter_config.json"),
            r#"{"r":2,"lora_alpha":2,"target_modules":["proj"]}"#,
        )
        .unwrap();
        std::fs::write(adapter_dir.path().join("adapter_model.safetensors"), []).unwrap();
        let limits = LoraRuntimeConfig {
            max_rank: 1,
            ..LoraRuntimeConfig::default()
        };
        let (runtime, _) = runtime_and_layer(limits);

        assert!(matches!(
            runtime.load_from_directory("production", adapter_dir.path()),
            Err(LoraAdapterError::RankLimit { rank: 2, max: 1 })
        ));
    }
}
