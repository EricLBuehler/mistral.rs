use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use candle_core::{DType, Device, Result};

use crate::Shard;

use super::{LoraExpertSiteHandle, LoraExpertSiteSpec};

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoraRuntimeId(u64);

impl LoraRuntimeId {
    fn next() -> Self {
        Self(NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoraSiteSlice {
    index: usize,
    count: usize,
}

impl LoraSiteSlice {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn count(self) -> usize {
        self.count
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoraSiteKey {
    path: Arc<str>,
    slice: Option<LoraSiteSlice>,
}

impl LoraSiteKey {
    pub fn new(path: impl Into<Arc<str>>) -> Self {
        Self {
            path: path.into(),
            slice: None,
        }
    }

    pub fn with_slice(path: impl Into<Arc<str>>, index: usize, count: usize) -> Result<Self> {
        if count == 0 || index >= count {
            candle_core::bail!("invalid LoRA site slice {index}/{count}");
        }
        Ok(Self {
            path: path.into(),
            slice: Some(LoraSiteSlice { index, count }),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn slice(&self) -> Option<LoraSiteSlice> {
        self.slice
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoraParallelism {
    Replicated,
    Column { output_shard: Shard },
    Row { input_shard: Shard },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraLinearSpec {
    in_features: usize,
    out_features: usize,
    parallelism: LoraParallelism,
}

impl LoraLinearSpec {
    pub fn replicated(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            parallelism: LoraParallelism::Replicated,
        }
    }

    pub fn column(in_features: usize, out_features: usize, output_shard: Shard) -> Self {
        Self {
            in_features,
            out_features,
            parallelism: LoraParallelism::Column { output_shard },
        }
    }

    pub fn row(in_features: usize, out_features: usize, input_shard: Shard) -> Self {
        Self {
            in_features,
            out_features,
            parallelism: LoraParallelism::Row { input_shard },
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub(crate) fn parallelism(&self) -> LoraParallelism {
        self.parallelism
    }
}

#[derive(Debug)]
pub struct LoraSiteHandle {
    runtime_id: LoraRuntimeId,
    key: LoraSiteKey,
    spec: LoraLinearSpec,
    activation_dtype: DType,
    device: Device,
    id: OnceLock<u32>,
}

impl LoraSiteHandle {
    pub(crate) fn runtime_id(&self) -> LoraRuntimeId {
        self.runtime_id
    }

    pub fn key(&self) -> &LoraSiteKey {
        &self.key
    }

    pub fn spec(&self) -> &LoraLinearSpec {
        &self.spec
    }

    pub fn activation_dtype(&self) -> DType {
        self.activation_dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn id(&self) -> Result<u32> {
        self.id
            .get()
            .copied()
            .ok_or_else(|| candle_core::Error::msg("LoRA layer registry has not been finalized"))
    }
}

#[derive(Default)]
struct RegistryState {
    sites: BTreeMap<LoraSiteKey, Arc<LoraSiteHandle>>,
    expert_sites: BTreeMap<LoraSiteKey, Arc<LoraExpertSiteHandle>>,
    finalized: bool,
}

#[derive(Debug)]
pub struct LoraLayerRegistry {
    runtime_id: LoraRuntimeId,
    state: Mutex<RegistryState>,
}

impl Default for LoraLayerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RegistryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryState")
            .field("sites", &self.sites)
            .field("expert_sites", &self.expert_sites)
            .field("finalized", &self.finalized)
            .finish()
    }
}

impl LoraLayerRegistry {
    pub fn new() -> Self {
        Self {
            runtime_id: LoraRuntimeId::next(),
            state: Mutex::new(RegistryState::default()),
        }
    }

    pub fn runtime_id(&self) -> LoraRuntimeId {
        self.runtime_id
    }

    pub fn register(
        &self,
        key: LoraSiteKey,
        spec: LoraLinearSpec,
        activation_dtype: DType,
        device: Device,
    ) -> Result<Arc<LoraSiteHandle>> {
        let mut state = self.state.lock().expect("LoRA layer registry poisoned");
        if state.expert_sites.contains_key(&key) {
            candle_core::bail!(
                "LoRA site `{}` was registered as both a linear and an expert group",
                key.path()
            );
        }
        if let Some(site) = state.sites.get(&key) {
            if site.spec != spec
                || site.activation_dtype != activation_dtype
                || site.device.location() != device.location()
            {
                candle_core::bail!(
                    "LoRA site `{}` was registered with incompatible specifications",
                    key.path()
                );
            }
            return Ok(site.clone());
        }
        if state.finalized {
            candle_core::bail!(
                "cannot register LoRA site `{}` after registry finalization",
                key.path()
            );
        }

        let site = Arc::new(LoraSiteHandle {
            runtime_id: self.runtime_id,
            key: key.clone(),
            spec,
            activation_dtype,
            device,
            id: OnceLock::new(),
        });
        state.sites.insert(key, site.clone());
        Ok(site)
    }

    pub fn register_expert(
        &self,
        key: LoraSiteKey,
        spec: LoraExpertSiteSpec,
        activation_dtype: DType,
        device: Device,
    ) -> Result<Arc<LoraExpertSiteHandle>> {
        if key.slice().is_some() {
            candle_core::bail!("expert LoRA group sites cannot be sliced");
        }
        let mut state = self.state.lock().expect("LoRA layer registry poisoned");
        if state.sites.contains_key(&key) {
            candle_core::bail!(
                "LoRA site `{}` was registered as both a linear and an expert group",
                key.path()
            );
        }
        if let Some(site) = state.expert_sites.get(&key) {
            if site.spec() != &spec
                || site.activation_dtype() != activation_dtype
                || site.device().location() != device.location()
            {
                candle_core::bail!(
                    "LoRA expert site `{}` was registered with incompatible specifications",
                    key.path()
                );
            }
            return Ok(site.clone());
        }
        if state.finalized {
            candle_core::bail!(
                "cannot register LoRA expert site `{}` after registry finalization",
                key.path()
            );
        }

        let site = Arc::new(LoraExpertSiteHandle::new(
            self.runtime_id,
            key.clone(),
            spec,
            activation_dtype,
            device,
        ));
        state.expert_sites.insert(key, site.clone());
        Ok(site)
    }

    pub fn finalize(&self) -> Result<Vec<Arc<LoraSiteHandle>>> {
        let mut state = self.state.lock().expect("LoRA layer registry poisoned");
        if !state.finalized {
            for (id, site) in state.sites.values().enumerate() {
                let id = u32::try_from(id).map_err(candle_core::Error::wrap)?;
                site.id
                    .set(id)
                    .map_err(|_| candle_core::Error::msg("LoRA site ID was already assigned"))?;
            }
            let first_expert_id = state.sites.len();
            for (offset, site) in state.expert_sites.values().enumerate() {
                let id =
                    u32::try_from(first_expert_id + offset).map_err(candle_core::Error::wrap)?;
                site.assign_id(id)?;
            }
            state.finalized = true;
        }
        Ok(state.sites.values().cloned().collect())
    }

    pub fn sites(&self) -> Vec<Arc<LoraSiteHandle>> {
        self.state
            .lock()
            .expect("LoRA layer registry poisoned")
            .sites
            .values()
            .cloned()
            .collect()
    }

    pub fn expert_sites(&self) -> Vec<Arc<LoraExpertSiteHandle>> {
        self.state
            .lock()
            .expect("LoRA layer registry poisoned")
            .expert_sites
            .values()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert_spec() -> Result<LoraExpertSiteSpec> {
        LoraExpertSiteSpec::new(
            2,
            4,
            8,
            super::super::LoraExpertProjectionNames::new("gate", "up", "down"),
            Shard::default(),
            Shard::default(),
        )
    }

    #[test]
    fn site_ids_are_independent_of_registration_order() -> Result<()> {
        fn ids(paths: &[&str]) -> Result<BTreeMap<String, u32>> {
            let registry = LoraLayerRegistry::new();
            for path in paths {
                registry.register(
                    LoraSiteKey::new(*path),
                    LoraLinearSpec::replicated(4, 8),
                    DType::F32,
                    Device::Cpu,
                )?;
            }
            registry.finalize()?;
            registry
                .sites()
                .into_iter()
                .map(|site| Ok((site.key().path().to_string(), site.id()?)))
                .collect()
        }

        assert_eq!(ids(&["c", "a", "b"])?, ids(&["a", "b", "c"])?);
        Ok(())
    }

    #[test]
    fn linear_and_expert_sites_share_one_id_namespace() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let expert = registry.register_expert(
            LoraSiteKey::new("experts"),
            expert_spec()?,
            DType::F32,
            Device::Cpu,
        )?;
        let linear = registry.register(
            LoraSiteKey::new("linear"),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;

        assert_eq!(linear.id()?, 0);
        assert_eq!(expert.id()?, 1);
        assert_ne!(linear.id()?, expert.id()?);
        Ok(())
    }

    #[test]
    fn duplicate_sites_must_have_the_same_specification() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let key = LoraSiteKey::new("model.layers.0.self_attn.q_proj");
        let first = registry.register(
            key.clone(),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        let second = registry.register(
            key.clone(),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        assert!(Arc::ptr_eq(&first, &second));
        assert!(registry
            .register(
                key.clone(),
                LoraLinearSpec::replicated(4, 8),
                DType::BF16,
                Device::Cpu,
            )
            .is_err());
        assert!(registry
            .register(
                key,
                LoraLinearSpec::replicated(8, 8),
                DType::F32,
                Device::Cpu,
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn new_sites_cannot_be_registered_after_finalization() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        registry.register(
            LoraSiteKey::new("a"),
            LoraLinearSpec::replicated(4, 8),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;
        assert!(registry
            .register(
                LoraSiteKey::new("b"),
                LoraLinearSpec::replicated(4, 8),
                DType::F32,
                Device::Cpu,
            )
            .is_err());
        Ok(())
    }
}
