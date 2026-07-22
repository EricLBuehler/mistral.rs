use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::{Arc, Mutex, RwLock, Weak},
};

use mistralrs_quant::{LoraAdapterWeights, LoraRuntimeId, LoraSlotId};
use thiserror::Error;

use super::{AdapterGenerationId, LoraAdapterInfo, LoraResidentGenerationInfo};

#[derive(Debug)]
pub(crate) struct ResidentBudget {
    limits: super::LoraRuntimeConfig,
    state: Mutex<ResidentBudgetState>,
}

#[derive(Debug, Default)]
struct ResidentBudgetState {
    adapters: usize,
    bytes: u64,
}

impl ResidentBudget {
    pub(crate) fn new(limits: super::LoraRuntimeConfig) -> Self {
        Self {
            limits,
            state: Mutex::new(ResidentBudgetState::default()),
        }
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<ResidentReservation, AdapterBudgetError> {
        let mut state = self.state.lock().expect("LoRA resident budget poisoned");
        let next_adapters =
            state
                .adapters
                .checked_add(1)
                .ok_or(AdapterBudgetError::AdapterLimit {
                    max: self.limits.max_adapters,
                })?;
        if next_adapters > self.limits.max_adapters {
            return Err(AdapterBudgetError::AdapterLimit {
                max: self.limits.max_adapters,
            });
        }
        let next_bytes = state
            .bytes
            .checked_add(bytes)
            .ok_or(AdapterBudgetError::ByteLimit {
                requested: bytes,
                resident: state.bytes,
                max: self.limits.max_bytes,
            })?;
        if next_bytes > self.limits.max_bytes {
            return Err(AdapterBudgetError::ByteLimit {
                requested: bytes,
                resident: state.bytes,
                max: self.limits.max_bytes,
            });
        }
        state.adapters += 1;
        state.bytes = next_bytes;
        Ok(ResidentReservation {
            budget: self.clone(),
            bytes,
            active: true,
        })
    }

    fn release(&self, bytes: u64) {
        let mut state = self.state.lock().expect("LoRA resident budget poisoned");
        state.adapters = state
            .adapters
            .checked_sub(1)
            .expect("LoRA resident adapter count underflow");
        state.bytes = state
            .bytes
            .checked_sub(bytes)
            .expect("LoRA resident byte count underflow");
    }
}

#[derive(Debug)]
pub(crate) struct ResidentReservation {
    budget: Arc<ResidentBudget>,
    bytes: u64,
    active: bool,
}

impl ResidentReservation {
    fn commit(mut self) -> Arc<ResidentBudget> {
        self.active = false;
        self.budget.clone()
    }
}

impl Drop for ResidentReservation {
    fn drop(&mut self) {
        if self.active {
            self.budget.release(self.bytes);
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum AdapterBudgetError {
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
}

#[derive(Debug)]
pub(crate) struct ResidentAdapterGeneration {
    runtime_id: LoraRuntimeId,
    generation: AdapterGenerationId,
    slot: LoraSlotId,
    rank: usize,
    bytes: u64,
    weights: Arc<LoraAdapterWeights>,
    budget: Arc<ResidentBudget>,
}

impl ResidentAdapterGeneration {
    pub(crate) fn new(
        runtime_id: LoraRuntimeId,
        generation: AdapterGenerationId,
        slot: LoraSlotId,
        rank: usize,
        bytes: u64,
        weights: Arc<LoraAdapterWeights>,
        reservation: ResidentReservation,
    ) -> Self {
        assert_eq!(reservation.bytes, bytes, "LoRA reservation size mismatch");
        let budget = reservation.commit();
        Self {
            runtime_id,
            generation,
            slot,
            rank,
            bytes,
            weights,
            budget,
        }
    }

    pub(crate) fn generation(&self) -> AdapterGenerationId {
        self.generation
    }

    pub(crate) fn runtime_id(&self) -> LoraRuntimeId {
        self.runtime_id
    }

    pub(crate) fn slot(&self) -> LoraSlotId {
        self.slot
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn weights(&self) -> &Arc<LoraAdapterWeights> {
        &self.weights
    }
}

impl Drop for ResidentAdapterGeneration {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AdapterLease(Arc<ResidentAdapterGeneration>);

impl AdapterLease {
    fn new(resident: Arc<ResidentAdapterGeneration>) -> Self {
        Self(resident)
    }

    pub(crate) fn generation(&self) -> AdapterGenerationId {
        self.0.generation()
    }

    pub(crate) fn slot(&self) -> LoraSlotId {
        self.0.slot()
    }

    pub(crate) fn resident(&self) -> &ResidentAdapterGeneration {
        &self.0
    }

    fn resident_arc(&self) -> Arc<ResidentAdapterGeneration> {
        self.0.clone()
    }
}

#[derive(Clone)]
struct AdapterAlias {
    source: String,
    revision: Option<String>,
    resident: Arc<ResidentAdapterGeneration>,
}

#[derive(Default)]
struct AdapterRegistryState {
    aliases: HashMap<String, AdapterAlias>,
    generations: HashMap<AdapterGenerationId, Arc<ResidentAdapterGeneration>>,
    slots: HashMap<LoraSlotId, Weak<ResidentAdapterGeneration>>,
    next_slot: LoraSlotId,
}

pub(crate) struct AdapterRegistry {
    state: RwLock<AdapterRegistryState>,
    max_aliases: usize,
    max_alias_bytes: usize,
}

pub(crate) struct AdapterRegistrySnapshot {
    pub(crate) adapters: Vec<LoraAdapterInfo>,
    pub(crate) generations: Vec<LoraResidentGenerationInfo>,
    pub(crate) resident_generations: usize,
    pub(crate) retired_generations: usize,
    pub(crate) resident_bytes: u64,
}

impl fmt::Debug for AdapterRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterRegistry")
            .field("adapters", &self.list())
            .finish()
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum AdapterRegistryError {
    #[error("adapter alias must not be empty")]
    EmptyAlias,
    #[error("adapter alias is {bytes} bytes, exceeding the {max}-byte limit")]
    AliasTooLong { bytes: usize, max: usize },
    #[error("loaded LoRA adapter alias limit {max} reached")]
    AliasLimit { max: usize },
    #[error("adapter alias `{0}` is not registered")]
    AliasNotFound(String),
    #[error("adapter generation `{0}` is not resident")]
    GenerationNotFound(AdapterGenerationId),
    #[error("adapter generation `{0}` is already resident with different data")]
    GenerationConflict(AdapterGenerationId),
    #[error("LoRA slot space is exhausted")]
    SlotExhausted,
}

impl AdapterRegistry {
    pub(crate) fn new(max_aliases: usize, max_alias_bytes: usize) -> Self {
        Self {
            state: RwLock::new(AdapterRegistryState::default()),
            max_aliases,
            max_alias_bytes,
        }
    }

    pub(crate) fn allocate_slot(&self) -> Result<LoraSlotId, AdapterRegistryError> {
        let mut state = self.state.write().expect("adapter registry poisoned");
        allocate_slot(&mut state)
    }

    pub(crate) fn register(
        &self,
        alias: String,
        source: String,
        revision: Option<String>,
        resident: Arc<ResidentAdapterGeneration>,
    ) -> Result<AdapterLease, AdapterRegistryError> {
        let mut state = self.state.write().expect("adapter registry poisoned");
        let (lease, retired) = register(
            &mut state,
            self.max_aliases,
            self.max_alias_bytes,
            alias,
            source,
            revision,
            resident,
        )?;
        drop(state);
        drop(retired);
        Ok(lease)
    }

    pub(crate) fn validate_registration(&self, alias: &str) -> Result<(), AdapterRegistryError> {
        let state = self.state.read().expect("adapter registry poisoned");
        validate_registration(&state, self.max_aliases, self.max_alias_bytes, alias)
    }

    pub(crate) fn resolve_alias(&self, alias: &str) -> Result<AdapterLease, AdapterRegistryError> {
        self.state
            .read()
            .expect("adapter registry poisoned")
            .aliases
            .get(alias)
            .map(|entry| AdapterLease::new(entry.resident.clone()))
            .ok_or_else(|| AdapterRegistryError::AliasNotFound(alias.to_string()))
    }

    pub(crate) fn register_lease(
        &self,
        alias: String,
        source: String,
        revision: Option<String>,
        lease: &AdapterLease,
    ) -> Result<AdapterLease, AdapterRegistryError> {
        self.register(alias, source, revision, lease.resident_arc())
    }

    pub(crate) fn resolve_generation(
        &self,
        generation: AdapterGenerationId,
    ) -> Result<AdapterLease, AdapterRegistryError> {
        let state = self.state.read().expect("adapter registry poisoned");
        state
            .generations
            .get(&generation)
            .cloned()
            .or_else(|| {
                state
                    .slots
                    .values()
                    .filter_map(Weak::upgrade)
                    .find(|resident| resident.generation() == generation)
            })
            .map(AdapterLease::new)
            .ok_or(AdapterRegistryError::GenerationNotFound(generation))
    }

    pub(crate) fn unload(&self, alias: &str) -> Result<LoraAdapterInfo, AdapterRegistryError> {
        let mut state = self.state.write().expect("adapter registry poisoned");
        let entry = state
            .aliases
            .remove(alias)
            .ok_or_else(|| AdapterRegistryError::AliasNotFound(alias.to_string()))?;
        let generation = entry.resident.generation();
        let still_aliased = state
            .aliases
            .values()
            .any(|candidate| candidate.resident.generation() == generation);
        let retired = if still_aliased {
            None
        } else {
            state.generations.remove(&generation)
        };
        let info = adapter_info(alias, &entry);
        drop(state);
        drop(retired);
        drop(entry);
        Ok(info)
    }

    pub(crate) fn list(&self) -> Vec<LoraAdapterInfo> {
        self.snapshot().adapters
    }

    pub(crate) fn snapshot(&self) -> AdapterRegistrySnapshot {
        let state = self.state.read().expect("adapter registry poisoned");
        let adapters = state
            .aliases
            .iter()
            .map(|(alias, entry)| (alias.clone(), adapter_info(alias, entry)))
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect();
        let residents = state
            .slots
            .values()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        let mut generations = residents
            .iter()
            .map(|resident| {
                let generation = resident.generation();
                let mut aliases = state
                    .aliases
                    .iter()
                    .filter(|(_, entry)| entry.resident.generation() == generation)
                    .map(|(alias, _)| alias.clone())
                    .collect::<Vec<_>>();
                aliases.sort();
                let registry_refs =
                    aliases.len() + usize::from(state.generations.contains_key(&generation)) + 1;
                LoraResidentGenerationInfo {
                    generation,
                    aliases,
                    rank: resident.rank(),
                    bytes: resident.bytes(),
                    retired: !state.generations.contains_key(&generation),
                    active_leases: Arc::strong_count(resident).saturating_sub(registry_refs),
                }
            })
            .collect::<Vec<_>>();
        generations.sort_by_key(|generation| generation.generation);
        let retired_generations = generations
            .iter()
            .filter(|generation| generation.retired)
            .count();
        let resident_bytes = residents.iter().map(|resident| resident.bytes()).sum();
        let snapshot = AdapterRegistrySnapshot {
            adapters,
            generations,
            resident_generations: residents.len(),
            retired_generations,
            resident_bytes,
        };
        drop(state);
        drop(residents);
        snapshot
    }

    pub(crate) fn info(&self, alias: &str) -> Result<LoraAdapterInfo, AdapterRegistryError> {
        self.state
            .read()
            .expect("adapter registry poisoned")
            .aliases
            .get(alias)
            .map(|entry| adapter_info(alias, entry))
            .ok_or_else(|| AdapterRegistryError::AliasNotFound(alias.to_string()))
    }
}

fn allocate_slot(state: &mut AdapterRegistryState) -> Result<LoraSlotId, AdapterRegistryError> {
    let reusable = state
        .slots
        .iter()
        .find_map(|(slot, resident)| (resident.strong_count() == 0).then_some(*slot));
    if let Some(slot) = reusable {
        state.slots.remove(&slot);
        return Ok(slot);
    }
    let slot = state.next_slot;
    state.next_slot = state
        .next_slot
        .checked_add(1)
        .ok_or(AdapterRegistryError::SlotExhausted)?;
    Ok(slot)
}

fn register(
    state: &mut AdapterRegistryState,
    max_aliases: usize,
    max_alias_bytes: usize,
    alias: String,
    source: String,
    revision: Option<String>,
    resident: Arc<ResidentAdapterGeneration>,
) -> Result<(AdapterLease, Option<Arc<ResidentAdapterGeneration>>), AdapterRegistryError> {
    validate_registration(state, max_aliases, max_alias_bytes, &alias)?;
    if let Some(existing) = state.generations.get(&resident.generation()) {
        if !Arc::ptr_eq(existing, &resident) {
            return Err(AdapterRegistryError::GenerationConflict(
                resident.generation(),
            ));
        }
    }

    state
        .generations
        .insert(resident.generation(), resident.clone());
    state
        .slots
        .insert(resident.slot(), Arc::downgrade(&resident));
    let replaced = state.aliases.insert(
        alias,
        AdapterAlias {
            source,
            revision,
            resident: resident.clone(),
        },
    );
    let mut retired = None;
    if let Some(previous) = replaced {
        let generation = previous.resident.generation();
        let still_aliased = state
            .aliases
            .values()
            .any(|candidate| candidate.resident.generation() == generation);
        if !still_aliased {
            retired = state.generations.remove(&generation);
        }
    }
    Ok((AdapterLease::new(resident), retired))
}

fn validate_registration(
    state: &AdapterRegistryState,
    max_aliases: usize,
    max_alias_bytes: usize,
    alias: &str,
) -> Result<(), AdapterRegistryError> {
    if alias.trim().is_empty() {
        return Err(AdapterRegistryError::EmptyAlias);
    }
    if alias.len() > max_alias_bytes {
        return Err(AdapterRegistryError::AliasTooLong {
            bytes: alias.len(),
            max: max_alias_bytes,
        });
    }
    if !state.aliases.contains_key(alias) && state.aliases.len() >= max_aliases {
        return Err(AdapterRegistryError::AliasLimit { max: max_aliases });
    }
    Ok(())
}

fn adapter_info(alias: &str, entry: &AdapterAlias) -> LoraAdapterInfo {
    LoraAdapterInfo {
        alias: alias.to_string(),
        source: entry.source.clone(),
        revision: entry.revision.clone(),
        generation: entry.resident.generation(),
        rank: entry.resident.rank(),
        bytes: entry.resident.bytes(),
    }
}
