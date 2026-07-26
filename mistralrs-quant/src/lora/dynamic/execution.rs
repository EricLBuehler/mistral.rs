use std::{
    cell::RefCell,
    collections::{hash_map::Entry, BTreeMap, HashMap},
    ops::Range,
    sync::{Arc, Mutex},
};

#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{Device, DeviceLocation, Result, Tensor};

use super::{
    DynamicLoraWeights, LoraExpertSiteHandle, LoraExpertWeights, LoraRuntimeId, LoraSiteHandle,
};

pub type LoraSlotId = u32;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum LoraRowRemap {
    Range { start: usize, end: usize },
    Repeat { row: usize, rows: usize },
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(super) struct PreparedExpertAdapters {
    pub slots: Vec<(LoraSlotId, Arc<LoraExpertWeights>)>,
    pub token_slots: Vec<u32>,
}

#[cfg(feature = "cuda")]
static NEXT_LORA_EXECUTION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
pub struct LoraExecutionArena {
    #[cfg(feature = "cuda")]
    expert_cuda: Mutex<super::expert_cuda::ExpertCudaCache>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LoraExecutionArenaStats {
    pub cached_routing_resources: usize,
    pub cached_weight_tables: usize,
    pub weight_table_uploads: usize,
    pub token_slot_uploads: usize,
    pub metadata_builds: usize,
}

impl std::fmt::Debug for LoraExecutionArena {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LoraExecutionArena")
    }
}

impl LoraExecutionArena {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "cuda")]
    pub fn cached_expert_cuda_resources(&self) -> usize {
        self.expert_cuda
            .lock()
            .expect("expert LoRA CUDA arena poisoned")
            .len()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_stats(&self) -> LoraExecutionArenaStats {
        let cache = self
            .expert_cuda
            .lock()
            .expect("expert LoRA CUDA arena poisoned");
        let (routing, weights, uploads, slots, metadata) = cache.stats();
        LoraExecutionArenaStats {
            cached_routing_resources: routing,
            cached_weight_tables: weights,
            weight_table_uploads: uploads,
            token_slot_uploads: slots,
            metadata_builds: metadata,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoraWeights {
    pub(crate) a: Tensor,
    pub(crate) b: Tensor,
    pub(crate) scale: f64,
}

impl LoraWeights {
    pub fn new(a: Tensor, b: Tensor, scale: f64) -> Result<Self> {
        let (rank, _) = a.dims2()?;
        let (_, b_rank) = b.dims2()?;
        if rank == 0 {
            candle_core::bail!("LoRA rank must be nonzero");
        }
        if rank != b_rank {
            candle_core::bail!("LoRA A rank {rank} does not match B rank {b_rank}");
        }
        if a.dtype() != b.dtype() {
            candle_core::bail!(
                "LoRA A dtype {:?} does not match B dtype {:?}",
                a.dtype(),
                b.dtype()
            );
        }
        if a.device().location() != b.device().location() {
            candle_core::bail!("LoRA A and B must be on the same device");
        }
        if !scale.is_finite() {
            candle_core::bail!("LoRA scale must be finite");
        }
        Ok(Self { a, b, scale })
    }

    pub fn a(&self) -> &Tensor {
        &self.a
    }

    pub fn b(&self) -> &Tensor {
        &self.b
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Debug)]
pub struct LoraAdapterWeights {
    runtime_id: LoraRuntimeId,
    weights: HashMap<u32, Arc<LoraWeights>>,
    expert_weights: HashMap<u32, Arc<LoraExpertWeights>>,
}

impl LoraAdapterWeights {
    pub fn new(runtime_id: LoraRuntimeId, weights: impl Into<DynamicLoraWeights>) -> Result<Self> {
        let weights = weights.into();
        let mut by_site = HashMap::with_capacity(weights.linear.len());
        for (site, weights) in weights.linear {
            if site.runtime_id() != runtime_id {
                candle_core::bail!("LoRA weights belong to a different runtime");
            }
            let site_id = site.id()?;
            if by_site.insert(site_id, Arc::new(weights)).is_some() {
                candle_core::bail!("LoRA weights contain a duplicate site");
            }
        }
        let mut experts_by_site = HashMap::with_capacity(weights.experts.len());
        for (site, weights) in weights.experts {
            if site.runtime_id() != runtime_id {
                candle_core::bail!("expert LoRA weights belong to a different runtime");
            }
            weights.validate_for_site(&site)?;
            let site_id = site.id()?;
            if experts_by_site.insert(site_id, Arc::new(weights)).is_some() {
                candle_core::bail!("expert LoRA weights contain a duplicate site");
            }
        }
        Ok(Self {
            runtime_id,
            weights: by_site,
            expert_weights: experts_by_site,
        })
    }

    fn weights(&self, site_id: u32) -> Option<&Arc<LoraWeights>> {
        self.weights.get(&site_id)
    }

    fn expert_weights(&self, site_id: u32) -> Option<&Arc<LoraExpertWeights>> {
        self.expert_weights.get(&site_id)
    }
}

#[derive(Debug)]
pub struct LoraExecution {
    #[cfg(feature = "cuda")]
    execution_id: u64,
    runtime_id: LoraRuntimeId,
    row_slots: Arc<[Option<LoraSlotId>]>,
    rows_by_slot: BTreeMap<LoraSlotId, Arc<[usize]>>,
    adapters: HashMap<LoraSlotId, Arc<LoraAdapterWeights>>,
    weights: HashMap<(u32, LoraSlotId), Arc<LoraWeights>>,
    expert_weights: HashMap<(u32, LoraSlotId), Arc<LoraExpertWeights>>,
    row_indices: Mutex<HashMap<(DeviceLocation, LoraSlotId), Tensor>>,
    row_remaps: Mutex<HashMap<LoraRowRemap, Arc<LoraExecution>>>,
    #[cfg(feature = "cuda")]
    prepared_expert_adapters: Mutex<HashMap<u32, Arc<PreparedExpertAdapters>>>,
    arena: Arc<LoraExecutionArena>,
}

impl LoraExecution {
    fn site_id(&self, site: &LoraSiteHandle) -> Result<u32> {
        if site.runtime_id() != self.runtime_id {
            candle_core::bail!("LoRA site and execution belong to different runtimes");
        }
        site.id()
    }

    fn expert_site_id(&self, site: &LoraExpertSiteHandle) -> Result<u32> {
        if site.runtime_id() != self.runtime_id {
            candle_core::bail!("expert LoRA site and execution belong to different runtimes");
        }
        site.id()
    }

    pub fn new(runtime_id: LoraRuntimeId, row_slots: Vec<Option<LoraSlotId>>) -> Self {
        Self::new_with_arena(runtime_id, row_slots, Arc::new(LoraExecutionArena::new()))
    }

    pub fn new_with_arena(
        runtime_id: LoraRuntimeId,
        row_slots: Vec<Option<LoraSlotId>>,
        arena: Arc<LoraExecutionArena>,
    ) -> Self {
        let mut rows_by_slot = BTreeMap::<LoraSlotId, Vec<usize>>::new();
        for (row, slot) in row_slots.iter().enumerate() {
            if let Some(slot) = slot {
                rows_by_slot.entry(*slot).or_default().push(row);
            }
        }
        Self {
            #[cfg(feature = "cuda")]
            execution_id: NEXT_LORA_EXECUTION_ID.fetch_add(1, Ordering::Relaxed),
            runtime_id,
            row_slots: row_slots.into(),
            rows_by_slot: rows_by_slot
                .into_iter()
                .map(|(slot, rows)| (slot, rows.into()))
                .collect(),
            adapters: HashMap::new(),
            weights: HashMap::new(),
            expert_weights: HashMap::new(),
            row_indices: Mutex::new(HashMap::new()),
            row_remaps: Mutex::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            prepared_expert_adapters: Mutex::new(HashMap::new()),
            arena,
        }
    }

    pub fn from_sequence_slots(
        runtime_id: LoraRuntimeId,
        sequence_slots: &[Option<LoraSlotId>],
        sequence_length: usize,
    ) -> Self {
        let row_slots = sequence_slots
            .iter()
            .flat_map(|slot| std::iter::repeat_n(*slot, sequence_length))
            .collect();
        Self::new(runtime_id, row_slots)
    }

    pub fn from_sequence_slots_with_arena(
        runtime_id: LoraRuntimeId,
        sequence_slots: &[Option<LoraSlotId>],
        sequence_length: usize,
        arena: Arc<LoraExecutionArena>,
    ) -> Self {
        let row_slots = sequence_slots
            .iter()
            .flat_map(|slot| std::iter::repeat_n(*slot, sequence_length))
            .collect();
        Self::new_with_arena(runtime_id, row_slots, arena)
    }

    pub fn from_ragged_sequence_slots_with_arena(
        runtime_id: LoraRuntimeId,
        sequence_slots: &[Option<LoraSlotId>],
        sequence_lengths: &[usize],
        arena: Arc<LoraExecutionArena>,
    ) -> Self {
        assert_eq!(sequence_slots.len(), sequence_lengths.len());
        let row_slots = sequence_slots
            .iter()
            .zip(sequence_lengths)
            .flat_map(|(slot, &length)| std::iter::repeat_n(*slot, length))
            .collect();
        Self::new_with_arena(runtime_id, row_slots, arena)
    }

    pub fn runtime_id(&self) -> LoraRuntimeId {
        self.runtime_id
    }

    #[cfg(feature = "cuda")]
    pub(super) fn execution_id(&self) -> u64 {
        self.execution_id
    }

    pub fn arena(&self) -> &Arc<LoraExecutionArena> {
        &self.arena
    }

    pub fn row_slots(&self) -> &[Option<LoraSlotId>] {
        &self.row_slots
    }

    fn remap_rows(&self, remap: LoraRowRemap) -> Result<Arc<Self>> {
        if let Some(execution) = self
            .row_remaps
            .lock()
            .expect("LoRA row remap cache poisoned")
            .get(&remap)
        {
            return Ok(execution.clone());
        }
        let row_slots = match remap {
            LoraRowRemap::Range { start, end } => {
                if start > end {
                    candle_core::bail!("LoRA route row range is reversed");
                }
                self.row_slots
                    .get(start..end)
                    .ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "LoRA route row range {start}..{end} exceeds {} rows",
                            self.row_slots.len()
                        ))
                    })?
                    .to_vec()
            }
            LoraRowRemap::Repeat { row, rows } => {
                let slot = self.row_slots.get(row).copied().ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "LoRA route row {row} is outside {} rows",
                        self.row_slots.len()
                    ))
                })?;
                vec![slot; rows]
            }
        };
        let mut execution = Self::new_with_arena(self.runtime_id, row_slots, self.arena.clone());
        execution.adapters = self.adapters.clone();
        execution.weights = self.weights.clone();
        execution.expert_weights = self.expert_weights.clone();
        let execution = Arc::new(execution);
        let mut cache = self
            .row_remaps
            .lock()
            .expect("LoRA row remap cache poisoned");
        Ok(cache.entry(remap).or_insert(execution).clone())
    }

    pub(crate) fn rows_by_slot(&self) -> &BTreeMap<LoraSlotId, Arc<[usize]>> {
        &self.rows_by_slot
    }

    #[cfg(feature = "cuda")]
    pub(super) fn expert_cuda_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, super::expert_cuda::ExpertCudaCache> {
        self.arena
            .expert_cuda
            .lock()
            .expect("expert LoRA CUDA arena poisoned")
    }

    pub(crate) fn row_indices(&self, slot: LoraSlotId, device: &Device) -> Result<Option<Tensor>> {
        let Some(rows) = self.rows_by_slot.get(&slot) else {
            return Ok(None);
        };
        let key = (device.location(), slot);
        let mut cache = self
            .row_indices
            .lock()
            .expect("LoRA row index cache poisoned");
        if let Some(indices) = cache.get(&key) {
            return Ok(Some(indices.clone()));
        }
        let rows = rows
            .iter()
            .map(|row| u32::try_from(*row).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        let row_count = rows.len();
        let indices = Tensor::from_vec(rows, row_count, device)?;
        cache.insert(key, indices.clone());
        Ok(Some(indices))
    }

    pub fn insert(
        &mut self,
        site: &LoraSiteHandle,
        slot: LoraSlotId,
        weights: LoraWeights,
    ) -> Result<()> {
        self.insert_shared(site, slot, Arc::new(weights))
    }

    pub fn insert_shared(
        &mut self,
        site: &LoraSiteHandle,
        slot: LoraSlotId,
        weights: Arc<LoraWeights>,
    ) -> Result<()> {
        let site_id = self.site_id(site)?;
        if self.adapters.contains_key(&slot) {
            candle_core::bail!("LoRA adapter weights were already installed for slot");
        }
        match self.weights.entry((site_id, slot)) {
            Entry::Occupied(_) => {
                candle_core::bail!("LoRA weights were already installed for site and slot")
            }
            Entry::Vacant(entry) => {
                entry.insert(weights);
            }
        }
        self.row_remaps
            .get_mut()
            .expect("LoRA row remap cache poisoned")
            .clear();
        Ok(())
    }

    pub fn insert_adapter(
        &mut self,
        slot: LoraSlotId,
        weights: Arc<LoraAdapterWeights>,
    ) -> Result<()> {
        if weights.runtime_id != self.runtime_id {
            candle_core::bail!("LoRA adapter weights belong to a different runtime");
        }
        if self
            .weights
            .keys()
            .any(|(_, installed_slot)| *installed_slot == slot)
            || self
                .expert_weights
                .keys()
                .any(|(_, installed_slot)| *installed_slot == slot)
        {
            candle_core::bail!("LoRA site weights were already installed for slot");
        }
        match self.adapters.entry(slot) {
            Entry::Occupied(_) => candle_core::bail!("LoRA adapter slot was already installed"),
            Entry::Vacant(entry) => {
                entry.insert(weights);
                self.row_remaps
                    .get_mut()
                    .expect("LoRA row remap cache poisoned")
                    .clear();
                #[cfg(feature = "cuda")]
                self.prepared_expert_adapters
                    .get_mut()
                    .expect("expert LoRA prepared cache poisoned")
                    .clear();
                Ok(())
            }
        }
    }

    pub fn weights(
        &self,
        site: &LoraSiteHandle,
        slot: LoraSlotId,
    ) -> Result<Option<&Arc<LoraWeights>>> {
        let site_id = self.site_id(site)?;
        Ok(self
            .weights
            .get(&(site_id, slot))
            .or_else(|| self.adapters.get(&slot)?.weights(site_id)))
    }

    pub fn insert_expert(
        &mut self,
        site: &LoraExpertSiteHandle,
        slot: LoraSlotId,
        weights: LoraExpertWeights,
    ) -> Result<()> {
        self.insert_expert_shared(site, slot, Arc::new(weights))
    }

    pub fn insert_expert_shared(
        &mut self,
        site: &LoraExpertSiteHandle,
        slot: LoraSlotId,
        weights: Arc<LoraExpertWeights>,
    ) -> Result<()> {
        let site_id = self.expert_site_id(site)?;
        weights.validate_for_site(site)?;
        if self.adapters.contains_key(&slot) {
            candle_core::bail!("LoRA adapter weights were already installed for slot");
        }
        match self.expert_weights.entry((site_id, slot)) {
            Entry::Occupied(_) => {
                candle_core::bail!("expert LoRA weights were already installed for site and slot")
            }
            Entry::Vacant(entry) => {
                entry.insert(weights);
                self.row_remaps
                    .get_mut()
                    .expect("LoRA row remap cache poisoned")
                    .clear();
                #[cfg(feature = "cuda")]
                self.prepared_expert_adapters
                    .get_mut()
                    .expect("expert LoRA prepared cache poisoned")
                    .remove(&site_id);
            }
        }
        Ok(())
    }

    pub fn expert_weights(
        &self,
        site: &LoraExpertSiteHandle,
        slot: LoraSlotId,
    ) -> Result<Option<&Arc<LoraExpertWeights>>> {
        let site_id = self.expert_site_id(site)?;
        Ok(self
            .expert_weights
            .get(&(site_id, slot))
            .or_else(|| self.adapters.get(&slot)?.expert_weights(site_id)))
    }

    #[cfg(feature = "cuda")]
    pub(super) fn prepared_expert_adapters(
        &self,
        site: &LoraExpertSiteHandle,
    ) -> Result<Arc<PreparedExpertAdapters>> {
        let site_id = self.expert_site_id(site)?;
        let mut cache = self
            .prepared_expert_adapters
            .lock()
            .expect("expert LoRA prepared cache poisoned");
        if let Some(prepared) = cache.get(&site_id) {
            return Ok(prepared.clone());
        }
        let mut slots = Vec::new();
        for slot in self.rows_by_slot.keys() {
            if let Some(weights) = self.expert_weights(site, *slot)? {
                slots.push((*slot, weights.clone()));
            }
        }
        let compact = slots
            .iter()
            .enumerate()
            .map(|(index, (slot, _))| (*slot, index as u32))
            .collect::<HashMap<_, _>>();
        let token_slots = self
            .row_slots
            .iter()
            .map(|slot| {
                slot.and_then(|slot| compact.get(&slot).copied())
                    .unwrap_or(super::ROUTED_LORA_BASE_SLOT)
            })
            .collect();
        let prepared = Arc::new(PreparedExpertAdapters { slots, token_slots });
        cache.insert(site_id, prepared.clone());
        Ok(prepared)
    }

    pub fn site_is_active(&self, site: &LoraSiteHandle) -> Result<bool> {
        let site_id = self.site_id(site)?;
        Ok(self.rows_by_slot.keys().any(|slot| {
            self.weights.contains_key(&(site_id, *slot))
                || self
                    .adapters
                    .get(slot)
                    .is_some_and(|adapter| adapter.weights(site_id).is_some())
        }))
    }

    pub fn expert_site_is_active(&self, site: &LoraExpertSiteHandle) -> Result<bool> {
        let site_id = self.expert_site_id(site)?;
        Ok(self.rows_by_slot.keys().any(|slot| {
            self.expert_weights.contains_key(&(site_id, *slot))
                || self
                    .adapters
                    .get(slot)
                    .is_some_and(|adapter| adapter.expert_weights(site_id).is_some())
        }))
    }
}

thread_local! {
    static LORA_EXECUTIONS: RefCell<Vec<Arc<LoraExecution>>> = const { RefCell::new(Vec::new()) };
}

struct ExecutionGuard {
    count: usize,
}

impl Drop for ExecutionGuard {
    fn drop(&mut self) {
        LORA_EXECUTIONS.with(|executions| {
            let mut executions = executions.borrow_mut();
            let retained = executions
                .len()
                .checked_sub(self.count)
                .expect("LoRA execution scope stack underflow");
            executions.truncate(retained);
        });
    }
}

pub fn with_lora_execution<T>(execution: Option<Arc<LoraExecution>>, f: impl FnOnce() -> T) -> T {
    let Some(execution) = execution else {
        return f();
    };
    LORA_EXECUTIONS.with(|executions| executions.borrow_mut().push(execution));
    let _guard = ExecutionGuard { count: 1 };
    f()
}

fn with_remapped_lora_execution<T>(
    remap: LoraRowRemap,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let scoped = LORA_EXECUTIONS.with(|executions| {
        executions
            .borrow()
            .iter()
            .map(|execution| execution.remap_rows(remap))
            .collect::<Result<Vec<_>>>()
    })?;
    if scoped.is_empty() {
        return f();
    }
    let count = scoped.len();
    LORA_EXECUTIONS.with(|executions| executions.borrow_mut().extend(scoped));
    let _guard = ExecutionGuard { count };
    f()
}

pub fn with_lora_execution_row_range<T>(
    rows: Range<usize>,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if rows.start > rows.end {
        candle_core::bail!("LoRA route row range is reversed");
    }
    with_remapped_lora_execution(
        LoraRowRemap::Range {
            start: rows.start,
            end: rows.end,
        },
        f,
    )
}

pub fn with_lora_execution_repeated_row<T>(
    row: usize,
    rows: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_remapped_lora_execution(LoraRowRemap::Repeat { row, rows }, f)
}

pub(crate) fn current_lora_execution(runtime_id: LoraRuntimeId) -> Option<Arc<LoraExecution>> {
    LORA_EXECUTIONS.with(|executions| {
        executions
            .borrow()
            .iter()
            .rev()
            .find(|execution| execution.runtime_id == runtime_id)
            .cloned()
    })
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        panic::{catch_unwind, AssertUnwindSafe},
    };

    use super::*;
    use crate::{
        DynamicLoraWeights, LoraExpertProjectionNames, LoraExpertProjectionWeights,
        LoraExpertSiteSpec, LoraLayerRegistry, Shard,
    };

    fn finalized_site() -> Result<(LoraLayerRegistry, Arc<LoraSiteHandle>)> {
        let registry = LoraLayerRegistry::new();
        let site = registry.register(
            crate::LoraSiteKey::new("proj"),
            crate::LoraLinearSpec::replicated(1, 1),
            candle_core::DType::F32,
            candle_core::Device::Cpu,
        )?;
        registry.finalize()?;
        Ok((registry, site))
    }

    #[test]
    fn ragged_sequence_slots_expand_in_packed_order() -> Result<()> {
        let (registry, _) = finalized_site()?;
        let execution = LoraExecution::from_ragged_sequence_slots_with_arena(
            registry.runtime_id(),
            &[Some(3), None, Some(7)],
            &[2, 1, 3],
            Arc::new(LoraExecutionArena::new()),
        );

        assert_eq!(
            execution.row_slots(),
            &[Some(3), Some(3), None, Some(7), Some(7), Some(7)]
        );
        assert_eq!(execution.rows_by_slot()[&3].as_ref(), &[0, 1]);
        assert_eq!(execution.rows_by_slot()[&7].as_ref(), &[3, 4, 5]);
        Ok(())
    }

    fn scalar_weights(value: f32) -> Result<LoraWeights> {
        LoraWeights::new(
            Tensor::new(&[[value]], &candle_core::Device::Cpu)?,
            Tensor::new(&[[value]], &candle_core::Device::Cpu)?,
            1.0,
        )
    }

    fn adapter_weights(
        registry: &LoraLayerRegistry,
        site: Arc<LoraSiteHandle>,
        value: f32,
    ) -> Result<Arc<LoraAdapterWeights>> {
        Ok(Arc::new(LoraAdapterWeights::new(
            registry.runtime_id(),
            vec![(site, scalar_weights(value)?)],
        )?))
    }

    #[test]
    fn scopes_nest_and_restore() {
        let first_id = LoraLayerRegistry::new().runtime_id();
        let second_id = LoraLayerRegistry::new().runtime_id();
        let first = Arc::new(LoraExecution::new(first_id, vec![Some(1)]));
        let second = Arc::new(LoraExecution::new(second_id, vec![Some(2)]));

        with_lora_execution(Some(first), || {
            assert!(current_lora_execution(first_id).is_some());
            with_lora_execution(Some(second), || {
                assert!(current_lora_execution(first_id).is_some());
                assert!(current_lora_execution(second_id).is_some());
            });
            assert!(current_lora_execution(first_id).is_some());
            assert!(current_lora_execution(second_id).is_none());
        });
        assert!(current_lora_execution(first_id).is_none());
    }

    #[test]
    fn row_range_scopes_all_executions_and_restores_routes() -> Result<()> {
        let first_id = LoraLayerRegistry::new().runtime_id();
        let second_id = LoraLayerRegistry::new().runtime_id();
        let first = Arc::new(LoraExecution::new(
            first_id,
            vec![Some(1), None, Some(3), Some(3)],
        ));
        let second = Arc::new(LoraExecution::new(
            second_id,
            vec![Some(5), Some(6), Some(6), None],
        ));

        with_lora_execution(Some(first), || {
            with_lora_execution(Some(second), || {
                with_lora_execution_row_range(1..4, || {
                    let first = current_lora_execution(first_id).unwrap();
                    let second = current_lora_execution(second_id).unwrap();
                    assert_eq!(first.row_slots(), &[None, Some(3), Some(3)]);
                    assert_eq!(first.rows_by_slot()[&3].as_ref(), &[1, 2]);
                    assert_eq!(second.row_slots(), &[Some(6), Some(6), None]);
                    assert_eq!(second.rows_by_slot()[&6].as_ref(), &[0, 1]);
                    Ok(())
                })?;

                assert_eq!(
                    current_lora_execution(first_id).unwrap().row_slots(),
                    &[Some(1), None, Some(3), Some(3)]
                );
                assert_eq!(
                    current_lora_execution(second_id).unwrap().row_slots(),
                    &[Some(5), Some(6), Some(6), None]
                );
                Result::<()>::Ok(())
            })
        })
    }

    #[test]
    fn repeated_row_scope_expands_one_route() -> Result<()> {
        let runtime_id = LoraLayerRegistry::new().runtime_id();
        let execution = Arc::new(LoraExecution::new(runtime_id, vec![Some(3), None, Some(7)]));
        with_lora_execution(Some(execution), || {
            with_lora_execution_repeated_row(2, 4, || {
                let scoped = current_lora_execution(runtime_id).unwrap();
                assert_eq!(scoped.row_slots(), &[Some(7), Some(7), Some(7), Some(7)]);
                assert_eq!(scoped.rows_by_slot()[&7].as_ref(), &[0, 1, 2, 3]);
                Ok(())
            })
        })
    }

    #[test]
    fn row_remaps_reuse_execution_local_resources() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let mut execution =
            LoraExecution::new(registry.runtime_id(), vec![Some(3), None, Some(7), Some(7)]);
        let range = LoraRowRemap::Range { start: 1, end: 4 };
        let first = execution.remap_rows(range)?;
        let second = execution.remap_rows(range)?;
        let repeated = execution.remap_rows(LoraRowRemap::Repeat { row: 2, rows: 3 })?;

        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(&first, &repeated));
        assert_eq!(first.row_slots(), &[None, Some(7), Some(7)]);
        assert_eq!(repeated.row_slots(), &[Some(7), Some(7), Some(7)]);
        execution.insert(&site, 7, scalar_weights(2.0)?)?;
        let refreshed = execution.remap_rows(range)?;
        assert!(!Arc::ptr_eq(&first, &refreshed));
        assert!(refreshed.site_is_active(&site)?);
        Ok(())
    }

    #[test]
    fn row_scope_rejects_out_of_range_routes_without_running() {
        let runtime_id = LoraLayerRegistry::new().runtime_id();
        let execution = Arc::new(LoraExecution::new(runtime_id, vec![Some(1), Some(2)]));
        let called = Cell::new(false);
        let result = with_lora_execution(Some(execution.clone()), || {
            with_lora_execution_row_range(1..3, || {
                called.set(true);
                Result::<()>::Ok(())
            })
        });
        assert!(result.is_err());
        assert!(!called.get());
        let result = with_lora_execution(Some(execution), || {
            with_lora_execution_repeated_row(2, 4, || {
                called.set(true);
                Result::<()>::Ok(())
            })
        });
        assert!(result.is_err());
        assert!(!called.get());
        assert!(current_lora_execution(runtime_id).is_none());
        let invalid_range = std::ops::Range { start: 2, end: 1 };
        let result = with_lora_execution_row_range(invalid_range, || {
            called.set(true);
            Result::<()>::Ok(())
        });
        assert!(result.is_err());
        assert!(!called.get());
    }

    #[test]
    fn scope_is_restored_after_panic() {
        let runtime_id = LoraLayerRegistry::new().runtime_id();
        let execution = Arc::new(LoraExecution::new(runtime_id, vec![Some(1)]));
        let result = catch_unwind(AssertUnwindSafe(|| {
            with_lora_execution(Some(execution), || panic!("test panic"));
        }));
        assert!(result.is_err());
        assert!(current_lora_execution(runtime_id).is_none());
    }

    #[test]
    fn row_scope_is_restored_after_panic() {
        let runtime_id = LoraLayerRegistry::new().runtime_id();
        let execution = Arc::new(LoraExecution::new(
            runtime_id,
            vec![Some(1), Some(2), Some(3)],
        ));
        with_lora_execution(Some(execution), || {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let _: Result<()> =
                    with_lora_execution_row_range(1..3, || panic!("row scope panic"));
            }));
            assert!(result.is_err());
            assert_eq!(
                current_lora_execution(runtime_id).unwrap().row_slots(),
                &[Some(1), Some(2), Some(3)]
            );
        });
        assert!(current_lora_execution(runtime_id).is_none());
    }

    #[test]
    fn scopes_are_thread_local() {
        let first_id = LoraLayerRegistry::new().runtime_id();
        let second_id = LoraLayerRegistry::new().runtime_id();
        let first = Arc::new(LoraExecution::new(first_id, vec![Some(1)]));
        let second = Arc::new(LoraExecution::new(second_id, vec![Some(2)]));

        let a = std::thread::spawn(move || {
            with_lora_execution(Some(first), || {
                assert!(current_lora_execution(first_id).is_some());
                assert!(current_lora_execution(second_id).is_none());
            });
        });
        let b = std::thread::spawn(move || {
            with_lora_execution(Some(second), || {
                assert!(current_lora_execution(second_id).is_some());
                assert!(current_lora_execution(first_id).is_none());
            });
        });
        a.join().unwrap();
        b.join().unwrap();
    }

    #[test]
    fn duplicate_insert_does_not_replace_installed_weights() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert(&site, 3, scalar_weights(2.0)?)?;
        assert!(execution.insert(&site, 3, scalar_weights(7.0)?).is_err());
        let installed = execution
            .weights(&site, 3)?
            .expect("first weights remain installed");
        assert_eq!(installed.a.to_vec2::<f32>()?, vec![vec![2.0]]);
        Ok(())
    }

    #[test]
    fn persistent_adapter_table_routes_installed_weights() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let adapter = adapter_weights(&registry, site.clone(), 2.0)?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert_adapter(3, adapter)?;

        assert!(execution.site_is_active(&site)?);
        let installed = execution
            .weights(&site, 3)?
            .expect("adapter table contains site weights");
        assert_eq!(installed.a.to_vec2::<f32>()?, vec![vec![2.0]]);
        Ok(())
    }

    #[test]
    fn persistent_adapter_table_for_an_unrouted_slot_is_inactive() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let adapter = adapter_weights(&registry, site.clone(), 2.0)?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert_adapter(4, adapter)?;

        assert!(!execution.site_is_active(&site)?);
        assert!(execution.weights(&site, 4)?.is_some());
        Ok(())
    }

    #[test]
    fn adapter_table_rejects_duplicate_sites() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let weights = LoraAdapterWeights::new(
            registry.runtime_id(),
            vec![
                (site.clone(), scalar_weights(2.0)?),
                (site, scalar_weights(7.0)?),
            ],
        );
        assert!(weights.is_err());
        Ok(())
    }

    #[test]
    fn adapter_tables_must_match_their_runtime() -> Result<()> {
        let (registry, _) = finalized_site()?;
        let (foreign_registry, foreign_site) = finalized_site()?;
        assert!(LoraAdapterWeights::new(
            registry.runtime_id(),
            vec![(foreign_site.clone(), scalar_weights(2.0)?)],
        )
        .is_err());

        let foreign_adapter = adapter_weights(&foreign_registry, foreign_site, 2.0)?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        assert!(execution.insert_adapter(3, foreign_adapter).is_err());
        Ok(())
    }

    #[test]
    fn adapter_table_installation_is_unique_per_slot() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let first = adapter_weights(&registry, site.clone(), 2.0)?;
        let second = adapter_weights(&registry, site.clone(), 7.0)?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert_adapter(3, first)?;

        assert!(execution.insert_adapter(3, second).is_err());
        assert!(execution.insert(&site, 3, scalar_weights(7.0)?).is_err());
        let installed = execution
            .weights(&site, 3)?
            .expect("first adapter table remains installed");
        assert_eq!(installed.a.to_vec2::<f32>()?, vec![vec![2.0]]);
        Ok(())
    }

    #[test]
    fn adapter_table_owns_linear_and_expert_weights() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let linear = registry.register(
            crate::LoraSiteKey::new("proj"),
            crate::LoraLinearSpec::replicated(1, 1),
            candle_core::DType::F32,
            candle_core::Device::Cpu,
        )?;
        let expert = registry.register_expert(
            crate::LoraSiteKey::new("experts"),
            LoraExpertSiteSpec::new(
                1,
                1,
                1,
                LoraExpertProjectionNames::new("gate", "up", "down"),
                Shard::default(),
                Shard::default(),
            )?,
            candle_core::DType::F32,
            candle_core::Device::Cpu,
        )?;
        registry.finalize()?;
        let expert_projection = LoraExpertProjectionWeights::new(
            Tensor::new(&[[[2f32]]], &candle_core::Device::Cpu)?,
            Tensor::new(&[[[3f32]]], &candle_core::Device::Cpu)?,
            Tensor::new(&[0.5f32], &candle_core::Device::Cpu)?,
        )?;
        let expert_weights = LoraExpertWeights::new(&expert, Some(expert_projection), None, None)?;
        let adapter = Arc::new(LoraAdapterWeights::new(
            registry.runtime_id(),
            DynamicLoraWeights {
                linear: vec![(linear.clone(), scalar_weights(2.0)?)],
                experts: vec![(expert.clone(), expert_weights)],
            },
        )?);
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert_adapter(3, adapter)?;

        assert!(execution.site_is_active(&linear)?);
        assert!(execution.expert_site_is_active(&expert)?);
        assert!(execution.weights(&linear, 3)?.is_some());
        assert!(execution.expert_weights(&expert, 3)?.is_some());
        Ok(())
    }

    #[test]
    fn expert_weights_cannot_be_rebound_to_another_site() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let spec = || {
            LoraExpertSiteSpec::new(
                1,
                1,
                1,
                LoraExpertProjectionNames::new("gate", "up", "down"),
                Shard::default(),
                Shard::default(),
            )
        };
        let first = registry.register_expert(
            crate::LoraSiteKey::new("first.experts"),
            spec()?,
            candle_core::DType::F32,
            candle_core::Device::Cpu,
        )?;
        let second = registry.register_expert(
            crate::LoraSiteKey::new("second.experts"),
            spec()?,
            candle_core::DType::F32,
            candle_core::Device::Cpu,
        )?;
        registry.finalize()?;
        let projection = LoraExpertProjectionWeights::new(
            Tensor::new(&[[[1f32]]], &candle_core::Device::Cpu)?,
            Tensor::new(&[[[1f32]]], &candle_core::Device::Cpu)?,
            Tensor::new(&[1f32], &candle_core::Device::Cpu)?,
        )?;
        let weights = LoraExpertWeights::new(&first, Some(projection), None, None)?;

        assert!(LoraAdapterWeights::new(
            registry.runtime_id(),
            DynamicLoraWeights {
                linear: Vec::new(),
                experts: vec![(second, weights)],
            },
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn site_weights_block_adapter_table_for_the_same_slot() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let adapter = adapter_weights(&registry, site.clone(), 7.0)?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert(&site, 3, scalar_weights(2.0)?)?;

        assert!(execution.insert_adapter(3, adapter).is_err());
        let installed = execution
            .weights(&site, 3)?
            .expect("site weights remain installed");
        assert_eq!(installed.a.to_vec2::<f32>()?, vec![vec![2.0]]);
        Ok(())
    }

    #[test]
    fn weights_for_an_unrouted_slot_do_not_activate_the_site() -> Result<()> {
        let (registry, site) = finalized_site()?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert(&site, 4, scalar_weights(2.0)?)?;
        assert!(!execution.site_is_active(&site)?);
        Ok(())
    }

    #[test]
    fn sites_from_another_runtime_are_rejected() -> Result<()> {
        let (registry, _) = finalized_site()?;
        let (_, foreign_site) = finalized_site()?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        assert!(execution
            .insert(&foreign_site, 0, scalar_weights(2.0)?)
            .is_err());
        assert!(execution.site_is_active(&foreign_site).is_err());
        Ok(())
    }
}
