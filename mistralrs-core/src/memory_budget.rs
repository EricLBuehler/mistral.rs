use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use candle_core::{Device, DeviceLocation};

use crate::{
    memory_plan::device_memory_budget, utils::memory_usage::DeviceMemory, MemoryGoal, MemoryUsage,
};

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Default)]
struct RuntimeBudgetState {
    members: Vec<(std::sync::Weak<()>, bool)>,
    reservations: Vec<std::sync::Weak<RuntimeMemoryReservation>>,
    capture_disabled: bool,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug)]
pub struct RuntimeMemoryReservation {
    owner: Arc<ModelMemoryBudget>,
    bytes: HashMap<DeviceLocation, usize>,
}

#[cfg(any(feature = "cuda", test))]
impl RuntimeMemoryReservation {
    pub(crate) fn capture_ready(&self) -> bool {
        let mut runtime = self.owner.group.runtime.lock().unwrap();
        runtime
            .members
            .retain(|(member, _)| member.strong_count() > 0);
        !runtime.capture_disabled && runtime.members.iter().all(|(_, pending)| !pending)
    }

    pub(crate) fn other_workspace_bytes(&self, device: DeviceLocation) -> Result<usize> {
        let reservations = self
            .owner
            .group
            .runtime
            .lock()
            .unwrap()
            .reservations
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .collect::<Vec<_>>();
        reservations
            .iter()
            .filter(|reservation| !std::ptr::eq(reservation.as_ref(), self))
            .try_fold(0usize, |total, reservation| {
                total
                    .checked_add(reservation.bytes.get(&device).copied().unwrap_or_default())
                    .ok_or_else(|| anyhow::anyhow!("group runtime reservation overflow"))
            })
    }
}

#[cfg(any(feature = "cuda", test))]
impl Drop for RuntimeMemoryReservation {
    fn drop(&mut self) {
        self.owner
            .group
            .runtime
            .lock()
            .unwrap()
            .reservations
            .retain(|reservation| !std::ptr::eq(reservation.as_ptr(), self));
    }
}

#[derive(Debug, Default)]
pub(crate) struct MemoryBudgetGroup {
    goal: MemoryGoal,
    snapshots: Mutex<HashMap<DeviceLocation, DeviceMemory>>,
    #[cfg(any(feature = "cuda", test))]
    runtime: Mutex<RuntimeBudgetState>,
}

impl MemoryBudgetGroup {
    pub(crate) fn new(goal: MemoryGoal) -> Self {
        Self {
            goal,
            snapshots: Mutex::default(),
            #[cfg(any(feature = "cuda", test))]
            runtime: Mutex::default(),
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn disable_runtime_capture(&self) {
        self.runtime.lock().unwrap().capture_disabled = true;
    }
}

#[derive(Debug)]
pub(crate) struct ModelMemoryBudget {
    group: Arc<MemoryBudgetGroup>,
    share: f64,
    baselines: Mutex<HashMap<DeviceLocation, usize>>,
    #[cfg(any(feature = "cuda", test))]
    runtime_member: Arc<()>,
}

#[cfg(any(feature = "cuda", test))]
impl Drop for ModelMemoryBudget {
    fn drop(&mut self) {
        self.group
            .runtime
            .lock()
            .unwrap()
            .members
            .retain(|(member, _)| {
                !std::ptr::eq(member.as_ptr(), Arc::as_ptr(&self.runtime_member))
            });
    }
}

impl ModelMemoryBudget {
    pub(crate) fn new(group: Arc<MemoryBudgetGroup>, share: f64) -> Self {
        #[cfg(any(feature = "cuda", test))]
        let runtime_member = {
            let member = Arc::new(());
            group
                .runtime
                .lock()
                .unwrap()
                .members
                .push((Arc::downgrade(&member), true));
            member
        };
        Self {
            group,
            share,
            baselines: Mutex::new(HashMap::new()),
            #[cfg(any(feature = "cuda", test))]
            runtime_member,
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn reserve_runtime(
        self: &Arc<Self>,
        bytes: HashMap<DeviceLocation, usize>,
    ) -> Arc<RuntimeMemoryReservation> {
        let reservation = Arc::new(RuntimeMemoryReservation {
            owner: self.clone(),
            bytes,
        });
        let mut runtime = self.group.runtime.lock().unwrap();
        for (member, pending) in &mut runtime.members {
            if std::ptr::eq(member.as_ptr(), Arc::as_ptr(&self.runtime_member)) {
                *pending = false;
            }
        }
        runtime.reservations.push(Arc::downgrade(&reservation));
        reservation
    }

    pub(crate) fn memory_goal(&self) -> MemoryGoal {
        self.group.goal
    }

    pub(crate) fn initialize(&self, devices: &[Device]) -> Result<()> {
        #[cfg(any(feature = "cuda", test))]
        for (member, pending) in &mut self.group.runtime.lock().unwrap().members {
            if std::ptr::eq(member.as_ptr(), Arc::as_ptr(&self.runtime_member)) {
                *pending = true;
            }
        }
        for device in devices {
            self.begin_device_load(device.location(), MemoryUsage.query(device)?);
        }
        Ok(())
    }

    fn begin_device_load(&self, location: DeviceLocation, snapshot: DeviceMemory) {
        self.group
            .snapshots
            .lock()
            .unwrap()
            .entry(location)
            .or_insert(snapshot);
        self.baselines
            .lock()
            .unwrap()
            .insert(location, snapshot.allocated());
    }

    pub(crate) fn allocation_budget(
        &self,
        location: DeviceLocation,
        snapshot: DeviceMemory,
    ) -> Result<usize> {
        let initial = *self
            .group
            .snapshots
            .lock()
            .unwrap()
            .entry(location)
            .or_insert(snapshot);
        let baseline = *self
            .baselines
            .lock()
            .unwrap()
            .entry(location)
            .or_insert(snapshot.allocated());
        let initial_available = device_memory_budget(self.group.goal, initial)?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let quota = ((initial_available as f64 * self.share) as usize).min(initial_available);
        let loaded_bytes = snapshot.allocated().saturating_sub(baseline);
        Ok(quota
            .saturating_sub(loaded_bytes)
            .min(device_memory_budget(self.group.goal, snapshot)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1024 * 1024 * 1024;
    const TOTAL: usize = 100 * GIB;
    const WEIGHTS: usize = 10 * GIB;
    const CUDA: DeviceLocation = DeviceLocation::Cuda { gpu_id: 0 };
    const OTHER_CUDA: DeviceLocation = DeviceLocation::Cuda { gpu_id: 1 };

    fn discrete(occupied: usize) -> DeviceMemory {
        DeviceMemory::Discrete {
            total: TOTAL,
            free: TOTAL - occupied,
        }
    }

    #[test]
    fn equal_models_keep_equal_total_budgets_after_sequential_loads() -> Result<()> {
        let group = Arc::new(MemoryBudgetGroup::default());
        let first = ModelMemoryBudget::new(group.clone(), 0.5);
        let second = ModelMemoryBudget::new(group, 0.5);
        first.begin_device_load(CUDA, discrete(0));
        let quota = 95 * GIB / 2;
        assert_eq!(first.allocation_budget(CUDA, discrete(0))?, quota);
        let first_kv = first.allocation_budget(CUDA, discrete(WEIGHTS))?;
        assert_eq!(first_kv, quota - WEIGHTS);
        second.begin_device_load(CUDA, discrete(quota));
        assert_eq!(second.allocation_budget(CUDA, discrete(quota))?, quota);
        assert_eq!(
            second.allocation_budget(CUDA, discrete(quota + WEIGHTS))?,
            first_kv
        );
        Ok(())
    }

    #[test]
    fn live_pressure_and_each_device_still_limit_allocations() -> Result<()> {
        let model = ModelMemoryBudget::new(Arc::new(MemoryBudgetGroup::default()), 0.5);
        model.begin_device_load(CUDA, discrete(0));
        model.begin_device_load(OTHER_CUDA, discrete(20 * GIB));
        assert_eq!(
            model.allocation_budget(OTHER_CUDA, discrete(30 * GIB))?,
            75 * GIB / 2 - WEIGHTS
        );
        assert_eq!(model.allocation_budget(CUDA, discrete(96 * GIB))?, 0);
        Ok(())
    }

    #[test]
    fn reload_replaces_its_baseline_without_recomputing_the_group_share() -> Result<()> {
        let model = ModelMemoryBudget::new(Arc::new(MemoryBudgetGroup::default()), 0.5);
        model.begin_device_load(CUDA, discrete(0));
        model.begin_device_load(CUDA, discrete(40 * GIB));
        assert_eq!(
            model.allocation_budget(CUDA, discrete(50 * GIB))?,
            95 * GIB / 2 - WEIGHTS
        );
        Ok(())
    }
    #[test]
    fn shared_utilization_target_controls_every_member() -> Result<()> {
        let model = ModelMemoryBudget::new(
            Arc::new(MemoryBudgetGroup::new(MemoryGoal::Utilization(0.5))),
            0.5,
        );
        model.begin_device_load(CUDA, discrete(0));
        assert_eq!(
            model.allocation_budget(CUDA, discrete(WEIGHTS))?,
            25 * GIB - WEIGHTS
        );
        Ok(())
    }

    #[test]
    fn runtime_reservations_follow_loaded_models_and_device_placement() -> Result<()> {
        let group = Arc::new(MemoryBudgetGroup::default());
        let first = Arc::new(ModelMemoryBudget::new(group.clone(), 0.5));
        let second = Arc::new(ModelMemoryBudget::new(group.clone(), 0.5));
        let first_runtime = first.reserve_runtime(HashMap::from([(CUDA, 2 * GIB)]));
        assert!(!first_runtime.capture_ready());
        let second_runtime =
            second.reserve_runtime(HashMap::from([(CUDA, 3 * GIB), (OTHER_CUDA, 4 * GIB)]));
        assert!(first_runtime.capture_ready());
        assert_eq!(first_runtime.other_workspace_bytes(CUDA)?, 3 * GIB);
        assert_eq!(first_runtime.other_workspace_bytes(OTHER_CUDA)?, 4 * GIB);
        assert_eq!(second_runtime.other_workspace_bytes(CUDA)?, 2 * GIB);
        assert_eq!(second_runtime.other_workspace_bytes(OTHER_CUDA)?, 0);
        drop(second_runtime);
        assert!(first_runtime.capture_ready());
        assert_eq!(first_runtime.other_workspace_bytes(CUDA)?, 0);
        assert_eq!(group.runtime.lock().unwrap().reservations.len(), 1);
        second.initialize(&[])?;
        assert!(!first_runtime.capture_ready());
        drop(second);
        assert!(first_runtime.capture_ready());
        assert_eq!(group.runtime.lock().unwrap().members.len(), 1);
        Ok(())
    }

    #[test]
    fn runtime_reservations_sum_other_models_with_checked_arithmetic() -> Result<()> {
        let group = Arc::new(MemoryBudgetGroup::default());
        let first = Arc::new(ModelMemoryBudget::new(group.clone(), 0.4));
        let second = Arc::new(ModelMemoryBudget::new(group.clone(), 0.4));
        let third = Arc::new(ModelMemoryBudget::new(group, 0.2));
        let first_runtime = first.reserve_runtime(HashMap::from([(CUDA, 2)]));
        let second_runtime = second.reserve_runtime(HashMap::from([(CUDA, 3)]));
        let third_runtime = third.reserve_runtime(HashMap::from([(CUDA, 5)]));
        assert_eq!(first_runtime.other_workspace_bytes(CUDA)?, 8);
        drop(third_runtime);
        let third_runtime = third.reserve_runtime(HashMap::from([(CUDA, usize::MAX)]));
        assert!(first_runtime.other_workspace_bytes(CUDA).is_err());
        drop(second_runtime);
        assert_eq!(first_runtime.other_workspace_bytes(CUDA)?, usize::MAX);
        drop(third_runtime);
        assert_eq!(first_runtime.other_workspace_bytes(CUDA)?, 0);
        Ok(())
    }
}
