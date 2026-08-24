use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdmissionClass {
    Workload { sequences: usize },
    OrderedControl,
    BypassControl,
    Shutdown,
}

#[derive(Debug)]
struct PendingRequest<T> {
    class: AdmissionClass,
    request: T,
}

#[derive(Debug)]
pub(super) struct AdmissionPolicy {
    max_active_sequences: usize,
    max_pending_requests: usize,
    max_dispatches_per_step: usize,
}

impl AdmissionPolicy {
    pub(super) fn new(max_active_sequences: usize, max_pending_requests: usize) -> Self {
        assert!(max_active_sequences > 0);
        assert!(max_pending_requests > 0);
        Self {
            max_active_sequences,
            max_pending_requests,
            max_dispatches_per_step: max_active_sequences,
        }
    }

    pub(super) fn max_dispatches_per_step(&self) -> usize {
        self.max_dispatches_per_step
    }

    fn available_sequences(&self, active_sequences: usize) -> usize {
        self.max_active_sequences.saturating_sub(active_sequences)
    }
}

#[derive(Debug)]
pub(super) struct AdmissionQueue<T> {
    ordered: VecDeque<PendingRequest<T>>,
    bypass_controls: VecDeque<T>,
    shutdown: VecDeque<T>,
    policy: AdmissionPolicy,
}

impl<T> AdmissionQueue<T> {
    pub(super) fn new(policy: AdmissionPolicy) -> Self {
        Self {
            ordered: VecDeque::new(),
            bypass_controls: VecDeque::new(),
            shutdown: VecDeque::new(),
            policy,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn remaining_capacity(&self) -> usize {
        self.policy.max_pending_requests.saturating_sub(self.len())
    }

    pub(super) fn max_dispatches_per_step(&self) -> usize {
        self.policy.max_dispatches_per_step()
    }

    #[cfg(any(feature = "cuda", test))]
    pub(super) fn blocks_decode_continuation(&self) -> bool {
        !self.bypass_controls.is_empty()
            || !self.shutdown.is_empty()
            || self
                .ordered
                .iter()
                .any(|pending| matches!(pending.class, AdmissionClass::OrderedControl))
    }

    pub(super) fn push(&mut self, request: T, class: AdmissionClass) -> Result<(), T> {
        if self.len() == self.policy.max_pending_requests {
            return Err(request);
        }
        match class {
            AdmissionClass::BypassControl => self.bypass_controls.push_back(request),
            AdmissionClass::Shutdown => self.shutdown.push_back(request),
            AdmissionClass::Workload { .. } | AdmissionClass::OrderedControl => {
                self.ordered.push_back(PendingRequest { class, request });
            }
        }
        Ok(())
    }

    pub(super) fn take_shutdown(&mut self) -> Option<T> {
        self.shutdown.pop_front()
    }

    pub(super) fn take_bypass_control(&mut self) -> Option<T> {
        self.bypass_controls.pop_front()
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.ordered.retain(|pending| keep(&pending.request));
        self.bypass_controls.retain(&mut keep);
        self.shutdown.retain(keep);
    }

    pub(super) fn pop_admissible(&mut self, active_sequences: usize) -> Option<T> {
        let pending = self.ordered.front()?;
        let admissible = match pending.class {
            AdmissionClass::Workload { sequences } => {
                let sequences = sequences.max(1);
                let available = self.policy.available_sequences(active_sequences);
                sequences <= available
                    || active_sequences == 0 && sequences > self.policy.max_active_sequences
            }
            AdmissionClass::OrderedControl
            | AdmissionClass::BypassControl
            | AdmissionClass::Shutdown => true,
        };
        admissible.then(|| self.ordered.pop_front().unwrap().request)
    }

    pub(super) fn pop_admissible_workload(&mut self, active_sequences: usize) -> Option<T> {
        let pending = self.ordered.front()?;
        let AdmissionClass::Workload { sequences } = pending.class else {
            return None;
        };
        let sequences = sequences.max(1);
        let available = self.policy.available_sequences(active_sequences);
        (sequences <= available
            || active_sequences == 0 && sequences > self.policy.max_active_sequences)
            .then(|| self.ordered.pop_front().unwrap().request)
    }

    pub(super) fn len(&self) -> usize {
        self.ordered.len() + self.bypass_controls.len() + self.shutdown.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload(id: usize) -> (usize, AdmissionClass) {
        (id, AdmissionClass::Workload { sequences: 1 })
    }

    #[test]
    fn idle_burst_fills_one_serving_wave_in_order() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(16, 64));
        for id in 0..32 {
            let (request, class) = workload(id);
            queue.push(request, class).unwrap();
        }

        let mut active = 0;
        let mut admitted = Vec::new();
        while admitted.len() < queue.max_dispatches_per_step() {
            let Some(request) = queue.pop_admissible(active) else {
                break;
            };
            admitted.push(request);
            active += 1;
        }

        assert_eq!(admitted, (0..16).collect::<Vec<_>>());
        assert_eq!(queue.pop_admissible(active), None);
    }

    #[test]
    fn full_decode_batch_backpressures_work_but_not_cancellation() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(8, 16));
        for id in 0..4 {
            let (request, class) = workload(id);
            queue.push(request, class).unwrap();
        }
        queue.push(99, AdmissionClass::BypassControl).unwrap();

        assert_eq!(queue.take_bypass_control(), Some(99));
        assert_eq!(queue.pop_admissible(8), None);
        assert_eq!(queue.pop_admissible(6), Some(0));
        assert_eq!(queue.pop_admissible(7), Some(1));
        assert_eq!(queue.pop_admissible(8), None);
    }

    #[test]
    fn head_request_is_not_skipped_when_it_needs_multiple_slots() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(4, 8));
        queue
            .push(0, AdmissionClass::Workload { sequences: 2 })
            .unwrap();
        queue
            .push(1, AdmissionClass::Workload { sequences: 1 })
            .unwrap();

        assert_eq!(queue.pop_admissible(3), None);
        assert_eq!(queue.pop_admissible(2), Some(0));
        assert_eq!(queue.pop_admissible(3), Some(1));
    }

    #[test]
    fn mutable_controls_remain_fifo_barriers() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(2, 8));
        let (request, class) = workload(0);
        queue.push(request, class).unwrap();
        queue.push(10, AdmissionClass::OrderedControl).unwrap();
        let (request, class) = workload(1);
        queue.push(request, class).unwrap();

        assert_eq!(queue.pop_admissible(2), None);
        assert_eq!(queue.pop_admissible(1), Some(0));
        assert_eq!(queue.pop_admissible(2), Some(10));
        assert_eq!(queue.pop_admissible(1), Some(1));
    }

    #[test]
    fn shutdown_bypasses_the_pending_workload() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(2, 8));
        for id in 0..4 {
            let (request, class) = workload(id);
            queue.push(request, class).unwrap();
        }
        queue.push(99, AdmissionClass::Shutdown).unwrap();

        assert_eq!(queue.take_shutdown(), Some(99));
        assert_eq!(queue.pop_admissible(0), Some(0));
    }

    #[test]
    fn dispatch_budget_bounds_rejected_request_work() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(4, 16));
        for id in 0..12 {
            let (request, class) = workload(id);
            queue.push(request, class).unwrap();
        }

        let dispatched = (0..queue.max_dispatches_per_step())
            .map(|_| queue.pop_admissible(0).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(dispatched, vec![0, 1, 2, 3]);
        assert!(!queue.is_empty());
    }

    #[test]
    fn pending_queue_applies_backpressure_at_its_configured_limit() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(2, 2));
        let (request, class) = workload(0);
        queue.push(request, class).unwrap();
        let (request, class) = workload(1);
        queue.push(request, class).unwrap();
        let (request, class) = workload(2);

        assert_eq!(queue.push(request, class), Err(2));
        assert_eq!(queue.remaining_capacity(), 0);
    }

    #[test]
    fn controls_stop_resident_decode_continuation() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(2, 8));
        let (request, class) = workload(0);
        queue.push(request, class).unwrap();
        assert!(!queue.blocks_decode_continuation());

        queue.push(10, AdmissionClass::OrderedControl).unwrap();
        assert!(queue.blocks_decode_continuation());
        assert_eq!(queue.pop_admissible_workload(0), Some(0));
        assert_eq!(queue.pop_admissible_workload(0), None);
    }

    #[test]
    fn retain_removes_abandoned_work_without_reordering() {
        let mut queue = AdmissionQueue::new(AdmissionPolicy::new(4, 8));
        for id in 0..4 {
            let (request, class) = workload(id);
            queue.push(request, class).unwrap();
        }
        queue.retain(|id| id % 2 == 0);

        assert_eq!(queue.pop_admissible(0), Some(0));
        assert_eq!(queue.pop_admissible(1), Some(2));
        assert!(queue.is_empty());
    }
}
