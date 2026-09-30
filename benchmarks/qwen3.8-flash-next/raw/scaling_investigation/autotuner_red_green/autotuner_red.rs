//! Adapts speculative depth from acceptance rates and the measured cost of drafting and verification.

#![allow(clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

// Half-lives count draft trials, not steps, so the time scale does not depend on the depth in use
const SEQUENCE_HALF_LIFE_TRIALS: f64 = 48.0;
const POPULATION_HALF_LIFE_TRIALS: f64 = 512.0;
// Pseudo-counts pulling a position toward the one before it, and a sequence toward the population curve
const PRIOR_TRIALS: f64 = 8.0;
const INITIAL_ACCEPTANCE: f64 = 0.8;
// Realized over predicted tokens per depth, decayed per verified sequence and pulled toward 1 by this many
const CALIBRATION_HALF_LIFE_OUTCOMES: f64 = 256.0;
const CALIBRATION_PRIOR_OUTCOMES: f64 = 16.0;
const MAX_ACCEPTANCE: f64 = 0.999;
// Estimates average their samples until they hold this many, then track at this weight
const COST_MIN_WEIGHT: f64 = 0.25;
// Step periods this many times the estimate are stalls, unless a streak of them shows the level moved; a sample
// this many times below it means the estimate itself came from a stall
const COST_OUTLIER_RATIO: f64 = 3.0;
const COST_OUTLIER_STREAK: u32 = 3;
// Batches are timed per exact size up to this, then in steps of the granularity, like the CUDA graph buckets
const EXACT_BATCH_KEYS: usize = 8;
const BATCH_KEY_GRANULARITY: usize = 8;
// A new depth must beat the current one by this factor before the batch switches
const SWITCH_MARGIN: f64 = 1.03;
// Every this many decisions per batch size the stalest neighboring depth is measured once
const PROBE_INTERVAL: u64 = 32;
// Sequences not seen for this many decisions are forgotten
const STALE_DECISIONS: u64 = 4096;
// Every candidate needs direct samples because graph and kernel boundaries can make costs non-monotonic
const WARMUP_SAMPLES: u32 = 2;
const WARMUP_PROBE_INTERVAL: u64 = 4;

#[cfg(test)]
const TEST_POPULATION_RATE: f64 = 0.9;

static INTERRUPTIONS: AtomicU64 = AtomicU64::new(0);

/// Marks work that runs between drafting and verification (a prompt step) so that step is not timed.
pub(crate) fn note_interruption() {
    INTERRUPTIONS.fetch_add(1, Ordering::Relaxed);
}

/// Depths a drafter left on automatic depth chooses between; the last is its maximum.
pub const AUTO_DEPTHS: [usize; 4] = [2, 3, 4, 6];
pub const AUTO_MAX_DEPTH: usize = AUTO_DEPTHS[AUTO_DEPTHS.len() - 1];
// Depths other than a drafter's primary one are graphed up to this batch; larger batches run them eagerly
const AUTO_EXTRA_DEPTH_GRAPH_MAX_BATCH: usize = 8;

/// `candidates` below `max_depth`, plus `max_depth` itself.
pub fn depths_up_to(candidates: &[usize], max_depth: usize) -> Vec<usize> {
    let mut depths = candidates
        .iter()
        .copied()
        .filter(|depth| *depth < max_depth)
        .collect::<Vec<_>>();
    depths.push(max_depth);
    depths
}

/// Graph plans for autotuned `depths`: `primary` at every batch size, the others for small batches.
pub fn auto_depth_graph_plans(
    depths: &[usize],
    primary: usize,
) -> Vec<super::SpeculativeGraphPlan> {
    depths
        .iter()
        .map(|&depth| {
            super::SpeculativeGraphPlan::new(
                depth,
                (depth != primary).then_some(AUTO_EXTRA_DEPTH_GRAPH_MAX_BATCH),
            )
        })
        .collect()
}

/// Acceptance of one draft position given every earlier draft was accepted.
#[derive(Clone, Copy, Debug, Default)]
struct PositionRate {
    hits: f64,
    trials: f64,
}

/// A sequence's hits against the hits the population curve predicted for the same trials.
#[derive(Clone, Copy, Debug, Default)]
struct SequenceAcceptance {
    hits: f64,
    predicted: f64,
    last_seen: u64,
}

impl SequenceAcceptance {
    fn scale(&self) -> f64 {
        (self.hits + PRIOR_TRIALS) / (self.predicted + PRIOR_TRIALS)
    }
}

/// Expected tokens one step commits when draft `k` lands with probability `rates[k]` given the ones before.
pub fn expected_tokens(rates: &[f64]) -> f64 {
    1.0 + rates
        .iter()
        .scan(1.0, |landed, rate| {
            *landed *= rate;
            Some(*landed)
        })
        .sum::<f64>()
}

/// One verified sequence of a step: drafts proposed and how many of them the target accepted.
#[derive(Clone, Copy, Debug)]
pub struct DraftOutcome {
    pub seq_id: usize,
    pub proposed: usize,
    pub accepted: usize,
}

#[derive(Clone, Copy, Debug)]
struct CostEstimate {
    secs: f64,
    outliers: u32,
    samples: u32,
    // Decision clock of the last sample
    updated: u64,
}

impl CostEstimate {
    fn update(&mut self, sample: f64, clock: u64) {
        self.updated = clock;
        self.samples = self.samples.saturating_add(1);
        // The first run of a shape pays one-off costs (kernel JIT, first graph replay) and noise only adds
        // time, so the second sample starts the estimate at the faster of the two
        if self.samples == 1 {
            self.secs = self.secs.min(sample);
            return;
        }
        if sample * COST_OUTLIER_RATIO < self.secs {
            self.secs = sample;
            self.outliers = 0;
        } else if sample > self.secs * COST_OUTLIER_RATIO {
            self.outliers += 1;
            if self.outliers >= COST_OUTLIER_STREAK {
                self.secs = sample;
                self.outliers = 0;
            }
        } else {
            let weight = (1.0 / f64::from(self.samples)).max(COST_MIN_WEIGHT);
            self.secs += weight * (sample - self.secs);
            self.outliers = 0;
        }
    }
}

fn record_cost<K: std::hash::Hash + Eq>(
    costs: &mut HashMap<K, CostEstimate>,
    key: K,
    sample: f64,
    clock: u64,
) {
    costs
        .entry(key)
        .and_modify(|cost| cost.update(sample, clock))
        .or_insert(CostEstimate {
            secs: sample,
            outliers: 0,
            samples: 0,
            updated: clock,
        });
}

#[derive(Clone, Debug)]
struct StepBoundary {
    seq_ids: Vec<usize>,
    depth: usize,
    at: Instant,
    interruptions: u64,
}

/// Prompt steps plus CUDA graph captures so far; a timed step that spans a change is not representative.
fn interruptions() -> u64 {
    #[cfg(feature = "cuda")]
    let captures = crate::pipeline::cuda_graph::cuda_graph_capture_count();
    #[cfg(not(feature = "cuda"))]
    let captures = 0;
    INTERRUPTIONS.load(Ordering::Relaxed) + captures
}

#[derive(Default)]
pub struct SpeculativeAutotuner {
    candidates: Vec<usize>,
    sequences: HashMap<usize, SequenceAcceptance>,
    positions: Vec<PositionRate>,
    // depth -> (realized, predicted) tokens, which corrects the curve where it is biased (deep positions are
    // mostly measured on the sequences easy enough to draft deep)
    calibration: HashMap<usize, (f64, f64)>,
    // (batch key, depth) -> seconds from the start of drafting to the end of verification
    step_cost: HashMap<(usize, usize), CostEstimate>,
    current: HashMap<usize, usize>,
    decisions: HashMap<usize, u64>,
    clock: u64,
    boundary: Option<StepBoundary>,
}

fn batch_key(batch: usize) -> usize {
    if batch <= EXACT_BATCH_KEYS {
        batch.max(1)
    } else {
        batch.div_ceil(BATCH_KEY_GRANULARITY) * BATCH_KEY_GRANULARITY
    }
}

impl SpeculativeAutotuner {
    /// Adopts a new candidate set, forgetting everything learned for the old one.
    pub fn set_candidates(&mut self, candidates: &[usize]) {
        let mut candidates = candidates.to_vec();
        candidates.sort_unstable();
        candidates.dedup();
        if candidates != self.candidates {
            *self = Self {
                candidates,
                ..Self::default()
            };
        }
    }

    pub fn is_active(&self) -> bool {
        self.candidates.len() > 1
    }

    /// Population acceptance of the first `depth` draft positions.
    fn position_rates(&self, depth: usize) -> Vec<f64> {
        let mut prior = INITIAL_ACCEPTANCE;
        (0..depth)
            .map(|position| {
                let observed = self.positions.get(position).copied().unwrap_or_default();
                prior = ((observed.hits + PRIOR_TRIALS * prior) / (observed.trials + PRIOR_TRIALS))
                    .min(MAX_ACCEPTANCE);
                prior
            })
            .collect()
    }

    fn sequence_tokens(&self, seq_id: usize, population: &[f64]) -> f64 {
        let scale = self
            .sequences
            .get(&seq_id)
            .map_or(1.0, SequenceAcceptance::scale);
        let rates = population
            .iter()
            .map(|rate| (rate * scale).min(MAX_ACCEPTANCE))
            .collect::<Vec<_>>();
        expected_tokens(&rates)
    }

    fn calibration(&self, depth: usize) -> f64 {
        let (realized, predicted) = self.calibration.get(&depth).copied().unwrap_or_default();
        let prior = CALIBRATION_PRIOR_OUTCOMES;
        (realized + prior) / (predicted + prior)
    }

    fn observe(&mut self, outcome: &DraftOutcome) {
        let predicted = self.position_rates(outcome.proposed);
        let expected = self.sequence_tokens(outcome.seq_id, &predicted);
        let decay = 0.5f64.powf(1.0 / CALIBRATION_HALF_LIFE_OUTCOMES);
        let calibration = self.calibration.entry(outcome.proposed).or_default();
        calibration.0 = calibration.0 * decay + (outcome.accepted + 1) as f64 / expected;
        calibration.1 = calibration.1 * decay + 1.0;
        let tried = outcome.accepted + usize::from(outcome.accepted < outcome.proposed);
        if self.positions.len() < tried {
            self.positions.resize(tried, PositionRate::default());
        }
        let population_decay = 0.5f64.powf(1.0 / POPULATION_HALF_LIFE_TRIALS);
        let sequence_decay = 0.5f64.powf(1.0 / SEQUENCE_HALF_LIFE_TRIALS);
        let sequence = self.sequences.entry(outcome.seq_id).or_default();
        for (position, predicted) in predicted.iter().enumerate().take(tried) {
            let hit = if position < outcome.accepted {
                1.0
            } else {
                0.0
            };
            let rate = &mut self.positions[position];
            rate.hits = rate.hits * population_decay + hit;
            rate.trials = rate.trials * population_decay + 1.0;
            sequence.hits = sequence.hits * sequence_decay + hit;
            sequence.predicted = sequence.predicted * sequence_decay + predicted;
        }
    }

    fn cell(&self, key: usize, depth: usize) -> Option<f64> {
        self.step_cost
            .get(&(key, depth))
            .filter(|cost| cost.samples > 0)
            .map(|cost| cost.secs)
    }

    /// Measured batch keys other than `key`, nearest first and larger first on ties.
    fn neighbor_keys(&self, key: usize) -> Vec<usize> {
        let mut keys = self
            .step_cost
            .iter()
            .filter(|((k, _), cost)| *k != key && cost.samples > 0)
            .map(|((k, _), _)| *k)
            .collect::<Vec<_>>();
        keys.sort_unstable_by_key(|k| (k.abs_diff(key), std::cmp::Reverse(*k)));
        keys.dedup();
        keys
    }

    /// Candidates measured at `key`, nearest to `depth` first.
    fn measured_near(&self, key: usize, depth: usize) -> Vec<(usize, f64)> {
        let mut measured = self
            .candidates
            .iter()
            .filter_map(|d| Some((*d, self.cell(key, *d)?)))
            .collect::<Vec<_>>();
        measured.sort_unstable_by_key(|(d, _)| d.abs_diff(depth));
        measured
    }

    /// Cost of an unmeasured depth from the nearest measured one at the same batch: a deeper draft verifies
    /// proportionally more tokens, a shallower one is taken as no cheaper until measured.
    fn within_batch(&self, key: usize, depth: usize) -> Option<f64> {
        let &(near, secs) = self.measured_near(key, depth).first()?;
        Some(if depth > near {
            secs * (1 + depth) as f64 / (1 + near) as f64
        } else {
            secs
        })
    }

    /// Estimated seconds of one step at `depth` for a batch key, or `None` before anything was measured.
    fn step_seconds(&self, key: usize, depth: usize) -> Option<f64> {
        if let Some(secs) = self.cell(key, depth) {
            return Some(secs);
        }
        let neighbors = self.neighbor_keys(key);
        // The nearest batch that measured this depth, rescaled by a depth both batches measured
        for &other in &neighbors {
            let Some(secs) = self.cell(other, depth) else {
                continue;
            };
            let shared = self
                .measured_near(key, depth)
                .into_iter()
                .find_map(|(d, here)| Some(here / self.cell(other, d)?));
            if let Some(scale) = shared {
                return Some(secs * scale);
            }
        }
        if let Some(secs) = self.within_batch(key, depth) {
            return Some(secs);
        }
        // A batch never timed borrows the nearest one's costs; deeper drafts cost more per extra draft in
        // proportion to the batch, so large batches start shallow
        let other = *neighbors.first()?;
        let shallowest = self.candidates.first()?;
        let base = self.step_seconds(other, *shallowest)?;
        let secs = self.step_seconds(other, depth)?;
        let growth = (key as f64 / other as f64).max(1.0);
        Some(base + (secs - base).max(0.0) * growth)
    }

    fn score(&self, seq_ids: &[usize], key: usize, depth: usize) -> Option<f64> {
        let population = self.position_rates(depth);
        let tokens = seq_ids
            .iter()
            .map(|seq_id| self.sequence_tokens(*seq_id, &population))
            .sum::<f64>()
            * self.calibration(depth);
        self.step_seconds(key, depth)
            .map(|secs| tokens / secs.max(f64::MIN_POSITIVE))
    }

    pub(crate) fn choose_with_hint(
        &mut self,
        seq_ids: &[usize],
        max_depth: usize,
        depth_hint: Option<usize>,
    ) -> usize {
        if !self.is_active() {
            return max_depth;
        }
        if let Some(depth) =
            depth_hint.filter(|depth| *depth <= max_depth && self.candidates.contains(depth))
        {
            tracing::debug!(
                batch = seq_ids.len(),
                chosen = depth,
                reason = "cohort_hint",
                "speculative depth decision"
            );
            return depth;
        }
        self.choose(seq_ids, max_depth)
    }

    /// The depth the batch of `seq_ids` should draft at, at most `max_depth`.
    pub fn choose(&mut self, seq_ids: &[usize], max_depth: usize) -> usize {
        self.clock += 1;
        let clock = self.clock;
        for seq_id in seq_ids {
            self.sequences.entry(*seq_id).or_default().last_seen = clock;
        }
        if clock.is_multiple_of(STALE_DECISIONS) {
            self.sequences
                .retain(|_, seq| clock - seq.last_seen < STALE_DECISIONS);
        }
        let candidates = self
            .candidates
            .iter()
            .copied()
            .filter(|depth| *depth <= max_depth)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return max_depth;
        }
        let key = batch_key(seq_ids.len());
        let current = self
            .current
            .get(&key)
            .copied()
            .filter(|depth| candidates.contains(depth));
        let mut ranked = candidates
            .iter()
            .filter_map(|depth| Some((*depth, self.score(seq_ids, key, *depth)?)))
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let chosen = match (ranked.first(), current) {
            // Nothing measured yet: start at the middle candidate
            (None, _) => current.unwrap_or(candidates[candidates.len() / 2]),
            (Some(&(best, best_score)), Some(current)) if candidates.contains(&current) => {
                let current_score = ranked
                    .iter()
                    .find(|(depth, _)| *depth == current)
                    .map_or(0.0, |(_, score)| *score);
                if best_score > current_score * SWITCH_MARGIN {
                    best
                } else {
                    current
                }
            }
            (Some(&(best, _)), _) => best,
        };
        self.current.insert(key, chosen);
        let decisions = self.decisions.entry(key).or_default();
        *decisions += 1;
        let decisions = *decisions;
        let position = candidates
            .iter()
            .position(|depth| *depth == chosen)
            .unwrap_or_default();
        let neighbors = [
            position.checked_sub(1).map(|i| candidates[i]),
            candidates.get(position + 1).copied(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let samples = |depth: usize| {
            self.step_cost
                .get(&(key, depth))
                .map_or(0, |cost| cost.samples)
        };
        let undersampled = neighbors
            .iter()
            .copied()
            .filter(|depth| samples(*depth) < WARMUP_SAMPLES)
            .min_by_key(|depth| samples(*depth));
        if let Some(depth) = undersampled {
            if decisions.is_multiple_of(WARMUP_PROBE_INTERVAL) {
                return depth;
            }
        }
        if decisions.is_multiple_of(PROBE_INTERVAL) {
            let stalest = neighbors.iter().copied().min_by_key(|depth| {
                self.step_cost
                    .get(&(key, *depth))
                    .map_or(0, |cost| cost.updated)
            });
            if let Some(depth) = stalest {
                return depth;
            }
        }
        chosen
    }

    /// The sequences in `seq_ids` started drafting `depth` tokens at `started`.
    pub fn begin_step(&mut self, seq_ids: &[usize], depth: usize, started: Instant) {
        self.boundary = (depth > 0).then(|| StepBoundary {
            seq_ids: seq_ids.to_vec(),
            depth,
            at: started,
            interruptions: interruptions(),
        });
    }

    pub fn cancel_step(&mut self) {
        self.boundary = None;
    }

    /// The target step verifying the last proposal finished at `verified_at` with these outcomes.
    pub fn record_verification(&mut self, verified_at: Instant, outcomes: &[DraftOutcome]) {
        if let Some(boundary) = self.boundary.take() {
            if boundary.seq_ids.len() == outcomes.len()
                && boundary
                    .seq_ids
                    .iter()
                    .zip(outcomes)
                    .all(|(seq_id, outcome)| {
                        *seq_id == outcome.seq_id && boundary.depth == outcome.proposed
                    })
                && boundary.interruptions == interruptions()
            {
                let secs = verified_at
                    .saturating_duration_since(boundary.at)
                    .as_secs_f64();
                record_cost(
                    &mut self.step_cost,
                    (batch_key(boundary.seq_ids.len()), boundary.depth),
                    secs,
                    self.clock,
                );
            }
        }
        for outcome in outcomes {
            self.observe(outcome);
        }
    }

    pub fn release(&mut self, seq_ids: &[usize]) {
        for seq_id in seq_ids {
            self.sequences.remove(seq_id);
        }
    }

    #[cfg(test)]
    fn set_costs(&mut self, key: usize, costs: &[(usize, f64)]) {
        let estimate = |secs| CostEstimate {
            secs,
            outliers: 0,
            samples: WARMUP_SAMPLES,
            updated: 0,
        };
        for (depth, secs) in costs {
            self.step_cost.insert((key, *depth), estimate(*secs));
        }
    }

    #[cfg(test)]
    fn set_positions(&mut self, rates: &[f64]) {
        let trials = 1e6;
        self.positions = rates
            .iter()
            .map(|rate| PositionRate {
                hits: rate * trials,
                trials,
            })
            .collect();
    }

    /// A flat population curve at `TEST_POPULATION_RATE`, scaled to `rate` for this sequence.
    #[cfg(test)]
    fn set_rate(&mut self, seq_id: usize, rate: f64) {
        if self.positions.is_empty() {
            self.set_positions(&[TEST_POPULATION_RATE; 8]);
        }
        let trials = 1e6;
        self.sequences.insert(
            seq_id,
            SequenceAcceptance {
                hits: rate / TEST_POPULATION_RATE * trials,
                predicted: trials,
                last_seen: 0,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn tuner(candidates: &[usize]) -> SpeculativeAutotuner {
        let mut tuner = SpeculativeAutotuner::default();
        tuner.set_candidates(candidates);
        tuner
    }

    #[test]
    fn expected_tokens_is_the_geometric_series() {
        assert!((expected_tokens(&[0.9; 2]) - 2.71).abs() < 1e-9);
        assert!((expected_tokens(&[0.5; 6]) - 1.984375).abs() < 1e-9);
        assert_eq!(expected_tokens(&[0.0; 4]), 1.0);
        assert!((expected_tokens(&[0.5, 0.5]) - 1.75).abs() < 1e-9);
    }

    #[test]
    fn batch_weighs_each_sequence_by_its_gain() {
        let mut t = tuner(&[2, 4, 6]);
        t.set_costs(2, &[(2, 1.0), (4, 1.25), (6, 1.5)]);
        t.set_rate(0, 0.9);
        t.set_rate(1, 0.5);
        assert_eq!(t.choose(&[0, 1], 6), 4);
        let mut t = tuner(&[2, 4, 6]);
        t.set_costs(2, &[(2, 1.0), (4, 1.25), (6, 1.5)]);
        t.set_rate(0, 0.9);
        t.set_rate(1, 0.3);
        assert_eq!(t.choose(&[0, 1], 6), 6);
    }

    #[test]
    fn steeper_step_cost_picks_shallower_drafts() {
        let mut t = tuner(&[2, 4, 6]);
        t.set_costs(8, &[(2, 1.0), (4, 2.0), (6, 3.0)]);
        let ids = (0..8).collect::<Vec<_>>();
        for id in &ids {
            t.set_rate(*id, 0.8);
        }
        assert_eq!(t.choose(&ids, 6), 2);
    }

    #[test]
    fn censored_outcomes_estimate_the_per_draft_rate() {
        let mut t = tuner(&[2, 4]);
        for _ in 0..400 {
            t.record_verification(
                Instant::now(),
                &[DraftOutcome {
                    seq_id: 7,
                    proposed: 4,
                    accepted: 2,
                }],
            );
        }
        // two hits, then one miss every step: the curve drops at the third position, and the sequence matches it
        let rates = t.position_rates(4);
        assert!(rates[0] > 0.95 && rates[1] > 0.95 && rates[2] < 0.1);
        assert!((t.sequences[&7].scale() - 1.0).abs() < 0.05);
    }

    #[test]
    fn small_gains_do_not_flip_the_depth() {
        let mut t = tuner(&[2, 3]);
        t.set_costs(1, &[(2, 1.0), (3, 1.0)]);
        t.set_rate(0, 0.9);
        assert_eq!(t.choose(&[0], 3), 3);
        // depth 2 now scores 1% better than 3: not enough to switch
        t.set_costs(
            1,
            &[
                (2, 1.0),
                (
                    3,
                    1.0 * expected_tokens(&[0.9; 3]) / expected_tokens(&[0.9; 2]) * 1.01,
                ),
            ],
        );
        assert_eq!(t.choose(&[0], 3), 3);
    }

    #[test]
    fn unmeasured_start_uses_the_middle_candidate_within_the_cap() {
        let mut t = tuner(&[2, 3, 4, 6]);
        assert_eq!(t.choose(&[0], 6), 4);
        let mut t = tuner(&[2, 3, 4, 6]);
        assert_eq!(t.choose(&[0], 3), 3);
    }

    #[test]
    fn cohort_hint_preserves_fixed_depth_and_candidate_limits() {
        let mut fixed = tuner(&[]);
        assert_eq!(fixed.choose_with_hint(&[0], 6, Some(2)), 6);
        let mut adaptive = tuner(&[2, 3, 4, 6]);
        assert_eq!(adaptive.choose_with_hint(&[0], 6, Some(2)), 2);
        assert_eq!(adaptive.choose_with_hint(&[0], 6, Some(5)), 4);
        assert_eq!(adaptive.choose_with_hint(&[0], 2, Some(6)), 2);
    }

    #[test]
    fn unmeasured_depth_respects_a_lowered_cap() {
        let mut t = tuner(&[2, 3, 4, 6]);
        assert_eq!(t.choose(&[0], 6), 4);
        assert_eq!(t.choose(&[0], 3), 3);
        assert_eq!(t.choose(&[0], 2), 2);
    }

    #[test]
    fn interrupted_steps_update_acceptance_without_recording_cost() {
        let mut t = tuner(&[2, 4]);
        let started = Instant::now();
        t.begin_step(&[7], 4, started);
        t.boundary.as_mut().unwrap().interruptions = interruptions().wrapping_sub(1);
        t.record_verification(
            started + Duration::from_secs(1),
            &[DraftOutcome {
                seq_id: 7,
                proposed: 4,
                accepted: 2,
            }],
        );
        assert!(t.step_cost.is_empty());
        assert_eq!(t.positions.len(), 3);
        assert_eq!(t.positions[2].hits, 0.0);
        assert_eq!(t.positions[2].trials, 1.0);
    }

    #[test]
    fn timing_rejects_interleaved_batches_with_the_same_shape() {
        let mut t = tuner(&[2, 4]);
        let started = Instant::now();
        let outcome = DraftOutcome {
            seq_id: 2,
            proposed: 4,
            accepted: 2,
        };
        t.begin_step(&[1], 4, started);
        t.record_verification(started + Duration::from_millis(20), &[outcome]);
        assert!(t.step_cost.is_empty());
        t.begin_step(&[2], 4, started);
        t.record_verification(started + Duration::from_millis(20), &[outcome]);
        assert!(t.step_cost.contains_key(&(1, 4)));
    }

    #[test]
    fn canceled_proposals_do_not_contribute_timing_samples() {
        let mut t = tuner(&[2, 4]);
        let started = Instant::now();
        t.begin_step(&[7], 4, started);
        t.cancel_step();
        t.record_verification(
            started + Duration::from_millis(20),
            &[DraftOutcome {
                seq_id: 7,
                proposed: 4,
                accepted: 2,
            }],
        );
        assert!(t.step_cost.is_empty());
        assert_eq!(t.positions.len(), 3);
    }

    #[test]
    fn step_timing_ignores_mismatched_batches_and_outliers() {
        let mut t = tuner(&[2, 4]);
        let outcome = |seq_id| DraftOutcome {
            seq_id,
            proposed: 4,
            accepted: 4,
        };
        t.begin_step(&[0], 4, Instant::now());
        t.record_verification(
            t.boundary.as_ref().unwrap().at + Duration::from_millis(20),
            &[outcome(0)],
        );
        let cost = t.step_cost[&(1, 4)].secs;
        assert!((cost - 0.020).abs() < 1e-6);
        t.begin_step(&[0], 4, Instant::now());
        t.record_verification(
            t.boundary.as_ref().unwrap().at + Duration::from_secs(2),
            &[outcome(0)],
        );
        assert!((t.step_cost[&(1, 4)].secs - cost).abs() < 1e-9);
        t.begin_step(&[0], 4, Instant::now());
        t.record_verification(
            t.boundary.as_ref().unwrap().at + Duration::from_millis(40),
            &[outcome(0), outcome(1)],
        );
        assert!((t.step_cost[&(1, 4)].secs - cost).abs() < 1e-9);
    }

    #[test]
    fn a_persistent_shift_replaces_the_cost_estimate() {
        let mut t = tuner(&[2, 4]);
        t.set_costs(1, &[(4, 0.001)]);
        for _ in 0..COST_OUTLIER_STREAK {
            record_cost(&mut t.step_cost, (1, 4), 0.05, 0);
        }
        assert!((t.step_cost[&(1, 4)].secs - 0.05).abs() < 1e-9);
    }

    #[test]
    fn a_cold_first_sample_is_corrected_by_the_next_fast_one() {
        let mut t = tuner(&[2, 4]);
        record_cost(&mut t.step_cost, (1, 2), 0.300, 0);
        record_cost(&mut t.step_cost, (1, 2), 0.020, 1);
        record_cost(&mut t.step_cost, (1, 2), 0.020, 2);
        assert!(t.step_cost[&(1, 2)].secs < 0.1);
    }

    #[test]
    fn steady_state_probes_only_neighboring_depths() {
        let mut t = tuner(&[2, 3, 4, 6]);
        t.set_costs(1, &[(2, 1.0), (3, 1.0), (4, 1.0), (6, 1.0)]);
        t.step_cost.get_mut(&(1, 4)).unwrap().updated = 9;
        t.set_rate(0, 0.9);
        let decisions = (0..PROBE_INTERVAL)
            .map(|_| t.choose(&[0], 6))
            .collect::<Vec<_>>();
        assert_eq!(decisions[0], 6);
        assert!(decisions[..decisions.len() - 1]
            .iter()
            .all(|depth| *depth == 6));
        // depth 2 is the stalest overall, but only 4 sits next to 6
        assert_eq!(decisions[decisions.len() - 1], 4);
    }

    #[test]
    fn warmup_finds_a_faster_depth_across_a_slower_neighbor() {
        let mut t = tuner(&AUTO_DEPTHS);
        let ids = (0..8).collect::<Vec<_>>();
        t.set_positions(&[0.9; AUTO_MAX_DEPTH]);
        t.set_costs(8, &[(4, 0.20), (6, 0.21)]);
        assert_eq!(t.choose(&ids, AUTO_MAX_DEPTH), 6);
        let costs = [(2, 0.13), (3, 0.10), (4, 0.20), (6, 0.21)];
        let bound =
            WARMUP_PROBE_INTERVAL * u64::from(WARMUP_SAMPLES + 1) * AUTO_DEPTHS.len() as u64;
        for _ in 0..bound {
            let depth = t.choose(&ids, AUTO_MAX_DEPTH);
            let secs = costs
                .iter()
                .find(|(candidate, _)| *candidate == depth)
                .unwrap()
                .1;
            record_cost(&mut t.step_cost, (8, depth), secs, t.clock);
        }
        assert!(AUTO_DEPTHS.iter().all(|depth| {
            t.step_cost
                .get(&(8, *depth))
                .is_some_and(|cost| cost.samples >= WARMUP_SAMPLES)
        }));
        assert_eq!(t.current[&8], 3);
    }

    #[test]
    fn warmup_probes_other_depths_before_the_preferred_cost_finishes_warming() {
        let mut t = tuner(&AUTO_DEPTHS);
        let ids = (0..8).collect::<Vec<_>>();
        t.set_positions(&[0.9; AUTO_MAX_DEPTH]);
        t.set_costs(8, &[(4, 0.20), (6, 0.21)]);
        t.step_cost.get_mut(&(8, 6)).unwrap().samples = WARMUP_SAMPLES - 1;
        let decisions = (0..WARMUP_PROBE_INTERVAL)
            .map(|_| t.choose(&ids, AUTO_MAX_DEPTH))
            .collect::<Vec<_>>();
        assert!(decisions[..decisions.len() - 1]
            .iter()
            .all(|depth| *depth == 6));
        assert_eq!(decisions[decisions.len() - 1], 3);
    }

    #[test]
    fn large_batches_start_shallow_from_small_batch_costs() {
        let mut t = tuner(&[2, 3, 4, 6]);
        t.set_costs(1, &[(2, 1.0), (3, 1.05), (4, 1.1), (6, 1.2)]);
        let ids = (0..8).collect::<Vec<_>>();
        for id in &ids {
            t.set_rate(*id, 0.8);
        }
        assert_eq!(t.choose(&[0], 6), 6);
        assert_eq!(t.choose(&ids, 6), 2);
    }

    #[test]
    fn unmeasured_depths_transfer_from_other_batch_sizes() {
        let mut t = tuner(&[3, 6]);
        t.set_costs(4, &[(3, 1.0), (6, 2.0)]);
        t.set_costs(8, &[(3, 2.0)]);
        assert!((t.step_seconds(8, 6).unwrap() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn nearby_batch_sizes_keep_separate_costs() {
        let mut t = tuner(&[3, 6]);
        let outcomes = |n: usize| {
            (0..n)
                .map(|seq_id| DraftOutcome {
                    seq_id,
                    proposed: 6,
                    accepted: 6,
                })
                .collect::<Vec<_>>()
        };
        t.begin_step(&[0, 1, 2, 3, 4], 6, Instant::now());
        t.record_verification(
            t.boundary.as_ref().unwrap().at + Duration::from_millis(10),
            &outcomes(5),
        );
        assert!(t.step_cost.contains_key(&(5, 6)));
        assert!(!t.step_cost.contains_key(&(8, 6)));
        assert_eq!(batch_key(12), 16);
    }

    #[test]
    fn warmup_probes_depths_without_clean_samples() {
        let mut t = tuner(&[2, 4]);
        record_cost(&mut t.step_cost, (1, 4), 0.02, 0);
        t.set_rate(0, 0.9);
        let decisions = (0..WARMUP_PROBE_INTERVAL)
            .map(|_| t.choose(&[0], 4))
            .collect::<Vec<_>>();
        assert!(decisions.contains(&2));
    }

    #[test]
    fn falling_acceptance_by_position_prefers_shallower_drafts() {
        let costs = [(2, 1.2), (3, 1.3), (4, 1.4), (6, 1.6)];
        let ids = (0..8).collect::<Vec<_>>();
        let mut flat = tuner(&[2, 3, 4, 6]);
        flat.set_costs(8, &costs);
        flat.set_positions(&[0.9; 6]);
        assert_eq!(flat.choose(&ids, 6), 6);
        let mut falling = tuner(&[2, 3, 4, 6]);
        falling.set_costs(8, &costs);
        falling.set_positions(&[0.95, 0.9, 0.7, 0.5, 0.35, 0.25]);
        assert_eq!(falling.choose(&ids, 6), 4);
    }

    #[test]
    fn calibration_discounts_depths_that_underdeliver() {
        let mut t = tuner(&[3, 6]);
        t.set_positions(&[0.9; 6]);
        for _ in 0..200 {
            t.record_verification(
                Instant::now(),
                &[DraftOutcome {
                    seq_id: 1,
                    proposed: 6,
                    accepted: 1,
                }],
            );
            t.set_positions(&[0.9; 6]);
            t.sequences.clear();
        }
        assert!(t.calibration(6) < 0.6);
        assert!((t.calibration(3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_single_candidate_is_inactive() {
        assert!(!tuner(&[3]).is_active());
        assert!(tuner(&[2, 3]).is_active());
    }
}
