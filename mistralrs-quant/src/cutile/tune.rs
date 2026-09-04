//! Startup autotuner for cuTile kernels, built on `cutile::tune`.
//!
//! A kernel plugs in with three things: the token buckets its launch policy distinguishes, a
//! declarative [`Space`] per bucket (joint parameter groups searched as a grid, independent axes
//! descended from the winner, constraints, and the policy config), and a launcher that turns a
//! cuTile [`Config`] into a timed launch set. Search bookkeeping, measurement (CUDA events,
//! medians, a paired runoff between the two best) and persistence (provenance-checked records)
//! come from cuTile; this module adds the buckets, the coordinate-descent searcher, a correctness
//! gate against the policy config, and the table kernels read at launch.

use candle_core::{CudaDevice, DType, Result, Tensor};
use cutile::bench::BenchOptions;
use cutile::cuda_core::Stream;
use cutile::error::Error as CutileError;
use cutile::tune::{
    space_hash, Autotuner, Config, Oracle, ParamValue, Record, RecordEntry, Searcher, Trial,
    TrialState, Workspace,
};
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

/// `off` uses the static policies, `auto` (default) tunes when no valid record exists, `force` retunes.
pub const TUNE_MODE_ENV: &str = "MISTRALRS_CUTILE_TUNE";
/// Overrides the directory holding the tuning records.
pub const TUNE_CACHE_ENV: &str = "MISTRALRS_CUTILE_TUNE_CACHE";
const RECORD_DIR: &str = "cutile_tune";
/// A candidate replaces the incumbent only when it is at least this much faster.
const MIN_GAIN: f64 = 0.03;
/// A candidate's output must stay within this relative distance of the policy config's output.
const GATE_TOLERANCE: f32 = 2e-2;
/// Wall-clock cap per bucket; the descent keeps its incumbent when the budget runs out, so a
/// model with many shapes still loads in bounded time.
const BUCKET_BUDGET: Duration = Duration::from_secs(20);
/// Launch sets enqueued per timed rep. One set per rep leaves the GPU idle between reps and
/// charges launch latency and allocation to the kernel, which inflated a 5 ms kernel to 12 and
/// compressed the spread between configs; a burst overlaps host work the way a forward pass does.
const LAUNCHES_PER_REP: usize = 8;
/// Weight sets a kernel keeps per shape so timed launches rotate through distinct layers and read
/// weights cold from HBM instead of L2.
pub(super) const TUNE_WEIGHT_SETS: usize = 8;
/// Below this many tokens a probe uses `TuneRouting::Hot`.
pub(super) const HOT_ROUTING_TOKENS: usize = 96;

/// Routing used to time a MoE config: `Hot` sends every token to the same top_k experts, the
/// shape of a batch of similar prompts where tile choice matters for decode; `Spread` scatters
/// tokens over all experts with mild imbalance, like a long prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TuneRouting {
    Hot,
    Spread,
}

impl TuneRouting {
    pub(super) fn for_tokens(m: usize) -> Self {
        if m <= HOT_ROUTING_TOKENS {
            Self::Hot
        } else {
            Self::Spread
        }
    }

    /// Expert ids for `m * top_k` routing slots.
    pub(super) fn expert_ids(self, m: usize, top_k: usize, num_experts: usize) -> Vec<u32> {
        (0..m * top_k)
            .map(|i| match self {
                Self::Hot => (i % top_k) as u32,
                Self::Spread => (((i * 2654435761usize) >> 7) % num_experts) as u32,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuneMode {
    Off,
    Auto,
    Force,
}

impl TuneMode {
    pub fn from_env() -> Self {
        std::env::var(TUNE_MODE_ENV)
            .ok()
            .and_then(|value| Self::parse(&value))
            .unwrap_or(Self::Auto)
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" => Some(Self::Off),
            "auto" | "on" | "1" | "true" | "yes" => Some(Self::Auto),
            "force" => Some(Self::Force),
            _ => None,
        }
    }
}

/// One token-count interval the launch policy keeps a single config for; `upper` is inclusive and
/// `probe` is the token count timed on behalf of the whole interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bucket {
    pub upper: usize,
    pub probe: usize,
}

/// Buckets between consecutive breakpoints (inclusive uppers), each probed at its upper end
/// capped at `probe_cap`, plus an unbounded tail probed at `tail_probe`.
pub fn buckets_from_breakpoints(
    breakpoints: &[usize],
    probe_cap: usize,
    tail_probe: usize,
) -> Vec<Bucket> {
    let mut sorted = breakpoints.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut buckets: Vec<Bucket> = sorted
        .iter()
        .map(|&upper| Bucket {
            upper,
            probe: upper.min(probe_cap),
        })
        .collect();
    let lo = sorted.last().map_or(1, |upper| upper + 1);
    buckets.push(Bucket {
        upper: usize::MAX,
        probe: tail_probe.max(lo),
    });
    buckets
}

/// Human-readable interval labels, `<=96`, `<=512`, `>512`, one per bucket in order.
fn bucket_labels(buckets: &[Bucket]) -> Vec<String> {
    let mut lower = 1usize;
    buckets
        .iter()
        .map(|bucket| {
            let label = if bucket.upper == usize::MAX {
                if lower == 1 {
                    "all".to_string()
                } else {
                    format!(">{}", lower - 1)
                }
            } else {
                format!("<={}", bucket.upper)
            };
            lower = bucket.upper.saturating_add(1);
            label
        })
        .collect()
}

/// Builds a cuTile config from integer parameters.
pub fn config(params: impl IntoIterator<Item = (&'static str, i64)>) -> Config {
    Config::new(
        params
            .into_iter()
            .map(|(key, value)| (key, ParamValue::Int(value))),
    )
}

/// The search space for one bucket, declared by the kernel.
///
/// Joint groups are parameters that only make sense together (tile shapes) and are searched as a
/// grid with every axis at its policy value. Axes are independent knobs, descended one at a time
/// from the grid winner. Constraints prune combinations the kernel cannot launch. The policy is
/// the static config every candidate must beat by a margin.
type Constraint = Box<dyn Fn(&Config) -> bool>;

#[derive(Default)]
pub struct Space {
    joints: Vec<(Vec<&'static str>, Vec<Vec<i64>>)>,
    axes: Vec<(&'static str, Vec<i64>)>,
    constraints: Vec<Constraint>,
    policy: Option<Config>,
}

impl Space {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parameters searched together, one row per candidate.
    pub fn joint<const N: usize>(
        mut self,
        names: [&'static str; N],
        rows: impl IntoIterator<Item = [i64; N]>,
    ) -> Self {
        self.joints.push((
            names.to_vec(),
            rows.into_iter().map(|r| r.to_vec()).collect(),
        ));
        self
    }

    /// An independent knob and the values to try for it.
    pub fn axis(mut self, name: &'static str, values: impl IntoIterator<Item = i64>) -> Self {
        self.axes.push((name, values.into_iter().collect()));
        self
    }

    /// Drops candidates the predicate rejects; applies to the grid and to every axis variant.
    pub fn constrain(mut self, keep: impl Fn(&Config) -> bool + 'static) -> Self {
        self.constraints.push(Box::new(keep));
        self
    }

    /// The static config the launch policy uses today; always a candidate and never pruned.
    pub fn policy(mut self, policy: Config) -> Self {
        self.policy = Some(policy);
        self
    }

    fn keeps(&self, candidate: &Config) -> bool {
        self.policy.as_ref().is_some_and(|p| p.id == candidate.id)
            || self.constraints.iter().all(|keep| keep(candidate))
    }

    /// Every axis at the policy's value, or its first listed value when the policy lacks it.
    fn axis_defaults(&self) -> BTreeMap<&'static str, i64> {
        self.axes
            .iter()
            .map(|(name, values)| {
                let default = self
                    .policy
                    .as_ref()
                    .and_then(|p| p.int(name))
                    .unwrap_or(values[0]);
                (*name, default)
            })
            .collect()
    }

    /// The grid phase: the policy first, then every joint row with the axes at their defaults.
    fn tiles(&self) -> Vec<Config> {
        let defaults = self.axis_defaults();
        let mut rows: Vec<BTreeMap<&'static str, i64>> = vec![BTreeMap::new()];
        for (names, group) in &self.joints {
            let mut grown = Vec::new();
            for row in &rows {
                for values in group {
                    let mut next = row.clone();
                    next.extend(names.iter().copied().zip(values.iter().copied()));
                    grown.push(next);
                }
            }
            rows = grown;
        }
        let mut tiles: Vec<Config> = self.policy.iter().cloned().collect();
        for mut row in rows {
            row.extend(defaults.iter().map(|(k, v)| (*k, *v)));
            let candidate = config(row);
            if self.keeps(&candidate) && !tiles.iter().any(|t| t.id == candidate.id) {
                tiles.push(candidate);
            }
        }
        tiles
    }

    /// `base` with one axis swept over its values.
    fn variants(&self, base: &Config, axis: usize) -> Vec<Config> {
        let (name, values) = &self.axes[axis];
        values
            .iter()
            .map(|&value| {
                let mut params: BTreeMap<String, ParamValue> = base.params.clone();
                params.insert((*name).to_string(), ParamValue::Int(value));
                Config::new(params)
            })
            .filter(|candidate| self.keeps(candidate))
            .collect()
    }

    /// Every config the descent can reach, for cuTile's declared candidate list.
    fn configs(&self) -> Vec<Config> {
        let mut configs = self.tiles();
        for axis in 0..self.axes.len() {
            let mut grown = configs.clone();
            for base in &configs {
                for variant in self.variants(base, axis) {
                    if !grown.iter().any(|c| c.id == variant.id) {
                        grown.push(variant);
                    }
                }
            }
            configs = grown;
        }
        configs
    }
}

/// Enqueues one launch set on the stream it is given.
pub type LaunchSet = Box<dyn FnMut(&Arc<Stream>) -> std::result::Result<(), CutileError>>;

/// One candidate ready to time: `run` enqueues one launch set and `sample` is that launch set's
/// output, compared against the policy config's by the gate.
pub struct Prepared {
    pub run: LaunchSet,
    pub sample: Tensor,
}

/// What a kernel asks the tuner to decide for one registered shape.
pub struct TuneRequest<'a> {
    pub kernel: &'static str,
    /// The kernel module's `_SOURCE_HASH`; a record tuned for other kernel source is refused.
    pub source_hash: &'static str,
    pub shape: String,
    pub buckets: &'a [Bucket],
    pub space: &'a dyn Fn(Bucket) -> Space,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Record,
    Measured,
    Fallback,
}

#[derive(Clone, Debug)]
pub struct Tuned {
    pub bucket: Bucket,
    pub config: Config,
    pub source: Source,
    pub ms: f64,
    pub policy_ms: f64,
}

/// Decides one config per bucket. `prepare(m, config)` builds a candidate's launch set at `m`
/// tokens; a candidate whose preparation fails or whose output disagrees with the policy config's
/// is recorded as invalid and skipped.
pub fn tune(
    dev: &CudaDevice,
    mode: TuneMode,
    request: &TuneRequest,
    mut prepare: impl FnMut(usize, &Config) -> Result<Prepared>,
) -> Vec<Tuned> {
    let labels = bucket_labels(request.buckets);
    let spaces: Vec<Space> = request
        .buckets
        .iter()
        .map(|&bucket| (request.space)(bucket))
        .collect();
    let policy_of = |space: &Space| {
        space
            .policy
            .clone()
            .expect("a tuning space declares its policy config")
    };
    let fallback = |bucket: Bucket, space: &Space| Tuned {
        bucket,
        config: policy_of(space),
        source: Source::Fallback,
        ms: 0.0,
        policy_ms: 0.0,
    };
    if mode == TuneMode::Off {
        return request
            .buckets
            .iter()
            .zip(&spaces)
            .map(|(&b, s)| fallback(b, s))
            .collect();
    }
    let all: Vec<Config> = spaces.iter().flat_map(Space::configs).collect();
    let workspace = Workspace {
        kernel: format!("{}/{}", request.kernel, request.shape),
        source_hash: request.source_hash.to_string(),
        arch: arch(dev),
        tileiras_fingerprint:
            cutile::cutile_compiler::cuda_tile_runtime_utils::tileiras_fingerprint().to_string(),
        space_hash: Some(space_hash(&all)),
    };
    let path = record_path(dev, request);
    let mut record = if mode == TuneMode::Auto && path.exists() {
        match Record::load_verified(&path, &workspace, |_| Ok(None)) {
            Ok((record, warnings)) => {
                for warning in warnings {
                    tracing::debug!("cuTile autotune {}: {warning}", workspace.kernel);
                }
                record
            }
            Err(err) => {
                tracing::info!("cuTile autotune {}: {err}", workspace.kernel);
                Record::new(&workspace)
            }
        }
    } else {
        Record::new(&workspace)
    };

    let stream = super::context::stream(dev);
    let mut results = Vec::with_capacity(request.buckets.len());
    let mut measured_any = false;
    for ((&bucket, label), space) in request.buckets.iter().zip(&labels).zip(&spaces) {
        let policy = policy_of(space);
        let candidates = space.configs();
        if let Some(entry) = record.get(label).filter(|entry| {
            candidates
                .iter()
                .any(|candidate| candidate == &entry.config)
        }) {
            results.push(Tuned {
                bucket,
                config: entry.config.clone(),
                source: Source::Record,
                ms: f64::from(entry.median_ms),
                policy_ms: 0.0,
            });
            continue;
        }
        if record.get(label).is_some() {
            tracing::info!(
                "cuTile autotune {} {label}: cached config is outside the current search space",
                workspace.kernel
            );
        }
        let reference = match prepare(bucket.probe, &policy) {
            Ok(prepared) => prepared.sample,
            Err(err) => {
                tracing::warn!(
                    "cuTile autotune {} {label}: policy config failed to run, keeping it untimed: {err}",
                    workspace.kernel
                );
                results.push(fallback(bucket, space));
                continue;
            }
        };
        let output = Autotuner::new(&workspace.kernel)
            .configs(candidates)
            .bench(bench_options())
            .budget(BUCKET_BUDGET)
            .run_with(Descent { space }, &stream, |_, candidate| {
                let prepared = prepare(bucket.probe, candidate).map_err(cutile_error)?;
                if candidate.id != policy.id && !within_tolerance(&prepared.sample, &reference)? {
                    return Err(cutile::error::tensor_error(
                        "output disagrees with the policy config",
                    ));
                }
                Ok(burst(prepared.run))
            });
        let output = match output {
            Ok(output) => output,
            Err(err) => {
                tracing::warn!(
                    "cuTile autotune {} {label}: search failed, keeping the policy: {err}",
                    workspace.kernel
                );
                results.push(fallback(bucket, space));
                continue;
            }
        };
        let per_set =
            |id: &str| latest_median(&output.trials, id).map(|ms| ms / LAUNCHES_PER_REP as f64);
        let policy_ms = per_set(&policy.id).unwrap_or(0.0);
        let winner = output
            .best
            .as_ref()
            .and_then(|best| per_set(&best.id).map(|ms| (best.clone(), ms)));
        let (chosen, ms) = match winner {
            Some((best, ms)) if keeps_policy(ms, policy_ms) => {
                tracing::debug!(
                    "cuTile autotune {} {label}: {} within the margin of the policy, keeping the policy",
                    workspace.kernel,
                    best.id
                );
                (policy.clone(), policy_ms)
            }
            Some((best, ms)) => (best, ms),
            None => (policy.clone(), policy_ms),
        };
        record.insert(RecordEntry {
            bucket: label.clone(),
            median_ms: ms as f32,
            samples: samples(&output.trials, &chosen.id),
            config: chosen.clone(),
            l2_key: None,
        });
        measured_any = true;
        results.push(Tuned {
            bucket,
            config: chosen,
            source: Source::Measured,
            ms,
            policy_ms,
        });
    }
    if measured_any {
        if let Err(err) = save_record(&record, &path) {
            tracing::warn!("cuTile autotune: could not write {}: {err}", path.display());
        }
    }
    log_summary(request, &labels, &spaces, &results);
    results
}

/// Grid over the tiles, then one axis at a time from the winner; the incumbent stays unless a
/// variant beats it by `MIN_GAIN`. Every trial goes back to cuTile, whose paired runoff between
/// the two best medians picks the final winner.
struct Descent<'a> {
    space: &'a Space,
}

impl Searcher for Descent<'_> {
    fn search(&mut self, oracle: &mut dyn Oracle) -> Vec<Trial> {
        let configs: Vec<Config> = oracle.configs().to_vec();
        let index_of = |id: &str| configs.iter().position(|c| c.id == id);
        let mut trials = Vec::new();
        let mut visit =
            |candidates: Vec<Config>, trials: &mut Vec<Trial>| -> Option<(Config, f32)> {
                let mut best: Option<(Config, f32)> = None;
                for candidate in candidates {
                    let Some(index) = index_of(&candidate.id) else {
                        continue;
                    };
                    if oracle.budget_remaining() == Some(Duration::ZERO) {
                        break;
                    }
                    let trial = oracle.measure(index);
                    if let Some(ms) = trial.median_ms() {
                        if best.as_ref().is_none_or(|(_, b)| ms < *b) {
                            best = Some((configs[index].clone(), ms));
                        }
                    }
                    trials.push(trial);
                }
                best
            };
        let Some((mut incumbent, mut incumbent_ms)) = visit(self.space.tiles(), &mut trials) else {
            return trials;
        };
        for axis in 0..self.space.axes.len() {
            let variants = self
                .space
                .variants(&incumbent, axis)
                .into_iter()
                .filter(|c| c.id != incumbent.id)
                .collect();
            if let Some((candidate, ms)) = visit(variants, &mut trials) {
                if f64::from(ms) < f64::from(incumbent_ms) * (1.0 - MIN_GAIN) {
                    incumbent = candidate;
                    incumbent_ms = ms;
                }
            }
        }
        trials
    }
}

/// The most recent measurement of a config; after the runoff that is its paired timing, so the
/// winner and the policy are compared from the same moment.
fn latest_median(trials: &[Trial], id: &str) -> Option<f64> {
    trials
        .iter()
        .rev()
        .filter(|t| t.config_id == id)
        .find_map(Trial::median_ms)
        .map(f64::from)
}

/// A winner within `MIN_GAIN` of the policy is not worth a new kernel variant; the policy stays.
fn keeps_policy(winner_ms: f64, policy_ms: f64) -> bool {
    policy_ms > 0.0 && winner_ms >= policy_ms * (1.0 - MIN_GAIN)
}

fn samples(trials: &[Trial], id: &str) -> usize {
    trials
        .iter()
        .filter(|t| t.config_id == id)
        .filter_map(|t| match &t.state {
            TrialState::Measured { reps, .. } => Some(*reps),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Weights rotate through distinct layers, so cuTile's L2 flush and its buffer are not needed.
fn bench_options() -> BenchOptions {
    BenchOptions {
        clear_l2: false,
        ..BenchOptions::default()
    }
}

/// `LAUNCHES_PER_REP` launch sets per call.
fn burst(mut run: LaunchSet) -> LaunchSet {
    Box::new(move |stream| {
        for _ in 0..LAUNCHES_PER_REP {
            run(stream)?;
        }
        Ok(())
    })
}

/// Median milliseconds of one prepared launch set, timed as the tuner times it, for the sweep tests.
#[cfg(test)]
pub(super) fn bench_ms(dev: &CudaDevice, prepared: Prepared) -> Result<f64> {
    let stream = super::context::stream(dev);
    let mut run = burst(prepared.run);
    let measurement = cutile::bench::do_bench(&stream, &bench_options(), |s| run(s))
        .map_err(|e| candle_core::Error::Msg(format!("cutile bench: {e}")))?;
    Ok(f64::from(measurement.median_ms()) / LAUNCHES_PER_REP as f64)
}

pub(super) fn cutile_error(err: candle_core::Error) -> CutileError {
    cutile::error::tensor_error(&err.to_string())
}

fn within_tolerance(sample: &Tensor, reference: &Tensor) -> std::result::Result<bool, CutileError> {
    let distance = |a: &Tensor, b: &Tensor| -> Result<(f32, f32)> {
        let a = a.to_dtype(DType::F32)?;
        let b = b.to_dtype(DType::F32)?;
        let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = b.abs()?.max_all()?.to_scalar::<f32>()?;
        Ok((diff, scale))
    };
    let (diff, scale) = distance(sample, reference).map_err(cutile_error)?;
    Ok(diff <= GATE_TOLERANCE * scale.max(1e-3))
}

fn log_summary(request: &TuneRequest, labels: &[String], spaces: &[Space], results: &[Tuned]) {
    if results.iter().all(|r| r.source == Source::Fallback) {
        return;
    }
    let parts: Vec<String> = results
        .iter()
        .zip(labels)
        .zip(spaces)
        .map(|((r, label), space)| {
            let source = match r.source {
                Source::Record => "recorded",
                Source::Measured => "measured",
                Source::Fallback => "policy",
            };
            let is_policy = space.policy.as_ref().is_some_and(|p| p.id == r.config.id);
            if r.policy_ms > 0.0 && !is_policy {
                format!(
                    "{label} {} {:.3}ms vs policy {:.3}ms ({source})",
                    r.config.id, r.ms, r.policy_ms
                )
            } else {
                format!("{label} {} ({source})", r.config.id)
            }
        })
        .collect();
    tracing::info!(
        "cuTile autotune {} {}: {}",
        request.kernel,
        request.shape,
        parts.join(", ")
    );
}

/// Configs by inclusive token-count upper bound, in ascending order.
type BucketConfigs<C> = Vec<(usize, C)>;

/// Tuned configs a kernel consults at launch time, keyed by shape.
pub struct TunedTable<K, C> {
    inner: OnceLock<RwLock<HashMap<K, BucketConfigs<C>>>>,
}

impl<K: Hash + Eq + Copy, C: Copy> TunedTable<K, C> {
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    /// Stores the tuned configs converted to the kernel's launch type; a config the kernel cannot
    /// read is left out, so that bucket falls back to the policy at launch.
    pub fn set(&self, key: K, tuned: &[Tuned], convert: impl Fn(&Config) -> Option<C>) {
        let configs = tuned
            .iter()
            .filter_map(|t| convert(&t.config).map(|c| (t.bucket.upper, c)))
            .collect();
        self.inner
            .get_or_init(|| RwLock::new(HashMap::new()))
            .write()
            .unwrap()
            .insert(key, configs);
    }

    pub fn get(&self, key: K, m: usize) -> Option<C> {
        let table = self.inner.get()?.read().unwrap();
        let configs = table.get(&key)?;
        configs
            .iter()
            .find(|(upper, _)| m <= *upper)
            .map(|(_, config)| *config)
    }
}

impl<K: Hash + Eq + Copy, C: Copy> Default for TunedTable<K, C> {
    fn default() -> Self {
        Self::new()
    }
}

fn arch(dev: &CudaDevice) -> String {
    let (major, minor) = super::device_compute_capability(dev);
    format!("sm_{major}{minor}")
}

/// GPU name and SM count as a file-name-safe slug; records are per device model, not just per arch.
fn device_slug(dev: &CudaDevice) -> String {
    use candle_core::cuda::cudarc::driver::result;
    let cu_device = dev.cuda_stream().context().cu_device();
    let name = result::device::get_name(cu_device).unwrap_or_else(|_| "unknown-gpu".to_string());
    let sms = super::device_multiprocessor_count(dev);
    let mut slug: String = name
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    format!("{}-{sms}sm", slug.trim_matches('-'))
}

fn record_dir() -> PathBuf {
    if let Some(path) = std::env::var_os(TUNE_CACHE_ENV) {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("mistralrs").join(RECORD_DIR);
    }
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path)
            .join(".cache")
            .join("mistralrs")
            .join(RECORD_DIR);
    }
    std::env::temp_dir().join("mistralrs").join(RECORD_DIR)
}

fn record_path(dev: &CudaDevice, request: &TuneRequest) -> PathBuf {
    record_dir().join(format!(
        "{}.{}.{}.json",
        request.kernel,
        request.shape,
        device_slug(dev)
    ))
}

fn save_record(record: &Record, path: &PathBuf) -> std::result::Result<(), CutileError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| cutile::error::tensor_error(&format!("record dir: {e}")))?;
    }
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    record.save(&tmp)?;
    std::fs::rename(&tmp, path)
        .map_err(|e| cutile::error::tensor_error(&format!("record rename: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Costs by config: tile 2 is fastest, knob 1 improves it, knob 2 regresses, tile 3 fails.
    fn cost(c: &Config) -> Option<f32> {
        let tile = match c.int("tile")? {
            1 => 1.0,
            2 => 0.8,
            _ => return None,
        };
        Some(
            tile * match c.int("knob")? {
                1 => 0.9,
                2 => 1.5,
                _ => 1.0,
            },
        )
    }

    struct FakeOracle {
        configs: Vec<Config>,
        visited: Vec<String>,
    }

    impl Oracle for FakeOracle {
        fn configs(&self) -> &[Config] {
            &self.configs
        }

        // cuTile's trial types are non-exhaustive, so a fake oracle builds them through serde
        fn measure(&mut self, index: usize) -> Trial {
            let config = &self.configs[index];
            self.visited.push(config.id.clone());
            let state = match cost(config) {
                Some(ms) => {
                    serde_json::json!({"Measured": {"median_ms": ms, "min_ms": ms, "reps": 5}})
                }
                None => serde_json::json!({"Invalid": {"reason": "unsupported"}}),
            };
            serde_json::from_value(serde_json::json!({"config_id": config.id, "state": state}))
                .unwrap()
        }

        fn budget_remaining(&self) -> Option<Duration> {
            None
        }
    }

    fn space() -> Space {
        Space::new()
            .joint(["tile"], [[1], [2], [3]])
            .axis("knob", [0, 1, 2])
            .constrain(|c| !(c.int("tile") == Some(3) && c.int("knob") == Some(2)))
            .policy(config([("tile", 1), ("knob", 0)]))
    }

    #[test]
    fn mode_parses_documented_values() {
        assert_eq!(TuneMode::parse("off"), Some(TuneMode::Off));
        assert_eq!(TuneMode::parse(" Auto "), Some(TuneMode::Auto));
        assert_eq!(TuneMode::parse("FORCE"), Some(TuneMode::Force));
        assert_eq!(TuneMode::parse("sometimes"), None);
    }

    #[test]
    fn buckets_cover_every_token_count_once() {
        let buckets = buckets_from_breakpoints(&[512, 96, 96, 16511], 4096, 4096);
        assert_eq!(
            buckets,
            vec![
                Bucket {
                    upper: 96,
                    probe: 96
                },
                Bucket {
                    upper: 512,
                    probe: 512
                },
                Bucket {
                    upper: 16511,
                    probe: 4096
                },
                Bucket {
                    upper: usize::MAX,
                    probe: 16512
                },
            ]
        );
        // the tail probe never falls inside a bounded bucket
        let tight = buckets_from_breakpoints(&[8192], 4096, 4096);
        assert_eq!(tight[1].probe, 8193);
    }

    #[test]
    fn labels_name_each_interval() {
        let buckets = buckets_from_breakpoints(&[96, 512], 4096, 4096);
        assert_eq!(bucket_labels(&buckets), vec!["<=96", "<=512", ">512"]);
        let single = buckets_from_breakpoints(&[], 4096, 4096);
        assert_eq!(bucket_labels(&single), vec!["all"]);
    }

    #[test]
    fn routing_shapes_match_their_regimes() {
        assert_eq!(TuneRouting::for_tokens(96), TuneRouting::Hot);
        assert_eq!(TuneRouting::for_tokens(97), TuneRouting::Spread);
        let hot = TuneRouting::Hot.expert_ids(4, 2, 128);
        assert_eq!(hot, vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let spread = TuneRouting::Spread.expert_ids(64, 8, 128);
        assert_eq!(spread.len(), 512);
        assert!(spread.iter().all(|&e| e < 128));
        assert!(
            spread
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 32
        );
    }

    #[test]
    fn space_grid_holds_the_policy_first_with_axes_at_policy_values() {
        let ids: Vec<String> = space().tiles().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, vec!["knob=0,tile=1", "knob=0,tile=2", "knob=0,tile=3"]);
        // a policy outside the joint rows is still the first candidate
        let odd = space().policy(config([("tile", 9), ("knob", 1)]));
        let ids: Vec<String> = odd.tiles().into_iter().map(|c| c.id).collect();
        assert_eq!(
            ids,
            vec![
                "knob=1,tile=9",
                "knob=1,tile=1",
                "knob=1,tile=2",
                "knob=1,tile=3"
            ]
        );
    }

    #[test]
    fn space_variants_and_constraints_prune_but_never_the_policy() {
        let s = space();
        let base = config([("tile", 3), ("knob", 0)]);
        let ids: Vec<String> = s.variants(&base, 0).into_iter().map(|c| c.id).collect();
        assert_eq!(ids, vec!["knob=0,tile=3", "knob=1,tile=3"]);
        assert_eq!(s.configs().len(), 8);
        let policy_constrained = space().constrain(|_| false);
        let ids: Vec<String> = policy_constrained
            .configs()
            .into_iter()
            .map(|c| c.id)
            .collect();
        assert_eq!(ids, vec!["knob=0,tile=1"]);
    }

    #[test]
    fn descent_visits_tiles_then_each_axis_from_the_winner() {
        let s = space();
        let mut oracle = FakeOracle {
            configs: s.configs(),
            visited: Vec::new(),
        };
        let trials = Descent { space: &s }.search(&mut oracle);
        // three tiles, then the two knob variants of tile 2; the incumbent knob is not re-timed
        assert_eq!(
            oracle.visited,
            vec![
                "knob=0,tile=1",
                "knob=0,tile=2",
                "knob=0,tile=3",
                "knob=1,tile=2",
                "knob=2,tile=2"
            ]
        );
        assert_eq!(trials.len(), 5);
        let best = cutile::tune::best_config(oracle.configs(), &trials).unwrap();
        assert_eq!(best.id, "knob=1,tile=2");
        assert_eq!(latest_median(&trials, "knob=0,tile=1"), Some(1.0));
        assert_eq!(samples(&trials, "knob=1,tile=2"), 5);
        assert_eq!(latest_median(&trials, "knob=0,tile=3"), None);
    }

    #[test]
    fn near_ties_keep_the_policy() {
        assert!(keeps_policy(0.98, 1.0));
        assert!(!keeps_policy(0.96, 1.0));
        assert!(!keeps_policy(0.5, 0.0));
    }

    #[test]
    fn table_picks_the_first_bucket_holding_m() {
        let table: TunedTable<u8, i64> = TunedTable::new();
        assert_eq!(table.get(0, 5), None);
        let tuned = [
            Tuned {
                bucket: Bucket {
                    upper: 96,
                    probe: 32,
                },
                config: config([("v", 1)]),
                source: Source::Fallback,
                ms: 0.0,
                policy_ms: 0.0,
            },
            Tuned {
                bucket: Bucket {
                    upper: usize::MAX,
                    probe: 4096,
                },
                config: config([("v", 2)]),
                source: Source::Fallback,
                ms: 0.0,
                policy_ms: 0.0,
            },
        ];
        table.set(0, &tuned, |c| c.int("v"));
        assert_eq!(table.get(0, 96), Some(1));
        assert_eq!(table.get(0, 97), Some(2));
        assert_eq!(table.get(1, 97), None);
    }

    #[test]
    fn gate_accepts_reordered_sums_and_rejects_garbage() {
        let dev = candle_core::Device::Cpu;
        let reference = Tensor::new(&[1.0f32, -2.0, 4.0], &dev).unwrap();
        let close = Tensor::new(&[1.01f32, -2.0, 4.02], &dev).unwrap();
        let wrong = Tensor::new(&[1.0f32, -2.0, 5.0], &dev).unwrap();
        assert!(within_tolerance(&close, &reference).unwrap());
        assert!(!within_tolerance(&wrong, &reference).unwrap());
    }
}
