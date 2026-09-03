//! Startup autotuner for cuTile kernels.
//!
//! A kernel describes, per registered shape, the token buckets its launch policy keeps one config
//! for and the candidate configs worth trying in each. During warmup, on the engine thread, the
//! tuner times the candidates against the static policy, keeps a winner only when it beats the
//! policy by a margin, and persists the answer in a JSON cache keyed by GPU, crate version, kernel
//! and shape, so the measurement is paid once per machine.

use candle_core::cuda::cudarc::driver::result;
use candle_core::{CudaDevice, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

/// `off` uses the static policies, `auto` (default) tunes on a cache miss, `force` retunes.
pub const TUNE_MODE_ENV: &str = "MISTRALRS_CUTILE_TUNE";
/// Overrides the cache file path.
pub const TUNE_CACHE_ENV: &str = "MISTRALRS_CUTILE_TUNE_CACHE";
const CACHE_FILE: &str = "cutile_tune.json";
const CACHE_FORMAT_VERSION: u32 = 1;
/// A candidate replaces the static policy only when it is at least this much faster.
const MIN_GAIN: f64 = 0.03;
/// Each config is timed this many times and the fastest run counts; noise only ever adds time.
const TIMING_REPEATS: usize = 3;
/// Timed launches per weight set; sets rotate so decode-sized probes read weights cold from HBM.
pub(super) const TUNE_ITERS: usize = 10;
/// Weight sets a kernel keeps per shape for that rotation; beyond this the L2 is cold anyway.
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

/// What a kernel asks the tuner to decide for one registered shape.
pub struct TuneRequest<'a, C> {
    /// Cache namespace; bump `version` whenever the kernel or its candidate lists change.
    pub kernel: &'static str,
    pub version: u32,
    pub shape: String,
    pub buckets: &'a [Bucket],
    pub fallback: &'a dyn Fn(usize) -> C,
    pub candidates: &'a dyn Fn(Bucket) -> Vec<C>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Cache,
    Measured,
    Fallback,
}

#[derive(Clone, Copy, Debug)]
pub struct Tuned<C> {
    pub bucket: Bucket,
    pub config: C,
    pub source: Source,
    pub ms: f64,
    pub fallback_ms: f64,
}

/// Decides one config per bucket. `time(m, config)` runs the kernel at `m` tokens with `config`
/// and returns milliseconds per launch set; a failing candidate is skipped, a failing fallback
/// keeps the static policy for that bucket.
pub fn tune<C>(
    dev: &CudaDevice,
    mode: TuneMode,
    request: &TuneRequest<C>,
    mut time: impl FnMut(usize, C) -> Result<f64>,
) -> Vec<Tuned<C>>
where
    C: Copy + PartialEq + Display + Serialize + DeserializeOwned,
{
    let device_key = device_key(dev);
    let labels = bucket_labels(request.buckets);
    let mut results = Vec::with_capacity(request.buckets.len());
    for (&bucket, label) in request.buckets.iter().zip(&labels) {
        let fallback = (request.fallback)(bucket.probe);
        if mode == TuneMode::Off {
            results.push(Tuned {
                bucket,
                config: fallback,
                source: Source::Fallback,
                ms: 0.0,
                fallback_ms: 0.0,
            });
            continue;
        }
        let candidates = (request.candidates)(bucket);
        let key = cache_key(&device_key, request, label);
        if mode == TuneMode::Auto {
            if let Some(entry) = cache().lock().unwrap().get::<C>(&key) {
                // a config outside today's candidate list means the list changed without a version bump
                if entry.config == fallback || candidates.contains(&entry.config) {
                    results.push(Tuned {
                        bucket,
                        config: entry.config,
                        source: Source::Cache,
                        ms: entry.ms,
                        fallback_ms: entry.fallback_ms,
                    });
                    continue;
                }
            }
        }
        let fallback_ms = match best_of(&mut time, bucket.probe, fallback) {
            Ok(ms) => ms,
            Err(err) => {
                tracing::warn!(
                    "cuTile autotune {} {} {label}: policy config failed to run, keeping it untimed: {err}",
                    request.kernel,
                    request.shape
                );
                results.push(Tuned {
                    bucket,
                    config: fallback,
                    source: Source::Fallback,
                    ms: 0.0,
                    fallback_ms: 0.0,
                });
                continue;
            }
        };
        let mut timings = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if candidate == fallback {
                continue;
            }
            match best_of(&mut time, bucket.probe, candidate) {
                Ok(ms) => timings.push((candidate, ms)),
                Err(err) => tracing::debug!(
                    "cuTile autotune {} {}: candidate {candidate} skipped: {err}",
                    request.kernel,
                    request.shape
                ),
            }
        }
        let (config, ms) = select(fallback, fallback_ms, &timings);
        cache().lock().unwrap().insert(
            key,
            Entry {
                config: serde_json::to_value(config).unwrap_or_default(),
                ms,
                fallback_ms,
            },
        );
        results.push(Tuned {
            bucket,
            config,
            source: Source::Measured,
            ms,
            fallback_ms,
        });
    }
    log_summary(request, &labels, &results);
    results
}

fn best_of<C: Copy>(
    time: &mut impl FnMut(usize, C) -> Result<f64>,
    m: usize,
    config: C,
) -> Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..TIMING_REPEATS {
        best = best.min(time(m, config)?);
    }
    Ok(best)
}

/// The fastest candidate wins only when it beats the policy by `MIN_GAIN`; otherwise the policy stays.
fn select<C: Copy>(fallback: C, fallback_ms: f64, timings: &[(C, f64)]) -> (C, f64) {
    let best = timings.iter().copied().min_by(|a, b| a.1.total_cmp(&b.1));
    match best {
        Some((config, ms)) if ms < fallback_ms * (1.0 - MIN_GAIN) => (config, ms),
        _ => (fallback, fallback_ms),
    }
}

fn log_summary<C: Copy + PartialEq + Display>(
    request: &TuneRequest<C>,
    labels: &[String],
    results: &[Tuned<C>],
) {
    if results.iter().all(|r| r.source == Source::Fallback) {
        return;
    }
    let parts: Vec<String> = results
        .iter()
        .zip(labels)
        .map(|(r, label)| {
            let source = match r.source {
                Source::Cache => "cached",
                Source::Measured => "measured",
                Source::Fallback => "policy",
            };
            let policy = (request.fallback)(r.bucket.probe);
            if r.ms > 0.0 && r.config != policy {
                format!(
                    "{label} {} {:.3}ms vs policy {:.3}ms ({source})",
                    r.config, r.ms, r.fallback_ms
                )
            } else {
                format!("{label} {} ({source})", r.config)
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

    pub fn set(&self, key: K, tuned: &[Tuned<C>]) {
        let configs = tuned.iter().map(|t| (t.bucket.upper, t.config)).collect();
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

fn device_key(dev: &CudaDevice) -> String {
    let cu_device = dev.cuda_stream().context().cu_device();
    let name = result::device::get_name(cu_device)
        .map(|name| name.trim().to_string())
        .unwrap_or_else(|_| "unknown-gpu".to_string());
    let (major, minor) = super::device_compute_capability(dev);
    let sms = super::device_multiprocessor_count(dev);
    format!("{name}|sm{major}{minor}|{sms}sm")
}

fn cache_key<C>(device_key: &str, request: &TuneRequest<C>, label: &str) -> String {
    format!(
        "{device_key}|{}|{}@{}|{}|{label}",
        env!("CARGO_PKG_VERSION"),
        request.kernel,
        request.version,
        request.shape
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    config: serde_json::Value,
    ms: f64,
    fallback_ms: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    entries: BTreeMap<String, Entry>,
}

struct TypedEntry<C> {
    config: C,
    ms: f64,
    fallback_ms: f64,
}

/// The on-disk cache, loaded once per process and merged with the file on every write so
/// concurrent processes only ever add entries.
struct TuneCache {
    path: PathBuf,
    file: CacheFile,
}

impl TuneCache {
    fn load(path: PathBuf) -> Self {
        let file = read_cache_file(&path).unwrap_or_default();
        Self { path, file }
    }

    fn get<C: DeserializeOwned>(&self, key: &str) -> Option<TypedEntry<C>> {
        let entry = self.file.entries.get(key)?;
        let config = serde_json::from_value(entry.config.clone()).ok()?;
        Some(TypedEntry {
            config,
            ms: entry.ms,
            fallback_ms: entry.fallback_ms,
        })
    }

    fn insert(&mut self, key: String, entry: Entry) {
        if let Some(on_disk) = read_cache_file(&self.path) {
            for (k, v) in on_disk.entries {
                self.file.entries.entry(k).or_insert(v);
            }
        }
        self.file.entries.insert(key, entry);
        self.file.version = CACHE_FORMAT_VERSION;
        if let Err(err) = write_cache_file(&self.path, &self.file) {
            tracing::warn!(
                "cuTile autotune: could not write {}: {err}",
                self.path.display()
            );
        }
    }
}

fn read_cache_file(path: &Path) -> Option<CacheFile> {
    let text = std::fs::read_to_string(path).ok()?;
    let file: CacheFile = serde_json::from_str(&text).ok()?;
    (file.version == CACHE_FORMAT_VERSION).then_some(file)
}

fn write_cache_file(path: &Path, file: &CacheFile) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    std::fs::write(&tmp, serde_json::to_string_pretty(file)?)?;
    std::fs::rename(&tmp, path)
}

fn cache_path() -> PathBuf {
    if let Some(path) = std::env::var_os(TUNE_CACHE_ENV) {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("mistralrs").join(CACHE_FILE);
    }
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path)
            .join(".cache")
            .join("mistralrs")
            .join(CACHE_FILE);
    }
    std::env::temp_dir().join("mistralrs").join(CACHE_FILE)
}

fn cache() -> &'static Mutex<TuneCache> {
    static CACHE: OnceLock<Mutex<TuneCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(TuneCache::load(cache_path())))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn winner_needs_a_margin_over_the_policy() {
        let timings = [(2u8, 0.98), (3u8, 0.99)];
        assert_eq!(select(1u8, 1.0, &timings), (1, 1.0));
        let timings = [(2u8, 0.96), (3u8, 0.90)];
        assert_eq!(select(1u8, 1.0, &timings), (3, 0.90));
        assert_eq!(select(1u8, 1.0, &[]), (1, 1.0));
    }

    #[test]
    fn table_picks_the_first_bucket_holding_m() {
        let table: TunedTable<u8, i32> = TunedTable::new();
        assert_eq!(table.get(0, 5), None);
        let tuned = [
            Tuned {
                bucket: Bucket {
                    upper: 96,
                    probe: 32,
                },
                config: 1,
                source: Source::Fallback,
                ms: 0.0,
                fallback_ms: 0.0,
            },
            Tuned {
                bucket: Bucket {
                    upper: usize::MAX,
                    probe: 4096,
                },
                config: 2,
                source: Source::Fallback,
                ms: 0.0,
                fallback_ms: 0.0,
            },
        ];
        table.set(0, &tuned);
        assert_eq!(table.get(0, 96), Some(1));
        assert_eq!(table.get(0, 97), Some(2));
        assert_eq!(table.get(1, 97), None);
    }

    #[test]
    fn cache_round_trips_and_merges_with_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join(CACHE_FILE);
        let mut a = TuneCache::load(path.clone());
        a.insert(
            "k1".to_string(),
            Entry {
                config: serde_json::json!({"bm": 16}),
                ms: 0.5,
                fallback_ms: 0.6,
            },
        );
        // a second process that loaded before k1 existed must not drop it
        let mut b = TuneCache::load(dir.path().join("elsewhere.json"));
        b.path = path.clone();
        b.insert(
            "k2".to_string(),
            Entry {
                config: serde_json::json!({"bm": 64}),
                ms: 1.0,
                fallback_ms: 1.0,
            },
        );
        let reloaded = TuneCache::load(path);
        assert_eq!(reloaded.file.entries.len(), 2);
        let entry = reloaded.get::<serde_json::Value>("k1").unwrap();
        assert_eq!(entry.config["bm"], 16);
        assert_eq!(entry.fallback_ms, 0.6);
        assert!(reloaded.get::<serde_json::Value>("missing").is_none());
    }
}
