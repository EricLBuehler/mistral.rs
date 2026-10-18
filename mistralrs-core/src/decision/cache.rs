use std::collections::{BTreeMap, HashMap};

use candle_core::{DType, Device, Tensor};
use sha2::{Digest, Sha256};

use super::DecisionPair;

const DEFAULT_CACHE_BUDGET: &str = "64MiB";
const FREE_MEMORY_PERCENT: usize = 90;
const MAX_SCORE_CACHE_BYTES: usize = 1 << 20;
const SCORE_CACHE_BUDGET_DIVISOR: usize = 64;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Role {
    State,
    Action,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct CacheKey(Role, [u8; 32]);

impl CacheKey {
    pub fn new(role: Role, text: &str) -> Self {
        Self(role, Sha256::digest(text.as_bytes()).into())
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct ScoreKey([u8; 32]);

impl ScoreKey {
    pub fn new(pairs: &[DecisionPair], temperature: f64) -> Self {
        let mut hash = Sha256::new();
        hash.update(temperature.to_bits().to_le_bytes());
        hash.update(pairs.len().to_le_bytes());
        for pair in pairs {
            hash.update(pair.candidates.len().to_le_bytes());
            for text in std::iter::once(&pair.state).chain(&pair.candidates) {
                hash.update(text.len().to_le_bytes());
                hash.update(text.as_bytes());
            }
        }
        Self(hash.finalize().into())
    }
}

struct ScoreEntry {
    values: Vec<f32>,
    stamp: u64,
}

impl ScoreEntry {
    fn bytes(&self) -> usize {
        std::mem::size_of::<(ScoreKey, Self)>() + self.values.len() * DType::F32.size_in_bytes()
    }
}

struct Entry {
    row: u32,
    stamp: u64,
}

pub(super) struct VectorCache {
    buffer: Option<Tensor>,
    entries: HashMap<CacheKey, Entry>,
    recency: BTreeMap<u64, CacheKey>,
    clock: u64,
    capacity: usize,
    scores: HashMap<ScoreKey, ScoreEntry>,
    score_recency: BTreeMap<u64, ScoreKey>,
    score_bytes: usize,
    score_budget: usize,
}

impl VectorCache {
    pub fn from_env(dim: usize, device: &Device, workspace_bytes: usize) -> anyhow::Result<Self> {
        let spec =
            std::env::var("CLM_ACTION_CACHE").unwrap_or_else(|_| DEFAULT_CACHE_BUDGET.to_string());
        let memory = crate::embedding_models::memory::encoder_memory_usage(device)?;
        let requested = parse_budget(&spec, memory.total())?;
        let bytes = if device.is_cpu() {
            requested
        } else {
            requested
                .min(memory.available() / 100 * FREE_MEMORY_PERCENT)
                .min(
                    crate::paged_attention::device_memory_cap(memory.available(), device)
                        .saturating_sub(workspace_bytes),
                )
        };
        let cache = Self::new(bytes, dim, device)?;
        tracing::info!(
            rows = cache.capacity,
            bytes = cache.capacity * dim * DType::F32.size_in_bytes(),
            ?device,
            "Reserved CLM state/action vector cache"
        );
        Ok(cache)
    }

    pub fn new(bytes: usize, dim: usize, device: &Device) -> candle_core::Result<Self> {
        let capacity = (bytes / (dim * DType::F32.size_in_bytes())).min(u32::MAX as usize);
        Ok(Self {
            buffer: if capacity == 0 {
                None
            } else {
                Some(Tensor::zeros((capacity, dim), DType::F32, device)?)
            },
            entries: HashMap::new(),
            recency: BTreeMap::new(),
            clock: 0,
            capacity,
            scores: HashMap::new(),
            score_recency: BTreeMap::new(),
            score_bytes: 0,
            score_budget: (bytes / SCORE_CACHE_BUDGET_DIVISOR).min(MAX_SCORE_CACHE_BYTES),
        })
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
        self.clock = 0;
        self.clear_scores();
    }

    fn clear_scores(&mut self) {
        self.scores.clear();
        self.score_recency.clear();
        self.score_bytes = 0;
    }

    pub fn scores(&mut self, key: &ScoreKey, vectors: &[CacheKey]) -> Option<Vec<f32>> {
        let entry = self.scores.get_mut(key)?;
        for key in vectors {
            let vector = self.entries.get_mut(key).expect("cached score vector");
            self.recency.remove(&vector.stamp);
            self.clock += 1;
            vector.stamp = self.clock;
            self.recency.insert(vector.stamp, *key);
        }
        self.score_recency.remove(&entry.stamp);
        self.clock += 1;
        entry.stamp = self.clock;
        self.score_recency.insert(entry.stamp, *key);
        Some(entry.values.clone())
    }

    pub fn insert_scores(&mut self, key: ScoreKey, vectors: &[CacheKey], values: &[f32]) {
        let bytes = std::mem::size_of::<(ScoreKey, ScoreEntry)>()
            + values.len() * DType::F32.size_in_bytes();
        if bytes > self.score_budget || vectors.iter().any(|key| !self.entries.contains_key(key)) {
            return;
        }
        if let Some(entry) = self.scores.remove(&key) {
            self.score_recency.remove(&entry.stamp);
            self.score_bytes -= entry.bytes();
        }
        while self.score_bytes + bytes > self.score_budget {
            let (_, oldest) = self.score_recency.pop_first().expect("score cache is full");
            self.score_bytes -= self.scores.remove(&oldest).expect("score entry").bytes();
        }
        self.clock += 1;
        self.score_bytes += bytes;
        self.score_recency.insert(self.clock, key);
        self.scores.insert(
            key,
            ScoreEntry {
                values: values.to_vec(),
                stamp: self.clock,
            },
        );
    }

    pub fn get(&mut self, keys: &[CacheKey]) -> candle_core::Result<Vec<Option<Tensor>>> {
        let mut result = vec![None; keys.len()];
        let Some(buffer) = &self.buffer else {
            return Ok(result);
        };
        let mut rows = Vec::new();
        let mut hits = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if let Some(entry) = self.entries.get_mut(key) {
                self.recency.remove(&entry.stamp);
                self.clock += 1;
                entry.stamp = self.clock;
                self.recency.insert(entry.stamp, *key);
                rows.push(entry.row);
                hits.push(i);
            }
        }
        if !rows.is_empty() {
            let indices = Tensor::new(rows.as_slice(), buffer.device())?;
            // Gather before inserting misses so evictions cannot overwrite this request's hits.
            let vectors = buffer.index_select(&indices, 0)?;
            for (row, index) in hits.into_iter().enumerate() {
                result[index] = Some(vectors.narrow(0, row, 1)?);
            }
        }
        Ok(result)
    }

    pub fn insert(&mut self, keys: &[CacheKey], vectors: &Tensor) -> candle_core::Result<()> {
        // Replacing any vector invalidates scores that may depend on it.
        if keys.iter().any(|key| self.entries.contains_key(key))
            || keys.len() > self.capacity.saturating_sub(self.entries.len())
        {
            self.clear_scores();
        }
        let Some(buffer) = &self.buffer else {
            return Ok(());
        };
        if keys.is_empty() {
            return Ok(());
        }
        let start = keys.len().saturating_sub(self.capacity);
        let keys = &keys[start..];
        let vectors = vectors.narrow(0, start, keys.len())?.contiguous()?;
        let mut rows = Vec::with_capacity(keys.len());
        for key in keys {
            let row = if let Some(entry) = self.entries.remove(key) {
                self.recency.remove(&entry.stamp);
                entry.row
            } else if self.entries.len() == self.capacity {
                let (_, oldest) = self.recency.pop_first().expect("full cache has entries");
                self.entries.remove(&oldest).expect("LRU entry").row
            } else {
                u32::try_from(self.entries.len()).expect("bounded cache row")
            };
            self.clock += 1;
            self.entries.insert(
                *key,
                Entry {
                    row,
                    stamp: self.clock,
                },
            );
            self.recency.insert(self.clock, *key);
            rows.push(row);
        }
        let inserted = (|| {
            let indices = Tensor::new(rows.as_slice(), buffer.device())?
                .unsqueeze(1)?
                .broadcast_as(vectors.shape())?
                .contiguous()?;
            buffer.scatter_set(&indices, &vectors, 0)
        })();
        if let Err(error) = inserted {
            self.clear();
            return Err(error);
        }
        Ok(())
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn parse_budget(spec: &str, total: usize) -> anyhow::Result<usize> {
    let spec = spec.trim();
    let split = spec
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(spec.len());
    let value: f64 = spec[..split].parse()?;
    let unit = spec[split..].trim().to_ascii_uppercase();
    anyhow::ensure!(
        value.is_finite() && value >= 0.0,
        "invalid CLM_ACTION_CACHE budget"
    );
    let multiplier = match unit.as_str() {
        "" => {
            anyhow::ensure!(
                value < 1.0,
                "CLM_ACTION_CACHE without a unit must be a fraction in [0, 1)"
            );
            total as f64
        }
        "B" => 1.0,
        "KB" => 1_000.0,
        "MB" => 1_000_000.0,
        "GB" => 1_000_000_000.0,
        "KIB" => 1024.0,
        "MIB" => 1024.0 * 1024.0,
        "GIB" => 1024.0 * 1024.0 * 1024.0,
        _ => anyhow::bail!("unknown CLM_ACTION_CACHE unit; use B, KB, MB, GB, KiB, MiB, or GiB"),
    };
    let bytes = value * multiplier;
    anyhow::ensure!(
        bytes.is_finite() && bytes < usize::MAX as f64,
        "CLM_ACTION_CACHE budget is too large"
    );
    Ok(bytes as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets() {
        assert_eq!(parse_budget("0", 1000).unwrap(), 0);
        assert_eq!(parse_budget("0.02", 1000).unwrap(), 20);
        assert_eq!(parse_budget("1 MiB", 1000).unwrap(), 1 << 20);
        for invalid in ["", "-1", "1", "NaN", "1TB", "1e99"] {
            assert!(parse_budget(invalid, 1000).is_err(), "{invalid}");
        }
    }

    #[test]
    fn scores_are_bounded_and_invalidated_with_vectors() -> anyhow::Result<()> {
        let mut cache = VectorCache::new(16_384, 2, &Device::Cpu)?;
        let vectors = [
            CacheKey::new(Role::State, "state"),
            CacheKey::new(Role::Action, "action"),
        ];
        cache.insert(
            &vectors,
            &Tensor::new(&[[1_f32, 2.], [3., 4.]], &Device::Cpu)?,
        )?;
        let pairs = [DecisionPair {
            state: "state".into(),
            keys: vec!["a".into()],
            candidates: vec!["action".into()],
        }];
        let keys = [1., 2., 3., 4.].map(|t| ScoreKey::new(&pairs, t));
        for key in &keys[..3] {
            cache.insert_scores(*key, &vectors, &[1., 2.]);
        }
        assert_eq!(cache.scores(&keys[0], &vectors).unwrap(), [1., 2.]);
        cache.insert_scores(keys[3], &vectors, &[3., 4.]);
        assert!(cache.scores(&keys[1], &vectors).is_none());
        assert_eq!(cache.scores(&keys[3], &vectors).unwrap(), [3., 4.]);
        assert!(cache.score_bytes <= cache.score_budget);
        cache.insert_scores(keys[1], &vectors, &[0.; 256]);
        assert!(cache.scores(&keys[1], &vectors).is_none());
        cache.insert_scores(keys[1], &[CacheKey::new(Role::Action, "missing")], &[0.]);
        assert!(cache.scores(&keys[1], &vectors).is_none());
        cache.insert(&vectors[..1], &Tensor::new(&[[5_f32, 6.]], &Device::Cpu)?)?;
        assert!(cache.scores(&keys[0], &vectors).is_none());
        cache.insert_scores(keys[0], &vectors, &[5., 6.]);
        cache.clear();
        assert!(cache.scores(&keys[0], &vectors).is_none());
        Ok(())
    }

    fn eviction_and_ownership(device: Device) -> anyhow::Result<()> {
        let a = CacheKey::new(Role::State, "a");
        let b = CacheKey::new(Role::Action, "a");
        let c = CacheKey::new(Role::State, "c");
        let mut cache = VectorCache::new(2 * 2 * 4, 2, &device)?;
        cache.insert(&[a, b], &Tensor::new(&[[1_f32, 2.], [3., 4.]], &device)?)?;
        let saved = cache.get(&[a])?.remove(0).unwrap();
        cache.insert(&[c], &Tensor::new(&[[5_f32, 6.]], &device)?)?;
        let result = cache.get(&[a, b, c])?;
        assert!(result[0].is_some());
        assert!(result[1].is_none());
        assert_eq!(result[2].as_ref().unwrap().to_vec2::<f32>()?, [[5., 6.]]);
        cache.insert(
            &[a, b, c],
            &Tensor::new(&[[7_f32, 8.], [9., 10.], [11., 12.]], &device)?,
        )?;
        assert_eq!(saved.to_vec2::<f32>()?, [[1., 2.]]);
        let result = cache.get(&[a, b, c])?;
        assert!(result[0].is_none());
        assert_eq!(result[1].as_ref().unwrap().to_vec2::<f32>()?, [[9., 10.]]);
        assert_eq!(result[2].as_ref().unwrap().to_vec2::<f32>()?, [[11., 12.]]);
        cache.clear();
        assert!(cache.get(&[a, b, c])?.iter().all(Option::is_none));
        Ok(())
    }

    #[test]
    fn cpu_eviction_and_ownership() -> anyhow::Result<()> {
        eviction_and_ownership(Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_eviction_and_ownership() -> anyhow::Result<()> {
        eviction_and_ownership(Device::new_metal(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_eviction_and_ownership() -> anyhow::Result<()> {
        eviction_and_ownership(Device::new_cuda(0)?)
    }
}
