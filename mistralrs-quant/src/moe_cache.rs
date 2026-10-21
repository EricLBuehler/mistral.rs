//! Optional residency cache for routed expert slabs.
//!
//! A memory-mapped safetensors checkpoint leaves the model in reclaimable page cache, which is
//! the right default: it costs nothing and the kernel evicts experts a token stopped routing to.
//! It is the wrong thing on a *shared* box. When another tenant's working set evicts our
//! pages, a decode step starts taking major faults, and at 6.82 GB per step that is unbounded
//! latency. This is the mechanism for pinning the hot subset instead of hoping the kernel keeps
//! it: an anonymous copy is not file-backed page cache and cannot be reclaimed, and with no
//! swap it cannot be paged out either.
//!
//! It is deliberately off unless `MISTRALRS_MOE_CACHE_MB` says otherwise, because the cost is
//! real on the same contended box: N GB of anonymous memory is N GB less for whoever else is
//! running. This buys residency, which is a latency and reproducibility property, not
//! throughput.
//!
//! Sized from the measured reuse rather than from the model size. A decode step touches about
//! half the expert pool (211 of 384 on Kolibri-1) and 57.6% of selections repeat from the
//! previous step, with 96.8% within five steps. At 2.36 GB of expert weight per step that puts
//! the knee near 5-10 GB; 12-16 GB is comfortably past it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::{CpuStorage, Storage, Tensor};

/// Megabytes of expert weight to keep resident. Unset or 0 disables the cache.
const ENV: &str = "MISTRALRS_MOE_CACHE_MB";

struct Entry {
    tensor: Tensor,
    bytes: usize,
    /// Access counter, so eviction can find the coldest entry without maintaining an order.
    last: u64,
}

#[derive(Default)]
struct State {
    /// `(owner identity, expert index) -> resident copy`. The owner is the address of the
    /// `Arc` holding the slab list, which is stable for the life of the linear and unique to it.
    entries: HashMap<(usize, usize), Entry>,
    bytes: usize,
    tick: u64,
}

pub struct MoECache {
    budget: usize,
    state: Mutex<State>,
}

fn budget_bytes() -> Option<usize> {
    std::env::var(ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|mb| *mb > 0)
        .map(|mb| mb.saturating_mul(1024 * 1024))
}

/// The process-wide cache, or `None` when it is not configured.
pub fn cache() -> Option<&'static MoECache> {
    static CACHE: OnceLock<Option<MoECache>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            budget_bytes().map(|budget| MoECache {
                budget,
                state: Mutex::new(State::default()),
            })
        })
        .as_ref()
}

/// Bytes one slab occupies once resident.
fn slab_bytes(slab: &Tensor) -> usize {
    slab.elem_count() * slab.dtype().size_in_bytes()
}

/// A private copy of `src`, so reads bypass the file-backed mapping.
///
/// Falls back to sharing `src` for anything this does not know how to copy byte-for-byte; the
/// cache then holds no private copy of that slab and the caller reads it as it would have
/// anyway, which is correct and merely not accelerated.
fn copy_slab(src: &Tensor) -> Option<Tensor> {
    let (guard, layout) = src.storage_and_layout();
    let Storage::Cpu(cpu) = &*guard else {
        return None;
    };
    let start = layout.start_offset();
    let n = src.elem_count();
    let dims = src.dims().to_vec();
    match cpu {
        CpuStorage::BF16(v) => Tensor::from_slice(&v[start..start + n], dims, src.device()).ok(),
        CpuStorage::F32(v) => Tensor::from_slice(&v[start..start + n], dims, src.device()).ok(),
        _ => None,
    }
}

impl MoECache {
    /// A resident slab for `(owner, idx)`.
    ///
    /// On a hit this is a clone of a `Tensor`, which is an `Arc` bump. On a miss it copies the
    /// slab once and then evicts the coldest entries until the cache is back inside its budget,
    /// so the footprint is bounded no matter how large the model or how long the run.
    pub fn slab(&self, owner: usize, idx: usize, src: &Tensor) -> Tensor {
        if let Some(resident) = self.get(owner, idx) {
            return resident;
        }

        // Copied outside the lock: this is a memcpy of a few MB and there is no reason to hold
        // the cache still for it.
        let Some(copy) = copy_slab(src) else {
            return src.clone();
        };
        self.insert(owner, idx, copy)
    }

    /// The resident slab for `(owner, idx)`, if the cache holds one.
    ///
    /// This is the peek half of a peek-then-fill: a caller that has to *compute* the slab
    /// wants to know whether it already has one before it spends the arithmetic, and the
    /// whole-stack callers cannot use `slab` for that because they are handed a source
    /// tensor they have already paid for.
    pub fn get(&self, owner: usize, idx: usize) -> Option<Tensor> {
        let Ok(mut state) = self.state.lock() else {
            return None;
        };
        let tick = state.tick;
        state.tick += 1;
        let entry = state.entries.get_mut(&(owner, idx))?;
        entry.last = tick;
        Some(entry.tensor.clone())
    }

    /// Inserts a slab the caller already owns, and returns it.
    ///
    /// The tensor is stored as it is, not copied: the caller is handing over something it
    /// computed and nobody else holds, so a copy would be pure waste. It is the fill half of
    /// a peek-then-fill.
    pub fn insert(&self, owner: usize, idx: usize, tensor: Tensor) -> Tensor {
        let key = (owner, idx);
        {
            let Ok(mut state) = self.state.lock() else {
                return tensor;
            };
            let tick = state.tick;
            state.tick += 1;
            if let Some(entry) = state.entries.get_mut(&key) {
                entry.last = tick;
                return entry.tensor.clone();
            }
        }

        let bytes = slab_bytes(&tensor);
        let Ok(mut state) = self.state.lock() else {
            return tensor;
        };
        let tick = state.tick;
        state.tick += 1;
        state.bytes += bytes;
        state.entries.insert(
            key,
            Entry {
                tensor: tensor.clone(),
                bytes,
                last: tick,
            },
        );
        while state.bytes > self.budget {
            let victim = state
                .entries
                .iter()
                .filter(|(k, _)| **k != key || state.entries.len() > 1)
                .min_by_key(|(_, e)| e.last)
                .map(|(k, _)| *k);
            let Some(victim) = victim else { break };
            if let Some(evicted) = state.entries.remove(&victim) {
                state.bytes = state.bytes.saturating_sub(evicted.bytes);
            }
        }
        tensor
    }

    /// Resident bytes and entry count, for a test or a progress line.
    pub fn stats(&self) -> (usize, usize) {
        match self.state.lock() {
            Ok(state) => (state.bytes, state.entries.len()),
            Err(_) => (0, 0),
        }
    }
}

/// Bytes the cache is configured to hold, or `None` when it is disabled.
pub fn configured_budget() -> Option<usize> {
    budget_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_core::{DType, Result};

    /// A slab that is resident must read back identically to the source, for both dtypes the
    /// gather kernel reads. A silent dtype mismatch here would corrupt every expert output.
    #[test]
    fn copies_round_trip_bit_exactly() -> Result<()> {
        let cache = MoECache {
            budget: 1 << 20,
            state: Mutex::new(State::default()),
        };
        for (idx, dtype) in [DType::F32, DType::BF16].into_iter().enumerate() {
            let src = Tensor::rand(-1f32, 1f32, (8, 16), &Device::Cpu)?.to_dtype(dtype)?;
            let widen = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
            // A distinct key per dtype: the cache is keyed by (owner, index) and does not know
            // about dtype, so reusing one key would hand back the previous slab.
            assert_eq!(widen(&src)?, widen(&cache.slab(1, idx, &src))?, "{dtype:?}");
        }
        Ok(())
    }

    /// The peek-then-fill pair has to agree with the copy-and-fill path it sits beside: a
    /// caller that asks first and inserts on a miss must end up with the same values, and a
    /// miss must report a miss rather than an empty slab.
    #[test]
    fn peek_then_fill_agrees_with_copy_and_fill() -> Result<()> {
        let cache = MoECache {
            budget: 1 << 20,
            state: Mutex::new(State::default()),
        };
        let src = Tensor::rand(-1f32, 1f32, (8, 16), &Device::Cpu)?;
        let widen = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();

        assert!(
            cache.get(1, 0).is_none(),
            "an empty cache must report a miss"
        );
        let filled = cache.insert(1, 0, src.clone());
        assert_eq!(
            widen(&src)?,
            widen(&filled)?,
            "the inserted slab must read back"
        );
        let peeked = cache.get(1, 0).expect("the inserted slab must be found");
        assert_eq!(
            widen(&src)?,
            widen(&peeked)?,
            "the peeked slab must read back"
        );
        assert!(
            cache.get(1, 1).is_none(),
            "a different index must still miss"
        );
        assert!(
            cache.get(2, 0).is_none(),
            "a different owner must still miss"
        );
        Ok(())
    }

    /// The cache must stay inside its budget however many distinct slabs it is shown, or it is
    /// just a slower way of loading the whole model into anonymous memory.
    #[test]
    fn evicts_to_stay_within_budget() -> Result<()> {
        // Room for four 8x16 f32 slabs (512 bytes each) plus one.
        let cache = MoECache {
            budget: 4 * 512 + 256,
            state: Mutex::new(State::default()),
        };
        for idx in 0..64usize {
            let src = Tensor::rand(-1f32, 1f32, (8, 16), &Device::Cpu)?;
            let got = cache.slab(1, idx, &src);
            let (bytes, entries) = cache.stats();
            assert!(
                bytes <= cache.budget,
                "idx {idx}: {bytes} > {}",
                cache.budget
            );
            assert!(entries <= 6, "idx {idx}: {entries} entries");
            // Whatever comes back must still be the right slab, hit or miss.
            assert_eq!(
                src.flatten_all()?.to_vec1::<f32>()?,
                got.flatten_all()?.to_vec1::<f32>()?,
                "idx {idx}"
            );
        }
        Ok(())
    }

    /// The coldest entry goes first, not the oldest inserted, or a hot expert that was resident
    /// early would be evicted while a cold one survived.
    #[test]
    fn evicts_the_coldest_entry() -> Result<()> {
        let cache = MoECache {
            budget: 2 * 512 + 64,
            state: Mutex::new(State::default()),
        };
        let slab = || Tensor::rand(-1f32, 1f32, (8, 16), &Device::Cpu);
        cache.slab(1, 0, &slab()?);
        cache.slab(1, 1, &slab()?);
        // Touch 0 so 1 becomes the coldest, then force an eviction with a third.
        cache.slab(1, 0, &slab()?);
        cache.slab(1, 2, &slab()?);
        let keys: Vec<usize> = {
            let state = cache.state.lock().expect("lock");
            state.entries.keys().map(|(_, e)| *e).collect()
        };
        assert!(keys.contains(&0), "the touched entry was evicted: {keys:?}");
        assert!(keys.contains(&2));
        Ok(())
    }
}
