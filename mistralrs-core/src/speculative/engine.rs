//! Engine-side speculative completion driver.
//!
//! Called from `Pipeline::try_speculative_completion`. For each eligible
//! sequence, runs one MTP step:
//!   1. Get the seq's stashed `(last_token, last_hidden)` (seeded from the
//!      prompt forward on the first call).
//!   2. Extract sliding + full donor KV from the target's `EitherCache::Normal`.
//!   3. Call the proposer to get K drafts.
//!   4. Reserve `K+1` cache slots via `SpeculativeCacheGuard`.
//!   5. Push the K drafts onto `seq.tokens` and build a K+1-token completion
//!      input via [`make_spec_completion_chunk`].
//!   6. Call `forward_inputs` with `return_hidden_states=true`.
//!   7. For each draft position i, sample from target logits[i] using the
//!      seq's sampler. If sampled == draft[i], accept; else reject and the
//!      sampled token becomes the bonus.
//!   8. Pop unaccepted drafts off `seq.tokens`.
//!   9. Commit cache to `committed + accepted + 1`; sample bonus and add
//!      via `finish_or_add_toks_to_seq` for each emitted token.
//!  10. Stash `hidden[accepted]` as `seq.mtp_seed_hidden` and notify the
//!      proposer via `on_accept`.

use candle_core::Result;

use crate::kv_cache::EitherCache;
use crate::speculative::mtp::head::SharedKv;

/// Extract `(k, v)` from a single layer of an `EitherCache::Normal` cache.
/// Returns `SharedKv` shaped `(batch, num_kv_heads, kv_len, head_dim)`.
pub fn extract_donor_kv(cache: &EitherCache, layer_idx: usize) -> Result<SharedKv> {
    let EitherCache::Normal(arc) = cache else {
        candle_core::bail!(
            "MTP donor KV extraction currently only supports EitherCache::Normal; \
             use --paged-attn off when MTP is active"
        );
    };
    let nc = arc.lock().unwrap();
    let layer = nc.0.get(layer_idx).ok_or_else(|| {
        candle_core::Error::Msg(format!("MTP donor layer index {layer_idx} out of range"))
    })?;
    let k = layer.appended_k()?.ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "MTP donor layer {layer_idx} has no appended_k (cache not warmed up?)"
        ))
    })?;
    let v = layer.appended_v()?.ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "MTP donor layer {layer_idx} has no appended_v (cache not warmed up?)"
        ))
    })?;
    Ok(SharedKv { k, v })
}
