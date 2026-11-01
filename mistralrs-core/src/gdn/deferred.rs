use candle_core::{DType, Result};

use super::GatedDeltaNet;
use crate::kv_cache::{HybridCache, HybridLayerCache};

pub(crate) fn reserve_deferred_decode_state<'a>(
    cache: &mut HybridCache,
    layers: impl IntoIterator<Item = (usize, &'a GatedDeltaNet)>,
    dtype: DType,
) -> Result<bool> {
    let mut specs = Vec::new();
    for (layer_idx, gdn) in layers {
        let Some(HybridLayerCache::Recurrent(pool)) = cache.get(layer_idx) else {
            candle_core::bail!("GDN layer {layer_idx} has no recurrent state pool");
        };
        if gdn.deferred_decode_supported(pool, dtype) {
            specs.push((layer_idx, gdn.deferred_state_spec()));
        }
    }
    cache.reserve_gdn_deferred_state(&specs)
}
