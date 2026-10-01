#[cfg(any(feature = "metal", test))]
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "metal", test))]
use candle_core::{DType, DeviceLocation, Result, Tensor};

#[cfg(any(feature = "metal", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrefixMaskKey {
    pub query_lens: Vec<usize>,
    pub kv_lens: Vec<usize>,
    pub mm_prefix_ranges: Option<Vec<Vec<(usize, usize)>>>,
    pub q_max: usize,
    pub kv_max: usize,
    pub sliding_window: Option<usize>,
    pub packed: bool,
    pub dtype: DType,
    pub device: DeviceLocation,
}

#[cfg(any(feature = "metal", test))]
#[derive(Clone, Debug, Default)]
pub(crate) struct PrefixMaskCache(Arc<Mutex<Vec<(PrefixMaskKey, Tensor)>>>);

#[cfg(not(any(feature = "metal", test)))]
#[derive(Clone, Debug, Default)]
pub(crate) struct PrefixMaskCache;

#[cfg(any(feature = "metal", test))]
impl PrefixMaskCache {
    pub(crate) fn get_or_try_insert(
        &self,
        key: PrefixMaskKey,
        make_mask: impl FnOnce() -> Result<Tensor>,
    ) -> Result<Tensor> {
        let mut masks = self
            .0
            .lock()
            .map_err(|_| candle_core::Error::msg("prefix mask cache mutex poisoned"))?;
        if let Some((_, mask)) = masks.iter().find(|(stored, _)| stored == &key) {
            return Ok(mask.clone());
        }
        let mask = make_mask()?;
        masks.push((key, mask.clone()));
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn key() -> PrefixMaskKey {
        PrefixMaskKey {
            query_lens: vec![2, 1],
            kv_lens: vec![4, 3],
            mm_prefix_ranges: None,
            q_max: 2,
            kv_max: 4,
            sliding_window: None,
            packed: false,
            dtype: DType::F32,
            device: DeviceLocation::Cpu,
        }
    }

    #[test]
    fn metadata_clones_share_masks_but_new_steps_do_not() -> Result<()> {
        let cache = PrefixMaskCache::default();
        let mask = cache.get_or_try_insert(key(), || Tensor::new(1f32, &Device::Cpu))?;
        let reused = cache.clone().get_or_try_insert(key(), || {
            panic!("a shared prefix mask must not be uploaded twice")
        })?;
        assert_eq!(mask.id(), reused.id());
        let next = PrefixMaskCache::default()
            .get_or_try_insert(key(), || Tensor::new(2f32, &Device::Cpu))?;
        assert_ne!(mask.id(), next.id());
        Ok(())
    }

    #[test]
    fn mask_geometry_and_semantics_select_distinct_entries() -> Result<()> {
        let original = key();
        let mut variants = Vec::new();
        let mut changed = original.clone();
        changed.query_lens = vec![1, 2];
        variants.push(changed);
        let mut changed = original.clone();
        changed.kv_lens = vec![3, 4];
        variants.push(changed);
        let mut changed = original.clone();
        changed.mm_prefix_ranges = Some(vec![vec![(0, 2)], vec![]]);
        variants.push(changed);
        let mut changed = original.clone();
        changed.sliding_window = Some(2);
        variants.push(changed);
        let mut changed = original.clone();
        changed.packed = true;
        variants.push(changed);
        let mut changed = original.clone();
        changed.q_max = 3;
        variants.push(changed);
        let mut changed = original.clone();
        changed.kv_max = 5;
        variants.push(changed);
        let mut changed = original.clone();
        changed.dtype = DType::BF16;
        variants.push(changed);
        let mut changed = original.clone();
        changed.device = DeviceLocation::Metal { gpu_id: 1 };
        variants.push(changed);
        let cache = PrefixMaskCache::default();
        cache.get_or_try_insert(original, || Tensor::new(0f32, &Device::Cpu))?;
        for (index, key) in variants.into_iter().enumerate() {
            let expected = f32::from(u16::try_from(index).unwrap()) + 1.0;
            let mask = cache.get_or_try_insert(key, || Tensor::new(expected, &Device::Cpu))?;
            assert_eq!(mask.to_scalar::<f32>()?, expected);
        }
        Ok(())
    }
}
