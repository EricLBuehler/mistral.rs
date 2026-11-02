pub(crate) fn with_legacy_fp8_backend(
    base: std::sync::Arc<dyn ModelConfigLike + Send + Sync>,
    paged_config: Option<&super::PagedAttentionConfig>,
    device: &candle_core::Device,
) -> std::sync::Arc<dyn ModelConfigLike + Send + Sync> {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    let legacy_sm61 = mistralrs_paged_attn::USE_FP8_LEGACY_SM61;
    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    let legacy_sm61 = false;
    let standard = legacy_fp8_requires_standard(
        legacy_sm61,
        device.is_cuda(),
        paged_config.is_some_and(|config| config.cache_type == super::PagedCacheType::F8E4M3),
    );
    if standard {
        tracing::info!("SM61 software FP8 KV uses the standard paged-attention backend");
        std::sync::Arc::new(AttentionFeaturesConfig {
            base,
            fallback: None,
            force_standard: true,
        })
    } else {
        base
    }
}

fn legacy_fp8_requires_standard(legacy_sm61: bool, cuda: bool, fp8_cache: bool) -> bool {
    legacy_sm61 && cuda && fp8_cache
}

#[cfg(test)]
mod campaign_legacy_fp8_backend_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn campaign_cpu_fp8_kv_backend_selection_is_dtype_and_build_specific() {
        for legacy in [false, true] {
            for cuda in [false, true] {
                for fp8 in [false, true] {
                    assert_eq!(
                        legacy_fp8_requires_standard(legacy, cuda, fp8),
                        legacy && cuda && fp8
                    );
                }
            }
        }
    }

    struct HndModel;

    impl ModelConfigLike for HndModel {
        fn max_seq_len(&self) -> usize {
            2048
        }
        fn num_layers(&self) -> usize {
            3
        }
        fn hidden_size(&self) -> usize {
            2048
        }
        fn num_kv_heads(&self) -> usize {
            4
        }
        fn num_attn_heads(&self) -> usize {
            16
        }
        fn k_head_dim(&self) -> usize {
            128
        }
        fn v_head_dim(&self) -> usize {
            128
        }
        fn attention_backend_kind(&self) -> AttentionBackendKind {
            AttentionBackendKind::FlashInfer
        }
        fn attention_backend_kind_for_layer(&self, _: usize) -> AttentionBackendKind {
            AttentionBackendKind::FlashInfer
        }
        fn kv_cache_layout(&self) -> KvCacheLayout {
            KvCacheLayout::FlashInferHnd
        }
        fn kv_cache_layout_for_layer(&self, _: usize) -> KvCacheLayout {
            KvCacheLayout::FlashInferHnd
        }
        fn layer_has_paged_kv_cache(&self, layer: usize) -> bool {
            layer != 1
        }
        fn sliding_window_for_layer(&self, _: usize) -> Option<usize> {
            Some(128)
        }
        fn prefix_prefill_attention_features(
            &self,
            layer: usize,
        ) -> Option<PrefixPrefillAttentionFeatures> {
            (layer != 1).then_some(PrefixPrefillAttentionFeatures::default())
        }
    }

    fn model() -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(HndModel)
    }

    #[test]
    fn campaign_cpu_fp8_kv_backend_preserves_hybrid_metadata_and_uses_one_layout() {
        let base = model();
        assert_eq!(base.kv_cache_layout(), KvCacheLayout::FlashInferHnd);
        assert_eq!(
            base.attention_backend_kind(),
            AttentionBackendKind::FlashInfer
        );
        let wrapped = AttentionFeaturesConfig {
            base: base.clone(),
            fallback: None,
            force_standard: true,
        };
        assert_eq!(
            wrapped.attention_backend_kind(),
            AttentionBackendKind::Standard
        );
        assert_eq!(wrapped.kv_cache_layout(), KvCacheLayout::Standard);
        assert_eq!(wrapped.kv_cache_group_ids(), base.kv_cache_group_ids());
        assert_eq!(
            wrapped.total_kv_cache_elements_per_token(),
            base.total_kv_cache_elements_per_token()
        );
        for layer in 0..base.num_layers() {
            assert_eq!(
                wrapped.attention_backend_kind_for_layer(layer),
                AttentionBackendKind::Standard
            );
            assert_eq!(
                wrapped.kv_cache_layout_for_layer(layer),
                KvCacheLayout::Standard
            );
            assert_eq!(
                wrapped.layer_has_paged_kv_cache(layer),
                base.layer_has_paged_kv_cache(layer)
            );
            assert_eq!(
                wrapped.prefix_prefill_attention_features(layer),
                base.prefix_prefill_attention_features(layer)
            );
            assert_eq!(
                wrapped.k_head_dim_for_layer(layer),
                base.k_head_dim_for_layer(layer)
            );
            assert_eq!(
                wrapped.sliding_window_for_layer(layer),
                base.sliding_window_for_layer(layer)
            );
        }
        for cache_type in [
            super::super::PagedCacheType::Auto,
            super::super::PagedCacheType::F8E4M3,
        ] {
            let paged =
                super::super::PagedAttentionConfig::new(None, crate::MemoryGoal::Auto, cache_type)
                    .unwrap();
            assert!(Arc::ptr_eq(
                &base,
                &with_legacy_fp8_backend(base.clone(), Some(&paged), &candle_core::Device::Cpu)
            ));
        }
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn campaign_cuda_fp8_kv_sm61_production_cache_layout() -> candle_core::Result<()> {
        use crate::paged_attention::{CacheConfig, CacheEngine, PagedCacheType};
        use candle_core::{DType, Device, Tensor};
        let device = Device::new_cuda(0)?;
        assert!(mistralrs_paged_attn::USE_FP8_LEGACY_SM61);
        let base = model();
        assert_eq!(
            base.attention_backend_kind(),
            AttentionBackendKind::FlashInfer
        );
        assert_eq!(base.kv_cache_layout(), KvCacheLayout::FlashInferHnd);
        let auto = super::super::PagedAttentionConfig::new(
            None,
            crate::MemoryGoal::Auto,
            PagedCacheType::Auto,
        )
        .unwrap();
        let fp8 = super::super::PagedAttentionConfig::new(
            None,
            crate::MemoryGoal::Auto,
            PagedCacheType::F8E4M3,
        )
        .unwrap();
        assert!(Arc::ptr_eq(
            &base,
            &with_legacy_fp8_backend(base.clone(), Some(&auto), &device)
        ));
        let metadata = with_legacy_fp8_backend(base, Some(&fp8), &device);
        assert_eq!(
            metadata.attention_backend_kind(),
            AttentionBackendKind::Standard
        );
        let config = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 2,
            cache_type: PagedCacheType::F8E4M3,
            kv_cache_group_ids: metadata.kv_cache_group_ids(),
        };
        let engine = CacheEngine::new(
            metadata.as_ref(),
            &config,
            DType::F16,
            &device,
            vec![None; 3],
        )?;
        let caches = engine.get_kv_cache();
        assert_eq!(caches[0].0.dims(), &[2, 4, 8, 32, 16]);
        assert_eq!(caches[1].0.dim(0)?, 0);
        assert_eq!(
            AttentionBackendKind::from_cache(&caches[0].0, &caches[0].1),
            AttentionBackendKind::Standard
        );
        let key = Tensor::zeros((32, 4, 128), DType::F16, &device)?;
        let value = Tensor::ones((32, 4, 128), DType::F16, &device)?;
        let slots = Tensor::from_vec((0i64..32).collect(), 32, &device)?;
        let scale = Tensor::new(1f32, &device)?;
        mistralrs_paged_attn::reshape_and_cache(
            &key,
            &value,
            Some(&scale),
            Some(&scale),
            &caches[0].0,
            &caches[0].1,
            &slots,
        )?;
        let query = Tensor::zeros((1, 16, 128), DType::F16, &device)?;
        let table = Tensor::new(&[[0u32]], &device)?;
        let lengths = Tensor::new(&[32u32], &device)?;
        let output = mistralrs_paged_attn::paged_attention(
            &query,
            Some(&scale),
            Some(&scale),
            &caches[0].0,
            &caches[0].1,
            &table,
            &lengths,
            None,
            32,
            1.0,
            1.0,
            None,
        )?;
        device.synchronize()?;
        assert!(output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| (x - 1.0).abs() < 0.01));
        Ok(())
    }
}
