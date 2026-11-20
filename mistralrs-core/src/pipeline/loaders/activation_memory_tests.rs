use super::*;

#[test]
fn primary_text_reservation_includes_planned_attention_and_logits() -> Result<()> {
    let baseline = ActivationMemoryEstimate {
        mapped_bytes: 30,
        non_mapped_bytes: 100,
        mapped_non_attention_bytes: 20,
        non_mapped_non_attention_bytes: 100,
        encoder_bytes: 0,
        planned_query_tokens: 512,
        logits_bytes_per_query_token: 7,
        custom_prefill_mask: false,
        metal_prefill_mask_is_structured: false,
    };
    for attention in [10, 80, 200] {
        let estimate = ActivationMemoryEstimate {
            mapped_bytes: baseline.mapped_non_attention_bytes + attention,
            ..baseline
        }
        .reserve_attention_inventory(0, 1)?;
        assert_eq!(estimate.non_mapped_bytes, 107 + attention);
        assert_eq!(estimate.mapped_bytes, 20 + attention);
        let mut paged = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?;
        apply_model_activation_memory(&mut paged, estimate);
        let plan = paged.runtime_memory_plan();
        assert_eq!(
            plan.activation_bytes(true, baseline.planned_query_tokens)?,
            plan.non_attention_bytes(true, baseline.planned_query_tokens, 1)? + attention
        );
        assert_eq!(plan.mapped_activation_bytes, 20 + attention);
        assert_eq!(plan.primary_non_attention_bytes, 100);
        assert_eq!(plan.mapped_non_attention_bytes, 20);
        assert_eq!(plan.logits_bytes_per_query_token, 7);
    }
    Ok(())
}

#[test]
fn primary_attention_normalization_preserves_larger_phase_reservations() -> Result<()> {
    let estimate = ActivationMemoryEstimate {
        mapped_bytes: 90,
        non_mapped_bytes: 200,
        mapped_non_attention_bytes: 80,
        non_mapped_non_attention_bytes: 50,
        encoder_bytes: 0,
        planned_query_tokens: 512,
        logits_bytes_per_query_token: 0,
        custom_prefill_mask: false,
        metal_prefill_mask_is_structured: false,
    };
    assert_eq!(estimate.reserve_attention_inventory(0, 1)?, estimate);
    assert!(ActivationMemoryEstimate {
        non_mapped_non_attention_bytes: usize::MAX,
        ..estimate
    }
    .reserve_attention_inventory(0, 1)
    .is_err());
    Ok(())
}

#[test]
fn encoder_and_text_attention_use_separate_startup_peaks() -> Result<()> {
    let estimate = ActivationMemoryEstimate {
        mapped_bytes: 80,
        mapped_non_attention_bytes: 20,
        non_mapped_non_attention_bytes: 40,
        encoder_bytes: 100,
        planned_query_tokens: 512,
        logits_bytes_per_query_token: 5,
        ..Default::default()
    }
    .reserve_attention_inventory(0, 2)?;
    assert_eq!(estimate.non_mapped_bytes, 140);
    assert_eq!(estimate.non_mapped_non_attention_bytes, 40);
    let mut paged = PagedAttentionConfig::new(
        Some(32),
        MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?;
    apply_model_activation_memory(&mut paged, estimate);
    let plan = paged.runtime_memory_plan();
    assert_eq!(plan.encoder_activation_bytes, 100);
    assert_eq!(plan.non_attention_bytes(true, 512, 2)?, 50);
    Ok(())
}

#[test]
fn cache_write_inventory_is_reserved_on_primary_and_mapped_devices() -> Result<()> {
    const QUERY_TOKENS: usize = 512;
    const PREFIX_ATTENTION: usize = 80;
    const CACHE_WRITE: usize = 32;
    let estimate = ActivationMemoryEstimate {
        mapped_bytes: 20 + PREFIX_ATTENTION,
        non_mapped_bytes: 100,
        mapped_non_attention_bytes: 20,
        non_mapped_non_attention_bytes: 100,
        encoder_bytes: 0,
        planned_query_tokens: QUERY_TOKENS,
        logits_bytes_per_query_token: 0,
        custom_prefill_mask: false,
        metal_prefill_mask_is_structured: false,
    }
    .reserve_attention_inventory(CACHE_WRITE, 1)?;
    let mut paged = PagedAttentionConfig::new(
        Some(32),
        MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?;
    apply_model_activation_memory(&mut paged, estimate);
    let plan = paged.runtime_memory_plan();
    for primary in [false, true] {
        assert_eq!(
            plan.activation_bytes(primary, QUERY_TOKENS)?,
            plan.non_attention_bytes(primary, QUERY_TOKENS, 1)? + PREFIX_ATTENTION + CACHE_WRITE
        );
    }
    assert!(ActivationMemoryEstimate {
        mapped_bytes: usize::MAX,
        ..estimate
    }
    .reserve_attention_inventory(CACHE_WRITE, 1)
    .is_err());
    Ok(())
}

#[test]
fn structured_mask_metadata_reaches_runtime_without_changing_non_attention() -> Result<()> {
    let mut paged = PagedAttentionConfig::new(
        Some(32),
        MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?;
    apply_model_activation_memory(
        &mut paged,
        ActivationMemoryEstimate {
            mapped_bytes: 120,
            non_mapped_bytes: 220,
            mapped_non_attention_bytes: 100,
            non_mapped_non_attention_bytes: 200,
            planned_query_tokens: 512,
            custom_prefill_mask: true,
            metal_prefill_mask_is_structured: true,
            ..Default::default()
        },
    );
    let plan = paged.runtime_memory_plan();
    assert!(plan.custom_prefill_mask);
    assert!(plan.metal_prefill_mask_is_structured);
    assert_eq!(plan.non_attention_bytes(true, 512, 1)?, 200);
    assert_eq!(plan.non_attention_bytes(false, 512, 1)?, 100);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a Metal device"]
fn native_metal_startup_reservation_has_no_full_prefix_gather() -> Result<()> {
    let device = Device::new_metal(0)?;
    let metadata = crate::paged_attention::ModelConfigMetadata {
        max_seq_len: 65_536,
        num_layers: 2,
        hidden_size: 1024,
        num_kv_heads: 2,
        num_attn_heads: 8,
        sliding_window: None,
        k_head_dim: 512,
        v_head_dim: 512,
        kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
    };
    let model = crate::paged_attention::AttentionFeaturesConfig::new(
        &metadata,
        Some(PrefixPrefillAttentionFeatures::default()),
    );
    let paged = PagedAttentionConfig::new(
        Some(32),
        MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?
    .with_execution_limits(ExecutionMemoryLimits {
        max_num_seqs: 1,
        max_num_batched_tokens: 512,
        max_prefill_chunk_tokens: 512,
    })?;
    let params = AutoDeviceMapParams::default_text();
    let estimate = |capacity, custom, structured| {
        model_activation_memory(
            &ActivationMemoryContext {
                params: &params,
                limits: paged.execution_limits,
                dtype: DType::BF16,
                device: &device,
                paged: &paged,
                cache_capacity_tokens: Some(capacity),
            },
            ModelActivationConfig {
                model: &model,
                architecture: ActivationModelConfig {
                    intermediate_size: 4096,
                    vocab_size: 1024,
                    custom_prefill_mask: custom,
                    metal_prefill_mask_is_structured: structured,
                    ..Default::default()
                },
            },
        )
    };
    let short = estimate(512, false, false)?;
    let long = estimate(65_536, false, false)?;
    assert_eq!(
        long.mapped_non_attention_bytes,
        short.mapped_non_attention_bytes
    );
    assert!(long.mapped_bytes < short.mapped_bytes * 2);
    let structured = estimate(65_536, true, true)?;
    let custom = estimate(65_536, true, false)?;
    assert!(custom.mapped_bytes > structured.mapped_bytes * 8);
    assert_eq!(
        custom.mapped_non_attention_bytes,
        structured.mapped_non_attention_bytes
    );
    Ok(())
}

#[test]
fn prefix_capacity_uses_usable_blocks_and_actual_cache_dtype() -> Result<()> {
    let model = crate::paged_attention::ModelConfigMetadata {
        max_seq_len: 8192,
        num_layers: 2,
        hidden_size: 512,
        num_kv_heads: 2,
        num_attn_heads: 8,
        sliding_window: None,
        k_head_dim: 64,
        v_head_dim: 96,
        kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
    };
    let params = AutoDeviceMapParams::default_text();
    let capacity = |goal, cache_type| -> Result<_> {
        let paged = PagedAttentionConfig::new(Some(32), goal, cache_type)?;
        explicit_prefix_context_capacity(
            &ActivationMemoryContext {
                params: &params,
                limits: paged.execution_limits,
                dtype: DType::BF16,
                device: &Device::Cpu,
                paged: &paged,
                cache_capacity_tokens: None,
            },
            &model,
        )
    };
    let auto = crate::paged_attention::PagedCacheType::Auto;
    let fp8 = crate::paged_attention::PagedCacheType::F8E4M3;
    let bytes_per_block = 2 * 2 * (64 + 96) * 32 * 2;
    assert_eq!(capacity(MemoryGoal::KvTokens(33), auto)?, Some(64));
    assert_eq!(capacity(MemoryGoal::KvTokens(9000), auto)?, Some(8192));
    assert_eq!(
        capacity(MemoryGoal::KvBytes(3 * bytes_per_block), auto)?,
        Some(64)
    );
    assert_eq!(
        capacity(MemoryGoal::KvBytes(3 * bytes_per_block), fp8)?,
        Some(160)
    );
    assert_eq!(capacity(MemoryGoal::Auto, auto)?, None);
    assert_eq!(capacity(MemoryGoal::Utilization(0.9), auto)?, None);
    assert!(capacity(MemoryGoal::KvBytes(bytes_per_block), auto).is_err());
    Ok(())
}

#[test]
fn startup_prefix_reservation_respects_dense_cap_and_query_lower_bound() -> Result<()> {
    struct CappedConfig {
        cap: Option<usize>,
    }
    impl ModelConfigLike for CappedConfig {
        fn max_seq_len(&self) -> usize {
            65_536
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn hidden_size(&self) -> usize {
            512
        }
        fn num_kv_heads(&self) -> usize {
            2
        }
        fn num_attn_heads(&self) -> usize {
            8
        }
        fn k_head_dim(&self) -> usize {
            64
        }
        fn v_head_dim(&self) -> usize {
            64
        }
        fn prefix_prefill_kv_len_cap(&self) -> Option<usize> {
            self.cap
        }
    }

    let params = AutoDeviceMapParams::default_text();
    let paged = PagedAttentionConfig::new(
        Some(32),
        MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?
    .with_execution_limits(ExecutionMemoryLimits {
        max_num_seqs: 1,
        max_num_batched_tokens: 512,
        max_prefill_chunk_tokens: 512,
    })?;
    let architecture = ActivationModelConfig {
        intermediate_size: 1536,
        vocab_size: 1024,
        attention_features: Some(PrefixPrefillAttentionFeatures::default()),
        ..Default::default()
    };
    let estimate = |cap, capacity| {
        let model = CappedConfig { cap };
        model_activation_memory(
            &ActivationMemoryContext {
                params: &params,
                limits: paged.execution_limits,
                dtype: DType::BF16,
                device: &Device::Cpu,
                paged: &paged,
                cache_capacity_tokens: Some(capacity),
            },
            ModelActivationConfig {
                model: &model,
                architecture,
            },
        )
    };
    let uncapped = estimate(None, 65_536)?;
    for (cap, capacity, dense_capacity) in
        [(1024, 65_536, 1024), (128, 65_536, 512), (4096, 1024, 1024)]
    {
        let capped = estimate(Some(cap), capacity)?;
        assert_eq!(capped, estimate(None, dense_capacity)?);
        assert!(capped.mapped_bytes < uncapped.mapped_bytes);
        assert_eq!(
            capped.mapped_non_attention_bytes,
            uncapped.mapped_non_attention_bytes
        );
    }
    Ok(())
}

#[test]
fn shared_estimator_reserves_prefix_attention_and_logits() -> Result<()> {
    let model = crate::paged_attention::ModelConfigMetadata {
        max_seq_len: 8192,
        num_layers: 2,
        hidden_size: 512,
        num_kv_heads: 2,
        num_attn_heads: 8,
        sliding_window: None,
        k_head_dim: 64,
        v_head_dim: 64,
        kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
    };
    let params = AutoDeviceMapParams::default_text();
    let paged = PagedAttentionConfig::new(
        Some(32),
        MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?;
    let mut context = ActivationMemoryContext {
        params: &params,
        limits: paged.execution_limits,
        dtype: DType::BF16,
        device: &Device::Cpu,
        paged: &paged,
        cache_capacity_tokens: None,
    };
    let architecture = ActivationModelConfig {
        intermediate_size: 1536,
        vocab_size: 1024,
        ..Default::default()
    };
    let baseline = model_activation_memory(
        &context,
        ModelActivationConfig {
            model: &model,
            architecture,
        },
    )?;
    context.cache_capacity_tokens = Some(8192);
    let prefix = model_activation_memory(
        &context,
        ModelActivationConfig {
            model: &model,
            architecture,
        },
    )?;
    assert!(prefix.mapped_bytes > baseline.mapped_bytes);
    assert_eq!(
        prefix.mapped_non_attention_bytes,
        baseline.mapped_non_attention_bytes
    );
    assert_eq!(prefix.non_mapped_bytes, baseline.non_mapped_bytes);
    assert_eq!(prefix.logits_bytes_per_query_token, 1024 * (2 + 4));
    Ok(())
}

fn attention_shape() -> AttentionMemoryShape {
    AttentionMemoryShape {
        batch: 1,
        query_len: 1024,
        kv_len: 8192,
        query_heads: 16,
        kv_heads: 8,
        k_head_dim: 128,
        v_head_dim: 128,
        dtype: DType::BF16,
        backend: AttentionWorkspaceKind::Eager,
        materialized_mask: true,
    }
}

#[test]
fn retained_token_state_is_reserved_at_f32_width() -> Result<()> {
    let model = crate::paged_attention::ModelConfigMetadata {
        max_seq_len: 8192,
        num_layers: 2,
        hidden_size: 512,
        num_kv_heads: 2,
        num_attn_heads: 8,
        sliding_window: None,
        k_head_dim: 64,
        v_head_dim: 64,
        kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
    };
    let params = AutoDeviceMapParams::default_text();
    let paged = PagedAttentionConfig::new(
        Some(32),
        MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?;
    let estimate = |dtype, extra_hidden_elements_per_token| {
        model_activation_memory(
            &ActivationMemoryContext {
                params: &params,
                limits: paged.execution_limits,
                dtype,
                device: &Device::Cpu,
                paged: &paged,
                cache_capacity_tokens: None,
            },
            ModelActivationConfig {
                model: &model,
                architecture: ActivationModelConfig {
                    intermediate_size: 1536,
                    vocab_size: 1024,
                    extra_hidden_elements_per_token,
                    ..Default::default()
                },
            },
        )
    };
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let baseline = estimate(dtype, 0)?;
        let retained = estimate(dtype, 512)?;
        let expected = baseline.planned_query_tokens * 512 * DType::F32.size_in_bytes();
        assert_eq!(
            retained.mapped_non_attention_bytes - baseline.mapped_non_attention_bytes,
            expected
        );
        assert_eq!(
            retained.non_mapped_non_attention_bytes - baseline.non_mapped_non_attention_bytes,
            expected
        );
    }
    assert!(estimate(DType::BF16, usize::MAX).is_err());
    Ok(())
}

#[test]
fn eager_attention_retains_full_keys_and_f32_scores() -> Result<()> {
    let shape = attention_shape();
    let scores = 16 * 1024 * 8192 * (2 * 4 + 2);
    let repeated_kv = 8192 * 16 * (128 + 128) * 2;
    let mask = 1024 * 8192 * (4 + 2);
    let output = 1024 * 16 * 128 * 2;
    assert_eq!(
        attention_workspace_bytes(shape)?,
        scores + repeated_kv + mask + output
    );
    let shorter = AttentionMemoryShape {
        kv_len: 1024,
        ..shape
    };
    assert!(attention_workspace_bytes(shape)? > 7 * attention_workspace_bytes(shorter)?);
    Ok(())
}

#[test]
fn eager_sinks_retains_all_query_scores() -> Result<()> {
    let shape = AttentionMemoryShape {
        query_len: 2 * ATTENTION_CHUNK_SIZE,
        ..attention_shape()
    };
    let sinks = AttentionMemoryShape {
        backend: AttentionWorkspaceKind::EagerSinks,
        ..shape
    };
    assert_eq!(
        attention_workspace_bytes(sinks)? - attention_workspace_bytes(shape)?,
        shape.batch
            * shape.query_heads
            * ATTENTION_CHUNK_SIZE
            * shape.kv_len
            * (EAGER_F32_SCORE_BUFFERS * DType::F32.size_in_bytes() + shape.dtype.size_in_bytes())
    );
    Ok(())
}

#[test]
fn fused_workspace_does_not_reserve_attention_scores() -> Result<()> {
    let eager = attention_shape();
    let fused = AttentionMemoryShape {
        backend: AttentionWorkspaceKind::Fused,
        materialized_mask: false,
        ..eager
    };
    assert!(attention_workspace_bytes(fused)? < attention_workspace_bytes(eager)? / 8);
    Ok(())
}

#[test]
fn large_gqa_prefill_reserves_expanded_kv_instead_of_eager_scores() -> Result<()> {
    let shape = AttentionMemoryShape {
        batch: 1,
        query_len: 4096,
        kv_len: 4096,
        query_heads: 32,
        kv_heads: 2,
        k_head_dim: 256,
        v_head_dim: 256,
        dtype: DType::BF16,
        backend: AttentionWorkspaceKind::Eager,
        materialized_mask: false,
    };
    let features = PrefixPrefillAttentionFeatures::default();
    let backend = cuda_prefill_backend(&shape, features, false, true, false).unwrap();
    assert_eq!(backend, AttentionWorkspaceKind::FusedExpandedKv);
    let fused = attention_workspace_bytes(AttentionMemoryShape { backend, ..shape })?;
    assert_eq!(
        fused,
        4096 * 2 * (32 * 256 + 2 * 512 + 32 * 512 + 32 * 256) + 4096 * 32 * 4
    );
    assert!(fused < attention_workspace_bytes(shape)? / 4);
    assert!(cuda_prefill_backend(&shape, features, true, true, false).is_none());
    assert!(cuda_prefill_backend(&shape, features, false, false, false).is_none());
    Ok(())
}

#[test]
fn cuda_sinks_prefill_uses_fused_workspace_without_flash_attention() -> Result<()> {
    let features = PrefixPrefillAttentionFeatures {
        has_sinks: true,
        has_sliding_window: true,
        ..Default::default()
    };
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let shape = AttentionMemoryShape {
            dtype,
            materialized_mask: false,
            ..attention_shape()
        };
        let backend = cuda_prefill_backend(&shape, features, false, false, true).unwrap();
        assert_eq!(backend, AttentionWorkspaceKind::Fused);
        assert!(
            attention_workspace_bytes(AttentionMemoryShape { backend, ..shape })?
                < attention_workspace_bytes(shape)? / 8
        );
        assert!(cuda_prefill_backend(&shape, features, true, true, true).is_none());
        assert!(cuda_prefill_backend(&shape, features, false, true, false).is_none());
        for unsupported in [
            PrefixPrefillAttentionFeatures {
                has_alibi: true,
                ..features
            },
            PrefixPrefillAttentionFeatures {
                has_softcap: true,
                ..features
            },
        ] {
            assert!(cuda_prefill_backend(&shape, unsupported, false, true, true).is_none());
        }
        assert!(cuda_prefill_backend(
            &AttentionMemoryShape {
                v_head_dim: 256,
                ..shape
            },
            features,
            false,
            true,
            true,
        )
        .is_none());
    }
    Ok(())
}

#[test]
fn attention_workspace_rejects_size_overflow() {
    assert!(attention_workspace_bytes(AttentionMemoryShape {
        batch: usize::MAX,
        ..attention_shape()
    })
    .is_err());
}

#[test]
fn metal_masks_count_distinct_full_and_window_variants() -> Result<()> {
    assert_eq!(additional_metal_mask_bytes(2, 16, 32, DType::BF16, 1)?, 0);
    assert_eq!(
        additional_metal_mask_bytes(2, 16, 32, DType::BF16, 2)?,
        2 * 16 * 32 * (4 + 2) * METAL_BUFFER_ROUNDING_BOUND
    );
    assert!(additional_metal_mask_bytes(usize::MAX, 2, 2, DType::BF16, 2).is_err());
    Ok(())
}

#[test]
fn idle_prefill_uses_batch_budget_not_latency_chunk_target() -> Result<()> {
    let params = AutoDeviceMapParams::default_text();
    let paged = PagedAttentionConfig::new(
        Some(32),
        crate::MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?
    .with_execution_limits(ExecutionMemoryLimits {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
        max_prefill_chunk_tokens: 512,
    })?;
    let context = ActivationMemoryContext {
        params: &params,
        limits: paged.execution_limits,
        dtype: DType::BF16,
        device: &Device::Cpu,
        paged: &paged,
        cache_capacity_tokens: None,
    };
    assert_eq!(context.query_tokens(8192)?, 4096);
    assert_eq!(context.query_tokens(128)?, 512);
    Ok(())
}

#[test]
fn qwen_reservations_follow_execution_limits_and_include_logits() -> Result<()> {
    let config = crate::models::qwen3::Config {
        vocab_size: 1024,
        hidden_size: 512,
        intermediate_size: 1536,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        max_position_embeddings: 8192,
        head_dim: Some(128),
        ..Default::default()
    };
    let config = serde_json::to_string(&config)?;
    let metadata = super::super::Qwen3Loader.model_config(&config)?;
    assert_eq!(metadata.k_head_dim(), 128);
    assert_eq!(metadata.v_head_dim(), 128);
    let mut paged = PagedAttentionConfig::new(
        Some(32),
        crate::MemoryGoal::Auto,
        crate::paged_attention::PagedCacheType::Auto,
    )?
    .with_execution_limits(ExecutionMemoryLimits {
        max_num_seqs: 4,
        max_num_batched_tokens: 1024,
        max_prefill_chunk_tokens: 128,
    })?;
    let first = reserve_model_activation_memory(
        &super::super::Qwen3Loader,
        &config,
        &mut paged,
        DType::BF16,
        &Device::Cpu,
        &AutoDeviceMapParams::Text {
            max_seq_len: 4,
            max_batch_size: 999,
        },
    )?;
    let second = reserve_model_activation_memory(
        &super::super::Qwen3Loader,
        &config,
        &mut paged,
        DType::BF16,
        &Device::Cpu,
        &AutoDeviceMapParams::default_text(),
    )?;
    assert_eq!(first, second);
    assert_eq!(first.planned_query_tokens, 1024);
    assert_eq!(paged.planned_query_tokens, 1024);
    assert_eq!(first.logits_bytes_per_query_token, 1024 * (2 + 4));
    assert!(first.mapped_non_attention_bytes > 0);
    assert!(first.non_mapped_non_attention_bytes > 0);
    assert!(first.mapped_bytes > first.mapped_non_attention_bytes);
    assert_eq!(
        paged.mapped_activation_memory_reservation_bytes,
        first.mapped_bytes
    );
    Ok(())
}

#[test]
fn prefix_reservation_follows_capacity_without_inflating_auto_placement() -> Result<()> {
    let config = serde_json::to_string(&crate::models::qwen3::Config {
        vocab_size: 1024,
        hidden_size: 512,
        intermediate_size: 1536,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        max_position_embeddings: 8192,
        head_dim: Some(128),
        ..Default::default()
    })?;
    let params = AutoDeviceMapParams::default_text();
    let estimate = |goal, capacity| -> Result<_> {
        let paged = PagedAttentionConfig::new(
            Some(32),
            goal,
            crate::paged_attention::PagedCacheType::Auto,
        )?
        .with_execution_limits(ExecutionMemoryLimits {
            max_num_seqs: 4,
            max_num_batched_tokens: 512,
            max_prefill_chunk_tokens: 128,
        })?;
        estimate_model_activation_memory(
            &super::super::Qwen3Loader,
            &config,
            ActivationMemoryContext {
                params: &params,
                limits: paged.execution_limits,
                dtype: DType::BF16,
                device: &Device::Cpu,
                paged: &paged,
                cache_capacity_tokens: capacity,
            },
        )
    };
    let baseline = estimate(MemoryGoal::Auto, None)?;
    let explicit = estimate(MemoryGoal::KvTokens(8192), None)?;
    let resolved = estimate(MemoryGoal::Auto, Some(8192))?;
    assert_eq!(explicit, resolved);
    assert_eq!(baseline.planned_query_tokens, resolved.planned_query_tokens);
    assert_eq!(
        baseline.mapped_non_attention_bytes,
        resolved.mapped_non_attention_bytes
    );
    assert!(resolved.mapped_bytes > baseline.mapped_bytes);
    let mut previous = baseline.mapped_bytes;
    for capacity in [32, 64, 512, 1024, 2048, 8192] {
        let next = estimate(MemoryGoal::Auto, Some(capacity))?.mapped_bytes;
        assert!(
            next >= previous,
            "capacity {capacity} reduced the reservation"
        );
        previous = next;
    }
    assert_eq!(estimate(MemoryGoal::Auto, Some(16384))?, resolved);
    assert!(estimate(MemoryGoal::Auto, Some(0)).is_err());
    Ok(())
}
