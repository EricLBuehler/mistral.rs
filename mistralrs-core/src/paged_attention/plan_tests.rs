use super::*;
use crate::paged_attention::{
    HybridPagedKvCacheConfig, KvCacheLayout, KvCacheTopology, ModelConfigMetadata,
};

struct DonorWorkspaceModel<const HEAD_DIM: usize>;

impl<const HEAD_DIM: usize> ModelConfigLike for DonorWorkspaceModel<HEAD_DIM> {
    fn max_seq_len(&self) -> usize {
        131_072
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn hidden_size(&self) -> usize {
        4096
    }

    fn num_kv_heads(&self) -> usize {
        2
    }

    fn num_attn_heads(&self) -> usize {
        8
    }

    fn k_head_dim(&self) -> usize {
        HEAD_DIM
    }

    fn v_head_dim(&self) -> usize {
        HEAD_DIM
    }

    fn has_kv_cache_sharing(&self) -> bool {
        true
    }

    fn kv_cache_topology(&self) -> KvCacheTopology {
        KvCacheTopology::from_layer_owners(vec![0, 0])
    }

    fn prefix_prefill_attention_features(
        &self,
        _layer_idx: usize,
    ) -> Option<PrefixPrefillAttentionFeatures> {
        Some(PrefixPrefillAttentionFeatures::default())
    }
}

fn workspace_model(features: Option<PrefixPrefillAttentionFeatures>) -> HybridPagedKvCacheConfig {
    workspace_model_with_window(features, None)
}

fn workspace_model_with_window(
    features: Option<PrefixPrefillAttentionFeatures>,
    sliding_window: Option<usize>,
) -> HybridPagedKvCacheConfig {
    let model = HybridPagedKvCacheConfig::new(
        ModelConfigMetadata {
            max_seq_len: 131_072,
            num_layers: 4,
            hidden_size: 4096,
            num_kv_heads: 4,
            num_attn_heads: 16,
            sliding_window,
            k_head_dim: 256,
            v_head_dim: 256,
            kv_cache_layout: KvCacheLayout::FlashInferHnd,
        },
        vec![true, false, true, false],
    );
    match features {
        Some(features) => model.with_uniform_prefix_prefill_attention_features(features),
        None => model,
    }
}

fn workspace_input<'a>(
    query_lens: &'a [usize],
    full_context_lens: &'a [usize],
) -> PromptPrefillWorkspaceInput<'a> {
    PromptPrefillWorkspaceInput {
        activation_dtype: DType::BF16,
        cache_dtype: DType::F8E4M3,
        device_is_cuda: true,
        device_is_metal: false,
        block_size: 32,
        query_lens,
        full_context_lens,
        max_pages_per_sequence: full_context_lens
            .iter()
            .copied()
            .max()
            .unwrap_or_default()
            .div_ceil(32),
        requires_prefix_attention: true,
        is_causal: true,
        causality_known: true,
        has_custom_mask: false,
        has_noncausal_mm_context: false,
        has_sliding_window: false,
        fa3_num_sm_by_layer: &[Some(132), None, Some(132), None],
        hnd_by_layer: &[],
    }
}

fn prefix_plan(head_size: usize, has_softcap: bool, has_sliding_window: bool) -> PrefixPrefillPlan {
    PrefixPrefillPlan::choose(PrefixPrefillPlanInput {
        device_is_cuda: true,
        dtype: DType::F16,
        cache_dtype: DType::F16,
        has_alibi: false,
        has_sinks: false,
        has_custom_mask: false,
        causality_known: true,
        head_size,
        has_softcap,
        has_sliding_window,
        query_layout_is_dense: true,
        query_len: 64,
        q_heads: 24,
        kv_heads: 4,
        writes_cache: true,
        is_causal: true,
        has_noncausal_mm_context: false,
        fa3_supported: true,
        block_size: 32,
        attention_backend: AttentionBackendKind::FlashInfer,
    })
}

#[test]
fn paged_prefix_rejects_disabled_large_head_features() {
    assert!(matches!(
        prefix_plan(320, true, false),
        PrefixPrefillPlan::GatherSdpa
    ));
    assert!(matches!(
        prefix_plan(320, false, true),
        PrefixPrefillPlan::GatherSdpa
    ));
}

#[test]
fn paged_prefix_gathers_mixed_dtype_cache() {
    let plan = PrefixPrefillPlan::choose(PrefixPrefillPlanInput {
        device_is_cuda: true,
        dtype: DType::BF16,
        cache_dtype: DType::F8E4M3,
        has_alibi: false,
        has_sinks: false,
        has_custom_mask: false,
        causality_known: true,
        head_size: 128,
        has_softcap: false,
        has_sliding_window: false,
        query_layout_is_dense: true,
        query_len: 64,
        q_heads: 24,
        kv_heads: 4,
        writes_cache: true,
        is_causal: true,
        has_noncausal_mm_context: false,
        fa3_supported: false,
        block_size: 32,
        attention_backend: AttentionBackendKind::FlashInfer,
    });
    assert!(matches!(plan, PrefixPrefillPlan::GatherSdpa));
}

#[cfg(cuda_unix)]
#[test]
fn fp8_dense_short_prefix_uses_fa3_when_available() {
    let input = PrefixPrefillPlanInput {
        device_is_cuda: true,
        dtype: DType::BF16,
        cache_dtype: DType::F8E4M3,
        has_alibi: false,
        has_sinks: false,
        has_custom_mask: false,
        causality_known: true,
        head_size: 256,
        has_softcap: false,
        has_sliding_window: false,
        query_layout_is_dense: true,
        query_len: 128,
        q_heads: 24,
        kv_heads: 4,
        writes_cache: true,
        is_causal: true,
        has_noncausal_mm_context: false,
        fa3_supported: true,
        block_size: 32,
        attention_backend: AttentionBackendKind::FlashInfer,
    };
    assert!(fa3_paged_prefill_supported(input));
    assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
        query_len: 129,
        ..input
    }));
    assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
        query_layout_is_dense: false,
        ..input
    }));
    assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
        writes_cache: false,
        ..input
    }));
    assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
        has_noncausal_mm_context: true,
        ..input
    }));
    assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
        q_heads: 20,
        ..input
    }));
    let plan = PrefixPrefillPlan::choose(input);
    if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
        assert!(matches!(plan, PrefixPrefillPlan::Fa3Fp8Paged));
    } else {
        assert!(matches!(plan, PrefixPrefillPlan::GatherSdpa));
    }
}

#[test]
fn prefix_gather_workspace_is_checked_and_exact() {
    let packed = GatherPrefillWorkspaceInput {
        batch: 16,
        total_q: 2_048,
        max_q: 128,
        total_kv: 1_600_000,
        max_kv: 100_000,
        q_heads: 16,
        kv_heads: 4,
        k_head_dim: 256,
        v_head_dim: 256,
        dtype: DType::BF16,
        backend: GatherAttentionBackend::PackedVarlen,
    };
    assert_eq!(
        gather_prefill_workspace_bytes(packed).unwrap(),
        6_604_062_720
    );

    let mixed_fallback = GatherPrefillWorkspaceInput {
        total_kv: 115_000,
        backend: GatherAttentionBackend::Conservative,
        ..packed
    };
    assert_eq!(
        gather_prefill_workspace_bytes(mixed_fallback).unwrap(),
        66_494_857_216
    );
    assert!(gather_prefill_workspace_bytes(GatherPrefillWorkspaceInput {
        total_kv: usize::MAX,
        ..packed
    })
    .is_err());
}

#[test]
fn single_token_cuda_prefix_uses_the_same_fused_path_as_longer_queries() -> Result<()> {
    let input = PrefixPrefillPlanInput {
        device_is_cuda: true,
        dtype: DType::BF16,
        cache_dtype: DType::F8E4M3,
        has_alibi: false,
        has_sinks: false,
        has_custom_mask: false,
        causality_known: true,
        head_size: 256,
        has_softcap: false,
        has_sliding_window: false,
        query_layout_is_dense: true,
        query_len: 1,
        q_heads: 24,
        kv_heads: 4,
        writes_cache: true,
        is_causal: false,
        has_noncausal_mm_context: false,
        fa3_supported: false,
        block_size: 32,
        attention_backend: AttentionBackendKind::FlashInfer,
    };
    assert!(gather_prefill_uses_packed_varlen_with_backend(
        input, true, true
    ));
    assert!(!gather_prefill_uses_packed_varlen_with_backend(
        input, false, true
    ));
    assert!(!gather_prefill_uses_packed_varlen_with_backend(
        input, true, false
    ));
    for rejected in [
        PrefixPrefillPlanInput {
            has_custom_mask: true,
            ..input
        },
        PrefixPrefillPlanInput {
            has_noncausal_mm_context: true,
            ..input
        },
        PrefixPrefillPlanInput {
            query_len: 2,
            ..input
        },
        PrefixPrefillPlanInput {
            has_sliding_window: true,
            ..input
        },
    ] {
        assert!(!gather_prefill_uses_packed_varlen_with_backend(
            rejected, true, true
        ));
    }
    let shape = GatherPrefillWorkspaceInput {
        batch: 1,
        total_q: 1,
        max_q: 1,
        total_kv: 451_008,
        max_kv: 451_008,
        q_heads: input.q_heads,
        kv_heads: input.kv_heads,
        k_head_dim: input.head_size,
        v_head_dim: input.head_size,
        dtype: input.dtype,
        backend: GatherAttentionBackend::PackedVarlen,
    };
    let fused = gather_prefill_workspace_bytes(shape)?;
    let eager = gather_prefill_workspace_bytes(GatherPrefillWorkspaceInput {
        backend: GatherAttentionBackend::Conservative,
        ..shape
    })?;
    assert!(fused < eager / 10);
    assert_eq!(fused, 451_008 * 4 * 256 * 2 * 2 + 3 * 24 * 256 * 2 + 24 * 4);
    Ok(())
}

#[test]
fn cold_cuda_prefill_reports_attention_workspace() -> Result<()> {
    let model = workspace_model(Some(PrefixPrefillAttentionFeatures::default()));
    let result = prompt_prefill_workspace(
        Some(&model),
        PromptPrefillWorkspaceInput {
            requires_prefix_attention: false,
            fa3_num_sm_by_layer: &[],
            ..workspace_input(&[512], &[512])
        },
    )?;
    assert!(result.bytes > 0);
    assert_eq!(result.attention_is_complete, crate::using_flash_attn());
    Ok(())
}

fn metal_gather_request<'a>(
    query_lens: &'a [usize],
    kv_lens: &'a [usize],
    head_dim: usize,
) -> GatherPrefillWorkspaceRequest<'a> {
    GatherPrefillWorkspaceRequest {
        device_is_metal: true,
        packed_queries: false,
        metal_mask_is_compatible: true,
        query_lens,
        kv_lens,
        q_heads: 8,
        kv_heads: 2,
        k_head_dim: head_dim,
        v_head_dim: head_dim,
        dtype: DType::BF16,
        plan_input: PrefixPrefillPlanInput {
            device_is_cuda: false,
            dtype: DType::BF16,
            cache_dtype: DType::BF16,
            has_alibi: false,
            has_sinks: false,
            has_custom_mask: false,
            causality_known: true,
            head_size: head_dim,
            has_softcap: false,
            has_sliding_window: false,
            query_layout_is_dense: true,
            query_len: query_lens.iter().copied().max().unwrap(),
            q_heads: 8,
            kv_heads: 2,
            writes_cache: true,
            is_causal: true,
            has_noncausal_mm_context: false,
            fa3_supported: false,
            block_size: 32,
            attention_backend: AttentionBackendKind::Standard,
        },
    }
}

#[test]
fn metal_fused_gather_reserves_kv_and_masks_without_eager_scores() -> Result<()> {
    let request = metal_gather_request(&[64, 32], &[512, 256], 128);
    assert_eq!(
        gather_attention_backend(request),
        GatherAttentionBackend::MetalFused
    );
    let bytes = gather_prefill_workspace_for_lengths(request)?;
    let gathered_kv = (512 + 256) * 2 * 128 * 2 * 2;
    let padded_kv = 2 * 512 * 2 * 128 * 2 * 2;
    let output = 2 * 64 * 8 * 128 * 2;
    let mask = 2 * 64 * 512 * (4 + 2);
    assert_eq!(bytes, gathered_kv + 3 * padded_kv + 3 * output + mask);
    let twice_heads = GatherPrefillWorkspaceRequest {
        q_heads: 16,
        ..request
    };
    assert_eq!(
        gather_prefill_workspace_for_lengths(twice_heads)? - bytes,
        3 * output
    );
    let eager = GatherPrefillWorkspaceRequest {
        metal_mask_is_compatible: false,
        ..request
    };
    assert!(gather_prefill_workspace_for_lengths(eager)? > bytes);
    for rejected in [
        GatherPrefillWorkspaceRequest {
            packed_queries: true,
            ..request
        },
        GatherPrefillWorkspaceRequest {
            query_lens: &[1, 1],
            ..request
        },
        GatherPrefillWorkspaceRequest {
            v_head_dim: 256,
            ..request
        },
        GatherPrefillWorkspaceRequest {
            device_is_metal: false,
            ..request
        },
    ] {
        assert_eq!(
            gather_attention_backend(rejected),
            GatherAttentionBackend::Conservative
        );
    }
    assert_eq!(
        gather_attention_backend(GatherPrefillWorkspaceRequest {
            plan_input: PrefixPrefillPlanInput {
                has_sinks: true,
                ..request.plan_input
            },
            ..request
        }),
        GatherAttentionBackend::Sinks
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_d512_gather_covers_extension_scratch_and_aligned_capacity_boundaries() -> Result<()> {
    use mistralrs_quant::metal_kernels::{flash_attn_ext_blk_scratch_size, FA_NCPSG};

    let request = metal_gather_request(&[16], &[65], 512);
    assert_eq!(
        gather_attention_backend(request),
        GatherAttentionBackend::MetalD512
    );
    let gathered_kv = 65 * 2 * 512 * 2 * 2;
    let output = 16 * 8 * 512 * 2;
    let masks = 16 * 65 * (4 + 2);
    let f16_mask = 16 * 65 * 2;
    let tail = 2 * 512 * 2 * FA_NCPSG * 2 + 2 * FA_NCPSG * 16;
    let blocks = flash_attn_ext_blk_scratch_size(16, 65, 1, 1);
    assert_eq!(
        gather_prefill_workspace_for_lengths(request)?,
        4 * gathered_kv + 3 * output + masks + f16_mask + tail + blocks
    );
    let mut previous = 0;
    for context in 63..=129 {
        let bytes = gather_prefill_workspace_for_lengths(GatherPrefillWorkspaceRequest {
            kv_lens: &[context],
            ..request
        })?;
        assert!(bytes >= previous);
        previous = bytes;
    }
    for dtype in [DType::F16, DType::F32] {
        assert_eq!(
            gather_attention_backend(GatherPrefillWorkspaceRequest { dtype, ..request }),
            GatherAttentionBackend::Conservative
        );
    }
    Ok(())
}

#[test]
fn mixed_short_query_gather_covers_value_transpose_peak() {
    let expected = [
        (1, 46_467_811_072usize),
        (8, 47_318_808_576),
        (32, 50_236_514_304),
        (64, 54_126_788_608),
        (128, 66_494_857_216),
    ];
    for (query_len, expected_bytes) in expected {
        let input = GatherPrefillWorkspaceInput {
            batch: 16,
            total_q: 16 * query_len,
            max_q: query_len,
            total_kv: 115_000,
            max_kv: 100_000,
            q_heads: 16,
            kv_heads: 4,
            k_head_dim: 256,
            v_head_dim: 256,
            dtype: DType::BF16,
            backend: GatherAttentionBackend::Conservative,
        };
        assert_eq!(
            gather_prefill_workspace_bytes(input).unwrap(),
            expected_bytes
        );
    }

    assert_eq!(
        gather_prefill_workspace_bytes(GatherPrefillWorkspaceInput {
            batch: 16,
            total_q: 16,
            max_q: 1,
            total_kv: 115_000,
            max_kv: 100_000,
            q_heads: 16,
            kv_heads: 16,
            k_head_dim: 256,
            v_head_dim: 256,
            dtype: DType::BF16,
            backend: GatherAttentionBackend::Conservative,
        })
        .unwrap(),
        41_327_331_072
    );
}

#[test]
fn sinks_gather_reserves_unchunked_scores_and_contiguous_qkv() -> Result<()> {
    const QUERY_LEN: usize = 2 * crate::attention::ATTENTION_CHUNK_SIZE;
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    let shape = GatherPrefillWorkspaceInput {
        batch: 1,
        total_q: QUERY_LEN,
        max_q: QUERY_LEN,
        total_kv: QUERY_LEN,
        max_kv: QUERY_LEN,
        q_heads: HEADS,
        kv_heads: HEADS,
        k_head_dim: HEAD_DIM,
        v_head_dim: HEAD_DIM,
        dtype: DType::BF16,
        backend: GatherAttentionBackend::Sinks,
    };
    let score_growth = HEADS
        * QUERY_LEN
        * (QUERY_LEN / 2)
        * (2 * DType::F32.size_in_bytes() + DType::BF16.size_in_bytes());
    let contiguous_qkv = 3 * QUERY_LEN * HEADS * HEAD_DIM * DType::BF16.size_in_bytes();
    assert_eq!(
        gather_prefill_workspace_bytes(shape)?
            - gather_prefill_workspace_bytes(GatherPrefillWorkspaceInput {
                backend: GatherAttentionBackend::Conservative,
                ..shape
            })?,
        score_growth + contiguous_qkv
    );
    Ok(())
}

#[test]
fn cuda_sinks_workspace_is_complete_for_cold_and_cached_prompts() -> Result<()> {
    let model = workspace_model_with_window(
        Some(PrefixPrefillAttentionFeatures {
            has_sinks: true,
            has_sliding_window: true,
            ..Default::default()
        }),
        Some(513),
    );
    for (query, context) in [(16, 16), (512, 1024), (4096, 32_768)] {
        let query_lens = [query];
        let context_lens = [context];
        let input = PromptPrefillWorkspaceInput {
            requires_prefix_attention: context > query,
            cache_dtype: DType::BF16,
            ..workspace_input(&query_lens, &context_lens)
        };
        let workspace = prompt_prefill_workspace(Some(&model), input)?;
        assert!(workspace.attention_is_complete);
        assert!(workspace.bytes > 0);
        let masked = prompt_prefill_workspace(
            Some(&model),
            PromptPrefillWorkspaceInput {
                has_custom_mask: true,
                ..input
            },
        )?;
        assert!(!masked.attention_is_complete);
    }
    Ok(())
}

#[test]
fn cold_prompt_workspace_preserves_unknown_attention_fallback() {
    let model = workspace_model(None);
    let query_lens = [128, 128];
    let context_lens = [128, 128];
    let mut input = workspace_input(&query_lens, &context_lens);
    input.requires_prefix_attention = false;
    let workspace = prompt_prefill_workspace(Some(&model), input).unwrap();
    assert!(workspace.bytes > 0);
    assert!(!workspace.attention_is_complete);
    assert_eq!(
        prompt_prefill_workspace(None, input).unwrap(),
        PromptPrefillWorkspace::default()
    );
}

#[test]
fn donor_cache_first_prompt_requires_gather_workspace() {
    let model = DonorWorkspaceModel::<256>;
    assert!(model_has_donor_paged_cache_layers(&model));
    let query_lens = [128];
    let context_lens = [128];
    let workspace =
        prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
            .unwrap();
    assert!(workspace.gather_workspace_bytes > 0);
    assert!(workspace.bytes >= workspace.gather_workspace_bytes);
}

#[cfg(feature = "metal")]
#[test]
fn donor_reduced_prompt_reserves_decode_workspace() -> Result<()> {
    let model = DonorWorkspaceModel::<512>;
    let query_lens = [15];
    let context_lens = [7087];
    let input = PromptPrefillWorkspaceInput {
        device_is_cuda: false,
        device_is_metal: true,
        cache_dtype: DType::BF16,
        hnd_by_layer: &[true, true],
        fa3_num_sm_by_layer: &[],
        ..workspace_input(&query_lens, &context_lens)
    };
    let native = metal_paged_prefill_workspace(MetalPagedPrefillWorkspaceInput {
        query_lens: &query_lens,
        context_lens: &context_lens,
        query_heads: 8,
        kv_heads: 2,
        key_head_dim: 512,
        value_head_dim: 512,
        block_size: 32,
        query_dtype: DType::BF16,
        cache_dtype: DType::BF16,
        hnd: true,
        has_custom_mask: false,
        has_multimodal_ranges: false,
    })?
    .unwrap();
    let packing = metal_cache_write_workspace(Some(&model), query_lens[0], DType::BF16)?;
    let donor_decode = mistralrs_paged_attn::MetalPagedAttentionWorkspace {
        num_seqs: 1,
        num_heads: 8,
        num_kv_heads: 2,
        head_size: 512,
        block_size: 32,
        max_context_len: context_lens[0],
        sliding_window: None,
        dtype: DType::BF16,
        cache_dtype: DType::BF16,
        hnd: true,
        packed_x: 1,
        has_alibi: false,
        has_sinks: false,
        has_scales: false,
        softcapping: 1.0,
    }
    .bytes()?;
    assert_eq!(native + packing, 532_480);
    assert_eq!(donor_decode, 536_576);
    let planned = prompt_prefill_workspace(Some(&model), input)?;
    assert!(planned.bytes >= donor_decode);
    assert_eq!(planned.gather_workspace_bytes, 0);
    assert!(planned.attention_is_complete);
    Ok(())
}

#[test]
fn prompt_workspace_rejects_prefix_without_model_metadata() {
    let query_lens = [128, 128];
    let context_lens = [1_000, 8_000];
    assert!(prompt_prefill_workspace(None, workspace_input(&query_lens, &context_lens)).is_err());
}

#[test]
fn prompt_workspace_is_conservative_for_unknown_attention_features() {
    let model = workspace_model(None);
    let query_lens = [128, 128];
    let context_lens = [1_000, 8_000];
    assert_eq!(
        prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
            .unwrap()
            .bytes,
        739_889_152
    );

    let model = workspace_model(Some(PrefixPrefillAttentionFeatures {
        has_sliding_window: true,
        ..Default::default()
    }));
    assert_eq!(
        prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
            .unwrap()
            .bytes,
        739_889_152
    );
}

#[cfg(feature = "metal")]
#[test]
fn metal_native_prefill_reserves_actual_outputs_without_gathers_or_masks() -> Result<()> {
    let model = workspace_model(None);
    let query_lens = [64, 32];
    let contexts = [512, 256];
    let mut input = workspace_input(&query_lens, &contexts);
    input.device_is_cuda = false;
    input.device_is_metal = true;
    input.fa3_num_sm_by_layer = &[];
    let prefix = prompt_prefill_workspace(Some(&model), input)?;
    assert!(prefix.attention_is_complete);
    assert!(prefix.bytes > 0);
    assert_eq!(prefix.gather_workspace_bytes, 0);
    let long_contexts = [65_536, 16_384];
    let long = prompt_prefill_workspace(
        Some(&model),
        PromptPrefillWorkspaceInput {
            full_context_lens: &long_contexts,
            hnd_by_layer: &[true; 4],
            ..input
        },
    )?;
    assert_eq!(long, prefix);
    let cold = prompt_prefill_workspace(
        Some(&model),
        PromptPrefillWorkspaceInput {
            requires_prefix_attention: false,
            full_context_lens: &query_lens,
            ..input
        },
    )?;
    assert_eq!(cold, prefix);
    let features = PrefixPrefillAttentionFeatures {
        has_alibi: true,
        has_sinks: true,
        has_softcap: true,
        has_sliding_window: true,
    };
    let featured = workspace_model(Some(features));
    assert_eq!(prompt_prefill_workspace(Some(&featured), input)?, prefix);
    let masked = prompt_prefill_workspace(
        Some(&model),
        PromptPrefillWorkspaceInput {
            has_custom_mask: true,
            ..input
        },
    )?;
    assert!(masked.attention_is_complete);
    assert!(masked.bytes > prefix.bytes);
    assert_eq!(masked.gather_workspace_bytes, 0);
    let unsupported = prompt_prefill_workspace(
        Some(&model),
        PromptPrefillWorkspaceInput {
            requires_prefix_attention: false,
            activation_dtype: DType::F64,
            ..input
        },
    )?;
    assert!(!unsupported.attention_is_complete);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_native_prefill_capacity_bound_is_monotonic_with_compact_ranges() -> Result<()> {
    let query_lens = [1];
    let mut previous = 0;
    for context in [1, 31, 32, 33, 511, 512, 513, 8192, 65_536] {
        let bytes = metal_paged_prefill_workspace(MetalPagedPrefillWorkspaceInput {
            query_lens: &query_lens,
            context_lens: &[context],
            query_heads: 12,
            kv_heads: 4,
            key_head_dim: 96,
            value_head_dim: 192,
            block_size: 32,
            query_dtype: DType::BF16,
            cache_dtype: DType::F8E4M3,
            hnd: true,
            has_custom_mask: false,
            has_multimodal_ranges: true,
        })?
        .unwrap();
        assert!(bytes >= previous);
        previous = bytes;
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_fallback_workspace_retains_distinct_full_and_window_masks() -> Result<()> {
    let full = PrefixPrefillAttentionFeatures::default();
    let window = PrefixPrefillAttentionFeatures {
        has_sliding_window: true,
        ..full
    };
    let model = workspace_model_with_window(Some(full), Some(512));
    let mut input = workspace_input(&[64, 32], &[512, 256]);
    input.device_is_cuda = false;
    input.device_is_metal = true;
    input.activation_dtype = DType::F64;
    input.has_sliding_window = true;
    input.fa3_num_sm_by_layer = &[];
    let single = prompt_prefill_workspace(Some(&model), input)?;
    let mixed = model.with_prefix_prefill_attention_features(vec![full, full, window, window]);
    let both = prompt_prefill_workspace(Some(&mixed), input)?;
    let mask_elements: usize = (64 + 32) * (512 + 256);
    assert_eq!(
        both.bytes - single.bytes,
        mask_elements
            * (DType::F32.size_in_bytes() + DType::F64.size_in_bytes())
            * METAL_BACKING_ROUNDING_BOUND
    );
    assert_eq!(both.gather_workspace_bytes, single.gather_workspace_bytes);
    let repeated = mixed.with_prefix_prefill_attention_features(vec![window; 4]);
    assert_eq!(
        prompt_prefill_workspace(Some(&repeated), input)?.bytes,
        single.bytes
    );
    let unknown_window = workspace_model(Some(window));
    assert_eq!(
        prompt_prefill_workspace(Some(&unknown_window), input)?.bytes,
        both.bytes
    );
    Ok(())
}

#[cfg(cuda_flash_attn)]
#[test]
fn prompt_workspace_falls_back_for_ineligible_known_attention() {
    let model = workspace_model(Some(PrefixPrefillAttentionFeatures::default()));
    let query_lens = [129, 129];
    let context_lens = [1_000, 8_000];
    assert_eq!(
        prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
            .unwrap()
            .bytes,
        36_864_000 + 3 * 258 * 16 * 256 * 2 + 258 * 16 * 4
    );

    let model = workspace_model(Some(PrefixPrefillAttentionFeatures {
        has_sinks: true,
        ..Default::default()
    }));
    let query_lens = [128, 128];
    let sinks = prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
        .unwrap();
    assert!(sinks.attention_is_complete);
    assert_eq!(
        sinks.bytes,
        2 * 8000 * 4 * 256 * 2 * 2 + 3 * 256 * 16 * 256 * 2 + 16 * 4 + 2 * 3 * 4
    );
}

#[cfg(cuda_flash_attn)]
#[test]
fn prompt_workspace_matches_direct_fa3_availability() {
    let model = workspace_model(Some(PrefixPrefillAttentionFeatures::default()));
    let query_lens = [128, 128];
    let context_lens = [1_000, 8_000];
    let workspace =
        prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
            .unwrap();
    assert_eq!(
        workspace.bytes,
        if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
            crate::flashinfer::fa3_prefill_workspace_bytes(2, 128, 16, 4, 256, 250, 132).unwrap()
        } else {
            36_864_000 + 3 * 256 * 16 * 256 * 2 + 256 * 16 * 4
        }
    );
    assert_eq!(
        workspace.gather_workspace_bytes,
        if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
            0
        } else {
            workspace.bytes
        }
    );
}

#[cfg(cuda_unix)]
#[test]
fn prompt_workspace_tracks_fa3_support_per_layer() {
    let model = workspace_model(Some(PrefixPrefillAttentionFeatures::default()));
    let query_lens = [128];
    let context_lens = [256];
    let mut input = workspace_input(&query_lens, &context_lens);
    input.device_is_cuda = false;
    input.fa3_num_sm_by_layer = &[Some(132), None, None, None];
    input.max_pages_per_sequence = 3_125;
    let workspace = prompt_prefill_workspace(Some(&model), input).unwrap();
    let expected_bytes = if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
        let fa3 =
            crate::flashinfer::fa3_prefill_workspace_components(1, 128, 16, 4, 256, 3_125, 132)
                .unwrap();
        fa3.pool().bytes().unwrap() + fa3.transient_bytes().max(12_648_448)
    } else {
        12_648_448
    };
    let gather_bytes = 12_648_448;
    assert_eq!(workspace.bytes, expected_bytes);
    assert_eq!(workspace.gather_workspace_bytes, gather_bytes);
}

#[cfg(not(cuda_unix))]
#[test]
fn standard_sliding_decode_uses_exact_gather_path() {
    assert!(DecodePlan::requires_host_context_lengths(
        AttentionBackendKind::Standard,
        128,
        true,
    ));
    let plan = DecodePlan::choose(DecodePlanInput {
        attention_backend: AttentionBackendKind::Standard,
        head_size: 128,
        has_alibi: false,
        has_sinks: false,
        has_sliding_window: true,
    })
    .unwrap();

    assert!(matches!(plan, DecodePlan::GatherSdpa));
}

#[cfg(cuda_unix)]
#[test]
fn standard_sliding_decode_keeps_device_context_lengths() {
    for head_size in [64, 80, 96, 112, 128, 192, 256, 512] {
        assert!(!DecodePlan::requires_host_context_lengths(
            AttentionBackendKind::Standard,
            head_size,
            true,
        ));
        for has_sinks in [false, true] {
            let plan = DecodePlan::choose(DecodePlanInput {
                attention_backend: AttentionBackendKind::Standard,
                head_size,
                has_alibi: false,
                has_sinks,
                has_sliding_window: true,
            })
            .unwrap();
            assert!(matches!(plan, DecodePlan::PagedAttention));
        }
    }
}

#[cfg(cuda_unix)]
#[test]
fn standard_sliding_decode_gathers_unsupported_paged_head_sizes() {
    for head_size in [160, 384] {
        assert!(DecodePlan::requires_host_context_lengths(
            AttentionBackendKind::Standard,
            head_size,
            true,
        ));
        let plan = DecodePlan::choose(DecodePlanInput {
            attention_backend: AttentionBackendKind::Standard,
            head_size,
            has_alibi: false,
            has_sinks: true,
            has_sliding_window: true,
        })
        .unwrap();
        assert!(matches!(plan, DecodePlan::GatherSdpa));
    }
}

#[test]
fn standard_full_decode_keeps_paged_kernel() {
    assert!(!DecodePlan::requires_host_context_lengths(
        AttentionBackendKind::Standard,
        128,
        false,
    ));
    let plan = DecodePlan::choose(DecodePlanInput {
        attention_backend: AttentionBackendKind::Standard,
        head_size: 128,
        has_alibi: false,
        has_sinks: false,
        has_sliding_window: false,
    })
    .unwrap();

    assert!(matches!(plan, DecodePlan::PagedAttention));
}

#[cfg(cuda_unix)]
#[test]
fn flashinfer_sliding_decode_keeps_device_context_lengths() {
    assert!(!DecodePlan::requires_host_context_lengths(
        AttentionBackendKind::FlashInfer,
        128,
        true,
    ));
    let plan = DecodePlan::choose(DecodePlanInput {
        attention_backend: AttentionBackendKind::FlashInfer,
        head_size: 128,
        has_alibi: false,
        has_sinks: false,
        has_sliding_window: true,
    })
    .unwrap();
    assert!(matches!(plan, DecodePlan::FlashInfer(_)));
}

#[test]
fn paged_mm_prefix_requires_range_metadata() {
    assert!(!select_paged_mm_prefix_path(true, true, true, false, false).unwrap());
    assert!(select_paged_mm_prefix_path(true, true, true, true, false).is_err());
    assert!(select_paged_mm_prefix_path(true, true, true, true, true).unwrap());
    assert!(!select_paged_mm_prefix_path(false, true, true, true, false).unwrap());
}
