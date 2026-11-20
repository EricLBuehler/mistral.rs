use std::{collections::HashSet, sync::Arc};

use candle_core::{Device, Result, Tensor, D};
use mistralrs_quant::QuantMethod;

use super::{
    contiguous_row_range, copy_dflash_graph_output_rows, dflash_adaptive_env_value,
    dflash_adaptive_supported, dflash_graph_host_rows, dflash_graph_plans,
    dflash_graph_positions_fit, dflash_graph_precapture_shapes, dflash_prefix_replay,
    dflash_rope_from_positions, drain_dflash_lru_entries, gather_ctx_taps, linear_from_weight,
    resolve_dflash_sampling_policy, select_ctx_kv_rows, select_dflash_depth,
    update_dormant_sequences, DFlashConfig, DFlashGraphHostInput, DFlashSamplingInputs,
    DFlashSequenceEviction, ADAPT_FULL_DEPTH_MAX_BATCH,
};
#[cfg(cuda_flash_attn)]
use super::{release_dflash_cuda_graph_resources, windowed_kv_checkpoint_capacity};
#[cfg(feature = "cuda")]
use super::{validate_candidate_selector_cuda, CandidateSelectorCudaSpec};
use crate::layers::{yarn_inv_freq_and_attention_factor, YarnRopeConfig};
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    cuda_graph_precapture_batches, CudaGraphComponent, CUDA_GRAPH_MAX_BATCH_BUCKET,
};
use crate::speculative::MtpDraftSamplingMethod;
use crate::speculative::{SpeculativeGraphPlan, SpeculativePrefixReplay};

#[cfg(cuda_flash_attn)]
#[test]
fn prefix_checkpoint_capacity_includes_transactional_staging() -> Result<()> {
    assert_eq!(windowed_kv_checkpoint_capacity(0)?, 0);
    assert_eq!(windowed_kv_checkpoint_capacity(16)?, 17);
    assert!(windowed_kv_checkpoint_capacity(usize::MAX).is_err());
    Ok(())
}

#[cfg(feature = "cuda")]
fn selector_cuda_spec(vocab_size: usize, top_k: usize) -> CandidateSelectorCudaSpec {
    CandidateSelectorCudaSpec {
        device_is_cuda: true,
        logits_dtype: candle_core::DType::BF16,
        hidden_projection_dtype: candle_core::DType::BF16,
        predecessor_dtype: candle_core::DType::BF16,
        successor_dtype: candle_core::DType::BF16,
        top_k,
        vocab_size,
        predecessor_vocab_size: Some(vocab_size),
        successor_vocab_size: Some(vocab_size),
    }
}

#[test]
fn dflash_draft_sampling_resolves_once_at_load() {
    assert_eq!(
        resolve_dflash_sampling_policy(MtpDraftSamplingMethod::Auto, Ok(())).unwrap(),
        MtpDraftSamplingMethod::Probabilistic
    );
    assert_eq!(
        resolve_dflash_sampling_policy(
            MtpDraftSamplingMethod::Auto,
            Err("missing selector".to_string()),
        )
        .unwrap(),
        MtpDraftSamplingMethod::Greedy
    );
    assert_eq!(
        resolve_dflash_sampling_policy(
            MtpDraftSamplingMethod::Greedy,
            Err("unsupported dtype".to_string()),
        )
        .unwrap(),
        MtpDraftSamplingMethod::Greedy
    );
    assert_eq!(
        resolve_dflash_sampling_policy(MtpDraftSamplingMethod::Probabilistic, Ok(())).unwrap(),
        MtpDraftSamplingMethod::Probabilistic
    );

    let missing = resolve_dflash_sampling_policy(
        MtpDraftSamplingMethod::Probabilistic,
        Err("a DFlash2 checkpoint with a candidate selector is required".to_string()),
    )
    .expect_err("probabilistic drafting must not silently use greedy selection");
    assert!(missing
        .to_string()
        .contains("DFlash2 checkpoint with a candidate selector is required"));

    let unsupported = resolve_dflash_sampling_policy(
        MtpDraftSamplingMethod::Probabilistic,
        Err("unsupported dtype".to_string()),
    )
    .expect_err("probabilistic drafting must reject an unsupported CUDA selector");
    assert!(unsupported.to_string().contains("unsupported dtype"));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_selector_top_k_respects_the_ranked_workspace_bound() {
    let vocab_size = 1_000_000;
    let max_top_k = crate::ops::cuda_topk_ranked_packed_max_k(vocab_size)
        .expect("test vocabulary is representable")
        .min(crate::ops::CUDA_DFLASH_SELECTOR_MAX_K);
    assert!(validate_candidate_selector_cuda(selector_cuda_spec(vocab_size, max_top_k)).is_ok());
    let error = validate_candidate_selector_cuda(selector_cuda_spec(vocab_size, max_top_k + 1))
        .expect_err("top_k above the ranked workspace bound must be rejected");
    assert!(error.contains(&format!("[1, {max_top_k}]")));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_selector_rejects_unrepresentable_vocabularies() {
    let vocab_size = crate::ops::CUDA_TOPK_MAX_EXACT_PACKED_VOCAB + 1;
    let error = validate_candidate_selector_cuda(selector_cuda_spec(vocab_size, 1))
        .expect_err("packed F32 indices must represent every vocabulary id exactly");
    assert!(error.contains("cannot be represented exactly"));
}

#[test]
fn final_sequence_release_does_not_leave_dormant_ids() {
    let mut dormant = HashSet::from([1, 2]);
    update_dormant_sequences(&mut dormant, &[2, 3], DFlashSequenceEviction::Dormant);
    assert_eq!(dormant, HashSet::from([1, 2, 3]));

    update_dormant_sequences(&mut dormant, &[2, 3], DFlashSequenceEviction::Released);
    assert_eq!(dormant, HashSet::from([1]));
}

#[test]
fn packed_projection_preserves_component_layout() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[2f32, 3., 5.], [7., 11., 13.]], &device)?;
    let weights = [
        Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?,
        Tensor::new(&[[7f32, 8., 9.]], &device)?,
        Tensor::new(&[[10f32, 11., 12.]], &device)?,
    ];
    let components = weights
        .iter()
        .map(|weight| linear_from_weight(weight.clone(), None, &device))
        .collect::<Result<Vec<Arc<dyn QuantMethod>>>>()?;
    let packed_weight = Tensor::cat(&weights.iter().collect::<Vec<_>>(), 0)?;
    let packed = linear_from_weight(packed_weight, None, &device)?.forward(&input)?;

    let mut offset = 0;
    for component in components {
        let expected = component.forward(&input)?;
        let rows = expected.dim(D::Minus1)?;
        assert_eq!(
            packed.narrow(D::Minus1, offset, rows)?.to_vec2::<f32>()?,
            expected.to_vec2::<f32>()?
        );
        offset += rows;
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn context_tap_gather_preserves_flat_row_order() -> Result<()> {
    let device = Device::Cpu;
    let tap_a = Tensor::from_vec(
        (0..12)
            .flat_map(|row| [row as f32, 100.0 + row as f32])
            .collect::<Vec<_>>(),
        (3, 4, 2),
        &device,
    )?;
    let tap_b = Tensor::from_vec(
        (0..12).map(|row| -(row as f32)).collect::<Vec<_>>(),
        (3, 4, 1),
        &device,
    )?;
    let packed = gather_ctx_taps(&[tap_a, tap_b], vec![9, 10, 1, 6, 7], &device)?;
    assert_eq!(
        packed.to_vec3::<f32>()?,
        [vec![
            vec![9.0, 109.0, -9.0],
            vec![10.0, 110.0, -10.0],
            vec![1.0, 101.0, -1.0],
            vec![6.0, 106.0, -6.0],
            vec![7.0, 107.0, -7.0],
        ]]
    );
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn context_tap_gather_handles_contiguous_rows_without_reordering() -> Result<()> {
    let device = Device::Cpu;
    let tap = Tensor::from_vec((0..12).map(|row| row as f32).collect(), (3, 4, 1), &device)?;
    let packed = gather_ctx_taps(&[tap], vec![4, 5, 6, 7], &device)?;
    assert_eq!(
        packed.to_vec3::<f32>()?,
        [vec![vec![4.0], vec![5.0], vec![6.0], vec![7.0]]]
    );
    Ok(())
}

#[test]
fn prepared_context_row_selection_detects_only_ordered_ranges() {
    assert_eq!(contiguous_row_range(&[4, 5, 6, 7]), Some((4, 4)));
    assert_eq!(contiguous_row_range(&[4, 6, 7]), None);
    assert_eq!(contiguous_row_range(&[7, 6, 5]), None);
    assert_eq!(contiguous_row_range(&[]), None);
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn context_kv_row_selection_handles_narrowed_sources() -> Result<()> {
    let device = Device::Cpu;
    let source = Tensor::from_vec(
        (0..32).map(|value| value as f32).collect(),
        (2, 2, 4, 2),
        &device,
    )?;
    let narrowed = source.narrow(2, 1, 3)?;
    assert!(!narrowed.is_contiguous());
    let (k, v) = select_ctx_kv_rows(&narrowed, &narrowed, &[0, 2])?;
    assert_eq!(
        k.flatten_all()?.to_vec1::<f32>()?,
        v.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        k.flatten_all()?.to_vec1::<f32>()?,
        [
            2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0, 18.0, 19.0, 22.0, 23.0, 26.0, 27.0, 30.0,
            31.0,
        ]
    );
    assert_eq!(k.dims(), &[2, 2, 2, 2]);
    let (empty_k, empty_v) = select_ctx_kv_rows(&narrowed, &narrowed, &[])?;
    assert_eq!(empty_k.dims(), &[2, 2, 0, 2]);
    assert_eq!(empty_v.dims(), &[2, 2, 0, 2]);
    Ok(())
}

#[test]
fn adaptive_depth_is_full_for_small_batches() {
    assert_eq!(select_dflash_depth(true, 7, 1, 1), 7);
    assert_eq!(
        select_dflash_depth(true, 7, ADAPT_FULL_DEPTH_MAX_BATCH, 1),
        7
    );
}

#[test]
fn adaptive_depth_keeps_a_live_draft_at_large_batches() {
    for max_n in 1..=7 {
        for batch in [9, 16, 64] {
            assert!(select_dflash_depth(true, max_n, batch, batch) > 0);
        }
    }
    assert_eq!(select_dflash_depth(true, 7, 16, 16), 3);
}

#[test]
fn adaptive_depth_uses_global_live_load_for_fragmented_batches() {
    assert_eq!(select_dflash_depth(true, 7, 1, 9), 3);
    assert_eq!(select_dflash_depth(true, 7, 8, 9), 3);
    assert_eq!(select_dflash_depth(true, 7, 8, 8), 7);
}

#[test]
fn fixed_depth_ignores_batch_size() {
    assert_eq!(select_dflash_depth(false, 7, 1, 1), 7);
    assert_eq!(select_dflash_depth(false, 7, 16, 16), 7);
}

#[test]
fn prefix_replay_covers_every_dflash_attention_window() {
    assert_eq!(
        dflash_prefix_replay([Some(1024), Some(2048), Some(512)]),
        SpeculativePrefixReplay::Suffix(2048)
    );
    assert_eq!(
        dflash_prefix_replay([Some(2048), None, Some(1024)]),
        SpeculativePrefixReplay::Full
    );
}

#[test]
fn adaptive_env_parses_explicit_values() {
    assert!(dflash_adaptive_env_value("1"));
    assert!(dflash_adaptive_env_value("true"));
    assert!(dflash_adaptive_env_value("TRUE"));
    for value in ["", "0", "false", "yes", "on", "7"] {
        assert!(!dflash_adaptive_env_value(value));
    }
}

#[test]
fn adaptive_mode_requires_depth_and_serving_headroom() {
    assert!(dflash_adaptive_supported(7, 16));
    assert!(!dflash_adaptive_supported(3, 16));
    assert!(!dflash_adaptive_supported(7, 8));
}

#[test]
fn graph_plans_cover_every_batch_without_a_zero_depth_gap() {
    assert_eq!(
        dflash_graph_plans(false, 7),
        vec![SpeculativeGraphPlan::new(7, None)]
    );
    assert_eq!(
        dflash_graph_plans(true, 7),
        vec![
            SpeculativeGraphPlan::new(7, Some(ADAPT_FULL_DEPTH_MAX_BATCH)),
            SpeculativeGraphPlan::new(3, None),
        ]
    );
    assert_eq!(
        dflash_graph_plans(true, 3),
        vec![SpeculativeGraphPlan::new(3, None)]
    );
}

#[test]
fn graph_precapture_shapes_follow_depth_batch_limits_and_pool_capacity() {
    let batches = (1..=8).chain(std::iter::once(16)).collect::<Vec<_>>();
    assert_eq!(
        dflash_graph_precapture_shapes(&dflash_graph_plans(false, 7), batches.clone(), 16),
        vec![
            (1, 8),
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (6, 8),
            (7, 8),
            (8, 8),
            (16, 8),
        ]
    );
    assert_eq!(
        dflash_graph_precapture_shapes(&dflash_graph_plans(true, 7), batches, 16),
        vec![
            (1, 4),
            (1, 8),
            (2, 4),
            (2, 8),
            (3, 4),
            (3, 8),
            (4, 4),
            (4, 8),
            (5, 4),
            (5, 8),
            (6, 4),
            (6, 8),
            (7, 4),
            (7, 8),
            (8, 4),
            (8, 8),
            (16, 4),
        ]
    );
}

#[cfg(feature = "cuda")]
#[test]
fn graph_precapture_shapes_cover_large_shared_batch_buckets() {
    let shapes = dflash_graph_precapture_shapes(
        &dflash_graph_plans(false, 7),
        cuda_graph_precapture_batches(CudaGraphComponent::DFlash, 8),
        CUDA_GRAPH_MAX_BATCH_BUCKET,
    );
    assert_eq!(
        shapes.len(),
        cuda_graph_precapture_batches(CudaGraphComponent::DFlash, 8).count()
    );
    assert_eq!(shapes.last(), Some(&(CUDA_GRAPH_MAX_BATCH_BUCKET, 8)));
}

#[test]
fn graph_pressure_eviction_drains_lru_entries_in_one_batch() {
    let mut entries = vec![10, 20, 30, 40];
    assert!(drain_dflash_lru_entries(&mut entries, 0).is_empty());
    assert_eq!(drain_dflash_lru_entries(&mut entries, 2), vec![10, 20]);
    assert_eq!(entries, vec![30, 40]);
    assert_eq!(
        drain_dflash_lru_entries(&mut entries, usize::MAX),
        vec![30, 40]
    );
    assert!(entries.is_empty());
}

#[test]
fn graph_outputs_do_not_alias_reclaimable_buffers() -> Result<()> {
    use candle_core::Var;

    let source = Var::from_tensor(&Tensor::from_vec(
        vec![1u32, 2, 3, 4, 5, 6],
        (2, 3),
        &Device::Cpu,
    )?)?;
    let output = copy_dflash_graph_output_rows(&source.as_detached_tensor(), 1)?;
    source.set(&Tensor::from_vec(
        vec![11u32, 12, 13, 14, 15, 16],
        (2, 3),
        &Device::Cpu,
    )?)?;
    assert_eq!(output.to_vec2::<u32>()?, vec![vec![1, 2, 3]]);
    Ok(())
}

#[cfg(cuda_flash_attn)]
#[test]
fn graph_release_waits_for_detached_output_copies() -> anyhow::Result<()> {
    use candle_core::{cuda_backend::cudarc::driver::sys, Var};

    use crate::pipeline::cuda_graph::{
        disable_event_tracking_for_capture, prepare_cuda_graph_memory_pool,
        restore_event_tracking_after_capture, CudaGraphHandle, CudaGraphHostStaging,
    };

    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
    let input = Var::from_tensor(&Tensor::from_vec(
        vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        &device,
    )?)?;
    let output = Var::zeros((2, 3), candle_core::DType::F32, &device)?;
    device.synchronize()?;

    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let captured = input.as_detached_tensor().affine(2.0, 1.0)?;
    crate::cuda::graph::copy_tensor(&captured, &output.as_detached_tensor())?;
    let graph = CudaGraphHandle::end_capture(&stream)?
        .ok_or_else(|| anyhow::anyhow!("DFlash graph capture returned no graph"))?;
    restore_event_tracking_after_capture(&stream, restore_event_tracking);
    graph.upload()?;

    let mut staging = CudaGraphHostStaging::new(stream)?;
    graph.launch()?;
    staging.record_graph_complete()?;
    let detached = copy_dflash_graph_output_rows(&output.as_detached_tensor(), 1)?;
    let (_, release_result) =
        release_dflash_cuda_graph_resources(graph, (staging, input, output, captured));
    release_result?;

    assert_eq!(detached.to_vec2::<f32>()?, vec![vec![3.0, 5.0, 7.0]]);
    Ok(())
}

#[test]
fn graph_positions_support_long_contexts_and_preserve_fallback_bounds() {
    assert!(dflash_graph_positions_fit(&[65_529, 100_000, 250_000], 8));
    assert!(dflash_graph_positions_fit(&[u32::MAX as usize - 7], 8));
    assert!(!dflash_graph_positions_fit(&[u32::MAX as usize - 6], 8));
    assert!(!dflash_graph_positions_fit(&[100_000], 0));
}

fn qwen35_dflash_config() -> DFlashConfig {
    serde_json::from_str(
        r#"{
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 5,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "vocab_size": 248320,
                "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
                "num_target_layers": 64
            }"#,
    )
    .unwrap()
}

fn qwen35_target_yarn() -> YarnRopeConfig {
    YarnRopeConfig {
        base: 10_000_000.0,
        head_dim: 64,
        max_position_embeddings: 1_048_576,
        original_max_position_embeddings: 262_144,
        factor: 4.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        mscale: 1.0,
        mscale_all_dim: 0.0,
        attention_factor: None,
    }
}

#[test]
fn dflash_yarn_uses_draft_geometry_and_target_scaling() -> Result<()> {
    let cfg = qwen35_dflash_config();
    let yarn = cfg
        .yarn_rope_config(Some(&qwen35_target_yarn()))?
        .expect("target uses YaRN");
    assert_eq!(yarn.base, 10_000_000.0);
    assert_eq!(yarn.head_dim, 128);
    assert_eq!(yarn.factor, 4.0);
    assert_eq!(yarn.original_max_position_embeddings, 262_144);

    let (inv_freq, attention_factor) = yarn_inv_freq_and_attention_factor(&yarn, &Device::Cpu)?;
    let inv_freq = inv_freq.to_vec1::<f32>()?;
    assert_eq!(inv_freq.len(), 64);
    assert!((inv_freq[0] - 1.0).abs() < 1e-6);
    let unscaled_last = 1.0 / yarn.base.powf(126.0 / 128.0);
    assert!((inv_freq[63] - unscaled_last / yarn.factor).abs() < 1e-8);
    assert!((attention_factor - (1.0 + 0.1 * yarn.factor.ln())).abs() < 1e-6);
    Ok(())
}

#[test]
fn dflash_yarn_rejects_a_mismatched_native_context() {
    let cfg = qwen35_dflash_config();
    let mut target = qwen35_target_yarn();
    target.original_max_position_embeddings = 131_072;
    let error = cfg
        .yarn_rope_config(Some(&target))
        .expect_err("mismatched native context must not silently change draft RoPE");
    assert!(error.to_string().contains("native context length"));
}

#[test]
fn dflash_native_rope_does_not_require_a_declared_context() -> Result<()> {
    let mut cfg = qwen35_dflash_config();
    cfg.max_position_embeddings = None;
    assert!(cfg.yarn_rope_config(None)?.is_none());
    let error = cfg
        .yarn_rope_config(Some(&qwen35_target_yarn()))
        .expect_err("YaRN must validate the draft's native context");
    assert!(error
        .to_string()
        .contains("max_position_embeddings is required"));
    Ok(())
}

#[test]
fn dflash_rejects_draft_side_rope_scaling() {
    for key in ["rope_type", "type"] {
        let mut cfg = qwen35_dflash_config();
        cfg.rope_parameters = Some(serde_json::json!({
            "rope_theta": 10_000_000,
            (key): "yarn",
        }));
        let error = cfg
            .validate_rope_type()
            .expect_err("draft-side scaling must not be silently ignored");
        assert!(error
            .to_string()
            .contains("configure RoPE scaling on the target"));
    }
}

#[test]
fn dflash_rejects_conflicting_rope_type_aliases() {
    let mut cfg = qwen35_dflash_config();
    cfg.rope_parameters = Some(serde_json::json!({
        "rope_type": "default",
        "type": "yarn",
    }));
    let error = cfg
        .validate_rope_type()
        .expect_err("conflicting aliases must not be accepted");
    assert!(error.to_string().contains("conflicts with legacy type"));
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn graph_rope_uses_each_replayed_long_position() -> Result<()> {
    let device = Device::Cpu;
    let positions = [65_535u32, 65_536, 100_000, 131_071];
    let frequencies = [1.0f32, 0.5, 0.01];
    let positions_tensor = Tensor::from_vec(positions.to_vec(), (positions.len(),), &device)?;
    let inv_freq = Tensor::from_vec(frequencies.to_vec(), (frequencies.len(),), &device)?;
    let (cos, sin) =
        dflash_rope_from_positions(&positions_tensor, &inv_freq, candle_core::DType::F32, 1.0)?;
    let cos = cos.to_vec2::<f32>()?;
    let sin = sin.to_vec2::<f32>()?;

    for (row, position) in positions.into_iter().enumerate() {
        for (column, frequency) in frequencies.into_iter().enumerate() {
            let angle = position as f32 * frequency;
            assert!((cos[row][column] - angle.cos()).abs() < 1e-5);
            assert!((sin[row][column] - angle.sin()).abs() < 1e-5);
        }
    }
    Ok(())
}

#[test]
fn graph_rope_applies_yarn_attention_scaling() -> Result<()> {
    let positions = Tensor::from_vec(vec![0u32, 1], 2, &Device::Cpu)?;
    let inv_freq = Tensor::from_vec(vec![1.0f32, 0.25], 2, &Device::Cpu)?;
    let attention_factor = 1.25;
    let (cos, sin) = dflash_rope_from_positions(
        &positions,
        &inv_freq,
        candle_core::DType::F32,
        attention_factor,
    )?;
    let cos = cos.to_vec2::<f32>()?;
    let sin = sin.to_vec2::<f32>()?;
    assert_eq!(cos[0], vec![attention_factor, attention_factor]);
    assert_eq!(sin[0], vec![0.0, 0.0]);
    assert!((cos[1][0] - attention_factor * 1.0f32.cos()).abs() < 1e-6);
    assert!((sin[1][1] - attention_factor * 0.25f32.sin()).abs() < 1e-6);
    Ok(())
}

#[cfg(cuda_flash_attn)]
#[test]
fn graph_rope_replays_mixed_long_positions_on_cuda() -> anyhow::Result<()> {
    use candle_core::{cuda_backend::cudarc::driver::sys, Var};

    use crate::pipeline::cuda_graph::{
        disable_event_tracking_for_capture, prepare_cuda_graph_memory_pool,
        restore_event_tracking_after_capture, CudaGraphHandle,
    };

    const BATCH: usize = 16;
    const BLOCK: usize = 8;
    const HEAD_DIM: usize = 128;
    const ROPE_THETA: f32 = 10_000_000.0;
    const BENCH_WARMUP: usize = 100;
    const BENCH_ITERATIONS: usize = 10_000;

    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
    let inv_freq_values = (0..HEAD_DIM)
        .step_by(2)
        .map(|index| 1.0 / ROPE_THETA.powf(index as f32 / HEAD_DIM as f32))
        .collect::<Vec<_>>();
    let replay_positions = |bases: [u32; BATCH]| {
        bases
            .into_iter()
            .flat_map(|base| (0..BLOCK).map(move |offset| base + offset as u32))
            .collect::<Vec<_>>()
    };
    let first_positions = replay_positions([
        65_532, 65_536, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000, 130_000, 140_000,
        150_000, 160_000, 180_000, 200_000, 225_000, 250_000,
    ]);
    let positions = Var::from_tensor(&Tensor::from_vec(
        first_positions.clone(),
        BATCH * BLOCK,
        &device,
    )?)?;
    let inv_freq = Tensor::from_vec(inv_freq_values.clone(), HEAD_DIM / 2, &device)?;
    let warmup = dflash_rope_from_positions(
        &positions.as_detached_tensor(),
        &inv_freq,
        candle_core::DType::BF16,
        1.0,
    )?;
    device.synchronize()?;
    drop(warmup);

    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let (cos, sin) = dflash_rope_from_positions(
        &positions.as_detached_tensor(),
        &inv_freq,
        candle_core::DType::BF16,
        1.0,
    )?;
    let graph = CudaGraphHandle::end_capture(&stream)?
        .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
    restore_event_tracking_after_capture(&stream, restore_event_tracking);
    graph.upload()?;

    for replayed in [
        first_positions,
        replay_positions([
            100_003, 110_000, 120_000, 130_000, 131_072, 140_000, 150_000, 160_000, 175_000,
            190_000, 200_000, 210_000, 220_000, 230_000, 240_000, 250_000,
        ]),
    ] {
        positions.set(&Tensor::from_vec(replayed.clone(), BATCH * BLOCK, &device)?)?;
        graph.launch()?;
        stream.synchronize()?;
        let cos = cos.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        let sin = sin.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        for (row, position) in replayed.into_iter().enumerate() {
            for (column, frequency) in inv_freq_values.iter().copied().enumerate() {
                let angle = position as f32 * frequency;
                assert!((cos[row][column] - angle.cos()).abs() < 0.005);
                assert!((sin[row][column] - angle.sin()).abs() < 0.005);
            }
        }
    }

    for _ in 0..BENCH_WARMUP {
        graph.launch()?;
    }
    stream.synchronize()?;
    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..BENCH_ITERATIONS {
        graph.launch()?;
    }
    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    end.synchronize()?;
    let latency_us = f64::from(start.elapsed_ms(&end)?) * 1_000.0 / BENCH_ITERATIONS as f64;
    eprintln!("DFlash C16/block8 RoPE graph latency: {latency_us:.3} us");

    drop(graph);
    drop(cos);
    drop(sin);
    device.synchronize()?;
    Ok(())
}

#[test]
fn graph_rows_keep_mixed_positions_across_the_old_table_boundary() -> Result<()> {
    let rows = dflash_graph_host_rows(DFlashGraphHostInput {
        anchors: &[5, 7],
        start_positions: &[65_532, 100_000],
        mask_token_id: 9,
        block: 8,
        batch_bucket: 2,
        sampling: None,
    })?;
    assert_eq!(
        rows.rope_indices,
        [
            65_532, 65_533, 65_534, 65_535, 65_536, 65_537, 65_538, 65_539, 100_000, 100_001,
            100_002, 100_003, 100_004, 100_005, 100_006, 100_007,
        ]
    );
    Ok(())
}

#[test]
fn graph_rows_pad_with_a_valid_alias() -> Result<()> {
    let rows = dflash_graph_host_rows(DFlashGraphHostInput {
        anchors: &[5, 7],
        start_positions: &[100, 200],
        mask_token_id: 9,
        block: 4,
        batch_bucket: 4,
        sampling: None,
    })?;
    assert_eq!(
        rows.token_ids,
        [5, 9, 9, 9, 7, 9, 9, 9, 7, 9, 9, 9, 7, 9, 9, 9]
    );
    assert_eq!(rows.anchors, [5, 7, 7, 7]);
    assert!(rows.selector_inverse_temperatures.is_none());
    assert!(rows.selector_uniforms.is_none());
    assert_eq!(
        rows.rope_indices,
        [100, 101, 102, 103, 200, 201, 202, 203, 200, 201, 202, 203, 200, 201, 202, 203,]
    );
    Ok(())
}

#[test]
fn graph_rows_pad_selector_sampling_inputs_with_inert_rows() -> Result<()> {
    let rows = dflash_graph_host_rows(DFlashGraphHostInput {
        anchors: &[5, 7],
        start_positions: &[100, 200],
        mask_token_id: 9,
        block: 4,
        batch_bucket: 4,
        sampling: Some(DFlashSamplingInputs {
            inverse_temperatures: &[1.0, 0.5],
            uniforms: &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }),
    })?;
    assert_eq!(
        rows.selector_inverse_temperatures.as_deref(),
        Some(&[1.0, 0.5, 0.0, 0.0][..])
    );
    assert_eq!(
        rows.selector_uniforms.as_deref(),
        Some(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..])
    );
    Ok(())
}

#[test]
fn graph_rows_reuse_bucket_storage() -> Result<()> {
    let mut rows = dflash_graph_host_rows(DFlashGraphHostInput {
        anchors: &[5, 7],
        start_positions: &[100, 200],
        mask_token_id: 9,
        block: 4,
        batch_bucket: 4,
        sampling: Some(DFlashSamplingInputs {
            inverse_temperatures: &[1.0, 0.5],
            uniforms: &[0.1; 6],
        }),
    })?;
    let capacities = (
        rows.token_ids.capacity(),
        rows.rope_indices.capacity(),
        rows.anchors.capacity(),
        rows.selector_inverse_temperatures
            .as_ref()
            .expect("sampling temperatures")
            .capacity(),
        rows.selector_uniforms
            .as_ref()
            .expect("sampling uniforms")
            .capacity(),
    );
    rows.update(DFlashGraphHostInput {
        anchors: &[11, 13],
        start_positions: &[300, 400],
        mask_token_id: 17,
        block: 4,
        batch_bucket: 4,
        sampling: Some(DFlashSamplingInputs {
            inverse_temperatures: &[0.25, 0.75],
            uniforms: &[0.2; 6],
        }),
    })?;
    assert_eq!(rows.token_ids[0], 11);
    assert_eq!(rows.token_ids[4], 13);
    assert_eq!(rows.rope_indices[0], 300);
    assert_eq!(rows.rope_indices[4], 400);
    assert_eq!(
        capacities,
        (
            rows.token_ids.capacity(),
            rows.rope_indices.capacity(),
            rows.anchors.capacity(),
            rows.selector_inverse_temperatures
                .as_ref()
                .expect("sampling temperatures")
                .capacity(),
            rows.selector_uniforms
                .as_ref()
                .expect("sampling uniforms")
                .capacity(),
        )
    );
    Ok(())
}
