use super::*;

#[test]
fn decode_graphs_support_standard_sliding_attention_and_sinks() {
    use crate::paged_attention::{
        HybridPagedKvCacheConfig, KvCacheLayout, ModelConfigMetadata,
        PrefixPrefillAttentionFeatures,
    };

    let mut config = ModelConfigMetadata {
        max_seq_len: 4096,
        num_layers: 2,
        hidden_size: 2048,
        num_kv_heads: 4,
        num_attn_heads: 32,
        sliding_window: None,
        k_head_dim: 128,
        v_head_dim: 128,
        kv_cache_layout: KvCacheLayout::StandardNoFlashInfer,
    };
    assert!(cuda_decode_graph_supported_for_model(Some(&config)));
    config.sliding_window = Some(513);
    assert!(cuda_decode_graph_supported_for_model(Some(&config)));

    config.sliding_window = None;
    let mixed = HybridPagedKvCacheConfig::new(config, vec![true, true])
        .with_prefix_prefill_attention_features(vec![
            PrefixPrefillAttentionFeatures {
                has_sinks: true,
                ..Default::default()
            },
            PrefixPrefillAttentionFeatures {
                has_sinks: true,
                has_sliding_window: true,
                ..Default::default()
            },
        ]);
    assert!(cuda_decode_graph_supported_for_model(Some(&mixed)));
}

fn spec_usage(bytes: &[(usize, usize)]) -> CudaGraphSpecStateUsage {
    CudaGraphSpecStateUsage {
        bytes: bytes
            .iter()
            .map(|(gpu_id, bytes)| (DeviceLocation::Cuda { gpu_id: *gpu_id }, *bytes))
            .collect(),
        device_totals: HashMap::new(),
    }
}

#[test]
fn graph_event_labels_have_fixed_cardinality() {
    use std::collections::HashSet;

    let mut labels = HashSet::new();
    for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
        for event in [
            CudaGraphEvent::Capture,
            CudaGraphEvent::Replay,
            CudaGraphEvent::EagerFallback,
        ] {
            for outcome in [CudaGraphOutcome::Success, CudaGraphOutcome::Failure] {
                assert!(labels.insert(cuda_graph_event_labels(component, event, outcome)));
            }
        }
    }

    assert_eq!(
        CUDA_GRAPH_EVENTS_METRIC,
        "mistralrs_cuda_graph_events_total"
    );
    assert_eq!(labels.len(), 12);
    assert_eq!(
        labels
            .iter()
            .map(|labels| labels.component)
            .collect::<HashSet<_>>(),
        HashSet::from(["target", "dflash"])
    );
    assert_eq!(
        labels
            .iter()
            .map(|labels| labels.event)
            .collect::<HashSet<_>>(),
        HashSet::from(["capture", "replay", "eager_fallback"])
    );
    assert_eq!(
        labels
            .iter()
            .map(|labels| labels.outcome)
            .collect::<HashSet<_>>(),
        HashSet::from(["success", "failure"])
    );
}

#[test]
fn graph_dispatch_labels_have_fixed_cardinality() {
    use std::collections::HashSet;

    let reasons = [
        CudaGraphDispatchReason::CacheHit,
        CudaGraphDispatchReason::Disabled,
        CudaGraphDispatchReason::ModelUnsupported,
        CudaGraphDispatchReason::SpeculativeConflict,
        CudaGraphDispatchReason::PagedAttentionUnavailable,
        CudaGraphDispatchReason::Prefill,
        CudaGraphDispatchReason::IncompatibleShape,
        CudaGraphDispatchReason::BatchUnsupported,
        CudaGraphDispatchReason::CacheConfigUnavailable,
        CudaGraphDispatchReason::RuntimeDisabled,
        CudaGraphDispatchReason::PaddingUnavailable,
        CudaGraphDispatchReason::CachePopulation,
        CudaGraphDispatchReason::Fallback,
    ];
    let mut labels = HashSet::new();
    for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
        for mode in [
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchMode::Eager,
            CudaGraphDispatchMode::Skipped,
        ] {
            for reason in reasons {
                assert!(labels.insert(cuda_graph_dispatch_labels(component, mode, reason)));
            }
        }
    }

    assert_eq!(
        CUDA_GRAPH_DISPATCH_METRIC,
        "mistralrs_cuda_graph_dispatch_total"
    );
    assert_eq!(labels.len(), 78);
    assert_eq!(
        labels
            .iter()
            .map(|labels| labels.mode)
            .collect::<HashSet<_>>(),
        HashSet::from(["replay", "eager", "skipped"])
    );
    assert_eq!(
        labels
            .iter()
            .map(|labels| labels.reason)
            .collect::<HashSet<_>>()
            .len(),
        reasons.len()
    );
}

#[test]
fn graph_eviction_labels_have_fixed_cardinality() {
    use std::collections::HashSet;

    let labels = [
        (
            CudaGraphComponent::Target,
            CudaGraphEvictionReason::Capacity,
        ),
        (
            CudaGraphComponent::Target,
            CudaGraphEvictionReason::MemoryPressure,
        ),
        (
            CudaGraphComponent::Target,
            CudaGraphEvictionReason::SpecStateBudget,
        ),
        (
            CudaGraphComponent::DFlash,
            CudaGraphEvictionReason::Capacity,
        ),
        (
            CudaGraphComponent::DFlash,
            CudaGraphEvictionReason::MemoryPressure,
        ),
    ]
    .into_iter()
    .map(|(component, reason)| (component.label(), reason.label()))
    .collect::<HashSet<_>>();

    assert_eq!(
        CUDA_GRAPH_EVICTIONS_METRIC,
        "mistralrs_cuda_graph_evictions_total"
    );
    assert_eq!(
        CUDA_GRAPH_RESIDENT_ENTRIES_METRIC,
        "mistralrs_cuda_graph_resident_entries"
    );
    assert_eq!(labels.len(), 5);
    assert!(labels.contains(&("target", "capacity")));
    assert!(labels.contains(&("dflash", "capacity")));
    assert!(labels.contains(&("target", "memory_pressure")));
    assert!(labels.contains(&("dflash", "memory_pressure")));
    assert!(labels.contains(&("target", "spec_state_budget")));
}

#[test]
fn capacity_eviction_removes_one_lru_entry_and_preserves_the_bound() {
    let mut entries = vec![10, 20, 30];
    assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), None);
    entries.push(40);
    assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), Some(10));
    entries.push(50);
    assert_eq!(entries, vec![20, 30, 40, 50]);
    assert_eq!(entries.len(), 4);
}

#[test]
fn target_cache_retains_64_entries_before_lru_eviction() {
    let capacity = TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY;
    assert_eq!(capacity, 64);

    let mut entries = (0..capacity - 1).collect::<Vec<_>>();
    assert_eq!(
        take_cuda_graph_capacity_eviction(&mut entries, capacity),
        None
    );
    entries.push(capacity - 1);
    assert_eq!(entries.len(), capacity);

    assert_eq!(
        take_cuda_graph_capacity_eviction(&mut entries, capacity),
        Some(0),
    );
    entries.push(capacity);

    assert_eq!(entries.len(), capacity);
    assert_eq!(entries.first(), Some(&1));
    assert_eq!(entries.last(), Some(&capacity));
}

#[test]
fn target_cache_capacity_tracks_large_speculative_working_sets() {
    assert_eq!(target_cuda_graph_cache_capacity(23, 23), 64);
    assert_eq!(target_cuda_graph_cache_capacity(38, 15), 64);
    assert_eq!(target_cuda_graph_cache_capacity(54, 23), 80);
    assert_eq!(target_cuda_graph_cache_capacity(61, 30), 96);
    assert_eq!(target_cuda_graph_cache_capacity(80, 23), 96);
    assert_eq!(target_cuda_graph_cache_capacity(usize::MAX, usize::MAX), 96);
}

#[test]
fn target_single_token_batch_buckets_are_exact_through_16() {
    assert_eq!(
        cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 0),
        None
    );
    for batch in 1..=TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS {
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, batch),
            Some(batch)
        );
    }
    assert_eq!(
        cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 17),
        Some(24)
    );
    assert_eq!(
        cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 31),
        Some(32)
    );
    assert_eq!(
        cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, CUDA_GRAPH_MAX_BATCH_BUCKET),
        Some(CUDA_GRAPH_MAX_BATCH_BUCKET)
    );
    assert_eq!(
        cuda_graph_batch_bucket(
            CudaGraphComponent::Target,
            1,
            CUDA_GRAPH_MAX_BATCH_BUCKET + 1
        ),
        None
    );
}

#[test]
fn multi_token_and_dflash_batches_keep_coarse_buckets() {
    for (component, q_len) in [
        (CudaGraphComponent::Target, 4),
        (CudaGraphComponent::DFlash, 1),
        (CudaGraphComponent::DFlash, 8),
    ] {
        assert_eq!(cuda_graph_batch_bucket(component, q_len, 8), Some(8));
        assert_eq!(cuda_graph_batch_bucket(component, q_len, 9), Some(16));
        assert_eq!(cuda_graph_batch_bucket(component, q_len, 17), Some(24));
    }
}

#[test]
fn precapture_batches_follow_component_and_query_width() {
    assert_eq!(
        cuda_graph_precapture_batches(CudaGraphComponent::Target, 1).collect::<Vec<_>>(),
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 40, 48, 56, 64, 72, 80,
            88, 96, 104, 112, 120, 128,
        ]
    );
    assert_eq!(
        cuda_graph_precapture_batches(CudaGraphComponent::Target, 4).collect::<Vec<_>>(),
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
        ]
    );
    assert_eq!(
        cuda_graph_precapture_batches(CudaGraphComponent::DFlash, 1).collect::<Vec<_>>(),
        cuda_graph_precapture_batches(CudaGraphComponent::Target, 4).collect::<Vec<_>>()
    );
}

#[test]
fn precapture_max_batch_uses_the_shape_policy() {
    assert_eq!(
        cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 0),
        0
    );
    assert_eq!(
        cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 9),
        9
    );
    assert_eq!(
        cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 4, 9),
        16
    );
    assert_eq!(
        cuda_graph_precapture_max_batch(CudaGraphComponent::DFlash, 8, 9),
        16
    );
    assert_eq!(
        cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 17),
        24
    );
    assert_eq!(
        cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 129),
        128
    );
}

#[test]
fn target_single_token_buckets_fit_the_default_cache() {
    let bucket_count = cuda_graph_precapture_batches(CudaGraphComponent::Target, 1).count();
    assert_eq!(bucket_count, 30);
    assert_eq!(
        target_cuda_graph_cache_capacity(bucket_count, bucket_count),
        TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY
    );
}

#[test]
fn startup_precapture_accepts_decode_and_verification_widths() {
    assert!(!cuda_graph_startup_capture_allowed(0));
    assert!(cuda_graph_startup_capture_allowed(1));
    assert!(cuda_graph_startup_capture_allowed(4));
    assert!(cuda_graph_startup_capture_allowed(8));
}

#[test]
fn decode_graph_batch_kind_rejects_every_prefill_chunk() {
    assert!(!cuda_decode_graph_batch_kind_supported(
        RecurrentBatchKind::Prefill
    ));
    assert!(cuda_decode_graph_batch_kind_supported(
        RecurrentBatchKind::Decode
    ));
    assert!(cuda_decode_graph_batch_kind_supported(
        RecurrentBatchKind::SpeculativeDecode
    ));
}

#[test]
fn graph_context_len_tracks_paged_attention_partitions() {
    assert_eq!(graph_context_len(Some(1), Some(2048)), Some(512));
    assert_eq!(graph_context_len(Some(512), Some(2048)), Some(512));
    assert_eq!(graph_context_len(Some(513), Some(2048)), Some(1024));
    assert_eq!(graph_context_len(Some(1537), Some(2048)), Some(2048));
}

#[test]
fn graph_context_len_preserves_nonstandard_metadata() {
    assert_eq!(graph_context_len(Some(513), None), Some(513));
    assert_eq!(graph_context_len(None, Some(2048)), Some(2048));
    assert_eq!(graph_context_len(None, None), None);
}

#[test]
fn decode_row_graph_key_is_independent_of_materialization() {
    let table = vec![1, 2, 3, 4];
    let rows = Arc::new(DecodePagedRows {
        slot_mappings: vec![vec![127]],
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
        context_lens: vec![128],
        full_context_lens: vec![128],
        query_len: 1,
        block_size: 32,
        use_standard_metadata: false,
        max_paged_context_len: 1_604_288,
        sliding_window: None,
        decode_window: 1,
        devices: vec![Device::Cpu],
        num_kv_heads: 4,
    });
    let staged = rows.build_graph_staged().unwrap();
    let materialized = rows.build_materialized().unwrap();
    let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
    let staged_key =
        CudaDecodeGraphKey::new(&input_ids, &staged, 32, RecurrentBatchKind::Decode).unwrap();
    let materialized_key =
        CudaDecodeGraphKey::new(&input_ids, &materialized, 32, RecurrentBatchKind::Decode).unwrap();
    assert_eq!(staged_key, materialized_key);
    assert!(staged_key.tensors.is_empty());
    assert!(staged_key.decode_rows.is_some());
    let speculative_key = CudaDecodeGraphKey::new(
        &input_ids,
        &materialized,
        32,
        RecurrentBatchKind::SpeculativeDecode,
    )
    .unwrap();
    assert_ne!(staged_key, speculative_key);
    assert!(!staged_key.has_same_spec_state_shape(&speculative_key));

    let mut next_bucket_rows = (*rows).clone();
    next_bucket_rows.block_tables =
        BlockTableSnapshot::from_owned_sequence_tables(vec![(1..=65).collect()], 1);
    next_bucket_rows.context_lens = vec![2049];
    next_bucket_rows.full_context_lens = vec![2049];
    let next_bucket = Arc::new(next_bucket_rows).build_graph_staged().unwrap();
    let next_bucket_key =
        CudaDecodeGraphKey::new(&input_ids, &next_bucket, 32, RecurrentBatchKind::Decode).unwrap();
    assert_ne!(staged_key, next_bucket_key);
    assert!(staged_key.has_same_spec_state_shape(&next_bucket_key));
}

const DECODE_CONTEXT_TEST_PAGE_SIZE: usize = 32;
const DECODE_CONTEXT_TEST_MAX_BATCH: usize = 64;
const DECODE_CONTEXT_TEST_FLOOR: usize = 2048;
const DECODE_CONTEXT_TEST_MAX_CONTEXT: usize = 128 * 1024;
const DECODE_CONTEXT_TEST_KV_HEADS: usize = 8;

fn decode_context_rows(batch: usize, context: usize) -> DecodePagedRows {
    DecodePagedRows {
        slot_mappings: vec![vec![_PAD_SLOT_ID]; batch],
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(
            vec![vec![0; context.div_ceil(DECODE_CONTEXT_TEST_PAGE_SIZE)]; batch],
            1,
        ),
        context_lens: vec![context; batch],
        full_context_lens: vec![context; batch],
        query_len: 1,
        block_size: DECODE_CONTEXT_TEST_PAGE_SIZE,
        use_standard_metadata: false,
        max_paged_context_len: DECODE_CONTEXT_TEST_MAX_CONTEXT,
        sliding_window: None,
        decode_window: 1,
        devices: vec![Device::Cpu],
        num_kv_heads: DECODE_CONTEXT_TEST_KV_HEADS,
    }
}

fn decode_context_key(rows: DecodePagedRows) -> CudaDecodeGraphKey {
    let inputs = Tensor::zeros((rows.slot_mappings.len(), 1), DType::U32, &Device::Cpu).unwrap();
    let metadata = Arc::new(rows).build_graph_staged().unwrap();
    CudaDecodeGraphKey::new(
        &inputs,
        &metadata,
        DECODE_CONTEXT_TEST_PAGE_SIZE,
        RecurrentBatchKind::Decode,
    )
    .unwrap()
}

#[test]
fn standard_decode_row_graph_keys_track_partition_launch_bounds() {
    let standard_key = |context| {
        let mut rows = decode_context_rows(1, context);
        rows.use_standard_metadata = true;
        decode_context_key(rows)
    };
    let captured = standard_key(4097);
    assert_eq!(captured.max_context_len, Some(4608));
    assert_eq!(captured.full_max_context_len, Some(4608));
    for context in [4098, 4200, 4608] {
        assert_eq!(captured, standard_key(context));
    }

    let next_partition = standard_key(4609);
    assert_eq!(captured.decode_rows, next_partition.decode_rows);
    assert_ne!(captured, next_partition);
    assert_eq!(next_partition.max_context_len, Some(5120));
    for context in [4610, 5025, 5120] {
        assert_eq!(next_partition, standard_key(context));
    }

    assert_eq!(
        decode_context_key(decode_context_rows(1, 4097)),
        decode_context_key(decode_context_rows(1, 5025)),
    );
}

#[test]
fn standard_decode_row_graph_keys_preserve_materialization_and_full_context_bounds() {
    let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
    let key = |metadata: &PagedAttentionInputMetadata| {
        CudaDecodeGraphKey::new(
            &input_ids,
            metadata,
            DECODE_CONTEXT_TEST_PAGE_SIZE,
            RecurrentBatchKind::Decode,
        )
        .unwrap()
    };
    for context in [31, 512, 513, 4097, 5025] {
        let mut rows = decode_context_rows(1, context);
        rows.use_standard_metadata = true;
        let rows = Arc::new(rows);
        assert_eq!(
            key(&rows.build_graph_staged().unwrap()),
            key(&rows.build_materialized().unwrap()),
        );
    }

    let sliding_key = |context| {
        let mut rows = decode_context_rows(1, context);
        rows.use_standard_metadata = true;
        rows.context_lens = vec![513];
        rows.sliding_window = Some(513);
        decode_context_key(rows)
    };
    let captured = sliding_key(4097);
    let replay = sliding_key(5025);
    assert_eq!(captured.decode_rows, replay.decode_rows);
    assert_eq!(captured.max_context_len, replay.max_context_len);
    assert_ne!(captured.full_max_context_len, replay.full_max_context_len);
    assert_ne!(captured, replay);
}

#[test]
fn startup_context_keys_cover_changed_admission_and_drain_batches() {
    let startup = cuda_graph_precapture_batches(CudaGraphComponent::Target, 1)
        .filter(|batch| *batch <= DECODE_CONTEXT_TEST_MAX_BATCH)
        .map(|batch| decode_context_key(decode_context_rows(1, 1).padded(batch)))
        .collect::<Vec<_>>();
    let even_warmup = [
        2, 6, 10, 14, 18, 22, 26, 30, 32, 30, 26, 22, 18, 14, 10, 6, 2,
    ];
    let odd_measured = [
        1, 5, 9, 13, 17, 21, 25, 29, 32, 31, 27, 23, 19, 15, 11, 7, 3,
    ];
    for batch in even_warmup.into_iter().chain(odd_measured) {
        let bucket = cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, batch).unwrap();
        for context in [
            1,
            128,
            512,
            513,
            1024,
            1025,
            1151,
            1536,
            DECODE_CONTEXT_TEST_FLOOR,
        ] {
            let rows = decode_context_rows(batch, context).padded(bucket);
            assert!(startup.contains(&decode_context_key(rows)));
        }
    }
    assert_eq!(startup.len(), 22);
    assert!(!startup.contains(&decode_context_key(decode_context_rows(
        32,
        DECODE_CONTEXT_TEST_FLOOR + 1
    ))));
}

#[test]
fn decode_context_floor_preserves_live_csr_splits_and_pad_slots() {
    let mut live_rows = decode_context_rows(3, 1025);
    live_rows.slot_mappings = vec![vec![1024]; 3];
    let rows = Arc::new(live_rows.padded(8));
    let metadata = rows.build_materialized().unwrap();
    let view = &metadata.flashinfer.as_ref().unwrap().views.logical;
    let location = Device::Cpu.location();
    let indptr = view.paged_kv.indptr[&location].to_vec1::<i32>().unwrap();
    assert_eq!(indptr, (0..=8).map(|row| row * 33).collect::<Vec<_>>());
    assert_eq!(
        view.paged_kv.indices[&location].dims(),
        &[8 * (DECODE_CONTEXT_TEST_FLOOR / DECODE_CONTEXT_TEST_PAGE_SIZE)]
    );
    assert_eq!(
        view.paged_kv.last_page_len[&location]
            .to_vec1::<i32>()
            .unwrap(),
        vec![1; 8]
    );
    let chunk = view.tile_plan.kv_chunk_size[&location]
        .to_vec1::<i32>()
        .unwrap()[0];
    let chunks_per_row = 1025usize.div_ceil(chunk as usize);
    let ends = view.tile_plan.o_indptr[&location].to_vec1::<i32>().unwrap();
    assert_eq!(
        ends,
        (0..=8)
            .map(|row| i32::try_from(row * chunks_per_row).unwrap())
            .collect::<Vec<_>>()
    );
    let mask = view.tile_plan.block_valid_mask[&location]
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(mask.len(), 8 * 8);
    assert!(mask[..8 * chunks_per_row].iter().all(|valid| *valid == 1));
    assert!(mask[8 * chunks_per_row..].iter().all(|valid| *valid == 0));
    assert_eq!(rows.slot_mappings[..3], vec![vec![1024]; 3]);
    assert!(rows.slot_mappings[3..]
        .iter()
        .all(|row| row == &[_PAD_SLOT_ID]));
    assert_eq!(
        decode_context_key((*rows).clone()),
        decode_context_key(decode_context_rows(1, 1).padded(8))
    );
}

#[test]
fn decode_context_floor_preserves_sliding_and_full_capacities() {
    let mut startup = decode_context_rows(1, 1);
    startup.sliding_window = Some(128);
    let mut live = decode_context_rows(1, 1025);
    live.sliding_window = Some(128);
    live.context_lens = vec![129];
    assert_eq!(
        decode_context_key(startup),
        decode_context_key(live.clone())
    );
    let metadata = Arc::new(live).build_materialized().unwrap();
    let views = &metadata.flashinfer.as_ref().unwrap().views;
    let location = Device::Cpu.location();
    let sliding = views.sliding.as_ref().unwrap();
    assert_eq!(sliding.paged_kv.indices[&location].dims(), &[5]);
    assert_eq!(
        views.logical.paged_kv.indices[&location].dims(),
        &[DECODE_CONTEXT_TEST_FLOOR / DECODE_CONTEXT_TEST_PAGE_SIZE]
    );
    assert_eq!(
        sliding.paged_kv.indptr[&location].to_vec1::<i32>().unwrap(),
        vec![0, 5]
    );
    assert_eq!(
        views.logical.paged_kv.indptr[&location]
            .to_vec1::<i32>()
            .unwrap(),
        vec![0, 33]
    );
}

#[test]
fn decode_context_floor_respects_small_caps_and_standard_metadata() {
    let mut short = decode_context_rows(1, 1);
    short.max_paged_context_len = 1024;
    let short_key = decode_context_key(short.clone());
    let mut later = decode_context_rows(1, 1024);
    later.max_paged_context_len = 1024;
    assert_eq!(short_key, decode_context_key(later));
    let mut standard = short.clone();
    standard.use_standard_metadata = true;
    let mut standard_later = decode_context_rows(1, 1025);
    standard_later.use_standard_metadata = true;
    standard_later.max_paged_context_len = 1024;
    assert_ne!(
        decode_context_key(standard),
        decode_context_key(standard_later)
    );
    let mut staged_decode = short;
    staged_decode.decode_window = 2;
    let mut staged_later = decode_context_rows(1, 1025);
    staged_later.decode_window = 2;
    staged_later.max_paged_context_len = 1024;
    assert_ne!(
        decode_context_key(staged_decode),
        decode_context_key(staged_later)
    );
}

#[test]
fn graph_buffers_share_unsliding_flashinfer_metadata() {
    let table = vec![1, 2, 3, 4];
    let metadata = Arc::new(DecodePagedRows {
        slot_mappings: vec![vec![127]],
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
        context_lens: vec![128],
        full_context_lens: vec![128],
        query_len: 1,
        block_size: 32,
        use_standard_metadata: false,
        max_paged_context_len: 1_604_288,
        sliding_window: None,
        decode_window: 1,
        devices: vec![Device::Cpu],
        num_kv_heads: 4,
    })
    .build()
    .unwrap();
    let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
    let key =
        CudaDecodeGraphKey::new(&input_ids, &metadata, 32, RecurrentBatchKind::Decode).unwrap();
    assert!(key
        .tensors
        .iter()
        .all(|tensor| !tensor.name.starts_with("full_")));

    let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
        metadata: &metadata,
        seqlen_offsets: &[127],
        position_ids: &[128],
        seq_len: 1,
        block_size: 32,
        kv_cache: &[],
        model_metadata: None,
        activation_dtype: DType::F32,
    })
    .unwrap();
    assert!(buffers.flashinfer_views_alias);
    let location = Device::Cpu.location();
    assert_eq!(
        buffers.paged_kv_indices.as_ref().unwrap()[&location].id(),
        buffers.full_paged_kv_indices.as_ref().unwrap()[&location].id()
    );
    assert_eq!(
        buffers.paged_kv_request_indices.as_ref().unwrap()[&location].id(),
        buffers.full_paged_kv_request_indices.as_ref().unwrap()[&location].id()
    );
}

#[test]
fn graph_rope_positions_expand_adjusted_ends_for_verification() {
    let table = vec![1, 2];
    let metadata = Arc::new(DecodePagedRows {
        slot_mappings: vec![vec![37, 38, 39], vec![37, 38, 39]],
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table.clone(), table], 3),
        context_lens: vec![38, 39, 40, 38, 39, 40],
        full_context_lens: vec![38, 39, 40, 38, 39, 40],
        query_len: 3,
        block_size: 32,
        use_standard_metadata: false,
        max_paged_context_len: 128,
        sliding_window: None,
        decode_window: 1,
        devices: vec![Device::Cpu],
        num_kv_heads: 4,
    })
    .build_materialized()
    .unwrap();

    let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
        metadata: &metadata,
        seqlen_offsets: &[97, 97],
        position_ids: &[100, 52],
        seq_len: 3,
        block_size: 32,
        kv_cache: &[],
        model_metadata: None,
        activation_dtype: DType::F32,
    })
    .unwrap();
    let positions = buffers.rope_positions[&Device::Cpu.location()]
        .as_detached_tensor()
        .to_vec1::<u32>()
        .unwrap();

    assert_eq!(positions, vec![97, 98, 99, 49, 50, 51]);
}

#[test]
fn graph_metadata_replay_updates_persistent_cuda_buffers() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let rows = |table: Vec<usize>, context_len: usize| -> anyhow::Result<_> {
        Ok(Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![i64::try_from(context_len - 1)?]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
            context_lens: vec![context_len],
            full_context_lens: vec![context_len],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 1_604_288,
            sliding_window: None,
            decode_window: 1,
            devices: vec![device.clone()],
            num_kv_heads: 4,
        }))
    };
    let initial = rows(vec![1, 2, 3, 4], 128)?.build_materialized()?;
    let updated = rows(vec![5, 6, 7, 8, 9, 10, 11, 12], 256)?.build()?;
    assert!(updated.has_host_staged_decode_tensors());

    let (mut buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
        metadata: &initial,
        seqlen_offsets: &[127],
        position_ids: &[128],
        seq_len: 1,
        block_size: 32,
        kv_cache: &[],
        model_metadata: None,
        activation_dtype: DType::F32,
    })?;
    let location = device.location();
    let state_indices = Var::from_tensor(&Tensor::zeros((3,), DType::U32, &device)?)?;
    let mut state_indices_map = HashMap::from([(location, state_indices)]);
    let mut host_staging = CudaGraphHostStaging::new(device.as_cuda_device()?.cuda_stream())?;
    host_staging.update(|host_staging| {
        buffers.copy_from(&updated, &[256], 1, host_staging)?;
        copy_state_indices(&state_indices_map, &[3, 5, 7], host_staging)
    })?;
    host_staging.update(|host_staging| {
        buffers.copy_from(&updated, &[256], 1, host_staging)?;
        copy_state_indices(&state_indices_map, &[11, 13, 17], host_staging)
    })?;
    device.synchronize()?;

    let indices = buffers.paged_kv_indices.as_ref().unwrap()[&location]
        .as_detached_tensor()
        .to_device(&Device::Cpu)?
        .to_vec1::<i32>()?;
    assert_eq!(&indices[..8], &[5, 6, 7, 8, 9, 10, 11, 12]);
    let positions = buffers.rope_positions[&location]
        .as_detached_tensor()
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    assert_eq!(positions, vec![255]);
    assert!(host_staging.buffers.len() > 1);
    assert_eq!(host_staging.completions.len(), 1);

    let state_indices = state_indices_map
        .remove(&location)
        .unwrap()
        .as_detached_tensor()
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    assert_eq!(state_indices, vec![11, 13, 17]);
    Ok(())
}

#[test]
fn graph_staging_orders_secondary_streams_both_directions() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let graph_stream = device.as_cuda_device()?.cuda_stream();
    let copy_stream = graph_stream.fork()?;
    assert!(!same_cuda_stream(&graph_stream, &copy_stream));
    let location = device.location();
    let mut staging = CudaGraphHostStaging::new(graph_stream.clone())?;

    staging.record_graph_complete()?;
    staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
    assert!(staging.completions[&location].ordered_after_graph);
    staging.order_before_graph()?;

    staging.record_graph_complete()?;
    staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
    assert!(staging.completions[&location].ordered_after_graph);
    staging.order_before_graph()?;
    graph_stream.synchronize()?;
    Ok(())
}

#[test]
fn graph_memory_pool_scope_restores_release_threshold() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let pool = cuda_memory_pool(&stream)?;
    let original = memory_pool_release_threshold(pool)?;

    let first = prepare_cuda_graph_memory_pool(&stream)?;
    let second = prepare_cuda_graph_memory_pool(&stream)?;
    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
    drop(first);
    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
    drop(second);
    assert_eq!(memory_pool_release_threshold(pool)?, original);
    Ok(())
}

#[test]
fn engine_memory_pool_retention_coordinates_nested_graph_scopes() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let alias = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let alias_stream = alias.as_cuda_device()?.cuda_stream();
    let pool = cuda_memory_pool(&stream)?;
    assert_eq!(cuda_memory_pool(&alias_stream)?, pool);
    let original = memory_pool_release_threshold(pool)?;

    for graph_first in [false, true] {
        for engines_drop_first in [false, true] {
            let prior_graph = graph_first
                .then(|| prepare_cuda_graph_memory_pool(&stream))
                .transpose()?;
            let first = retain_cuda_memory_pool(&stream)?;
            let second = retain_cuda_memory_pool(&alias_stream)?;
            assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
            assert_eq!(cuda_graph_memory_pool_scope_active(&device)?, graph_first);
            let outer = match prior_graph {
                Some(guard) => guard,
                None => prepare_cuda_graph_memory_pool(&stream)?,
            };
            let inner = prepare_cuda_graph_memory_pool(&alias_stream)?;
            assert!(cuda_graph_memory_pool_scope_active(&alias)?);

            if engines_drop_first {
                drop(second);
                assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                drop(first);
                assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                drop(outer);
                assert!(cuda_graph_memory_pool_scope_active(&device)?);
                assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                drop(inner);
            } else {
                drop(outer);
                assert!(cuda_graph_memory_pool_scope_active(&device)?);
                drop(inner);
                assert!(!cuda_graph_memory_pool_scope_active(&device)?);
                assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                drop(first);
                assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                drop(second);
            }
            assert!(!cuda_graph_memory_pool_scope_active(&device)?);
            assert_eq!(memory_pool_release_threshold(pool)?, original);
        }
    }
    Ok(())
}

#[test]
fn graph_memory_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
    let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
    trim_cuda_graph_memory(&stream)?;
    let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
    let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

    let pool = cuda_memory_pool(&stream)?;
    let original = memory_pool_release_threshold(pool)?;
    let retention = retain_cuda_memory_pool(&stream)?;
    let guard = prepare_cuda_graph_memory_pool(&stream)?;
    let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let output = input.as_detached_tensor().affine(2.0, 1.0)?;
    let graph = CudaGraphHandle::end_capture(&stream)?
        .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
    restore_event_tracking_after_capture(&stream, restore_event_tracking);
    graph.upload()?;
    graph.launch()?;
    stream.synchronize()?;
    drop(output);
    stream.synchronize()?;
    drop(graph);
    drop(guard);
    assert!(!cuda_graph_memory_pool_scope_active(&device)?);
    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);

    trim_cuda_graph_memory(&stream)?;
    assert_eq!(
        cuda_graph_memory_attribute(&stream, used_attribute)?,
        used_before
    );
    assert_eq!(
        cuda_graph_memory_attribute(&stream, reserved_attribute)?,
        reserved_before
    );
    drop(retention);
    assert_eq!(memory_pool_release_threshold(pool)?, original);
    Ok(())
}

#[test]
fn unlaunched_graph_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
    let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
    trim_cuda_graph_memory(&stream)?;
    let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
    let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

    let guard = prepare_cuda_graph_memory_pool(&stream)?;
    let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let logits = input.as_detached_tensor().affine(2.0, 1.0)?;
    let graph = CudaGraphHandle::end_capture(&stream)?
        .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
    restore_event_tracking_after_capture(&stream, restore_event_tracking);
    graph.upload()?;

    let mut release_result = Ok(());
    drop_cuda_graph_entry_output(logits, &stream, 0, "logits", &mut release_result);
    release_result?;
    drop(graph);
    drop(guard);
    trim_cuda_graph_memory(&stream)?;

    assert_eq!(
        cuda_graph_memory_attribute(&stream, used_attribute)?,
        used_before
    );
    assert_eq!(
        cuda_graph_memory_attribute(&stream, reserved_attribute)?,
        reserved_before
    );
    Ok(())
}

#[test]
fn graph_replay_retains_captured_output_storage() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let cuda_device = device.as_cuda_device()?;
    let stream = cuda_device.cuda_stream();
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

    let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
    let warmup = input.as_detached_tensor().affine(2.0, 1.0)?;
    device.synchronize()?;
    drop(warmup);

    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let output = input.as_detached_tensor().affine(2.0, 1.0)?;
    let graph = CudaGraphHandle::end_capture(&stream)?
        .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
    restore_event_tracking_after_capture(&stream, restore_event_tracking);
    graph.upload()?;

    for (values, expected) in [
        (vec![3f32, 5.0], vec![7f32, 11.0]),
        (vec![8f32, 13.0], vec![17f32, 27.0]),
    ] {
        input.set(&Tensor::from_vec(values, 2, &device)?)?;
        graph.launch()?;
        stream.synchronize()?;
        assert_eq!(output.to_vec1::<f32>()?, expected);
    }

    drop(graph);
    drop(output);
    device.synchronize()?;
    Ok(())
}

#[test]
fn replayed_graph_output_releases_without_driver_error() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
    let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;

    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let output = input.as_detached_tensor().affine(2.0, 1.0)?;
    let graph = CudaGraphHandle::end_capture(&stream)?
        .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
    restore_event_tracking_after_capture(&stream, restore_event_tracking);
    graph.upload()?;
    graph.launch()?;
    stream.synchronize()?;

    let mut release_result = Ok(());
    drop_cuda_graph_entry_output(output, &stream, 1, "test output", &mut release_result);
    release_result?;
    drop(graph);
    Ok(())
}

#[test]
fn graph_copy_supports_dense_row_source() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let cuda_device = device.as_cuda_device()?;
    let stream = cuda_device.cuda_stream();
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

    let input = Var::from_tensor(&Tensor::from_vec(
        (1u16..=16).map(f32::from).collect::<Vec<_>>(),
        (2, 2, 4),
        &device,
    )?)?;
    let output = Var::from_tensor(&Tensor::zeros((2, 2, 2), DType::F32, &device)?)?;
    let source = input.as_detached_tensor().narrow(2, 1, 2)?;
    assert!(!source.is_contiguous());

    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    crate::cuda::graph::copy_tensor(&source, &output.as_detached_tensor())?;
    let graph = CudaGraphHandle::end_capture(&stream)?
        .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
    restore_event_tracking_after_capture(&stream, restore_event_tracking);
    graph.upload()?;

    graph.launch()?;
    stream.synchronize()?;
    assert_eq!(
        output.as_detached_tensor().to_vec3::<f32>()?,
        vec![
            vec![vec![2.0, 3.0], vec![6.0, 7.0]],
            vec![vec![10.0, 11.0], vec![14.0, 15.0]],
        ]
    );

    input.set(&Tensor::from_vec(
        (1u16..=16).rev().map(f32::from).collect::<Vec<_>>(),
        (2, 2, 4),
        &device,
    )?)?;
    graph.launch()?;
    stream.synchronize()?;
    assert_eq!(
        output.as_detached_tensor().to_vec3::<f32>()?,
        vec![
            vec![vec![15.0, 14.0], vec![11.0, 10.0]],
            vec![vec![7.0, 6.0], vec![3.0, 2.0]],
        ]
    );
    Ok(())
}

#[test]
fn graph_suspension_does_not_clear_permanent_disable() {
    let mut state = CudaDecodeGraphState::default();
    state.suspend();
    assert!(state.disabled());
    state.resume();
    assert!(!state.disabled());
    state.disable();
    state.suspend();
    state.resume();
    assert!(state.disabled());
}

#[test]
fn eager_retry_block_applies_to_one_failed_step() {
    let mut state = CudaDecodeGraphState::default();
    assert!(state.take_eager_retry_allowed());
    state.block_eager_retry();
    assert!(!state.take_eager_retry_allowed());
    assert!(state.take_eager_retry_allowed());
    state.block_eager_retry();
    state.clear();
    assert!(state.take_eager_retry_allowed());
}

#[test]
fn graph_generations_are_not_reused_after_cleanup() {
    let mut state = CudaDecodeGraphState::default();
    let first = state.allocate_generation();
    state.clear();
    let second = state.allocate_generation();
    state.suspend();
    state.resume();
    let third = state.allocate_generation();
    assert!(first < second && second < third);
    assert!(!cuda_graph_replay_version_matches(second, 0, first, 1));
    assert!(!cuda_graph_replay_version_matches(third, 0, second, 1));
    assert!(cuda_graph_replay_version_matches(third, 7, third, 7));
}

#[test]
fn memory_pressure_eviction_drains_lru_entries_in_one_batch() {
    let mut entries = vec![10, 20, 30, 40];
    assert!(drain_lru_entries(&mut entries, 0).is_empty());
    assert_eq!(drain_lru_entries(&mut entries, 2), vec![10, 20]);
    assert_eq!(entries, vec![30, 40]);
    assert_eq!(drain_lru_entries(&mut entries, usize::MAX), vec![30, 40]);
    assert!(entries.is_empty());
}

#[test]
fn graph_reclaim_quota_is_shared_between_target_and_dflash() {
    let mut target_entries = 2usize;
    let mut dflash_entries = 4usize;
    let reclaimed = reclaim_cuda_graph_entries(
        3,
        |limit| {
            let reclaimed = target_entries.min(limit);
            target_entries -= reclaimed;
            reclaimed
        },
        |limit| {
            let reclaimed = dflash_entries.min(limit);
            dflash_entries -= reclaimed;
            reclaimed
        },
    );
    assert_eq!(reclaimed, 3);
    assert_eq!(target_entries, 0);
    assert_eq!(dflash_entries, 3);

    let mut dflash_called = false;
    assert_eq!(
        reclaim_cuda_graph_entries(
            2,
            |_| 2,
            |_| {
                dflash_called = true;
                0
            }
        ),
        2
    );
    assert!(!dflash_called);

    let mut target_called = false;
    assert_eq!(
        reclaim_cuda_graph_entries(
            0,
            |_| {
                target_called = true;
                0
            },
            |_| 0,
        ),
        0
    );
    assert!(!target_called);
}

#[test]
fn speculative_state_budget_scales_with_the_largest_entry() {
    assert_eq!(default_spec_state_budget(10_000, 0), 400);
    assert_eq!(default_spec_state_budget(10_000, 50), 500);
    assert_eq!(default_spec_state_budget(10_000, 100), 800);
    assert_eq!(default_spec_state_budget(10_000, 200), 800);
    assert_eq!(default_spec_state_budget(10_000, 300), 800);
    assert_eq!(default_spec_state_budget(10_000, 900), 900);
}

#[test]
fn only_unmaterialized_graph_output_frees_ignore_invalid_value() {
    assert!(unmaterialized_graph_output_drop_error(
        0,
        sys::CUresult::CUDA_ERROR_INVALID_VALUE
    ));
    assert!(!unmaterialized_graph_output_drop_error(
        1,
        sys::CUresult::CUDA_ERROR_INVALID_VALUE
    ));
    assert!(!unmaterialized_graph_output_drop_error(
        0,
        sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
    ));
}

#[test]
fn speculative_state_budget_evicts_lru_entries_per_device() {
    let gpu0 = DeviceLocation::Cuda { gpu_id: 0 };
    let gpu1 = DeviceLocation::Cuda { gpu_id: 1 };
    let budgets = HashMap::from([(gpu0, 100), (gpu1, 100)]);
    let existing = vec![
        spec_usage(&[]),
        spec_usage(&[(0, 40)]),
        spec_usage(&[(1, 60)]),
    ];

    assert_eq!(
        spec_state_eviction_plan(&existing, &spec_usage(&[(1, 50)]), &budgets),
        vec![2]
    );
    assert_eq!(
        spec_state_eviction_plan(&existing, &spec_usage(&[(0, 120)]), &budgets),
        vec![1]
    );
    let existing = vec![
        spec_usage(&[(0, 30)]),
        spec_usage(&[(0, 50)]),
        spec_usage(&[(0, 10)]),
    ];
    assert_eq!(
        spec_state_eviction_plan(&existing, &spec_usage(&[(0, 60)]), &budgets),
        vec![0, 1]
    );
}

#[test]
fn one_token_continuation_advances_host_decode_state() -> anyhow::Result<()> {
    let rows = Arc::new(
        DecodePagedRows {
            slot_mappings: vec![vec![39]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![9, 17]], 1),
            context_lens: vec![4],
            full_context_lens: vec![4],
            query_len: 1,
            block_size: 4,
            use_standard_metadata: false,
            max_paged_context_len: 64,
            sliding_window: Some(4),
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 1,
        }
        .padded(2),
    );
    let step = CudaGraphDecodeStep {
        input_ids: Tensor::zeros((2, 1), DType::U32, &Device::Cpu)?,
        seqlen_offsets: vec![100, 100],
        context_lens: vec![(0, 1), (0, 1)],
        position_ids: vec![43, 43],
        metadata: rows.build_graph_staged()?,
        state_indices: Some(vec![2, 5]),
        real_batch: 1,
    };
    let continuation = step
        .one_token_continuation(step.input_ids.clone())?
        .expect("next allocated block should permit one token");
    let rows = continuation.metadata.decode_rows.as_ref().unwrap();
    assert_eq!(rows.slot_mappings, vec![vec![68], vec![_PAD_SLOT_ID]]);
    assert_eq!(
        rows.materialized_block_tables(),
        vec![vec![9, 17], vec![9, 17]]
    );
    assert_eq!(rows.context_lens, vec![5, 5]);
    assert_eq!(rows.full_context_lens, vec![5, 5]);
    assert_eq!(continuation.seqlen_offsets, vec![101, 101]);
    assert_eq!(continuation.context_lens, vec![(0, 1), (0, 1)]);
    assert_eq!(continuation.position_ids, vec![44, 44]);
    assert_eq!(continuation.state_indices, Some(vec![2, 5]));
    Ok(())
}

#[test]
fn one_token_continuation_requires_an_allocated_boundary_slot() -> anyhow::Result<()> {
    let rows = Arc::new(DecodePagedRows {
        slot_mappings: vec![vec![39]],
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![9]], 1),
        context_lens: vec![4],
        full_context_lens: vec![4],
        query_len: 1,
        block_size: 4,
        use_standard_metadata: false,
        max_paged_context_len: 64,
        sliding_window: None,
        decode_window: 1,
        devices: vec![Device::Cpu],
        num_kv_heads: 1,
    });
    let step = CudaGraphDecodeStep {
        input_ids: Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?,
        seqlen_offsets: vec![3],
        context_lens: vec![(0, 1)],
        position_ids: vec![4],
        metadata: rows.build_graph_staged()?,
        state_indices: None,
        real_batch: 1,
    };
    assert!(step
        .one_token_continuation(step.input_ids.clone())?
        .is_none());
    Ok(())
}

#[test]
fn resident_replay_skips_the_host_input_update() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let metadata = Arc::new(DecodePagedRows {
        slot_mappings: vec![vec![0]],
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![0]], 1),
        context_lens: vec![1],
        full_context_lens: vec![1],
        query_len: 1,
        block_size: 32,
        use_standard_metadata: false,
        max_paged_context_len: 32,
        sliding_window: None,
        decode_window: 1,
        devices: vec![device.clone()],
        num_kv_heads: 1,
    })
    .build_materialized()?;
    let initial_ids = Tensor::from_vec(vec![1u32], (1, 1), &device)?;
    let key = CudaDecodeGraphKey::new(&initial_ids, &metadata, 32, RecurrentBatchKind::Decode)?;
    let warmup_logits = initial_ids.to_dtype(DType::F32)?;
    let entry = capture_cuda_decode_graph(
        CudaDecodeGraphCaptureCtx {
            key: key.clone(),
            input_ids: &initial_ids,
            seqlen_offsets: &[0],
            position_ids: &[1],
            block_size: 32,
            kv_cache: &[],
            metadata: &metadata,
            model_metadata: None,
            activation_dtype: DType::F32,
            warmup_logits: &warmup_logits,
            state_indices: None,
            real_batch: 1,
        },
        |input_ids, _| input_ids.to_dtype(DType::F32),
    )?;
    let mut state = CudaDecodeGraphState::default();
    state.insert(entry);

    let step = |token: u32| -> candle_core::Result<CudaGraphDecodeStep> {
        Ok(CudaGraphDecodeStep {
            input_ids: Tensor::from_vec(vec![token], (1, 1), &device)?,
            seqlen_offsets: vec![0],
            context_lens: vec![(0, 1)],
            position_ids: vec![1],
            metadata: metadata.clone(),
            state_indices: None,
            real_batch: 1,
        })
    };
    let host_step = step(7)?;
    let host_replay = state
        .replay(&key, &host_step, CudaDecodeGraphReplayInput::Host)?
        .expect("captured graph missing");
    let launch = host_replay.launch.expect("qlen=1 launch missing");
    launch.graph_stream().synchronize()?;
    assert_eq!(host_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);

    let resident_step = step(13)?;
    let resident_replay = state
        .replay(
            &key,
            &resident_step,
            CudaDecodeGraphReplayInput::Resident(&launch),
        )?
        .expect("resident graph entry changed");
    resident_replay
        .launch
        .as_ref()
        .unwrap()
        .graph_stream()
        .synchronize()?;
    assert_eq!(resident_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
    let resident_launch = resident_replay.launch.unwrap();
    assert_eq!(resident_launch.generation(), launch.generation());
    assert!(resident_launch.one_token_continuation()?.is_some());
    assert!(state.replay_one_token(launch)?.is_none());
    let lookahead = state
        .replay_one_token(resident_launch)?
        .expect("one-token continuation should retain the graph key");
    lookahead
        .launch
        .as_ref()
        .unwrap()
        .graph_stream()
        .synchronize()?;
    assert_eq!(lookahead.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
    Ok(())
}
