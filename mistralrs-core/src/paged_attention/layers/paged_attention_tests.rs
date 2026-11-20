use super::*;
use candle_core::D;

#[test]
fn cumulative_seqlens_match_checked_host_token_count() -> Result<()> {
    let lengths = [3usize, 0, 5];
    let num_tokens = checked_sequence_token_count(&lengths)?;
    let cu_seqlens = cumulative_seqlens_from_lengths(&lengths, &Device::Cpu)?;

    assert_eq!(num_tokens, 8);
    assert_eq!(cu_seqlens.to_vec1::<u32>()?, vec![0, 3, 3, 8]);
    Ok(())
}

#[test]
fn checked_sequence_token_count_rejects_invalid_kernel_sizes() {
    let kernel_limit = usize::try_from(i32::MAX).unwrap();
    assert!(checked_sequence_token_count(&[kernel_limit, 1]).is_err());
    assert!(checked_sequence_token_count(&[usize::MAX, 1]).is_err());
}

#[cfg(cuda_unix)]
fn supported_fa3_decode_candidate() -> Fa3DecodeCandidate {
    Fa3DecodeCandidate {
        key: Fa3PagedScheduleShape {
            device: DeviceLocation::Cuda { gpu_id: 0 },
            view: Fa3DecodeView::Logical,
            batch: 8,
            query_len: 1,
            causal: false,
            q_heads: 16,
            kv_heads: 4,
            head_dim: 256,
            page_size: 32,
        }
        .decode_schedule_key()
        .expect("supported FA3 decode schedule"),
        query_dtype: DType::BF16,
        query_contiguous: true,
        key_cache_dtype: DType::F8E4M3,
        value_cache_dtype: DType::F8E4M3,
        mask_is_none: true,
        shapes_match: true,
        has_alibi: false,
        has_sinks: false,
        has_softcap: false,
        has_sliding_window: false,
        has_noncausal_mm_context: false,
    }
}

#[cfg(cuda_unix)]
#[test]
fn fa3_decode_candidate_requires_supported_full_decode() {
    let supported = supported_fa3_decode_candidate();
    assert_eq!(supported.schedule_key(), Some(supported.key));

    macro_rules! reject {
        ($field:ident, $value:expr) => {{
            let mut candidate = supported;
            candidate.$field = $value;
            assert!(candidate.schedule_key().is_none());
        }};
    }

    reject!(query_dtype, DType::F16);
    reject!(query_contiguous, false);
    reject!(key_cache_dtype, DType::BF16);
    reject!(value_cache_dtype, DType::BF16);
    reject!(mask_is_none, false);
    reject!(shapes_match, false);
    reject!(has_alibi, true);
    reject!(has_sinks, true);
    reject!(has_softcap, true);
    reject!(has_sliding_window, true);
    reject!(has_noncausal_mm_context, true);

    let mut unsupported_shape = supported;
    unsupported_shape.key.head_dim = 128;
    assert!(unsupported_shape.schedule_key().is_none());

    let mut speculative = supported;
    speculative.key.query_len = 8;
    speculative.key.causal = true;
    assert_eq!(speculative.schedule_key(), Some(speculative.key));
}

#[cfg(cuda_unix)]
#[test]
fn fa3_decode_schedule_lookup_is_exact() {
    let candidate = supported_fa3_decode_candidate();
    let schedules = HashMap::from([(candidate.key, ())]);
    assert!(schedules.contains_key(&candidate.schedule_key().unwrap()));

    let mut different_batch = candidate;
    different_batch.key.batch /= 2;
    assert!(!schedules.contains_key(&different_batch.schedule_key().unwrap()));

    let mut different_heads = candidate;
    different_heads.key.q_heads /= 2;
    assert!(!schedules.contains_key(&different_heads.schedule_key().unwrap()));

    let mut different_query = candidate;
    different_query.key.query_len = 8;
    different_query.key.causal = true;
    assert!(!schedules.contains_key(&different_query.schedule_key().unwrap()));
}

#[test]
fn owns_validated_fp8_attention_scales() -> Result<()> {
    let scales = Fp8AttentionScales {
        q: 0.25,
        k: 0.5,
        v: 0.75,
    };
    let attention =
        PagedAttention::new_with_fp8_attention_scales(128, &Device::Cpu, None, Some(scales))?;
    assert_eq!(attention.fp8_attention_scales(), scales);
    assert!(attention.has_calibrated_fp8_attention_scales());
    let [q_scale, k_scale, v_scale] = attention.fp8_scale_tensors();
    assert_eq!(q_scale.to_scalar::<f32>()?, scales.q);
    assert_eq!(k_scale.to_scalar::<f32>()?, scales.k);
    assert_eq!(v_scale.to_scalar::<f32>()?, scales.v);
    #[cfg(cuda_unix)]
    {
        let fp8_cache = Tensor::zeros((1,), DType::F8E4M3, &Device::Cpu)?;
        let flashinfer_scales = attention.cache_scales(&fp8_cache).flashinfer(&fp8_cache);
        assert_eq!(flashinfer_scales.k, scales.k);
        assert_eq!(flashinfer_scales.v, scales.v);
    }

    let default = PagedAttention::new(128, &Device::Cpu, None)?;
    assert_eq!(default.fp8_attention_scales(), Fp8AttentionScales::UNIT);
    assert!(!default.has_calibrated_fp8_attention_scales());

    assert!(PagedAttention::new_with_fp8_attention_scales(
        128,
        &Device::Cpu,
        None,
        Some(Fp8AttentionScales {
            q: 1.0,
            k: 0.0,
            v: 1.0,
        }),
    )
    .is_err());
    Ok(())
}

#[test]
fn cache_write_accepts_row_strided_dense_heads() -> Result<()> {
    let packed = Tensor::zeros((2, 3, 16), DType::F32, &Device::Cpu)?;
    let row_strided = packed.narrow(D::Minus1, 4, 8)?.unfold(D::Minus1, 4, 4)?;
    assert_eq!(row_strided.dims(), &[2, 3, 2, 4]);
    assert_eq!(row_strided.stride(), &[48, 16, 4, 1]);
    assert_eq!(cache_input_shape(&row_strided)?, (6, 2, 4));
    assert_eq!(
        cache_input_can_write_directly(&row_strided)?,
        cfg!(cuda_unix)
    );

    let row_strided = packed
        .narrow(1, 0, 1)?
        .squeeze(1)?
        .narrow(D::Minus1, 4, 8)?
        .unfold(D::Minus1, 4, 4)?;
    assert_eq!(row_strided.stride(), &[48, 4, 1]);
    assert!(cache_input_can_write_directly(&row_strided)?);
    Ok(())
}

#[test]
fn cache_write_packs_singleton_token_stride() -> Result<()> {
    let source = Tensor::zeros((8, 1, 128), DType::F32, &Device::Cpu)?;
    let singleton = source.transpose(0, 1)?;
    assert_eq!(singleton.dims(), &[1, 8, 128]);
    assert!(!cache_input_can_write_directly(&singleton)?);

    let packed = singleton.force_contiguous()?;
    assert_eq!(packed.stride(), &[1024, 128, 1]);
    assert!(cache_input_can_write_directly(&packed)?);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_cache_packing_preserves_offset_and_strided_inputs() -> Result<()> {
    const HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const BLOCK_SIZE: usize = 32;
    const BLOCKS: usize = 2;
    const PACK: usize = 8;
    const SOURCE_BATCH: usize = 3;
    const VALUE_PERIOD: usize = 97;
    const VALUE_MIDPOINT: f32 = 48.;
    const SENTINEL: f32 = -97.;
    const SLOTS: [i64; 4] = [39, 2, 63, 32];

    let device = Device::new_metal(0)?;
    for dtype in [DType::BF16, DType::F16] {
        for query_len in [1, 2] {
            let source_shape = (SOURCE_BATCH, HEADS, query_len, HEAD_DIM);
            let elements = SOURCE_BATCH * HEADS * query_len * HEAD_DIM;
            let source = |factor: usize| -> Result<Tensor> {
                let data = (0..elements)
                    .map(|index| {
                        f32::from(u8::try_from((index * factor) % VALUE_PERIOD).unwrap())
                            - VALUE_MIDPOINT
                    })
                    .collect::<Vec<_>>();
                Tensor::from_vec(data, source_shape, &device)?
                    .to_dtype(dtype)?
                    .narrow(0, 1, query_len)?
                    .transpose(1, 2)
            };
            let key = source(7)?;
            let value = source(11)?;
            assert!(!cache_input_can_write_directly(&key)?);
            assert!(!cache_input_can_write_directly(&value)?);
            assert_eq!(key.is_contiguous(), query_len == 1);
            assert_eq!(value.is_contiguous(), query_len == 1);
            assert!(key.layout().start_offset() > 0);
            assert!(value.layout().start_offset() > 0);
            let (tokens, _, _) = cache_input_shape(&key)?;
            let read =
                |tensor: &Tensor| tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
            let key_values = read(&key)?;
            let value_values = read(&value)?;
            let mut expected_key = vec![SENTINEL; BLOCKS * HEADS * HEAD_DIM * BLOCK_SIZE];
            let mut expected_value = expected_key.clone();
            let mut expected_hnd_key = expected_key.clone();
            let mut expected_hnd_value = expected_key.clone();
            for (token, &slot) in SLOTS.iter().take(tokens).enumerate() {
                let slot = usize::try_from(slot).unwrap();
                let (block, offset) = (slot / BLOCK_SIZE, slot % BLOCK_SIZE);
                for head in 0..HEADS {
                    for dim in 0..HEAD_DIM {
                        let source = (token * HEADS + head) * HEAD_DIM + dim;
                        let key_index = (((block * HEADS + head) * (HEAD_DIM / PACK) + dim / PACK)
                            * BLOCK_SIZE
                            + offset)
                            * PACK
                            + dim % PACK;
                        let value_index =
                            ((block * HEADS + head) * HEAD_DIM + dim) * BLOCK_SIZE + offset;
                        expected_key[key_index] = key_values[source];
                        expected_value[value_index] = value_values[source];
                        let hnd_index =
                            ((block * HEADS + head) * BLOCK_SIZE + offset) * HEAD_DIM + dim;
                        expected_hnd_key[hnd_index] = key_values[source];
                        expected_hnd_value[hnd_index] = value_values[source];
                    }
                }
            }
            let mut padded_slots = vec![-1];
            padded_slots.extend(SLOTS);
            let slots = Tensor::new(padded_slots, &device)?.narrow(0, 1, tokens)?;
            let packed_key = pack_cache_input(&key)?;
            let packed_value = pack_cache_input(&value)?;
            assert_eq!(packed_key.dims(), &[tokens, HEADS, HEAD_DIM]);
            assert_eq!(packed_key.stride(), &[HEADS * HEAD_DIM, HEAD_DIM, 1]);
            let expected_offset = if key.is_contiguous() {
                key.layout().start_offset()
            } else {
                0
            };
            assert_eq!(packed_key.layout().start_offset(), expected_offset);
            for hnd in [false, true] {
                let (key_shape, value_shape, expected_key, expected_value) = if hnd {
                    (
                        vec![BLOCKS, HEADS, BLOCK_SIZE, HEAD_DIM],
                        vec![BLOCKS, HEADS, BLOCK_SIZE, HEAD_DIM],
                        &expected_hnd_key,
                        &expected_hnd_value,
                    )
                } else {
                    (
                        vec![BLOCKS, HEADS, HEAD_DIM / PACK, BLOCK_SIZE, PACK],
                        vec![BLOCKS, HEADS, HEAD_DIM, BLOCK_SIZE],
                        &expected_key,
                        &expected_value,
                    )
                };
                let key_cache = Tensor::full(SENTINEL, key_shape, &device)?.to_dtype(dtype)?;
                let value_cache = Tensor::full(SENTINEL, value_shape, &device)?.to_dtype(dtype)?;
                assert_eq!(
                    AttentionBackendKind::from_cache(&key_cache, &value_cache),
                    AttentionBackendKind::Standard,
                );
                assert_eq!(cache_block_size(&key_cache, &value_cache)?, BLOCK_SIZE);
                assert_eq!(cache_kv_shape(&key_cache, &value_cache)?, (HEADS, HEAD_DIM));
                reshape_and_cache(
                    &packed_key,
                    &packed_value,
                    None,
                    None,
                    &key_cache,
                    &value_cache,
                    &slots,
                )?;
                assert_eq!(
                    &read(&key_cache)?,
                    expected_key,
                    "{dtype:?} query_len={query_len} hnd={hnd}: key cache mismatch"
                );
                assert_eq!(
                    &read(&value_cache)?,
                    expected_value,
                    "{dtype:?} query_len={query_len} hnd={hnd}: value cache mismatch"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn varlen_segments_partition_each_logical_query() {
    assert_eq!(
        validate_varlen_segment_partition(&[3, 2], &[2, 1, 2]).unwrap(),
        2
    );
}

#[test]
fn varlen_segments_reject_boundary_crossing_and_trailing_segments() {
    assert!(validate_varlen_segment_partition(&[3, 2], &[2, 2, 1]).is_err());
    assert!(validate_varlen_segment_partition(&[3, 2], &[2, 1, 2, 1]).is_err());
    assert!(validate_varlen_segment_partition(&[3, 2], &[2, 1]).is_err());
    assert!(validate_varlen_segment_partition(&[3, 0], &[2, 1]).is_err());
}

#[test]
fn ragged_decode_mask_excludes_padding() {
    let mask = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[1, 1],
            kv_lens: &[2, 4],
            mm_prefix_ranges: None,
            sliding_window: None,
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        1,
        4,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(mask) = mask else {
        panic!("expected custom mask");
    };
    let mask = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(
        mask,
        vec![
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            0.0
        ]
    );
}

#[cfg(feature = "metal")]
#[test]
fn metal_prefix_masks_reuse_uploads_without_reusing_different_windows() -> Result<()> {
    let device = Device::new_metal(0)?;
    let cache = PrefixMaskCache::default();
    let mm_prefix_ranges = [vec![(1, 3)], vec![]];
    let mut previous = None;
    for sliding_window in [None, Some(2)] {
        let request = PrefixMaskRequest {
            query_lens: &[2, 1],
            kv_lens: &[4, 3],
            mm_prefix_ranges: Some(&mm_prefix_ranges),
            sliding_window,
            dtype: DType::BF16,
            device: &device,
        };
        let AttentionMask::Custom(mask) = prefix_gather_causal_mask(request, 2, 4, &cache)? else {
            unreachable!()
        };
        let AttentionMask::Custom(reused) =
            prefix_gather_causal_mask(request, 2, 4, &cache.clone())?
        else {
            unreachable!()
        };
        assert_eq!(mask.id(), reused.id());
        assert_ne!(previous, Some(mask.id()));
        previous = Some(mask.id());
        let AttentionMask::Custom(expected) = prefix_gather_causal_mask(
            PrefixMaskRequest {
                device: &Device::Cpu,
                dtype: DType::F32,
                ..request
            },
            2,
            4,
            &PrefixMaskCache::default(),
        )?
        else {
            unreachable!()
        };
        assert_eq!(
            mask.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        let packed = packed_prefix_gather_causal_mask(request, &cache)?;
        let reused = packed_prefix_gather_causal_mask(request, &cache)?;
        assert_eq!(packed.id(), reused.id());
        assert_ne!(packed.id(), mask.id());
        let expected = packed_prefix_gather_causal_mask(
            PrefixMaskRequest {
                device: &Device::Cpu,
                dtype: DType::F32,
                ..request
            },
            &PrefixMaskCache::default(),
        )?;
        assert_eq!(
            packed
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
    }
    Ok(())
}

#[test]
fn ragged_prompt_mask_keeps_padding_rows_finite() {
    let mask = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[2],
            kv_lens: &[2],
            mm_prefix_ranges: None,
            sliding_window: None,
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        4,
        2,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(mask) = mask else {
        panic!("expected custom mask");
    };
    let mask = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(
        mask,
        vec![
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY
        ]
    );
}

#[test]
fn sliding_mask_uses_window_as_token_capacity() {
    let mask = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[1],
            kv_lens: &[6],
            mm_prefix_ranges: None,
            sliding_window: Some(4),
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        1,
        6,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(mask) = mask else {
        panic!("expected custom mask");
    };

    assert_eq!(
        mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn sliding_mask_moves_with_each_chunked_query_row() {
    let mask = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[3],
            kv_lens: &[8],
            mm_prefix_ranges: None,
            sliding_window: Some(4),
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        3,
        8,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(mask) = mask else {
        panic!("expected custom mask");
    };
    let mask = mask.squeeze(0).unwrap().squeeze(0).unwrap();

    for (row, visible_start) in [2, 3, 4].into_iter().enumerate() {
        let values = mask.get(row).unwrap().to_vec1::<f32>().unwrap();
        for (column, value) in values.into_iter().enumerate() {
            let visible = column >= visible_start && column <= row + 5;
            assert_eq!(value == 0.0, visible);
        }
    }
}

#[test]
fn sliding_mask_handles_unit_window_and_query_longer_than_window() {
    let unit = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[1],
            kv_lens: &[3],
            mm_prefix_ranges: None,
            sliding_window: Some(1),
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        1,
        3,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(unit) = unit else {
        panic!("expected custom mask");
    };
    assert_eq!(
        unit.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
    );

    let chunk = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[5],
            kv_lens: &[7],
            mm_prefix_ranges: None,
            sliding_window: Some(2),
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        5,
        7,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(chunk) = chunk else {
        panic!("expected custom mask");
    };
    let chunk = chunk.squeeze(0).unwrap().squeeze(0).unwrap();
    for row in 0..5 {
        let values = chunk.get(row).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values.iter().filter(|&&value| value == 0.0).count(), 2);
    }
}

#[test]
fn noncausal_media_range_overrides_both_sliding_window_edges() {
    let mask = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[2],
            kv_lens: &[8],
            mm_prefix_ranges: Some(&[vec![(1, 8)]]),
            sliding_window: Some(3),
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        2,
        8,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(mask) = mask else {
        panic!("expected custom mask");
    };
    let mask = mask.squeeze(0).unwrap().squeeze(0).unwrap();

    assert_eq!(
        mask.get(0).unwrap().to_vec1::<f32>().unwrap(),
        vec![f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        mask.get(1).unwrap().to_vec1::<f32>().unwrap(),
        vec![f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn packed_query_layout_is_dense_without_rectangular_lengths() {
    assert!(query_layout_is_dense(&[3, 5], 1, 8));
    assert!(query_layout_is_dense(&[5, 5], 2, 5));
    assert!(!query_layout_is_dense(&[3, 5], 2, 5));
}

#[test]
fn packed_prefix_selects_block_diagonal_mask_before_padded_mask() {
    assert!(should_use_packed_prefix_mask(
        true,
        true,
        &AttentionMask::CausalFlash,
        false
    ));
    assert!(!should_use_packed_prefix_mask(
        true,
        true,
        &AttentionMask::CausalFlash,
        true
    ));
    assert!(!should_use_packed_prefix_mask(
        false,
        true,
        &AttentionMask::CausalFlash,
        false
    ));
    assert!(!should_use_packed_prefix_mask(
        true,
        true,
        &AttentionMask::None,
        false
    ));
}

#[test]
fn cached_suffix_reconstructs_mask_and_decode_layout() {
    assert!(should_reconstruct_prefix_mask(
        &AttentionMask::None,
        true,
        false,
        false
    ));
    assert!(should_reconstruct_prefix_mask(
        &AttentionMask::None,
        false,
        true,
        false
    ));
    assert!(should_reconstruct_prefix_mask(
        &AttentionMask::None,
        false,
        false,
        true
    ));
    assert!(!should_reconstruct_prefix_mask(
        &AttentionMask::CausalFlash,
        true,
        false,
        true
    ));

    let output = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (1, 2, 3, 1), &Device::Cpu).unwrap();
    let output = prefix_attention_output_layout(output, &AttentionMask::None).unwrap();
    assert_eq!(output.dims(), &[1, 3, 2, 1]);
    assert_eq!(
        output.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        vec![0, 3, 1, 4, 2, 5]
    );
}

#[test]
fn single_token_suffix_mask_excludes_padding_and_sliding_history() {
    let mask = prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[1, 1],
            kv_lens: &[3, 5],
            mm_prefix_ranges: None,
            sliding_window: Some(3),
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        1,
        5,
        &PrefixMaskCache::default(),
    )
    .unwrap();
    let AttentionMask::Custom(mask) = mask else {
        panic!("expected custom mask");
    };
    assert_eq!(
        mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![
            0.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
        ]
    );
}

#[test]
fn gathered_decode_counts_each_query_token_as_a_row() {
    let query = Tensor::zeros((6, 4, 8), DType::F32, &Device::Cpu).unwrap();

    assert_eq!(
        decode_query_rows(&query, &[7, 8, 9, 10, 11, 12]).unwrap(),
        6
    );
    assert!(decode_query_rows(&query, &[7, 8]).is_err());
}

#[test]
fn packed_sinks_query_round_trips_through_padded_varlen_layout() {
    let query = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (1, 1, 6, 1), &Device::Cpu).unwrap();
    let padded = pad_packed_query(&query, &[2, 4]).unwrap();

    assert_eq!(padded.dims(), &[2, 1, 4, 1]);
    assert_eq!(
        padded.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        vec![0, 1, 0, 0, 2, 3, 4, 5]
    );
    let repacked = repack_padded_query(&padded, &[2, 4]).unwrap();
    assert_eq!(
        repacked.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
        query.flatten_all().unwrap().to_vec1::<u32>().unwrap()
    );
}

#[test]
fn packed_sinks_query_rejects_inconsistent_lengths() {
    let query = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();

    assert!(pad_packed_query(&query, &[2, 2]).is_err());
    assert!(pad_packed_query(&query, &[5, 0]).is_err());
}

#[test]
fn sinks_decode_uses_varlen_without_general_flash_support() {
    assert!(should_use_decode_gather_varlen(false, true, true, 2));
    assert!(!should_use_decode_gather_varlen(false, true, true, 1));
    assert!(!should_use_decode_gather_varlen(false, true, false, 2));
    assert!(!should_use_decode_gather_varlen(true, true, false, 2));
    assert!(!should_use_decode_gather_varlen(false, false, true, 2));
    assert!(should_use_decode_gather_varlen(true, false, false, 1));
}

#[test]
fn packed_varlen_requires_a_compatible_flash_backend() {
    assert!(!packed_varlen_flash_is_usable(
        false,
        true,
        DType::F16,
        128,
        false,
        false
    ));
    assert!(!packed_varlen_flash_is_usable(
        true,
        false,
        DType::F16,
        128,
        false,
        false
    ));
    assert!(!packed_varlen_flash_is_usable(
        true,
        true,
        DType::F32,
        128,
        false,
        false
    ));
    assert!(!packed_varlen_flash_is_usable(
        true,
        true,
        DType::F16,
        640,
        false,
        false
    ));
    assert!(!packed_varlen_flash_is_usable(
        true,
        true,
        DType::F16,
        512,
        true,
        false
    ));
    assert!(!packed_varlen_flash_is_usable(
        true,
        true,
        DType::F16,
        320,
        false,
        true
    ));
    assert_eq!(
        packed_varlen_flash_is_usable(true, true, DType::F16, 128, false, false),
        cfg!(any(feature = "flash-attn", feature = "flash-attn-v3"))
    );
}

#[test]
fn uncached_noncausal_prompt_uses_gather_path() {
    assert!(should_use_gather_path(true, false, true, true, false, true));
    assert!(!should_use_gather_path(
        true, false, false, false, false, true
    ));
}

#[test]
fn uncached_later_prompt_mask_uses_gather_path() {
    assert!(should_use_gather_path(
        true, false, false, true, false, true
    ));
    assert!(!should_use_gather_path(
        false, false, false, true, false, true
    ));
}

#[test]
fn exact_mm_ranges_keep_the_direct_prefix_path_causal() {
    assert!(prefix_prefill_is_causal(true, false, true));
    assert!(prefix_prefill_is_causal(true, true, false));
    assert!(!prefix_prefill_is_causal(true, false, false));
    assert!(!prefix_prefill_is_causal(false, true, true));
}

#[test]
fn full_mm_range_view_does_not_reuse_sliding_ranges() {
    let regular = 7;
    let selected = select_optional_view(true, None, Some(&regular));

    assert_eq!(selected, None);
    assert_eq!(
        select_optional_view(false, Some(&11), Some(&regular)),
        Some(&regular)
    );
}

#[test]
fn noncausal_mm_context_requires_the_selected_cache_view() {
    assert!(noncausal_mm_view_is_valid(false, false, false));
    assert!(noncausal_mm_view_is_valid(true, true, true));
    assert!(!noncausal_mm_view_is_valid(true, false, true));
    assert!(!noncausal_mm_view_is_valid(true, true, false));
}

#[test]
fn packed_custom_mask_preserves_boundaries_and_mm_prefixes() {
    let mask = packed_prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[2, 3],
            kv_lens: &[2, 3],
            mm_prefix_ranges: Some(&[vec![(0, 2)], vec![]]),
            sliding_window: Some(2),
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        &PrefixMaskCache::default(),
    )
    .unwrap()
    .squeeze(0)
    .unwrap()
    .squeeze(0)
    .unwrap()
    .to_vec2::<f32>()
    .unwrap();

    assert_eq!(
        mask[0],
        vec![
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY
        ]
    );
    assert_eq!(
        mask[2],
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY
        ]
    );
    assert_eq!(
        mask[4],
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0
        ]
    );
}

#[test]
fn overlapping_mm_ranges_do_not_merge_attention_groups() {
    let mask = packed_prefix_gather_causal_mask(
        PrefixMaskRequest {
            query_lens: &[6],
            kv_lens: &[6],
            mm_prefix_ranges: Some(&[vec![(0, 4), (2, 6)]]),
            sliding_window: None,
            dtype: DType::F32,
            device: &Device::Cpu,
        },
        &PrefixMaskCache::default(),
    )
    .unwrap()
    .squeeze(0)
    .unwrap()
    .squeeze(0)
    .unwrap()
    .to_vec2::<f32>()
    .unwrap();

    assert_eq!(
        mask[0],
        vec![0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
    );
    assert_eq!(mask[3], vec![0.0; 6]);
}
