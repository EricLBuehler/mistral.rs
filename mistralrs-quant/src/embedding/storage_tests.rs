use super::*;

const DEVICE_TEST_ROWS: usize = 2;
const DEVICE_TEST_COLUMNS: usize = 2;
#[cfg(feature = "cuda")]
const MAPPED_RELEASE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
#[cfg(feature = "cuda")]
const MAPPED_RELEASE_POLL: std::time::Duration = std::time::Duration::from_millis(1);
#[cfg(feature = "cuda")]
const LIFETIME_ROWS: usize = 1024;
#[cfg(feature = "cuda")]
const LIFETIME_COLUMNS: usize = 64;
#[cfg(feature = "cuda")]
const FALLBACK_ROWS: usize = 4;
#[cfg(feature = "cuda")]
const FALLBACK_COLUMNS: usize = 2 * SYMMETRIC_GROUP;
#[cfg(feature = "cuda")]
const FALLBACK_IDS: [u32; 4] = [3, 0, 3, 1];

fn device_test_rows() -> MappedRows {
    let owner = Arc::new([1f32, 2., 3., 4.]);
    let shard = MappedShard {
        data: owner.as_ptr().cast(),
        scales: std::ptr::null(),
        first_row: 0,
        rows: DEVICE_TEST_ROWS,
    };
    unsafe { MappedRows::new(vec![shard], owner) }
}

#[test]
fn mapped_device_index_rejects_cpu() {
    let rows = device_test_rows();
    assert!(
        EmbeddingTable::mapped(RowCodec::Dense(DType::F32), DEVICE_TEST_COLUMNS, &rows,)
            .gathers_on(&Device::Cpu)
    );
    assert!(rows.with_device_index(&Device::Cpu).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn mapped_device_index_dispatch_checks_device() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let other_device = Device::new_cuda_with_stream(0)?;
    let rows = device_test_rows();
    let ids = Tensor::new(&[1u32, 0], &other_device)?;
    let gather = |rows: &MappedRows| -> Result<()> {
        let table = EmbeddingTable::mapped(RowCodec::Dense(DType::F32), DEVICE_TEST_COLUMNS, rows);
        assert!(!table.gathers_on(&other_device));
        let output = table.gather(&ids, DType::F32.into())?;
        assert!(output.device().same_device(&other_device));
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![3., 4.], vec![1., 2.]]);
        Ok(())
    };
    gather(&rows)?;
    if !device_reads_pageable_memory(&device)? {
        assert!(rows.with_device_index(&device).is_err());
        return Ok(());
    }
    let rows = rows.with_device_index(&device)?;
    assert!(
        EmbeddingTable::mapped(RowCodec::Dense(DType::F32), DEVICE_TEST_COLUMNS, &rows,)
            .gathers_on(&device)
    );
    gather(&rows)
}

#[cfg(feature = "cuda")]
fn mapped_lifetime_table(
    device: &Device,
) -> Result<(
    tempfile::TempDir,
    ManagedEmbedding,
    std::sync::Weak<MmapedSafetensors>,
)> {
    let dir = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
    let path = dir.path().join("weight.safetensors");
    candle_core::safetensors::save(
        &std::collections::HashMap::from([(
            "weight".to_owned(),
            Tensor::ones((LIFETIME_ROWS, LIFETIME_COLUMNS), DType::F32, &Device::Cpu)?,
        )]),
        &path,
    )?;
    let raw = Arc::new(unsafe { MmapedSafetensors::new(&path)? });
    let owner = Arc::downgrade(&raw);
    let source = EmbeddingSource::dense_shards(raw, vec!["weight".to_owned()], LIFETIME_COLUMNS)?;
    Ok((dir, ManagedEmbedding::new(source, device)?, owner))
}

#[cfg(feature = "cuda")]
#[test]
fn unified_dense_fallback_preserves_mapped_rows() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cuda_device::cuda_device_info(device.as_cuda_device()?).integrated {
        return Ok(());
    }
    let ids = Tensor::new(&FALLBACK_IDS, &Device::Cpu)?;
    let device_ids = ids.to_device(&device)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let weights = Tensor::from_vec(
            (0..FALLBACK_ROWS * FALLBACK_COLUMNS)
                .map(|i| (i % 17) as f32 / 8.0 - 1.0)
                .collect::<Vec<_>>(),
            (FALLBACK_ROWS, FALLBACK_COLUMNS),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?;
        let expected = (weights.index_select(&ids, 0)?.to_dtype(DType::F32)? * 1.5)?
            .to_dtype(DType::BF16)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        let dir = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
        let path = dir.path().join("weight.safetensors");
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("weight", weights)]),
            &path,
        )?;
        let raw = Arc::new(unsafe { MmapedSafetensors::new(&path)? });
        for allow_resident in [false, true] {
            let source = EmbeddingSource::dense_shards(
                raw.clone(),
                vec!["weight".to_owned()],
                FALLBACK_COLUMNS,
            )?
            .with_packing(EmbeddingPacking::Symmetric {
                preferred_bits: 8,
                fallback_bits: 4,
            })?;
            assert_eq!(source.metadata().host_backing, EmbeddingBacking::FileMapped);
            let mut table = ManagedEmbedding::new(source, &device)?;
            table.prepare(
                &device,
                &|_| {
                    assert!(
                        allow_resident,
                        "nonresident choice queried a residency budget"
                    );
                    Ok(WeightAllocationBudget {
                        persistent_bytes: 0,
                        execution_bytes: 0,
                    })
                },
                allow_resident,
                true,
            )?;
            assert!(table.resident.is_none());
            assert!(matches!(table.source, TableSource::Shards { .. }));
            assert!(table.reclaimable_weight_cache_bytes().is_none());
            for mode in [EmbeddingGatherMode::Prefill, EmbeddingGatherMode::Decode] {
                let output = table.gather(
                    &device_ids,
                    super::super::EmbeddingOutput::scaled(DType::BF16, 1.5),
                    mode,
                )?;
                assert_eq!(output.flatten_all()?.to_vec1::<half::bf16>()?, expected);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mapped_table_drop_waits_for_eager_cuda_reads() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !device_reads_pageable_memory(&device)? {
        return Ok(());
    }
    let (_dir, table, owner) = mapped_lifetime_table(&device)?;
    let ids = Tensor::new(&[0u32, (LIFETIME_ROWS - 1) as u32], &device)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let _delay = crate::utils::cuda_test::delay_stream(&stream)?;
    let output = table.gather(&ids, DType::F32.into(), EmbeddingGatherMode::Decode)?;
    drop(table);
    assert!(owner.upgrade().is_none());
    assert_eq!(
        output.to_vec2::<f32>()?,
        vec![vec![1.; LIFETIME_COLUMNS]; 2]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn captured_mapped_table_outlives_its_owner_until_graph_release() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let device = Device::new_cuda_with_stream(0)?;
    if !device_reads_pageable_memory(&device)? {
        return Ok(());
    }
    let (_dir, table, owner) = mapped_lifetime_table(&device)?;
    let ids = Tensor::new(&[0u32, (LIFETIME_ROWS - 1) as u32], &device)?;
    let gather = || table.gather(&ids, DType::F32.into(), EmbeddingGatherMode::Decode);
    drop(gather()?);
    device.synchronize()?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    if let Err(error) =
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    {
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        return Err(candle_core::Error::wrap(error));
    }
    let output = gather().and_then(|first| first + gather()?);
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = output?;
    let graph = graph
        .map_err(candle_core::Error::wrap)?
        .expect("captured mapped gathers");
    drop(table);
    assert!(owner.upgrade().is_some());
    for _ in 0..2 {
        graph.launch().map_err(candle_core::Error::wrap)?;
        assert_eq!(
            output.to_vec2::<f32>()?,
            vec![vec![2.; LIFETIME_COLUMNS]; 2]
        );
    }
    drop(output);
    drop(graph);
    let started = std::time::Instant::now();
    while owner.upgrade().is_some() && started.elapsed() < MAPPED_RELEASE_TIMEOUT {
        std::thread::sleep(MAPPED_RELEASE_POLL);
    }
    assert!(
        owner.upgrade().is_none(),
        "graph release kept the mapped table alive"
    );
    Ok(())
}
