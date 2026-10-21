use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor};

use super::{CacheElement, CacheKey, CachedRecurrentState, MatchingCache, PrefixCacheManagerV2};
use crate::{
    device_map::DeviceMapper,
    kv_cache::{
        CacheManager, EitherCache, KvCache, NormalCache, NormalCacheManager,
        RecurrentStateSnapshot, RotatingCache, SingleCache,
    },
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{CacheManagerMixin, GeneralMetadata, MetadataMixin, Modalities, ModelKind},
    sampler::Sampler,
    sequence::{SeqStepType, Sequence, SequenceGroup, SequenceRecognizer},
    AdapterGenerationId,
};

const TEST_MAX_SEQ_LEN: usize = 1024;
const TEST_KV_HEADS: usize = 2;
const TEST_KV_HEAD_DIM: usize = 2;
const TEST_PROMPT_LEN: usize = 4;

fn element(len: usize, device: &Device) -> Result<CacheElement> {
    let data_len = u16::try_from(len).unwrap();
    let data = Tensor::from_vec(
        (1..=data_len).map(f32::from).collect::<Vec<_>>(),
        (1, 1, len, 1),
        device,
    )?;
    let mut k = SingleCache::new(2, TEST_MAX_SEQ_LEN, len);
    let mut v = SingleCache::new(2, TEST_MAX_SEQ_LEN, len);
    k.append(&data)?;
    v.append(&data)?;
    Ok(CacheElement {
        cache: vec![Some(KvCache::Normal { k, v })],
        recurrent_snapshots: None,
        audio_hashes: None,
        image_hashes: None,
        video_hashes: None,
    })
}

fn manager(len: usize, device: &Device) -> Result<PrefixCacheManagerV2> {
    let mut manager = PrefixCacheManagerV2::new(16, false, false);
    manager.caches.insert(
        CacheKey::new((1..=u32::try_from(len).unwrap()).collect(), None),
        element(len, device)?,
    );
    Ok(manager)
}

fn search(manager: &mut PrefixCacheManagerV2, tokens: &[u32]) -> Result<Option<MatchingCache>> {
    manager.search_for_matching_cache(tokens, None, &[], None, None, None)
}

fn hit(manager: &mut PrefixCacheManagerV2, tokens: &[u32]) -> Result<MatchingCache> {
    Ok(search(manager, tokens)?.expect("expected normal text prefix hit"))
}

fn values(cache: &KvCache) -> Result<Vec<f32>> {
    let k = cache
        .k()?
        .expect("normal K data")
        .flatten_all()?
        .to_vec1()?;
    let v: Vec<f32> = cache
        .v()?
        .expect("normal V data")
        .flatten_all()?
        .to_vec1()?;
    assert_eq!(k, v);
    Ok(k)
}

#[test]
fn exact_match_replays_one_token_and_preserves_retained_state() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    for _ in 0..3 {
        let MatchingCache::Normal {
            normal,
            toks,
            offset,
            ..
        } = hit(&mut manager, &[1, 2, 3, 4])?;
        assert_eq!(offset, 3);
        assert_eq!(toks, [4]);
        assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3.]);
        let retained = manager.caches.values().next().unwrap();
        assert_eq!(retained.coverage_len(), 4);
        assert_eq!(
            values(retained.cache[0].as_ref().unwrap())?,
            [1., 2., 3., 4.]
        );
    }
    Ok(())
}

#[test]
fn empty_prompt_misses_and_single_token_replays_from_zero() -> Result<()> {
    let mut manager = manager(1, &Device::Cpu)?;
    assert!(search(&mut manager, &[])?.is_none());
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &[1])?;
    assert_eq!(offset, 0);
    assert_eq!(toks, [1]);
    assert_eq!(normal[0].as_ref().unwrap().current_seq_len(), 0);
    let mut seq = sequence(normal, toks, offset);
    let pipeline = TestPipeline::new(Device::Cpu);
    pipeline.set_none_cache(&mut [&mut seq], false, false, true)?;
    append(&pipeline, 9.)?;
    assert_eq!(
        values(
            manager.caches.values().next().unwrap().cache[0]
                .as_ref()
                .unwrap()
        )?,
        [1.]
    );
    Ok(())
}

#[test]
fn partial_prefix_reuse_stops_at_forwarded_tokens() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &[1, 2, 3, 9])?;
    assert_eq!(offset, 3);
    assert_eq!(toks, [9]);
    assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3.]);
    let cached = manager.caches.shift_remove_index(0).unwrap().1;
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4, 5], None), cached);
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &[1, 2, 3, 4, 5])?;
    assert_eq!(offset, 4);
    assert_eq!(toks, [5]);
    assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3., 4.]);
    Ok(())
}

#[test]
fn exact_replay_isolates_adapter_generations_and_shared_layers() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    let mut cached = manager.caches.shift_remove_index(0).unwrap().1;
    cached.cache.push(Some(KvCache::Shared { owner: 0 }));
    let a = AdapterGenerationId::from_bytes([1; 32]);
    let b = AdapterGenerationId::from_bytes([2; 32]);
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], Some(a)), cached);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    assert!(manager
        .search_for_matching_cache(&[1, 2, 3, 4], Some(b), &[], None, None, None)?
        .is_none());
    let Some(MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    }) = manager.search_for_matching_cache(&[1, 2, 3, 4], Some(a), &[], None, None, None)?
    else {
        panic!("matching adapter missing")
    };
    assert_eq!(offset, 3);
    assert_eq!(toks, [4]);
    assert!(matches!(normal[1], Some(KvCache::Shared { owner: 0 })));
    Ok(())
}

#[test]
fn unsupported_cache_states_skip_exact_replay() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    manager.has_paged_attention = true;
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    manager.has_paged_attention = false;
    manager.no_prefix_cache = true;
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    manager.no_prefix_cache = false;
    for kind in [MultimodalKind::Image, MultimodalKind::Audio] {
        let features = [MultiModalFeature {
            kind,
            offset: 1,
            length: 2,
            hashes: vec![],
            item_range: 0..1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }];
        assert!(manager
            .search_for_matching_cache(&[1, 2, 3, 4], None, &features, None, None, None)?
            .is_none());
    }
    for kind in 0..3 {
        let mut cached = element(4, &Device::Cpu)?;
        match kind {
            0 => cached.image_hashes = Some(vec![]),
            1 => cached.audio_hashes = Some(vec![]),
            _ => cached.video_hashes = Some(vec![]),
        }
        manager
            .caches
            .insert(CacheKey::new(vec![1, 2, 3, 4], None), cached);
        assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    }
    manager.caches.insert(
        CacheKey::new(vec![1, 2, 3, 4], None),
        element(4, &Device::Cpu)?,
    );
    for kind in 0..3 {
        let image = (kind == 0).then_some([].as_slice());
        let audio = (kind == 1).then_some([].as_slice());
        let video = (kind == 2).then_some([].as_slice());
        assert!(manager
            .search_for_matching_cache(&[1, 2, 3, 4], None, &[], image, audio, video)?
            .is_none());
    }
    let mut recurrent = element(4, &Device::Cpu)?;
    recurrent.recurrent_snapshots = Some(CachedRecurrentState {
        len: 4,
        snapshots: vec![RecurrentStateSnapshot {
            conv_state: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?,
            recurrent_state: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?,
            state_layout: crate::kv_cache::RecurrentStateLayout::Opaque,
        }],
    });
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], None), recurrent);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    let mut missing = element(4, &Device::Cpu)?;
    missing.cache.push(None);
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], None), missing);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    let data = Tensor::zeros((1, 1, 4, 1), DType::F32, &Device::Cpu)?;
    let mut k = RotatingCache::new(2, 8, 8);
    let mut v = RotatingCache::new(2, 8, 8);
    k.append(&data)?;
    v.append(&data)?;
    let mut rotating = element(4, &Device::Cpu)?;
    rotating.cache = vec![Some(KvCache::Rotating { k, v })];
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], None), rotating);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    Ok(())
}

struct TestPipeline {
    cache: EitherCache,
    device: Device,
}

impl TestPipeline {
    fn new(device: Device) -> Self {
        Self {
            cache: EitherCache::Normal(NormalCache::new(1, TEST_MAX_SEQ_LEN)),
            device,
        }
    }
}

impl CacheManagerMixin for TestPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> Result<()> {
        NormalCacheManager.clone_in_cache(self, seqs, false)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        NormalCacheManager.clone_out_cache(self, seqs, false);
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        _: bool,
        draft: bool,
        preallocated: bool,
    ) -> Result<()> {
        NormalCacheManager.set_none_cache(self, seqs, draft, preallocated)
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
}

impl MetadataMixin for TestPipeline {
    fn device(&self) -> Device {
        self.device.clone()
    }
    fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
        None
    }
    fn name(&self) -> String {
        "replay-test".into()
    }
    fn reset_non_granular_state(&self) {}
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        Arc::new(GeneralMetadata {
            max_seq_len: TEST_MAX_SEQ_LEN,
            llg_factory: None,
            no_kv_cache: false,
            no_prefix_cache: false,
            num_hidden_layers: 1,
            eos_tok: vec![],
            kind: ModelKind::Normal,
            is_xlora: false,
            activation_dtype: DType::F32,
            sliding_window: None,
            cache_config: None,
            cache_engine: None,
            model_metadata: None,
            modalities: Modalities {
                input: vec![],
                output: vec![],
            },
            loaded_for_uqff_write: false,
        })
    }
}

fn sequence(normal: Vec<Option<KvCache>>, toks: Vec<u32>, offset: usize) -> Sequence {
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let sampler = Sampler::new(
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        32,
        1.,
        0.,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
        1, false, true, None,
    )));
    let original_tokens = (1..=u32::try_from(offset).unwrap())
        .chain(toks.iter().copied())
        .collect();
    Sequence::new_waiting(
        original_tokens,
        "prompt".into(),
        0,
        0,
        1,
        tx,
        sampler,
        vec![],
        vec![],
        None,
        false,
        false,
        group,
        0,
        0,
        SequenceRecognizer::None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        SeqStepType::PromptAndDecode,
        None,
        None,
        None,
        false,
        false,
        vec![],
        None,
    )
    .prefill_v2_normal(normal, toks, offset)
}

fn append(pipeline: &TestPipeline, value: f32) -> Result<()> {
    let source = Tensor::full(value, (1, 1, 1, 1), &pipeline.device)?;
    append_tensor(pipeline, &source)
}

fn append_tensor(pipeline: &TestPipeline, source: &Tensor) -> Result<()> {
    let mut cache = pipeline.cache.normal();
    if pipeline.device.is_metal() {
        cache.0[0].append(source, source)?;
        return Ok(());
    }
    let KvCache::Normal { k, v } = &mut cache.0[0] else {
        panic!("normal cache required")
    };
    k.append(source)?;
    v.append(source)?;
    Ok(())
}

fn batched_replay(device: Device) -> Result<()> {
    let block = NormalCache::CACHE_GROW_SIZE;
    for capacities in [[block, 2 * block], [2 * block, block], [block, block]] {
        for draft in [false, true] {
            let mut manager = PrefixCacheManagerV2::new(16, false, false);
            for (row, capacity) in capacities.into_iter().enumerate() {
                let token = u32::try_from(row + 1).unwrap();
                let data = Tensor::full(
                    [1f32, 2.][row],
                    (1, TEST_KV_HEADS, TEST_PROMPT_LEN, TEST_KV_HEAD_DIM),
                    &device,
                )?;
                let mut k = SingleCache::new(2, TEST_MAX_SEQ_LEN, capacity);
                let mut v = SingleCache::new(2, TEST_MAX_SEQ_LEN, capacity);
                k.append(&data)?;
                v.append(&data)?;
                manager.caches.insert(
                    CacheKey::new(vec![token; TEST_PROMPT_LEN], None),
                    CacheElement {
                        cache: vec![Some(KvCache::Normal { k, v })],
                        recurrent_snapshots: None,
                        audio_hashes: None,
                        image_hashes: None,
                        video_hashes: None,
                    },
                );
            }
            let mut seqs = Vec::new();
            for token in [1, 2] {
                let MatchingCache::Normal {
                    normal,
                    toks,
                    offset,
                    ..
                } = hit(&mut manager, &[token; TEST_PROMPT_LEN])?;
                assert_eq!(offset, TEST_PROMPT_LEN - 1);
                let mut seq = sequence(normal, toks, offset);
                if draft {
                    *seq.normal_draft_cache() = seq.normal_cache_ref().to_vec();
                }
                seqs.push(seq);
            }
            let pipeline = TestPipeline::new(device.clone());
            let mut refs = seqs.iter_mut().collect::<Vec<_>>();
            NormalCacheManager.clone_in_cache(&pipeline, &mut refs, draft)?;
            let data = Tensor::from_vec(
                [
                    vec![9f32; TEST_KV_HEADS * TEST_KV_HEAD_DIM],
                    vec![7.; TEST_KV_HEADS * TEST_KV_HEAD_DIM],
                ]
                .concat(),
                (2, TEST_KV_HEADS, 1, TEST_KV_HEAD_DIM),
                &device,
            )?;
            append_tensor(&pipeline, &data)?;
            drop(refs);
            for extra_tokens in [0, block] {
                if extra_tokens != 0 {
                    let extension = Tensor::full(
                        5f32,
                        (2, TEST_KV_HEADS, extra_tokens, TEST_KV_HEAD_DIM),
                        &device,
                    )?;
                    append_tensor(&pipeline, &extension)?;
                }
                let mut refs = seqs.iter_mut().collect::<Vec<_>>();
                NormalCacheManager.clone_out_cache(&pipeline, &mut refs, draft);
                drop(refs);
                for (row, seq) in seqs.iter_mut().enumerate() {
                    let cache = if draft {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    let KvCache::Normal { k, v } = cache[0].as_ref().unwrap() else {
                        panic!("normal cache required")
                    };
                    assert_eq!(k.capacity_seq_len, block + extra_tokens);
                    assert_eq!(v.capacity_seq_len, block + extra_tokens);
                    let mut head = vec![[1., 2.][row]; (TEST_PROMPT_LEN - 1) * TEST_KV_HEAD_DIM];
                    head.extend(vec![[9., 7.][row]; TEST_KV_HEAD_DIM]);
                    head.extend(vec![5.; extra_tokens * TEST_KV_HEAD_DIM]);
                    assert_eq!(
                        values(cache[0].as_ref().unwrap())?,
                        head.repeat(TEST_KV_HEADS)
                    );
                }
            }
            for (row, retained) in manager.caches.values().enumerate() {
                let KvCache::Normal { k, v } = retained.cache[0].as_ref().unwrap() else {
                    panic!("normal cache required")
                };
                assert_eq!(k.capacity_seq_len, capacities[row]);
                assert_eq!(v.capacity_seq_len, capacities[row]);
                assert_eq!(retained.coverage_len(), TEST_PROMPT_LEN);
                assert_eq!(
                    values(retained.cache[0].as_ref().unwrap())?,
                    vec![[1., 2.][row]; TEST_KV_HEADS * TEST_PROMPT_LEN * TEST_KV_HEAD_DIM]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn batched_replay_with_different_capacities_on_cpu() -> Result<()> {
    batched_replay(Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn batched_replay_with_different_capacities_on_metal() -> Result<()> {
    batched_replay(Device::new_metal(0)?)
}

fn concurrent_replay(device: Device) -> Result<()> {
    let mut manager = manager(4, &device)?;
    let mut requests = Vec::new();
    for value in [9., 7.] {
        let MatchingCache::Normal {
            normal,
            toks,
            offset,
            ..
        } = hit(&mut manager, &[1, 2, 3, 4])?;
        requests.push((sequence(normal, toks, offset), value));
    }
    std::thread::scope(|scope| -> Result<()> {
        let handles = requests
            .into_iter()
            .map(|(mut seq, value)| {
                let device = device.clone();
                scope.spawn(move || -> Result<()> {
                    let pipeline = TestPipeline::new(device);
                    pipeline.clone_in_cache(&mut [&mut seq])?;
                    append(&pipeline, value)?;
                    assert_eq!(
                        values(seq.normal_cache_ref()[0].as_ref().unwrap())?,
                        [1., 2., 3.]
                    );
                    pipeline.clone_out_cache(&mut [&mut seq]);
                    assert_eq!(
                        values(seq.normal_cache_ref()[0].as_ref().unwrap())?,
                        [1., 2., 3., value]
                    );
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap()?;
        }
        Ok(())
    })?;
    let retained = manager.caches.values().next().unwrap();
    assert_eq!(retained.coverage_len(), 4);
    assert_eq!(
        values(retained.cache[0].as_ref().unwrap())?,
        [1., 2., 3., 4.]
    );
    let MatchingCache::Normal {
        normal,
        offset,
        toks,
        ..
    } = hit(&mut manager, &[1, 2, 3, 4])?;
    assert_eq!(offset, 3);
    assert_eq!(toks, [4]);
    assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3.]);
    Ok(())
}

#[test]
fn concurrent_replay_clone_in_isolates_tensor_writes() -> Result<()> {
    concurrent_replay(Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn concurrent_replay_clone_in_isolates_metal_tensor_writes() -> Result<()> {
    concurrent_replay(Device::new_metal(0)?)
}

#[test]
fn restored_request_survives_eviction_without_mutating_retained_storage() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    let original = manager.caches.values().next().unwrap().clone();
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &[1, 2, 3, 4])?;
    let mut seq = sequence(normal, toks, offset);
    assert_eq!(manager.evict_all_caches()?, 1);
    let pipeline = TestPipeline::new(Device::Cpu);
    pipeline.clone_in_cache(&mut [&mut seq])?;
    append(&pipeline, 9.)?;
    drop(seq);
    assert_eq!(original.coverage_len(), 4);
    assert_eq!(
        values(original.cache[0].as_ref().unwrap())?,
        [1., 2., 3., 4.]
    );
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    Ok(())
}
