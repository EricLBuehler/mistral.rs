use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use super::{
    add_recurrent_prefix_memory_reservations, automatic_recurrent_checkpoint_lane_budget,
    automatic_recurrent_prefix_capacity, decode_positions_tensor,
    effective_recurrent_checkpoint_lanes, next_pipeline_prompt_chunk_group,
    paged_attention_memory_reservations, prompt_chunk_is_final, recurrent_batch_kind_for_input,
    recurrent_kv_floor_bytes, reserve_recurrent_serving_capacity, resolve_lora_execution,
    should_sample_step, should_try_speculative_sampling, CacheMemoryReservations, LogitsSelection,
    ModelForwardContext, RecurrentBatchKind, RecurrentCheckpointBudget, RecurrentPrefixBudget,
};
use crate::{
    kv_cache::{
        EitherCache, HybridCache, HybridCacheConfig, HybridLayerType, RecurrentLayerConfig,
        RecurrentStateSpec,
    },
    paged_attention::block_hash::MultimodalAttentionPolicy,
    pipeline::prompt_chunks::PromptChunkPlan,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
    MemoryGoal, MessageContent, PagedAttentionConfig, PagedCacheType,
};
use candle_core::{Device, DeviceLocation, Tensor};
use either::Either;
use indexmap::IndexMap;

fn cache_with_blocks(num_gpu_blocks: usize) -> crate::paged_attention::CacheConfig {
    crate::paged_attention::CacheConfig {
        block_size: 32,
        num_gpu_blocks,
        cache_type: crate::paged_attention::PagedCacheType::Auto,
        kv_cache_group_ids: Vec::new(),
    }
}

#[test]
fn max_model_len_is_the_tightest_limit() -> anyhow::Result<()> {
    use super::{resolve_max_model_len, MemoryGoal};
    let small_cache = cache_with_blocks(33);
    let auto = Some((&small_cache, MemoryGoal::Auto));
    assert_eq!(resolve_max_model_len(4096, None, None, true)?, 4096);
    assert_eq!(resolve_max_model_len(4096, Some(1000), None, true)?, 1000);
    assert_eq!(resolve_max_model_len(4096, None, auto, true)?, 1024);
    assert_eq!(resolve_max_model_len(512, None, auto, true)?, 512);
    assert_eq!(resolve_max_model_len(4096, Some(1024), auto, true)?, 1024);
    Ok(())
}

#[test]
fn explicit_max_model_len_must_fit_model_and_cache() {
    use super::{resolve_max_model_len, MemoryGoal};
    let small_cache = cache_with_blocks(33);
    assert!(resolve_max_model_len(4096, Some(8192), None, true)
        .unwrap_err()
        .to_string()
        .contains("context length of 4096"));
    let auto = resolve_max_model_len(
        4096,
        Some(2048),
        Some((&small_cache, MemoryGoal::Auto)),
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(auto.contains("holds 1024 tokens") && auto.contains("--pa-cache-type"));
    let explicit = resolve_max_model_len(
        4096,
        Some(2048),
        Some((&small_cache, MemoryGoal::KvTokens(1024))),
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(explicit.contains("raise --kv-cache"));
}

fn prompt_chunk(start: usize, end: usize) -> PromptChunkPlan {
    PromptChunkPlan {
        start,
        end,
        attention_policy: MultimodalAttentionPolicy::Causal,
    }
}

#[test]
fn scheduler_visible_prefill_samples_only_the_final_chunk() {
    assert!(!should_sample_step(true, true, false));
    assert!(should_sample_step(true, true, true));
    assert!(should_sample_step(true, false, false));
    assert!(should_sample_step(false, true, false));
}

#[test]
fn speculative_sampling_bootstraps_on_the_final_prompt_chunk() {
    assert!(!should_try_speculative_sampling(
        true, true, false, false, true
    ));
    assert!(should_try_speculative_sampling(
        true, true, true, false, true
    ));
    assert!(should_try_speculative_sampling(
        true, false, false, false, true
    ));
    assert!(!should_try_speculative_sampling(
        true, true, true, false, false
    ));
    assert!(should_try_speculative_sampling(
        false, true, false, false, false
    ));
    assert!(!should_try_speculative_sampling(
        true, true, true, true, true
    ));
}

#[test]
fn scheduler_visible_prefill_preserves_nonfinal_proposer_state() {
    assert!(!prompt_chunk_is_final(true, false, true));
    assert!(prompt_chunk_is_final(true, true, false));
    assert!(prompt_chunk_is_final(false, false, true));
}

#[test]
fn packed_hybrid_final_prompt_chunks_remain_ragged() {
    let plans = vec![
        vec![prompt_chunk(0, 128), prompt_chunk(128, 129)],
        vec![prompt_chunk(0, 128), prompt_chunk(128, 134)],
    ];

    assert_eq!(
        next_pipeline_prompt_chunk_group(&[1, 1], &plans, false, true, true),
        Some((vec![0, 1], MultimodalAttentionPolicy::Causal, true))
    );
}

#[test]
fn hybrid_nonfinal_checkpoint_chunks_remain_uniform() {
    let plans = vec![
        vec![prompt_chunk(0, 64), prompt_chunk(64, 129)],
        vec![prompt_chunk(0, 128), prompt_chunk(128, 134)],
    ];

    assert_eq!(
        next_pipeline_prompt_chunk_group(&[0, 0], &plans, false, true, true),
        Some((vec![0], MultimodalAttentionPolicy::Causal, false))
    );
}

#[test]
fn unsupported_packed_prefill_keeps_final_chunks_uniform() {
    let plans = vec![vec![prompt_chunk(0, 129)], vec![prompt_chunk(0, 134)]];

    assert_eq!(
        next_pipeline_prompt_chunk_group(&[0, 0], &plans, false, false, true),
        Some((vec![0], MultimodalAttentionPolicy::Causal, true))
    );
}

#[test]
fn recurrent_batch_kind_tracks_staged_speculative_completions() {
    assert_eq!(
        recurrent_batch_kind_for_input(false, true),
        RecurrentBatchKind::SpeculativeDecode
    );
    assert_eq!(
        recurrent_batch_kind_for_input(false, false),
        RecurrentBatchKind::Decode
    );
    assert_eq!(
        recurrent_batch_kind_for_input(true, true),
        RecurrentBatchKind::Prefill
    );
}

#[test]
fn unsupported_recurrent_checkpoint_placement_uses_one_lane() {
    assert_eq!(effective_recurrent_checkpoint_lanes(8, false), 1);
    assert_eq!(effective_recurrent_checkpoint_lanes(8, true), 8);
    assert_eq!(effective_recurrent_checkpoint_lanes(1, false), 1);
}

#[test]
fn automatic_recurrent_checkpoint_depth_preserves_kv_budget() {
    const GIB: usize = 1024 * 1024 * 1024;
    const SNAPSHOT_BYTES: usize = 147 * 1024 * 1024 + 3 * 256 * 1024;

    let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
        requested_lanes: 8,
        capacity: 65,
        snapshot_bytes: SNAPSHOT_BYTES,
        current_capacity: 9,
        current_lanes: 1,
        allocation_available: 53 * GIB,
        future_reserved_bytes: 7 * GIB,
        kv_floor_bytes: 5 * GIB,
    })
    .unwrap();

    assert_eq!(lanes, 4);
}

#[test]
fn automatic_recurrent_checkpoint_depth_keeps_requested_small_capacity() {
    const GIB: usize = 1024 * 1024 * 1024;
    const SNAPSHOT_BYTES: usize = 147 * 1024 * 1024 + 3 * 256 * 1024;

    let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
        requested_lanes: 8,
        capacity: 33,
        snapshot_bytes: SNAPSHOT_BYTES,
        current_capacity: 9,
        current_lanes: 1,
        allocation_available: 53 * GIB,
        future_reserved_bytes: 7 * GIB,
        kv_floor_bytes: 5 * GIB,
    })
    .unwrap();

    assert_eq!(lanes, 8);
}

#[test]
fn automatic_recurrent_checkpoint_depth_uses_replay_when_none_fit() {
    const GIB: usize = 1024 * 1024 * 1024;
    const SNAPSHOT_BYTES: usize = 147 * 1024 * 1024 + 3 * 256 * 1024;

    let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
        requested_lanes: 8,
        capacity: 512,
        snapshot_bytes: SNAPSHOT_BYTES,
        current_capacity: 9,
        current_lanes: 1,
        allocation_available: 53 * GIB,
        future_reserved_bytes: 7 * GIB,
        kv_floor_bytes: 5 * GIB,
    })
    .unwrap();

    assert_eq!(lanes, 1);
}

#[test]
fn automatic_recurrent_checkpoint_depth_uses_existing_minimum_capacity() {
    const GIB: usize = 1024 * 1024 * 1024;

    let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
        requested_lanes: 8,
        capacity: 2,
        snapshot_bytes: GIB,
        current_capacity: 9,
        current_lanes: 1,
        allocation_available: 15 * GIB,
        future_reserved_bytes: 0,
        kv_floor_bytes: 0,
    })
    .unwrap();

    assert_eq!(lanes, 1);
}

#[test]
fn recurrent_context_floor_preserves_requested_kv_bytes() {
    const MIB: usize = 1024 * 1024;

    let bytes =
        recurrent_kv_floor_bytes(MemoryGoal::KvTokens(1_000_000), 100 * 1024 * MIB, 1024).unwrap();

    assert_eq!(bytes, 977 * MIB);
}

#[test]
fn cpu_recurrent_reservation_does_not_allocate_checkpoint_lanes() {
    let hybrid = HybridCache::new(
        HybridCacheConfig {
            layer_types: vec![HybridLayerType::Recurrent],
            max_seq_len: 32,
            recurrent: RecurrentLayerConfig {
                conv_dim: 8,
                conv_width: 4,
                state: RecurrentStateSpec::Gdn {
                    heads: 2,
                    key_dim: 4,
                    value_dim: 4,
                },
                recurrent_dtype: Some(candle_core::DType::F32),
            },
        },
        candle_core::DType::BF16,
        &[Device::Cpu],
    )
    .unwrap();
    let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
    let config =
        PagedAttentionConfig::new(None, MemoryGoal::KvBytes(1024 * 1024), PagedCacheType::Auto)
            .unwrap()
            .with_serving_capacity(16)
            .unwrap()
            .with_recurrent_checkpoint_lanes(8)
            .unwrap();

    reserve_recurrent_serving_capacity(&cache, &config, false, false, &Device::Cpu, 1).unwrap();

    let cache = cache.hybrid();
    assert_eq!(cache.checkpoint_lanes(), 1);
    let pool = cache.get(0).unwrap().as_recurrent_pool().unwrap();
    assert_eq!(pool.capacity(), 17);
    assert_eq!(pool.physical_capacity(), 17);
}

#[test]
fn recurrent_transition_reservation_keeps_one_physical_lane() {
    let hybrid = HybridCache::new(
        HybridCacheConfig {
            layer_types: vec![HybridLayerType::Recurrent],
            max_seq_len: 32,
            recurrent: RecurrentLayerConfig {
                conv_dim: 8,
                conv_width: 4,
                state: RecurrentStateSpec::Gdn {
                    heads: 2,
                    key_dim: 4,
                    value_dim: 4,
                },
                recurrent_dtype: Some(candle_core::DType::F32),
            },
        },
        candle_core::DType::BF16,
        &[Device::Cpu],
    )
    .unwrap();
    let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
    let config =
        PagedAttentionConfig::new(None, MemoryGoal::KvBytes(1024 * 1024), PagedCacheType::Auto)
            .unwrap()
            .with_serving_capacity(64)
            .unwrap()
            .with_recurrent_checkpoint_lanes(8)
            .unwrap();

    reserve_recurrent_serving_capacity(&cache, &config, true, true, &Device::Cpu, 1).unwrap();

    let cache = cache.hybrid();
    assert_eq!(cache.recurrent_capacity(), 65);
    assert_eq!(cache.checkpoint_lanes(), 8);
    assert_eq!(cache.physical_checkpoint_lanes(), 1);
    assert!(cache.uses_recurrent_transition_log());
    let pool = cache.get(0).unwrap().as_recurrent_pool().unwrap();
    assert_eq!(pool.capacity(), 65);
    assert_eq!(pool.physical_capacity(), 65);
}

#[test]
fn one_lane_recurrent_reservation_omits_transition_log() {
    let hybrid = HybridCache::new(
        HybridCacheConfig {
            layer_types: vec![HybridLayerType::Recurrent],
            max_seq_len: 32,
            recurrent: RecurrentLayerConfig {
                conv_dim: 8,
                conv_width: 4,
                state: RecurrentStateSpec::Gdn {
                    heads: 2,
                    key_dim: 4,
                    value_dim: 4,
                },
                recurrent_dtype: Some(candle_core::DType::F32),
            },
        },
        candle_core::DType::BF16,
        &[Device::Cpu],
    )
    .unwrap();
    let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
    let config =
        PagedAttentionConfig::new(None, MemoryGoal::KvBytes(1024 * 1024), PagedCacheType::Auto)
            .unwrap()
            .with_serving_capacity(64)
            .unwrap()
            .with_recurrent_checkpoint_lanes(1)
            .unwrap();

    reserve_recurrent_serving_capacity(&cache, &config, true, true, &Device::Cpu, 1).unwrap();

    let cache = cache.hybrid();
    assert_eq!(cache.recurrent_capacity(), 65);
    assert_eq!(cache.checkpoint_lanes(), 1);
    assert!(!cache.uses_recurrent_transition_log());
    let pool = cache.get(0).unwrap().as_recurrent_pool().unwrap();
    assert_eq!(pool.capacity(), 65);
    assert_eq!(pool.physical_capacity(), 65);
}

#[test]
fn automatic_prefix_capacity_uses_the_limiting_recurrent_device() {
    let roomy = RecurrentPrefixBudget {
        snapshot_bytes: 10,
        persistent_bytes: 1000,
        execution_bytes: 1000,
    };
    let limited = RecurrentPrefixBudget {
        snapshot_bytes: 20,
        persistent_bytes: 240,
        execution_bytes: 100,
    };
    assert_eq!(automatic_recurrent_prefix_capacity(16, &[roomy]), 16);
    assert_eq!(
        automatic_recurrent_prefix_capacity(16, &[roomy, limited]),
        4
    );
    assert_eq!(automatic_recurrent_prefix_capacity(2, &[roomy, limited]), 2);
}

#[test]
fn recurrent_prefix_reservation_includes_staging_and_device_baselines() {
    let reservations = add_recurrent_prefix_memory_reservations(
        CacheMemoryReservations {
            primary_device_bytes: 100,
            secondary_device_bytes: 50,
            ..Default::default()
        },
        HashMap::from([
            (DeviceLocation::Cpu, 10),
            (DeviceLocation::Cuda { gpu_id: 0 }, 20),
            (DeviceLocation::Cuda { gpu_id: 1 }, 30),
        ]),
        DeviceLocation::Cpu,
        2,
    )
    .unwrap();

    assert_eq!(reservations.primary_device_bytes, 130);
    assert_eq!(reservations.secondary_device_bytes, 140);
    assert_eq!(reservations.primary_staging_bytes, 10);
    assert_eq!(reservations.secondary_staging_bytes, 30);
}

#[test]
fn loaded_hybrid_cache_drives_recurrent_prefix_reservation() {
    let hybrid = HybridCache::new(
        HybridCacheConfig {
            layer_types: vec![HybridLayerType::Recurrent],
            max_seq_len: 32,
            recurrent: RecurrentLayerConfig {
                conv_dim: 8,
                conv_width: 4,
                state: RecurrentStateSpec::Gdn {
                    heads: 2,
                    key_dim: 4,
                    value_dim: 4,
                },
                recurrent_dtype: Some(candle_core::DType::F32),
            },
        },
        candle_core::DType::BF16,
        &[Device::Cpu],
    )
    .unwrap();
    let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
    let config =
        PagedAttentionConfig::new(None, MemoryGoal::KvBytes(1024 * 1024), PagedCacheType::Auto)
            .unwrap()
            .with_base_device_memory_reservation(100)
            .unwrap()
            .with_recurrent_prefix_capacity(crate::PrefixCacheCapacity::Fixed(2));

    let reservations = paged_attention_memory_reservations(&cache, &config, &Device::Cpu).unwrap();

    assert_eq!(reservations.primary_device_bytes, 676);
    assert_eq!(reservations.secondary_device_bytes, 0);
    assert_eq!(reservations.primary_staging_bytes, 192);
    assert_eq!(reservations.secondary_staging_bytes, 0);
}
use serde_json::Value;

#[test]
fn base_lora_routes_still_validate_dense_batch_cardinality() -> candle_core::Result<()> {
    let input_ids = Tensor::zeros((2, 3), candle_core::DType::U32, &Device::Cpu)?;
    let flash_meta = FlashParams::empty(true);

    assert!(resolve_lora_execution(None, &input_ids, None, &flash_meta, &[None, None])?.is_none());
    let error = resolve_lora_execution(None, &input_ids, None, &flash_meta, &[None]).unwrap_err();
    assert!(error
        .to_string()
        .contains("adapter lease count 1 does not match model batch size 2"));
    Ok(())
}

#[test]
fn base_lora_routes_validate_packed_logical_shape() -> candle_core::Result<()> {
    let input_ids = Tensor::zeros((1, 5), candle_core::DType::U32, &Device::Cpu)?;
    let mut flash_meta = FlashParams::empty(true);
    flash_meta.packed = true;
    let mut paged_meta = PagedAttentionInputMetadata::dummy(&Device::Cpu)?;
    paged_meta.query_lens = Some(vec![2, 3]);

    assert!(resolve_lora_execution(
        None,
        &input_ids,
        Some(&paged_meta),
        &flash_meta,
        &[None, None],
    )?
    .is_none());

    let error = resolve_lora_execution(None, &input_ids, Some(&paged_meta), &flash_meta, &[None])
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("adapter lease count 1 does not match packed logical sequence count 2"));

    paged_meta.query_lens = Some(vec![2, 2]);
    let error = resolve_lora_execution(
        None,
        &input_ids,
        Some(&paged_meta),
        &flash_meta,
        &[None, None],
    )
    .unwrap_err();
    assert!(error.to_string().contains(
        "packed logical query lengths total 4 does not match physical sequence length 5"
    ));
    Ok(())
}

#[test]
fn packed_logits_select_each_logical_sequence() {
    let source = Tensor::from_vec(
        (0u8..8).map(f32::from).collect::<Vec<_>>(),
        (1, 8, 1),
        &Device::Cpu,
    )
    .unwrap();
    let selection = LogitsSelection::from_packed_context_lens(
        &source,
        &[(2, 1), (0, 1), (3, 1)],
        &[3, 1, 4],
        &[Device::Cpu],
    )
    .unwrap();
    let selected = selection.select(&source).unwrap();

    assert_eq!(selected.dims(), &[3, 1, 1]);
    assert_eq!(
        selected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![2.0, 3.0, 7.0]
    );
}

#[test]
fn packed_logits_select_multi_token_spans() {
    let source = Tensor::from_vec(
        (0u8..5).map(f32::from).collect::<Vec<_>>(),
        (1, 5, 1),
        &Device::Cpu,
    )
    .unwrap();
    let selection = LogitsSelection::from_packed_context_lens(
        &source,
        &[(1, 2), (0, 2)],
        &[3, 2],
        &[Device::Cpu],
    )
    .unwrap();
    let selected = selection.select(&source).unwrap();

    assert_eq!(selected.dims(), &[2, 2, 1]);
    assert_eq!(
        selected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn packed_positions_require_explicit_metadata() {
    let mut flash_params = FlashParams::empty(true);
    flash_params.packed = true;
    let seqlen_offsets = [0];
    let context_lens = [(0, 1)];
    let position_ids = [1];
    let mut context = ModelForwardContext::new(
        &seqlen_offsets,
        &context_lens,
        &position_ids,
        None,
        &flash_params,
    );

    let error = context.text_positions(&Device::Cpu, 1).unwrap_err();

    assert!(error
        .to_string()
        .contains("packed prefill is missing RoPE positions"));
}

#[test]
fn decode_positions_expand_exclusive_row_ends() {
    let positions = decode_positions_tensor(&[4, 10], 3, &Device::Cpu).unwrap();

    assert_eq!(positions.to_vec1::<u32>().unwrap(), vec![1, 2, 3, 7, 8, 9]);
    assert!(decode_positions_tensor(&[2], 3, &Device::Cpu).is_err());
}

macro_rules! hashmap {
        (@single $($x:tt)*) => (());
        (@count $($rest:expr),*) => (<[()]>::len(&[$(hashmap!(@single $rest)),*]));

        ($($key:expr => $value:expr,)+) => { hashmap!($($key => $value),+) };
        ($($key:expr => $value:expr),*) => {
            {
                let _cap = hashmap!(@count $($key),*);
                let mut _map = ::indexmap::IndexMap::with_capacity(_cap);
                $(
                    let _ = _map.insert($key, Value::String($value));
                )*
                _map
            }
        };
    }

#[cfg(test)]
#[track_caller]
fn test_with_inputs(
    templates: &[(bool, &str, &str, &str, &str)],
    expected_outputs: &[&str],
    inputs: Vec<IndexMap<String, MessageContent>>,
) {
    use crate::pipeline::chat_template::ChatTemplateValue;

    use super::chat_template::apply_chat_template_to;
    let mut failed = Vec::new();
    let n_templates = templates.len();
    for ((has_system, bos, eos, unk, template), expected) in templates.iter().zip(expected_outputs)
    {
        let output = match apply_chat_template_to(
            if !has_system {
                inputs[1..].to_vec()
            } else {
                inputs.clone()
            },
            true,
            None,
            None, // reasoning_effort
            &ChatTemplateValue(Either::Left(template.to_string())),
            Some(bos.to_string()),
            Some(eos.to_string()),
            Some(unk.to_string()),
            Vec::new(),
        ) {
            Ok(v) => v,
            Err(e) => {
                failed.push(format!("Failed with {e}."));
                continue;
            }
        };
        if output != *expected {
            failed.push(format!(
                "Expected: `{}` \n\nGot:      `{}`",
                expected.replace('\n', "\\n"),
                output.replace('\n', "\\n")
            ));
        }
    }
    if !failed.is_empty() {
        for (i, line) in failed.iter().enumerate() {
            println!("------------ Template {i} ------------");
            println!("{line}");
        }
        println!("------------------------");
        panic!("{}/{n_templates} chat templates failed.", failed.len());
    }
}

#[test]
/// Generating these cases:
/// ```py
/// >>> t=transformers.AutoTokenizer.from_pretrained(...)
/// # If non-system prompt model
/// >>> t.apply_chat_template([{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
/// # If system prompt model
/// >>> t.apply_chat_template([{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
/// ```
fn test_chat_templates() {
    let templates = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"),
            // mistralai/Mistral-7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // meta-llama/Llama-2-13b-chat-hf
            (true, "<s>", "</s>", "<unk>", "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"),
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // google/gemma-7b-it
            (false, "<bos>", "<eos>", "<unk>", "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"),
            // HuggingFaceM4/idefics2-8b-chatty
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"),
        ];
    let expected_outputs = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
            // mistralai/Mistral-7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s> [INST] Who are you [/INST]   I am an assistant   </s> [INST] Another question [/INST]",
            // meta-llama/Llama-2-13b-chat-hf
            "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s><s>[INST] Who are you [/INST] I am an assistant </s><s>[INST] Another question [/INST]",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
            // google/gemma-7b-it
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        ];
    let messages = [
        ["system", "You are a helpful assistant"],
        ["user", "Hello"],
        ["assistant", "Hi there"],
        ["user", "Who are you"],
        ["assistant", "   I am an assistant   "],
        ["user", "Another question"],
    ];
    let mut inputs = Vec::new();
    for [role, content] in messages {
        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left(role.to_string()));
        message.insert("content".to_string(), Either::Left(content.to_string()));
        inputs.push(message);
    }
    test_with_inputs(&templates, &expected_outputs, inputs);
}

#[test]
/// Generating these cases:
/// ```py
/// >>> processor=transformers.AutoProcessor.from_pretrained(...)
/// >>> processor.apply_chat_template([
///         {"role":"system","content":[{"type":"text", "text": "You are a helpful assistant"}]},
///         {"role":"user","content":[{"type":"image"}, {"type":"text", "text": "Hello, please describe the above."}]},
///         {"role":"assistant","content":[{"type":"text", "text": "Hi there"}]},
///         {"role":"user","content":[{"type":"text", "text": "Who are you"}]},
///         {"role":"assistant","content":[{"type":"text", "text": "   I am an assistant   "}]},
///         {"role":"user","content":[{"type":"text", "text": "Another question"}]}
///     ], add_generation_prompt=True, tokenize=False)
/// ```
fn test_image_chat_templates() {
    let templates = [
            // HuggingFaceM4/idefics2-8b-chatty
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"),
        ];
    let expected_outputs = [
            // HuggingFaceM4/idefics2-8b-chatty
            "System: You are a helpful assistant<end_of_utterance>\nUser:<image>Hello, please describe the above.<end_of_utterance>\nAssistant: Hi there<end_of_utterance>\nUser:<image>This is me, who are you<end_of_utterance>\nAssistant:    I am an assistant   <end_of_utterance>\nUser:<image>Another question, what is this?<end_of_utterance>\nAssistant:",
        ];

    let mut inputs = Vec::new();

    let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
        IndexMap::new();
    message.insert("role".to_string(), Either::Left("system".to_string()));
    message.insert(
        "content".to_string(),
        Either::Right(vec![hashmap! {
            "type".to_string() => "text".to_string(),
            "text".to_string() => "You are a helpful assistant".to_string()
        }]),
    );
    inputs.push(message);

    let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
        IndexMap::new();
    message.insert("role".to_string(), Either::Left("user".to_string()));
    message.insert(
        "content".to_string(),
        Either::Right(vec![
            hashmap! {
                "type".to_string() => "image".to_string()
            },
            hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "Hello, please describe the above.".to_string()
            },
        ]),
    );
    inputs.push(message);

    let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
        IndexMap::new();
    message.insert("role".to_string(), Either::Left("assistant".to_string()));
    message.insert(
        "content".to_string(),
        Either::Right(vec![hashmap! {
            "type".to_string() => "text".to_string(),
            "text".to_string() => "Hi there".to_string()
        }]),
    );
    inputs.push(message);

    let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
        IndexMap::new();
    message.insert("role".to_string(), Either::Left("user".to_string()));
    message.insert(
        "content".to_string(),
        Either::Right(vec![
            hashmap! {
                "type".to_string() => "image".to_string()
            },
            hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "This is me, who are you".to_string()
            },
        ]),
    );
    inputs.push(message);

    let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
        IndexMap::new();
    message.insert("role".to_string(), Either::Left("assistant".to_string()));
    message.insert(
        "content".to_string(),
        Either::Right(vec![hashmap! {
            "type".to_string() => "text".to_string(),
            "text".to_string() => "   I am an assistant   ".to_string()
        }]),
    );
    inputs.push(message);

    let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
        IndexMap::new();
    message.insert("role".to_string(), Either::Left("user".to_string()));
    message.insert(
        "content".to_string(),
        Either::Right(vec![
            hashmap! {
                "type".to_string() => "image".to_string()
            },
            hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "Another question, what is this?".to_string()
            },
        ]),
    );
    inputs.push(message);

    test_with_inputs(&templates, &expected_outputs, inputs);
}
