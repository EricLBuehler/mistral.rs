use super::{argmax_f32, partial_sort_top_k, ModelGenerationDefaults, SamplingParams};
use std::collections::HashMap;

#[test]
fn test_argmax() {
    use super::Sampler;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::Arc;
    use std::sync::Mutex;

    let sampler = Sampler::new(
        None,
        10,
        None,
        None,
        None,
        None,
        None,
        32,
        0.1,
        0.05,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let logits = Tensor::from_vec(vec![-3.0f32, -1.0, -2.0], 3, &Device::Cpu).unwrap();
    let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
    let res = sampler
        .sample(logits, &[0, 1, 2], 0, true, rng, false, false)
        .unwrap();
    assert_eq!(res.token, 1);
    let expected = -1.0f32 - ((-3.0f32).exp() + (-1.0f32).exp() + (-2.0f32).exp()).ln();
    assert!((res.logprob - expected).abs() < 1e-6);
    let selected = res
        .top_logprobs
        .unwrap()
        .into_iter()
        .find(|top| top.token == 1)
        .unwrap();
    assert!((selected.logprob - expected).abs() < 1e-6);
}

#[test]
fn argmax_uses_first_maximum_and_rejects_invalid_rows() {
    assert_eq!(argmax_f32(&[-2.0, 4.0, 4.0, 1.0]).unwrap(), 1);
    assert_eq!(argmax_f32(&[f32::NEG_INFINITY, -3.0, 2.0]).unwrap(), 2);
    assert!(argmax_f32(&[0.0, f32::NAN]).is_err());
    assert!(argmax_f32(&[0.0, f32::INFINITY]).is_err());
    assert!(argmax_f32(&[f32::NEG_INFINITY, f32::NEG_INFINITY]).is_err());
    assert!(argmax_f32(&[]).is_err());
}

#[test]
fn top_k_ties_prefer_lower_token_ids_at_the_boundary() {
    let mut probs = vec![0.1, 0.5, 0.5, 0.5, 0.2];
    let selected = partial_sort_top_k(&mut probs, 2, true);

    assert_eq!(selected, vec![(1, 0.5), (2, 0.5)]);
    assert_eq!(probs, vec![0.0, 0.5, 0.5, 0.0, 0.0]);
}

#[test]
fn test_gumbel_speculative() {
    use super::Sampler;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::Arc;
    use std::sync::Mutex;

    let sampler = Sampler::new(
        None,
        10,
        None,
        None,
        None,
        None,
        None,
        32,
        0.1,
        0.05,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let logits = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], 3, &Device::Cpu).unwrap();
    let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
    let res = sampler
        .sample(logits, &[0, 1, 2], 0, false, rng, true, false)
        .unwrap();
    assert_eq!(res.token, 2);
    assert_eq!(res.top_logprobs, None);
    let expected = 2.0f32 - (0.0f32.exp() + 1.0f32.exp() + 2.0f32.exp()).ln();
    assert!((res.logprob - expected).abs() < 1e-6);
}

#[test]
fn test_speculative_candidate_probs_use_sampling_filters() {
    use super::Sampler;
    use candle_core::{Device, Tensor};

    let sampler = Sampler::new(
        Some(1.0),
        10,
        None,
        None,
        None,
        None,
        None,
        1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let logits = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], 3, &Device::Cpu).unwrap();
    let context = [0u32];
    let target_probs = sampler
        .speculative_target_probs(logits.clone(), &context, 0)
        .unwrap();
    let candidate_probs = sampler
        .speculative_candidate_probs(logits, &context, 0)
        .unwrap();

    assert_eq!(candidate_probs, target_probs.sampling);
    assert_eq!(candidate_probs, vec![0.0, 0.0, 1.0]);
    let expected = [
        1.0f32 / (1.0 + 1.0f32.exp() + 2.0f32.exp()),
        1.0f32.exp() / (1.0 + 1.0f32.exp() + 2.0f32.exp()),
        2.0f32.exp() / (1.0 + 1.0f32.exp() + 2.0f32.exp()),
    ];
    for (actual, expected) in target_probs.reporting.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn test_min_p_applies_without_top_p() {
    use super::Sampler;
    use candle_core::{Device, Tensor};

    let sampler = Sampler::new(
        Some(1.0),
        0,
        None,
        None,
        None,
        None,
        None,
        3,
        1.0,
        0.4,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let logits = Tensor::from_vec(vec![0.0f32, 2.0f32.ln(), 4.0f32.ln()], 3, &Device::Cpu).unwrap();
    let probs = sampler
        .speculative_candidate_probs(logits, &[0], 0)
        .unwrap();

    assert!((probs[0] - 0.0).abs() < 1e-6);
    assert!((probs[1] - 1.0 / 3.0).abs() < 1e-6);
    assert!((probs[2] - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_top_logprobs_use_unfiltered_distribution() {
    use super::Sampler;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::{Arc, Mutex};

    let sampler = Sampler::new(
        Some(1.0),
        3,
        None,
        None,
        None,
        None,
        None,
        1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let logits = Tensor::from_vec(vec![0.0f32, 2.0f32.ln(), 3.0f32.ln()], 3, &Device::Cpu).unwrap();
    let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));

    let result = sampler
        .sample(logits, &[0], 0, true, rng, false, false)
        .unwrap();

    assert_eq!(result.token, 2);
    let top = result.top_logprobs.unwrap();
    assert_eq!(top.len(), 3);
    let expected = [1.0f32 / 6.0, 2.0 / 6.0, 3.0 / 6.0];
    for item in top {
        assert!((item.logprob - expected[item.token as usize].ln()).abs() < 1e-6);
    }
}

#[test]
fn test_logits_bias_suppresses_argmax_token() {
    use super::Sampler;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::Arc;
    use std::sync::Mutex;

    let sampler = Sampler::new(
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        -1,
        1.0,
        0.0,
        HashMap::from([(2, -1.0e9)]),
        vec![],
    )
    .unwrap();
    let logits = Tensor::from_vec(vec![0.0f32, 1.0, 10.0], 3, &Device::Cpu).unwrap();
    let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
    let res = sampler
        .sample(logits, &[0], 0, false, rng, false, false)
        .unwrap();

    assert_eq!(res.token, 1);
}

#[test]
fn test_apply_model_defaults() {
    let mut params = SamplingParams::neutral();
    params.apply_model_defaults(&ModelGenerationDefaults {
        do_sample: Some(true),
        temperature: Some(1.0),
        top_k: Some(32),
        top_p: Some(0.9),
        min_p: Some(0.05),
        repetition_penalty: Some(1.1),
        max_new_tokens: Some(256),
        max_length: None,
        suppress_tokens: Some(vec![258882, 258883]),
    });

    assert_eq!(params.temperature, Some(1.0));
    assert_eq!(params.top_k, Some(32));
    assert_eq!(params.top_p, Some(0.9));
    assert_eq!(params.min_p, Some(0.05));
    assert_eq!(params.repetition_penalty, Some(1.1));
    assert_eq!(params.max_len, Some(256));
    assert_eq!(
        params.logits_bias.as_ref().unwrap().get(&258882),
        Some(&-1.0e9)
    );
    assert_eq!(
        params.logits_bias.as_ref().unwrap().get(&258883),
        Some(&-1.0e9)
    );
}

#[test]
fn test_apply_model_defaults_disables_sampling_when_requested() {
    let mut params = SamplingParams {
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.9),
        min_p: Some(0.1),
        ..SamplingParams::neutral()
    };
    params.apply_model_defaults(&ModelGenerationDefaults {
        do_sample: Some(false),
        ..Default::default()
    });

    assert_eq!(params.temperature, None);
    assert_eq!(params.top_k, Some(1));
    assert_eq!(params.top_p, None);
    assert_eq!(params.min_p, None);
}

#[test]
fn sampling_params_ignore_eos_is_backward_compatible() {
    let mut serialized = serde_json::to_value(SamplingParams::neutral()).unwrap();
    serialized.as_object_mut().unwrap().remove("ignore_eos");

    let params: SamplingParams = serde_json::from_value(serialized).unwrap();
    assert!(!params.ignore_eos);

    let mut params = SamplingParams::neutral();
    params.ignore_eos = true;
    let params: SamplingParams =
        serde_json::from_value(serde_json::to_value(params).unwrap()).unwrap();
    assert!(params.ignore_eos);
}

#[test]
fn test_do_sample_false_overrides_sampling_defaults() {
    let mut params = SamplingParams::neutral();
    params.apply_model_defaults(&ModelGenerationDefaults {
        do_sample: Some(false),
        temperature: Some(0.6),
        top_k: Some(20),
        top_p: Some(0.9),
        min_p: Some(0.05),
        ..Default::default()
    });

    assert_eq!(params.temperature, None);
    assert_eq!(params.top_k, Some(1));
    assert_eq!(params.top_p, None);
    assert_eq!(params.min_p, None);
}

#[cfg(feature = "cuda")]
fn assert_device_sampling_matches_cpu(device: &candle_core::Device) {
    use super::Sampler;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::{Arc, Mutex};

    const VOCAB: usize = 1031;
    const PEAK: usize = 613;
    const TIED: [usize; 2] = [17, 902];
    const SEED: u64 = 7;

    let spread = (0..VOCAB)
        .map(|i| ((i * 37 + 11) % 257) as f32 / 64.0 - 2.0)
        .collect::<Vec<_>>();
    let mut peaked = spread.clone();
    peaked[PEAK] = 24.0;
    let mut tied = spread.clone();
    for index in TIED {
        tied[index] = 9.5;
    }
    let context = [PEAK as u32, PEAK as u32, 3, TIED[0] as u32];

    let sampler = |temperature: Option<f64>,
                   penalties: [Option<f32>; 3],
                   top_k: i64,
                   top_p: f64,
                   bias: HashMap<u32, f32>| {
        Sampler::new(
            temperature,
            0,
            None,
            penalties[0],
            penalties[1],
            penalties[2],
            None,
            top_k,
            top_p,
            0.0,
            bias,
            vec![],
        )
        .unwrap()
    };
    let no_penalty = [None; 3];
    let cases = [
        ("greedy", sampler(None, no_penalty, -1, 1.0, HashMap::new())),
        (
            "greedy penalties",
            sampler(
                None,
                [Some(6.0), Some(4.0), Some(1.3)],
                -1,
                1.0,
                HashMap::new(),
            ),
        ),
        (
            "greedy bias",
            sampler(
                None,
                no_penalty,
                -1,
                1.0,
                HashMap::from([(PEAK as u32, -1.0e9), (TIED[0] as u32, -1.0e9)]),
            ),
        ),
        (
            "top-1",
            sampler(Some(0.7), no_penalty, 1, 1.0, HashMap::new()),
        ),
        (
            "top-1 penalties",
            sampler(
                Some(0.7),
                [Some(6.0), Some(4.0), Some(1.3)],
                1,
                1.0,
                HashMap::new(),
            ),
        ),
    ];
    for (name, sampler) in &cases {
        for (kind, logits) in [("spread", &spread), ("peaked", &peaked), ("tied", &tied)] {
            let sample = |device: &Device| {
                let logits = Tensor::from_vec(logits.clone(), VOCAB, device).unwrap();
                let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
                sampler
                    .sample(logits, &context, 0, false, rng, false, false)
                    .unwrap()
                    .token
            };
            assert_eq!(sample(device), sample(&Device::Cpu), "{name} on {kind}");
        }
    }
    // top-p keeps only the dominant token, so the draw is deterministic on every backend.
    let nucleus = sampler(Some(1.0), no_penalty, 40, 0.5, HashMap::new());
    let sample = |device: &Device| {
        let logits = Tensor::from_vec(peaked.clone(), VOCAB, device).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        nucleus
            .sample(logits, &context, 0, false, rng, false, false)
            .unwrap()
            .token
    };
    assert_eq!(sample(device), PEAK as u32);
    assert_eq!(sample(&Device::Cpu), PEAK as u32);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sampling_matches_cpu() {
    assert_device_sampling_matches_cpu(&candle_core::Device::new_cuda(0).unwrap());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_plan_covers_greedy_top_k_and_categorical() {
    use super::{CudaBatchSamplingKind, Sampler};

    let greedy = Sampler::new(
        None,
        0,
        None,
        Some(0.0),
        Some(0.0),
        Some(1.0),
        None,
        -1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let greedy_plan = greedy.cuda_batch_sampling_plan(false).unwrap();
    assert!(matches!(greedy_plan.kind, CudaBatchSamplingKind::Greedy));
    assert_eq!(greedy_plan.inverse_temperature, 1.0);
    assert!(greedy.cuda_resident_sampling_plan(false).is_some());
    assert!(greedy.cuda_speculative_sampling_plan(false).is_none());

    let mut penalized = greedy.clone();
    penalized.repetition_penalty = Some(1.1);
    assert!(penalized.cuda_batch_sampling_plan(false).is_none());
    let mut biased = greedy.clone();
    biased.logits_bias.insert(1, 1.0);
    assert!(biased.cuda_batch_sampling_plan(false).is_none());
    let mut processed = greedy.clone();
    processed.logits_processors.push(std::sync::Arc::new(
        |logits: &candle_core::Tensor, _context: &[u32]| Ok(logits.clone()),
    ));
    assert!(processed.cuda_batch_sampling_plan(false).is_none());

    let top_k = Sampler::new(
        Some(0.5),
        0,
        None,
        None,
        None,
        None,
        None,
        64,
        0.9,
        0.05,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let top_k_plan = top_k.cuda_batch_sampling_plan(false).unwrap();
    assert!(matches!(
        top_k_plan.kind,
        CudaBatchSamplingKind::TopK { k: 64 }
    ));
    assert_eq!(top_k_plan.inverse_temperature, 2.0);
    assert!(top_k.cuda_resident_sampling_plan(false).is_some());
    assert!(top_k.cuda_batch_sampling_plan(true).is_none());
    let speculative_top_k = top_k.cuda_speculative_sampling_plan(false).unwrap();
    assert_eq!(speculative_top_k.inverse_temperature, 2.0);
    assert_eq!(speculative_top_k.top_k, 64);
    assert_eq!(speculative_top_k.top_p, 0.9);
    assert_eq!(speculative_top_k.min_p, 0.05);

    let top_one = Sampler::new(
        Some(1.0),
        0,
        None,
        None,
        None,
        None,
        None,
        1,
        0.9,
        0.05,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let top_one_plan = top_one.cuda_batch_sampling_plan(false).unwrap();
    assert!(top_one_plan.kind.is_argmax());

    let unbounded = Sampler::new(
        Some(1.0),
        0,
        None,
        None,
        None,
        None,
        None,
        -1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let unbounded_plan = unbounded.cuda_batch_sampling_plan(false).unwrap();
    assert!(matches!(
        unbounded_plan.kind,
        CudaBatchSamplingKind::Categorical
    ));
    assert!(unbounded.cuda_resident_sampling_plan(false).is_none());
    let speculative_unbounded = unbounded.cuda_speculative_sampling_plan(false).unwrap();
    assert_eq!(speculative_unbounded.top_k, 0);
    assert_eq!(speculative_unbounded.top_p, 1.0);
    assert_eq!(speculative_unbounded.min_p, 0.0);

    let mut filtered = unbounded;
    filtered.top_p = 0.9;
    assert!(filtered.cuda_batch_sampling_plan(false).is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_packed_row_uses_full_distribution_logprob() {
    use super::Sampler;
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;

    let sampler = Sampler::new(
        Some(2.0),
        0,
        None,
        None,
        None,
        None,
        None,
        1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let plan = sampler.cuda_batch_sampling_plan(false).unwrap();
    let packed = [4.0, 2.0, 1.0, 7.0, 8.0, 9.0, 2.0, 2.0];
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let sampled = sampler
        .sample_cuda_topk_packed_row(&packed, 3, plan, &mut rng)
        .unwrap();

    assert_eq!(sampled.token, 7);
    assert!((sampled.logprob - 0.5f32.ln()).abs() < 1e-6);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ranked_topk_matches_packed_reference_for_fixed_seeds() {
    use super::Sampler;
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;

    for k in [20, 32] {
        let sampler = Sampler::new(
            Some(0.8),
            0,
            None,
            None,
            None,
            None,
            None,
            k as i64,
            0.37,
            0.05,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let plan = sampler.cuda_batch_sampling_plan(false).unwrap();
        let values = (0..k)
            .map(|slot| 5.0 - (slot / 4) as f32 * 0.25)
            .collect::<Vec<_>>();
        let indices = (0..k).map(|slot| (1000 + slot) as f32).collect::<Vec<_>>();
        let scaled_max = values[0] * plan.inverse_temperature;
        let probabilities = values
            .iter()
            .map(|value| (*value * plan.inverse_temperature - scaled_max).exp())
            .collect::<Vec<_>>();
        let denominator = probabilities.iter().sum::<f32>();
        let cutoff = 0.37 * denominator;
        let mut cumulative = 0.0f32;
        let mut support = 0;
        for probability in &probabilities {
            if cumulative >= cutoff {
                break;
            }
            cumulative += probability;
            support += 1;
        }

        let mut ranked = Vec::with_capacity(2 * k);
        ranked.extend_from_slice(&values);
        ranked.extend_from_slice(&indices);
        let mut full = ranked.clone();
        full.extend_from_slice(&[denominator, scaled_max]);

        for seed in 0..256 {
            let mut reference_rng = Isaac64Rng::seed_from_u64(seed);
            let reference = sampler
                .sample_cuda_topk_packed_row(&full, k, plan, &mut reference_rng)
                .unwrap();
            let mut ranked_rng = Isaac64Rng::seed_from_u64(seed);
            let actual = sampler
                .sample_cuda_ranked_topk_packed_row(&ranked, k, plan, &mut ranked_rng)
                .unwrap();

            assert_eq!(actual.token, reference.token);
            assert_eq!(actual.logprob, reference.logprob);
            assert!((actual.token as usize) < 1000 + support);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn resident_unit_uniform_matches_weighted_index_for_fixed_seeds() {
    use super::weighted_index_from_unit_f32;
    use rand::{
        distr::{weighted::WeightedIndex, Distribution, Uniform},
        RngCore, SeedableRng,
    };
    use rand_isaac::Isaac64Rng;

    let top_p_truncated = |mut weights: Vec<f32>, top_p: f32| {
        let cutoff = top_p * weights.iter().sum::<f32>();
        let mut cumulative = 0.0f32;
        for weight in &mut weights {
            if cumulative >= cutoff {
                *weight = 0.0;
            } else {
                cumulative += *weight;
            }
        }
        weights
    };
    let distributions = [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.125, 0.0, 8.0, 1.0],
        vec![1.0e-7, 3.0e-5, 0.25, 9.0],
        top_p_truncated(vec![0.42, 0.31, 0.17, 0.07, 0.03], 0.72),
        top_p_truncated(vec![1.0; 20], 0.95),
    ];
    let unit = Uniform::new(0.0f32, 1.0f32).unwrap();

    for weights in distributions {
        let reference = WeightedIndex::new(&weights).unwrap();
        for seed in 0..4096 {
            let mut reference_rng = Isaac64Rng::seed_from_u64(seed);
            let expected = reference.sample(&mut reference_rng);
            let expected_next = reference_rng.next_u64();

            let mut resident_rng = Isaac64Rng::seed_from_u64(seed);
            let uniform = unit.sample(&mut resident_rng);
            let actual = weighted_index_from_unit_f32(&weights, uniform).unwrap();
            let actual_next = resident_rng.next_u64();

            assert_eq!(actual, expected, "seed={seed}, weights={weights:?}");
            assert_eq!(actual_next, expected_next, "seed={seed}");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_categorical_row_parses_token_and_logprob() {
    use super::Sampler;

    let sampler = Sampler::new(
        Some(1.0),
        0,
        None,
        None,
        None,
        None,
        None,
        -1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let sampled = sampler.sample_cuda_categorical_row(&[17.0, -2.5]).unwrap();

    assert_eq!(sampled.token, 17);
    assert_eq!(sampled.logprob, -2.5);
    assert!(sampler
        .sample_cuda_categorical_row(&[f32::NAN, f32::NAN])
        .is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_top1_parser_rejects_invalid_output() {
    use super::Sampler;

    assert_eq!(Sampler::cuda_top1_token([3.0, 9.0]).unwrap(), 9);
    assert!(Sampler::cuda_top1_token([f32::NAN, f32::NAN]).is_err());
    assert!(Sampler::cuda_top1_token([3.0, -1.0]).is_err());
    assert!(Sampler::cuda_top1_token([3.0, 1.5]).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_top1_row_uses_zero_logprob_and_validates_shape() {
    use super::Sampler;

    let sampler = Sampler::new(
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        -1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let sampled = sampler.sample_cuda_top1_row(&[4.5, 17.0]).unwrap();

    assert_eq!(sampled.token, 17);
    assert_eq!(sampled.logprob, 0.0);
    assert!(sampler.sample_cuda_top1_row(&[4.5]).is_err());
    assert!(sampler.sample_cuda_top1_row(&[f32::NAN, f32::NAN]).is_err());
}
