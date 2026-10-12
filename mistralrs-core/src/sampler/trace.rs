use std::sync::{Arc, Mutex};

use candle_core::{Result, Tensor};
use llm_watermarking::synthid::generation_tournament::{GenerationRngInfo, NoTournamentReason};
use llm_watermarking::trace::{SelectionScoreKind, TraceOptions, TraceStatus, TraceView};
use rand::RngCore;
use rand_isaac::Isaac64Rng;

use super::{argmax_f32, partial_sort_top_k, Logprobs, Sampler};
use crate::sampling_trace::{
    GenerationTournament, SamplingTraceConfig, SamplingTraceStep, TeachingTournament,
    TextGrainGeneration, TextGrainTrace, TraceCandidate, TraceInverse, TraceLayer, TraceMembership,
    TraceWatermark, GENERATION_RNG_VERSION,
};

pub(crate) struct TraceStepContext<'a> {
    pub context: &'a [u32],
    pub prompt_len: usize,
    pub options: SamplingTraceConfig,
    pub sampling_seed: Option<u64>,
}

impl Sampler {
    pub(crate) fn sample_traced(
        &self,
        logits: Tensor,
        step: TraceStepContext<'_>,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<(Logprobs, SamplingTraceStep)> {
        let input = logits.to_vec1::<f32>()?;
        let mut processed = self.apply_penalties(input.clone(), step.context, step.prompt_len)?;
        for processor in &self.logits_processors {
            processed = processor.apply(&processed, step.context)?;
        }
        let processed_values = processed.to_vec1::<f32>()?;
        let reporting = match self.temperature {
            Some(temperature) => candle_nn::ops::softmax_last_dim(&(&processed / temperature)?)?,
            None => candle_nn::ops::softmax_last_dim(&processed)?,
        }
        .to_vec1::<f32>()?;
        let mut sampling = if self.temperature.is_none() {
            let mut weights = vec![0.0; reporting.len()];
            weights[argmax_f32(&processed_values)? as usize] = 1.0;
            weights
        } else {
            reporting.clone()
        };
        if self.watermark.is_some() {
            self.filter_top_kp_min_p(&mut sampling);
            Self::normalize_probs(&mut sampling)?;
        } else if self.temperature.is_some() {
            // The ordinary non-watermark sampler receives F32 filter thresholds.
            let mut filter = self.clone();
            filter.top_p = f64::from(self.top_p as f32);
            filter.min_p = f64::from(self.min_p as f32);
            filter.filter_top_kp_min_p(&mut sampling);
        }
        let before = sampling.clone();
        let explicit = self.uses_tournament() && self.temperature.is_some();
        let explicit_textgrain = self.uses_textgrain_sampling() && self.temperature.is_some();
        let native_options = step.options.textgrain.unwrap_or_default();
        let mut native_trace = None;
        let library_trace = if let Some(watermark) = &self.watermark {
            if self.temperature.is_some()
                && watermark.scheme() == "textgrain"
                && !explicit_textgrain
            {
                native_trace = Some(
                    watermark
                        .resolve(sampling.len())?
                        .textgrain()
                        .map_err(candle_core::Error::wrap)?
                        .apply_traced(
                            &mut sampling,
                            step.context,
                            step.prompt_len,
                            &native_options.options(),
                        )
                        .map_err(candle_core::Error::wrap)?,
                );
                Self::normalize_probs(&mut sampling)?;
                None
            } else if self.temperature.is_some() && !explicit && !explicit_textgrain {
                let trace = watermark.resolve(sampling.len())?.apply_traced(
                    &mut sampling,
                    step.context,
                    step.prompt_len,
                    &TraceOptions {
                        max_layers: step.options.max_layers,
                    },
                )?;
                Self::normalize_probs(&mut sampling)?;
                Some(trace)
            } else {
                None
            }
        } else {
            None
        };
        let mut generation_tournament = None;
        let mut native_generation = None;
        let selected = if explicit_textgrain {
            let watermark = self.watermark.as_ref().unwrap().resolve(before.len())?;
            let mut guard = rng.lock().expect("could not lock rng mutex");
            let trace = watermark
                .textgrain()
                .map_err(candle_core::Error::wrap)?
                .sample_traced(
                    &before,
                    step.context,
                    step.prompt_len,
                    &mut || guard.next_u64(),
                    &native_options.options(),
                )
                .map_err(candle_core::Error::wrap)?;
            drop(guard);
            let token = trace.token_id;
            sampling.copy_from_slice(trace.trace.output());
            Self::normalize_probs(&mut sampling)?;
            native_generation = Some(TextGrainGeneration::from_library(
                &trace,
                step.sampling_seed,
            ));
            native_trace = Some(trace.trace);
            self.logprobs_from_probs(token, &reporting, true)?
        } else if explicit {
            let watermark = self.watermark.as_ref().unwrap().resolve(before.len())?;
            let sampler = watermark.production_sampler()?;
            let mut guard = rng.lock().expect("could not lock rng mutex");
            let token = if let Some(config) = step.options.generation_tournament {
                let seed =
                    step.sampling_seed
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                    "explicit tournament tracing requires resolved sequence RNG provenance".into())
                        })?
                        .to_string();
                let (token, bracket) = sampler
                    .sample_traced(
                        &before,
                        step.context,
                        step.prompt_len,
                        &mut || guard.next_u64(),
                        &config.options(step.options.max_layers),
                        GenerationRngInfo {
                            rng_version: GENERATION_RNG_VERSION,
                            effective_seed: &seed,
                        },
                    )
                    .map_err(candle_core::Error::wrap)?;
                generation_tournament = Some(GenerationTournament::from_library(bracket, |id| {
                    self.tokenizer
                        .as_ref()
                        .map(|t| {
                            t.decode(&[id], false)
                                .map_err(|error| candle_core::Error::Msg(error.to_string()))
                        })
                        .transpose()
                }));
                token
            } else {
                sampler
                    .sample(&before, step.context, step.prompt_len, &mut || {
                        guard.next_u64()
                    })
                    .map_err(candle_core::Error::wrap)?
            };
            drop(guard);
            self.logprobs_from_probs(token, &reporting, true)?
        } else if self.watermark.is_some() {
            // Preserve the ordinary path's RNG draw even for a keyed or greedy point mass.
            self.sample_multinomial(&sampling, &reporting, true, rng)?
        } else if self.temperature.is_none() {
            self.sample_argmax(processed, true)?
        } else {
            self.sample_top_kp_min_p(
                &reporting,
                self.top_k,
                self.top_p as f32,
                self.min_p as f32,
                true,
                rng,
            )?
        };
        if step.options.generation_tournament.is_some() && generation_tournament.is_none() {
            let depth = self.watermark.as_ref().and_then(|w| w.synthid_depth());
            let reason = if self.watermark.is_none() {
                NoTournamentReason::WatermarkDisabled
            } else if self.temperature.is_none() {
                NoTournamentReason::Greedy
            } else if depth.is_some() {
                NoTournamentReason::ProbabilityUpdates
            } else {
                NoTournamentReason::UnsupportedSampling
            };
            generation_tournament = Some(GenerationTournament::not_run(
                depth,
                reason,
                step.sampling_seed,
            ));
        }
        let keyed = explicit
            || library_trace
                .as_ref()
                .is_some_and(|trace| trace.output_weights().is_none());
        let ranking_input = if self.temperature.is_none() {
            &reporting
        } else {
            &before
        };
        let ids = candidate_ids(
            selected.token,
            ranking_input,
            (!keyed).then_some(sampling.as_slice()),
            step.options.max_candidates,
        );
        let snapshot = library_trace
            .as_ref()
            .map(|trace| {
                trace
                    .snapshot(
                        Some(&ids),
                        &TraceView {
                            max_rows: step.options.max_candidates,
                            ..TraceView::default()
                        },
                    )
                    .map_err(candle_core::Error::wrap)
            })
            .transpose()?;
        let textgrain = native_trace
            .as_ref()
            .map(|trace| {
                trace
                    .snapshot(
                        Some(&ids),
                        &native_options.view(step.options.max_candidates),
                    )
                    .map(|snapshot| TextGrainTrace::from_library(snapshot, native_generation))
                    .map_err(candle_core::Error::wrap)
            })
            .transpose()?;
        let before_total: f64 = before.iter().map(|&p| f64::from(p)).sum();
        let after_total: f64 = sampling.iter().map(|&p| f64::from(p)).sum();
        let scheme = self.watermark.as_ref().map(|w| w.scheme());
        let mut candidates = Vec::with_capacity(ids.len());
        for (row, &token_id) in ids.iter().enumerate() {
            let index = token_id as usize;
            let after = (!keyed).then(|| f64::from(sampling[index]) / after_total);
            let score = snapshot
                .as_ref()
                .and_then(|s| s.selection_scores.as_ref())
                .map(|s| s[row]);
            let inverse = snapshot.as_ref().and_then(|s| s.inverse.as_ref());
            candidates.push(TraceCandidate {
                token_id,
                text: self
                    .tokenizer
                    .as_ref()
                    .map(|t| {
                        t.decode(&[token_id], false)
                            .map_err(|error| candle_core::Error::Msg(error.to_string()))
                    })
                    .transpose()?,
                input_logit: input[index].is_finite().then_some(input[index]),
                processed_logit: processed_values[index]
                    .is_finite()
                    .then_some(processed_values[index]),
                reporting_probability: reporting[index],
                pre_watermark_probability: f64::from(before[index]) / before_total,
                post_watermark_probability: after,
                post_watermark_log_probability: after.filter(|&p| p > 0.0).map(f64::ln),
                membership: snapshot
                    .as_ref()
                    .and_then(|s| s.favored_mask.as_ref())
                    .map(|mask| TraceMembership {
                        kind: if scheme == Some("mpac") {
                            "favored"
                        } else {
                            "green"
                        }
                        .into(),
                        favored: mask[row],
                    }),
                selection_score: score.filter(|s| s.is_finite()),
                selection_score_status: score
                    .filter(|s| !s.is_finite())
                    .map(|_| "negative_infinity".into()),
                inverse_rank: inverse.map(|i| i.ranks[row]),
                cdf_lower: inverse.map(|i| i.cdf_lower[row]),
                cdf_upper: inverse.map(|i| i.cdf_upper[row]),
            });
        }
        let watermark = scheme.map(|scheme| TraceWatermark {
            scheme: scheme.into(),
            status: snapshot
                .as_ref()
                .map(|s| match s.status {
                    TraceStatus::Applied => "applied",
                    TraceStatus::Warmup => "warmup",
                    TraceStatus::RepeatedContext => "repeated_context",
                })
                .or_else(|| textgrain.as_ref().map(|t| t.status.as_str()))
                .unwrap_or_else(|| {
                    if explicit {
                        generation_tournament
                            .as_ref()
                            .map_or("explicit_tournament", |t| t.status.as_str())
                    } else {
                        "greedy"
                    }
                })
                .into(),
            bias_delta: snapshot.as_ref().and_then(|s| s.bias_delta),
            payload_position: snapshot.as_ref().and_then(|s| s.payload_position),
            payload_symbol: snapshot.as_ref().and_then(|s| s.payload_symbol),
            key_position: snapshot.as_ref().and_then(|s| s.key_position),
            score_kind: snapshot.as_ref().and_then(|s| s.score_kind).map(|kind| {
                match kind {
                    SelectionScoreKind::NegativeExponentialCost => "negative_exponential_cost",
                    SelectionScoreKind::GumbelMax => "gumbel_max",
                    SelectionScoreKind::NegativeRank => "negative_rank",
                }
                .into()
            }),
            inverse: snapshot
                .as_ref()
                .and_then(|s| s.inverse.as_ref())
                .map(|i| TraceInverse {
                    uniform: i.uniform,
                    threshold: i.threshold,
                    total_weight: i.total_weight,
                }),
        });
        let layers_truncated = snapshot
            .as_ref()
            .is_some_and(|s| s.captured_layers < s.total_layers);
        let layers = snapshot
            .into_iter()
            .flat_map(|s| s.layers)
            .map(|l| TraceLayer {
                index: l.index,
                green_mass: l.green_mass,
                input_normalizer: l.input_normalizer,
                g_values: l.g_values,
                input_probabilities: l.input_probabilities,
                probabilities: l.probabilities,
            })
            .collect();
        let generated_index = step.context.len() - step.prompt_len;
        let teaching_tournament = step
            .options
            .teaching_tournament
            .map(|config| {
                let watermark = self.watermark.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg(
                        "teaching_tournament requires a SynthID watermark".into(),
                    )
                })?;
                let options = config.options(generated_index);
                if self.temperature.is_none() {
                    return Ok(TeachingTournament::greedy(
                        options,
                        watermark.teaching_depth()?,
                    ));
                }
                let demo = watermark.resolve(before.len())?.tournament_demo(
                    &before,
                    step.context,
                    step.prompt_len,
                    &options,
                )?;
                TeachingTournament::from_library(demo, |id| {
                    self.tokenizer
                        .as_ref()
                        .map(|t| {
                            t.decode(&[id], false)
                                .map_err(|error| candle_core::Error::Msg(error.to_string()))
                        })
                        .transpose()
                })
            })
            .transpose()?;
        let trace = SamplingTraceStep {
            generated_index,
            context_length: step.context.len(),
            attempt: 0,
            selected_token_id: selected.token,
            selection_rule: if self.temperature.is_none() {
                "greedy"
            } else if explicit {
                "synthid_tournament"
            } else if explicit_textgrain {
                "textgrain_block_then_token"
            } else if keyed {
                "keyed_argmax"
            } else {
                "categorical"
            }
            .into(),
            candidate_count: before.len(),
            candidates_truncated: ids.len() < before.len(),
            watermark,
            candidates,
            layers,
            layers_truncated,
            teaching_tournament,
            generation_tournament,
            textgrain,
        };
        Ok((selected, trace))
    }
}

fn candidate_ids(selected: u32, before: &[f32], after: Option<&[f32]>, limit: usize) -> Vec<u32> {
    let before = partial_sort_top_k(&mut before.to_vec(), limit, false);
    let after = after
        .map(|p| partial_sort_top_k(&mut p.to_vec(), limit, false))
        .unwrap_or_default();
    let mut ids = vec![selected];
    for index in 0..limit {
        for entry in [before.get(index), after.get(index)].into_iter().flatten() {
            if ids.len() < limit && !ids.contains(&entry.0) {
                ids.push(entry.0);
            }
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::{RngCore, SeedableRng};
    use std::collections::HashMap;

    const VOCAB: usize = 17;

    fn production_config() -> crate::WatermarkConfig {
        crate::WatermarkConfig::Synthid {
            key: "42".repeat(32),
            ngram_len: 5,
            depth: 4,
            generation_policy: crate::SynthIdGenerationPolicy::Tournament,
        }
    }

    fn production_parity(device: &Device) -> anyhow::Result<()> {
        use crate::sampling_trace::GenerationTournamentConfig;
        let logits = Tensor::from_vec(
            (0..VOCAB).map(|i| (i as f32 * 0.7).sin()).collect(),
            VOCAB,
            device,
        )?;
        let config = production_config();
        for temperature in [None, Some(0.7)] {
            let sampler = sampler(temperature).with_watermark(Some(&config))?;
            for (context, status) in [
                (vec![1], "warmup"),
                (vec![1, 2, 3, 4, 5], "applied"),
                (vec![1, 2, 3, 4, 1, 2, 3, 4], "repeated_context"),
            ] {
                for seed in 0..8 {
                    for capture in [
                        None,
                        Some((0, 32)),
                        Some((15, 0)),
                        Some((3, 2)),
                        Some((15, 32)),
                    ] {
                        let plain_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let traced_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let expected = sampler.sample(
                            logits.clone(),
                            &context,
                            1,
                            true,
                            plain_rng.clone(),
                            false,
                            false,
                        )?;
                        let (actual, trace) = sampler.sample_traced(
                            logits.clone(),
                            TraceStepContext {
                                context: &context,
                                prompt_len: 1,
                                sampling_seed: Some(seed),
                                options: SamplingTraceConfig {
                                    max_steps: 1,
                                    max_candidates: 1,
                                    max_layers: capture.map_or(0, |c| c.1),
                                    generation_tournament: capture
                                        .map(|c| GenerationTournamentConfig { max_matches: c.0 }),
                                    ..Default::default()
                                },
                            },
                            traced_rng.clone(),
                        )?;
                        assert_eq!(
                            serde_json::to_value(&expected)?,
                            serde_json::to_value(&actual)?
                        );
                        assert_eq!(
                            plain_rng.lock().unwrap().next_u64(),
                            traced_rng.lock().unwrap().next_u64()
                        );
                        if capture.is_none() {
                            assert!(serde_json::to_value(&trace)?
                                .get("generation_tournament")
                                .is_none());
                            continue;
                        }
                        let bracket = trace.generation_tournament.unwrap();
                        assert_eq!(bracket.effective_seed, Some(seed.to_string()));
                        assert_eq!(
                            bracket.status,
                            if temperature.is_none() {
                                "greedy"
                            } else {
                                status
                            }
                        );
                        if bracket.used_for_generation {
                            assert_eq!(bracket.winner.as_ref().unwrap().token_id, actual.token);
                            assert_eq!(bracket.total_draws, 16);
                            assert_eq!(bracket.total_matches, 15);
                            assert_eq!(bracket.rounds, 4);
                            assert_eq!(bracket.configured_depth, Some(4));
                            assert_eq!(trace.selection_rule, "synthid_tournament");
                            assert!(trace.candidates[0].post_watermark_probability.is_none());
                            assert!(trace.layers.is_empty());
                            let (matches, layers) = capture.unwrap();
                            assert!(bracket.matches.len() <= matches);
                            if matches == 0 || layers == 0 {
                                assert_eq!(bracket.collapsed_subtrees.len(), 1);
                                assert_eq!(bracket.draws.len(), 1);
                            }
                        } else {
                            assert!(bracket.winner.is_none());
                            assert!(bracket.draws.is_empty());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn generation_tournament_preserves_tokens_logprobs_and_live_rng() -> anyhow::Result<()> {
        production_parity(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn generation_tournament_metal_policy_is_independent_of_capture() -> anyhow::Result<()> {
        production_parity(&Device::new_metal(0)?)
    }

    #[test]
    fn generation_tournament_reports_actual_absence_without_changing_sampling() -> anyhow::Result<()>
    {
        use crate::sampling_trace::GenerationTournamentConfig;
        for (config, temperature, status) in [
            (None, Some(0.7), "watermark_disabled"),
            (
                Some(crate::watermark::token_configs(VOCAB)[0].clone()),
                Some(0.7),
                "no_production_bracket",
            ),
            (
                Some(crate::watermark::token_configs(VOCAB)[1].clone()),
                Some(0.7),
                "unsupported_sampling",
            ),
            (Some(production_config()), None, "greedy"),
        ] {
            let sampler = sampler(temperature).with_watermark(config.as_ref())?;
            let logits = Tensor::from_vec(vec![0.1f32; VOCAB], VOCAB, &Device::Cpu)?;
            let context = [1, 2, 3, 4];
            let plain_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
            let traced_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
            let expected = sampler.sample(
                logits.clone(),
                &context,
                4,
                true,
                plain_rng.clone(),
                false,
                false,
            )?;
            let (actual, trace) = sampler.sample_traced(
                logits,
                TraceStepContext {
                    context: &context,
                    prompt_len: 4,
                    sampling_seed: None,
                    options: SamplingTraceConfig {
                        max_steps: 1,
                        generation_tournament: Some(GenerationTournamentConfig::default()),
                        ..Default::default()
                    },
                },
                traced_rng.clone(),
            )?;
            assert_eq!(
                serde_json::to_value(expected)?,
                serde_json::to_value(actual)?
            );
            assert_eq!(
                plain_rng.lock().unwrap().next_u64(),
                traced_rng.lock().unwrap().next_u64()
            );
            let report = trace.generation_tournament.unwrap();
            assert_eq!(report.status, status);
            assert!(!report.used_for_generation);
            assert!(report.winner.is_none());
            assert!(report.effective_seed.is_none());
            assert_eq!(report.rng_provenance, "unavailable_shared_stream");
        }
        let mut config = serde_json::to_value(production_config())?;
        config["depth"] = 30.into();
        assert!(serde_json::from_value::<crate::WatermarkConfig>(config)?
            .validate_generation()
            .is_err());
        Ok(())
    }

    fn sampler(temperature: Option<f64>) -> Sampler {
        Sampler::new(
            temperature,
            4,
            None,
            Some(0.1),
            Some(0.2),
            Some(1.1),
            None,
            8,
            0.8,
            0.02,
            HashMap::from([(2, -0.5)]),
            vec![],
        )
        .unwrap()
    }

    fn parity(device: &Device) -> anyhow::Result<()> {
        let logits = Tensor::from_vec(
            (0..VOCAB).map(|i| (i as f32 * 0.7).sin()).collect(),
            VOCAB,
            device,
        )?;
        let configs = std::iter::once(None)
            .chain(crate::watermark::token_configs(VOCAB).into_iter().map(Some));
        for config in configs {
            for temperature in [None, Some(0.7)] {
                let sampler = sampler(temperature).with_watermark(config.as_ref())?;
                for context in [vec![1], vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 1, 2, 3, 4]] {
                    for seed in 0..12 {
                        let ordinary_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let traced_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let ordinary = sampler.sample(
                            logits.clone(),
                            &context,
                            1,
                            true,
                            ordinary_rng.clone(),
                            false,
                            false,
                        )?;
                        let (traced, trace) = sampler.sample_traced(
                            logits.clone(),
                            TraceStepContext {
                                sampling_seed: None,
                                context: &context,
                                prompt_len: 1,
                                options: SamplingTraceConfig {
                                    max_candidates: 4,
                                    max_layers: 2,
                                    ..Default::default()
                                },
                            },
                            traced_rng.clone(),
                        )?;
                        assert_eq!(
                            serde_json::to_value(ordinary)?,
                            serde_json::to_value(&traced)?,
                            "{config:?}"
                        );
                        assert_eq!(
                            ordinary_rng.lock().unwrap().next_u64(),
                            traced_rng.lock().unwrap().next_u64()
                        );
                        assert_eq!(trace.candidates[0].token_id, traced.token);
                        assert!(trace.candidates.len() <= 4);
                        assert_eq!(trace.generated_index, context.len() - 1);
                        assert!(trace.layers.len() <= 2);
                        for layer in &trace.layers {
                            for row in 0..trace.candidates.len() {
                                let expected = layer.input_probabilities[row]
                                    * (1.0 + f64::from(layer.g_values[row]) - layer.green_mass);
                                assert!((layer.probabilities[row] - expected).abs() < 1e-12);
                            }
                        }
                        let serialized = serde_json::to_string(&trace)?;
                        assert!(!serialized.contains(&"42".repeat(32)));
                        if trace.selection_rule == "keyed_argmax" {
                            assert!(trace
                                .candidates
                                .iter()
                                .all(|c| c.post_watermark_probability.is_none()));
                            assert!(trace.watermark.as_ref().unwrap().score_kind.is_some());
                        } else if config.is_none() {
                            for candidate in trace.candidates {
                                assert_eq!(
                                    Some(candidate.pre_watermark_probability),
                                    candidate.post_watermark_probability
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn teaching_tournament_preserves_generation_trace_and_rng() -> anyhow::Result<()> {
        teaching_parity(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn teaching_tournament_metal_host_path_parity() -> anyhow::Result<()> {
        teaching_parity(&Device::new_metal(0)?)
    }

    fn teaching_parity(device: &Device) -> anyhow::Result<()> {
        use crate::sampling_trace::TeachingTournamentConfig;
        let config = &crate::watermark::token_configs(VOCAB)[0];
        let logits = Tensor::from_vec(
            (0..VOCAB).map(|i| (i as f32 * 0.7).sin()).collect(),
            VOCAB,
            device,
        )?;
        for temperature in [None, Some(0.7)] {
            let sampler = sampler(temperature).with_watermark(Some(config))?;
            for (context, expected_status) in [
                (vec![1], "warmup"),
                (vec![1, 2, 3, 4, 5], "demonstrated"),
                (vec![1, 2, 3, 4, 1, 2, 3, 4], "repeated_context"),
            ] {
                for seed in 0..12 {
                    for rounds in [1, 4] {
                        let options = SamplingTraceConfig {
                            max_candidates: VOCAB,
                            max_layers: 4,
                            ..Default::default()
                        };
                        let plain_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let demo_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let (plain, trace) = sampler.sample_traced(
                            logits.clone(),
                            TraceStepContext {
                                sampling_seed: None,
                                context: &context,
                                prompt_len: 1,
                                options,
                            },
                            plain_rng.clone(),
                        )?;
                        let diagnostic = TeachingTournamentConfig { rounds, seed };
                        let (selected, mut with_demo) = sampler.sample_traced(
                            logits.clone(),
                            TraceStepContext {
                                sampling_seed: None,
                                context: &context,
                                prompt_len: 1,
                                options: SamplingTraceConfig {
                                    teaching_tournament: Some(diagnostic),
                                    ..options
                                },
                            },
                            demo_rng.clone(),
                        )?;
                        assert_eq!(
                            serde_json::to_value(plain)?,
                            serde_json::to_value(selected)?
                        );
                        assert_eq!(
                            plain_rng.lock().unwrap().next_u64(),
                            demo_rng.lock().unwrap().next_u64()
                        );
                        let demo = with_demo.teaching_tournament.take().unwrap();
                        assert_eq!(
                            serde_json::to_value(&trace)?,
                            serde_json::to_value(with_demo)?
                        );
                        assert_eq!(demo.origin, "teaching_simulation");
                        assert!(!demo.used_for_generation);
                        assert_eq!(
                            demo.status,
                            if temperature.is_none() {
                                "greedy"
                            } else {
                                expected_status
                            }
                        );
                        assert_eq!(
                            demo.effective_seed,
                            diagnostic.options(context.len() - 1).seed.to_string()
                        );
                        assert_eq!(demo.requested_rounds, rounds);
                        if demo.status != "demonstrated" {
                            assert_eq!(demo.rounds, 0);
                            assert!(
                                demo.draws.is_empty()
                                    && demo.matches.is_empty()
                                    && demo.winner.is_none()
                            );
                            continue;
                        }
                        assert_eq!(demo.draws.len(), 1 << rounds);
                        assert_eq!(demo.matches.len(), (1 << rounds) - 1);
                        for draw in &demo.draws {
                            let row = trace
                                .candidates
                                .iter()
                                .find(|c| c.token_id == draw.token_id)
                                .unwrap();
                            assert_eq!(draw.probability, row.pre_watermark_probability);
                        }
                        for m in &demo.matches {
                            for entrant in [&m.left, &m.right] {
                                let draw = &demo.draws[entrant.draw_id];
                                let row = trace
                                    .candidates
                                    .iter()
                                    .position(|c| c.token_id == draw.token_id)
                                    .unwrap();
                                assert_eq!(entrant.g_value, trace.layers[m.round].g_values[row]);
                                match entrant.source.kind.as_str() {
                                    "draw" => assert_eq!(entrant.source.id, entrant.draw_id),
                                    "match" => {
                                        assert!(entrant.source.id < m.match_id);
                                        assert_eq!(
                                            demo.matches[entrant.source.id].winner_draw_id,
                                            entrant.draw_id
                                        );
                                    }
                                    _ => panic!("unknown source"),
                                }
                            }
                            let winner = if m.winner == "left" {
                                &m.left
                            } else {
                                &m.right
                            };
                            assert_eq!(m.winner_draw_id, winner.draw_id);
                            assert_eq!(
                                m.reason,
                                if m.left.g_value == m.right.g_value {
                                    "random_tie"
                                } else {
                                    "higher_g"
                                }
                            );
                            assert!(
                                winner.g_value >= m.left.g_value
                                    && winner.g_value >= m.right.g_value
                            );
                        }
                        let winner = demo.winner.as_ref().unwrap();
                        assert_eq!(demo.matches[winner.match_id].winner_draw_id, winner.draw_id);
                        assert_eq!(demo.draws[winner.draw_id].token_id, winner.token_id);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn teaching_tournament_uses_full_support_with_replay_and_duplicates() -> anyhow::Result<()> {
        use crate::sampling_trace::TeachingTournamentConfig;
        let config = crate::WatermarkConfig::Synthid {
            generation_policy: Default::default(),
            key: "42".repeat(32),
            ngram_len: 2,
            depth: 4,
        };
        let sampler = Sampler::new(
            Some(1.0),
            1,
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
        )?
        .with_watermark(Some(&config))?;
        let options = SamplingTraceConfig {
            max_candidates: 1,
            max_layers: 0,
            teaching_tournament: Some(TeachingTournamentConfig {
                rounds: 4,
                seed: u64::MAX,
            }),
            ..Default::default()
        };
        let mut prior_demo = None;
        for generation_seed in [1, 2] {
            let (_, trace) = sampler.sample_traced(
                Tensor::new(&[0.0f32, 0.0, 0.0, f32::NEG_INFINITY], &Device::Cpu)?,
                TraceStepContext {
                    sampling_seed: None,
                    context: &[1],
                    prompt_len: 1,
                    options,
                },
                Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(generation_seed))),
            )?;
            assert!(trace.layers.is_empty());
            let demo = trace.teaching_tournament.unwrap();
            assert!(demo
                .draws
                .iter()
                .any(|d| d.token_id != trace.candidates[0].token_id));
            let mut seen = std::collections::HashSet::new();
            let mut duplicate = false;
            for (index, draw) in demo.draws.iter().enumerate() {
                assert_eq!(draw.draw_id, index);
                assert!(draw.token_id < 3);
                assert_eq!(draw.probability, 1.0 / 3.0);
                duplicate |= !seen.insert(draw.token_id);
            }
            assert!(duplicate);
            let serialized = serde_json::to_value(demo)?;
            if let Some(prior) = prior_demo {
                assert_eq!(prior, serialized);
            }
            prior_demo = Some(serialized);
        }
        Ok(())
    }

    #[test]
    fn sampling_trace_preserves_draws_reporting_and_rng() -> anyhow::Result<()> {
        parity(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn sampling_trace_metal_host_path_parity() -> anyhow::Result<()> {
        parity(&Device::new_metal(0)?)
    }

    #[test]
    fn sampling_trace_uses_full_support_and_real_logit_stages() -> anyhow::Result<()> {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let sampler = Sampler::new(
            Some(0.5),
            2,
            None,
            None,
            None,
            None,
            None,
            0,
            1.0,
            1.0 - 1e-10,
            HashMap::from([(3, -1.0)]),
            vec![],
        )?;
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let (_, trace) = sampler.sample_traced(
            logits,
            TraceStepContext {
                sampling_seed: None,
                context: &[1],
                prompt_len: 1,
                options: SamplingTraceConfig {
                    max_candidates: 2,
                    ..Default::default()
                },
            },
            rng,
        )?;
        assert!(
            trace
                .candidates
                .iter()
                .map(|c| c.pre_watermark_probability)
                .sum::<f64>()
                < 1.0
        );
        let row = trace.candidates.iter().find(|c| c.token_id == 3).unwrap();
        assert_eq!(row.input_logit, Some(4.0));
        assert_eq!(row.processed_logit, Some(3.0));
        Ok(())
    }

    #[test]
    fn sampling_trace_keyed_masked_scores_are_json_safe() -> anyhow::Result<()> {
        for config in crate::watermark::token_configs(4)
            .into_iter()
            .filter(|c| matches!(c.scheme(), "exponential" | "inverse_transform"))
        {
            let sampler = Sampler::new(
                Some(1.0),
                1,
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
            )?
            .with_watermark(Some(&config))?;
            let (_, trace) = sampler.sample_traced(
                Tensor::new(&[0.0f32, 1.0, 2.0, 3.0], &Device::Cpu)?,
                TraceStepContext {
                    sampling_seed: None,
                    context: &[1, 2, 3],
                    prompt_len: 2,
                    options: SamplingTraceConfig {
                        max_candidates: 4,
                        ..Default::default()
                    },
                },
                Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42))),
            )?;
            assert_eq!(trace.selected_token_id, 3);
            assert!(trace
                .candidates
                .iter()
                .any(
                    |c| c.selection_score_status.as_deref() == Some("negative_infinity")
                        && c.selection_score.is_none()
                ));
            let value = serde_json::to_value(trace)?;
            assert!(value["candidates"][0]
                .get("post_watermark_probability")
                .is_none());
        }
        Ok(())
    }

    #[test]
    fn textgrain_watermark_native_traces_preserve_sampling_and_expose_transport(
    ) -> anyhow::Result<()> {
        use crate::sampling_trace::TextGrainTraceConfig;
        let logits = Tensor::from_vec(
            (0..VOCAB).map(|i| (i as f32 * 0.7).sin()).collect(),
            VOCAB,
            &Device::Cpu,
        )?;
        for policy in ["probability_updates", "block_then_token"] {
            let config: crate::WatermarkConfig = serde_json::from_value(serde_json::json!({
                "scheme": "textgrain", "key": "42".repeat(32), "context_width": 2,
                "block_count": 4, "column_count": 5, "max_iterations": 16,
                "generation_policy": policy
            }))?;
            let detector = crate::Watermark::new(&config)?;
            let probability_check = sampler(Some(0.7))
                .with_watermark(Some(&config))?
                .speculative_probs(logits.clone(), &[1, 2, 3, 4], 0);
            assert_eq!(probability_check.is_ok(), policy == "probability_updates");
            for temperature in [None, Some(0.7)] {
                let sampler = sampler(temperature).with_watermark(Some(&config))?;
                for context in [vec![1], vec![1, 2, 3, 4], vec![1, 1, 1, 1]] {
                    for capture in [
                        None,
                        Some(TextGrainTraceConfig {
                            max_iterations: 1,
                            transport: false,
                        }),
                        Some(TextGrainTraceConfig {
                            max_iterations: 16,
                            transport: true,
                        }),
                    ] {
                        let seed = u64::MAX - 7;
                        let plain_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let traced_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                        let expected = sampler.sample(
                            logits.clone(),
                            &context,
                            0,
                            true,
                            plain_rng.clone(),
                            false,
                            false,
                        )?;
                        let (actual, trace) = sampler.sample_traced(
                            logits.clone(),
                            TraceStepContext {
                                context: &context,
                                prompt_len: 0,
                                sampling_seed: Some(seed),
                                options: SamplingTraceConfig {
                                    max_steps: 1,
                                    max_candidates: 3,
                                    textgrain: capture,
                                    ..Default::default()
                                },
                            },
                            traced_rng.clone(),
                        )?;
                        assert_eq!(
                            serde_json::to_value(expected)?,
                            serde_json::to_value(&actual)?
                        );
                        assert_eq!(
                            plain_rng.lock().unwrap().next_u64(),
                            traced_rng.lock().unwrap().next_u64()
                        );
                        assert!(trace.layers.is_empty());
                        assert!(trace.generation_tournament.is_none());
                        assert!(!serde_json::to_string(&trace)?.contains(&"42".repeat(32)));
                        if temperature.is_none() {
                            assert!(trace.textgrain.is_none());
                            assert_eq!(trace.watermark.unwrap().status, "greedy");
                            continue;
                        }
                        if policy == "block_then_token" {
                            let mut state =
                                crate::sampling_trace::TraceState::new(Default::default());
                            state.record(trace.clone())?;
                            let emitted = state.take(false).unwrap();
                            assert_eq!(
                                emitted.steps[0]
                                    .textgrain
                                    .as_ref()
                                    .unwrap()
                                    .generation
                                    .as_ref()
                                    .unwrap()
                                    .selected_token_id,
                                actual.token
                            );
                            let mut invalid = trace.clone();
                            invalid.selected_token_id = (actual.token + 1) % VOCAB as u32;
                            assert!(state.record(invalid).is_err());
                        }
                        let native = trace.textgrain.unwrap();
                        let options = capture.unwrap_or_default();
                        let mut weights = sampler
                            .pre_watermark_probs(logits.clone(), &context, 0)?
                            .sampling;
                        let reference = detector.textgrain()?.apply_traced(
                            &mut weights,
                            &context,
                            0,
                            &options.options(),
                        )?;
                        let reference =
                            reference.snapshot(Some(&native.token_ids), &options.view(3))?;
                        assert_eq!(native.token_ids[0], actual.token);
                        assert_eq!(
                            native.token_ids,
                            trace
                                .candidates
                                .iter()
                                .map(|c| c.token_id)
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(
                            native.status,
                            crate::sampling_trace::textgrain_status(reference.status)
                        );
                        assert_eq!(native.blocks, reference.blocks);
                        assert_eq!(native.output_probabilities, reference.output_probabilities);
                        assert_eq!(native.selected_costs, reference.selected_costs);
                        assert_eq!(native.detection_scores, reference.detection_scores);
                        assert_eq!(
                            native.solver.as_ref().map(|s| s.budget_satisfied),
                            reference.solver.as_ref().map(|s| s.budget_satisfied)
                        );
                        assert_eq!(native.iterations.len(), reference.iterations.len());
                        assert_eq!(native.omitted_iterations, reference.omitted_iterations);
                        assert!(native.iterations.len() <= options.max_iterations);
                        if native.status != "applied" {
                            assert!(native.blocks.is_none());
                            assert!(native.solver.is_none());
                            assert!(native.transport.is_none());
                        }
                        if let Some(transport) = native.transport {
                            let reference = reference.transport.unwrap();
                            assert_eq!(transport.costs, reference.costs);
                            assert_eq!(transport.coupling, reference.coupling);
                            assert_eq!(transport.coupling.len(), 20);
                            for (block, row) in transport.coupling.chunks(5).enumerate() {
                                assert!(
                                    (row.iter().sum::<f64>() - native.block_masses[block]).abs()
                                        < 1e-10
                                );
                            }
                            for column in 0..5 {
                                assert!(
                                    (transport.coupling.chunks(5).map(|r| r[column]).sum::<f64>()
                                        - 0.2)
                                        .abs()
                                        < 1e-10
                                );
                            }
                        }
                        if policy == "block_then_token" {
                            let draws = native.generation.unwrap();
                            assert!(draws.used_for_generation);
                            assert_eq!(draws.selected_token_id, actual.token);
                            assert_eq!(draws.effective_seed, Some(seed.to_string()));
                            assert_eq!(
                                draws.rng_draws,
                                if native.status == "applied" { 2 } else { 1 }
                            );
                            for draw in draws
                                .block_draw
                                .iter()
                                .chain(std::iter::once(&draws.token_draw))
                            {
                                assert!(
                                    draw.uniform >= draw.cdf_lower && draw.uniform < draw.cdf_upper
                                );
                            }
                            if let Some(block) = draws.selected_block {
                                assert_eq!(block, native.blocks.unwrap()[0]);
                            }
                        } else {
                            assert!(native.generation.is_none());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
