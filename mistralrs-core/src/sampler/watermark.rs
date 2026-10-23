use candle_core::{DType, Result, Tensor};
use llm_watermarking::tensor::IndexedCandidates;

use super::{argmax_f32, Logprobs, Sampler};
use crate::WatermarkTensor;

pub(super) struct WatermarkStep<'a> {
    pub context: &'a [u32],
    pub prompt_len: usize,
    pub return_logprobs: bool,
    pub sample_speculative: bool,
    pub multiple_sequences: bool,
}

impl Sampler {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn try_sample_watermarked_topk(
        &self,
        logits: &Tensor,
        step: &WatermarkStep<'_>,
        rng: std::sync::Arc<std::sync::Mutex<rand_isaac::Isaac64Rng>>,
    ) -> Result<Option<Logprobs>> {
        if self.watermark.is_none()
            || !self.can_sample_topk_on_device(
                step.return_logprobs,
                step.sample_speculative,
                step.multiple_sequences,
                logits.device().is_cuda(),
            )
        {
            return Ok(None);
        }
        let temperature = self.temperature.expect("device top-k requires temperature");
        let vocab_size = logits.elem_count();
        #[cfg(feature = "cuda")]
        if logits.device().is_cuda() {
            let logits = self.apply_device_sparse_penalties_if_needed(
                logits.clone(),
                step.context,
                step.prompt_len,
            )?;
            let logits = self.apply_device_logits_bias_if_needed(logits)?;
            let topk =
                crate::ops::cuda_topk_logits_f32_packed(&logits, self.top_k as usize, temperature)?;
            return self
                .sample_watermarked_candidates(&topk.packed, topk.k, vocab_size, step, rng)
                .map(Some);
        }
        #[cfg(feature = "metal")]
        if logits.device().is_metal() {
            let logits = self.apply_device_sparse_penalties_if_needed_metal(
                logits.clone(),
                step.context,
                step.prompt_len,
            )?;
            // The existing Metal top-k kernel takes a base buffer without a byte offset.
            let logits = if logits.layout().start_offset() == 0 {
                logits
            } else {
                logits.force_contiguous()?
            };
            let topk =
                crate::ops::metal_topk_logits_packed(&logits, self.top_k as usize, temperature)?;
            return self
                .sample_watermarked_candidates(&topk.packed, topk.k, vocab_size, step, rng)
                .map(Some);
        }
        Ok(None)
    }

    #[cfg(any(feature = "cuda", feature = "metal", test))]
    fn sample_watermarked_candidates(
        &self,
        packed: &Tensor,
        k: usize,
        vocab_size: usize,
        step: &WatermarkStep<'_>,
        rng: std::sync::Arc<std::sync::Mutex<rand_isaac::Isaac64Rng>>,
    ) -> Result<Logprobs> {
        use rand::distr::Distribution;

        let candidates = self.watermark_candidates(packed, k, vocab_size, step)?;
        let selected = if candidates.keyed_selection {
            argmax_f32(&candidates.values)? as usize
        } else {
            let distribution = rand::distr::weighted::WeightedIndex::new(&candidates.values)
                .map_err(candle_core::Error::wrap)?;
            distribution.sample(&mut *rng.lock().expect("could not lock rng mutex"))
        };
        let token = candidates.token_ids[selected];
        let bytes = self
            .tokenizer
            .as_ref()
            .map(|tokenizer| {
                tokenizer
                    .decode(&[token], false)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            })
            .transpose()?;
        Ok(Logprobs {
            token,
            logprob: candidates.reporting[selected].ln(),
            top_logprobs: None,
            bytes,
        })
    }

    #[cfg(any(feature = "cuda", feature = "metal", test))]
    fn watermark_candidates(
        &self,
        packed: &Tensor,
        k: usize,
        vocab_size: usize,
        step: &WatermarkStep<'_>,
    ) -> Result<WatermarkCandidates> {
        debug_assert!(
            !step.return_logprobs && !step.sample_speculative && !step.multiple_sequences
        );
        let temperature = self.temperature.expect("device top-k requires temperature");
        let values = packed.narrow(0, 0, k)?;
        let token_ids = packed.narrow(0, k, k)?;
        let indices = token_ids.to_dtype(DType::U32)?;
        let denominator = packed.narrow(0, 2 * k, 1)?;
        let maximum = packed.narrow(0, 2 * k + 1, 1)?;
        let reporting = values
            .affine(1.0 / temperature, 0.0)?
            .broadcast_sub(&maximum)?
            .exp()?
            .broadcast_div(&denominator)?;
        let mut sampling = reporting.clone();
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let cutoff = sampling.sum_keepdim(0)?.affine(self.top_p, 0.0)?;
            let cumulative = sampling.cumsum(0)?;
            let preceding = if k == 1 {
                Tensor::zeros(1, DType::F32, packed.device())?
            } else {
                Tensor::cat(
                    &[
                        &Tensor::zeros(1, DType::F32, packed.device())?,
                        &cumulative.narrow(0, 0, k - 1)?,
                    ],
                    0,
                )?
            };
            sampling = sampling.mul(&preceding.broadcast_lt(&cutoff)?.to_dtype(DType::F32)?)?;
        }
        if self.min_p > 0.0 && self.min_p < 1.0 {
            let threshold = reporting.narrow(0, 0, 1)?.affine(self.min_p, 0.0)?;
            sampling = sampling.mul(&sampling.broadcast_gt(&threshold)?.to_dtype(DType::F32)?)?;
        }
        let candidates = IndexedCandidates::new_trusted(&indices, vocab_size)?;
        let watermark = self
            .watermark
            .as_ref()
            .expect("watermark configured")
            .resolve(vocab_size)?;
        let (marked, keyed_selection) =
            match watermark.apply_indexed(&sampling, &candidates, step.context, step.prompt_len)? {
                WatermarkTensor::Probabilities(probs) => (probs, false),
                WatermarkTensor::SelectionScores(scores) => (scores, true),
            };
        // Keep the existing single compact readback for host selection and unwatermarked reporting probabilities.
        let result = Tensor::cat(&[&marked, &token_ids, &reporting], 0)?.to_vec1::<f32>()?;
        Ok(WatermarkCandidates {
            values: result[..k].to_vec(),
            token_ids: result[k..2 * k].iter().map(|id| *id as u32).collect(),
            reporting: result[2 * k..].to_vec(),
            keyed_selection,
        })
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
struct WatermarkCandidates {
    values: Vec<f32>,
    token_ids: Vec<u32>,
    reporting: Vec<f32>,
    keyed_selection: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    const TEST_VOCAB_SIZE: usize = 257;
    const TEST_MAX_TOP_K: usize = 128;
    const TEST_TEMPERATURE: f32 = 0.8;

    fn sampler(config: &crate::WatermarkConfig, top_p: f64, min_p: f64) -> Sampler {
        Sampler::new(
            Some(0.8),
            0,
            None,
            None,
            None,
            None,
            None,
            4,
            top_p,
            min_p,
            HashMap::new(),
            vec![],
        )
        .unwrap()
        .with_watermark(Some(config))
        .unwrap()
    }

    fn compact_parity(device: &Device) -> anyhow::Result<()> {
        let logits = (0..TEST_VOCAB_SIZE)
            .map(|id| ((id * 73 + 19) % TEST_VOCAB_SIZE) as f32 / 32.0 - 4.0)
            .collect::<Vec<_>>();
        let mut ids = (0..TEST_VOCAB_SIZE as u32).collect::<Vec<_>>();
        ids.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
        let maximum = logits[ids[0] as usize] / TEST_TEMPERATURE;
        let denominator: f32 = logits
            .iter()
            .map(|value| (value / TEST_TEMPERATURE - maximum).exp())
            .sum();
        let histories: &[(&[u32], usize)] = &[
            (&[1], 1),
            (&[1, 2, 3, 4], 4),
            (&[1, 2, 3, 4, 5], 4),
            (&[1, 2, 3, 4, 1, 2, 3, 4], 4),
        ];
        for k in [1, 4, TEST_MAX_TOP_K] {
            let ids = &ids[..k];
            let mut packed = ids
                .iter()
                .map(|id| logits[*id as usize])
                .collect::<Vec<_>>();
            packed.extend(ids.iter().map(|id| *id as f32));
            packed.extend([denominator, maximum]);
            let packed = Tensor::new(packed.as_slice(), device)?;
            for mut config in crate::watermark::token_configs(logits.len()) {
                match &mut config {
                    crate::WatermarkConfig::Kgw { context_width, .. }
                    | crate::WatermarkConfig::Mpac { context_width, .. } => {
                        *context_width = 2;
                    }
                    crate::WatermarkConfig::Exponential {
                        start_position,
                        sequence_len,
                        ..
                    }
                    | crate::WatermarkConfig::InverseTransform {
                        start_position,
                        sequence_len,
                        ..
                    } => {
                        *start_position = usize::MAX;
                        *sequence_len = 7;
                    }
                    _ => {}
                }
                for (top_p, min_p) in [(1.0, 0.0), (0.9, 0.01), (0.7, 0.1)] {
                    let mut sampler = sampler(&config, top_p, min_p);
                    sampler.top_k = k as i64;
                    for &(context, prompt_len) in histories {
                        let step = WatermarkStep {
                            context,
                            prompt_len,
                            return_logprobs: false,
                            sample_speculative: false,
                            multiple_sequences: false,
                        };
                        let expected = sampler.speculative_target_probs(
                            Tensor::new(logits.as_slice(), &Device::Cpu)?,
                            context,
                            prompt_len,
                        )?;
                        let actual =
                            sampler.watermark_candidates(&packed, k, logits.len(), &step)?;
                        assert_eq!(actual.token_ids, ids);
                        if actual.keyed_selection {
                            let selected = actual.token_ids[argmax_f32(&actual.values)? as usize];
                            assert_eq!(
                                expected.sampling[selected as usize],
                                1.0,
                                "{}",
                                config.scheme()
                            );
                        } else {
                            let mass: f32 = actual.values.iter().sum();
                            for (rank, id) in ids.iter().enumerate() {
                                assert!(
                                    (actual.values[rank] / mass - expected.sampling[*id as usize])
                                        .abs()
                                        < 3e-4,
                                    "{} K={k} context={context:?} rank={rank}",
                                    config.scheme()
                                );
                                if expected.sampling[*id as usize] == 0.0 {
                                    assert_eq!(actual.values[rank], 0.0);
                                }
                            }
                        }
                        for (rank, id) in ids.iter().enumerate() {
                            assert!(
                                (actual.reporting[rank] - expected.reporting[*id as usize]).abs()
                                    < 1e-6
                            );
                        }
                        let replay = sampler.clone().watermark_candidates(
                            &packed,
                            k,
                            logits.len(),
                            &step,
                        )?;
                        assert_eq!(actual.values, replay.values);
                        let sample = sampler.sample_watermarked_candidates(
                            &packed,
                            k,
                            logits.len(),
                            &step,
                            Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42))),
                        )?;
                        assert!(expected.sampling[sample.token as usize] > 0.0);
                        assert!(
                            (sample.logprob - expected.reporting[sample.token as usize].ln()).abs()
                                < 1e-5
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn watermark_all_schemes_normal_and_speculative_sampling_agree() -> anyhow::Result<()> {
        let logits = Tensor::new(
            &[2.0f32, -3.0, 0.0, -4.0, -2.0, 1.0, -5.0, 3.0],
            &Device::Cpu,
        )?;
        let context = [1, 2, 3, 4];
        for config in crate::watermark::token_configs(8) {
            let sampler = sampler(&config, 0.9, 0.01);
            let expected =
                sampler.speculative_target_probs(logits.clone(), &context, context.len())?;
            let mut tokens = Vec::new();
            for speculative in [false, true] {
                let result = sampler.sample(
                    logits.clone(),
                    &context,
                    context.len(),
                    true,
                    Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42))),
                    speculative,
                    false,
                )?;
                assert!(expected.sampling[result.token as usize] > 0.0);
                assert_eq!(
                    result.logprob,
                    expected.reporting[result.token as usize].ln()
                );
                tokens.push(result.token);
            }
            assert_eq!(tokens[0], tokens[1], "{}", config.scheme());
        }
        Ok(())
    }

    #[test]
    fn watermark_compact_cpu_preserves_ids_filters_and_reporting() -> anyhow::Result<()> {
        compact_parity(&Device::Cpu)
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn watermark_compact_metal_preserves_ids_filters_and_reporting() -> anyhow::Result<()> {
        compact_parity(&Device::new_metal(0)?)
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn watermark_compact_cuda_preserves_ids_filters_and_reporting() -> anyhow::Result<()> {
        compact_parity(&Device::new_cuda(0)?)
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn device_sampling(device: &Device) -> anyhow::Result<()> {
        let logits = Tensor::new(&[2.0f32, -3.0, 0.0, -4.0, -2.0, 1.0, -5.0, 3.0], device)?;
        let mut step = WatermarkStep {
            context: &[1, 2, 3, 4],
            prompt_len: 4,
            return_logprobs: false,
            sample_speculative: false,
            multiple_sequences: false,
        };
        for config in crate::watermark::token_configs(8) {
            let sampler = sampler(&config, 0.9, 0.01);
            let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
            let sample = sampler
                .try_sample_watermarked_topk(&logits, &step, rng.clone())?
                .expect("GPU watermark path selected");
            assert!([0, 5, 7].contains(&sample.token));
            let expected = sampler.speculative_target_probs(
                logits.to_device(&Device::Cpu)?,
                step.context,
                step.prompt_len,
            )?;
            let padded = Tensor::cat(&[&Tensor::full(-50.0f32, 8, device)?, &logits], 0)?;
            let view = padded.narrow(0, 8, 8)?;
            let sample = sampler.sample(
                view,
                step.context,
                step.prompt_len,
                false,
                rng.clone(),
                false,
                false,
            )?;
            assert!(
                (sample.logprob - expected.reporting[sample.token as usize].ln()).abs() < 1e-5,
                "{}: token {}, logprob {}, expected {}",
                config.scheme(),
                sample.token,
                sample.logprob,
                expected.reporting[sample.token as usize].ln()
            );
            assert!(expected.sampling[sample.token as usize] > 0.0);
            step.sample_speculative = true;
            assert!(sampler
                .try_sample_watermarked_topk(&logits, &step, rng)?
                .is_none());
            step.sample_speculative = false;
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn watermark_metal_uses_existing_sampling_kernel() -> anyhow::Result<()> {
        device_sampling(&Device::new_metal(0)?)
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn watermark_cuda_uses_existing_sampling_kernel() -> anyhow::Result<()> {
        device_sampling(&Device::new_cuda(0)?)
    }
}
