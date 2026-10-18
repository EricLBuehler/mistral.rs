use std::{sync::Mutex, time::Instant};

use candle_core::{Device, Tensor};
use indexmap::IndexMap;
use tokenizers::Tokenizer;

use crate::embedding_models::memory::EncoderMemory;

use super::{
    cache::{CacheKey, Role, ScoreKey, VectorCache},
    ClmHeads, DecisionPair, DecisionRequest, DecisionResponse, DecisionUsage,
    DecisionValidationError,
};

const ENCODER_BATCH_SIZE: usize = 32;
const ENCODER_BATCH_TOKENS: usize = 2048;
const MAX_PADDING_RATIO: usize = 4;
const CACHE_HITS_METRIC: &str = "mistralrs_decision_cache_hits_total";
const CACHE_MISSES_METRIC: &str = "mistralrs_decision_cache_misses_total";
const ENCODER_BATCHES_METRIC: &str = "mistralrs_decision_encoder_batches_total";
const ENCODER_TOKENS_METRIC: &str = "mistralrs_decision_encoder_tokens_total";
const INFERENCE_DURATION_METRIC: &str = "mistralrs_decision_inference_duration_seconds";
const SCORE_CACHE_HITS_METRIC: &str = "mistralrs_decision_score_cache_hits_total";

struct InferenceState {
    cache: VectorCache,
    calibrating: bool,
}

pub(crate) struct ClmInference {
    heads: ClmHeads,
    state: Mutex<InferenceState>,
    encoder_memory: EncoderMemory,
}

impl ClmInference {
    pub fn new(
        heads: ClmHeads,
        device: &Device,
        encoder_memory: EncoderMemory,
    ) -> anyhow::Result<Self> {
        let cache = VectorCache::from_env(
            heads.projection_dim(),
            device,
            encoder_memory.workspace_bytes(ENCODER_BATCH_TOKENS),
        )?;
        Ok(Self::with_cache(heads, cache, encoder_memory))
    }

    fn with_cache(heads: ClmHeads, cache: VectorCache, encoder_memory: EncoderMemory) -> Self {
        Self {
            heads,
            state: Mutex::new(InferenceState {
                cache,
                calibrating: false,
            }),
            encoder_memory,
        }
    }

    pub fn invalidate(&self, calibrating: bool) {
        let mut state = self.state.lock().expect("CLM cache lock poisoned");
        state.cache.clear();
        state.calibrating = calibrating;
    }

    pub fn decide(
        &self,
        request: &DecisionRequest,
        model_id: &str,
        tokenizer: &Tokenizer,
        max_tokens: usize,
        encode: impl Fn(&[&[u32]]) -> anyhow::Result<Tensor>,
    ) -> anyhow::Result<DecisionResponse> {
        let started = Instant::now();
        let pairs = request.pairs()?;
        let mut projections = IndexMap::new();
        for pair in &pairs {
            projections.insert((Role::State, pair.state.as_str()), ());
            for text in &pair.candidates {
                projections.insert((Role::Action, text.as_str()), ());
            }
        }
        let keys: Vec<_> = projections
            .keys()
            .map(|(role, text)| CacheKey::new(*role, text))
            .collect();
        let mut state = self.state.lock().expect("CLM cache lock poisoned");
        let score_key = ScoreKey::new(&pairs, request.temperature);
        if !state.calibrating {
            if let Some(logits) = state.cache.scores(&score_key, &keys) {
                drop(state);
                let response = decision_response(request, &pairs, &logits, 0, model_id)?;
                metrics::counter!(CACHE_HITS_METRIC, "model" => model_id.to_owned())
                    .increment(keys.len() as u64);
                metrics::counter!(SCORE_CACHE_HITS_METRIC, "model" => model_id.to_owned())
                    .increment(1);
                metrics::histogram!(INFERENCE_DURATION_METRIC, "model" => model_id.to_owned(), "cache" => "hit")
                    .record(started.elapsed().as_secs_f64());
                return Ok(response);
            }
        }
        let mut vectors = if state.calibrating {
            vec![None; keys.len()]
        } else {
            state.cache.get(&keys)?
        };
        let hits = vectors.iter().filter(|v| v.is_some()).count();
        metrics::counter!(CACHE_HITS_METRIC, "model" => model_id.to_owned()).increment(hits as u64);
        metrics::counter!(CACHE_MISSES_METRIC, "model" => model_id.to_owned())
            .increment((keys.len() - hits) as u64);
        let mut embeddings: IndexMap<&str, Option<Tensor>> = IndexMap::new();
        for ((_, text), vector) in projections.keys().zip(&vectors) {
            if vector.is_none() {
                embeddings.insert(text, None);
            }
        }
        let mut tokens = Vec::with_capacity(embeddings.len());
        let mut input_tokens = 0;
        for text in embeddings.keys() {
            let encoded = tokenizer.encode(*text, true).map_err(anyhow::Error::msg)?;
            let ids = encoded.get_ids().to_vec();
            if ids.is_empty() || ids.len() > max_tokens {
                return Err(DecisionValidationError(format!("Each CLM state with instructions and each candidate must contain 1..={max_tokens} tokens; got {}", ids.len())).into());
            }
            input_tokens += ids.len();
            tokens.push(ids);
        }
        let token_budget = if tokens.is_empty() {
            ENCODER_BATCH_TOKENS
        } else {
            self.encoder_memory.token_budget(ENCODER_BATCH_TOKENS)?
        };
        let batches = encoder_batches(&tokens, token_budget);
        for batch in &batches {
            let inputs: Vec<_> = batch.iter().map(|&i| tokens[i].as_slice()).collect();
            let xs = encode(&inputs)?;
            for (row, &index) in batch.iter().enumerate() {
                *embeddings
                    .get_index_mut(index)
                    .expect("missing text index")
                    .1 = Some(xs.narrow(0, row, 1)?);
            }
        }
        metrics::counter!(ENCODER_BATCHES_METRIC, "model" => model_id.to_owned())
            .increment(batches.len() as u64);
        metrics::counter!(ENCODER_TOKENS_METRIC, "model" => model_id.to_owned())
            .increment(input_tokens as u64);
        for role in [Role::State, Role::Action] {
            let missing: Vec<_> = projections
                .keys()
                .enumerate()
                .filter_map(|(i, (kind, _))| (*kind == role && vectors[i].is_none()).then_some(i))
                .collect();
            for batch in missing.chunks(ENCODER_BATCH_SIZE) {
                let xs: Vec<_> = batch
                    .iter()
                    .map(|&i| {
                        let (_, text) = projections.get_index(i).expect("projection index").0;
                        embeddings[text].as_ref().expect("encoded text")
                    })
                    .collect();
                let projected = self
                    .heads
                    .project(&Tensor::cat(&xs, 0)?, role == Role::State)?;
                if !state.calibrating {
                    let keys: Vec<_> = batch.iter().map(|&i| keys[i]).collect();
                    state.cache.insert(&keys, &projected)?;
                }
                for (row, &index) in batch.iter().enumerate() {
                    vectors[index] = Some(projected.narrow(0, row, 1)?);
                }
            }
        }
        let vector = |role, text| {
            vectors[projections
                .get_index_of(&(role, text))
                .expect("projection key")]
            .as_ref()
            .expect("projected text")
        };
        let mut scores = Vec::with_capacity(pairs.len());
        for pair in &pairs {
            let actions = Tensor::cat(
                &pair
                    .candidates
                    .iter()
                    .map(|text| vector(Role::Action, text.as_str()))
                    .collect::<Vec<_>>(),
                0,
            )?;
            scores.push(self.heads.score(
                vector(Role::State, pair.state.as_str()),
                &actions,
                request.temperature,
            )?);
        }
        let logits = Tensor::cat(&scores, 0)?.to_vec1::<f32>()?;
        if !state.calibrating {
            state.cache.insert_scores(score_key, &keys, &logits);
        }
        drop(state);
        let response = decision_response(request, &pairs, &logits, input_tokens, model_id)?;
        tracing::debug!(
            hits,
            misses = keys.len() - hits,
            encoder_batches = batches.len(),
            token_budget,
            input_tokens,
            "CLM inference"
        );
        metrics::histogram!(INFERENCE_DURATION_METRIC, "model" => model_id.to_owned(), "cache" => if hits == keys.len() { "hit" } else { "miss" })
            .record(started.elapsed().as_secs_f64());
        Ok(response)
    }
}

fn decision_response(
    request: &DecisionRequest,
    pairs: &[DecisionPair],
    logits: &[f32],
    input_tokens: usize,
    model_id: &str,
) -> anyhow::Result<DecisionResponse> {
    let mut offset = 0;
    let mut answers = IndexMap::new();
    for ((id, question), pair) in request.questions.iter().zip(pairs) {
        let end = offset + pair.keys.len();
        answers.insert(
            id.clone(),
            question.answer(&pair.keys, &logits[offset..end])?,
        );
        offset = end;
    }
    Ok(DecisionResponse {
        model: model_id.to_owned(),
        answers,
        usage: DecisionUsage {
            input_tokens,
            output_tokens: 0,
            billing_units: request.questions.len(),
        },
    })
}

fn encoder_batches(tokens: &[Vec<u32>], token_budget: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<_> = (0..tokens.len()).collect();
    order.sort_by_key(|&i| tokens[i].len());
    let mut batches = Vec::new();
    let mut batch: Vec<usize> = Vec::new();
    for index in order {
        let len = tokens[index].len();
        if !batch.is_empty()
            && (batch.len() == ENCODER_BATCH_SIZE
                || (batch.len() + 1) * len > token_budget
                || len > tokens[batch[0]].len() * MAX_PADDING_RATIO)
        {
            batches.push(std::mem::take(&mut batch));
        }
        batch.push(index);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, path::Path};

    use super::*;
    use crate::decision::{ClmConfig, CLM_MAX_TOKENS};

    fn assert_answers_close(a: &DecisionResponse, b: &DecisionResponse) -> anyhow::Result<()> {
        fn close(a: &serde_json::Value, b: &serde_json::Value) {
            match (a, b) {
                (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
                    assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 2e-6);
                }
                (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
                    assert_eq!(a.len(), b.len());
                    for (key, value) in a {
                        close(value, &b[key]);
                    }
                }
                _ => assert_eq!(a, b),
            }
        }
        close(
            &serde_json::to_value(&a.answers)?,
            &serde_json::to_value(&b.answers)?,
        );
        Ok(())
    }

    #[test]
    fn cached_misses_invalidation_and_disabled_cache_match() -> anyhow::Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clm");
        let config: ClmConfig =
            serde_json::from_str(&std::fs::read_to_string(path.join("config.json"))?)?;
        let make = |bytes| -> anyhow::Result<ClmInference> {
            let heads = ClmHeads::load(&path.join("heads.pt"), &config, &Device::Cpu)?;
            let cache = VectorCache::new(bytes, heads.projection_dim(), &Device::Cpu)?;
            Ok(ClmInference::with_cache(
                heads,
                cache,
                EncoderMemory::new(vec![Device::Cpu], 1)?,
            ))
        };
        let cached = make(8192)?;
        let uncached = make(0)?;
        let tiny = make(16)?;
        #[cfg(feature = "metal")]
        let constrained = {
            let mut inference = make(0)?;
            inference.encoder_memory = EncoderMemory::constrained(Device::new_metal(0)?);
            inference
        };
        let tokenizer = Tokenizer::from_file(path.join("encoder/tokenizer.json"))
            .map_err(anyhow::Error::msg)?;
        let calls = Cell::new(0);
        let encoded_texts = Cell::new(0);
        let encode = |tokens: &[&[u32]]| -> anyhow::Result<Tensor> {
            calls.set(calls.get() + 1);
            encoded_texts.set(encoded_texts.get() + tokens.len());
            let values: Vec<f32> = tokens
                .iter()
                .flat_map(|ids| {
                    (0_u32..8).map(move |i| {
                        f32::from(
                            u16::try_from(ids.iter().copied().sum::<u32>() + i * i + 1).unwrap(),
                        )
                    })
                })
                .collect();
            Ok(Tensor::from_vec(values, (tokens.len(), 8), &Device::Cpu)?)
        };
        let mut request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "model":"test", "state":"invoice", "questions": {
                "q":{"type":"choice", "criteria":{"a":"invoice", "b":"technical"}}
            }
        }))?;
        let run = |model: &ClmInference, request: &DecisionRequest| {
            model.decide(request, "test", &tokenizer, CLM_MAX_TOKENS, encode)
        };
        let cold = run(&cached, &request)?;
        assert_eq!(encoded_texts.get(), 2);
        assert!(cold.usage.input_tokens > 0);
        #[cfg(feature = "metal")]
        {
            let before = calls.get();
            let small = run(&constrained, &request)?;
            assert_eq!(calls.get() - before, 2);
            assert_answers_close(&small, &cold)?;
            assert_eq!(small.usage.input_tokens, cold.usage.input_tokens);
        }
        let before = calls.get();
        let warm = run(&cached, &request)?;
        assert_eq!(calls.get(), before);
        assert_eq!(warm.usage.input_tokens, 0);
        assert_eq!(
            serde_json::to_value(&cold.answers)?,
            serde_json::to_value(&warm.answers)?
        );
        let mut renamed = request.clone();
        let question = renamed.questions.shift_remove("q").unwrap();
        renamed.questions.insert("renamed".into(), question);
        let renamed = run(&cached, &renamed)?;
        assert!(renamed.answers.contains_key("renamed"));
        assert_eq!(renamed.usage.input_tokens, 0);
        request.temperature = 2.0;
        let warm = run(&cached, &request)?;
        assert_eq!(warm.usage.input_tokens, 0);
        let fresh = run(&uncached, &request)?;
        assert_eq!(
            serde_json::to_value(&warm.answers)?,
            serde_json::to_value(&fresh.answers)?
        );
        assert!(run(&uncached, &request)?.usage.input_tokens > 0);

        request.state = serde_json::json!("calm");
        let before = encoded_texts.get();
        let changed = run(&cached, &request)?;
        assert_eq!(encoded_texts.get() - before, 1);
        assert!(changed.usage.input_tokens < fresh.usage.input_tokens);
        for _ in 0..2 {
            let evicted = run(&tiny, &request)?;
            assert_answers_close(&evicted, &changed)?;
            assert!(evicted.usage.input_tokens > 0);
        }
        cached.invalidate(true);
        assert!(run(&cached, &request)?.usage.input_tokens > 0);
        assert!(run(&cached, &request)?.usage.input_tokens > 0);
        cached.invalidate(false);
        assert!(run(&cached, &request)?.usage.input_tokens > 0);
        assert_eq!(run(&cached, &request)?.usage.input_tokens, 0);
        let other_model = make(4096)?;
        assert!(run(&other_model, &request)?.usage.input_tokens > 0);
        Ok(())
    }

    #[test]
    fn batching_limits_padding_and_total_tokens() {
        let tokens = [2, 2, 4, 5, 6, 12, 12, 17, 19, 19, 2048].map(|len| vec![0; len]);
        assert_eq!(
            encoder_batches(&tokens, ENCODER_BATCH_TOKENS),
            [vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9], vec![10]]
        );
        let tokens = vec![vec![0; 4]; 65];
        assert_eq!(
            encoder_batches(&tokens, ENCODER_BATCH_TOKENS)
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [32, 32, 1]
        );
        let tokens = vec![vec![0; 1000]; 3];
        assert_eq!(
            encoder_batches(&tokens, ENCODER_BATCH_TOKENS),
            [vec![0, 1], vec![2]]
        );
        assert_eq!(encoder_batches(&tokens, 1000), [vec![0], vec![1], vec![2]]);
        assert_eq!(encoder_batches(&tokens, 1), [vec![0], vec![1], vec![2]]);
    }
}
