#![cfg(feature = "metal")]

use std::{
    env,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use candle_core::{metal_backend::MetalDevice, Device};
use futures::future::join_all;
use mistralrs::{
    model_builder_trait::build_model_from_pipeline, AnyModelBuilder, IsqBits, MemoryGoal, Model,
    ModelBuilder, ModelDType, RequestBuilder, TextMessageRole, TextMessages,
    UqffMultimodalModelBuilder, UqffTextModelBuilder,
};
use mistralrs_core::{memory_plan::resolve_device_budget, ChatCompletionResponse, MemoryUsage};
use serde::Serialize;
use serde_json::{json, Value};

const BF16_LOGPROB_TOLERANCE: f64 = 0.1;
const SAMPLE_INTERVAL: Duration = Duration::from_micros(100);
const OUTPUT_TOKENS: usize = 32;
const BATCH_SIZE: usize = 8;
const PREFIX_CACHE_SEQUENCES: usize = 32;
const SHORT_REPETITIONS: usize = 4;
const LONG_REPETITIONS: usize = 72;
const LONG_SINGLE_REPETITIONS: usize = 128;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const PASS_COUNT: usize = 2;
const PROMPT_SENTENCE: &str = "The observatory measures light from distant stars. ";
const COHORT_ADMISSION: &str = "sampling_sentinel_v1";
const COHORT_SENTINEL_PROMPT: &str = "Hi.";
const COHORT_SENTINEL_TOKENS: usize = 1;
const COHORT_GATE_TEST_TIMEOUT: Duration = Duration::from_secs(1);

struct CohortGate(Arc<(Mutex<bool>, Condvar)>);

impl CohortGate {
    fn new() -> Self {
        Self(Arc::new((Mutex::new(false), Condvar::new())))
    }

    fn release(&self) {
        let (released, wake) = &*self.0;
        *released.lock().unwrap() = true;
        wake.notify_all();
    }
}

impl Drop for CohortGate {
    fn drop(&mut self) {
        self.release();
    }
}

#[test]
fn cohort_gate_drop_releases_a_waiting_sampler() {
    let gate = CohortGate::new();
    let callback_gate = gate.0.clone();
    let (ready, waiting) = std::sync::mpsc::sync_channel(0);
    let callback = thread::spawn(move || {
        let (released, wake) = &*callback_gate;
        let released = released.lock().unwrap();
        ready.send(()).unwrap();
        let (released, _) = wake
            .wait_timeout_while(released, COHORT_GATE_TEST_TIMEOUT, |released| !*released)
            .unwrap();
        assert!(*released);
    });
    waiting.recv_timeout(COHORT_GATE_TEST_TIMEOUT).unwrap();
    drop(gate);
    callback.join().unwrap();
}

#[derive(Serialize)]
struct Observation {
    start_bytes: usize,
    end_bytes: usize,
    observed_peak_bytes: usize,
    samples: usize,
    max_sample_gap_us: u128,
    elapsed_ms: u128,
}

struct Sampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Observation>>,
}

impl Sampler {
    fn start(device: &MetalDevice) -> Self {
        let device = device.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let start_bytes = device.current_allocated_size();
        let started = Instant::now();
        let handle = thread::spawn(move || {
            let mut observation = Observation {
                start_bytes,
                end_bytes: start_bytes,
                observed_peak_bytes: start_bytes,
                samples: 1,
                max_sample_gap_us: 0,
                elapsed_ms: 0,
            };
            let mut previous = started;
            loop {
                let now = Instant::now();
                observation.end_bytes = device.current_allocated_size();
                observation.observed_peak_bytes =
                    observation.observed_peak_bytes.max(observation.end_bytes);
                observation.samples += 1;
                observation.max_sample_gap_us = observation
                    .max_sample_gap_us
                    .max(now.duration_since(previous).as_micros());
                previous = now;
                if thread_stop.load(Ordering::Relaxed) {
                    break;
                }
                thread::sleep(SAMPLE_INTERVAL);
            }
            observation.elapsed_ms = started.elapsed().as_millis();
            observation
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> Result<Observation> {
        self.stop.store(true, Ordering::Relaxed);
        self.handle
            .take()
            .context("missing allocation observer")?
            .join()
            .map_err(|_| anyhow::anyhow!("allocation observer panicked"))
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn max_sequences() -> Result<usize> {
    let sequences = env::var("MISTRALRS_MEMORY_MAX_SEQS")
        .map(|value| value.parse().context("invalid MISTRALRS_MEMORY_MAX_SEQS"))
        .unwrap_or_else(|_| Ok(mistralrs_core::ExecutionMemoryLimits::default().max_num_seqs))?;
    ensure!(sequences > 0, "MISTRALRS_MEMORY_MAX_SEQS must be nonzero");
    Ok(sequences)
}

fn builder(device: &Device, goal: MemoryGoal, max_num_seqs: usize) -> Result<AnyModelBuilder> {
    let model = env::var("MISTRALRS_MEMORY_MODEL")
        .context("set MISTRALRS_MEMORY_MODEL to a local model directory")?;
    ensure!(
        PathBuf::from(&model).is_dir(),
        "model must be a local directory"
    );
    let mut limits = mistralrs_core::ExecutionMemoryLimits {
        max_num_seqs,
        ..Default::default()
    };
    if let Ok(tokens) = env::var("MISTRALRS_MEMORY_BATCH_TOKENS") {
        limits.max_num_batched_tokens = tokens
            .parse()
            .context("invalid MISTRALRS_MEMORY_BATCH_TOKENS")?;
    }
    let paged = mistralrs::PagedAttentionMetaBuilder::default()
        .with_memory_goal(goal)
        .build()?
        .with_execution_limits(limits)?;
    let uqff = env::var_os("MISTRALRS_MEMORY_UQFF");
    let isq_bits = env::var("MISTRALRS_MEMORY_ISQ_BITS").ok();
    ensure!(
        uqff.is_none() || isq_bits.is_none(),
        "choose UQFF or ISQ, not both"
    );
    let kind = env::var("MISTRALRS_MEMORY_UQFF_KIND").unwrap_or_else(|_| "text".into());
    let builder = if let Some(uqff) = uqff {
        let files: Vec<_> = env::split_paths(&uqff).collect();
        ensure!(!files.is_empty(), "UQFF file list is empty");
        for file in &files {
            ensure!(
                file.is_file() || PathBuf::from(&model).join(file).is_file(),
                "UQFF file is not local: {}",
                file.display()
            );
        }
        match kind.as_str() {
            "text" => UqffTextModelBuilder::new(&model, files)
                .into_inner()
                .with_device(device.clone())
                .with_dtype(ModelDType::BF16)
                .with_max_num_seqs(max_num_seqs)
                .with_paged_attn(paged.clone())
                .with_prefix_cache_n(Some(PREFIX_CACHE_SEQUENCES))
                .with_logging()
                .into(),
            "multimodal" => UqffMultimodalModelBuilder::new(&model, files)
                .into_inner()
                .with_device(device.clone())
                .with_dtype(ModelDType::BF16)
                .with_max_num_seqs(max_num_seqs)
                .with_paged_attn(paged.clone())
                .with_prefix_cache_n(Some(PREFIX_CACHE_SEQUENCES))
                .with_logging()
                .into(),
            _ => anyhow::bail!("MISTRALRS_MEMORY_UQFF_KIND must be text or multimodal"),
        }
    } else {
        let mut builder = ModelBuilder::new(&model)
            .with_device(device.clone())
            .with_dtype(ModelDType::BF16)
            .with_max_num_seqs(max_num_seqs)
            .with_paged_attn(paged)
            .with_prefix_cache_n(Some(PREFIX_CACHE_SEQUENCES))
            .with_logging();
        if let Some(bits) = isq_bits.as_deref() {
            builder = builder.with_auto_isq(IsqBits::try_from(bits).map_err(|_| {
                anyhow::anyhow!("MISTRALRS_MEMORY_ISQ_BITS must be 2, 3, 4, 5, 6 or 8")
            })?);
        }
        builder.into()
    };
    println!(
        "{}",
        json!({
            "phase": "source",
            "model": model,
            "uqff": env::var_os("MISTRALRS_MEMORY_UQFF").map(|p| p.to_string_lossy().into_owned()),
            "uqff_kind": kind,
            "isq_bits": isq_bits,
            "activation_dtype": "bf16",
            "memory_goal": goal,
            "max_num_seqs": max_num_seqs,
            "cohort_admission": COHORT_ADMISSION,
            "requested_sample_interval_us": SAMPLE_INTERVAL.as_micros(),
            "measurement": "sampled Metal currentAllocatedSize, not an allocator high-water mark or process RSS",
        })
    );
    Ok(builder)
}

fn prompt(repetitions: usize, row: usize) -> String {
    format!(
        "Study {repetitions}, record {row}.\n{}\nExplain what the observatory measures in a short paragraph.",
        PROMPT_SENTENCE.repeat(repetitions)
    )
}

fn request(prompt: &str) -> RequestBuilder {
    RequestBuilder::new()
        .add_message(TextMessageRole::User, prompt)
        .enable_thinking(false)
        .set_sampler_temperature(0.0)
        .set_sampler_max_len(OUTPUT_TOKENS)
        .set_sampler_ignore_eos(true)
        .return_logprobs(true)
}

async fn run_cohort(
    model: &Model,
    prompts: &[String],
    device: &Device,
    synchronized: bool,
) -> Result<(Vec<ChatCompletionResponse>, Observation)> {
    let gate = CohortGate::new();
    let callback_gate = gate.0.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let ready_tx = Mutex::new(Some(ready_tx));
    let processor = Arc::new(move |logits: &candle_core::Tensor, _context: &[u32]| {
        if let Some(ready) = ready_tx.lock().unwrap().take() {
            let _ = ready.send(());
        }
        let (released, wake) = &*callback_gate;
        let (released, _) = wake
            .wait_timeout_while(released.lock().unwrap(), REQUEST_TIMEOUT, |released| {
                !*released
            })
            .unwrap();
        if !*released {
            candle_core::bail!("cohort admission gate timed out");
        }
        Ok(logits.clone())
    });
    let mut sentinel = Box::pin(
        model.send_chat_request(
            request(COHORT_SENTINEL_PROMPT)
                .set_sampler_max_len(COHORT_SENTINEL_TOKENS)
                .return_logprobs(false)
                .add_logits_processor(processor),
        ),
    );
    tokio::select! {
        response = &mut sentinel => {
            response?;
            anyhow::bail!("cohort sentinel completed before its sampling gate");
        }
        ready = tokio::time::timeout(REQUEST_TIMEOUT, ready_rx) => {
            ready.context("cohort sentinel did not reach sampling")?
                .context("cohort sentinel signal closed")?;
        }
    }
    let sampler = Sampler::start(device.as_metal_device()?);
    let sender = model.inner().get_sender(None)?;
    let capacity = sender.capacity();
    ensure!(
        capacity >= prompts.len(),
        "cohort does not fit the request queue"
    );
    let mut cohort = Box::pin(join_all(
        prompts
            .iter()
            .map(|prompt| model.send_chat_request(request(prompt))),
    ));
    // Sampling blocks the engine after collection, so these sends all precede its next queue snapshot.
    ensure!(
        futures::poll!(cohort.as_mut()).is_pending(),
        "cohort completed before admission"
    );
    ensure!(
        sender.capacity() == capacity - prompts.len(),
        "cohort requests were not all queued behind the sampling gate"
    );
    gate.release();
    let responses = tokio::time::timeout(REQUEST_TIMEOUT, async {
        let (sentinel, responses) = tokio::join!(sentinel, cohort);
        let sentinel = sentinel?;
        ensure!(
            sentinel.usage.completion_tokens == COHORT_SENTINEL_TOKENS,
            "cohort sentinel generated an unexpected token count"
        );
        responses
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::from)
    })
    .await;
    if synchronized {
        device.synchronize()?;
    }
    let observation = sampler.finish()?;
    Ok((responses.context("inference timed out")??, observation))
}

fn validate_response(response: &ChatCompletionResponse) -> Result<usize> {
    ensure!(response.choices.len() == 1, "expected one completion");
    let choice = &response.choices[0];
    ensure!(
        choice.finish_reason == "length",
        "unexpected finish: {}",
        choice.finish_reason
    );
    ensure!(
        response.usage.completion_tokens == OUTPUT_TOKENS,
        "output token count mismatch"
    );
    ensure!(response.usage.prompt_tokens > 0, "empty prompt token usage");
    ensure!(
        response.usage.total_tokens == response.usage.prompt_tokens + OUTPUT_TOKENS,
        "total token count mismatch"
    );
    ensure!(
        choice
            .message
            .content
            .as_ref()
            .is_some_and(|text| !text.is_empty())
            || choice
                .message
                .reasoning_content
                .as_ref()
                .is_some_and(|text| !text.is_empty()),
        "empty generated text"
    );
    let logprobs = choice
        .logprobs
        .as_ref()
        .and_then(|p| p.content.as_ref())
        .context("missing chosen-token logprobs")?;
    ensure!(logprobs.len() == OUTPUT_TOKENS, "logprob count mismatch");
    ensure!(
        logprobs.iter().all(|p| p.logprob.is_finite()),
        "nonfinite chosen-token logprob"
    );
    let cached = response
        .usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |p| p.cached_tokens);
    ensure!(
        cached <= response.usage.prompt_tokens,
        "cached tokens exceed prompt tokens"
    );
    Ok(cached)
}

struct Case {
    name: &'static str,
    prompts: Vec<String>,
    prompt_tokens: Vec<usize>,
    require_cached: bool,
}

async fn cases(model: &Model) -> Result<Vec<Case>> {
    let mut cases = vec![
        Case {
            name: "short_b1",
            prompts: vec![prompt(SHORT_REPETITIONS, 0)],
            prompt_tokens: vec![],
            require_cached: false,
        },
        Case {
            name: "long_b1",
            prompts: vec![prompt(LONG_SINGLE_REPETITIONS, 0)],
            prompt_tokens: vec![],
            require_cached: false,
        },
        Case {
            name: "short_b8",
            prompts: (0..BATCH_SIZE)
                .map(|i| prompt(SHORT_REPETITIONS, i))
                .collect(),
            prompt_tokens: vec![],
            require_cached: false,
        },
        Case {
            name: "long_b8",
            prompts: (0..BATCH_SIZE)
                .map(|i| prompt(LONG_REPETITIONS, i))
                .collect(),
            prompt_tokens: vec![],
            require_cached: false,
        },
    ];
    let mixed: Vec<_> = (0..BATCH_SIZE)
        .map(|i| {
            prompt(
                if i % 2 == 0 {
                    SHORT_REPETITIONS
                } else {
                    LONG_REPETITIONS
                },
                i,
            )
        })
        .collect();
    cases.push(Case {
        name: "mixed_b8",
        prompts: mixed.clone(),
        prompt_tokens: vec![],
        require_cached: false,
    });
    cases.push(Case {
        name: "mixed_b8_repeat",
        prompts: mixed,
        prompt_tokens: vec![],
        require_cached: true,
    });
    for case in &mut cases {
        for prompt in &case.prompts {
            let messages = TextMessages::new()
                .add_message(TextMessageRole::User, prompt)
                .enable_thinking(false);
            let tokens = model
                .tokenize(
                    either::Either::Left(messages),
                    None,
                    false,
                    true,
                    Some(false),
                )
                .await?;
            case.prompt_tokens.push(tokens.len());
        }
    }
    Ok(cases)
}

fn write_report(path: Option<&std::path::Path>, report: &Value) -> Result<()> {
    if let Some(path) = path {
        std::fs::write(path, serde_json::to_vec_pretty(report)?)?;
    }
    Ok(())
}

fn compare_case(record: &Value, reference: &Value) -> Result<Value> {
    let reference = reference["cases"]
        .as_array()
        .context("missing reference cases")?
        .iter()
        .find(|case| case["case"] == record["case"] && case["pass"] == record["pass"])
        .context("missing matched reference case")?;
    ensure!(
        record["prompts"] == reference["prompts"],
        "reference prompts differ"
    );
    let actual = record["responses"]
        .as_array()
        .context("missing responses")?;
    let expected = reference["responses"]
        .as_array()
        .context("missing reference responses")?;
    ensure!(
        actual.len() == expected.len(),
        "reference batch size differs"
    );
    let mut comparisons = Vec::new();
    let mut matches = true;
    for (actual, expected) in actual.iter().zip(expected) {
        ensure!(
            actual["usage"]["prompt_tokens"] == expected["usage"]["prompt_tokens"],
            "reference prompt tokens differ"
        );
        let actual_choice = &actual["choices"][0];
        let expected_choice = &expected["choices"][0];
        let actual_tokens = actual_choice["logprobs"]["content"]
            .as_array()
            .context("missing logprobs")?;
        let expected_tokens = expected_choice["logprobs"]["content"]
            .as_array()
            .context("missing reference logprobs")?;
        ensure!(
            actual_tokens.len() == expected_tokens.len(),
            "reference output token count differs"
        );
        let mut aligned_tokens = 0;
        let mut max_logprob_delta = 0.0f64;
        let mut first_difference = Value::Null;
        for (index, (actual, expected)) in actual_tokens.iter().zip(expected_tokens).enumerate() {
            if actual["token"] != expected["token"] {
                first_difference = json!({"index": index, "actual": actual["token"], "reference": expected["token"]});
                break;
            }
            aligned_tokens += 1;
            max_logprob_delta = max_logprob_delta.max(
                (actual["logprob"].as_f64().context("missing logprob")?
                    - expected["logprob"]
                        .as_f64()
                        .context("missing reference logprob")?)
                .abs(),
            );
        }
        let exact_text = actual_choice["message"] == expected_choice["message"];
        matches &= aligned_tokens == actual_tokens.len()
            && exact_text
            && max_logprob_delta <= BF16_LOGPROB_TOLERANCE;
        comparisons.push(json!({"aligned_tokens": aligned_tokens, "exact_text": exact_text,
            "max_aligned_logprob_delta": max_logprob_delta, "first_token_difference": first_difference}));
    }
    Ok(
        json!({"matches": matches, "logprob_tolerance": BF16_LOGPROB_TOLERANCE, "responses": comparisons}),
    )
}

#[tokio::test]
#[ignore = "requires a local model and exclusive Metal use; set MISTRALRS_MEMORY_MODEL and optional MISTRALRS_MEMORY_UQFF"]
async fn real_model_observed_memory_budget() -> Result<()> {
    let max_num_seqs = max_sequences()?;
    let goal_string = env::var("MISTRALRS_MEMORY_GOAL").unwrap_or_else(|_| "auto".into());
    let goal = goal_string.parse::<MemoryGoal>().or_else(|_| {
        goal_string
            .parse::<f32>()
            .map(MemoryGoal::Utilization)
            .map_err(anyhow::Error::from)
    })?;
    goal.validate()?;
    let report_path = env::var_os("MISTRALRS_MEMORY_REPORT").map(PathBuf::from);
    let reference: Option<Value> = env::var_os("MISTRALRS_MEMORY_REFERENCE")
        .map(|path| -> Result<_> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) })
        .transpose()?;
    let source = json!({
        "model": env::var("MISTRALRS_MEMORY_MODEL")?,
        "uqff": env::var("MISTRALRS_MEMORY_UQFF").ok(),
        "uqff_kind": env::var("MISTRALRS_MEMORY_UQFF_KIND").unwrap_or_else(|_| "text".into()),
        "isq_bits": env::var("MISTRALRS_MEMORY_ISQ_BITS").ok(),
        "memory_goal": goal,
        "max_num_seqs": max_num_seqs,
        "cohort_admission": COHORT_ADMISSION,
        "activation_dtype": "bf16",
    });
    if let Some(reference) = &reference {
        ensure!(
            reference["complete"] == true,
            "reference run did not complete"
        );
        ensure!(
            reference["source"] == source,
            "reference model/quantization/goal differs"
        );
    }
    let mut report = json!({"source": source, "complete": false, "cases": []});
    let mut comparison_failures = Vec::new();
    let device = Device::new_metal(0)?;
    let metal = device.as_metal_device()?;
    let budget = resolve_device_budget(goal, MemoryUsage.query(&device)?)?;
    let load_sampler = Sampler::start(metal);
    let (pipeline, scheduler, config) = builder(&device, goal, max_num_seqs)?
        .build_pipeline()
        .await?;
    let (plan, cache, logical_kv_bytes) = {
        let pipeline = pipeline.lock().await;
        let metadata = pipeline.get_metadata();
        let plan = metadata
            .memory_plan
            .context("model has no declared runtime memory plan")?;
        let cache = metadata
            .cache_config
            .clone()
            .context("model did not enable paging")?;
        let kv = metadata
            .cache_engine
            .as_ref()
            .context("model has no KV cache")?
            .get_kv_cache();
        let logical_kv_bytes: usize = kv
            .iter()
            .map(|(k, v)| {
                k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()
            })
            .sum();
        (plan, cache, logical_kv_bytes)
    };
    let model = build_model_from_pipeline(pipeline, scheduler, config).await;
    device.synchronize()?;
    let load = load_sampler.finish()?;
    let post_load_bytes = metal.current_allocated_size();
    let declared_reserve_bytes = plan
        .primary_activation_bytes
        .max(plan.mapped_activation_bytes);
    ensure!(
        declared_reserve_bytes > 0,
        "model declares no forward activation reserve"
    );
    let predicted_forward_limit = post_load_bytes
        .checked_add(declared_reserve_bytes)
        .context("predicted allocation overflow")?;
    println!(
        "{}",
        json!({
            "phase": "load",
            "allocation": load,
            "post_load_bytes": post_load_bytes,
            "logical_kv_bytes": logical_kv_bytes,
            "kv_blocks": cache.num_gpu_blocks,
            "kv_block_tokens": cache.block_size,
            "declared_reserve_bytes": declared_reserve_bytes,
            "primary_non_attention_bytes": plan.primary_non_attention_bytes,
            "mapped_non_attention_bytes": plan.mapped_non_attention_bytes,
            "planned_query_tokens": plan.planned_query_tokens,
            "max_num_seqs": plan.limits.max_num_seqs,
            "max_num_batched_tokens": plan.limits.max_num_batched_tokens,
            "max_prefill_chunk_tokens": plan.limits.max_prefill_chunk_tokens,
            "predicted_forward_limit_bytes": predicted_forward_limit,
            "device_engine_limit_bytes": budget.engine_limit_bytes,
            "initial_allocatable_bytes": budget.allocatable_bytes,
            "load_peak_is_not_a_forward_budget_assertion": true,
        })
    );
    ensure!(
        post_load_bytes <= budget.engine_limit_bytes,
        "post-load allocation exceeds device goal"
    );
    report["limits"] = json!({
        "max_num_seqs": plan.limits.max_num_seqs,
        "planned_query_tokens": plan.planned_query_tokens,
        "max_num_batched_tokens": plan.limits.max_num_batched_tokens,
        "max_prefill_chunk_tokens": plan.limits.max_prefill_chunk_tokens,
        "predicted_forward_limit_bytes": predicted_forward_limit,
        "device_engine_limit_bytes": budget.engine_limit_bytes,
    });
    let cases = cases(&model).await?;
    let mut overall_peak = post_load_bytes;
    let mut completions = 0;
    let mut cached_requests = 0;
    for pass in 0..PASS_COUNT {
        for case in &cases {
            let synchronized = pass == 0;
            if synchronized {
                device.synchronize()?;
            }
            let (responses, observation) =
                run_cohort(&model, &case.prompts, &device, synchronized).await?;
            overall_peak = overall_peak.max(observation.observed_peak_bytes);
            println!(
                "{}",
                json!({
                    "phase": "forward_allocation", "case": case.name, "pass": pass,
                    "synchronized_boundaries": synchronized, "allocation": observation,
                    "prompt_tokens": case.prompt_tokens,
                    "predicted_forward_limit_bytes": predicted_forward_limit,
                    "device_engine_limit_bytes": budget.engine_limit_bytes,
                })
            );
            let mut usages = Vec::with_capacity(responses.len());
            let mut uncached_prompt_tokens = 0;
            for (response, expected_prompt_tokens) in responses.iter().zip(&case.prompt_tokens) {
                let cached = validate_response(response)?;
                ensure!(
                    response.usage.prompt_tokens == *expected_prompt_tokens,
                    "tokenization and usage disagree"
                );
                if case.require_cached {
                    ensure!(
                        cached > 0,
                        "repeated prompt did not reuse its cached prefix"
                    );
                }
                uncached_prompt_tokens += response.usage.prompt_tokens - cached;
                cached_requests += usize::from(cached > 0);
                completions += 1;
                usages.push(&response.usage);
            }
            println!(
                "{}",
                json!({
                    "phase": "forward", "case": case.name, "pass": pass,
                    "synchronized_boundaries": synchronized, "usage": usages,
                    "uncached_prompt_tokens": uncached_prompt_tokens,
                    "predicted_forward_limit_bytes": predicted_forward_limit,
                    "device_engine_limit_bytes": budget.engine_limit_bytes,
                })
            );
            let mut record = json!({
                "case": case.name, "pass": pass, "prompts": case.prompts,
                "allocation": observation, "uncached_prompt_tokens": uncached_prompt_tokens,
                "responses": responses,
            });
            if let Some(reference) = &reference {
                let comparison = compare_case(&record, reference)?;
                if comparison["matches"] != true {
                    comparison_failures.push(format!("{} pass {pass}: {comparison}", case.name));
                }
                println!(
                    "{}",
                    json!({"phase": "equivalence", "case": case.name, "pass": pass, "comparison": comparison})
                );
                record["comparison"] = comparison;
            }
            report["cases"].as_array_mut().unwrap().push(record);
            write_report(report_path.as_deref(), &report)?;
            if pass == 0
                && case.name == "long_b8"
                && case.prompt_tokens.iter().sum::<usize>() > plan.planned_query_tokens
            {
                ensure!(uncached_prompt_tokens > plan.planned_query_tokens,
                    "long_b8 no longer exceeds the per-step query-token budget after prefix reuse: {uncached_prompt_tokens} uncached tokens");
            }
            ensure!(
                observation.observed_peak_bytes <= predicted_forward_limit,
                "{} pass {pass}: observed {} bytes exceeds post-load {} + declared reserve {}",
                case.name,
                observation.observed_peak_bytes,
                post_load_bytes,
                declared_reserve_bytes
            );
            ensure!(
                observation.observed_peak_bytes <= budget.engine_limit_bytes,
                "{} pass {pass}: observed allocation exceeds device memory goal",
                case.name
            );
        }
    }
    println!(
        "{}",
        json!({
            "phase": "summary", "completions": completions,
            "chosen_token_logprobs_checked": completions * OUTPUT_TOKENS,
            "cached_requests": cached_requests, "observed_forward_peak_bytes": overall_peak,
            "predicted_forward_limit_bytes": predicted_forward_limit,
            "device_engine_limit_bytes": budget.engine_limit_bytes,
            "limitation": "sampling can miss transient peaks; passing checks observed allocation only",
        })
    );
    report["complete"] = json!(true);
    report["comparison_failures"] = json!(comparison_failures);
    write_report(report_path.as_deref(), &report)?;
    ensure!(
        comparison_failures.is_empty(),
        "chunked/reference differences: {}",
        comparison_failures.join("; ")
    );
    Ok(())
}
