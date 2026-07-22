//! Performance benchmarking command

use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use mistralrs_core::{
    initialize_logging, AdapterSelection, Constraint, NormalRequest, Request, RequestMessage,
    Response, SamplingParams,
};
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::mpsc::channel;
use tracing::info;

use crate::args::{BenchRuntimeOptions, GlobalOptions, ModelType};

use super::normalize_requested_adapter;
use super::serve::{
    apply_quant_resolution, convert_to_model_selected, extract_device_settings,
    extract_isq_setting, extract_paged_attn_settings,
};

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

/// Benchmark result for a single test
struct BenchResult {
    test_name: String,
    tok_per_sec: f32,
    std_dev: f32,
    latency_ms: f32,
    latency_kind: BenchLatencyKind,
}

#[derive(Clone, Copy)]
enum BenchLatencyKind {
    Ttft,
    Tpot,
}

struct BenchMeasurement {
    time_to_first_token: Duration,
    decode_duration: Duration,
    decode_intervals: usize,
}

const BENCH_TOKEN_BASE: u32 = 1000;
const BENCH_TOKEN_SPAN: u32 = 2048;
const BENCH_ITER_STRIDE: u32 = 131;
const BENCH_CASE_STRIDE: u32 = 719;

pub struct BenchRunConfig {
    pub prompt_lens: Vec<usize>,
    pub gen_len: usize,
    pub depths: Vec<usize>,
    pub iterations: usize,
    pub warmup: usize,
    pub adapter: Option<String>,
}

/// Extract model_id from ModelType
fn get_model_id(model_type: &ModelType) -> String {
    match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Multimodal { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model.model_id.clone(),
    }
}

/// Run the benchmark command
pub async fn run_bench(
    mut model_type: ModelType,
    runtime: BenchRuntimeOptions,
    global: GlobalOptions,
    config: BenchRunConfig,
) -> Result<()> {
    initialize_logging();

    let BenchRunConfig {
        prompt_lens,
        gen_len,
        depths,
        iterations,
        warmup,
        adapter: request_adapter,
    } = config;

    if prompt_lens.is_empty() {
        anyhow::bail!("--prompt-len must contain at least one value");
    }
    if depths.is_empty() {
        anyhow::bail!("--depth must contain at least one value");
    }
    if iterations == 0 {
        anyhow::bail!("--iterations must be greater than 0");
    }
    if prompt_lens.iter().all(|prompt_len| *prompt_len == 0) && gen_len <= 1 {
        anyhow::bail!("benchmark must enable at least one TTFT or decode measurement");
    }
    if gen_len > 1 && depths.contains(&0) {
        anyhow::bail!("--depth must be greater than 0 when decode metrics are enabled");
    }
    let request_adapter = normalize_requested_adapter(&model_type, request_adapter.as_deref())?;

    // Get model ID for display
    let model_id = get_model_id(&model_type);
    // Convert args and load model
    let matformer = runtime.matformer_selection();
    apply_quant_resolution(&mut model_type, &global.token_source, &matformer).await?;
    let model_selected = convert_to_model_selected(&model_type, &matformer)?;

    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = extract_paged_attn_settings(&model_type);

    let (cpu, device_layers) = extract_device_settings(&model_type);
    let isq = extract_isq_setting(&model_type);

    info!("Loading model for benchmarking...");

    // Build using the same infrastructure as serve
    let builder = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(1) // Single sequence for benchmarking
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(false)
        .with_prefix_cache_n(0) // Disable prefix cache for benchmarking
        .with_disable_eos_stop(true) // Always generate exactly gen_len tokens
        .with_mtp_config_optional(runtime.mtp_config())
        .set_paged_attn(paged_attn)
        .with_cpu(cpu)
        .with_seed_optional(global.seed)
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(isq)
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_paged_attn_cache_type(paged_cache_type);

    let mistralrs = builder.build().await?;
    if let Some(alias) = request_adapter.as_deref() {
        let adapters = mistralrs.list_lora_adapters(None).await?;
        if !adapters.iter().any(|adapter| adapter.alias == alias) {
            anyhow::bail!("LoRA adapter alias `{alias}` is not loaded");
        }
    }
    if let Some(max_seq_len) = mistralrs
        .config(None)
        .map_err(anyhow::Error::msg)?
        .max_seq_len
    {
        let longest_ttft = prompt_lens
            .iter()
            .copied()
            .map(|prompt_len| prompt_len.saturating_add(1))
            .max()
            .unwrap_or_default();
        let longest_decode = if gen_len > 1 {
            depths
                .iter()
                .copied()
                .map(|depth| depth.saturating_add(gen_len))
                .max()
                .unwrap_or_default()
        } else {
            0
        };
        let longest_request = longest_ttft.max(longest_decode);
        if longest_request > max_seq_len {
            anyhow::bail!(
                "benchmark request length {longest_request} exceeds model maximum {max_seq_len}"
            );
        }
    }
    info!("Model loaded.");

    if warmup > 0 {
        info!("Running {warmup} warmup iteration(s) per benchmark case...");
        for (prompt_idx, prompt_len) in prompt_lens.iter().copied().enumerate() {
            if prompt_len == 0 {
                continue;
            }
            for i in 0..warmup {
                let token_start = bench_token_start(i, prompt_idx, 0);
                run_single_bench(
                    &mistralrs,
                    prompt_len,
                    1,
                    token_start,
                    request_adapter.clone(),
                )
                .await?;
            }
        }
        if gen_len > 1 {
            for (depth_idx, depth) in depths.iter().copied().enumerate() {
                for i in 0..warmup {
                    let token_start = bench_token_start(i, depth_idx, prompt_lens.len());
                    run_single_bench(
                        &mistralrs,
                        depth,
                        gen_len,
                        token_start,
                        request_adapter.clone(),
                    )
                    .await?;
                }
            }
        }
        info!("Warmup complete.");

        // Reset logger counters so benchmark stats are clean
        if let Ok(logger) = mistralrs.get_logger(None) {
            logger.reset();
        }
    }

    // Run benchmarks
    info!(
        "Running {} iteration(s) with prompt lengths {:?}, {} generation tokens, decode depths {:?}...",
        iterations, prompt_lens, gen_len, depths
    );

    #[cfg(feature = "cuda")]
    let cuda_profiler_range = std::env::var_os("MISTRALRS_BENCH_CUDA_PROFILER_RANGE").is_some();
    #[cfg(feature = "cuda")]
    if cuda_profiler_range {
        unsafe {
            let _ = cudaProfilerStart();
        }
    }

    let mut ttft_results: Vec<(usize, Vec<(f32, f32)>)> =
        prompt_lens.iter().map(|&len| (len, Vec::new())).collect();
    let mut decode_results: Vec<(usize, Vec<(f32, f32)>)> =
        depths.iter().map(|&depth| (depth, Vec::new())).collect();

    for i in 0..iterations {
        info!("Iteration {}/{}...", i + 1, iterations);

        for (prompt_idx, (prompt_len, results)) in ttft_results.iter_mut().enumerate() {
            if *prompt_len == 0 {
                continue;
            }
            let token_start = bench_token_start(i + warmup, prompt_idx, 0);
            let measurement = run_single_bench(
                &mistralrs,
                *prompt_len,
                1,
                token_start,
                request_adapter.clone(),
            )
            .await?;
            let ttft_seconds = measurement.time_to_first_token.as_secs_f32();
            let tok_per_sec = if ttft_seconds > 0.0 {
                *prompt_len as f32 / ttft_seconds
            } else {
                0.0
            };
            results.push((
                tok_per_sec,
                measurement.time_to_first_token.as_secs_f32() * 1000.0,
            ));
        }

        if gen_len > 1 {
            for (depth_idx, (depth, results)) in decode_results.iter_mut().enumerate() {
                let token_start = bench_token_start(i + warmup, depth_idx, prompt_lens.len());
                let measurement = run_single_bench(
                    &mistralrs,
                    *depth,
                    gen_len,
                    token_start,
                    request_adapter.clone(),
                )
                .await?;
                let decode_seconds = measurement.decode_duration.as_secs_f32();
                let tok_per_sec = if decode_seconds > 0.0 {
                    measurement.decode_intervals as f32 / decode_seconds
                } else {
                    0.0
                };
                let ms_per_tok = if tok_per_sec > 0.0 {
                    1000.0 / tok_per_sec
                } else {
                    0.0
                };
                results.push((tok_per_sec, ms_per_tok));
            }
        }
    }

    #[cfg(feature = "cuda")]
    if cuda_profiler_range {
        unsafe {
            let _ = cudaProfilerStop();
        }
    }

    // Calculate statistics
    let mut results = Vec::new();

    for (prompt_len, ttft_result) in ttft_results {
        if ttft_result.is_empty() {
            continue;
        }
        let tok_per_sec_vals: Vec<f32> = ttft_result.iter().map(|(t, _)| *t).collect();
        let ttft_vals: Vec<f32> = ttft_result.iter().map(|(_, l)| *l).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let (mean_ttft, _) = calculate_stats(&ttft_vals);
        results.push(BenchResult {
            test_name: format!("TTFT ({} input tokens)", prompt_len),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: mean_ttft,
            latency_kind: BenchLatencyKind::Ttft,
        });
    }

    for (depth, decode_result) in decode_results {
        if decode_result.is_empty() {
            continue;
        }
        let tok_per_sec_vals: Vec<f32> = decode_result.iter().map(|(t, _)| *t).collect();
        let tpot_vals: Vec<f32> = decode_result.iter().map(|(_, l)| *l).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let (mean_tpot, _) = calculate_stats(&tpot_vals);
        results.push(BenchResult {
            test_name: format!("Decode ({} tokens @ d{})", gen_len, depth),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: mean_tpot,
            latency_kind: BenchLatencyKind::Tpot,
        });
    }

    // Print results
    print_results(&model_id, request_adapter.as_deref(), iterations, &results);

    Ok(())
}

/// Calculate mean and standard deviation
fn calculate_stats(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    (mean, std_dev)
}

fn bench_token_start(iteration: usize, case_idx: usize, group_offset: usize) -> u32 {
    ((iteration + 1) as u32 * BENCH_ITER_STRIDE
        + (case_idx + group_offset) as u32 * BENCH_CASE_STRIDE)
        % BENCH_TOKEN_SPAN
}

async fn run_single_bench(
    mistralrs: &Arc<mistralrs_core::MistralRs>,
    prompt_tokens: usize,
    gen_tokens: usize,
    token_start: u32,
    adapter: Option<String>,
) -> Result<BenchMeasurement> {
    let mut sampling_params = SamplingParams::deterministic();
    sampling_params.max_len = Some(gen_tokens);

    let sender = mistralrs.get_sender(None).unwrap();
    let (tx, mut rx) = channel(100);

    let tokens = bench_tokens(prompt_tokens, token_start);

    let req = Request::Normal(Box::new(NormalRequest {
        id: mistralrs.next_request_id(),
        messages: RequestMessage::CompletionTokens(tokens),
        sampling_params,
        response: tx,
        return_logprobs: false,
        is_streaming: true,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        enable_code_execution: false,
        enable_shell: false,
        shell_options: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        session_id: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: None,
        adapter: adapter.map(AdapterSelection::alias),
        truncate_sequence: false,
        files: None,
        input_files: Vec::new(),
    }));

    let request_start = Instant::now();
    sender.send(req).await?;

    recv_measurement(&mut rx, request_start, gen_tokens).await
}

fn bench_tokens(prompt_tokens: usize, token_start: u32) -> Vec<u32> {
    (0..prompt_tokens)
        .map(|idx| BENCH_TOKEN_BASE + (token_start + idx as u32) % BENCH_TOKEN_SPAN)
        .collect()
}

async fn recv_measurement(
    rx: &mut tokio::sync::mpsc::Receiver<Response>,
    request_start: Instant,
    expected_tokens: usize,
) -> Result<BenchMeasurement> {
    let mut first_token = None;

    let last_token = loop {
        match rx.recv().await {
            Some(Response::AgenticToolCallProgress { .. }) => continue,
            Some(Response::BlockDenoisingProgress(_)) => continue,
            Some(Response::File(_)) => continue,
            Some(Response::CompletionChunk(response)) => {
                let received = Instant::now();
                let finished = response
                    .choices
                    .iter()
                    .any(|choice| choice.finish_reason.is_some());
                if !response.choices.is_empty() {
                    first_token.get_or_insert(received);
                }
                if finished {
                    break received;
                }
            }
            Some(Response::InternalError(e)) => anyhow::bail!("Internal error: {e:?}"),
            Some(Response::ModelError(e, _)) => anyhow::bail!("Model error: {e}"),
            Some(Response::CompletionModelError(e, _)) => anyhow::bail!("Model error: {e}"),
            Some(Response::ValidationError(e)) => anyhow::bail!("Validation error: {e:?}"),
            Some(_) => anyhow::bail!("Unexpected response type"),
            None => anyhow::bail!("No response received"),
        }
    };

    let first_token = first_token.expect("finished response must contain a token");
    Ok(BenchMeasurement {
        time_to_first_token: first_token.duration_since(request_start),
        decode_duration: last_token.duration_since(first_token),
        decode_intervals: expected_tokens.saturating_sub(1),
    })
}

/// Print benchmark results in a nice table
#[allow(clippy::cast_precision_loss)]
fn print_results(
    model_id: &str,
    adapter: Option<&str>,
    iterations: usize,
    results: &[BenchResult],
) {
    println!();
    println!("Benchmark Results");
    println!("=================");
    println!();
    println!("Model: {}", model_id);
    println!("Adapter: {}", adapter.unwrap_or("base"));
    println!("Iterations: {}", iterations);
    println!();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Test"),
            Cell::new("T/s"),
            Cell::new("Latency"),
        ]);

    for result in results {
        let latency_str = match result.latency_kind {
            BenchLatencyKind::Ttft => format!("{:.2} ms TTFT", result.latency_ms),
            BenchLatencyKind::Tpot => format!("{:.2} ms TPOT", result.latency_ms),
        };

        table.add_row(vec![
            Cell::new(&result.test_name),
            Cell::new(format!("{:.1} ± {:.1}", result.tok_per_sec, result.std_dev))
                .fg(Color::Green),
            Cell::new(latency_str),
        ]);
    }

    println!("{table}");
    println!();
}
