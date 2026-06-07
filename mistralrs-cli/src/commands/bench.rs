//! Performance benchmarking command

use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use mistralrs_core::{
    initialize_logging, Constraint, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, Usage,
};
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
use std::sync::Arc;
use tokio::sync::mpsc::channel;
use tracing::info;

use crate::args::{BenchRuntimeOptions, GlobalOptions, ModelType};

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
    /// For prefill: model prefill time in ms; for decode: ms/tok
    latency_ms: f32,
}

pub struct BenchRunConfig {
    pub prompt_lens: Vec<usize>,
    pub gen_len: usize,
    pub depths: Vec<usize>,
    pub iterations: usize,
    pub warmup: usize,
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
    } = config;

    if prompt_lens.is_empty() {
        anyhow::bail!("--prompt-len must contain at least one value");
    }
    if depths.is_empty() {
        anyhow::bail!("--depth must contain at least one value");
    }
    if gen_len > 0 && depths.contains(&0) {
        anyhow::bail!("--depth must be greater than 0 when --gen-len is greater than 0");
    }

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
    info!("Model loaded.");

    // Warmup runs
    if warmup > 0 {
        info!("Running {} warmup iteration(s)...", warmup);
        for _ in 0..warmup {
            run_single_bench(&mistralrs, 32, 16).await?;
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

    let mut prefill_results: Vec<(usize, Vec<(f32, f32)>)> =
        prompt_lens.iter().map(|&len| (len, Vec::new())).collect();
    let mut decode_results: Vec<(usize, Vec<(f32, f32)>)> =
        depths.iter().map(|&depth| (depth, Vec::new())).collect();

    for i in 0..iterations {
        info!("Iteration {}/{}...", i + 1, iterations);

        for (prompt_len, results) in prefill_results.iter_mut() {
            if *prompt_len == 0 {
                continue;
            }
            let usage = run_single_bench(&mistralrs, *prompt_len, 1).await?;
            let tok_per_sec = usage.avg_prompt_tok_per_sec;
            let prefill_ms = usage.total_prompt_time_sec * 1000.0;
            results.push((tok_per_sec, prefill_ms));
        }

        if gen_len > 0 {
            for (depth, results) in decode_results.iter_mut() {
                let usage = run_single_bench(&mistralrs, *depth, gen_len).await?;
                let tok_per_sec = usage.avg_compl_tok_per_sec;
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

    for (prompt_len, prefill_result) in prefill_results {
        if prefill_result.is_empty() {
            continue;
        }
        let tok_per_sec_vals: Vec<f32> = prefill_result.iter().map(|(t, _)| *t).collect();
        let prefill_time_vals: Vec<f32> = prefill_result.iter().map(|(_, l)| *l).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let (mean_prefill_time, _) = calculate_stats(&prefill_time_vals);
        results.push(BenchResult {
            test_name: format!("Prefill ({} tokens)", prompt_len),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: mean_prefill_time,
        });
    }

    for (depth, decode_result) in decode_results {
        if decode_result.is_empty() {
            continue;
        }
        let tok_per_sec_vals: Vec<f32> = decode_result.iter().map(|(t, _)| *t).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let ms_per_tok = 1000.0 / mean_tps;
        results.push(BenchResult {
            test_name: format!("Decode ({} tokens @ d{})", gen_len, depth),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: ms_per_tok, // ms/tok
        });
    }

    // Print results
    print_results(&model_id, iterations, &results);

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

/// Run a single benchmark iteration.
async fn run_single_bench(
    mistralrs: &Arc<mistralrs_core::MistralRs>,
    prompt_tokens: usize,
    gen_tokens: usize,
) -> Result<Usage> {
    let mut sampling_params = SamplingParams::deterministic();
    sampling_params.max_len = Some(gen_tokens);

    let sender = mistralrs.get_sender(None).unwrap();
    let (tx, mut rx) = channel(100);

    // Use token IDs for prompt to ensure exact length
    let tokens: Vec<u32> = (1000..1000 + prompt_tokens as u32).collect();

    let req = Request::Normal(Box::new(NormalRequest {
        id: mistralrs.next_request_id(),
        messages: RequestMessage::CompletionTokens(tokens),
        sampling_params,
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        enable_code_execution: false,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        session_id: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: None,
        truncate_sequence: false,
        files: None,
    }));

    sender.send(req).await?;

    loop {
        match rx.recv().await {
            Some(Response::AgenticToolCallProgress { .. }) => continue,
            Some(Response::File(_)) => continue,
            Some(Response::CompletionDone(response)) => return Ok(response.usage),
            Some(Response::Done(response)) => return Ok(response.usage),
            Some(Response::InternalError(e)) => anyhow::bail!("Internal error: {e:?}"),
            Some(Response::ModelError(e, _)) => anyhow::bail!("Model error: {e}"),
            Some(Response::ValidationError(e)) => anyhow::bail!("Validation error: {e:?}"),
            Some(_) => anyhow::bail!("Unexpected response type"),
            None => anyhow::bail!("No response received"),
        }
    }
}

/// Print benchmark results in a nice table
#[allow(clippy::cast_precision_loss)]
fn print_results(model_id: &str, iterations: usize, results: &[BenchResult]) {
    println!();
    println!("Benchmark Results");
    println!("=================");
    println!();
    println!("Model: {}", model_id);
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
        // Determine latency label based on test type
        let latency_str = if result.test_name.contains("Prefill") {
            format!("{:.2} ms (prefill)", result.latency_ms)
        } else {
            format!("{:.2} ms/T", result.latency_ms)
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
