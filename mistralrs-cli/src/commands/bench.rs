//! Performance benchmarking command

use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use mistralrs_core::{
    initialize_logging, Constraint, DrySamplingParams, NormalRequest, Request, RequestMessage,
    Response, SamplingParams,
};
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::channel;
use tracing::info;

use crate::args::{GlobalOptions, ModelType, RuntimeOptions};

use super::serve::{
    convert_to_model_selected, extract_device_settings, extract_isq_setting,
    extract_paged_attn_settings,
};

/// Benchmark result for a single test
struct BenchResult {
    test_name: String,
    tok_per_sec: f32,
    std_dev: f32,
    /// For prefill: TTFT in ms; for decode: ms/tok
    latency_ms: f32,
}

/// Extract model_id from ModelType
fn get_model_id(model_type: &ModelType) -> String {
    match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Vision { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model.model_id.clone(),
    }
}

/// Run the benchmark command
pub async fn run_bench(
    model_type: ModelType,
    runtime: RuntimeOptions,
    global: GlobalOptions,
    prompt_len: usize,
    gen_len: usize,
    iterations: usize,
    warmup: usize,
) -> Result<()> {
    initialize_logging();

    // Get model ID for display
    let model_id = get_model_id(&model_type);

    // Convert args and load model
    let model_selected = convert_to_model_selected(&model_type)?;

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
            let _ = run_single_bench(&mistralrs, 32, 16).await;
        }
        info!("Warmup complete.");

        // Reset logger counters so benchmark stats are clean
        if let Ok(logger) = mistralrs.get_logger(None) {
            logger.reset();
        }
    }

    // Run benchmarks
    info!(
        "Running {} iteration(s) with {} prompt tokens, {} generation tokens...",
        iterations, prompt_len, gen_len
    );

    let mut prefill_results = Vec::new();
    let mut decode_results = Vec::new();

    for i in 0..iterations {
        info!("Iteration {}/{}...", i + 1, iterations);

        // Prefill benchmark (prompt processing)
        // Use external timing since internal Usage timing may not capture prompt time accurately
        if prompt_len > 0 {
            let start = Instant::now();
            run_single_bench(&mistralrs, prompt_len, 1).await?;
            let elapsed = start.elapsed();
            // Record both tok/s and TTFT (latency in ms)
            let tok_per_sec = prompt_len as f32 / elapsed.as_secs_f32();
            let ttft_ms = elapsed.as_secs_f32() * 1000.0;
            prefill_results.push((tok_per_sec, ttft_ms));
        }

        // Decode benchmark (token generation)
        if gen_len > 0 {
            let start = Instant::now();
            run_single_bench(&mistralrs, 4, gen_len).await?;
            let elapsed = start.elapsed();
            // Record both tok/s and ms/tok
            let tok_per_sec = gen_len as f32 / elapsed.as_secs_f32();
            let ms_per_tok = 1000.0 / tok_per_sec;
            decode_results.push((tok_per_sec, ms_per_tok));
        }
    }

    // Calculate statistics
    let mut results = Vec::new();

    if !prefill_results.is_empty() {
        let tok_per_sec_vals: Vec<f32> = prefill_results.iter().map(|(t, _)| *t).collect();
        let ttft_vals: Vec<f32> = prefill_results.iter().map(|(_, l)| *l).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let (mean_ttft, _) = calculate_stats(&ttft_vals);
        results.push(BenchResult {
            test_name: format!("Prefill ({} tokens)", prompt_len),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: mean_ttft, // TTFT
        });
    }

    if !decode_results.is_empty() {
        let tok_per_sec_vals: Vec<f32> = decode_results.iter().map(|(t, _)| *t).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let ms_per_tok = 1000.0 / mean_tps;
        results.push(BenchResult {
            test_name: format!("Decode ({} tokens)", gen_len),
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

/// Run a single benchmark iteration
async fn run_single_bench(
    mistralrs: &Arc<mistralrs_core::MistralRs>,
    prompt_tokens: usize,
    gen_tokens: usize,
) -> Result<()> {
    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        min_p: Some(0.05),
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        repetition_penalty: None,
        max_len: Some(gen_tokens),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
    };

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
        model_id: None,
        truncate_sequence: false,
    }));

    sender.send(req).await?;

    match rx.recv().await {
        Some(Response::CompletionDone(_)) | Some(Response::Done(_)) => Ok(()),
        Some(Response::InternalError(e)) => anyhow::bail!("Internal error: {e:?}"),
        Some(Response::ModelError(e, _)) => anyhow::bail!("Model error: {e}"),
        Some(Response::ValidationError(e)) => anyhow::bail!("Validation error: {e:?}"),
        Some(_) => anyhow::bail!("Unexpected response type"),
        None => anyhow::bail!("No response received"),
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
            format!("{:.2} ms (TTFT)", result.latency_ms)
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
