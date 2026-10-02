//! Read final-norm hidden states at chosen token positions from a GGUF model.
//!
//! Input JSON: `{"rows": [{"ids": [u32, ..], "positions": [usize, ..]}, ..]}` where positions
//! index into `ids`. Output JSON: `{"rows": [{"hidden": [[f32; hidden_size], ..]}, ..]}`, one
//! vector per requested position.
//!
//! Run with: `cargo run --release --features cuda --example hidden_states -p mistralrs -- \
//!   --model-dir DIR --file model.gguf --input rows.json --output hidden.json`

use std::{fs, time::Instant};

use anyhow::{Context, Result};
use candle_core::IndexOp;
use clap::Parser;
use mistralrs::{
    Constraint, GgufModelBuilder, ModelDType, NormalRequest, Request, RequestMessage, ResponseOk,
    SamplingParams, Tensor,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::channel;

#[derive(Parser)]
struct Args {
    /// Directory holding the GGUF file.
    #[arg(long)]
    model_dir: String,
    /// GGUF filename inside `model_dir`.
    #[arg(long)]
    file: String,
    /// Input rows (JSON).
    #[arg(long)]
    input: String,
    /// Output hidden states (JSON).
    #[arg(long)]
    output: String,
    /// Activation dtype: auto, f32, f16 or bf16.
    #[arg(long, default_value = "auto")]
    dtype: String,
    /// Disable the prefix cache.
    #[arg(long)]
    no_prefix_cache: bool,
    /// Run on the CPU.
    #[arg(long)]
    cpu: bool,
}

#[derive(Deserialize)]
struct Row {
    ids: Vec<u32>,
    positions: Vec<usize>,
}

#[derive(Deserialize)]
struct Input {
    rows: Vec<Row>,
}

#[derive(Serialize)]
struct OutRow {
    hidden: Vec<Vec<f32>>,
    returned_rows: usize,
    seconds: f64,
}

#[derive(Serialize)]
struct Output {
    rows: Vec<OutRow>,
}

async fn hidden_rows(model: &mistralrs::Model, ids: Vec<u32>) -> Result<Tensor> {
    let (tx, mut rx) = channel(1);
    let request = Request::Normal(Box::new(NormalRequest {
        messages: RequestMessage::CompletionTokens(ids),
        sampling_params: SamplingParams {
            max_len: Some(1),
            ..SamplingParams::deterministic()
        },
        seed: None,
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        queued_at: None,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: true,
        web_search_options: None,
        enable_code_execution: false,
        enable_shell: false,
        shell_options: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: None,
        adapter: None,
        truncate_sequence: false,
        session_id: None,
        files: None,
        input_files: Vec::new(),
    }));
    model.inner().get_sender(None)?.send(request).await?;
    let ResponseOk::Raw { logits_chunks, .. } =
        rx.recv().await.context("channel closed")?.as_result()?
    else {
        anyhow::bail!("unexpected response type");
    };
    // Chunks are [1, tokens, hidden] or [tokens, hidden]; flatten to [tokens, hidden].
    let chunks = logits_chunks
        .into_iter()
        .map(|t| {
            let t = if t.rank() == 3 { t.i(0)? } else { t };
            t.to_dtype(mistralrs::DType::F32)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::cat(&chunks, 0)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let dtype = match args.dtype.as_str() {
        "auto" => ModelDType::Auto,
        "f32" => ModelDType::F32,
        "f16" => ModelDType::F16,
        "bf16" => ModelDType::BF16,
        other => anyhow::bail!("unknown dtype {other}"),
    };
    let mut builder = GgufModelBuilder::new(&args.model_dir, vec![args.file.clone()])
        .with_dtype(dtype)
        .with_hidden_states_output()
        .with_logging();
    if args.cpu {
        builder = builder.with_force_cpu();
    }
    if args.no_prefix_cache {
        builder = builder.with_prefix_cache_n(None);
    }
    let model = builder.build().await?;

    let input: Input = serde_json::from_str(&fs::read_to_string(&args.input)?)?;
    let mut rows = Vec::new();
    for row in input.rows {
        let n = row.ids.len();
        let start = Instant::now();
        let hidden = hidden_rows(&model, row.ids).await?;
        let seconds = start.elapsed().as_secs_f64();
        let returned = hidden.dim(0)?;
        // Returned rows are the last `returned` prompt tokens (earlier ones came from the cache).
        let offset = n - returned;
        let mut out = Vec::new();
        for p in row.positions {
            anyhow::ensure!(p >= offset, "position {p} was served from the prefix cache");
            out.push(hidden.i(p - offset)?.to_vec1::<f32>()?);
        }
        rows.push(OutRow {
            hidden: out,
            returned_rows: returned,
            seconds,
        });
    }
    fs::write(&args.output, serde_json::to_string(&Output { rows })?)?;
    Ok(())
}
