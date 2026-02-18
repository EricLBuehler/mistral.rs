//! Compute perplexity of a text file using a loaded model.
//!
//! Run with: `cargo run --release --example perplexity -p mistralrs`

use std::{fs::read_to_string, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use either::Either;
use mistralrs::{
    cross_entropy_loss, parse_isq_value, Constraint, DType, Device, MistralRs, ModelBuilder,
    NormalRequest, Request, ResponseOk, SamplingParams, Tensor,
};
use tokio::sync::mpsc::channel;

/// Calculate perplexity of a model. By default, this uses the Llama 3.1 8B model.
#[derive(Parser)]
struct Args {
    /// The model to run.
    #[arg(short, long, default_value = "google/gemma-3-4b-it")]
    model_id: String,

    /// Filename to text to run the model on. This is recommended to be the Wikitext 2 dataset:
    /// https://huggingface.co/datasets/EricB/wikitext2
    #[arg(short, long)]
    file: String,

    /// ISQ quantization to run with.
    #[arg(short, long)]
    isq: Option<String>,

    /// Generate and utilize an imatrix to enhance GGUF quantizations.
    #[arg(short, long)]
    calibration_file: Option<PathBuf>,
}

async fn process_chunk(runner: &MistralRs, chunk: Vec<u32>) -> anyhow::Result<(Tensor, Vec<u32>)> {
    let (tx, mut rx) = channel(1);

    let request = Request::Normal(Box::new(NormalRequest {
        messages: mistralrs::RequestMessage::CompletionTokens(chunk),
        sampling_params: SamplingParams {
            max_len: Some(0),
            ..SamplingParams::deterministic()
        },
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: true,
        web_search_options: None,
        model_id: None,
        truncate_sequence: false,
    }));

    runner.get_sender(None)?.send(request).await?;

    let ResponseOk::Raw {
        logits_chunks,
        tokens,
    } = rx
        .recv()
        .await
        .context("Channel was erroneously closed!")?
        .as_result()?
    else {
        anyhow::bail!("Got unexpected response type.")
    };

    Ok((logits_chunks[0].clone(), tokens))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let quant = if let Some(isq) = &args.isq {
        Some(parse_isq_value(isq, None).map_err(anyhow::Error::msg)?)
    } else {
        None
    };

    let prompt_chunksize = 1024;
    let mut model_builder = ModelBuilder::new(&args.model_id).with_logging();
    if let Some(quant) = quant {
        model_builder = model_builder.with_isq(quant);
    }
    if let Some(calibration_file) = &args.calibration_file {
        model_builder = model_builder.with_calibration_file(calibration_file.clone());
    }

    let model = model_builder.build().await?;

    let text = read_to_string(&args.file)?;
    let tokens = model
        .tokenize(Either::Right(text), None, false, false, None)
        .await?;
    let bos_token = model
        .tokenize(Either::Right(" ".to_string()), None, true, false, None)
        .await?[0];
    let inner = model.inner();

    println!("Using bos token id `{bos_token}`.");

    let n_chunks = tokens.len().div_ceil(prompt_chunksize);
    let mut ppl_measurements = Vec::new();
    for (i, chunk) in tokens.chunks(prompt_chunksize).enumerate() {
        let start = Instant::now();
        let (logits, tokens) = {
            let chunk = [vec![bos_token], chunk.to_vec()].concat();
            process_chunk(inner, chunk).await?
        };

        // Upcast to float if we need to compute the loss to avoid potential precision issues
        let logits = logits.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        // Shift so that tokens < n predict n
        let shift_logits = logits.narrow(0, 0, logits.dim(0)? - 1)?.contiguous()?;
        let shift_labels = Tensor::from_slice(&tokens[1..], (tokens.len() - 1,), &Device::Cpu)?;

        let loss_fct = cross_entropy_loss(&shift_logits, &shift_labels)?;
        let perplexity = loss_fct.exp()?.to_scalar::<f32>()?;
        let end = Instant::now();

        ppl_measurements.push(perplexity);
        println!(
            "Chunk {i}/{n_chunks} ({} tokens): Perplexity for `{}`, ISQ `{:?}`, {}s: {perplexity}",
            tokens.len(),
            args.file,
            quant,
            end.duration_since(start).as_secs_f32(),
        );
    }

    let mean = ppl_measurements.iter().sum::<f32>() / ppl_measurements.len() as f32;
    let variance = ppl_measurements
        .iter()
        .map(|e| (mean - e).powf(2.))
        .sum::<f32>()
        / ppl_measurements.len() as f32;
    let std_dev = variance.sqrt();
    println!();
    println!(
        "Final perplexity for `{}`, ISQ `{:?}`: {}±{} ppl",
        args.file, quant, mean, std_dev
    );

    Ok(())
}
