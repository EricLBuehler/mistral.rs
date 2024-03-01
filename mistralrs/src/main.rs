use std::sync::{mpsc::channel, Arc};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    routing::get,
    Router,
};
use candle_core::Device;
use clap::Parser;
use mistralrs_core::{
    Loader, MistralLoader, MistralRs, MistralSpecificConfig, Request, Response, SamplingParams,
    SchedulerMethod, StopTokens as InternalStopTokens, TokenSource,
};
use openai::{ChatCompletionRequest, Message, StopTokens};
mod openai;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Port to serve on.
    #[arg(short, long)]
    port: String,

    /// Log all responses and requests to output.log
    #[clap(long, short, action)]
    log: bool,

    /// Model ID to load the
    #[arg(short, long, default_value = "mistralai/Mistral-7B-Instruct-v0.1")]
    model_id: String,

    /// Enable quantized.
    #[clap(long, short, action)]
    quantized: bool,

    /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
    #[arg(short, long, default_value = "lmz/candle-mistral")]
    quantized_model_id: Option<String>,

    /// Quantized filename, only applicable if `quantized` is set.
    #[arg(short, long, default_value = "model-q4k.gguf")]
    quantized_filename: Option<String>,
}

fn get_prompt(_messages: Vec<Message>) -> String {
    todo!()
}

async fn root(
    State(state): State<Arc<MistralRs>>,
    Json(request): Json<ChatCompletionRequest>,
) -> String {
    let (tx, rx) = channel();
    let stop_toks = match request.stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        Some(StopTokens::MultiId(m)) => Some(InternalStopTokens::Ids(m)),
        Some(StopTokens::SingleId(s)) => Some(InternalStopTokens::Ids(vec![s])),
        None => None,
    };
    let request = Request {
        prompt: get_prompt(request.messages),
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            top_n_logprobs: request.top_logprobs.unwrap_or(1),
            repeat_penalty: request.repetition_penalty,
            presence_penalty: request.presence_penalty,
            max_len: request.max_tokens,
            stop_toks,
        },
        response: tx,
    };

    MistralRs::maybe_log_request(state.clone(), &request);
    let sender = state.get_sender();
    sender.send(request).unwrap();
    let response = rx.recv().unwrap();

    match response {
        Response::Error(e) => {
            dbg!(&e);
            e.to_string()
        }
        Response::Done((reason, out)) => {
            MistralRs::maybe_log_response(state, (reason, &out));
            out
        }
    }
}

fn get_router(state: Arc<MistralRs>) -> Router {
    Router::new().route("/", get(root)).with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let loader = MistralLoader::new(
        args.model_id,
        MistralSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64,
        },
        if args.quantized {
            Some(
                args.quantized_model_id
                    .expect("Quantized model ID must be set if quantized is."),
            )
        } else {
            None
        },
        if args.quantized {
            Some(
                args.quantized_filename
                    .expect("Quantized filename must be set if quantized is."),
            )
        } else {
            None
        },
    );
    let pipeline = loader.load_model(
        None,
        TokenSource::CacheToken,
        None,
        &Device::cuda_if_available(0)?,
    )?;
    let mistralrs = MistralRs::new(
        pipeline,
        SchedulerMethod::Fixed(2.try_into().unwrap()),
        args.log,
    );

    let app = get_router(mistralrs);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
