use std::sync::{mpsc::channel, Arc};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    routing::get,
    Router,
};
use candle_core::{DType, Device};
use clap::Parser;
use mistralrs_core::{
    Loader, MistralLoader, MistralRs, MistralSpecificConfig, Request, Response, SamplingParams,
    SchedulerMethod, TokenSource,
};
use serde::Deserialize;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Port to serve on.
    #[arg(short, long)]
    port: String,

    /// Log all responses and requests to responses.log
    #[clap(long, short, action)]
    log: bool,
}

#[derive(Clone, Deserialize)]
struct RawRequest {
    pub prompt: String,
    pub sampling_params: SamplingParams,
}

async fn root(State(state): State<Arc<MistralRs>>, Json(request): Json<RawRequest>) -> String {
    let (tx, rx) = channel();
    let request = Request {
        prompt: request.prompt,
        sampling_params: request.sampling_params,
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

    let model_id = "mistralai/Mistral-7B-Instruct-v0.1";
    /*
    let loader = MistralLoader::new(
        model_id.to_string(),
        MistralSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64,
        },
        Some(DType::F32),
    );*/
    let loader = MistralLoader::new(
        model_id.to_string(),
        MistralSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64,
        },
        Some(DType::F32),
        Some("lmz/candle-mistral".to_string()),
        Some("model-q4k.gguf".to_string()),
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
