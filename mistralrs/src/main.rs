use anyhow::Result;
use axum::{routing::get, Router};
use candle_core::{DType, Device};
use clap::Parser;
use mistralrs_core::{Loader, MistralLoader, MistralRs, MistralSpecificConfig, TokenSource};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Port to serve on.
    #[arg(short, long)]
    port: String,
}

async fn root() -> &'static str {
    "Hello World!"
}

fn get_router() -> Router {
    Router::new().route("/", get(root))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let app = get_router();

    let model_id = "mistralai/Mistral-7B-Instruct-v0.1";
    let loader = MistralLoader::new(
        model_id.to_string(),
        MistralSpecificConfig {
            use_flash_attn: false,
        },
        Some(DType::F32),
    );
    let pipeline = loader.load_model(None, TokenSource::CacheToken, None, &Device::Cpu)?;
    let mistralrs = MistralRs::new(pipeline);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
