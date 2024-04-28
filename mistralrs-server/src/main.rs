use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::{self, Method},
    routing::{get, post},
    Router,
};
use candle_core::{quantized::GgmlDType, Device};
use clap::Parser;
use mistralrs_core::{
    get_tgt_non_granular_index, DeviceMapMetadata, Loader, LoaderBuilder, MistralRs,
    MistralRsBuilder, ModelKind, ModelSelected, SchedulerMethod, TokenSource,
};
use openai::{ChatCompletionRequest, Message, ModelObjects, StopTokens};
use std::sync::Arc;
mod chat_completion;
mod completions;
use crate::{chat_completion::__path_chatcompletions, completions::completions};

use crate::{chat_completion::chatcompletions, openai::ModelObject};
mod interactive_mode;
mod openai;

use interactive_mode::interactive_mode;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info, warn};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}

fn parse_isq(s: &str) -> Result<GgmlDType, String> {
    match s {
        "Q4_0" => Ok(GgmlDType::Q4_0),
        "Q4_1" => Ok(GgmlDType::Q4_1),
        "Q5_0" => Ok(GgmlDType::Q5_0),
        "Q5_1" => Ok(GgmlDType::Q5_1),
        "Q8_0" => Ok(GgmlDType::Q8_1),
        "Q8_1" => Ok(GgmlDType::Q4_0),
        "Q2K" => Ok(GgmlDType::Q2K),
        "Q3K" => Ok(GgmlDType::Q3K),
        "Q4K" => Ok(GgmlDType::Q4K),
        "Q5K" => Ok(GgmlDType::Q5K),
        "Q6K" => Ok(GgmlDType::Q6K),
        "Q8K" => Ok(GgmlDType::Q8K),
        _ => Err(format!("GGML type {s} unknown")),
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// IP to serve on. Defaults to "0.0.0.0"
    #[arg(long)]
    serve_ip: Option<String>,

    /// Port to serve on.
    #[arg(short, long)]
    port: Option<String>,

    /// Log all responses and requests to this file
    #[clap(long, short)]
    log: Option<String>,

    /// If a sequence is larger than the maximum model length, truncate the number
    /// of tokens such that the sequence will fit at most the maximum length.
    /// If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.
    #[clap(long, short, action)]
    truncate_sequence: bool,

    /// Model
    #[clap(subcommand)]
    model: ModelSelected,

    /// Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1.
    #[arg(long, default_value_t = 16)]
    max_seqs: usize,

    /// Use no KV cache.
    #[arg(long, default_value_t = false)]
    no_kv_cache: bool,

    /// JINJA chat template with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    /// Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
    #[arg(short, long)]
    chat_template: Option<String>,

    /// Source of the token for authentication.
    /// Can be in the formats: "literal:<value>", "env:<value>", "path:<value>", "cache" to use a cached token or "none" to use no token.
    /// Defaults to using a cached token.
    #[arg(long, default_value_t = TokenSource::CacheToken, value_parser = parse_token_source)]
    token_source: TokenSource,

    /// Enter interactive mode instead of serving a chat server.
    #[clap(long, short, action)]
    interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    #[arg(long, default_value_t = 16)]
    prefix_cache_n: usize,

    /// Number of device layers to load and run on the device. All others will be on the CPU.
    #[arg(short, long)]
    num_device_layers: Option<usize>,

    /// In-situ quantization to apply. You may specify one of the GGML data type (except F32 or F16): formatted like this: `Q4_0` or `Q4K`.
    #[arg(long = "isq", value_parser = parse_isq)]
    in_situ_quant: Option<GgmlDType>,
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/models",
    responses((status = 200, description = "Served model info", body = ModelObjects))
)]
async fn models(State(state): State<Arc<MistralRs>>) -> Json<ModelObjects> {
    Json(ModelObjects {
        object: "list",
        data: vec![ModelObject {
            id: state.get_id(),
            object: "model",
            created: state.get_creation_time(),
            owned_by: "local",
        }],
    })
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/health",
    responses((status = 200, description = "Server is healthy"))
)]
async fn health() -> &'static str {
    "OK"
}

fn get_router(state: Arc<MistralRs>) -> Router {
    #[derive(OpenApi)]
    #[openapi(
        paths(models, health, chatcompletions),
        components(
            schemas(ModelObjects, ModelObject, ChatCompletionRequest, StopTokens, Message)),
        tags(
            (name = "Mistral.rs", description = "Mistral.rs API")
        ),
        info(
            title = "Mistral.rs",
            license(
            name = "MIT",
        )
        )
    )]
    struct ApiDoc;

    let doc = { ApiDoc::openapi() };

    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", doc))
        .layer(cors_layer)
        .route("/v1/chat/completions", post(chatcompletions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/", get(health))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();

    #[cfg(not(feature = "flash-attn"))]
    let use_flash_attn = false;
    #[cfg(feature = "flash-attn")]
    let use_flash_attn = true;

    let tgt_non_granular_index = get_tgt_non_granular_index(&args.model);

    if tgt_non_granular_index.is_some() {
        args.max_seqs = 1;
    }

    let loader: Box<dyn Loader> = LoaderBuilder::new(args.model)
        .with_no_kv_cache(args.no_kv_cache)
        .with_chat_template(args.chat_template)
        .with_use_flash_attn(use_flash_attn)
        .build()?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::cuda_if_available(0)?;

    tracing_subscriber::fmt().init();

    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    info!("Sampling method: penalties -> temperature -> topk -> topp -> multinomial");
    info!("Loading model `{}` on {device:?}...", loader.get_id());
    if use_flash_attn {
        info!("Using flash attention.");
    }
    if use_flash_attn
        && matches!(
            loader.get_kind(),
            ModelKind::QuantizedGGML
                | ModelKind::QuantizedGGUF
                | ModelKind::XLoraGGML
                | ModelKind::XLoraGGUF
        )
    {
        warn!("Using flash attention with a quantized model has no effect!")
    }
    info!("Model kind is: {}", loader.get_kind().as_ref());
    let pipeline = loader.load_model(
        None,
        args.token_source,
        None,
        &device,
        false,
        args.num_device_layers
            .map(DeviceMapMetadata::from_num_device_layers)
            .unwrap_or(DeviceMapMetadata::dummy()),
        args.in_situ_quant,
    )?;
    info!("Model loaded.");

    let mistralrs = MistralRsBuilder::new(
        pipeline,
        SchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
    )
    .with_opt_log(args.log)
    .with_truncate_sequence(args.truncate_sequence)
    .with_no_kv_cache(args.no_kv_cache)
    .with_prefix_cache_n(args.prefix_cache_n)
    .build();

    if args.interactive_mode {
        interactive_mode(mistralrs).await;
        return Ok(());
    }

    let port = args.port.expect("Expected port to be specified.");

    let app = get_router(mistralrs);

    let ip = if let Some(ref ip) = args.serve_ip {
        ip.to_string()
    } else {
        "0.0.0.0".to_string()
    };
    let listener = tokio::net::TcpListener::bind(format!("{ip}:{}", port)).await?;
    info!("Serving on http://{ip}:{}.", port);
    axum::serve(listener, app).await?;

    Ok(())
}
