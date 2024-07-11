use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::{self, Method},
    routing::{get, post},
    Router,
};
use candle_core::{quantized::GgmlDType, Device};
use clap::Parser;
use mistralrs_core::{
    get_model_dtype, get_tgt_non_granular_index, initialize_logging, DefaultSchedulerMethod,
    DeviceLayerMapMetadata, DeviceMapMetadata, Loader, LoaderBuilder, MistralRs, MistralRsBuilder,
    ModelSelected, PagedAttentionConfig, Request, SchedulerConfig, TokenSource,
};
use openai::{ChatCompletionRequest, Message, ModelObjects, StopTokens};
use serde::{Deserialize, Serialize};
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
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
const MB_TO_B: usize = 1024 * 1024; // 1024 kb in a mb

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}

fn parse_isq(s: &str) -> Result<GgmlDType, String> {
    match s {
        "Q4_0" => Ok(GgmlDType::Q4_0),
        "Q4_1" => Ok(GgmlDType::Q4_1),
        "Q5_0" => Ok(GgmlDType::Q5_0),
        "Q5_1" => Ok(GgmlDType::Q5_1),
        "Q8_0" => Ok(GgmlDType::Q8_0),
        "Q8_1" => Ok(GgmlDType::Q8_1),
        "Q2K" => Ok(GgmlDType::Q2K),
        "Q3K" => Ok(GgmlDType::Q3K),
        "Q4K" => Ok(GgmlDType::Q4K),
        "Q5K" => Ok(GgmlDType::Q5K),
        "Q6K" => Ok(GgmlDType::Q6K),
        "Q8K" => Ok(GgmlDType::Q8K),
        _ => Err(format!("GGML type {s} unknown, choose one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`.")),
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

    /// Model selector
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
    /// Can be in the formats: `literal:<value>`, `env:<value>`, `path:<value>`, `cache` to use a cached token, or `none` to use no token.
    /// Defaults to `cache`.
    #[arg(long, default_value_t = TokenSource::CacheToken, value_parser = parse_token_source)]
    token_source: TokenSource,

    /// Enter interactive mode instead of serving a chat server. Exclusive to `--vi` (vision interactive mode).
    #[clap(long, short, action)]
    interactive_mode: bool,

    /// Enter vision interactive mode instead of serving a chat server. Exclusive to `--interactive-mode/-i`.
    #[clap(long = "vi", action)]
    vision_interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    #[arg(long, default_value_t = 16)]
    prefix_cache_n: usize,

    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    #[arg(short, long, value_parser, value_delimiter = ';')]
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply. You may specify one of the GGML data type (except F32 or F16): formatted like this: `Q4_0` or `Q4K`.
    #[arg(long = "isq", value_parser = parse_isq)]
    in_situ_quant: Option<GgmlDType>,

    /// GPU memory to allocate for KV cache with Paged Attention in MBs. If this is set, then so must `pa_blk_size` be to use Paged Attention.
    #[arg(long = "pa-gpu-mem")]
    paged_attn_gpu_mem: Option<usize>,

    /// Block size (number of tokens per block) for Paged Attention. If this is set, then so must `pa_gpu_mem` be to use Paged Attention.
    #[arg(long = "pa-blk-size")]
    paged_attn_block_size: Option<usize>,
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

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
struct AdapterActivationRequest {
    #[schema(example = json!(vec!["adapter_1","adapter_2"]))]
    adapter_names: Vec<String>,
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/activate_adapters",
    request_body = AdapterActivationRequest,
    responses((status = 200, description = "Activate a set of pre-loaded LoRA adapters"))
)]
async fn activate_adapters(
    State(state): State<Arc<MistralRs>>,
    Json(request): Json<AdapterActivationRequest>,
) -> String {
    let repr = format!("Adapter activation: {:?}", request.adapter_names);
    MistralRs::maybe_log_request(state.clone(), repr.clone());
    let request = Request::ActivateAdapters(request.adapter_names);
    state.get_sender().unwrap().send(request).await.unwrap();
    repr
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
struct ReIsqRequest {
    #[schema(example = "Q4K")]
    ggml_type: String,
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/re_isq",
    request_body = ReIsqRequest,
    responses((status = 200, description = "Reapply ISQ to a non GGUF or GGML model."))
)]
async fn re_isq(
    State(state): State<Arc<MistralRs>>,
    Json(request): Json<ReIsqRequest>,
) -> Result<String, String> {
    let repr = format!("Re ISQ: {:?}", request.ggml_type);
    MistralRs::maybe_log_request(state.clone(), repr.clone());
    let request = Request::ReIsq(parse_isq(&request.ggml_type)?);
    state.get_sender().unwrap().send(request).await.unwrap();
    Ok(repr)
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
        .route("/activate_adapters", post(activate_adapters))
        .route("/re_isq", post(re_isq))
        .layer(DefaultBodyLimit::max(N_INPUT_SIZE * MB_TO_B))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();
    initialize_logging();

    #[cfg(not(feature = "flash-attn"))]
    let use_flash_attn = false;
    #[cfg(feature = "flash-attn")]
    let use_flash_attn = true;

    let tgt_non_granular_index = get_tgt_non_granular_index(&args.model);
    let dtype = get_model_dtype(&args.model)?;

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

    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    info!("Sampling method: penalties -> temperature -> topk -> topp -> multinomial");
    if use_flash_attn {
        info!("Using flash attention.");
    }
    if use_flash_attn && loader.get_kind().is_quantized() {
        warn!("Using flash attention with a quantized model has no effect!")
    }
    info!("Model kind is: {}", loader.get_kind().to_string());

    // Parse device mapper
    let mapper = if let Some(device_layers) = args.num_device_layers {
        if device_layers.len() == 1 && device_layers[0].parse::<usize>().is_ok() {
            let layers = device_layers[0].parse::<usize>().unwrap();
            DeviceMapMetadata::from_num_device_layers_multi_gpu(vec![DeviceLayerMapMetadata {
                ordinal: 0,
                layers,
            }])
        } else {
            let mut mapping = Vec::new();
            for layer in device_layers {
                let split = layer.splitn(2, ':').collect::<Vec<_>>();
                if split.len() < 2 {
                    panic!("Expected layer to be of format ORD:NUM, got {layer}");
                }
                let ord = split[0]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[0]));
                let num = split[1]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[1]));
                for DeviceLayerMapMetadata { ordinal, layers: _ } in &mapping {
                    if *ordinal == ord {
                        panic!("Duplicate ordinal {ord}");
                    }
                }
                mapping.push(DeviceLayerMapMetadata {
                    ordinal: ord,
                    layers: num,
                });
            }
            DeviceMapMetadata::from_num_device_layers_multi_gpu(mapping)
        }
    } else {
        DeviceMapMetadata::dummy()
    };

    let cache_config = if args.paged_attn_block_size.is_some() && args.paged_attn_gpu_mem.is_some()
    {
        // Allocate 0.5 GB of CPU memory just as a placeholder.
        // Nothing happens here as we have no `swap_out`, see `_preempt_by_swap`.
        Some(PagedAttentionConfig::new(
            args.paged_attn_block_size,
            512,
            args.paged_attn_gpu_mem.unwrap(),
        )?)
    } else {
        None
    };
    // let cache_config = None;
    let pipeline = loader.load_model_from_hf(
        None,
        args.token_source,
        &dtype,
        &device,
        false,
        mapper,
        args.in_situ_quant,
        cache_config,
    )?;
    info!("Model loaded.");

    let scheduler_config = if cache_config.is_some() {
        SchedulerConfig::PagedAttentionMeta {
            max_num_seqs: args.max_seqs,
            config: pipeline
                .lock()
                .await
                .get_metadata()
                .cache_config
                .as_ref()
                .unwrap()
                .clone(),
        }
    } else {
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
        }
    };
    let mistralrs = MistralRsBuilder::new(pipeline, scheduler_config)
        .with_opt_log(args.log)
        .with_truncate_sequence(args.truncate_sequence)
        .with_no_kv_cache(args.no_kv_cache)
        .with_prefix_cache_n(args.prefix_cache_n)
        .build();

    if args.interactive_mode && args.vision_interactive_mode {
        anyhow::bail!("Interactive mode and vision interactive mode are exclusive.");
    } else if args.interactive_mode {
        interactive_mode(mistralrs, false).await;
        return Ok(());
    } else if args.vision_interactive_mode {
        interactive_mode(mistralrs, true).await;
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
