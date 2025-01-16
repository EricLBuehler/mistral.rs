use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::{self, Method},
    routing::{get, post},
    Router,
};
use candle_core::Device;
use clap::Parser;
use mistralrs_core::{
    get_auto_device_map_params, get_model_dtype, get_tgt_non_granular_index, initialize_logging,
    paged_attn_supported, parse_isq_value, DefaultSchedulerMethod, DeviceLayerMapMetadata,
    DeviceMapMetadata, DeviceMapSetting, IsqType, Loader, LoaderBuilder, MemoryGpuConfig,
    MistralRs, MistralRsBuilder, ModelSelected, PagedAttentionConfig, Request, SchedulerConfig,
    TokenSource,
};
use openai::{
    ChatCompletionRequest, CompletionRequest, ImageGenerationRequest, Message, ModelObjects,
    StopTokens,
};
use serde::{Deserialize, Serialize};
use std::{num::NonZeroUsize, sync::Arc};

mod chat_completion;
mod completions;
mod image_generation;
mod interactive_mode;
mod openai;
mod util;

use crate::openai::ModelObject;
use crate::{
    chat_completion::{__path_chatcompletions, chatcompletions},
    completions::completions,
    image_generation::image_generation,
};

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

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// IP to serve on. Defaults to "0.0.0.0"
    #[arg(long)]
    serve_ip: Option<String>,

    /// Integer seed to ensure reproducible random number generation.
    #[arg(short, long)]
    seed: Option<u64>,

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

    /// Enter interactive mode instead of serving a chat server.
    #[clap(long, short, action)]
    interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    #[arg(long, default_value_t = 16)]
    prefix_cache_n: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    #[arg(short, long, value_parser, value_delimiter = ';')]
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    #[arg(long = "isq", value_parser = parse_isq_value)]
    in_situ_quant: Option<IsqType>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-gpu-mem-usage` (default = 0.9) > `pa-ctxt-len` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem")]
    paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-gpu-mem-usage` (default = 0.9) > `pa-ctxt-len` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem-usage")]
    paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-gpu-mem-usage` (default = 0.9) > `pa-ctxt-len` > `pa-gpu-mem`.
    #[arg(long = "pa-ctxt-len")]
    paged_ctxt_len: Option<usize>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    #[arg(long = "pa-blk-size")]
    paged_attn_block_size: Option<usize>,

    /// Disable PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
    #[arg(long = "no-paged-attn", default_value_t = false)]
    no_paged_attn: bool,

    /// Enable PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
    #[arg(long = "paged-attn", default_value_t = false)]
    paged_attn: bool,

    /// Enable server throughput logging, supported in the server and with interactive mode
    #[arg(long = "throughput", default_value_t = false)]
    throughput_log: bool,

    /// Number of tokens to batch the prompt step into. This can help with OOM errors when in the prompt step, but reduces performance.
    #[arg(long = "prompt-batchsize")]
    prompt_batchsize: Option<usize>,

    /// Use CPU only
    #[arg(long)]
    cpu: bool,
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
    let request = Request::ReIsq(parse_isq_value(&request.ggml_type)?);
    state.get_sender().unwrap().send(request).await.unwrap();
    Ok(repr)
}

fn get_router(state: Arc<MistralRs>) -> Router {
    #[derive(OpenApi)]
    #[openapi(
        paths(models, health, chatcompletions),
        components(
            schemas(ModelObjects, ModelObject, ChatCompletionRequest, CompletionRequest, ImageGenerationRequest, StopTokens, Message)),
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
        .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
        .allow_origin(allow_origin);

    Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", doc))
        .route("/v1/chat/completions", post(chatcompletions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/", get(health))
        .route("/activate_adapters", post(activate_adapters))
        .route("/re_isq", post(re_isq))
        .route("/v1/images/generations", post(image_generation))
        .layer(cors_layer)
        .layer(DefaultBodyLimit::max(N_INPUT_SIZE * MB_TO_B))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();
    initialize_logging();

    let setting_server = if !args.interactive_mode {
        let port = args.port.expect("Interactive mode was not specified, so expected port to be specified. Perhaps you forgot `-i` or `--port`?");
        let ip = args.serve_ip.unwrap_or_else(|| "0.0.0.0".to_string());

        // Create listener early to validate address before model loading
        let listener = tokio::net::TcpListener::bind(format!("{ip}:{port}")).await?;
        Some((listener, ip, port))
    } else {
        None
    };

    #[cfg(not(feature = "flash-attn"))]
    let use_flash_attn = false;
    #[cfg(feature = "flash-attn")]
    let use_flash_attn = true;

    let tgt_non_granular_index = get_tgt_non_granular_index(&args.model);
    let dtype = get_model_dtype(&args.model)?;
    let auto_device_map_params = get_auto_device_map_params(&args.model)?;

    if tgt_non_granular_index.is_some() {
        args.max_seqs = 1;
    }

    let prompt_batchsize = match args.prompt_batchsize {
        Some(0) => {
            anyhow::bail!("`prompt_batchsize` must be a strictly positive integer, got 0.",)
        }
        Some(x) => Some(NonZeroUsize::new(x).unwrap()),
        None => None,
    };

    let loader: Box<dyn Loader> = LoaderBuilder::new(args.model)
        .with_no_kv_cache(args.no_kv_cache)
        .with_chat_template(args.chat_template)
        .with_use_flash_attn(use_flash_attn)
        .with_prompt_batchsize(prompt_batchsize)
        .build()?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = if args.cpu {
        args.no_paged_attn = true;
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
    }

    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    info!("Sampling method: penalties -> temperature -> topk -> topp -> minp -> multinomial");
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
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(vec![
                DeviceLayerMapMetadata { ordinal: 0, layers },
            ]))
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
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(mapping))
        }
    } else {
        DeviceMapSetting::Auto(auto_device_map_params)
    };

    let no_paged_attn = if device.is_cuda() {
        args.no_paged_attn
    } else if device.is_metal() {
        !args.paged_attn
    } else {
        true
    };

    // Allocate 0.5 GB of CPU memory just as a placeholder.
    // Nothing happens here as we have no `swap_out`, see `_preempt_by_swap`.
    let cache_config = match (
        args.paged_attn_block_size,
        args.paged_attn_gpu_mem,
        args.paged_attn_gpu_mem_usage,
        args.paged_ctxt_len,
        paged_attn_supported(),
        no_paged_attn,
    ) {
        (block_size, None, None, None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::Utilization(0.9), // NOTE(EricLBuehler): default is to use 90% of memory
        )?),
        (block_size, None, None, Some(ctxt), true, false) => Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::ContextSize(ctxt),
        )?),
        (block_size, None, Some(f), None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::Utilization(f),
        )?),
        (block_size, Some(m), None, None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::MbAmount(m),
        )?),
        (block_size, Some(_m), Some(f), None, true, false) => {
            info!("Both memory size, and usage were specified, defaulting to the usage value.");
            Some(PagedAttentionConfig::new(
                block_size,
                512,
                MemoryGpuConfig::Utilization(f),
            )?)
        }
        (block_size, Some(_m), None, Some(ctxt), true, false) => {
            info!("All memory size and ctxt len, defaulting to the context len value.");
            Some(PagedAttentionConfig::new(
                block_size,
                512,
                MemoryGpuConfig::ContextSize(ctxt),
            )?)
        }
        (block_size, None, Some(f), Some(_ctxt), true, false) => {
            info!("Both ctxt len and usage were specified, defaulting to the usage value.");
            Some(PagedAttentionConfig::new(
                block_size,
                512,
                MemoryGpuConfig::Utilization(f),
            )?)
        }
        (_, _, _, _, _, _) => None,
    };

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
        // Handle case where we may have device mapping
        if let Some(ref cache_config) = pipeline.lock().await.get_metadata().cache_config {
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: args.max_seqs,
                config: cache_config.clone(),
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
            }
        }
    } else {
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
        }
    };
    // Throughput logging in the server
    let builder = MistralRsBuilder::new(pipeline, scheduler_config)
        .with_opt_log(args.log)
        .with_truncate_sequence(args.truncate_sequence)
        .with_no_kv_cache(args.no_kv_cache)
        .with_prefix_cache_n(args.prefix_cache_n)
        .with_gemm_full_precision_f16(args.cpu)
        .with_gemm_full_precision_f16(args.cpu); // Required to allow `cuda` build to use `--cpu`, #1056

    if args.interactive_mode {
        interactive_mode(builder.build(), args.throughput_log).await;
        return Ok(());
    }

    let builder = if args.throughput_log {
        builder.with_throughput_logging()
    } else {
        builder
    };
    let mistralrs = builder.build();

    let app = get_router(mistralrs);
    if let Some((listener, ip, port)) = setting_server {
        info!("Serving on http://{ip}:{}.", port);
        axum::serve(listener, app).await?;
    };

    Ok(())
}
