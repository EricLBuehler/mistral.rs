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
    paged_attn_supported, parse_isq_value, AutoDeviceMapParams, BertEmbeddingModel,
    DefaultSchedulerMethod, DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting, IsqType,
    Loader, LoaderBuilder, MemoryGpuConfig, MistralRs, MistralRsBuilder, ModelSelected,
    PagedAttentionConfig, Pipeline, Request, SchedulerConfig, TokenSource,
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
mod openai;
mod util;

use crate::openai::ModelObject;
use crate::{
    chat_completion::{__path_chatcompletions, chatcompletions},
    completions::completions,
    image_generation::image_generation,
};

use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::info;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
const MB_TO_B: usize = 1024 * 1024; // 1024 kb in a mb

type LoadedPipeline = Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// IP to serve on. Defaults to "0.0.0.0"
    #[arg(long)]
    pub serve_ip: Option<String>,

    /// Integer seed to ensure reproducible random number generation.
    #[arg(short, long)]
    pub seed: Option<u64>,

    /// Port to serve on.
    #[arg(short, long)]
    pub port: Option<String>,

    /// Log all responses and requests to this file
    #[clap(long, short)]
    pub log: Option<String>,

    /// If a sequence is larger than the maximum model length, truncate the number
    /// of tokens such that the sequence will fit at most the maximum length.
    /// If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.
    #[clap(long, short, action)]
    pub truncate_sequence: bool,

    /// Model selector
    #[clap(subcommand)]
    pub model: ModelSelected,

    /// Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1.
    #[arg(long, default_value_t = 16)]
    pub max_seqs: usize,

    /// Use no KV cache.
    #[arg(long, default_value_t = false)]
    pub no_kv_cache: bool,

    /// Chat template file with a JINJA file with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    /// Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
    #[arg(short, long)]
    pub chat_template: Option<String>,

    /// Explicit JINJA chat template file (.jinja) to be used. If specified, this overrides all other chat templates.
    #[arg(short, long)]
    pub jinja_explicit: Option<String>,

    /// Source of the token for authentication.
    /// Can be in the formats: `literal:<value>`, `env:<value>`, `path:<value>`, `cache` to use a cached token, or `none` to use no token.
    /// Defaults to `cache`.
    #[arg(long, default_value_t = TokenSource::CacheToken, value_parser = parse_token_source)]
    pub token_source: TokenSource,

    /// Enter interactive mode instead of serving a chat server.
    #[clap(long, short, action)]
    pub interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    #[arg(long, default_value_t = 16)]
    pub prefix_cache_n: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    #[arg(short, long, value_parser, value_delimiter = ';')]
    pub num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    #[arg(long = "isq", value_parser = parse_isq_value)]
    pub in_situ_quant: Option<IsqType>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem")]
    pub paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem-usage")]
    pub paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    /// This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
    #[arg(long = "pa-ctxt-len")]
    pub paged_ctxt_len: Option<usize>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    #[arg(long = "pa-blk-size")]
    pub paged_attn_block_size: Option<usize>,

    /// Disable PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
    #[arg(long = "no-paged-attn", default_value_t = false)]
    pub no_paged_attn: bool,

    /// Enable PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
    #[arg(long = "paged-attn", default_value_t = false)]
    pub paged_attn: bool,

    /// Enable server throughput logging, supported in the server and with interactive mode
    #[arg(long = "throughput", default_value_t = false)]
    pub throughput_log: bool,

    /// Number of tokens to batch the prompt step into. This can help with OOM errors when in the prompt step, but reduces performance.
    #[arg(long = "prompt-batchsize")]
    pub prompt_chunksize: Option<usize>,

    /// Use CPU only
    #[arg(long)]
    pub cpu: bool,

    /// Enable searching compatible with the OpenAI `web_search_options` setting. This uses the BERT model specified below or the default.
    #[arg(long = "enable-search")]
    pub enable_search: bool,

    /// Specify a Hugging Face model ID for a BERT model to assist web searching. Defaults to Snowflake Arctic Embed L.
    #[arg(long = "search-bert-model")]
    pub search_bert_model: Option<String>,

    /// Enable thinking for interactive mode and models that support it.
    #[arg(long = "enable-thinking")]
    pub enable_thinking: bool,
}

pub type SharedMistralState = Arc<MistralRs>;
pub type ExtractedMistralState = State<SharedMistralState>;

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/v1/models",
  responses((status = 200, description = "Served model info", body = ModelObjects))
)]
async fn models(State(state): ExtractedMistralState) -> Json<ModelObjects> {
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
    State(state): ExtractedMistralState,
    Json(request): Json<ReIsqRequest>,
) -> Result<String, String> {
    let repr = format!("Re ISQ: {:?}", request.ggml_type);
    MistralRs::maybe_log_request(state.clone(), repr.clone());
    let request = Request::ReIsq(parse_isq_value(&request.ggml_type)?);
    state.get_sender().unwrap().send(request).await.unwrap();
    Ok(repr)
}

pub fn get_openapi_doc(base_path: Option<&str>) -> utoipa::openapi::OpenApi {
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

    let mut doc = ApiDoc::openapi();

    if let Some(prefix) = base_path {
        if !prefix.is_empty() {
            let mut prefixed_paths = utoipa::openapi::Paths::default();

            let original_paths = std::mem::take(&mut doc.paths.paths);

            for (path, item) in original_paths {
                let prefixed_path = format!("{}{}", prefix, path);
                prefixed_paths.paths.insert(prefixed_path, item);
            }

            prefixed_paths.extensions = doc.paths.extensions.clone();

            doc.paths = prefixed_paths;
        }
    }

    doc
}

pub async fn bootstrap_mistralrs_router_from_state(
    mistralrs: SharedMistralState,
    include_swagger_routes: bool,
    base_path: Option<&str>,
) -> Result<Router> {
    initialize_logging();

    // if args.interactive_mode {
    //     interactive_mode(mistralrs, args.throughput_log, args.interactive_search).await;
    //     return Ok(());
    // }

    // // Needs to be after the .build call as that is where the daemon waits.
    // let setting_server = if !args.interactive_mode {
    //     let port = args.port.expect("Interactive mode was not specified, so expected port to be specified. Perhaps you forgot `-i` or `--port`?");
    //     let ip = args.serve_ip.unwrap_or_else(|| "0.0.0.0".to_string());

    //     // Create listener early to validate address before model loading
    //     let listener = tokio::net::TcpListener::bind(format!("{ip}:{port}")).await?;
    //     Some((listener, ip, port))
    // } else {
    //     None
    // };

    let app = get_router(mistralrs, include_swagger_routes, base_path);

    Ok(app)
}

pub async fn bootstrap_mistralrs_router_from_args(
    args: Args,
    include_swagger_routes: bool,
    base_path: Option<&str>,
) -> Result<Router> {
    initialize_logging();

    let mistralrs = bootstrap_mistralrs(args).await?;

    // if args.interactive_mode {
    //     interactive_mode(mistralrs, args.throughput_log, args.interactive_search).await;
    //     return Ok(());
    // }

    // // Needs to be after the .build call as that is where the daemon waits.
    // let setting_server = if !args.interactive_mode {
    //     let port = args.port.expect("Interactive mode was not specified, so expected port to be specified. Perhaps you forgot `-i` or `--port`?");
    //     let ip = args.serve_ip.unwrap_or_else(|| "0.0.0.0".to_string());

    //     // Create listener early to validate address before model loading
    //     let listener = tokio::net::TcpListener::bind(format!("{ip}:{port}")).await?;
    //     Some((listener, ip, port))
    // } else {
    //     None
    // };

    let app = get_router(mistralrs, include_swagger_routes, base_path);

    Ok(app)
}

pub async fn bootstrap_mistralrs(mut args: Args) -> Result<SharedMistralState> {
    args = configure_args(args);

    let tgt_non_granular_index = get_tgt_non_granular_index(&args.model);
    let dtype = get_model_dtype(&args.model)?;
    let auto_device_map_params = get_auto_device_map_params(&args.model)?;

    if tgt_non_granular_index.is_some() {
        args.max_seqs = 1;
    }

    let prompt_chunksize = match args.prompt_chunksize {
        Some(0) => {
            anyhow::bail!("`prompt_chunksize` must be a strictly positive integer, got 0.",)
        }
        Some(x) => Some(NonZeroUsize::new(x).unwrap()),
        None => None,
    };

    let max_seq_len = auto_device_map_params.max_seq_len();

    let device = init_device(args.cpu, args.seed)?;
    let mapper = init_mapper(&args.num_device_layers, &auto_device_map_params);
    let no_paged_attn = configure_no_paged_attn(&device, args.no_paged_attn, args.paged_attn);

    // Allocate 0.5 GB of CPU memory just as a placeholder.
    // Nothing happens here as we have no `swap_out`, see `_preempt_by_swap`.
    let cache_config = init_cache_config(
        args.paged_attn_block_size,
        args.paged_attn_gpu_mem,
        args.paged_attn_gpu_mem_usage,
        args.paged_ctxt_len,
        no_paged_attn,
        max_seq_len,
    )?;

    // Configure this last to prevent arg moves
    let loader: Box<dyn Loader> = LoaderBuilder::new(args.model)
        .with_no_kv_cache(args.no_kv_cache)
        .with_chat_template(args.chat_template)
        .with_prompt_chunksize(prompt_chunksize)
        .with_jinja_explicit(args.jinja_explicit)
        .build()?;

    print_mistral_server_info(&loader);

    let pipeline: LoadedPipeline = loader.load_model_from_hf(
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

    let scheduler_config = init_scheduler_config(&cache_config, &pipeline, args.max_seqs).await;

    let bert_model = if args.enable_search {
        Some(
            args.search_bert_model
                .map(BertEmbeddingModel::Custom)
                .unwrap_or_default(),
        )
    } else {
        None
    };

    // Throughput logging in the server
    Ok(build_mistralrs(
        pipeline,
        scheduler_config,
        args.interactive_mode,
        bert_model.clone(),
        args.log,
        args.truncate_sequence,
        args.no_kv_cache,
        args.prefix_cache_n,
    ))
}

// This was originally with the device config
fn configure_args(mut args: Args) -> Args {
    #[cfg(not(feature = "metal"))]
    if args.cpu {
        args.no_paged_attn = true;
    }

    args
}

fn init_device(force_cpu: bool, seed: Option<u64>) -> Result<candle_core::Device> {
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    #[allow(clippy::if_same_then_else)]
    let device = if force_cpu {
        Device::Cpu
    } else if mistralrs_core::distributed::use_nccl() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }

    Ok(device)
}

fn init_mapper(
    num_device_layers: &Option<Vec<String>>,
    auto_device_map_params: &AutoDeviceMapParams,
) -> DeviceMapSetting {
    // Parse device mapper
    if let Some(device_layers) = num_device_layers {
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
        DeviceMapSetting::Auto(auto_device_map_params.clone())
    }
}

#[allow(clippy::borrowed_box)]
fn print_mistral_server_info(loader: &Box<dyn Loader>) {
    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    info!("Sampling method: penalties -> temperature -> topk -> topp -> minp -> multinomial");
    info!("Model kind is: {}", loader.get_kind().to_string());
}

fn configure_no_paged_attn(device: &Device, no_paged_attn: bool, paged_attn: bool) -> bool {
    if device.is_cuda() || mistralrs_core::distributed::use_nccl() {
        no_paged_attn
    } else if device.is_metal() {
        !paged_attn
    } else {
        true
    }
}

fn init_cache_config(
    paged_attn_block_size: Option<usize>,
    paged_attn_gpu_mem: Option<usize>,
    paged_attn_gpu_mem_usage: Option<f32>,
    paged_ctxt_len: Option<usize>,
    no_paged_attn: bool,
    max_seq_len: usize,
) -> Result<Option<PagedAttentionConfig>> {
    match (
        paged_attn_block_size,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_supported(),
        no_paged_attn,
    ) {
        (block_size, None, None, None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::ContextSize(max_seq_len),
        )?)),
        (block_size, None, None, Some(ctxt), true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::ContextSize(ctxt),
        )?)),
        (block_size, None, Some(f), None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::Utilization(f),
        )?)),
        (block_size, Some(m), None, None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            512,
            MemoryGpuConfig::MbAmount(m),
        )?)),
        (block_size, Some(_m), Some(f), None, true, false) => {
            info!("Both memory size, and usage were specified, defaulting to the usage value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                512,
                MemoryGpuConfig::Utilization(f),
            )?))
        }
        (block_size, Some(_m), None, Some(ctxt), true, false) => {
            info!("All memory size and ctxt len, defaulting to the context len value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                512,
                MemoryGpuConfig::ContextSize(ctxt),
            )?))
        }
        (block_size, None, Some(f), Some(_ctxt), true, false) => {
            info!("Both ctxt len and usage were specified, defaulting to the usage value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                512,
                MemoryGpuConfig::Utilization(f),
            )?))
        }
        (_, _, _, _, _, _) => Ok(None),
    }
}

async fn init_scheduler_config(
    cache_config: &Option<PagedAttentionConfig>,
    pipeline: &LoadedPipeline,
    args_max_seqs: usize,
) -> SchedulerConfig {
    if cache_config.is_some() {
        // Handle case where we may have device mapping
        if let Some(ref cache_config) = pipeline.lock().await.get_metadata().cache_config {
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: args_max_seqs,
                config: cache_config.clone(),
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(args_max_seqs.try_into().unwrap()),
            }
        }
    } else {
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(args_max_seqs.try_into().unwrap()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_mistralrs(
    pipeline: LoadedPipeline,
    scheduler_config: SchedulerConfig,
    interactive_mode: bool,
    bert_model: Option<BertEmbeddingModel>,
    log: Option<String>,
    truncate_sequence: bool,
    no_kv_cache: bool,
    prefix_cache_n: usize,
) -> SharedMistralState {
    MistralRsBuilder::new(pipeline, scheduler_config, !interactive_mode, bert_model)
        .with_opt_log(log)
        .with_truncate_sequence(truncate_sequence)
        .with_no_kv_cache(no_kv_cache)
        .with_prefix_cache_n(prefix_cache_n)
        .build()
}

fn get_router(
    state: SharedMistralState,
    include_swagger_routes: bool,
    base_path: Option<&str>,
) -> Router {
    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
        .allow_origin(allow_origin);

    // Use the provided base path or default to ""
    let prefix = base_path.unwrap_or("");

    let mut router = Router::new()
        .route("/v1/chat/completions", post(chatcompletions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/", get(health))
        .route("/re_isq", post(re_isq))
        .route("/v1/images/generations", post(image_generation))
        .layer(cors_layer)
        .layer(DefaultBodyLimit::max(N_INPUT_SIZE * MB_TO_B))
        .with_state(state);

    if include_swagger_routes {
        let doc = get_openapi_doc(None);

        router = router.merge(
            SwaggerUi::new(format!("{prefix}/docs"))
                .url(format!("{prefix}/api-doc/openapi.json"), doc),
        );
    }

    router
}
