use std::{fs::File, sync::Arc};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::{self, Method},
    routing::{get, post},
    Router,
};
use candle_core::Device;
use clap::Parser;
use mistralrs_core::{
    GemmaLoader, GemmaSpecificConfig, LlamaLoader, LlamaSpecificConfig, Loader, MistralLoader,
    MistralRs, MistralSpecificConfig, MixtralLoader, MixtralSpecificConfig, ModelKind, Phi2Loader,
    Phi2SpecificConfig, SchedulerMethod, TokenSource,
};
use model_selected::ModelSelected;
use openai::{ChatCompletionRequest, Message, ModelObjects, StopTokens};
mod chat_completion;
mod completions;
use crate::{chat_completion::__path_chatcompletions, completions::completions};

use crate::{chat_completion::chatcompletions, openai::ModelObject};
mod interactive_mode;
mod model_selected;
mod openai;
mod prompt_mode;

use interactive_mode::interactive_mode;
use prompt_mode::prompt_mode;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info, warn};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
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

    /// Run a single prompt. This cannot be used with interactive mode.
    #[clap(long)]
    prompt: Option<String>,

    /// Requires --prompt. Number of prompt completions to run concurrently in prompt mode.
    #[clap(long, default_value_t = 1, requires = "prompt")]
    prompt_concurrency: usize,

    /// Requires --prompt. Number of prompt tokens to generate.
    #[clap(long, default_value_t = 128, requires = "prompt")]
    prompt_max_tokens: usize,
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

    let tgt_non_granular_index = match args.model {
        ModelSelected::Gemma { .. }
        | ModelSelected::Llama { .. }
        | ModelSelected::LlamaGGML { .. }
        | ModelSelected::LlamaGGUF { .. }
        | ModelSelected::Mistral { .. }
        | ModelSelected::MistralGGUF { .. }
        | ModelSelected::Mixtral { .. }
        | ModelSelected::MixtralGGUF { .. }
        | ModelSelected::Phi2 { .. }
        | ModelSelected::XLoraPhi2 { .. }
        | ModelSelected::LoraMistralGGUF { .. }
        | ModelSelected::LoraMistral { .. }
        | ModelSelected::LoraLlama { .. }
        | ModelSelected::LoraLlamaGGML { .. }
        | ModelSelected::LoraLlamaGGUF { .. }
        | ModelSelected::LoraMixtral { .. }
        | ModelSelected::LoraMixtralGGUF { .. } => None,
        ModelSelected::XLoraGemma {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraLlama {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraLlamaGGML {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraLlamaGGUF {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraMistral {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraMistralGGUF {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraMixtral {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraMixtralGGUF {
            tgt_non_granular_index,
            ..
        } => tgt_non_granular_index,
    };
    if tgt_non_granular_index.is_some() {
        args.max_seqs = 1;
    }
    let args = args;

    let loader: Box<dyn Loader> = match args.model {
        ModelSelected::Mistral {
            model_id,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(MistralLoader::new(
            model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::MistralGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(MistralLoader::new(
            tok_model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            quantized_model_id,
            quantized_filename,
            None,
            ModelKind::QuantizedGGUF,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::XLoraMistral {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(MistralLoader::new(
            model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            None,
            None,
            Some(xlora_model_id),
            ModelKind::XLoraNormal,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::Gemma {
            model_id,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(GemmaLoader::new(
            model_id,
            GemmaSpecificConfig { repeat_last_n },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::XLoraGemma {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(GemmaLoader::new(
            model_id,
            GemmaSpecificConfig { repeat_last_n },
            None,
            None,
            Some(xlora_model_id),
            ModelKind::Normal,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::Llama {
            model_id,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(LlamaLoader::new(
            model_id,
            LlamaSpecificConfig {
                repeat_last_n,
                use_flash_attn,
                gqa: 0,
            },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::LlamaGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                repeat_last_n,
                use_flash_attn,
                gqa: 0,
            },
            quantized_model_id,
            quantized_filename,
            None,
            ModelKind::QuantizedGGUF,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::LlamaGGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            gqa,
            tokenizer_json,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                repeat_last_n,
                use_flash_attn,
                gqa,
            },
            quantized_model_id,
            quantized_filename,
            None,
            ModelKind::QuantizedGGML,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::XLoraLlama {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(LlamaLoader::new(
            model_id,
            LlamaSpecificConfig {
                repeat_last_n,
                use_flash_attn,
                gqa: 0,
            },
            None,
            None,
            Some(xlora_model_id),
            ModelKind::QuantizedGGML,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::Mixtral {
            model_id,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(MixtralLoader::new(
            model_id,
            MixtralSpecificConfig {
                repeat_last_n,
                use_flash_attn,
            },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::MixtralGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(MixtralLoader::new(
            tok_model_id,
            MixtralSpecificConfig {
                repeat_last_n,
                use_flash_attn,
            },
            quantized_model_id,
            quantized_filename,
            None,
            ModelKind::QuantizedGGUF,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::XLoraMixtral {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(MixtralLoader::new(
            model_id,
            MixtralSpecificConfig {
                repeat_last_n,
                use_flash_attn,
            },
            None,
            None,
            Some(xlora_model_id),
            ModelKind::XLoraNormal,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::XLoraMistralGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(MistralLoader::new(
            tok_model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            quantized_model_id,
            quantized_filename,
            Some(xlora_model_id),
            ModelKind::XLoraGGUF,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::XLoraMixtralGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(MixtralLoader::new(
            tok_model_id,
            MixtralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            quantized_model_id,
            quantized_filename,
            Some(xlora_model_id),
            ModelKind::XLoraGGUF,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::XLoraLlamaGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                use_flash_attn,
                repeat_last_n,
                gqa: 0,
            },
            quantized_model_id,
            quantized_filename,
            Some(xlora_model_id),
            ModelKind::XLoraGGUF,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::XLoraLlamaGGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            gqa,
            tokenizer_json,
            tgt_non_granular_index,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                use_flash_attn,
                repeat_last_n,
                gqa,
            },
            quantized_model_id,
            quantized_filename,
            Some(xlora_model_id),
            ModelKind::XLoraGGML,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::Phi2 {
            model_id,
            repeat_last_n,
            tokenizer_json,
        } => Box::new(Phi2Loader::new(
            model_id,
            Phi2SpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            None,
        )),
        ModelSelected::XLoraPhi2 {
            model_id,
            tokenizer_json,
            xlora_model_id,
            repeat_last_n,
            order,
            tgt_non_granular_index,
        } => Box::new(Phi2Loader::new(
            model_id,
            Phi2SpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            None,
            None,
            Some(xlora_model_id),
            ModelKind::XLoraNormal,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::LoraMistralGGUF {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            repeat_last_n,
            order,
        } => Box::new(MistralLoader::new(
            tok_model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            quantized_model_id,
            quantized_filename,
            Some(adapters_model_id),
            ModelKind::LoraGGUF,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::LoraMistral {
            model_id,
            tokenizer_json,
            adapters_model_id,
            repeat_last_n,
            order,
        } => Box::new(MistralLoader::new(
            model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            None,
            None,
            Some(adapters_model_id),
            ModelKind::LoraNormal,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::LoraMixtral {
            model_id,
            tokenizer_json,
            adapters_model_id,
            repeat_last_n,
            order,
        } => Box::new(MistralLoader::new(
            model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            None,
            None,
            Some(adapters_model_id),
            ModelKind::LoraNormal,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::LoraMixtralGGUF {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            repeat_last_n,
            order,
        } => Box::new(MistralLoader::new(
            tok_model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            quantized_model_id,
            quantized_filename,
            Some(adapters_model_id),
            ModelKind::LoraGGUF,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::LoraLlama {
            model_id,
            tokenizer_json,
            adapters_model_id,
            repeat_last_n,
            order,
        } => Box::new(MistralLoader::new(
            model_id,
            MistralSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            None,
            None,
            Some(adapters_model_id),
            ModelKind::LoraNormal,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::LoraLlamaGGUF {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            repeat_last_n,
            order,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                use_flash_attn,
                repeat_last_n,
                gqa: 0,
            },
            quantized_model_id,
            quantized_filename,
            Some(adapters_model_id),
            ModelKind::LoraGGUF,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
        ModelSelected::LoraLlamaGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            repeat_last_n,
            order,
            gqa,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                use_flash_attn,
                repeat_last_n,
                gqa,
            },
            quantized_model_id,
            quantized_filename,
            Some(adapters_model_id),
            ModelKind::LoraGGML,
            Some(serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
            tgt_non_granular_index,
        )),
    };

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
    let pipeline = loader.load_model(None, args.token_source, None, &device)?;
    info!("Model loaded.");

    let mistralrs = MistralRs::new(
        pipeline,
        SchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
        args.log,
        args.truncate_sequence,
        args.no_kv_cache,
        false,
        args.prefix_cache_n,
    );

    if let Some(prompt) = args.prompt {
        prompt_mode(
            mistralrs,
            prompt,
            args.prompt_concurrency,
            args.prompt_max_tokens,
        );
        return Ok(());
    }
    if args.interactive_mode {
        interactive_mode(mistralrs);
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
