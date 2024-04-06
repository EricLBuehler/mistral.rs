use std::{
    env,
    error::Error,
    fs::File,
    pin::Pin,
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use aici::iface::AsyncCmdChannel;
use aici_abi::toktree::TokTrie;
use aicirt::{
    api::{AuthInfo, GetTagsResp, MkModuleReq, MkModuleResp, SetTagsReq},
    bintokens::find_tokenizer,
};
use anyhow::Result;
use axum::{
    body::Bytes,
    extract::{Json, State},
    http::HeaderMap,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
    routing::{get, post},
    Router,
};
use base64::Engine;
use candle_core::Device;
use clap::Parser;
use indexmap::IndexMap;
use mistralrs_core::{
    ChatCompletionResponse, GemmaLoader, GemmaSpecificConfig, LlamaLoader, LlamaSpecificConfig,
    Loader, MistralLoader, MistralRs, MistralSpecificConfig, MixtralLoader, MixtralSpecificConfig,
    ModelKind, Request, Response, SamplingParams, SchedulerMethod,
    StopTokens as InternalStopTokens, TokenSource,
};
use model_selected::ModelSelected;
use openai::{ChatCompletionRequest, Message, ModelObjects, StopTokens};

use crate::{
    aici::iface::{self, AiciRtIface},
    openai::ModelObject,
};
mod aici;
mod model_selected;
mod openai;

use tracing::{info, warn};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}

struct ServerState {
    mistralrs: Arc<MistralRs>,
    side_cmd_ch: AsyncCmdChannel,
}

impl ServerState {
    fn new(mistralrs: Arc<MistralRs>, side_cmd_ch: AsyncCmdChannel) -> Self {
        Self {
            mistralrs,
            side_cmd_ch,
        }
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
    port: String,

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
    /// Can be in the formats: "literal:<value>", "env:<value>", "path:<value>", or "cache" to use a cached token.
    /// Defaults to using a cached token.
    #[arg(long, default_value_t = TokenSource::CacheToken, value_parser = parse_token_source)]
    token_source: TokenSource,

    // TODO: below are only used for aici
    /// Path to the aicirt binary.
    #[arg(long, help_heading = "AICI settings")]
    pub aicirt: Option<String>,

    /// Size of JSON comm buffer in megabytes
    #[arg(long, default_value = "128", help_heading = "AICI settings")]
    json_size: usize,

    /// Size of binary comm buffer in megabytes
    #[arg(long, default_value = "32", help_heading = "AICI settings")]
    bin_size: usize,

    /// How many milliseconds to spin-wait for a message over IPC and SHM.
    #[arg(long, default_value = "200", help_heading = "AICI settings")]
    busy_wait_time: u64,

    /// Shm/semaphore name prefix; default /aici-PORT-
    #[arg(long, help_heading = "AICI settings")]
    shm_prefix: Option<String>,

    /// Pass additional option to aicirt
    #[arg(long, short = 'A', help_heading = "AICI settings")]
    aicirt_arg: Vec<String>,
}

struct Streamer {
    rx: Receiver<Response>,
    is_done: bool,
    state: Arc<ServerState>,
}

impl futures::Stream for Streamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_done {
            return Poll::Ready(None);
        }
        match self.rx.try_recv() {
            Ok(resp) => match resp {
                Response::Error(e) => {
                    MistralRs::maybe_log_error(self.state.mistralrs.clone(), &*e);
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                }
                Response::Chunk(response) => {
                    if response.choices.iter().all(|x| x.stopreason.is_some()) {
                        self.is_done = true;
                    }
                    MistralRs::maybe_log_response(self.state.mistralrs.clone(), &response);
                    Poll::Ready(Some(Event::default().json_data(response)))
                }
                _ => unreachable!(),
            },
            Err(_) => Poll::Pending,
        }
    }
}

enum ChatCompletionResponder {
    Sse(Sse<Streamer>),
    Json(ChatCompletionResponse),
    Error(Box<dyn Error>),
}

impl IntoResponse for ChatCompletionResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatCompletionResponder::Sse(s) => s.into_response(),
            ChatCompletionResponder::Json(s) => Json(s).into_response(),
            ChatCompletionResponder::Error(e) => e.to_string().into_response(),
        }
    }
}

fn parse_request(
    oairequest: ChatCompletionRequest,
    state: Arc<ServerState>,
    tx: Sender<Response>,
) -> Request {
    let repr = serde_json::to_string(&oairequest).unwrap();
    MistralRs::maybe_log_request(state.mistralrs.clone(), repr);

    let stop_toks = match oairequest.stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        Some(StopTokens::MultiId(m)) => Some(InternalStopTokens::Ids(m)),
        Some(StopTokens::SingleId(s)) => Some(InternalStopTokens::Ids(vec![s])),
        None => None,
    };
    let mut messages = Vec::new();
    for message in oairequest.messages {
        let mut message_map = IndexMap::new();
        message_map.insert("role".to_string(), message.role);
        message_map.insert("content".to_string(), message.content);
        messages.push(message_map);
    }

    Request {
        id: state.mistralrs.next_request_id(),
        messages,
        sampling_params: SamplingParams {
            temperature: oairequest.temperature,
            top_k: oairequest.top_k,
            top_p: oairequest.top_p,
            top_n_logprobs: oairequest.top_logprobs.unwrap_or(1),
            repeat_penalty: oairequest.repetition_penalty,
            presence_penalty: oairequest.presence_penalty,
            max_len: oairequest.max_tokens,
            stop_toks,
            logits_bias: oairequest.logit_bias,
            n_choices: oairequest.n_choices,
        },
        response: tx,
        return_logprobs: oairequest.logprobs,
        is_streaming: oairequest.stream.unwrap_or(false),
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
async fn chatcompletions(
    State(state): State<Arc<ServerState>>,
    Json(oairequest): Json<ChatCompletionRequest>,
) -> ChatCompletionResponder {
    let (tx, rx) = channel();
    let request = parse_request(oairequest, state.clone(), tx);
    let is_streaming = request.is_streaming;
    let sender = state.mistralrs.get_sender();
    sender.send(request).unwrap();

    if is_streaming {
        let streamer = Streamer {
            rx,
            is_done: false,
            state,
        };

        ChatCompletionResponder::Sse(
            Sse::new(streamer).keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(1000))
                            .unwrap_or(1000),
                    ))
                    .text("keep-alive-text"),
            ),
        )
    } else {
        let response = rx.recv().unwrap();

        match response {
            Response::Error(e) => {
                MistralRs::maybe_log_error(state.mistralrs.clone(), &*e);
                ChatCompletionResponder::Error(e)
            }
            Response::Done(response) => {
                MistralRs::maybe_log_response(state.mistralrs.clone(), &response);
                ChatCompletionResponder::Json(response)
            }
            Response::Chunk(_) => unreachable!(),
        }
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/models",
    responses((status = 200, description = "Served model info", body = ModelObjects))
)]
async fn models(State(state): State<Arc<ServerState>>) -> Json<ModelObjects> {
    Json(ModelObjects {
        object: "list",
        data: vec![ModelObject {
            id: state.mistralrs.get_id(),
            object: "model",
            created: state.mistralrs.get_creation_time(),
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

fn auth_info(headers: HeaderMap) -> AuthInfo {
    // we default to localhost/admin when no headers given
    let user = headers
        .get("x-user-id")
        .map_or("localhost", |v| v.to_str().unwrap_or("(invalid header)"));
    let role = headers
        .get("x-user-role")
        .map_or("admin", |v| v.to_str().unwrap_or("(invalid header)"));
    let is_admin = role == "admin";
    AuthInfo {
        user: user.to_string(),
        is_admin,
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/controllers/tags",
    responses((status = 200, description = "Get controller tags", body=GetTagsResp))
)]
async fn get_controllers_tags(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Json<GetTagsResp> {
    let r = state.side_cmd_ch.get_tags(auth_info(headers)).await;
    Json(r.unwrap()) // TODO: handle error
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/controllers/tags",
    responses((status = 200, description = "Create controller tags", body=GetTagsResp))
)]
async fn tag_controller(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(body): Json<SetTagsReq>,
) -> Json<GetTagsResp> {
    let r = state.side_cmd_ch.set_tags(body, auth_info(headers)).await;
    Json(r.unwrap()) // TODO: handle error
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/controllers",
    responses((status = 200, description = "Upload controller", body=MkModuleResp))
)]
async fn upload_controller(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Json<MkModuleResp> {
    let binary = base64::engine::general_purpose::STANDARD.encode(body);
    let r = state
        .side_cmd_ch
        .mk_module(MkModuleReq { binary }, auth_info(headers))
        .await;
    Json(r.unwrap()) // TODO: handle error
}

fn get_router(state: Arc<ServerState>) -> Router {
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
    Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", doc))
        .route("/v1/chat/completions", post(chatcompletions))
        .route("/v1/models", get(models))
        .route("/controllers/tags", get(get_controllers_tags))
        .route("/controllers/tags", post(tag_controller))
        .route("/controllers", post(upload_controller))
        .route("/health", get(health))
        .route("/", get(health))
        .with_state(state)
}

fn guess_aicirt() -> Result<String> {
    let mut path = std::env::current_exe()?;
    path.pop();
    path.push("aicirt");
    if path.to_str().is_some() && path.exists() {
        Ok(path.to_str().unwrap().to_string())
    } else {
        Err(anyhow::anyhow!(
            "can't find aicirt binary (tried {:?})",
            path
        ))
    }
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
        | ModelSelected::MixtralGGUF { .. } => None,
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
            Some(serde_json::from_reader(File::open(order)?)?),
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
    );

    let aicirt = match &args.aicirt {
        Some(v) => v.clone(),
        None => match guess_aicirt() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("can't find aicirt; specify with --aicirt=PATH\n{e}");
                std::process::exit(10);
            }
        },
    };

    let shm_prefix = match &args.shm_prefix {
        Some(v) => {
            if v.starts_with("/") {
                v.clone()
            } else {
                format!("/{}", v)
            }
        }
        None => format!("/aici-{}-", args.port),
    };

    // TODO: do not hardcode mistral
    let tokenizer = find_tokenizer("mistralai/Mixtral-8x7B-Instruct-v0.1").unwrap();
    let tokens = tokenizer.token_bytes();

    let tok_trie = TokTrie::from(&tokenizer.tokrx_info(), &tokens);

    let rt_args = iface::Args {
        aicirt,
        tokenizer: "mistral".to_string(), // TODO: do not hardcode mistral
        json_size: args.json_size,
        bin_size: args.bin_size,
        shm_prefix,
        busy_wait_time: args.busy_wait_time,
        add_args: args.aicirt_arg.clone(),
    };

    let iface = AiciRtIface::start_aicirt(&rt_args, &tok_trie).expect("failed to start aicirt");

    let side_cmd_ch = iface.side_cmd.clone();

    let app = get_router(ServerState::new(mistralrs, side_cmd_ch).into());

    let ip = if let Some(ref ip) = args.serve_ip {
        ip.to_string()
    } else {
        "0.0.0.0".to_string()
    };
    let listener = tokio::net::TcpListener::bind(format!("{ip}:{}", args.port)).await?;
    info!("Serving on http://{ip}:{}.", args.port);
    axum::serve(listener, app).await?;

    Ok(())
}
