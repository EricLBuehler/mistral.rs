use std::{
    error::Error,
    fs::File,
    pin::Pin,
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc,
    },
    task::{Context, Poll},
};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
    routing::post,
    Router,
};
use candle_core::Device;
use clap::Parser;
use indexmap::IndexMap;
use mistralrs_core::{
    GemmaLoader, GemmaSpecificConfig, LlamaLoader, LlamaSpecificConfig, Loader, MistralLoader,
    MistralRs, MistralSpecificConfig, MixtralLoader, MixtralSpecificConfig, ModelKind, Request,
    Response, SamplingParams, SchedulerMethod, StopTokens as InternalStopTokens, TokenSource,
};
use model_selected::ModelSelected;
use openai::{ChatCompletionRequest, StopTokens};
mod model_selected;
mod openai;

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
}

struct Streamer {
    rx: Receiver<Response>,
    is_done: bool,
    state: Arc<MistralRs>,
}

impl futures::Stream for Streamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.is_done {
            return Poll::Ready(None);
        }

        match self.rx.try_recv() {
            Ok(resp) => {
                self.is_done = true;
                match resp {
                    Response::Error(e) => {
                        MistralRs::maybe_log_error(self.state.clone(), &*e);
                        Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                    }
                    Response::Chunk(response) => {
                        MistralRs::maybe_log_response(self.state.clone(), &response);
                        Poll::Ready(Some(Event::default().json_data(response)))
                    }
                    _ => unreachable!(),
                }
            }
            Err(_) => Poll::Pending,
        }
    }
}

enum ChatCompletionResponder {
    Sse(Sse<Streamer>),
    String(String),
    Error(Box<dyn Error>),
}

impl IntoResponse for ChatCompletionResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatCompletionResponder::Sse(s) => s.into_response(),
            ChatCompletionResponder::String(s) => s.into_response(),
            ChatCompletionResponder::Error(e) => e.to_string().into_response(),
        }
    }
}

fn parse_request(
    oairequest: ChatCompletionRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Request {
    let repr = serde_json::to_string(&oairequest).unwrap();
    MistralRs::maybe_log_request(state.clone(), repr);

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
        is_streaming: oairequest.stream,
    }
}

async fn chatcompletions(
    State(state): State<Arc<MistralRs>>,
    Json(oairequest): Json<ChatCompletionRequest>,
) -> ChatCompletionResponder {
    let (tx, rx) = channel();
    let is_streaming = oairequest.stream;
    let request = parse_request(oairequest, state.clone(), tx);
    let sender = state.get_sender();
    sender.send(request).unwrap();

    if is_streaming {
        let streamer = Streamer {
            rx,
            is_done: false,
            state,
        };

        ChatCompletionResponder::Sse(Sse::new(streamer).keep_alive(KeepAlive::new()))
    } else {
        let response = rx.recv().unwrap();

        match response {
            Response::Error(e) => {
                MistralRs::maybe_log_error(state, &*e);
                ChatCompletionResponder::Error(e)
            }
            Response::Done(response) => {
                MistralRs::maybe_log_response(state, &response);
                ChatCompletionResponder::String(serde_json::to_string(&response).unwrap())
            }
            Response::Chunk(_) => unreachable!(),
        }
    }
}

fn get_router(state: Arc<MistralRs>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chatcompletions))
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

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!("Loading model `{}` on {device:?}...", loader.get_id());
    let pipeline = loader.load_model(None, args.token_source, None, &device)?;
    println!("Model loaded.");
    let mistralrs = MistralRs::new(
        pipeline,
        SchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
        args.log,
        args.truncate_sequence,
        args.no_kv_cache,
    );

    let app = get_router(mistralrs);

    let ip = if let Some(ref ip) = args.serve_ip {
        ip.to_string()
    } else {
        "0.0.0.0".to_string()
    };
    let listener = tokio::net::TcpListener::bind(format!("{ip}:{}", args.port)).await?;
    println!("Serving on http://{ip}:{}.", args.port);
    axum::serve(listener, app).await?;

    Ok(())
}
