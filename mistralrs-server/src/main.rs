use std::{
    collections::HashMap,
    fs::File,
    sync::{mpsc::channel, Arc},
};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    routing::post,
    Router,
};
use candle_core::Device;
use clap::{Parser, Subcommand};
use mistralrs_core::{
    GemmaLoader, GemmaSpecificConfig, LlamaLoader, LlamaSpecificConfig, Loader, MistralLoader,
    MistralRs, MistralSpecificConfig, MixtralLoader, MixtralSpecificConfig, ModelKind, Request,
    Response, SamplingParams, SchedulerMethod, StopTokens as InternalStopTokens, TokenSource,
};
use openai::{ChatCompletionRequest, StopTokens};
mod openai;

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the mistral model.
    Mistral {
        /// Model ID to load from
        #[arg(short, long, default_value = "mistralai/Mistral-7B-Instruct-v0.1")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized mistral model with gguf.
    MistralGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "mistralai/Mistral-7B-Instruct-v0.1")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(
            short = 'm',
            long,
            default_value = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        )]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the mistral model, with X-LoRA.
    XLoraMistral {
        /// Model ID to load from
        #[arg(short, long, default_value = "HuggingFaceH4/zephyr-7b-beta")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load Xlora from
        #[arg(short, long, default_value = "lamm-mit/x-lora")]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the gemma model.
    Gemma {
        /// Model ID to load from
        #[arg(short, long, default_value = "google/gemma-7b-it")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the gemma model, with X-LoRA.
    XLoraGemma {
        /// Model ID to load from
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load Xlora from
        #[arg(short, long)]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the llama model.
    Llama {
        /// Model ID to load from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized llama model with gguf.
    LlamaGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "llama-2-13b-chat.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized llama model with gguf.
    LlamaGGML {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGML file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGML")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "llama-2-13b-chat.ggmlv3.q4_K_M.bin"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// GQA
        #[arg(long, default_value_t = 1)]
        gqa: usize,
    },

    /// Select the llama model, with X-LoRA.
    XLoraLlama {
        /// Model ID to load from
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load Xlora from
        #[arg(short, long)]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the mixtral model.
    Mixtral {
        /// Model ID to load from
        #[arg(short, long, default_value = "mistralai/Mixtral-8x7B-Instruct-v0.1")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized mixtral model with gguf.
    MixtralGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "mistralai/Mixtral-8x7B-Instruct-v0.1")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Mixtral-8x7B-v0.1-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "mixtral-8x7b-v0.1.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the mixtral model, with X-LoRA.
    XLoraMixtral {
        /// Model ID to load from
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load Xlora from
        #[arg(short, long)]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraMistralGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "HuggingFaceH4/zephyr-7b-beta")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/zephyr-7B-beta-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "zephyr-7b-beta.Q8_0.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load Xlora from
        #[arg(short, long, default_value = "lamm-mit/x-lora")]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraLlamaGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "llama-2-13b-chat.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load Xlora from
        #[arg(short, long)]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraLlamaGGML {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGML file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGML")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "llama-2-13b-chat.ggmlv3.q4_K_M.bin"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load Xlora from
        #[arg(short, long)]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// GQA
        #[arg(long, default_value_t = 1)]
        gqa: usize,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraMixtralGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "mistralai/Mixtral-8x7B-Instruct-v0.1")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Mixtral-8x7B-v0.1-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "mixtral-8x7b-v0.1.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load Xlora from
        #[arg(short, long)]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
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

    /// Maximum running sequences at any time
    #[arg(long, default_value_t = 2)]
    max_seqs: usize,

    /// Use no KV cache.
    #[arg(long, default_value_t = false)]
    no_kv_cache: bool,

    /// JINJA chat template with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    /// Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
    #[arg(short, long)]
    chat_template: Option<String>,
}

async fn chatcompletions(
    State(state): State<Arc<MistralRs>>,
    Json(oairequest): Json<ChatCompletionRequest>,
) -> String {
    let (tx, rx) = channel();
    let repr = serde_json::to_string(&oairequest).unwrap();
    let stop_toks = match oairequest.stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        Some(StopTokens::MultiId(m)) => Some(InternalStopTokens::Ids(m)),
        Some(StopTokens::SingleId(s)) => Some(InternalStopTokens::Ids(vec![s])),
        None => None,
    };
    let mut messages = Vec::new();
    for message in oairequest.messages {
        let mut message_map = HashMap::new();
        message_map.insert("role".to_string(), message.role);
        message_map.insert("content".to_string(), message.content);
        messages.push(message_map);
    }
    let request = Request {
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
        },
        response: tx,
        return_logprobs: oairequest.logprobs,
    };

    MistralRs::maybe_log_request(state.clone(), repr);
    let sender = state.get_sender();
    sender.send(request).unwrap();
    let response = rx.recv().unwrap();

    match response {
        Response::Error(e) => {
            dbg!(&e);
            e.to_string()
        }
        Response::Done(response) => {
            MistralRs::maybe_log_response(state, &response);
            serde_json::to_string(&response).unwrap()
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
    let args = Args::parse();

    #[cfg(not(feature = "flash-attn"))]
    let use_flash_attn = false;
    #[cfg(feature = "flash-attn")]
    let use_flash_attn = true;

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
        )),
        ModelSelected::XLoraMistral {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
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
        )),
        ModelSelected::XLoraGemma {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
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
        )),
        ModelSelected::XLoraLlama {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
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
            Some(xlora_model_id),
            ModelKind::QuantizedGGML,
            Some(serde_json::from_reader(File::open(order)?)?),
            args.no_kv_cache,
            args.chat_template,
            tokenizer_json,
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
        )),
        ModelSelected::XLoraMixtral {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
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
        )),
        ModelSelected::XLoraMistralGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tokenizer_json,
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
        )),
        ModelSelected::XLoraMixtralGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tokenizer_json,
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
        )),
        ModelSelected::XLoraLlamaGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tokenizer_json,
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
        )),
    };

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::cuda_if_available(0)?;

    let pipeline = loader.load_model(None, TokenSource::CacheToken, None, &device)?;
    let mistralrs = MistralRs::new(
        pipeline,
        SchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
        args.log,
        args.truncate_sequence,
        args.no_kv_cache,
    );

    let app = get_router(mistralrs);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    eprintln!("Serving on http://127.0.0.1:{}.", args.port);
    axum::serve(listener, app).await?;

    Ok(())
}
