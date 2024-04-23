#![allow(clippy::too_many_arguments)]

use candle_core::Result;
use either::Either;
use indexmap::IndexMap;
use message::{Message, Role};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::Debug,
    str::FromStr,
    sync::{mpsc::channel, Arc, Mutex},
};
use stream::ChatCompletionStreamer;

use candle_core::Device;
use mistralrs_core::{
    ChatCompletionResponse, CompletionResponse, Constraint, GemmaLoader, GemmaSpecificConfig,
    GgmlLoader, GgmlSpecificConfig, GgufLoader, GgufSpecificConfig, LlamaLoader,
    LlamaSpecificConfig, Loader, MistralLoader, MistralRs, MistralRsBuilder, MistralSpecificConfig,
    MixtralLoader, MixtralSpecificConfig, ModelKind, Phi2Loader, Phi2SpecificConfig,
    Request as _Request, RequestMessage, Response, SamplingParams, SchedulerMethod, StopTokens,
    TokenSource,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyList, PyString},
};
use std::fs::File;
mod stream;
mod which;
use which::Which;
mod message;

#[cfg(not(feature = "metal"))]
static CUDA_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);
#[cfg(feature = "metal")]
static METAL_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);

#[cfg(not(feature = "metal"))]
fn get_device() -> Result<Device> {
    let mut device = CUDA_DEVICE.lock().unwrap();
    if let Some(device) = device.as_ref() {
        return Ok(device.clone());
    };
    let res = Device::cuda_if_available(0)?;
    *device = Some(res.clone());
    Ok(res)
}
#[cfg(feature = "metal")]
fn get_device() -> Result<Device> {
    let mut device = METAL_DEVICE.lock().unwrap();
    if let Some(device) = device.as_ref() {
        return Ok(device.clone());
    };
    let res = Device::new_metal(0)?;
    *device = Some(res.clone());
    Ok(res)
}

#[pyclass]
/// An object wrapping the underlying Rust system to handle requests and process conversations.
struct Runner {
    runner: Arc<MistralRs>,
}

static NEXT_REQUEST_ID: Mutex<RefCell<usize>> = Mutex::new(RefCell::new(0));

#[pymethods]
impl Runner {
    #[new]
    #[pyo3(signature = (which, max_seqs = 16, no_kv_cache = false, prefix_cache_n = 16, token_source = "cache", chat_template = None))]
    fn new(
        which: Which,
        max_seqs: usize,
        no_kv_cache: bool,
        prefix_cache_n: usize,
        token_source: &str,
        chat_template: Option<String>,
    ) -> PyResult<Self> {
        const REPEAT_LAST_N_DEFAULT: usize = 64;
        const GQA_DEFAULT: usize = 1;

        #[cfg(not(feature = "flash-attn"))]
        let use_flash_attn = false;
        #[cfg(feature = "flash-attn")]
        let use_flash_attn = true;

        let tgt_non_granular_index = match which {
            Which::Gemma { .. }
            | Which::Llama { .. }
            | Which::Mistral { .. }
            | Which::Mixtral { .. }
            | Which::Phi2 { .. }
            | Which::LoraMistral { .. }
            | Which::LoraLlama { .. }
            | Which::LoraMixtral { .. }
            | Which::GGUF { .. }
            | Which::LoraGGUF { .. }
            | Which::GGML { .. }
            | Which::LoraGGML { .. } => None,
            Which::XLoraGemma {
                tgt_non_granular_index,
                ..
            }
            | Which::XLoraLlama {
                tgt_non_granular_index,
                ..
            }
            | Which::XLoraMistral {
                tgt_non_granular_index,
                ..
            }
            | Which::XLoraMixtral {
                tgt_non_granular_index,
                ..
            }
            | Which::XLoraPhi2 {
                tgt_non_granular_index,
                ..
            }
            | Which::XLoraGGUF {
                tgt_non_granular_index,
                ..
            }
            | Which::XLoraGGML {
                tgt_non_granular_index,
                ..
            } => tgt_non_granular_index,
        };
        let max_seqs = if tgt_non_granular_index.is_some() {
            1
        } else {
            max_seqs
        };

        let loader: Box<dyn Loader> = match which {
            Which::Mistral {
                model_id,
                repeat_last_n,
                tokenizer_json,
            } => Box::new(MistralLoader::new(
                Some(model_id),
                MistralSpecificConfig {
                    use_flash_attn,
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                None,
                ModelKind::Normal,
                None,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                None,
            )),
            Which::XLoraMistral {
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
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                Some(xlora_model_id),
                ModelKind::XLoraNormal,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::Gemma {
                model_id,
                repeat_last_n,
                tokenizer_json,
            } => Box::new(GemmaLoader::new(
                Some(model_id),
                GemmaSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                None,
                ModelKind::Normal,
                None,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                None,
            )),
            Which::XLoraGemma {
                model_id,
                xlora_model_id,
                repeat_last_n,
                order,
                tokenizer_json,
                tgt_non_granular_index,
            } => Box::new(GemmaLoader::new(
                model_id,
                GemmaSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                Some(xlora_model_id),
                ModelKind::Normal,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::Llama {
                model_id,
                repeat_last_n,
                tokenizer_json,
            } => Box::new(LlamaLoader::new(
                Some(model_id),
                LlamaSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                    use_flash_attn,
                    gqa: 0,
                },
                None,
                None,
                None,
                ModelKind::Normal,
                None,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                None,
            )),
            Which::XLoraLlama {
                model_id,
                xlora_model_id,
                repeat_last_n,
                order,
                tokenizer_json,
                tgt_non_granular_index,
            } => Box::new(LlamaLoader::new(
                model_id,
                LlamaSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                    use_flash_attn,
                    gqa: 0,
                },
                None,
                None,
                Some(xlora_model_id),
                ModelKind::QuantizedGGML,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::Mixtral {
                model_id,
                repeat_last_n,
                tokenizer_json,
            } => Box::new(MixtralLoader::new(
                Some(model_id),
                MixtralSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                    use_flash_attn,
                },
                None,
                None,
                None,
                ModelKind::Normal,
                None,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                None,
            )),
            Which::XLoraMixtral {
                model_id,
                xlora_model_id,
                repeat_last_n,
                order,
                tokenizer_json,
                tgt_non_granular_index,
            } => Box::new(MixtralLoader::new(
                model_id,
                MixtralSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                    use_flash_attn,
                },
                None,
                None,
                Some(xlora_model_id),
                ModelKind::XLoraNormal,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::Phi2 {
                model_id,
                repeat_last_n,
                tokenizer_json,
            } => Box::new(Phi2Loader::new(
                Some(model_id),
                Phi2SpecificConfig {
                    use_flash_attn,
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                None,
                ModelKind::Normal,
                None,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                None,
            )),
            Which::XLoraPhi2 {
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
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                Some(xlora_model_id),
                ModelKind::XLoraNormal,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::LoraMistral {
                model_id,
                tokenizer_json,
                adapters_model_id,
                repeat_last_n,
                order,
            } => Box::new(MistralLoader::new(
                model_id,
                MistralSpecificConfig {
                    use_flash_attn,
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                Some(adapters_model_id),
                ModelKind::LoraNormal,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::LoraMixtral {
                model_id,
                tokenizer_json,
                adapters_model_id,
                repeat_last_n,
                order,
            } => Box::new(MistralLoader::new(
                model_id,
                MistralSpecificConfig {
                    use_flash_attn,
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                Some(adapters_model_id),
                ModelKind::LoraNormal,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::LoraLlama {
                model_id,
                tokenizer_json,
                adapters_model_id,
                repeat_last_n,
                order,
            } => Box::new(MistralLoader::new(
                model_id,
                MistralSpecificConfig {
                    use_flash_attn,
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                None,
                None,
                Some(adapters_model_id),
                ModelKind::LoraNormal,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::GGUF {
                tok_model_id,
                tokenizer_json,
                quantized_model_id,
                quantized_filename,
                repeat_last_n,
            } => Box::new(GgufLoader::new(
                Some(tok_model_id),
                GgufSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                Some(quantized_model_id),
                Some(quantized_filename),
                None,
                ModelKind::QuantizedGGUF,
                None,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                None,
            )),
            Which::XLoraGGUF {
                tok_model_id,
                tokenizer_json,
                quantized_model_id,
                quantized_filename,
                repeat_last_n,
                xlora_model_id,
                order,
                tgt_non_granular_index,
            } => Box::new(GgufLoader::new(
                Some(tok_model_id),
                GgufSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                Some(quantized_model_id),
                Some(quantized_filename),
                Some(xlora_model_id),
                ModelKind::XLoraGGUF,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::LoraGGUF {
                tok_model_id,
                tokenizer_json,
                quantized_model_id,
                quantized_filename,
                repeat_last_n,
                adapters_model_id,
                order,
                tgt_non_granular_index,
            } => Box::new(GgufLoader::new(
                Some(tok_model_id),
                GgufSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                },
                Some(quantized_model_id),
                Some(quantized_filename),
                Some(adapters_model_id),
                ModelKind::LoraGGUF,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::GGML {
                tok_model_id,
                tokenizer_json,
                quantized_model_id,
                quantized_filename,
                repeat_last_n,
                gqa,
            } => Box::new(GgmlLoader::new(
                Some(tok_model_id),
                GgmlSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                    gqa: gqa.unwrap_or(GQA_DEFAULT),
                },
                Some(quantized_model_id),
                Some(quantized_filename),
                None,
                ModelKind::QuantizedGGUF,
                None,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                None,
            )),
            Which::XLoraGGML {
                tok_model_id,
                tokenizer_json,
                quantized_model_id,
                quantized_filename,
                repeat_last_n,
                xlora_model_id,
                order,
                tgt_non_granular_index,
                gqa,
            } => Box::new(GgmlLoader::new(
                Some(tok_model_id),
                GgmlSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                    gqa: gqa.unwrap_or(GQA_DEFAULT),
                },
                Some(quantized_model_id),
                Some(quantized_filename),
                Some(xlora_model_id),
                ModelKind::XLoraGGUF,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
            Which::LoraGGML {
                tok_model_id,
                tokenizer_json,
                quantized_model_id,
                quantized_filename,
                repeat_last_n,
                adapters_model_id,
                order,
                tgt_non_granular_index,
                gqa,
            } => Box::new(GgmlLoader::new(
                Some(tok_model_id),
                GgmlSpecificConfig {
                    repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                    gqa: gqa.unwrap_or(GQA_DEFAULT),
                },
                Some(quantized_model_id),
                Some(quantized_filename),
                Some(adapters_model_id),
                ModelKind::LoraGGUF,
                Some(
                    serde_json::from_reader(
                        File::open(order.clone())
                            .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            )),
        };

        let device = get_device().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pipeline = loader
            .load_model(
                None,
                TokenSource::from_str(token_source)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                None,
                &device,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mistralrs = MistralRsBuilder::new(
            pipeline,
            SchedulerMethod::Fixed(
                max_seqs
                    .try_into()
                    .map_err(|e| PyValueError::new_err(format!("{e:?}")))?,
            ),
        )
        .with_no_kv_cache(no_kv_cache)
        .with_prefix_cache_n(prefix_cache_n)
        .build();

        Ok(Self { runner: mistralrs })
    }

    /// Send an OpenAI API compatible request, returning the result.
    fn send_chat_completion_request(
        &mut self,
        request: Py<ChatCompletionRequest>,
    ) -> PyResult<Either<ChatCompletionResponse, ChatCompletionStreamer>> {
        let (tx, rx) = channel();
        Python::with_gil(|py| {
            let request = request.bind(py).borrow();
            let stop_toks = request
                .stop_seqs
                .as_ref()
                .map(|x| StopTokens::Seqs(x.to_vec()));
            let constraint = if request.grammar_type == Some("regex".to_string()) {
                if request.grammar.is_none() {
                    return Err(PyValueError::new_err(
                        "Grammar type is specified but not grammar text",
                    ));
                }
                Constraint::Regex(request.grammar.as_ref().unwrap().clone())
            } else if request.grammar_type == Some("yacc".to_string()) {
                if request.grammar.is_none() {
                    return Err(PyValueError::new_err(
                        "Grammar type is specified but not grammar text",
                    ));
                }
                Constraint::Yacc(request.grammar.as_ref().unwrap().clone())
            } else if request.grammar_type.is_some() {
                return Err(PyValueError::new_err(
                    "Grammar type is specified but is not `regex` or `yacc`",
                ));
            } else {
                Constraint::None
            };
            let model_request = _Request {
                id: {
                    let l = NEXT_REQUEST_ID.lock().unwrap();
                    let last = &mut *l.borrow_mut();
                    let last_v = *last;
                    *last += 1;
                    last_v
                },
                messages: match request.messages {
                    Either::Left(ref messages) => {
                        let mut messages_vec = Vec::new();
                        for message in messages {
                            let mut message_map = IndexMap::new();
                            let role = match message.role {
                                Role::Assistant => "assistant",
                                Role::User => "user",
                            };
                            message_map.insert("role".to_string(), role.to_string());
                            message_map.insert("content".to_string(), message.content.clone());
                            messages_vec.push(message_map);
                        }
                        RequestMessage::Chat(messages_vec)
                    }
                    Either::Right(ref prompt) => {
                        let mut messages = Vec::new();
                        let mut message_map = IndexMap::new();
                        message_map.insert("role".to_string(), "user".to_string());
                        message_map.insert("content".to_string(), prompt.to_string());
                        messages.push(message_map);
                        RequestMessage::Chat(messages)
                    }
                },
                sampling_params: SamplingParams {
                    temperature: request.temperature,
                    top_k: request.top_k,
                    top_p: request.top_p,
                    top_n_logprobs: request.top_logprobs.unwrap_or(1),
                    frequency_penalty: request.frequency_penalty,
                    presence_penalty: request.presence_penalty,
                    max_len: request.max_tokens,
                    stop_toks,
                    logits_bias: request.logit_bias.clone(),
                    n_choices: request.n_choices,
                },
                response: tx,
                return_logprobs: request.logprobs,
                is_streaming: request.stream,
                constraint,
                suffix: None,
            };

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender();
            sender.send(model_request).unwrap();

            if request.stream {
                Ok(Either::Right(ChatCompletionStreamer::from_rx(rx)))
            } else {
                let response = rx.recv().unwrap();

                match response {
                    Response::ValidationError(e) | Response::InternalError(e) => {
                        Err(PyValueError::new_err(e.to_string()))
                    }
                    Response::Done(response) => Ok(Either::Left(response)),
                    Response::ModelError(msg, _) => Err(PyValueError::new_err(msg.to_string())),
                    Response::Chunk(_) => unreachable!(),
                    Response::CompletionDone(_) => unreachable!(),
                    Response::CompletionModelError(_, _) => unreachable!(),
                }
            }
        })
    }

    /// Send an OpenAI API compatible request, returning the result.
    fn send_completion_request(
        &mut self,
        request: Py<CompletionRequest>,
    ) -> PyResult<CompletionResponse> {
        let (tx, rx) = channel();
        Python::with_gil(|py| {
            let request = request.bind(py).borrow();
            let stop_toks = request
                .stop_seqs
                .as_ref()
                .map(|x| StopTokens::Seqs(x.to_vec()));
            let constraint = if request.grammar_type == Some("regex".to_string()) {
                if request.grammar.is_none() {
                    return Err(PyValueError::new_err(
                        "Grammar type is specified but not grammar text",
                    ));
                }
                Constraint::Regex(request.grammar.as_ref().unwrap().clone())
            } else if request.grammar_type == Some("yacc".to_string()) {
                if request.grammar.is_none() {
                    return Err(PyValueError::new_err(
                        "Grammar type is specified but not grammar text",
                    ));
                }
                Constraint::Yacc(request.grammar.as_ref().unwrap().clone())
            } else if request.grammar_type.is_some() {
                return Err(PyValueError::new_err(
                    "Grammar type is specified but is not `regex` or `yacc`",
                ));
            } else {
                Constraint::None
            };
            let model_request = _Request {
                id: {
                    let l = NEXT_REQUEST_ID.lock().unwrap();
                    let last = &mut *l.borrow_mut();
                    let last_v = *last;
                    *last += 1;
                    last_v
                },
                messages: RequestMessage::Completion {
                    text: request.prompt.clone(),
                    echo_prompt: request.echo_prompt,
                    best_of: request.best_of,
                },
                sampling_params: SamplingParams {
                    temperature: request.temperature,
                    top_k: request.top_k,
                    top_p: request.top_p,
                    top_n_logprobs: 1,
                    frequency_penalty: request.frequency_penalty,
                    presence_penalty: request.presence_penalty,
                    max_len: request.max_tokens,
                    stop_toks,
                    logits_bias: request.logit_bias.clone(),
                    n_choices: request.n_choices,
                },
                response: tx,
                return_logprobs: false,
                is_streaming: false,
                constraint,
                suffix: request.suffix.clone(),
            };

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender();
            sender.send(model_request).unwrap();
            let response = rx.recv().unwrap();

            match response {
                Response::ValidationError(e) | Response::InternalError(e) => {
                    Err(PyValueError::new_err(e.to_string()))
                }
                Response::CompletionDone(response) => Ok(response),
                Response::CompletionModelError(msg, _) => {
                    Err(PyValueError::new_err(msg.to_string()))
                }
                Response::Chunk(_) => unreachable!(),
                Response::Done(_) => unreachable!(),
                Response::ModelError(_, _) => unreachable!(),
            }
        })
    }
}

#[pyclass]
#[derive(Debug)]
/// An OpenAI API compatible completion request.
struct CompletionRequest {
    _model: String,
    prompt: String,
    best_of: usize,
    echo_prompt: bool,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    logit_bias: Option<HashMap<u32, f32>>,
    max_tokens: Option<usize>,
    n_choices: usize,
    stop_seqs: Option<Vec<String>>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    suffix: Option<String>,
    top_k: Option<usize>,
    grammar: Option<String>,
    grammar_type: Option<String>,
}

#[pymethods]
impl CompletionRequest {
    #[new]
    #[pyo3(signature = (
        prompt,
        model,
        best_of = 1,
        echo_prompt = false,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        max_tokens=None,
        n_choices=1,
        stop_seqs=None,
        temperature=None,
        top_p=None,
        suffix=None,
        top_k=None,
        grammar = None,
        grammar_type = None
    ))]
    fn new(
        prompt: String,
        model: String,
        best_of: usize,
        echo_prompt: bool,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        logit_bias: Option<HashMap<u32, f32>>,
        max_tokens: Option<usize>,
        n_choices: usize,
        stop_seqs: Option<Vec<String>>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        suffix: Option<String>,
        top_k: Option<usize>,
        grammar: Option<String>,
        grammar_type: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            prompt,
            best_of,
            echo_prompt,
            suffix,
            _model: model,
            logit_bias,
            max_tokens,
            n_choices,
            presence_penalty,
            frequency_penalty,
            stop_seqs,
            temperature,
            top_p,
            top_k,
            grammar,
            grammar_type,
        })
    }
}

#[pyclass]
#[derive(Debug)]
/// An OpenAI API compatible chat completion request.
struct ChatCompletionRequest {
    messages: Either<Vec<Message>, String>,
    _model: String,
    logit_bias: Option<HashMap<u32, f32>>,
    logprobs: bool,
    top_logprobs: Option<usize>,
    max_tokens: Option<usize>,
    n_choices: usize,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    stop_seqs: Option<Vec<String>>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    stream: bool,
    top_k: Option<usize>,
    grammar: Option<String>,
    grammar_type: Option<String>,
}

#[pymethods]
impl ChatCompletionRequest {
    #[new]
    #[pyo3(signature = (
        messages,
        model,
        logprobs = false,
        n_choices = 1,
        logit_bias = None,
        top_logprobs = None,
        max_tokens = None,
        presence_penalty = None,
        frequency_penalty = None,
        stop_seqs = None,
        temperature = None,
        top_p = None,
        top_k = None,
        stream=false,
        grammar = None,
        grammar_type = None
    ))]
    fn new(
        messages: Py<PyAny>,
        model: String,
        logprobs: bool,
        n_choices: usize,
        logit_bias: Option<HashMap<u32, f32>>,
        top_logprobs: Option<usize>,
        max_tokens: Option<usize>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        stop_seqs: Option<Vec<String>>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        stream: Option<bool>,
        grammar: Option<String>,
        grammar_type: Option<String>,
    ) -> PyResult<Self> {
        let messages = Python::with_gil(|py| {
            if let Ok(messages) = messages.bind(py).downcast_exact::<PyList>() {
                let mut messages_vec = Vec::new();
                for message in messages {
                    messages_vec.push(message.extract::<Message>()?);
                }
                Ok::<Either<Vec<Message>, String>, PyErr>(Either::Left(messages_vec))
            } else if let Ok(messages) = messages.bind(py).downcast_exact::<PyString>() {
                let prompt = messages.extract::<String>()?;
                Ok::<Either<Vec<Message>, String>, PyErr>(Either::Right(prompt))
            } else {
                return Err(PyTypeError::new_err("Expected a string or list of dicts."));
            }
        })?;
        Ok(Self {
            messages,
            _model: model,
            logit_bias,
            logprobs,
            top_logprobs,
            max_tokens,
            n_choices,
            presence_penalty,
            frequency_penalty,
            stop_seqs,
            temperature,
            top_p,
            top_k,
            stream: stream.unwrap_or(false),
            grammar,
            grammar_type,
        })
    }
}

#[pymodule]
fn mistralrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Runner>()?;
    m.add_class::<Which>()?;
    m.add_class::<ChatCompletionRequest>()?;
    m.add_class::<CompletionRequest>()?;
    m.add_class::<Message>()?;
    m.add_class::<Role>()?;
    Ok(())
}
