#![allow(clippy::too_many_arguments)]

use base64::{engine::general_purpose, Engine};
use candle_core::{quantized::GgmlDType, Result};
use either::Either;
use indexmap::IndexMap;
use reqwest::StatusCode;
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::Debug,
    str::FromStr,
    sync::{Arc, Mutex},
};
use stream::ChatCompletionStreamer;
use tokio::sync::mpsc::channel;

use candle_core::Device;
use mistralrs_core::{
    ChatCompletionResponse, CompletionResponse, Constraint, DeviceMapMetadata, GGMLLoaderBuilder,
    GGMLSpecificConfig, GGUFLoaderBuilder, GGUFSpecificConfig, Loader, MistralRs, MistralRsBuilder,
    ModelDType, NormalLoaderBuilder, NormalRequest, NormalSpecificConfig, Request as _Request,
    RequestMessage, Response, SamplingParams, SchedulerMethod, SpeculativeConfig,
    SpeculativeLoader, StopTokens, TokenSource, VisionLoaderBuilder, VisionSpecificConfig,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyList, PyString},
};
use std::fs::File;
mod stream;
mod which;
use which::{Architecture, VisionArchitecture, Which};

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

fn parse_isq(s: &str) -> std::result::Result<GgmlDType, String> {
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
        _ => Err(format!("GGML type {s} unknown")),
    }
}

#[pyclass]
/// An object wrapping the underlying Rust system to handle requests and process conversations.
struct Runner {
    runner: Arc<MistralRs>,
}

static NEXT_REQUEST_ID: Mutex<RefCell<usize>> = Mutex::new(RefCell::new(0));

fn parse_which(
    which: Which,
    no_kv_cache: bool,
    chat_template: Option<String>,
) -> PyResult<Box<dyn Loader>> {
    const REPEAT_LAST_N_DEFAULT: usize = 64;
    const GQA_DEFAULT: usize = 1;

    #[cfg(not(feature = "flash-attn"))]
    let use_flash_attn = false;
    #[cfg(feature = "flash-attn")]
    let use_flash_attn = true;

    Ok(match which {
        Which::Plain {
            model_id,
            repeat_last_n,
            tokenizer_json,
            arch,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
            },
            chat_template,
            tokenizer_json,
            Some(model_id),
        )
        .build(arch.into()),
        Which::XLora {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
            tgt_non_granular_index,
            arch,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
            },
            chat_template,
            tokenizer_json,
            model_id,
        )
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
            no_kv_cache,
            tgt_non_granular_index,
        )
        .build(arch.into()),
        Which::Lora {
            model_id,
            tokenizer_json,
            adapters_model_id,
            repeat_last_n,
            order,
            arch,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
            },
            chat_template,
            tokenizer_json,
            model_id,
        )
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
        .build(arch.into()),
        Which::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig {
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
            },
            chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .build(),
        Which::XLoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tgt_non_granular_index,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig {
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
            },
            chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
            no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        Which::LoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            adapters_model_id,
            order,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig {
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
            },
            chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
        .build(),
        Which::GGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            gqa,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                gqa: gqa.unwrap_or(GQA_DEFAULT),
            },
            chat_template,
            tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
        )
        .build(),
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
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                gqa: gqa.unwrap_or(GQA_DEFAULT),
            },
            chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
            no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        Which::LoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            adapters_model_id,
            order,
            gqa,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
                gqa: gqa.unwrap_or(GQA_DEFAULT),
            },
            chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
        .build(),
        Which::VisionPlain {
            model_id,
            repeat_last_n,
            tokenizer_json,
            arch,
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
                use_flash_attn,
                repeat_last_n: repeat_last_n.unwrap_or(REPEAT_LAST_N_DEFAULT),
            },
            chat_template,
            tokenizer_json,
            Some(model_id),
        )
        .build(arch.into()),
    })
}

#[pymethods]
impl Runner {
    #[new]
    #[pyo3(signature = (
        which,
        max_seqs = 16,
        no_kv_cache = false,
        prefix_cache_n = 16,
        token_source = "cache",
        speculative_gamma = 32,
        which_draft = None,
        chat_template = None,
        num_device_layers = None,
        in_situ_quant = None
    ))]
    fn new(
        which: Which,
        max_seqs: usize,
        no_kv_cache: bool,
        prefix_cache_n: usize,
        token_source: &str,
        speculative_gamma: usize,
        which_draft: Option<Which>,
        chat_template: Option<String>,
        num_device_layers: Option<usize>,
        in_situ_quant: Option<String>,
    ) -> PyResult<Self> {
        let tgt_non_granular_index = match which {
            Which::Plain { .. }
            | Which::Lora { .. }
            | Which::GGUF { .. }
            | Which::LoraGGUF { .. }
            | Which::GGML { .. }
            | Which::LoraGGML { .. }
            | Which::VisionPlain { .. } => None,
            Which::XLora {
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

        let loader = parse_which(which, no_kv_cache, chat_template.clone())?;
        let loader = if let Some(draft_which) = which_draft {
            let draft = parse_which(draft_which, no_kv_cache, chat_template)?;
            Box::new(SpeculativeLoader {
                target: loader,
                draft,
                config: SpeculativeConfig {
                    gamma: speculative_gamma,
                },
            })
        } else {
            loader
        };

        let device = get_device().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let isq = if let Some(isq) = in_situ_quant {
            Some(parse_isq(&isq).map_err(|e| PyValueError::new_err(e.to_string()))?)
        } else {
            None
        };
        let pipeline = loader
            .load_model_from_hf(
                None,
                TokenSource::from_str(token_source)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                &ModelDType::Auto,
                &device,
                true, // Silent for jupyter
                num_device_layers
                    .map(DeviceMapMetadata::from_num_device_layers)
                    .unwrap_or(DeviceMapMetadata::dummy()),
                isq,
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
        let (tx, mut rx) = channel(10_000);
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
            let model_request = _Request::Normal(NormalRequest {
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
                        let mut image_urls = Vec::new();
                        for message in messages {
                            match &message["content"] {
                                Either::Left(content) => {
                                    let mut message_map: IndexMap<
                                        String,
                                        Either<String, Vec<IndexMap<String, String>>>,
                                    > = IndexMap::new();
                                    message_map.insert(
                                        "role".to_string(),
                                        Either::Left(
                                            message["role"].as_ref().left().unwrap().clone(),
                                        ),
                                    );
                                    message_map.insert(
                                        "content".to_string(),
                                        Either::Left(content.to_string()),
                                    );
                                    messages_vec.push(message_map);
                                }
                                Either::Right(image_messages) => {
                                    if image_messages.len() != 2 {
                                        return Err(PyValueError::new_err(
                                        "Expected 2 items for the content of a message with an image."
                                    .to_string()));
                                    }
                                    if message["role"].as_ref().left().unwrap() != "user" {
                                        return Err(PyValueError::new_err(format!(
                                        "Role for an image message must be `user`, but it is {}",
                                        &message["role"].as_ref().left().unwrap()
                                    )));
                                    }

                                    let mut items = Vec::new();
                                    for image_message in image_messages {
                                        if image_message.len() != 2 {
                                            return Err(PyValueError::new_err("Expected 2 items for the sub-content of a message with an image.".to_string()));
                                        }
                                        if !image_message.contains_key("type") {
                                            return Err(PyValueError::new_err(
                                                "Expected `type` key in input message.".to_string(),
                                            ));
                                        }
                                        if image_message["type"].is_right() {
                                            return Err(PyValueError::new_err(
                                                "Expected string value in `type`.".to_string(),
                                            ));
                                        }
                                        items.push(
                                            image_message["type"].as_ref().unwrap_left().clone(),
                                        )
                                    }

                                    #[allow(clippy::type_complexity)]
                                    fn get_content_and_url(
                                        text_idx: usize,
                                        url_idx: usize,
                                        image_messages: &[HashMap<
                                            String,
                                            Either<String, HashMap<String, String>>,
                                        >],
                                    ) -> PyResult<(String, String)>
                                    {
                                        if image_messages[text_idx]["text"].is_right() {
                                            return Err(PyValueError::new_err(
                                                "Expected string value in `text`.".to_string(),
                                            ));
                                        }
                                        let content = image_messages[text_idx]["text"]
                                            .as_ref()
                                            .unwrap_left()
                                            .clone();
                                        if image_messages[url_idx]["image_url"].is_left()
                                            || !image_messages[url_idx]["image_url"]
                                                .as_ref()
                                                .unwrap_right()
                                                .contains_key("url")
                                        {
                                            return Err(PyValueError::new_err("Expected content of format {{`type`: `text`, `text`: ...}} and {{`type`: `url`, `image_url`: {{`url`: ...}}}}".to_string()));
                                        }
                                        let url = image_messages[url_idx]["image_url"]
                                            .as_ref()
                                            .unwrap_right()["url"]
                                            .clone();
                                        Ok((content, url))
                                    }
                                    let mut message_map: IndexMap<
                                        String,
                                        Either<String, Vec<IndexMap<String, String>>>,
                                    > = IndexMap::new();
                                    message_map.insert(
                                        "role".to_string(),
                                        Either::Left(
                                            message["role"].as_ref().left().unwrap().clone(),
                                        ),
                                    );
                                    let (content, url) = if items[0] == "text" {
                                        get_content_and_url(0, 1, image_messages)?
                                    } else {
                                        get_content_and_url(1, 0, image_messages)?
                                    };

                                    let mut content_map = Vec::new();
                                    let mut content_image_map = IndexMap::new();
                                    content_image_map
                                        .insert("type".to_string(), "image".to_string());
                                    content_map.push(content_image_map);
                                    let mut content_text_map = IndexMap::new();
                                    content_text_map.insert("type".to_string(), "text".to_string());
                                    content_text_map.insert("text".to_string(), content);
                                    content_map.push(content_text_map);

                                    message_map
                                        .insert("content".to_string(), Either::Right(content_map));
                                    messages_vec.push(message_map);
                                    image_urls.push(url);
                                }
                            }
                        }
                        if !image_urls.is_empty() {
                            let mut images = Vec::new();
                            for url in image_urls {
                                let bytes = match reqwest::blocking::get(url.clone()) {
                                    Ok(http_resp) => http_resp
                                        .bytes()
                                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                                        .to_vec(),
                                    Err(e) => {
                                        if e.status()
                                            .is_some_and(|code| code == StatusCode::NOT_FOUND)
                                        {
                                            general_purpose::STANDARD
                                                .decode(url)
                                                .map_err(|e| PyValueError::new_err(e.to_string()))?
                                        } else {
                                            return Err(PyValueError::new_err(e.to_string()));
                                        }
                                    }
                                };
                                images.push(
                                    image::load_from_memory(&bytes)
                                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                                );
                            }
                            RequestMessage::VisionChat {
                                messages: messages_vec,
                                images,
                            }
                        } else {
                            RequestMessage::Chat(messages_vec)
                        }
                    }
                    Either::Right(ref prompt) => {
                        let mut messages = Vec::new();
                        let mut message_map: IndexMap<
                            String,
                            Either<String, Vec<IndexMap<String, String>>>,
                        > = IndexMap::new();
                        message_map.insert("role".to_string(), Either::Left("user".to_string()));
                        message_map.insert("content".to_string(), Either::Left(prompt.to_string()));
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
                adapters: request.adapters.clone(),
            });

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender();
            sender.blocking_send(model_request).unwrap();

            if request.stream {
                Ok(Either::Right(ChatCompletionStreamer::from_rx(rx)))
            } else {
                let response = rx.blocking_recv().unwrap();

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
        let (tx, mut rx) = channel(10_000);
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
            let model_request = _Request::Normal(NormalRequest {
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
                adapters: request.adapters.clone(),
            });

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender();
            sender.blocking_send(model_request).unwrap();
            let response = rx.blocking_recv().unwrap();

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

    /// Send a request to re-ISQ the model. If the model was loaded as GGUF or GGML
    /// then nothing will happen.
    fn send_re_isq(&self, dtype: String) -> PyResult<()> {
        let request =
            _Request::ReIsq(parse_isq(&dtype).map_err(|e| PyValueError::new_err(e.to_string()))?);
        self.runner.get_sender().blocking_send(request).unwrap();
        Ok(())
    }

    /// Send a request to make the specified adapters the active adapters for the model.
    fn activate_adapters(&self, adapter_names: Vec<String>) {
        let request = _Request::ActivateAdapters(adapter_names);
        self.runner.get_sender().blocking_send(request).unwrap();
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
    adapters: Option<Vec<String>>,
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
        grammar_type = None,
        adapters = None
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
        adapters: Option<Vec<String>>,
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
            adapters,
        })
    }
}

#[pyclass]
#[derive(Debug)]
/// An OpenAI API compatible chat completion request.
struct ChatCompletionRequest {
    #[allow(clippy::type_complexity)]
    messages: Either<
        Vec<
            HashMap<
                String,
                Either<String, Vec<HashMap<String, Either<String, HashMap<String, String>>>>>,
            >,
        >,
        String,
    >,
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
    adapters: Option<Vec<String>>,
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
        grammar_type = None,
        adapters = None
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
        adapters: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let messages = Python::with_gil(|py| {
            if let Ok(messages) = messages.bind(py).downcast_exact::<PyList>() {
                let mut messages_vec = Vec::new();
                for message in messages {
                    messages_vec.push(message.extract::<HashMap<
                        String,
                        Either<
                            String,
                            Vec<HashMap<String, Either<String, HashMap<String, String>>>>,
                        >,
                    >>()?);
                }
                Ok::<
                    Either<
                        Vec<
                            HashMap<
                                String,
                                Either<
                                    String,
                                    Vec<HashMap<String, Either<String, HashMap<String, String>>>>,
                                >,
                            >,
                        >,
                        String,
                    >,
                    PyErr,
                >(Either::Left(messages_vec))
            } else if let Ok(messages) = messages.bind(py).downcast_exact::<PyString>() {
                let prompt = messages.extract::<String>()?;
                Ok::<
                    Either<
                        Vec<
                            HashMap<
                                String,
                                Either<
                                    String,
                                    Vec<HashMap<String, Either<String, HashMap<String, String>>>>,
                                >,
                            >,
                        >,
                        String,
                    >,
                    PyErr,
                >(Either::Right(prompt))
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
            adapters,
        })
    }
}

#[pymodule]
fn mistralrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Runner>()?;
    m.add_class::<Which>()?;
    m.add_class::<ChatCompletionRequest>()?;
    m.add_class::<CompletionRequest>()?;
    m.add_class::<Architecture>()?;
    m.add_class::<VisionArchitecture>()?;

    m.add_class::<mistralrs_core::ResponseMessage>()?;
    m.add_class::<mistralrs_core::Delta>()?;
    m.add_class::<mistralrs_core::ResponseLogprob>()?;
    m.add_class::<mistralrs_core::Logprobs>()?;
    m.add_class::<mistralrs_core::Choice>()?;
    m.add_class::<mistralrs_core::ChunkChoice>()?;
    m.add_class::<mistralrs_core::Usage>()?;
    m.add_class::<mistralrs_core::ChatCompletionResponse>()?;
    m.add_class::<mistralrs_core::ChatCompletionChunkResponse>()?;
    m.add_class::<mistralrs_core::CompletionChoice>()?;
    m.add_class::<mistralrs_core::CompletionResponse>()?;
    m.add_class::<mistralrs_core::TopLogprob>()?;
    Ok(())
}
