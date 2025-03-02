#![allow(clippy::too_many_arguments)]

use anyhow::Context;
use anymoe::{AnyMoeConfig, AnyMoeExpertType};
use either::Either;
use indexmap::IndexMap;
use requests::{ChatCompletionRequest, CompletionRequest, ToolChoice};
use serde_json::Value;
use std::{
    cell::RefCell,
    collections::HashMap,
    num::NonZeroUsize,
    str::FromStr,
    sync::{Arc, Mutex, OnceLock},
};
use stream::ChatCompletionStreamer;
use tokio::sync::mpsc::channel;
use util::{PyApiErr, PyApiResult};

use candle_core::{Device, Result};
use mistralrs_core::{
    initialize_logging, paged_attn_supported, parse_isq_value, AnyMoeLoader, AutoDeviceMapParams,
    ChatCompletionResponse, CompletionResponse, Constraint, DefaultSchedulerMethod,
    DetokenizationRequest, DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting,
    DiffusionGenerationParams, DiffusionLoaderBuilder, DiffusionSpecificConfig, DrySamplingParams,
    GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder, GGUFSpecificConfig,
    ImageGenerationResponse, ImageGenerationResponseFormat, LlguidanceGrammar, Loader,
    MemoryGpuConfig, MistralRs, MistralRsBuilder, NormalLoaderBuilder, NormalRequest,
    NormalSpecificConfig, PagedAttentionConfig, Request as _Request, RequestMessage, Response,
    ResponseOk, SamplingParams, SchedulerConfig, SpeculativeConfig, SpeculativeLoader, StopTokens,
    TokenSource, TokenizationRequest, Tool, Topology, VisionLoaderBuilder, VisionSpecificConfig,
};
use pyo3::prelude::*;
use std::fs::File;
mod anymoe;
mod requests;
mod stream;
mod util;
mod which;
use which::{Architecture, DiffusionArchitecture, VisionArchitecture, Which};

static DEVICE: OnceLock<Result<Device>> = OnceLock::new();

#[cfg(not(feature = "metal"))]
fn get_device(seed: Option<u64>) -> &'static Result<Device> {
    DEVICE.get_or_init(|| {
        let device = Device::cuda_if_available(0)?;
        if let Some(seed) = seed {
            device.set_seed(seed)?;
        }
        Ok(device)
    })
}

#[cfg(feature = "metal")]
fn get_device(seed: Option<u64>) -> &'static Result<Device> {
    DEVICE.get_or_init(|| {
        let device = Device::new_metal(0)?;
        if let Some(seed) = seed {
            device.set_seed(seed)?;
        }
        Ok(device)
    })
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
    prompt_chunksize: Option<NonZeroUsize>,
) -> PyApiResult<Box<dyn Loader>> {
    let use_flash_attn = mistralrs_core::using_flash_attn();

    Ok(match which {
        Which::Plain {
            model_id,
            tokenizer_json,
            arch,
            topology,
            organization,
            write_uqff,
            from_uqff,
            dtype: _,
            imatrix,
            calibration_file,
            auto_map_params: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
                organization: organization.map(Into::into).unwrap_or(Default::default()),
                write_uqff,
                from_uqff,
                imatrix,
                calibration_file,
            },
            chat_template,
            tokenizer_json,
            Some(model_id),
        )
        .with_no_kv_cache(no_kv_cache)
        .build(arch.map(Into::into))?,
        Which::XLora {
            model_id,
            xlora_model_id,
            order,
            tokenizer_json,
            tgt_non_granular_index,
            arch,
            topology,
            write_uqff,
            from_uqff,
            dtype: _,
            auto_map_params: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff,
                imatrix: None,
                calibration_file: None,
            },
            chat_template,
            tokenizer_json,
            model_id,
        )
        .with_no_kv_cache(no_kv_cache)
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
            no_kv_cache,
            tgt_non_granular_index,
        )
        .build(arch.map(Into::into))?,
        Which::Lora {
            model_id,
            tokenizer_json,
            adapters_model_id,
            order,
            arch,
            topology,
            write_uqff,
            from_uqff,
            dtype: _,
            auto_map_params: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff,
                imatrix: None,
                calibration_file: None,
            },
            chat_template,
            tokenizer_json,
            model_id,
        )
        .with_no_kv_cache(no_kv_cache)
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
        )
        .build(arch.map(Into::into))?,
        Which::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            topology,
            dtype: _,
            auto_map_params: _,
        } => GGUFLoaderBuilder::new(
            chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename.map_left(|f| vec![f]).into_inner(),
            GGUFSpecificConfig {
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
        )
        .with_no_kv_cache(no_kv_cache)
        .build(),
        Which::XLoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            topology,
            dtype: _,
            auto_map_params: _,
        } => GGUFLoaderBuilder::new(
            chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename.map_left(|f| vec![f]).into_inner(),
            GGUFSpecificConfig {
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
        )
        .with_no_kv_cache(no_kv_cache)
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
            no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        Which::LoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
            topology,
            dtype: _,
            auto_map_params: _,
        } => GGUFLoaderBuilder::new(
            chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename.map_left(|f| vec![f]).into_inner(),
            GGUFSpecificConfig {
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
        )
        .with_no_kv_cache(no_kv_cache)
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
        )
        .build(),
        Which::GGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            gqa,
            topology,
            dtype: _,
            auto_map_params: _,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
            chat_template,
            tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
        )
        .with_no_kv_cache(no_kv_cache)
        .build(),
        Which::XLoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            gqa,
            topology,
            dtype: _,
            auto_map_params: _,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
            chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .with_no_kv_cache(no_kv_cache)
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
            no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        Which::LoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
            gqa,
            topology,
            dtype: _,
            auto_map_params: _,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
            chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .with_no_kv_cache(no_kv_cache)
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
        )
        .build(),
        Which::VisionPlain {
            model_id,
            tokenizer_json,
            arch,
            topology,
            write_uqff,
            from_uqff,
            dtype: _,
            max_edge,
            calibration_file,
            imatrix,
            auto_map_params: _,
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
                use_flash_attn,
                prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff,
                max_edge,
                calibration_file,
                imatrix,
            },
            chat_template,
            tokenizer_json,
            Some(model_id),
        )
        .build(arch.into()),
        Which::DiffusionPlain {
            model_id,
            arch,
            dtype: _,
        } => {
            DiffusionLoaderBuilder::new(DiffusionSpecificConfig { use_flash_attn }, Some(model_id))
                .build(arch.into())
        }
    })
}

fn build_constraint(grammar: Option<&str>, grammar_type: Option<&str>) -> PyApiResult<Constraint> {
    if grammar_type.is_none() {
        if grammar.is_some() {
            return Err(PyApiErr::from(
                "Grammar text is specified but not grammar type",
            ));
        }
        return Ok(Constraint::None);
    }

    let grammar =
        grammar.ok_or_else(|| PyApiErr::from("Grammar type is specified but not grammar text"))?;

    let constraint = match grammar_type.unwrap() {
        "regex" => Constraint::Regex(grammar.to_string()),
        "lark" => Constraint::Lark(grammar.to_string()),
        "json_schema" => {
            let value = serde_json::from_str::<serde_json::Value>(grammar)
                .map_err(|e| PyApiErr::from(format!("Failed to parse JSON schema: {e}")))?;
            Constraint::JsonSchema(value)
        }
        "llguidance" => {
            let value = serde_json::from_str::<LlguidanceGrammar>(grammar).map_err(|e| {
                PyApiErr::from(format!("Failed to parse JSON llguidance object: {e}"))
            })?;
            Constraint::Llguidance(value)
        }
        _ => return Err(PyApiErr::from(
            "Grammar type is specified but is not `regex`, `lark`, `json_schema`, nor `llguidance`",
        )),
    };

    Ok(constraint)
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
        in_situ_quant = None,
        anymoe_config = None,
        pa_gpu_mem = None,
        pa_gpu_mem_usage = None,
        pa_ctxt_len = None,
        pa_blk_size = None,
        no_paged_attn = false,
        paged_attn = false,
        prompt_chunksize = None,
        seed = None,
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
        num_device_layers: Option<Vec<String>>,
        in_situ_quant: Option<String>,
        anymoe_config: Option<AnyMoeConfig>,
        pa_gpu_mem: Option<usize>,
        pa_gpu_mem_usage: Option<f32>,
        pa_ctxt_len: Option<usize>,
        pa_blk_size: Option<usize>,
        no_paged_attn: bool,
        paged_attn: bool,
        prompt_chunksize: Option<usize>,
        seed: Option<u64>,
    ) -> PyApiResult<Self> {
        let tgt_non_granular_index = match which {
            Which::Plain { .. }
            | Which::Lora { .. }
            | Which::GGUF { .. }
            | Which::LoraGGUF { .. }
            | Which::GGML { .. }
            | Which::LoraGGML { .. }
            | Which::VisionPlain { .. }
            | Which::DiffusionPlain { .. } => None,
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
        let dtype = match which {
            Which::Plain { dtype, .. }
            | Which::Lora { dtype, .. }
            | Which::GGUF { dtype, .. }
            | Which::LoraGGUF { dtype, .. }
            | Which::GGML { dtype, .. }
            | Which::LoraGGML { dtype, .. }
            | Which::VisionPlain { dtype, .. }
            | Which::DiffusionPlain { dtype, .. }
            | Which::XLora { dtype, .. }
            | Which::XLoraGGUF { dtype, .. }
            | Which::XLoraGGML { dtype, .. } => dtype,
        };
        let auto_map_params = match &which {
            Which::Plain {
                auto_map_params, ..
            }
            | Which::Lora {
                auto_map_params, ..
            }
            | Which::GGUF {
                auto_map_params, ..
            }
            | Which::LoraGGUF {
                auto_map_params, ..
            }
            | Which::GGML {
                auto_map_params, ..
            }
            | Which::LoraGGML {
                auto_map_params, ..
            }
            | Which::XLora {
                auto_map_params, ..
            }
            | Which::XLoraGGUF {
                auto_map_params, ..
            }
            | Which::XLoraGGML {
                auto_map_params, ..
            } => auto_map_params
                .clone()
                .map(|p| AutoDeviceMapParams::Text {
                    max_seq_len: p.max_seq_len,
                    max_batch_size: p.max_batch_size,
                })
                .unwrap_or(AutoDeviceMapParams::default_text()),
            Which::VisionPlain {
                auto_map_params, ..
            } => auto_map_params
                .clone()
                .map(|p| AutoDeviceMapParams::Vision {
                    max_seq_len: p.max_seq_len,
                    max_batch_size: p.max_batch_size,
                    max_image_shape: (p.max_image_length, p.max_image_length),
                    max_num_images: p.max_num_images,
                })
                .unwrap_or(AutoDeviceMapParams::default_vision()),
            Which::DiffusionPlain { .. } => AutoDeviceMapParams::default_text(),
        };

        let max_seq_len = auto_map_params.max_seq_len();

        let max_seqs = if tgt_non_granular_index.is_some() {
            1
        } else {
            max_seqs
        };

        let prompt_chunksize = match prompt_chunksize {
            Some(0) => {
                return Err(PyApiErr::from(
                    "`prompt_chunksize` must be a strictly positive integer, got 0.",
                ))
            }
            Some(x) => Some(NonZeroUsize::new(x).unwrap()),
            None => None,
        };

        let loader = parse_which(which, no_kv_cache, chat_template.clone(), prompt_chunksize)?;
        let loader = if let Some(draft_which) = which_draft {
            let draft = parse_which(draft_which, no_kv_cache, chat_template, prompt_chunksize)?;
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
        let loader = if let Some(amoe_conf) = anymoe_config {
            Box::new(AnyMoeLoader {
                target: loader,
                config: mistralrs_core::AnyMoeConfig {
                    hidden_size: amoe_conf.hidden_size,
                    lr: amoe_conf.lr,
                    epochs: amoe_conf.epochs,
                    batch_size: amoe_conf.batch_size,
                    expert_type: amoe_conf.expert_type.into(),
                    gate_model_id: amoe_conf.gate_model_id.clone(),
                    training: amoe_conf.training,
                    loss_csv_path: amoe_conf.loss_csv_path.clone(),
                },
                path: amoe_conf.dataset_json,
                prefix: amoe_conf.prefix,
                mlp: amoe_conf.mlp,
                model_ids: amoe_conf.model_ids,
                layers: amoe_conf.layers,
            })
        } else {
            loader
        };

        let device = get_device(seed).as_ref().map_err(PyApiErr::from)?;
        let isq = if let Some(isq) = in_situ_quant {
            Some(parse_isq_value(&isq).map_err(PyApiErr::from)?)
        } else {
            None
        };

        let mapper = match num_device_layers {
            Some(device_layers) => {
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
            }
            None => DeviceMapSetting::Auto(auto_map_params),
        };

        let no_paged_attn = if device.is_cuda() {
            no_paged_attn
        } else if device.is_metal() {
            !paged_attn
        } else {
            true
        };

        // Allocate 0.5 GB of CPU memory just as a placeholder.
        // Nothing happens here as we have no `swap_out`, see `_preempt_by_swap`.
        let cache_config =
            match (
                pa_blk_size,
                pa_gpu_mem,
                pa_gpu_mem_usage,
                pa_ctxt_len,
                paged_attn_supported(),
                no_paged_attn,
            ) {
                (block_size, None, None, None, true, false) => Some(PagedAttentionConfig::new(
                    block_size,
                    512,
                    MemoryGpuConfig::ContextSize(max_seq_len),
                )?),
                (block_size, None, None, Some(ctxt), true, false) => Some(
                    PagedAttentionConfig::new(block_size, 512, MemoryGpuConfig::ContextSize(ctxt))?,
                ),
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
                (block_size, Some(_m), Some(f), None, true, false) => Some(
                    PagedAttentionConfig::new(block_size, 512, MemoryGpuConfig::Utilization(f))?,
                ),
                (block_size, Some(_m), None, Some(ctxt), true, false) => Some(
                    PagedAttentionConfig::new(block_size, 512, MemoryGpuConfig::ContextSize(ctxt))?,
                ),
                (block_size, None, Some(f), Some(_ctxt), true, false) => Some(
                    PagedAttentionConfig::new(block_size, 512, MemoryGpuConfig::Utilization(f))?,
                ),
                (_, _, _, _, _, _) => None,
            };

        let pipeline = loader
            .load_model_from_hf(
                None,
                TokenSource::from_str(token_source).map_err(PyApiErr::from)?,
                &dtype,
                device,
                true, // Silent for jupyter
                mapper,
                isq,
                cache_config,
            )
            .map_err(PyApiErr::from)?;

        let scheduler_config = if cache_config.is_some() {
            // Handle case where we may have device mapping
            if let Some(ref cache_config) = pipeline.blocking_lock().get_metadata().cache_config {
                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: max_seqs,
                    config: cache_config.clone(),
                }
            } else {
                SchedulerConfig::DefaultScheduler {
                    method: DefaultSchedulerMethod::Fixed(
                        max_seqs
                            .try_into()
                            .map_err(|e| PyApiErr::from(format!("{e:?}")))?,
                    ),
                }
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(
                    max_seqs
                        .try_into()
                        .map_err(|e| PyApiErr::from(format!("{e:?}")))?,
                ),
            }
        };
        let mistralrs = MistralRsBuilder::new(pipeline, scheduler_config)
            .with_no_kv_cache(no_kv_cache)
            .with_prefix_cache_n(prefix_cache_n)
            .build();

        Ok(Self { runner: mistralrs })
    }

    /// Send an OpenAI API compatible request, returning the result.
    fn send_chat_completion_request(
        &mut self,
        request: Py<ChatCompletionRequest>,
    ) -> PyApiResult<Either<ChatCompletionResponse, ChatCompletionStreamer>> {
        let (tx, mut rx) = channel(10_000);
        Python::with_gil(|py| {
            let request = request.bind(py).borrow();
            let stop_toks = request
                .stop_seqs
                .as_ref()
                .map(|x| StopTokens::Seqs(x.to_vec()));
            let constraint =
                build_constraint(request.grammar.as_deref(), request.grammar_type.as_deref())?;

            let dry_params = if let Some(dry_multiplier) = request.dry_multiplier {
                Some(DrySamplingParams::new_with_defaults(
                    dry_multiplier,
                    request.dry_sequence_breakers.clone(),
                    request.dry_base,
                    request.dry_allowed_length,
                )?)
            } else {
                None
            };

            let messages = match request.messages {
                Either::Left(ref messages) => {
                    let mut messages_vec = Vec::new();
                    let mut image_urls = Vec::new();
                    for message in messages {
                        match &message["content"] {
                            Either::Left(content) => {
                                let mut message_map: IndexMap<
                                    String,
                                    Either<String, Vec<IndexMap<String, Value>>>,
                                > = IndexMap::new();
                                message_map.insert(
                                    "role".to_string(),
                                    Either::Left(message["role"].as_ref().left().unwrap().clone()),
                                );
                                message_map.insert(
                                    "content".to_string(),
                                    Either::Left(content.to_string()),
                                );
                                messages_vec.push(message_map);
                            }
                            Either::Right(image_messages) => {
                                if image_messages.len() != 2 {
                                    return Err(PyApiErr::from(
                                        "Expected 2 items for the content of a message with an image."
                                    ));
                                }
                                if message["role"].as_ref().left().unwrap() != "user" {
                                    return Err(PyApiErr::from(format!(
                                        "Role for an image message must be `user`, but it is {}",
                                        &message["role"].as_ref().left().unwrap()
                                    )));
                                }

                                let mut items = Vec::new();
                                for image_message in image_messages {
                                    if image_message.len() != 2 {
                                        return Err(PyApiErr::from("Expected 2 items for the sub-content of a message with an image.".to_string()));
                                    }
                                    if !image_message.contains_key("type") {
                                        return Err(PyApiErr::from(
                                            "Expected `type` key in input message.".to_string(),
                                        ));
                                    }
                                    if image_message["type"].is_right() {
                                        return Err(PyApiErr::from(
                                            "Expected string value in `type`.".to_string(),
                                        ));
                                    }
                                    items.push(image_message["type"].as_ref().unwrap_left().clone())
                                }

                                #[allow(clippy::type_complexity)]
                                fn get_content_and_url(
                                    text_idx: usize,
                                    url_idx: usize,
                                    image_messages: &[HashMap<
                                        String,
                                        Either<String, HashMap<String, String>>,
                                    >],
                                ) -> PyApiResult<(String, String)> {
                                    if image_messages[text_idx]["text"].is_right() {
                                        return Err(PyApiErr::from(
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
                                        return Err(PyApiErr::from("Expected content of format {{`type`: `text`, `text`: ...}} and {{`type`: `url`, `image_url`: {{`url`: ...}}}}".to_string()));
                                    }
                                    let url = image_messages[url_idx]["image_url"]
                                        .as_ref()
                                        .unwrap_right()["url"]
                                        .clone();
                                    Ok((content, url))
                                }
                                let mut message_map: IndexMap<
                                    String,
                                    Either<String, Vec<IndexMap<String, Value>>>,
                                > = IndexMap::new();
                                message_map.insert(
                                    "role".to_string(),
                                    Either::Left(message["role"].as_ref().left().unwrap().clone()),
                                );
                                let (content, url) = if items[0] == "text" {
                                    get_content_and_url(0, 1, image_messages)?
                                } else {
                                    get_content_and_url(1, 0, image_messages)?
                                };

                                let mut content_map = Vec::new();
                                let mut content_image_map = IndexMap::new();
                                content_image_map
                                    .insert("type".to_string(), Value::String("image".to_string()));
                                content_map.push(content_image_map);
                                let mut content_text_map = IndexMap::new();
                                content_text_map
                                    .insert("type".to_string(), Value::String("text".to_string()));
                                content_text_map.insert("text".to_string(), Value::String(content));
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
                            let url_unparsed = url.trim();

                            let image = util::parse_image_url(url_unparsed)?;
                            images.push(image);
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
                        Either<String, Vec<IndexMap<String, Value>>>,
                    > = IndexMap::new();
                    message_map.insert("role".to_string(), Either::Left("user".to_string()));
                    message_map.insert("content".to_string(), Either::Left(prompt.to_string()));
                    messages.push(message_map);
                    RequestMessage::Chat(messages)
                }
            };

            let tool_choice = request.tool_choice.as_ref().map(|x| match x {
                ToolChoice::Auto => mistralrs_core::ToolChoice::Auto,
                ToolChoice::NoTools => mistralrs_core::ToolChoice::None,
            });

            let tools = if let Some(tools) = &request.tool_schemas {
                let mut new_tools = Vec::new();
                for schema in tools {
                    new_tools.push(serde_json::from_str::<Tool>(schema)?);
                }
                Some(new_tools)
            } else {
                None
            };

            let model_request = _Request::Normal(NormalRequest {
                id: {
                    let l = NEXT_REQUEST_ID.lock().unwrap();
                    let last = &mut *l.borrow_mut();
                    let last_v = *last;
                    *last += 1;
                    last_v
                },
                messages,
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
                    min_p: request.min_p,
                    dry_params,
                },
                response: tx,
                return_logprobs: request.logprobs,
                is_streaming: request.stream,
                constraint,
                suffix: None,
                adapters: request.adapters.clone(),
                tool_choice,
                tools,
                logits_processors: None,
                return_raw_logits: false,
            });

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender()?;
            sender.blocking_send(model_request).unwrap();

            if request.stream {
                Ok(Either::Right(ChatCompletionStreamer::from_rx(rx)))
            } else {
                let response = rx.blocking_recv().unwrap();

                match response {
                    Response::ValidationError(e) | Response::InternalError(e) => {
                        Err(PyApiErr::from(e.to_string()))
                    }
                    Response::Done(response) => Ok(Either::Left(response)),
                    Response::ModelError(msg, _) => Err(PyApiErr::from(msg.to_string())),
                    Response::Chunk(_) => unreachable!(),
                    Response::CompletionDone(_) => unreachable!(),
                    Response::CompletionModelError(_, _) => unreachable!(),
                    Response::CompletionChunk(_) => unreachable!(),
                    Response::ImageGeneration(_) => unreachable!(),
                    Response::Raw { .. } => unreachable!(),
                }
            }
        })
    }

    /// Send an OpenAI API compatible request, returning the result.
    fn send_completion_request(
        &mut self,
        request: Py<CompletionRequest>,
    ) -> PyApiResult<CompletionResponse> {
        let (tx, mut rx) = channel(10_000);
        Python::with_gil(|py| {
            let request = request.bind(py).borrow();
            let stop_toks = request
                .stop_seqs
                .as_ref()
                .map(|x| StopTokens::Seqs(x.to_vec()));
            let constraint =
                build_constraint(request.grammar.as_deref(), request.grammar_type.as_deref())?;

            let tool_choice = request.tool_choice.as_ref().map(|x| match x {
                ToolChoice::Auto => mistralrs_core::ToolChoice::Auto,
                ToolChoice::NoTools => mistralrs_core::ToolChoice::None,
            });

            let tools = if let Some(tools) = &request.tool_schemas {
                let mut new_tools = Vec::new();
                for schema in tools {
                    new_tools.push(serde_json::from_str::<Tool>(schema)?);
                }
                Some(new_tools)
            } else {
                None
            };

            let dry_params = if let Some(dry_multiplier) = request.dry_multiplier {
                Some(DrySamplingParams::new_with_defaults(
                    dry_multiplier,
                    request.dry_sequence_breakers.clone(),
                    request.dry_base,
                    request.dry_allowed_length,
                )?)
            } else {
                None
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
                    min_p: request.min_p,
                    dry_params,
                },
                response: tx,
                return_logprobs: false,
                is_streaming: false,
                constraint,
                suffix: request.suffix.clone(),
                adapters: request.adapters.clone(),
                tool_choice,
                tools,
                logits_processors: None,
                return_raw_logits: false,
            });

            MistralRs::maybe_log_request(self.runner.clone(), format!("{request:?}"));
            let sender = self.runner.get_sender()?;
            sender.blocking_send(model_request).unwrap();
            let response = rx.blocking_recv().unwrap();

            match response {
                Response::ValidationError(e) | Response::InternalError(e) => {
                    Err(PyApiErr::from(e.to_string()))
                }
                Response::CompletionDone(response) => Ok(response),
                Response::CompletionModelError(msg, _) => Err(PyApiErr::from(msg.to_string())),
                Response::Chunk(_) => unreachable!(),
                Response::Done(_) => unreachable!(),
                Response::ModelError(_, _) => unreachable!(),
                Response::CompletionChunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
                Response::Raw { .. } => unreachable!(),
            }
        })
    }

    /// Generate an image.
    #[pyo3(signature = (
        prompt,
        response_format,
        height = 720,
        width = 1280,
    ))]
    fn generate_image(
        &self,
        prompt: String,
        response_format: ImageGenerationResponseFormat,
        height: usize,
        width: usize,
    ) -> PyApiResult<ImageGenerationResponse> {
        let (tx, mut rx) = channel(1);

        let request = _Request::Normal(NormalRequest {
            id: 0,
            messages: RequestMessage::ImageGeneration {
                prompt: prompt.to_string(),
                format: response_format,
                generation_params: DiffusionGenerationParams { height, width },
            },
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            suffix: None,
            constraint: Constraint::None,
            adapters: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
        });

        let sender = self.runner.get_sender()?;
        sender.blocking_send(request).unwrap();

        let ResponseOk::ImageGeneration(response) = rx
            .blocking_recv()
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            return Err(PyApiErr::from("Got unexpected response type."));
        };

        Ok(response)
    }

    /// Send a request to re-ISQ the model. If the model was loaded as GGUF or GGML
    /// then nothing will happen.
    fn send_re_isq(&self, dtype: String) -> PyApiResult<()> {
        let request = _Request::ReIsq(parse_isq_value(&dtype)?);
        self.runner.get_sender()?.blocking_send(request).unwrap();
        Ok(())
    }

    /// Send a request to make the specified adapters the active adapters for the model.
    fn activate_adapters(&self, adapter_names: Vec<String>) {
        let request = _Request::ActivateAdapters(adapter_names);
        self.runner
            .get_sender()
            .unwrap()
            .blocking_send(request)
            .unwrap();
    }

    /// Tokenize some text, returning raw tokens.
    fn tokenize_text(&self, text: String, add_special_tokens: bool) -> PyApiResult<Vec<u32>> {
        let (tx, mut rx) = channel(1);
        let request = _Request::Tokenize(TokenizationRequest {
            text: Either::Right(text),
            tools: None,
            add_generation_prompt: true,
            add_special_tokens,
            response: tx,
        });

        self.runner.get_sender()?.blocking_send(request).unwrap();

        rx.blocking_recv()
            .context("Channel was erroneously closed!")?
            .map_err(PyApiErr::from)
    }

    /// Detokenize some tokens, returning text.
    fn detokenize_text(&self, tokens: Vec<u32>, skip_special_tokens: bool) -> PyApiResult<String> {
        let (tx, mut rx) = channel(1);
        let request = _Request::Detokenize(DetokenizationRequest {
            tokens,
            skip_special_tokens,
            response: tx,
        });

        self.runner.get_sender()?.blocking_send(request).unwrap();

        rx.blocking_recv()
            .context("Channel was erroneously closed!")?
            .map_err(PyApiErr::from)
    }
}

#[pymodule]
fn mistralrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    initialize_logging();

    m.add_class::<Runner>()?;
    m.add_class::<Which>()?;
    m.add_class::<ChatCompletionRequest>()?;
    m.add_class::<CompletionRequest>()?;
    m.add_class::<Architecture>()?;
    m.add_class::<VisionArchitecture>()?;
    m.add_class::<DiffusionArchitecture>()?;
    m.add_class::<AnyMoeConfig>()?;
    m.add_class::<AnyMoeExpertType>()?;
    m.add_class::<ToolChoice>()?;

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
    m.add_class::<mistralrs_core::ModelDType>()?;
    m.add_class::<mistralrs_core::ImageGenerationResponseFormat>()?;
    Ok(())
}
