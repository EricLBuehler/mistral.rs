#![allow(clippy::too_many_arguments)]

use anyhow::Context;
use anymoe::{AnyMoeConfig, AnyMoeExpertType};
use either::Either;
use indexmap::IndexMap;
use itertools::Itertools;
use requests::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, PythonEmbeddingInputs, ToolChoice,
};
use serde_json::Value;
use std::{
    cell::RefCell,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex, OnceLock},
};
use stream::ChatCompletionStreamer;
use tokio::{
    runtime::Runtime,
    sync::mpsc::{channel, Receiver},
};
use util::{PyApiErr, PyApiResult};

use candle_core::{Device, Result};
use mistralrs_core::{
    initialize_logging, paged_attn_supported, parse_isq_value, AnyMoeLoader, AutoDeviceMapParams,
    ChatCompletionResponse, CompletionResponse, Constraint, DefaultSchedulerMethod,
    DetokenizationRequest, DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting,
    DiffusionGenerationParams, DiffusionLoaderBuilder, DrySamplingParams, EmbeddingLoaderBuilder,
    EmbeddingSpecificConfig, GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder,
    GGUFSpecificConfig, ImageGenerationResponse, ImageGenerationResponseFormat, LlguidanceGrammar,
    Loader, MemoryGpuConfig, MistralRs, MistralRsBuilder, NormalLoaderBuilder, NormalRequest,
    NormalSpecificConfig, PagedAttentionConfig, PagedCacheType, ReasoningEffort,
    Request as _Request, RequestMessage, Response, ResponseOk, SamplingParams, SchedulerConfig,
    SearchEmbeddingModel, SpeculativeConfig, SpeculativeLoader, SpeechLoader, StopTokens,
    TokenSource, TokenizationRequest, Tool, Topology, VisionLoaderBuilder, VisionSpecificConfig,
};
use mistralrs_core::{
    CalledFunction, SearchCallback, SearchFunctionParameters, SearchResult, ToolCallback,
    ToolCallbacks,
};
use mistralrs_mcp::{McpClientConfig, McpServerConfig, McpServerSource};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Bound;
use pyo3::PyObject;
use std::fs::File;
mod anymoe;
mod requests;
mod stream;
mod util;
mod which;
use which::{Architecture, DiffusionArchitecture, SpeechLoaderType, VisionArchitecture, Which};

/// Parse reasoning effort string to ReasoningEffort enum
fn parse_reasoning_effort(effort: &Option<String>) -> Option<ReasoningEffort> {
    effort
        .as_ref()
        .and_then(|e| match e.to_lowercase().as_str() {
            "low" => Some(ReasoningEffort::Low),
            "medium" => Some(ReasoningEffort::Medium),
            "high" => Some(ReasoningEffort::High),
            _ => None,
        })
}

static DEVICE: OnceLock<Result<Device>> = OnceLock::new();

#[cfg(not(feature = "metal"))]
fn get_device(seed: Option<u64>) -> &'static Result<Device> {
    DEVICE.get_or_init(|| {
        let device = if mistralrs_core::distributed::use_nccl() {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };
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
#[pyo3(get_all)]
#[derive(Debug, Clone)]
pub struct SpeechGenerationResponse {
    pub pcm: Vec<f32>,
    pub rate: usize,
    pub channels: usize,
}

#[pyclass]
#[derive(Clone)]
/// An object wrapping the underlying Rust system to handle requests and process conversations.
struct Runner {
    runner: Arc<MistralRs>,
}

static NEXT_REQUEST_ID: Mutex<RefCell<usize>> = Mutex::new(RefCell::new(0));

fn wrap_search_callback(cb: PyObject) -> Arc<SearchCallback> {
    Arc::new(move |params: &SearchFunctionParameters| {
        Python::with_gil(|py| {
            let obj = cb.call1(py, (params.query.clone(),))?;
            let list = obj.downcast_bound::<PyList>(py)?;
            let mut results = Vec::new();
            for item in list.iter() {
                let title: String = item.get_item("title")?.extract()?;
                let description: String = item.get_item("description")?.extract()?;
                let url: String = item.get_item("url")?.extract()?;
                let content: String = item.get_item("content")?.extract()?;
                results.push(SearchResult {
                    title,
                    description,
                    url,
                    content,
                });
            }
            Ok(results)
        })
        .map_err(|e: PyErr| anyhow::anyhow!(e.to_string()))
    })
}

fn wrap_tool_callback(cb: PyObject) -> Arc<ToolCallback> {
    Arc::new(move |func: &CalledFunction| {
        Python::with_gil(|py| {
            let json = py.import("json")?;
            let args: Py<PyAny> = json
                .call_method1("loads", (func.arguments.clone(),))?
                .into();
            let obj = cb.call1(py, (func.name.clone(), args))?;
            obj.extract::<String>(py)
        })
        .map_err(|e: PyErr| anyhow::anyhow!(e.to_string()))
    })
}

fn wrap_tool_callbacks(obj: PyObject) -> anyhow::Result<ToolCallbacks> {
    Python::with_gil(|py| {
        let dict = obj
            .downcast_bound::<pyo3::types::PyDict>(py)
            .map_err(|e| anyhow::anyhow!("Failed to downcast to PyDict: {}", e))?;

        let mut map = ToolCallbacks::new();

        for (name, cb) in dict.iter() {
            let name: String = name
                .extract()
                .map_err(|e: PyErr| anyhow::anyhow!(e.to_string()))?;
            let cb_obj: PyObject = cb.into();
            map.insert(name, wrap_tool_callback(cb_obj));
        }
        Ok(map)
    })
}

fn parse_which(
    which: Which,
    no_kv_cache: bool,
    chat_template: Option<String>,
    jinja_explicit: Option<String>,
) -> PyApiResult<Box<dyn Loader>> {
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
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                organization: organization.map(Into::into).unwrap_or(Default::default()),
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.right_or_else(|l| vec![l])
                        .iter()
                        .map(|x| PathBuf::from_str(x).unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix,
                calibration_file,
                hf_cache_path,
                matformer_config_path,
                matformer_slice_name,
            },
            chat_template,
            tokenizer_json,
            Some(model_id),
            no_kv_cache,
            jinja_explicit,
        )
        .build(arch.map(Into::into))?,
        Which::Embedding {
            model_id,
            tokenizer_json,
            arch,
            topology,
            write_uqff,
            from_uqff,
            dtype: _,
            hf_cache_path,
        } => EmbeddingLoaderBuilder::new(
            EmbeddingSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.right_or_else(|l| vec![l])
                        .iter()
                        .map(|path| PathBuf::from_str(path).unwrap())
                        .collect::<Vec<_>>()
                }),
                hf_cache_path,
            },
            tokenizer_json,
            Some(model_id),
        )
        .build(arch.map(Into::into)),
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
            hf_cache_path,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.right_or_else(|l| vec![l])
                        .iter()
                        .map(|x| PathBuf::from_str(x).unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix: None,
                calibration_file: None,
                hf_cache_path,
                matformer_config_path: None,
                matformer_slice_name: None,
            },
            chat_template,
            tokenizer_json,
            model_id,
            no_kv_cache,
            jinja_explicit,
        )
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
            adapter_model_ids,
            arch,
            topology,
            write_uqff,
            from_uqff,
            dtype: _,
            auto_map_params: _,
            hf_cache_path,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.right_or_else(|l| vec![l])
                        .iter()
                        .map(|x| PathBuf::from_str(x).unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix: None,
                calibration_file: None,
                hf_cache_path,
                matformer_config_path: None,
                matformer_slice_name: None,
            },
            chat_template,
            tokenizer_json,
            model_id,
            no_kv_cache,
            jinja_explicit,
        )
        .with_lora(adapter_model_ids)
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
                topology: Topology::from_option_path(topology)?,
            },
            no_kv_cache,
            jinja_explicit,
        )
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
                topology: Topology::from_option_path(topology)?,
            },
            no_kv_cache,
            jinja_explicit,
        )
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
                topology: Topology::from_option_path(topology)?,
            },
            no_kv_cache,
            jinja_explicit,
        )
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
                topology: Topology::from_option_path(topology)?,
            },
            chat_template,
            tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
            no_kv_cache,
            jinja_explicit,
        )
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
                topology: Topology::from_option_path(topology)?,
            },
            chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            no_kv_cache,
            jinja_explicit,
        )
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
                topology: Topology::from_option_path(topology)?,
            },
            chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            no_kv_cache,
            jinja_explicit,
        )
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
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            organization,
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.right_or_else(|l| vec![l])
                        .iter()
                        .map(|x| PathBuf::from_str(x).unwrap())
                        .collect::<Vec<_>>()
                }),
                max_edge,
                calibration_file,
                imatrix,
                hf_cache_path,
                matformer_config_path,
                matformer_slice_name,
                organization: organization.map(Into::into).unwrap_or_default(),
            },
            chat_template,
            tokenizer_json,
            Some(model_id),
            jinja_explicit,
        )
        .build(arch.map(Into::into)),
        Which::DiffusionPlain {
            model_id,
            arch,
            dtype: _,
        } => DiffusionLoaderBuilder::new(Some(model_id)).build(arch.into()),
        Which::Speech {
            model_id,
            dac_model_id,
            arch,
            ..
        } => Box::new(SpeechLoader {
            model_id,
            dac_model_id,
            arch: arch.into(),
            cfg: None,
        }),
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
        jinja_explicit = None,
        num_device_layers = None,
        in_situ_quant = None,
        anymoe_config = None,
        pa_gpu_mem = None,
        pa_gpu_mem_usage = None,
        pa_ctxt_len = None,
        pa_blk_size = None,
        pa_cache_type = None,
        no_paged_attn = false,
        paged_attn = false,
        seed = None,
        enable_search = false,
        search_embedding_model = None,
        search_callback = None,
        tool_callbacks = None,
        mcp_client_config = None,
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
        jinja_explicit: Option<String>,
        num_device_layers: Option<Vec<String>>,
        in_situ_quant: Option<String>,
        anymoe_config: Option<AnyMoeConfig>,
        pa_gpu_mem: Option<usize>,
        pa_gpu_mem_usage: Option<f32>,
        pa_ctxt_len: Option<usize>,
        pa_blk_size: Option<usize>,
        pa_cache_type: Option<PagedCacheType>,
        no_paged_attn: bool,
        paged_attn: bool,
        seed: Option<u64>,
        enable_search: bool,
        search_embedding_model: Option<String>,
        search_callback: Option<PyObject>,
        tool_callbacks: Option<PyObject>,
        mcp_client_config: Option<McpClientConfigPy>,
    ) -> PyApiResult<Self> {
        let tgt_non_granular_index = match which {
            Which::Plain { .. }
            | Which::Lora { .. }
            | Which::GGUF { .. }
            | Which::LoraGGUF { .. }
            | Which::GGML { .. }
            | Which::LoraGGML { .. }
            | Which::Embedding { .. }
            | Which::VisionPlain { .. }
            | Which::DiffusionPlain { .. }
            | Which::Speech { .. } => None,
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
            | Which::Embedding { dtype, .. }
            | Which::VisionPlain { dtype, .. }
            | Which::DiffusionPlain { dtype, .. }
            | Which::Speech { dtype, .. }
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
            Which::Embedding { .. } | Which::DiffusionPlain { .. } | Which::Speech { .. } => {
                AutoDeviceMapParams::default_text()
            }
        };

        let max_seq_len = auto_map_params.max_seq_len();

        let max_seqs = if tgt_non_granular_index.is_some() {
            1
        } else {
            max_seqs
        };

        let loader = parse_which(
            which,
            no_kv_cache,
            chat_template.clone(),
            jinja_explicit.clone(),
        )?;
        let loader = if let Some(draft_which) = which_draft {
            let draft = parse_which(draft_which, no_kv_cache, chat_template, jinja_explicit)?;
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
            Some(parse_isq_value(&isq, Some(device)).map_err(PyApiErr::from)?)
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

        let no_paged_attn = if device.is_cuda() || mistralrs_core::distributed::use_nccl() {
            no_paged_attn
        } else if device.is_metal() {
            !paged_attn
        } else {
            true
        };

        let cache_config = match (
            pa_blk_size,
            pa_gpu_mem,
            pa_gpu_mem_usage,
            pa_ctxt_len,
            paged_attn_supported(),
            no_paged_attn,
        ) {
            (block_size, None, None, None, true, false) => Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::ContextSize(max_seq_len),
                pa_cache_type.unwrap_or_default(),
            )?),
            (block_size, None, None, Some(ctxt), true, false) => Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::ContextSize(ctxt),
                pa_cache_type.unwrap_or_default(),
            )?),
            (block_size, None, Some(f), None, true, false) => Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                pa_cache_type.unwrap_or_default(),
            )?),
            (block_size, Some(m), None, None, true, false) => Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::MbAmount(m),
                pa_cache_type.unwrap_or_default(),
            )?),
            (block_size, Some(_m), Some(f), None, true, false) => Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                pa_cache_type.unwrap_or_default(),
            )?),
            (block_size, Some(_m), None, Some(ctxt), true, false) => {
                Some(PagedAttentionConfig::new(
                    block_size,
                    MemoryGpuConfig::ContextSize(ctxt),
                    pa_cache_type.unwrap_or_default(),
                )?)
            }
            (block_size, None, Some(f), Some(_ctxt), true, false) => {
                Some(PagedAttentionConfig::new(
                    block_size,
                    MemoryGpuConfig::Utilization(f),
                    pa_cache_type.unwrap_or_default(),
                )?)
            }
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
        let search_embedding_model = if enable_search {
            Some(match search_embedding_model {
                Some(model) => {
                    SearchEmbeddingModel::from_str(model.as_str()).map_err(PyApiErr::from)?
                }
                None => SearchEmbeddingModel::default(),
            })
        } else {
            None
        };
        let cb = search_callback.map(wrap_search_callback);
        let tool_cbs = match tool_callbacks {
            Some(obj) => Some(wrap_tool_callbacks(obj)?),
            None => None,
        };
        let mut builder =
            MistralRsBuilder::new(pipeline, scheduler_config, false, search_embedding_model);
        if let Some(cb) = cb {
            builder = builder.with_search_callback(cb);
        }
        if let Some(map) = tool_cbs {
            for (name, cb) in map {
                builder = builder.with_tool_callback(name, cb);
            }
        }
        if let Some(mcp_config) = mcp_client_config {
            builder = builder.with_mcp_client(mcp_config.into());
        }
        let rt = Runtime::new().expect("Failed to create Runner::new runtime");
        let mistralrs = rt.block_on(async {
            builder
                .with_no_kv_cache(no_kv_cache)
                .with_prefix_cache_n(prefix_cache_n)
                .build()
                .await
        });

        Ok(Self { runner: mistralrs })
    }

    /// Send an OpenAI API compatible request, returning the result.
    #[pyo3(signature = (request, model_id = None))]
    fn send_chat_completion_request(
        &mut self,
        request: Py<ChatCompletionRequest>,
        model_id: Option<String>,
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
                    let mut audio_urls = Vec::new();
                    for message in messages {
                        let role = message["role"].as_ref().left().unwrap().clone();
                        match &message["content"] {
                            Either::Left(content) => {
                                let mut message_map: IndexMap<
                                    String,
                                    Either<String, Vec<IndexMap<String, Value>>>,
                                > = IndexMap::new();
                                message_map.insert("role".to_string(), Either::Left(role));
                                message_map.insert(
                                    "content".to_string(),
                                    Either::Left(content.to_string()),
                                );
                                messages_vec.push(message_map);
                            }
                            Either::Right(image_messages) => {
                                // If there is only one message, it is possible a text message
                                // found when rig is used as client. In this case, we need to check if
                                // the message is a text message or an image message.
                                if image_messages.len() == 1 {
                                    if !image_messages[0].contains_key("text") {
                                        return Err(PyApiErr::from(
                                            "Expected `text` key in input message.",
                                        ));
                                    }
                                    let content = match &image_messages[0]["text"] {
                                        Either::Left(left) => left.to_string(),
                                        Either::Right(right) => format!("{right:?}"),
                                    };
                                    let mut message_map: IndexMap<
                                        String,
                                        Either<String, Vec<IndexMap<String, Value>>>,
                                    > = IndexMap::new();
                                    message_map.insert("role".to_string(), Either::Left(role));
                                    message_map
                                        .insert("content".to_string(), Either::Left(content));
                                    messages_vec.push(message_map);
                                    continue;
                                }
                                if role != "user" {
                                    return Err(PyApiErr::from(
                                        "Role for an image message must be `user`, but it is {role}",
                                    ));
                                }

                                enum ContentPart {
                                    Text { text: String },
                                    Image { image_url: String },
                                    Audio { audio_url: String },
                                }

                                let mut items = Vec::new();
                                for image_message in image_messages {
                                    match image_message.get("type") {
                                        Some(Either::Left(x)) if x == "text" => {
                                            items.push(ContentPart::Text {
                                                text: image_message
                                                    .get("text").as_ref()
                                                    .context("Text sub-content must have `text` key.")?.as_ref()
                                                    .left().context("Text sub-content `text` key must be a string.")?.clone(),
                                            });
                                        }
                                        Some(Either::Left(x)) if x == "image_url" => {
                                            items.push(ContentPart::Image {
                                                image_url: image_message
                                                    .get("image_url")
                                                    .as_ref()
                                                    .context("Image sub-content must have `image_url` key.")?
                                                    .as_ref()
                                                    .right()
                                                    .context("Image sub-content `image_url` key must be an object.")?
                                                    .get("url")
                                                    .context("Image sub-content `image_url` object must have a `url` key.")?
                                                    .clone(),
                                            });
                                        }
                                        Some(Either::Left(x)) if x == "audio_url" => {
                                            items.push(ContentPart::Audio {
                                                audio_url: image_message
                                                    .get("audio_url")
                                                    .as_ref()
                                                    .context("Audio sub-content must have `audio_url` key.")?
                                                    .as_ref()
                                                    .right()
                                                    .context("Audio sub-content `audio_url` key must be an object.")?
                                                    .get("url")
                                                    .context("Audio sub-content `audio_url` object must have a `url` key.")?
                                                    .clone(),
                                            });
                                        }
                                        _ => return Err(PyApiErr::from("Expected array content sub-content to be of format {{`type`: `text`, `text`: ...}} and {{`type`: `url`, `image_url`: {{`url`: ...}}}}"))
                                    }
                                }

                                let text_content = items
                                    .iter()
                                    .filter_map(|item| match item {
                                        ContentPart::Text { text } => Some(text),
                                        _ => None,
                                    })
                                    .join(" ");
                                let image_urls_iter = items
                                    .iter()
                                    .filter_map(|item| match item {
                                        ContentPart::Image { image_url } => Some(image_url.clone()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>();

                                let audio_urls_iter = items
                                    .iter()
                                    .filter_map(|item| match item {
                                        ContentPart::Audio { audio_url } => Some(audio_url.clone()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>();

                                let mut message_map: IndexMap<
                                    String,
                                    Either<String, Vec<IndexMap<String, Value>>>,
                                > = IndexMap::new();
                                message_map.insert("role".to_string(), Either::Left(role));

                                let mut content_map: Vec<IndexMap<String, Value>> = Vec::new();
                                for _ in &image_urls_iter {
                                    let mut content_image_map = IndexMap::new();
                                    content_image_map.insert(
                                        "type".to_string(),
                                        Value::String("image".to_string()),
                                    );
                                    content_map.push(content_image_map);
                                }
                                for _ in &audio_urls_iter {
                                    let mut content_audio_map = IndexMap::new();
                                    content_audio_map.insert(
                                        "type".to_string(),
                                        Value::String("audio".to_string()),
                                    );
                                    content_map.push(content_audio_map);
                                }
                                {
                                    let mut content_text_map = IndexMap::new();
                                    content_text_map.insert(
                                        "type".to_string(),
                                        Value::String("text".to_string()),
                                    );
                                    content_text_map
                                        .insert("text".to_string(), Value::String(text_content));
                                    content_map.push(content_text_map);
                                }

                                message_map
                                    .insert("content".to_string(), Either::Right(content_map));
                                messages_vec.push(message_map);
                                image_urls.extend(image_urls_iter);
                                audio_urls.extend(audio_urls_iter);
                            }
                        }
                    }
                    if !image_urls.is_empty() || !audio_urls.is_empty() {
                        let mut images = Vec::new();
                        for url in image_urls {
                            let url_unparsed = url.trim();

                            let image = util::parse_image_url(url_unparsed)?;
                            images.push(image);
                        }
                        let mut audios = Vec::new();
                        for url in audio_urls {
                            let url_unparsed = url.trim();
                            let audio = util::parse_audio_url(url_unparsed)?;
                            audios.push(audio);
                        }
                        RequestMessage::VisionChat {
                            messages: messages_vec,
                            images,
                            audios,
                            enable_thinking: request.enable_thinking,
                            reasoning_effort: parse_reasoning_effort(&request.reasoning_effort),
                        }
                    } else {
                        RequestMessage::Chat {
                            messages: messages_vec,
                            enable_thinking: request.enable_thinking,
                            reasoning_effort: parse_reasoning_effort(&request.reasoning_effort),
                        }
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
                    RequestMessage::Chat {
                        messages,
                        enable_thinking: request.enable_thinking,
                        reasoning_effort: parse_reasoning_effort(&request.reasoning_effort),
                    }
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

            let model_request = _Request::Normal(Box::new(NormalRequest {
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
                    repetition_penalty: request.repetition_penalty,
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
                tool_choice,
                tools,
                logits_processors: None,
                return_raw_logits: false,
                web_search_options: request.web_search_options.clone(),
                model_id: model_id.clone(),
                truncate_sequence: request.truncate_sequence,
            }));

            let is_streaming = request.stream;
            let debug_repr = format!("{request:?}");
            drop(request);

            let runner = self.runner.clone();
            let send_recv_result = py
                .allow_threads(move || -> std::result::Result<either::Either<Response, Receiver<Response>>, String> {
                    MistralRs::maybe_log_request(runner.clone(), debug_repr);
                    let sender = runner
                        .get_sender(model_id.as_deref())
                        .map_err(|e| e.to_string())?;
                    sender
                        .blocking_send(model_request)
                        .map_err(|e| e.to_string())?;
                    if is_streaming {
                        Ok(either::Either::Right(rx))
                    } else {
                        let response = rx
                            .blocking_recv()
                            .ok_or_else(|| "Response channel closed unexpectedly".to_string())?;
                        Ok(either::Either::Left(response))
                    }
                })
                .map_err(PyApiErr::from)?;

            match send_recv_result {
                either::Either::Right(rx) => Ok(Either::Right(ChatCompletionStreamer::from_rx(rx))),
                either::Either::Left(response) => match response {
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
                    Response::Speech { .. } => unreachable!(),
                    Response::Raw { .. } => unreachable!(),
                    Response::Embeddings { .. } => unreachable!(),
                },
            }
        })
    }

    /// Send an embeddings request, returning embedding vectors in the same order they were provided.
    /// This returns the embeddings as [batch size, embedding dim]
    #[pyo3(signature = (request, model_id = None))]
    fn send_embedding_request(
        &mut self,
        request: Py<EmbeddingRequest>,
        model_id: Option<String>,
    ) -> PyApiResult<Vec<Vec<f32>>> {
        Python::with_gil(|py| {
            let (inputs, truncate_sequence, debug_repr) = {
                let request_ref = request.bind(py).borrow();
                (
                    request_ref.inputs.clone(),
                    request_ref.truncate_sequence,
                    format!("{:?}", &*request_ref),
                )
            };

            let runner = self.runner.clone();
            py.allow_threads(move || -> std::result::Result<Vec<Vec<f32>>, String> {
                MistralRs::maybe_log_request(runner.clone(), debug_repr);
                let sender = runner
                    .get_sender(model_id.as_deref())
                    .map_err(|e| e.to_string())?;

                let expected = match &inputs {
                    PythonEmbeddingInputs::Prompts(prompts) => prompts.len(),
                    PythonEmbeddingInputs::Tokens(batches) => batches.len(),
                };

                let mut receivers = Vec::with_capacity(expected);

                let mut enqueue = |message: RequestMessage| -> std::result::Result<(), String> {
                    let (tx, rx) = channel(1);
                    let request_id = {
                        let l = NEXT_REQUEST_ID.lock().unwrap();
                        let last = &mut *l.borrow_mut();
                        let last_v = *last;
                        *last += 1;
                        last_v
                    };

                    let model_request = _Request::Normal(Box::new(NormalRequest {
                        id: request_id,
                        messages: message,
                        sampling_params: SamplingParams::deterministic(),
                        response: tx,
                        return_logprobs: false,
                        is_streaming: false,
                        constraint: Constraint::None,
                        suffix: None,
                        tool_choice: None,
                        tools: None,
                        logits_processors: None,
                        return_raw_logits: false,
                        web_search_options: None,
                        model_id: model_id.clone(),
                        truncate_sequence,
                    }));

                    sender
                        .blocking_send(model_request)
                        .map_err(|e| e.to_string())?;
                    receivers.push(rx);
                    Ok(())
                };

                match inputs {
                    PythonEmbeddingInputs::Prompts(prompts) => {
                        for prompt in prompts {
                            enqueue(RequestMessage::Embedding { prompt })?;
                        }
                    }
                    PythonEmbeddingInputs::Tokens(batches) => {
                        for tokens in batches {
                            enqueue(RequestMessage::EmbeddingTokens { prompt: tokens })?;
                        }
                    }
                }

                let mut all_embeddings = Vec::with_capacity(receivers.len());

                for mut rx in receivers {
                    let response = rx.blocking_recv().ok_or_else(|| {
                        "Embedding response channel closed unexpectedly".to_string()
                    })?;

                    match response {
                        Response::Embeddings { embeddings, .. } => all_embeddings.push(embeddings),
                        Response::ValidationError(e) | Response::InternalError(e) => {
                            return Err(e.to_string())
                        }
                        Response::ModelError(msg, _) => return Err(msg.to_string()),
                        _ => {
                            return Err(
                                "Received unexpected response type from embeddings request."
                                    .to_string(),
                            )
                        }
                    }
                }

                Ok(all_embeddings)
            })
            .map_err(PyApiErr::from)
        })
    }

    /// Send an OpenAI API compatible request, returning the result.
    #[pyo3(signature = (request, model_id = None))]
    fn send_completion_request(
        &mut self,
        request: Py<CompletionRequest>,
        model_id: Option<String>,
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

            let model_request = _Request::Normal(Box::new(NormalRequest {
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
                    repetition_penalty: request.repetition_penalty,
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
                tool_choice,
                tools,
                logits_processors: None,
                return_raw_logits: false,
                web_search_options: None,
                model_id: model_id.clone(),
                truncate_sequence: request.truncate_sequence,
            }));

            let debug_repr = format!("{request:?}");
            drop(request);

            let runner = self.runner.clone();
            let response = py
                .allow_threads(move || -> std::result::Result<Response, String> {
                    MistralRs::maybe_log_request(runner.clone(), debug_repr);
                    let sender = runner
                        .get_sender(model_id.as_deref())
                        .map_err(|e| e.to_string())?;
                    sender
                        .blocking_send(model_request)
                        .map_err(|e| e.to_string())?;
                    rx.blocking_recv()
                        .ok_or_else(|| "Response channel closed unexpectedly".to_string())
                })
                .map_err(PyApiErr::from)?;

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
                Response::Speech { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
                Response::Embeddings { .. } => unreachable!(),
            }
        })
    }

    /// Generate an image.
    #[pyo3(signature = (
        prompt,
        response_format,
        height = 720,
        width = 1280,
        model_id = None,
        save_file = None,
    ))]
    fn generate_image(
        &self,
        py: Python<'_>,
        prompt: String,
        response_format: ImageGenerationResponseFormat,
        height: usize,
        width: usize,
        model_id: Option<String>,
        save_file: Option<PathBuf>,
    ) -> PyApiResult<ImageGenerationResponse> {
        let (tx, mut rx) = channel(1);

        let request = _Request::Normal(Box::new(NormalRequest {
            id: 0,
            messages: RequestMessage::ImageGeneration {
                prompt: prompt.to_string(),
                format: response_format,
                generation_params: DiffusionGenerationParams { height, width },
                save_file,
            },
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            suffix: None,
            constraint: Constraint::None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            model_id: model_id.clone(),
            truncate_sequence: false,
        }));

        let runner = self.runner.clone();
        let response = py
            .allow_threads(move || -> std::result::Result<Response, String> {
                let sender = runner
                    .get_sender(model_id.as_deref())
                    .map_err(|e| e.to_string())?;
                sender.blocking_send(request).map_err(|e| e.to_string())?;
                rx.blocking_recv()
                    .ok_or_else(|| "Channel was erroneously closed!".to_string())
            })
            .map_err(PyApiErr::from)?;

        let ResponseOk::ImageGeneration(response) = response.as_result()? else {
            return Err(PyApiErr::from("Got unexpected response type."));
        };

        Ok(response)
    }

    /// Generate audio.
    #[pyo3(signature = (
        prompt,
        model_id = None,
    ))]
    fn generate_audio(
        &self,
        py: Python<'_>,
        prompt: String,
        model_id: Option<String>,
    ) -> PyApiResult<SpeechGenerationResponse> {
        let (tx, mut rx) = channel(1);

        let request = _Request::Normal(Box::new(NormalRequest {
            id: 0,
            messages: RequestMessage::SpeechGeneration { prompt },
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            suffix: None,
            constraint: Constraint::None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            model_id: model_id.clone(),
            truncate_sequence: false,
        }));

        let runner = self.runner.clone();
        let response = py
            .allow_threads(move || -> std::result::Result<Response, String> {
                let sender = runner
                    .get_sender(model_id.as_deref())
                    .map_err(|e| e.to_string())?;
                sender.blocking_send(request).map_err(|e| e.to_string())?;
                rx.blocking_recv()
                    .ok_or_else(|| "Channel was erroneously closed!".to_string())
            })
            .map_err(PyApiErr::from)?;

        let ResponseOk::Speech {
            pcm,
            rate,
            channels,
        } = response.as_result()?
        else {
            return Err(PyApiErr::from("Got unexpected response type."));
        };

        Ok(SpeechGenerationResponse {
            pcm: (*pcm).clone(),
            rate,
            channels,
        })
    }

    /// Send a request to re-ISQ the model. If the model was loaded as GGUF or GGML
    /// then nothing will happen.
    #[pyo3(signature = (dtype, model_id = None))]
    fn send_re_isq(
        &self,
        py: Python<'_>,
        dtype: String,
        model_id: Option<String>,
    ) -> PyApiResult<()> {
        let request = _Request::ReIsq(parse_isq_value(&dtype, None)?);
        let runner = self.runner.clone();
        py.allow_threads(move || {
            runner
                .get_sender(model_id.as_deref())
                .map_err(|e| e.to_string())?
                .blocking_send(request)
                .map_err(|e| e.to_string())
        })
        .map_err(PyApiErr::from)
    }

    /// Tokenize some text, returning raw tokens.
    #[pyo3(signature = (text, add_special_tokens, enable_thinking, model_id = None))]
    fn tokenize_text(
        &self,
        py: Python<'_>,
        text: String,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        model_id: Option<String>,
    ) -> PyApiResult<Vec<u32>> {
        let (tx, mut rx) = channel(1);
        let request = _Request::Tokenize(TokenizationRequest {
            text: Either::Right(text),
            tools: None,
            add_generation_prompt: true,
            add_special_tokens,
            response: tx,
            enable_thinking,
            reasoning_effort: None,
        });

        let runner = self.runner.clone();
        py.allow_threads(
            move || -> std::result::Result<anyhow::Result<Vec<u32>>, String> {
                runner
                    .get_sender(model_id.as_deref())
                    .map_err(|e| e.to_string())?
                    .blocking_send(request)
                    .map_err(|e| e.to_string())?;
                rx.blocking_recv()
                    .ok_or_else(|| "Channel was erroneously closed!".to_string())
            },
        )
        .map_err(PyApiErr::from)?
        .map_err(PyApiErr::from)
    }

    /// Detokenize some tokens, returning text.
    #[pyo3(signature = (tokens, skip_special_tokens, model_id = None))]
    fn detokenize_text(
        &self,
        py: Python<'_>,
        tokens: Vec<u32>,
        skip_special_tokens: bool,
        model_id: Option<String>,
    ) -> PyApiResult<String> {
        let (tx, mut rx) = channel(1);
        let request = _Request::Detokenize(DetokenizationRequest {
            tokens,
            skip_special_tokens,
            response: tx,
        });

        let runner = self.runner.clone();
        py.allow_threads(
            move || -> std::result::Result<anyhow::Result<String>, String> {
                runner
                    .get_sender(model_id.as_deref())
                    .map_err(|e| e.to_string())?
                    .blocking_send(request)
                    .map_err(|e| e.to_string())?;
                rx.blocking_recv()
                    .ok_or_else(|| "Channel was erroneously closed!".to_string())
            },
        )
        .map_err(PyApiErr::from)?
        .map_err(PyApiErr::from)
    }

    /// List all available model IDs in multi-model mode (aliases if configured).
    fn list_models(&self) -> PyApiResult<Vec<String>> {
        self.runner.list_models().map_err(PyApiErr::from)
    }

    /// Return the maximum supported sequence length for the requested model, if available.
    #[pyo3(signature = (model_id = None))]
    fn max_sequence_length(&self, model_id: Option<String>) -> PyApiResult<Option<usize>> {
        self.runner
            .max_sequence_length(model_id.as_deref())
            .map_err(PyApiErr::from)
    }

    /// Get the default model ID in multi-model mode.
    fn get_default_model_id(&self) -> PyApiResult<Option<String>> {
        self.runner.get_default_model_id().map_err(PyApiErr::from)
    }

    /// Set the default model ID in multi-model mode.
    fn set_default_model_id(&self, model_id: String) -> PyApiResult<()> {
        self.runner
            .set_default_model_id(&model_id)
            .map_err(PyApiErr::from)
    }

    /// Remove a model by ID in multi-model mode.
    fn remove_model(&self, model_id: String) -> PyApiResult<()> {
        self.runner.remove_model(&model_id).map_err(PyApiErr::from)
    }

    /// Send an OpenAI API compatible request to a specific model, returning the result.
    fn send_chat_completion_request_to_model(
        &mut self,
        request: Py<ChatCompletionRequest>,
        model_id: String,
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
                    let mut audio_urls = Vec::new();
                    for message in messages {
                        let role = message["role"].as_ref().left().unwrap().clone();
                        match &message["content"] {
                            Either::Left(content) => {
                                let mut message_map: IndexMap<
                                    String,
                                    Either<String, Vec<IndexMap<String, Value>>>,
                                > = IndexMap::new();
                                message_map.insert("role".to_string(), Either::Left(role));
                                message_map.insert(
                                    "content".to_string(),
                                    Either::Left(content.to_string()),
                                );
                                messages_vec.push(message_map);
                            }
                            Either::Right(image_messages) => {
                                // If there is only one message, it is possible a text message
                                // found when rig is used as client. In this case, we need to check if
                                // the message is a text message or an image message.
                                if image_messages.len() == 1 {
                                    if !image_messages[0].contains_key("text") {
                                        return Err(PyApiErr::from(
                                            "Expected `text` key in input message.",
                                        ));
                                    }
                                    let content = match &image_messages[0]["text"] {
                                        Either::Left(left) => left.to_string(),
                                        Either::Right(right) => format!("{right:?}"),
                                    };
                                    let mut message_map: IndexMap<
                                        String,
                                        Either<String, Vec<IndexMap<String, Value>>>,
                                    > = IndexMap::new();
                                    message_map.insert("role".to_string(), Either::Left(role));
                                    message_map
                                        .insert("content".to_string(), Either::Left(content));
                                    messages_vec.push(message_map);
                                    continue;
                                }
                                if role != "user" {
                                    return Err(PyApiErr::from(
                                        "Role for an image message must be `user`, but it is {role}",
                                    ));
                                }

                                enum ContentPart {
                                    Text { text: String },
                                    Image { image_url: String },
                                    Audio { audio_url: String },
                                }

                                let mut items = Vec::new();
                                for image_message in image_messages {
                                    match image_message.get("type") {
                                        Some(Either::Left(x)) if x == "text" => {
                                            items.push(ContentPart::Text {
                                                text: image_message
                                                    .get("text").as_ref()
                                                    .context("Text sub-content must have `text` key.")?.as_ref()
                                                    .left().context("Text sub-content `text` key must be a string.")?.clone(),
                                            });
                                        }
                                        Some(Either::Left(x)) if x == "image_url" => {
                                            items.push(ContentPart::Image {
                                                image_url: image_message
                                                    .get("image_url")
                                                    .as_ref()
                                                    .context("Image sub-content must have `image_url` key.")?
                                                    .as_ref()
                                                    .right()
                                                    .context("Image sub-content `image_url` key must be an object.")?
                                                    .get("url")
                                                    .context("Image sub-content `image_url` object must have a `url` key.")?
                                                    .clone(),
                                            });
                                        }
                                        Some(Either::Left(x)) if x == "audio_url" => {
                                            items.push(ContentPart::Audio {
                                                audio_url: image_message
                                                    .get("audio_url")
                                                    .as_ref()
                                                    .context("Audio sub-content must have `audio_url` key.")?
                                                    .as_ref()
                                                    .right()
                                                    .context("Audio sub-content `audio_url` key must be an object.")?
                                                    .get("url")
                                                    .context("Audio sub-content `audio_url` object must have a `url` key.")?
                                                    .clone(),
                                            });
                                        }
                                        _ => return Err(PyApiErr::from("Expected array content sub-content to be of format {{`type`: `text`, `text`: ...}} and {{`type`: `url`, `image_url`: {{`url`: ...}}}}"))
                                    }
                                }

                                let text_content = items
                                    .iter()
                                    .filter_map(|item| match item {
                                        ContentPart::Text { text } => Some(text),
                                        _ => None,
                                    })
                                    .join(" ");
                                let image_urls_iter = items
                                    .iter()
                                    .filter_map(|item| match item {
                                        ContentPart::Image { image_url } => Some(image_url.clone()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>();

                                let audio_urls_iter = items
                                    .iter()
                                    .filter_map(|item| match item {
                                        ContentPart::Audio { audio_url } => Some(audio_url.clone()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>();

                                let mut message_map: IndexMap<
                                    String,
                                    Either<String, Vec<IndexMap<String, Value>>>,
                                > = IndexMap::new();
                                message_map.insert("role".to_string(), Either::Left(role));

                                let mut content_map: Vec<IndexMap<String, Value>> = Vec::new();
                                for _ in &image_urls_iter {
                                    let mut content_image_map = IndexMap::new();
                                    content_image_map.insert(
                                        "type".to_string(),
                                        Value::String("image".to_string()),
                                    );
                                    content_map.push(content_image_map);
                                }
                                for _ in &audio_urls_iter {
                                    let mut content_audio_map = IndexMap::new();
                                    content_audio_map.insert(
                                        "type".to_string(),
                                        Value::String("audio".to_string()),
                                    );
                                    content_map.push(content_audio_map);
                                }
                                {
                                    let mut content_text_map = IndexMap::new();
                                    content_text_map.insert(
                                        "type".to_string(),
                                        Value::String("text".to_string()),
                                    );
                                    content_text_map
                                        .insert("text".to_string(), Value::String(text_content));
                                    content_map.push(content_text_map);
                                }

                                message_map
                                    .insert("content".to_string(), Either::Right(content_map));
                                messages_vec.push(message_map);
                                image_urls.extend(image_urls_iter);
                                audio_urls.extend(audio_urls_iter);
                            }
                        }
                    }
                    if !image_urls.is_empty() || !audio_urls.is_empty() {
                        let mut images = Vec::new();
                        for url in image_urls {
                            let url_unparsed = url.trim();

                            let image = util::parse_image_url(url_unparsed)?;
                            images.push(image);
                        }
                        let mut audios = Vec::new();
                        for url in audio_urls {
                            let url_unparsed = url.trim();
                            let audio = util::parse_audio_url(url_unparsed)?;
                            audios.push(audio);
                        }
                        RequestMessage::VisionChat {
                            messages: messages_vec,
                            images,
                            audios,
                            enable_thinking: request.enable_thinking,
                            reasoning_effort: parse_reasoning_effort(&request.reasoning_effort),
                        }
                    } else {
                        RequestMessage::Chat {
                            messages: messages_vec,
                            enable_thinking: request.enable_thinking,
                            reasoning_effort: parse_reasoning_effort(&request.reasoning_effort),
                        }
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
                    RequestMessage::Chat {
                        messages,
                        enable_thinking: request.enable_thinking,
                        reasoning_effort: parse_reasoning_effort(&request.reasoning_effort),
                    }
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

            let model_request = _Request::Normal(Box::new(NormalRequest {
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
                    repetition_penalty: request.repetition_penalty,
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
                tool_choice,
                tools,
                logits_processors: None,
                return_raw_logits: false,
                web_search_options: request.web_search_options.clone(),
                model_id: Some(model_id.clone()),
                truncate_sequence: request.truncate_sequence,
            }));

            let is_streaming = request.stream;
            let debug_repr = format!("{request:?}");
            drop(request);

            let runner = self.runner.clone();
            let send_recv_result = py
                .allow_threads(move || -> std::result::Result<either::Either<Response, Receiver<Response>>, String> {
                    MistralRs::maybe_log_request(runner.clone(), debug_repr);
                    let sender = runner
                        .get_sender(Some(&model_id))
                        .map_err(|e| e.to_string())?;
                    sender
                        .blocking_send(model_request)
                        .map_err(|e| e.to_string())?;
                    if is_streaming {
                        Ok(either::Either::Right(rx))
                    } else {
                        let response = rx
                            .blocking_recv()
                            .ok_or_else(|| "Response channel closed unexpectedly".to_string())?;
                        Ok(either::Either::Left(response))
                    }
                })
                .map_err(PyApiErr::from)?;

            match send_recv_result {
                either::Either::Right(rx) => Ok(Either::Right(ChatCompletionStreamer::from_rx(rx))),
                either::Either::Left(response) => match response {
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
                    Response::Speech { .. } => unreachable!(),
                    Response::Raw { .. } => unreachable!(),
                    Response::Embeddings { .. } => unreachable!(),
                },
            }
        })
    }

    /// Send an OpenAI API compatible completion request to a specific model, returning the result.
    fn send_completion_request_to_model(
        &mut self,
        request: Py<CompletionRequest>,
        model_id: String,
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

            let model_request = _Request::Normal(Box::new(NormalRequest {
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
                    repetition_penalty: request.repetition_penalty,
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
                tool_choice,
                tools,
                logits_processors: None,
                return_raw_logits: false,
                web_search_options: None,
                model_id: Some(model_id.clone()),
                truncate_sequence: request.truncate_sequence,
            }));

            let debug_repr = format!("{request:?}");
            drop(request);

            let runner = self.runner.clone();
            let response = py
                .allow_threads(move || -> std::result::Result<Response, String> {
                    MistralRs::maybe_log_request(runner.clone(), debug_repr);
                    let sender = runner
                        .get_sender(Some(&model_id))
                        .map_err(|e| e.to_string())?;
                    sender
                        .blocking_send(model_request)
                        .map_err(|e| e.to_string())?;
                    rx.blocking_recv()
                        .ok_or_else(|| "Response channel closed unexpectedly".to_string())
                })
                .map_err(PyApiErr::from)?;

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
                Response::Speech { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
                Response::Embeddings { .. } => unreachable!(),
            }
        })
    }

    /// Unload a model from memory while preserving its configuration for later reload.
    /// The model can be reloaded automatically when a request is sent to it, or manually
    /// using `reload_model()`.
    fn unload_model(&self, model_id: String) -> PyApiResult<()> {
        self.runner.unload_model(&model_id).map_err(PyApiErr::from)
    }

    /// Manually reload a previously unloaded model.
    fn reload_model(&self, py: Python<'_>, model_id: String) -> PyApiResult<()> {
        let runner = self.runner.clone();
        py.allow_threads(move || {
            runner
                .reload_model_blocking(&model_id)
                .map_err(|e| e.to_string())
        })
        .map_err(PyApiErr::from)
    }

    /// List all unloaded model IDs.
    fn list_unloaded_models(&self) -> PyApiResult<Vec<String>> {
        self.runner.list_unloaded_models().map_err(PyApiErr::from)
    }

    /// Check if a model is currently loaded (as opposed to unloaded).
    fn is_model_loaded(&self, model_id: String) -> PyApiResult<bool> {
        self.runner
            .is_model_loaded(&model_id)
            .map_err(PyApiErr::from)
    }

    /// Get the status of a model: "loaded", "unloaded", "reloading", or None if not found.
    fn get_model_status(&self, model_id: String) -> PyApiResult<Option<String>> {
        self.runner
            .get_model_status(&model_id)
            .map(|s| s.map(|s| s.to_string()))
            .map_err(PyApiErr::from)
    }

    /// List all models with their status (loaded, unloaded, reloading).
    fn list_models_with_status(&self) -> PyApiResult<Vec<(String, String)>> {
        self.runner
            .list_models_with_status()
            .map(|v| v.into_iter().map(|(id, s)| (id, s.to_string())).collect())
            .map_err(PyApiErr::from)
    }
}

/// MCP server source configuration for different transport types
#[pyclass]
#[derive(Debug, Clone)]
pub enum McpServerSourcePy {
    /// HTTP-based MCP server
    #[pyo3(constructor = (url, timeout_secs, headers))]
    Http {
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<std::collections::HashMap<String, String>>,
    },
    /// Process-based MCP server
    #[pyo3(constructor = (command, args, work_dir, env))]
    Process {
        command: String,
        args: Vec<String>,
        work_dir: Option<String>,
        env: Option<std::collections::HashMap<String, String>>,
    },
    /// WebSocket-based MCP server
    #[pyo3(constructor = (url, timeout_secs, headers))]
    WebSocket {
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<std::collections::HashMap<String, String>>,
    },
}

impl From<McpServerSourcePy> for McpServerSource {
    fn from(source: McpServerSourcePy) -> Self {
        match source {
            McpServerSourcePy::Http {
                url,
                timeout_secs,
                headers,
            } => McpServerSource::Http {
                url,
                timeout_secs,
                headers,
            },
            McpServerSourcePy::Process {
                command,
                args,
                work_dir,
                env,
            } => McpServerSource::Process {
                command,
                args,
                work_dir,
                env,
            },
            McpServerSourcePy::WebSocket {
                url,
                timeout_secs,
                headers,
            } => McpServerSource::WebSocket {
                url,
                timeout_secs,
                headers,
            },
        }
    }
}

/// Configuration for an individual MCP server
#[pyclass]
#[derive(Debug, Clone)]
pub struct McpServerConfigPy {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub source: McpServerSourcePy,
    #[pyo3(get, set)]
    pub enabled: bool,
    #[pyo3(get, set)]
    pub tool_prefix: Option<String>,
    #[pyo3(get, set)]
    pub resources: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub bearer_token: Option<String>,
}

#[pymethods]
impl McpServerConfigPy {
    #[new]
    #[pyo3(signature = (id, name, source, enabled=true, tool_prefix=None, resources=None, bearer_token=None))]
    pub fn new(
        id: String,
        name: String,
        source: McpServerSourcePy,
        enabled: bool,
        tool_prefix: Option<String>,
        resources: Option<Vec<String>>,
        bearer_token: Option<String>,
    ) -> Self {
        Self {
            id,
            name,
            source,
            enabled,
            tool_prefix,
            resources,
            bearer_token,
        }
    }
}

impl From<McpServerConfigPy> for McpServerConfig {
    fn from(config: McpServerConfigPy) -> Self {
        McpServerConfig {
            id: config.id,
            name: config.name,
            source: config.source.into(),
            enabled: config.enabled,
            tool_prefix: config.tool_prefix,
            resources: config.resources,
            bearer_token: config.bearer_token,
        }
    }
}

/// Configuration for MCP client integration
#[pyclass]
#[derive(Debug, Clone)]
pub struct McpClientConfigPy {
    #[pyo3(get, set)]
    pub servers: Vec<McpServerConfigPy>,
    #[pyo3(get, set)]
    pub auto_register_tools: bool,
    #[pyo3(get, set)]
    pub tool_timeout_secs: Option<u64>,
    #[pyo3(get, set)]
    pub max_concurrent_calls: Option<usize>,
}

#[pymethods]
impl McpClientConfigPy {
    #[new]
    #[pyo3(signature = (servers, auto_register_tools=true, tool_timeout_secs=None, max_concurrent_calls=None))]
    pub fn new(
        servers: Vec<McpServerConfigPy>,
        auto_register_tools: bool,
        tool_timeout_secs: Option<u64>,
        max_concurrent_calls: Option<usize>,
    ) -> Self {
        Self {
            servers,
            auto_register_tools,
            tool_timeout_secs,
            max_concurrent_calls,
        }
    }
}

impl From<McpClientConfigPy> for McpClientConfig {
    fn from(config: McpClientConfigPy) -> Self {
        McpClientConfig {
            servers: config.servers.into_iter().map(|s| s.into()).collect(),
            auto_register_tools: config.auto_register_tools,
            tool_timeout_secs: config.tool_timeout_secs,
            max_concurrent_calls: config.max_concurrent_calls,
        }
    }
}

#[pymodule]
fn mistralrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    initialize_logging();

    m.add_class::<Runner>()?;
    m.add_class::<Which>()?;
    m.add_class::<ChatCompletionRequest>()?;
    m.add_class::<CompletionRequest>()?;
    m.add_class::<EmbeddingRequest>()?;
    m.add_class::<Architecture>()?;
    m.add_class::<which::EmbeddingArchitecture>()?;
    m.add_class::<VisionArchitecture>()?;
    m.add_class::<DiffusionArchitecture>()?;
    m.add_class::<AnyMoeConfig>()?;
    m.add_class::<AnyMoeExpertType>()?;
    m.add_class::<ToolChoice>()?;
    m.add_class::<SpeechGenerationResponse>()?;
    m.add_class::<SpeechLoaderType>()?;

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
    m.add_class::<McpServerSourcePy>()?;
    m.add_class::<McpServerConfigPy>()?;
    m.add_class::<McpClientConfigPy>()?;
    Ok(())
}
