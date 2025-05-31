/// Builder for mistral.rs for server
use std::num::NonZeroUsize;

use anyhow::{Context, Result};
use candle_core::Device;
use mistralrs_core::{
    get_auto_device_map_params, get_model_dtype, get_tgt_non_granular_index, paged_attn_supported,
    parse_isq_value, AutoDeviceMapParams, BertEmbeddingModel, DefaultSchedulerMethod,
    DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting, Loader, LoaderBuilder,
    MemoryGpuConfig, MistralRsBuilder, ModelSelected, PagedAttentionConfig, SchedulerConfig,
    TokenSource,
};
use tracing::info;

use crate::{
    defaults,
    types::{LoadedPipeline, SharedMistralState},
};

pub struct MistralRsForServerBuilder {
    device: Option<Device>,

    // Args
    /// IP to serve on.
    // serve_ip: Option<String>,

    /// Integer seed to ensure reproducible random number generation.
    seed: Option<u64>,

    /// Port to serve on.
    // port: Option<String>,

    /// Log all responses and requests to this file
    log: Option<String>,

    /// If a sequence is larger than the maximum model length, truncate the number
    /// of tokens such that the sequence will fit at most the maximum length.
    /// If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.
    truncate_sequence: bool,

    /// Model selector
    model: Option<ModelSelected>,

    /// Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1.
    max_seqs: usize,

    /// Use no KV cache.
    no_kv_cache: bool,

    /// Chat template file with a JINJA file with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    /// Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
    chat_template: Option<String>,

    /// Explicit JINJA chat template file (.jinja) to be used. If specified, this overrides all other chat templates.
    jinja_explicit: Option<String>,

    /// Source of the token for authentication.
    /// Can be in the formats: `literal:<value>`, `env:<value>`, `path:<value>`, `cache` to use a cached token, or `none` to use no token.
    /// Defaults to `cache`.
    token_source: TokenSource,

    /// Enter interactive mode instead of serving a chat server.
    interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    prefix_cache_n: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    in_situ_quant: Option<String>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    /// This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
    paged_ctxt_len: Option<usize>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    paged_attn_block_size: Option<usize>,

    /// Disable PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
    no_paged_attn: bool,

    /// Enable PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
    paged_attn: bool,

    /// Number of tokens to batch the prompt step into. This can help with OOM errors when in the prompt step, but reduces performance.
    prompt_chunksize: Option<usize>,

    /// Use CPU only
    cpu: bool,

    /// Enable searching compatible with the OpenAI `web_search_options` setting. This uses the BERT model specified below or the default.
    enable_search: bool,

    /// Specify a Hugging Face model ID for a BERT model to assist web searching. Defaults to Snowflake Arctic Embed L.
    search_bert_model: Option<String>,
    // Enable thinking for interactive mode and models that support it.
    // enable_thinking: bool,
}

impl Default for MistralRsForServerBuilder {
    fn default() -> Self {
        Self {
            device: defaults::DEVICE,
            // serve_ip: None,
            seed: defaults::SEED,
            // port: None,
            log: defaults::LOG,
            truncate_sequence: defaults::TRUNCATE_SEQUENCE,
            model: defaults::MODEL,
            max_seqs: defaults::MAX_SEQS,
            no_kv_cache: defaults::NO_KV_CACHE,
            chat_template: defaults::CHAT_TEMPLATE,
            jinja_explicit: defaults::JINJA_EXPLICIT,
            token_source: defaults::DEFAULT_TOKEN_SOURCE,
            interactive_mode: defaults::INTERACTIVE_MODE,
            prefix_cache_n: defaults::PREFIX_CACHE_N,
            num_device_layers: defaults::NUM_DEVICE_LAYERS,
            in_situ_quant: defaults::IN_SITU_QUANT,
            paged_attn_gpu_mem: defaults::PAGED_ATTN_GPU_MEM,
            paged_attn_gpu_mem_usage: defaults::PAGED_ATTN_GPU_MEM_USAGE,
            paged_ctxt_len: defaults::PAGED_CTXT_LEN,
            paged_attn_block_size: defaults::PAGED_ATTN_BLOCK_SIZE,
            no_paged_attn: defaults::NO_PAGED_ATTN,
            paged_attn: defaults::PAGED_ATTN,
            prompt_chunksize: defaults::PROMPT_CHUNKSIZE,
            cpu: defaults::CPU,
            enable_search: defaults::ENABLE_SEARCH,
            search_bert_model: defaults::SEARCH_BERT_MODEL,
            // enable_thinking: server_defaults::ENABLE_THINKING,
        }
    }
}

impl MistralRsForServerBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_log(mut self, log: String) -> Self {
        self.log = Some(log);
        self
    }

    pub fn with_truncate_sequence(mut self, truncate_sequence: bool) -> Self {
        self.truncate_sequence = truncate_sequence;
        self
    }

    pub fn with_model(mut self, model: ModelSelected) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_max_seqs(mut self, max_seqs: usize) -> Self {
        self.max_seqs = max_seqs;
        self
    }

    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }

    pub fn with_chat_template(mut self, chat_template: String) -> Self {
        self.chat_template = Some(chat_template);
        self
    }

    pub fn with_jinja_explicit(mut self, jinja_explicit: String) -> Self {
        self.jinja_explicit = Some(jinja_explicit);
        self
    }

    pub fn with_token_source(mut self, token_source: TokenSource) -> Self {
        self.token_source = token_source;
        self
    }

    pub fn with_interactive_mode(mut self, interactive_mode: bool) -> Self {
        self.interactive_mode = interactive_mode;
        self
    }

    pub fn with_prefix_cache_n(mut self, prefix_cache_n: usize) -> Self {
        self.prefix_cache_n = prefix_cache_n;
        self
    }

    pub fn with_num_device_layers(mut self, num_device_layers: Vec<String>) -> Self {
        self.num_device_layers = Some(num_device_layers);
        self
    }

    pub fn with_in_situ_quant(mut self, in_situ_quant: String) -> Self {
        self.in_situ_quant = Some(in_situ_quant);
        self
    }

    pub fn with_paged_attn_gpu_mem(mut self, paged_attn_gpu_mem: usize) -> Self {
        self.paged_attn_gpu_mem = Some(paged_attn_gpu_mem);
        self
    }

    pub fn with_paged_attn_gpu_mem_usage(mut self, paged_attn_gpu_mem_usage: f32) -> Self {
        self.paged_attn_gpu_mem_usage = Some(paged_attn_gpu_mem_usage);
        self
    }

    pub fn with_paged_ctxt_len(mut self, paged_ctxt_len: usize) -> Self {
        self.paged_ctxt_len = Some(paged_ctxt_len);
        self
    }

    pub fn with_paged_attn_block_size(mut self, paged_attn_block_size: usize) -> Self {
        self.paged_attn_block_size = Some(paged_attn_block_size);
        self
    }

    pub fn with_no_paged_attn(mut self, no_paged_attn: bool) -> Self {
        self.no_paged_attn = no_paged_attn;
        self
    }

    pub fn with_paged_attn(mut self, paged_attn: bool) -> Self {
        self.paged_attn = paged_attn;
        self
    }

    pub fn with_prompt_chunksize(mut self, prompt_chunksize: usize) -> Self {
        self.prompt_chunksize = Some(prompt_chunksize);
        self
    }

    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn with_enable_search(mut self, enable_search: bool) -> Self {
        self.enable_search = enable_search;
        self
    }

    pub fn with_search_bert_model(mut self, search_bert_model: String) -> Self {
        self.search_bert_model = Some(search_bert_model);
        self
    }

    pub async fn build(mut self) -> Result<SharedMistralState> {
        // This was originally with the device config
        if self.cpu {
            self.no_paged_attn = true;
        }

        let model = self.model.context("Model was None")?;

        let tgt_non_granular_index = get_tgt_non_granular_index(&model);
        let dtype = get_model_dtype(&model)?;
        let auto_device_map_params = get_auto_device_map_params(&model)?;

        if tgt_non_granular_index.is_some() {
            self.max_seqs = 1;
        }

        let prompt_chunksize = match self.prompt_chunksize {
            Some(0) => {
                anyhow::bail!("`prompt_chunksize` must be a strictly positive integer, got 0.",)
            }
            Some(x) => Some(NonZeroUsize::new(x).unwrap()),
            None => None,
        };

        let max_seq_len = auto_device_map_params.max_seq_len();

        let device = if let Some(device) = self.device {
            device
        } else {
            init_device(self.cpu, self.seed)?
        };

        let mapper = init_mapper(&self.num_device_layers, &auto_device_map_params);
        let no_paged_attn = configure_no_paged_attn(&device, self.no_paged_attn, self.paged_attn);

        // Allocate 0.5 GB of CPU memory just as a placeholder.
        // Nothing happens here as we have no `swap_out`, see `_preempt_by_swap`.
        let cache_config = init_cache_config(
            self.paged_attn_block_size,
            self.paged_attn_gpu_mem,
            self.paged_attn_gpu_mem_usage,
            self.paged_ctxt_len,
            no_paged_attn,
            max_seq_len,
        )?;

        // Configure this last to prevent arg moves
        let loader: Box<dyn Loader> = LoaderBuilder::new(model)
            .with_no_kv_cache(self.no_kv_cache)
            .with_chat_template(self.chat_template)
            .with_prompt_chunksize(prompt_chunksize)
            .with_jinja_explicit(self.jinja_explicit)
            .build()?;

        mistralrs_instance_info(&loader);

        let isq = self
            .in_situ_quant
            .as_ref()
            .and_then(|isq| parse_isq_value(isq, Some(&device)).ok());

        let pipeline: LoadedPipeline = loader.load_model_from_hf(
            None,
            self.token_source,
            &dtype,
            &device,
            false,
            mapper,
            isq,
            cache_config,
        )?;
        info!("Model loaded.");

        let scheduler_config = init_scheduler_config(&cache_config, &pipeline, self.max_seqs).await;

        let bert_model = if self.enable_search {
            Some(
                self.search_bert_model
                    .map(BertEmbeddingModel::Custom)
                    .unwrap_or_default(),
            )
        } else {
            None
        };

        let mistralrs = MistralRsBuilder::new(
            pipeline,
            scheduler_config,
            !self.interactive_mode,
            bert_model,
        )
        .with_opt_log(self.log)
        .with_truncate_sequence(self.truncate_sequence)
        .with_no_kv_cache(self.no_kv_cache)
        .with_prefix_cache_n(self.prefix_cache_n)
        .build();

        Ok(mistralrs)
    }
}

// TODO replace with best device?
fn init_device(force_cpu: bool, seed: Option<u64>) -> Result<candle_core::Device> {
    #[cfg(feature = "metal")]
    let device = if force_cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
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
fn mistralrs_instance_info(loader: &Box<dyn Loader>) {
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
