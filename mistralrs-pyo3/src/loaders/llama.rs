use candle_core::DType as _DType;
use mistralrs::{
    LlamaLoader as _LlamaLoader, LlamaSpecificConfig, Loader, MistralRs, ModelKind as _ModelKind,
    SchedulerMethod, TokenSource,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fs::File;

use crate::{get_device, DType, ModelKind, Runner};

#[pyclass]
/// A loader for a Runner.
pub struct LlamaLoader {
    loader: _LlamaLoader,
    no_kv_cache: bool,
    tgt_non_granular_index: Option<usize>,
}

#[pymethods]
impl LlamaLoader {
    /// - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    /// - `kind`: Model kind
    /// - `no_kv_cache=False`: Disable kv cache.
    /// - `use_flash_attn=<feature>`: Use flash attn, only used if feature is enabled.
    /// - `repeat_last_n=64`: Repeat last n context window.
    /// - `gqa=1`: GQA, irrelevant if non quantized model type.
    /// - `order_file=None`: Ordering JSON file.
    /// - `quantized_model_id=None`: Quantized model ID.
    /// - `quantized_filename=None`: Quantized filename (gguf/ggml),
    /// - `xlora_model_id=None`: X-LoRA model
    /// - `chat_template=None`: Chat template literal or file.
    /// - `tokenizer_json=None`: Tokenizer json file.
    /// - `tgt_non_granular_index=None`: Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached. If this is set then the max running sequences will be set to 1.
    #[new]
    #[pyo3(signature = (
        model_id,
        kind,
        no_kv_cache=false,
        use_flash_attn=cfg!(feature="flash-attn"),
        repeat_last_n=64,
        gqa=1,
        order_file=None,
        quantized_model_id=None,
        quantized_filename=None,
        xlora_model_id=None,
        chat_template=None,
        tokenizer_json=None,
        tgt_non_granular_index=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_id: String,
        kind: Py<ModelKind>,
        no_kv_cache: bool,
        use_flash_attn: Option<bool>,
        repeat_last_n: usize,
        gqa: usize,
        order_file: Option<String>,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        xlora_model_id: Option<String>,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        tgt_non_granular_index: Option<usize>,
    ) -> PyResult<Self> {
        let mut use_flash_attn = use_flash_attn.unwrap_or(false);
        use_flash_attn &= cfg!(feature = "flash-attn");
        let order = if let Some(ref order_file) = order_file {
            let f = File::open(order_file.clone())
                .unwrap_or_else(|_| panic!("Could not load ordering file at {order_file}"));
            match serde_json::from_reader(f) {
                Ok(x) => Some(x),
                Err(y) => return Err(PyValueError::new_err(y.to_string())),
            }
        } else {
            None
        };
        let kind = Python::with_gil(|py| match &*kind.bind(py).borrow() {
            ModelKind::Normal => _ModelKind::Normal,
            ModelKind::XLoraNormal => _ModelKind::XLoraNormal,
            ModelKind::QuantizedGGUF => _ModelKind::QuantizedGGUF,
            ModelKind::QuantizedGGML => _ModelKind::QuantizedGGML,
            ModelKind::XLoraGGUF => _ModelKind::XLoraGGUF,
            ModelKind::XLoraGGML => _ModelKind::XLoraGGML,
        });
        if matches!(kind, _ModelKind::Normal)
            && (order_file.is_some()
                || quantized_model_id.is_some()
                || quantized_filename.is_some()
                || xlora_model_id.is_some())
        {
            return Err(PyValueError::new_err("Expected no order file, no quantized model id, no quantized filename, and no xlora model id."));
        } else if matches!(kind, _ModelKind::XLoraNormal)
            && (order_file.is_none()
                || quantized_model_id.is_some()
                || quantized_filename.is_some()
                || xlora_model_id.is_none())
        {
            return Err(PyValueError::new_err("Expected an order file and xlora model id but no quantized model id and no quantized filename."));
        } else if (matches!(kind, _ModelKind::QuantizedGGUF)
            || matches!(kind, _ModelKind::QuantizedGGML))
            && (order_file.is_some()
                || quantized_model_id.is_none()
                || quantized_filename.is_none()
                || xlora_model_id.is_some())
        {
            return Err(PyValueError::new_err("Expected a quantized model id and quantized filename but no order file and no xlora model id."));
        } else if (matches!(kind, _ModelKind::XLoraGGUF) || matches!(kind, _ModelKind::XLoraGGML))
            && (order_file.is_none()
                || quantized_model_id.is_none()
                || quantized_filename.is_none()
                || xlora_model_id.is_none())
        {
            return Err(PyValueError::new_err("Expected a quantized model id and quantized filename and order file and xlora model id."));
        }
        Ok(Self {
            loader: _LlamaLoader::new(
                model_id,
                LlamaSpecificConfig {
                    use_flash_attn,
                    repeat_last_n,
                    gqa,
                },
                quantized_model_id,
                quantized_filename,
                xlora_model_id,
                kind,
                order,
                no_kv_cache,
                chat_template,
                tokenizer_json,
                tgt_non_granular_index,
            ),
            no_kv_cache,
            tgt_non_granular_index,
        })
    }

    /// Load a model.
    ///
    /// - `token_source="cache"`
    /// Specify token source and token source value as the following pairing:
    /// "cache" -> None
    /// "literal" -> str
    /// "envvar" -> str
    /// "path" -> str
    ///
    /// - `max_seqs=16`: Maximum running sequences at any time.
    ///
    /// - `prefix_cache_n=16`: Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    ///
    /// - `truncate_sequence=False`:
    /// If a sequence is larger than the maximum model length, truncate the number
    /// of tokens such that the sequence will fit at most the maximum length.
    /// If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.
    ///
    /// - `logfile=None`: Log all responses and requests to this file.
    ///
    /// - `revision=None`: HF revision.
    ///
    /// - `token_source_value=None`: Value of token source value for `token_source`
    ///
    /// - `dtype=None`: Datatype to load the model into, only applicable for non-quantized models.
    #[pyo3(signature = (token_source = "cache", max_seqs = 16, prefix_cache_n = 16, truncate_sequence = false, logfile = None, revision = None, token_source_value = None, dtype = None))]
    #[allow(clippy::too_many_arguments)]
    fn load(
        &mut self,
        token_source: &str,
        max_seqs: usize,
        prefix_cache_n: usize,
        truncate_sequence: bool,
        logfile: Option<String>,
        revision: Option<String>,
        token_source_value: Option<String>,
        dtype: Option<DType>,
    ) -> PyResult<Runner> {
        let device = get_device();
        let device = match device {
            Ok(x) => x,
            Err(y) => return Err(PyValueError::new_err(y.to_string())),
        };

        let source = match (token_source, &token_source_value) {
            ("cache", None) => TokenSource::CacheToken,
            ("literal", Some(v)) => TokenSource::Literal(v.clone()),
            ("envvar", Some(env)) => TokenSource::EnvVar(env.clone()),
            ("path", Some(p)) => TokenSource::Path(p.clone()),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "'{token_source}' and '{token_source_value:?}' are not compatible."
                )))
            }
        };

        let dtype = dtype.map(|dtype| match dtype {
            DType::BF16 => _DType::BF16,
            DType::F16 => _DType::F16,
            DType::F32 => _DType::F32,
            DType::F64 => _DType::F64,
            DType::U8 => _DType::U8,
            DType::U32 => _DType::U32,
            DType::I64 => _DType::I64,
        });

        let res = self.loader.load_model(revision, source, dtype, &device);
        let pipeline = match res {
            Ok(x) => x,
            Err(y) => return Err(PyValueError::new_err(y.to_string())),
        };

        let maxseqs = if self.tgt_non_granular_index.is_some() {
            1
        } else {
            max_seqs
        };

        let mistralrs = MistralRs::new(
            pipeline,
            SchedulerMethod::Fixed(maxseqs.try_into().unwrap()),
            logfile,
            truncate_sequence,
            self.no_kv_cache,
            prefix_cache_n,
        );

        Ok(Runner { runner: mistralrs })
    }
}
