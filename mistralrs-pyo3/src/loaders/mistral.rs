use std::fs::File;

use candle_core::Device;
use mistralrs::{
    Loader, MistralLoader as _MistralLoader, MistralRs, MistralSpecificConfig,
    ModelKind as _ModelKind, SchedulerMethod, TokenSource,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{MistralRunner, ModelKind};

#[pyclass]
pub struct MistralLoader {
    loader: _MistralLoader,
    no_kv_cache: bool,
}

#[pymethods]
impl MistralLoader {
    #[new]
    #[pyo3(signature = (model_id, kind, no_kv_cache=false, use_flash_attn=cfg!(feature="flash-attn"), repeat_last_n=64, order_file=None, quantized_model_id=None,quantized_filename=None,xlora_model_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_id: String,
        kind: Py<ModelKind>,
        no_kv_cache: bool,
        mut use_flash_attn: bool,
        repeat_last_n: usize,
        order_file: Option<String>,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        xlora_model_id: Option<String>,
    ) -> PyResult<Self> {
        use_flash_attn = use_flash_attn & cfg!(feature="flash-attn");
        let order = if let Some(order_file) = order_file {
            let f = File::open(order_file);
            let f = match f {
                Ok(x) => x,
                Err(y) => return Err(PyValueError::new_err(y)),
            };
            match serde_json::from_reader(f) {
                Ok(x) => Some(x),
                Err(y) => return Err(PyValueError::new_err(y.to_string())),
            }
        } else {
            None
        };
        Ok(Self {
            loader: _MistralLoader::new(
                model_id,
                MistralSpecificConfig {
                    use_flash_attn,
                    repeat_last_n,
                },
                quantized_model_id,
                quantized_filename,
                xlora_model_id,
                Python::with_gil(|py| match &*kind.as_ref(py).borrow() {
                    ModelKind::Normal => _ModelKind::Normal,
                    ModelKind::XLoraNormal => _ModelKind::XLoraNormal,
                    ModelKind::QuantizedGGUF => _ModelKind::QuantizedGGUF,
                    ModelKind::QuantizedGGML => _ModelKind::QuantizedGGML,
                    ModelKind::XLoraGGUF => _ModelKind::XLoraGGUF,
                    ModelKind::XLoraGGML => _ModelKind::XLoraGGML,
                }),
                order,
                no_kv_cache,
            ),
            no_kv_cache,
        })
    }

    /// Specify token source and token source value as the following pairing:
    /// "cache" -> None
    /// "literal" -> str
    /// "envvar" -> str
    /// "path" -> str
    ///
    /// `log`:
    /// Log all responses and requests to this file
    ///
    /// `truncate_sequence`:
    /// If a sequence is larger than the maximum model length, truncate the number
    /// of tokens such that the sequence will fit at most the maximum length.
    /// If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.
    ///
    /// `max_seqs`:
    /// Maximum running sequences at any time
    ///
    /// `no_kv_cache`:
    /// Use no KV cache.
    #[pyo3(signature = (token_source = "cache", max_seqs = 2, truncate_sequence = false, logfile = None, revision = None, token_source_value = None))]
    fn load(
        &mut self,
        token_source: &str,
        max_seqs: usize,
        truncate_sequence: bool,
        logfile: Option<String>,
        revision: Option<String>,
        token_source_value: Option<String>,
    ) -> PyResult<MistralRunner> {
        println!("Loading");
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0);
        #[cfg(not(feature = "metal"))]
        let device = Device::cuda_if_available(0);
        dbg!(&device);
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

        let res = self.loader.load_model(revision, source, None, &device);
        let (pipeline, conversation) = match res {
            Ok(x) => x,
            Err(y) => return Err(PyValueError::new_err(y.to_string())),
        };

        let mistralrs = MistralRs::new(
            pipeline,
            SchedulerMethod::Fixed(max_seqs.try_into().unwrap()),
            logfile,
            truncate_sequence,
            self.no_kv_cache,
        );

        Ok(MistralRunner {
            runner: mistralrs,
            conversation,
        })
    }
}
