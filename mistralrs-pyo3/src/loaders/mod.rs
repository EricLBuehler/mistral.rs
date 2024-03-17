use pyo3::{
    intern, pyclass, pymethods,
    types::{PyTuple, PyType},
    IntoPy, Py, PyAny, PyResult, Python,
};

use crate::{DType, ModelKind};

pub mod gemma;
pub mod llama;
pub mod mistral;
pub mod mixtral;

#[pyclass]
pub struct NormalLoader {
    inner: Py<PyAny>,
}

#[pymethods]
impl NormalLoader {
    /// - `class`: Loader class.
    /// - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    /// - `no_kv_cache=False`: Disable kv cache.
    /// - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
    /// - `repeat_last_n=64`: Repeat last n context window.
    /// - `gqa=None`: GQA, irrelevant if non quantized model type.
    /// - `chat_template=None`: Chat template literal or file.
    /// - `tokenizer_json=None`: Tokenizer json file.
    #[new]
    #[pyo3(signature = (
        class,
        model_id,
        no_kv_cache=false,
        use_flash_attn=None,
        repeat_last_n=64,
        gqa=None,
        chat_template=None,
        tokenizer_json=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        class: Py<PyType>,
        model_id: String,
        no_kv_cache: bool,
        mut use_flash_attn: Option<bool>,
        repeat_last_n: usize,
        gqa: Option<usize>,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
    ) -> PyResult<Self> {
        use_flash_attn = use_flash_attn.map(|x| x & cfg!(feature = "flash-attn"));
        let kind = ModelKind::Normal;

        let loader = Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                model_id.into_py(py),
                kind.into_py(py),
                no_kv_cache.into_py(py),
                use_flash_attn.into_py(py),
                repeat_last_n.into_py(py),
                gqa.into_py(py),
                py.None(), // order_file
                py.None(), // quantized_model_id
                py.None(), // quantized_filename
                py.None(), // xlora_model_id
                chat_template.into_py(py),
                tokenizer_json.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            class.call1(py, args)
        })?;

        Ok(Self { inner: loader })
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
    #[pyo3(signature = (token_source = "cache", max_seqs = 16, truncate_sequence = false, logfile = None, revision = None, token_source_value = None, dtype = None))]
    #[allow(clippy::too_many_arguments)]
    fn load(
        &mut self,
        token_source: &str,
        max_seqs: usize,
        truncate_sequence: bool,
        logfile: Option<String>,
        revision: Option<String>,
        token_source_value: Option<String>,
        dtype: Option<DType>,
    ) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                token_source.into_py(py),
                max_seqs.into_py(py),
                truncate_sequence.into_py(py),
                logfile.into_py(py),
                revision.into_py(py),
                token_source_value.into_py(py),
                dtype.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            self.inner.call_method1(py, intern!(py, "load"), args)
        })
    }
}

#[pyclass]
pub struct XLoraLoader {
    inner: Py<PyAny>,
}

#[pymethods]
impl XLoraLoader {
    /// - `class`: Loader class.
    /// - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    /// - `no_kv_cache=False`: Disable kv cache.
    /// - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
    /// - `repeat_last_n=64`: Repeat last n context window.
    /// - `gqa=None`: GQA, irrelevant if non quantized model type.
    /// - `order_file=None`: Ordering JSON file.
    /// - `xlora_model_id=None`: X-LoRA model
    /// - `chat_template=None`: Chat template literal or file.
    /// - `tokenizer_json=None`: Tokenizer json file.
    #[new]
    #[pyo3(signature = (
        class,
        model_id,
        no_kv_cache=false,
        use_flash_attn=None,
        repeat_last_n=64,
        gqa=None,
        order_file=None,
        xlora_model_id=None,
        chat_template=None,
        tokenizer_json=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        class: Py<PyType>,
        model_id: String,
        no_kv_cache: bool,
        mut use_flash_attn: Option<bool>,
        repeat_last_n: usize,
        gqa: Option<usize>,
        order_file: Option<String>,
        xlora_model_id: Option<String>,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
    ) -> PyResult<Self> {
        use_flash_attn = use_flash_attn.map(|x| x & cfg!(feature = "flash-attn"));
        let kind = ModelKind::XLoraNormal;

        let loader = Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                model_id.into_py(py),
                kind.into_py(py),
                no_kv_cache.into_py(py),
                use_flash_attn.into_py(py),
                repeat_last_n.into_py(py),
                gqa.into_py(py),
                order_file.into_py(py),
                py.None(), // quantized_model_id
                py.None(), // quantized_filename
                xlora_model_id.into_py(py),
                chat_template.into_py(py),
                tokenizer_json.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            class.call1(py, args)
        })?;

        Ok(Self { inner: loader })
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
    #[pyo3(signature = (token_source = "cache", max_seqs = 16, truncate_sequence = false, logfile = None, revision = None, token_source_value = None, dtype = None))]
    #[allow(clippy::too_many_arguments)]
    fn load(
        &mut self,
        token_source: &str,
        max_seqs: usize,
        truncate_sequence: bool,
        logfile: Option<String>,
        revision: Option<String>,
        token_source_value: Option<String>,
        dtype: Option<DType>,
    ) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                token_source.into_py(py),
                max_seqs.into_py(py),
                truncate_sequence.into_py(py),
                logfile.into_py(py),
                revision.into_py(py),
                token_source_value.into_py(py),
                dtype.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            self.inner.call_method1(py, intern!(py, "load"), args)
        })
    }
}

#[pyclass]
pub struct QuantizedLoader {
    inner: Py<PyAny>,
}

#[pymethods]
impl QuantizedLoader {
    /// - `class`: Loader class.
    /// - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    /// - `is_gguf`: Loading gguf or ggml.
    /// - `no_kv_cache=False`: Disable kv cache.
    /// - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
    /// - `repeat_last_n=64`: Repeat last n context window.
    /// - `gqa=None`: GQA, irrelevant if non quantized model type.
    /// - `quantized_model_id=None`: Quantized model ID.
    /// - `quantized_filename=None`: Quantized filename (gguf/ggml),
    /// - `chat_template=None`: Chat template literal or file.
    /// - `tokenizer_json=None`: Tokenizer json file.
    #[new]
    #[pyo3(signature = (
        class,
        model_id,
        is_gguf,
        no_kv_cache=false,
        use_flash_attn=None,
        repeat_last_n=64,
        gqa=None,
        quantized_model_id=None,
        quantized_filename=None,
        chat_template=None,
        tokenizer_json=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        class: Py<PyType>,
        model_id: String,
        is_gguf: bool,
        no_kv_cache: bool,
        mut use_flash_attn: Option<bool>,
        repeat_last_n: usize,
        gqa: Option<usize>,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
    ) -> PyResult<Self> {
        use_flash_attn = use_flash_attn.map(|x| x & cfg!(feature = "flash-attn"));
        let kind = if is_gguf {
            ModelKind::QuantizedGGUF
        } else {
            ModelKind::QuantizedGGML
        };

        let loader = Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                model_id.into_py(py),
                kind.into_py(py),
                no_kv_cache.into_py(py),
                use_flash_attn.into_py(py),
                repeat_last_n.into_py(py),
                gqa.into_py(py),
                py.None(), // order_file
                quantized_model_id.into_py(py),
                quantized_filename.into_py(py),
                py.None(), // xlora_model_id
                chat_template.into_py(py),
                tokenizer_json.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            class.call1(py, args)
        })?;

        Ok(Self { inner: loader })
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
    #[pyo3(signature = (token_source = "cache", max_seqs = 16, truncate_sequence = false, logfile = None, revision = None, token_source_value = None, dtype = None))]
    #[allow(clippy::too_many_arguments)]
    fn load(
        &mut self,
        token_source: &str,
        max_seqs: usize,
        truncate_sequence: bool,
        logfile: Option<String>,
        revision: Option<String>,
        token_source_value: Option<String>,
        dtype: Option<DType>,
    ) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                token_source.into_py(py),
                max_seqs.into_py(py),
                truncate_sequence.into_py(py),
                logfile.into_py(py),
                revision.into_py(py),
                token_source_value.into_py(py),
                dtype.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            self.inner.call_method1(py, intern!(py, "load"), args)
        })
    }
}
#[pyclass]
pub struct XLoraQuantizedLoader {
    inner: Py<PyAny>,
}

#[pymethods]
impl XLoraQuantizedLoader {
    /// - `class`: Loader class.
    /// - `model_id`: Base model ID, or tokenizer ID if quantized model type.
    /// - `is_gguf`: Loading gguf or ggml.
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
    #[new]
    #[pyo3(signature = (
        class,
        model_id,
        is_gguf,
        no_kv_cache=false,
        use_flash_attn=None,
        repeat_last_n=64,
        gqa=None,
        order_file=None,
        quantized_model_id=None,
        quantized_filename=None,
        xlora_model_id=None,
        chat_template=None,
        tokenizer_json=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        class: Py<PyType>,
        model_id: String,
        is_gguf: bool,
        no_kv_cache: bool,
        mut use_flash_attn: Option<bool>,
        repeat_last_n: usize,
        gqa: Option<usize>,
        order_file: Option<String>,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        xlora_model_id: Option<String>,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
    ) -> PyResult<Self> {
        use_flash_attn = use_flash_attn.map(|x| x & cfg!(feature = "flash-attn"));
        let kind = if is_gguf {
            ModelKind::XLoraGGUF
        } else {
            ModelKind::XLoraGGML
        };

        let loader = Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                model_id.into_py(py),
                kind.into_py(py),
                no_kv_cache.into_py(py),
                use_flash_attn.into_py(py),
                repeat_last_n.into_py(py),
                gqa.into_py(py),
                order_file.into_py(py),
                quantized_model_id.into_py(py),
                quantized_filename.into_py(py),
                xlora_model_id.into_py(py),
                chat_template.into_py(py),
                tokenizer_json.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            class.call1(py, args)
        })?;

        Ok(Self { inner: loader })
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
    #[pyo3(signature = (token_source = "cache", max_seqs = 16, truncate_sequence = false, logfile = None, revision = None, token_source_value = None, dtype = None))]
    #[allow(clippy::too_many_arguments)]
    fn load(
        &mut self,
        token_source: &str,
        max_seqs: usize,
        truncate_sequence: bool,
        logfile: Option<String>,
        revision: Option<String>,
        token_source_value: Option<String>,
        dtype: Option<DType>,
    ) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let elems: &[Py<PyAny>] = &[
                token_source.into_py(py),
                max_seqs.into_py(py),
                truncate_sequence.into_py(py),
                logfile.into_py(py),
                revision.into_py(py),
                token_source_value.into_py(py),
                dtype.into_py(py),
            ];
            let args = PyTuple::new(py, elems);

            self.inner.call_method1(py, intern!(py, "load"), args)
        })
    }
}
