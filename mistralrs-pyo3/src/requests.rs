use std::collections::HashMap;

use either::Either;
use mistralrs_core::WebSearchOptions;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods, PyList, PyString},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Debug, Clone)]
pub enum ToolChoice {
    NoTools,
    Auto,
}

#[pyclass]
#[derive(Debug)]
/// An OpenAI API compatible completion request.
pub struct CompletionRequest {
    pub(crate) _model: String,
    pub(crate) prompt: String,
    pub(crate) best_of: Option<usize>,
    pub(crate) echo_prompt: bool,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) logit_bias: Option<HashMap<u32, f32>>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) n_choices: usize,
    pub(crate) stop_seqs: Option<Vec<String>>,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) suffix: Option<String>,
    pub(crate) top_k: Option<usize>,
    pub(crate) grammar: Option<String>,
    pub(crate) grammar_type: Option<String>,
    pub(crate) min_p: Option<f64>,
    pub(crate) tool_schemas: Option<Vec<String>>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) dry_multiplier: Option<f32>,
    pub(crate) dry_base: Option<f32>,
    pub(crate) dry_allowed_length: Option<usize>,
    pub(crate) dry_sequence_breakers: Option<Vec<String>>,
    pub(crate) truncate_sequence: bool,
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
        repetition_penalty=None,
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
        min_p=None,
        tool_schemas=None,
        tool_choice=None,
        dry_multiplier=None,
        dry_base=None,
        dry_allowed_length=None,
        dry_sequence_breakers=None,
        truncate_sequence=false,
    ))]
    fn new(
        prompt: String,
        model: String,
        best_of: Option<usize>,
        echo_prompt: bool,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        repetition_penalty: Option<f32>,
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
        min_p: Option<f64>,
        tool_schemas: Option<Vec<String>>,
        tool_choice: Option<ToolChoice>,
        dry_multiplier: Option<f32>,
        dry_base: Option<f32>,
        dry_allowed_length: Option<usize>,
        dry_sequence_breakers: Option<Vec<String>>,
        truncate_sequence: Option<bool>,
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
            repetition_penalty,
            stop_seqs,
            temperature,
            top_p,
            top_k,
            grammar,
            grammar_type,
            min_p,
            tool_schemas,
            tool_choice,
            dry_multiplier,
            dry_allowed_length,
            dry_base,
            dry_sequence_breakers,
            truncate_sequence: truncate_sequence.unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone)]
pub enum PythonEmbeddingInputs {
    Prompts(Vec<String>),
    Tokens(Vec<Vec<u32>>),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub(crate) inputs: PythonEmbeddingInputs,
    pub(crate) truncate_sequence: bool,
}

#[pymethods]
impl EmbeddingRequest {
    #[new]
    #[pyo3(signature = (input, truncate_sequence=false))]
    fn new(input: Py<PyAny>, truncate_sequence: bool) -> PyResult<Self> {
        let inputs = Python::with_gil(|py| normalize_embedding_inputs(input.bind(py)))?;
        Ok(Self {
            inputs,
            truncate_sequence,
        })
    }
}

fn normalize_embedding_inputs(obj: &Bound<'_, PyAny>) -> PyResult<PythonEmbeddingInputs> {
    // Single string
    if let Ok(single) = obj.extract::<String>() {
        return Ok(PythonEmbeddingInputs::Prompts(vec![single]));
    }

    if let Ok(strings) = obj.extract::<Vec<String>>() {
        if strings.is_empty() {
            return Err(PyValueError::new_err(
                "embedding input sequence must not be empty",
            ));
        }
        return Ok(PythonEmbeddingInputs::Prompts(strings));
    }

    if let Ok(tokens) = obj.extract::<Vec<i64>>() {
        return Ok(PythonEmbeddingInputs::Tokens(vec![convert_tokens(tokens)?]));
    }

    if let Ok(batches) = obj.extract::<Vec<Vec<i64>>>() {
        if batches.is_empty() {
            return Err(PyValueError::new_err(
                "embedding token batches must not be empty",
            ));
        }
        let mut out = Vec::with_capacity(batches.len());
        for batch in batches {
            out.push(convert_tokens(batch)?);
        }
        return Ok(PythonEmbeddingInputs::Tokens(out));
    }

    Err(PyTypeError::new_err(
        "embedding input must be a string, list[str], list[int], or list[list[int]]",
    ))
}

fn convert_tokens(tokens: Vec<i64>) -> PyResult<Vec<u32>> {
    if tokens.is_empty() {
        return Err(PyValueError::new_err(
            "token list must contain at least one element",
        ));
    }
    let mut out = Vec::with_capacity(tokens.len());
    for token in tokens {
        out.push(convert_token(token)?);
    }
    Ok(out)
}

fn convert_token(value: i64) -> PyResult<u32> {
    if value < 0 || value > u32::MAX as i64 {
        Err(PyValueError::new_err(format!(
            "token value {value} is outside the allowed unsigned 32-bit range"
        )))
    } else {
        Ok(value as u32)
    }
}

#[pyclass]
#[derive(Debug)]
/// An OpenAI API compatible chat completion request.
pub struct ChatCompletionRequest {
    #[allow(clippy::type_complexity)]
    pub(crate) messages: Either<
        Vec<
            HashMap<
                String,
                Either<String, Vec<HashMap<String, Either<String, HashMap<String, String>>>>>,
            >,
        >,
        String,
    >,
    pub(crate) _model: String,
    pub(crate) logit_bias: Option<HashMap<u32, f32>>,
    pub(crate) logprobs: bool,
    pub(crate) top_logprobs: Option<usize>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) n_choices: usize,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) stop_seqs: Option<Vec<String>>,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) stream: bool,
    pub(crate) top_k: Option<usize>,
    pub(crate) grammar: Option<String>,
    pub(crate) grammar_type: Option<String>,
    pub(crate) min_p: Option<f64>,
    pub(crate) tool_schemas: Option<Vec<String>>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) dry_multiplier: Option<f32>,
    pub(crate) dry_base: Option<f32>,
    pub(crate) dry_allowed_length: Option<usize>,
    pub(crate) dry_sequence_breakers: Option<Vec<String>>,
    pub(crate) web_search_options: Option<WebSearchOptions>,
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) truncate_sequence: bool,
    /// Reasoning effort level for models that support extended thinking.
    /// Valid values: "low", "medium", "high"
    pub(crate) reasoning_effort: Option<String>,
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
        repetition_penalty = None,
        stop_seqs = None,
        temperature = None,
        top_p = None,
        top_k = None,
        stream=false,
        grammar = None,
        grammar_type = None,
        min_p=None,
        tool_schemas=None,
        tool_choice=None,
        dry_multiplier=None,
        dry_base=None,
        dry_allowed_length=None,
        dry_sequence_breakers=None,
        web_search_options=None,
        enable_thinking=false,
        truncate_sequence=false,
        reasoning_effort=None,
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
        repetition_penalty: Option<f32>,
        stop_seqs: Option<Vec<String>>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        stream: Option<bool>,
        grammar: Option<String>,
        grammar_type: Option<String>,
        min_p: Option<f64>,
        tool_schemas: Option<Vec<String>>,
        tool_choice: Option<ToolChoice>,
        dry_multiplier: Option<f32>,
        dry_base: Option<f32>,
        dry_allowed_length: Option<usize>,
        dry_sequence_breakers: Option<Vec<String>>,
        web_search_options: Option<WebSearchOptions>,
        enable_thinking: Option<bool>,
        truncate_sequence: Option<bool>,
        reasoning_effort: Option<String>,
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
                Err(PyTypeError::new_err("Expected a string or list of dicts."))
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
            repetition_penalty,
            stop_seqs,
            temperature,
            top_p,
            top_k,
            stream: stream.unwrap_or(false),
            grammar,
            grammar_type,
            min_p,
            tool_choice,
            tool_schemas,
            dry_multiplier,
            dry_allowed_length,
            dry_base,
            dry_sequence_breakers,
            web_search_options,
            enable_thinking,
            truncate_sequence: truncate_sequence.unwrap_or(false),
            reasoning_effort,
        })
    }
}
