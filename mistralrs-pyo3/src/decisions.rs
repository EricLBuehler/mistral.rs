use indexmap::IndexMap;
use mistralrs_core::{
    DecisionInstructions, DecisionMedia, DecisionQuestion as CoreQuestion,
    DecisionRequest as CoreRequest, DecisionResponse as CoreResponse,
};
use pyo3::{
    exceptions::{PyAttributeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde_json::{json, Value};

use crate::util::{parse_audio_url, parse_image_url, parse_video_url, PyApiResult};

fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let text: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&text).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn to_python(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("json")?
        .call_method1("loads", (value.to_string(),))?
        .unbind())
}

fn media_url(value: &Value) -> PyResult<(&str, String)> {
    let kind = value["type"]
        .as_str()
        .ok_or_else(|| PyValueError::new_err("Media requires a type"))?;
    let url = if kind == "input_audio" {
        let data = value[kind]["data"]
            .as_str()
            .ok_or_else(|| PyValueError::new_err("input_audio requires data"))?;
        let format = value[kind]["format"]
            .as_str()
            .ok_or_else(|| PyValueError::new_err("input_audio requires format"))?;
        mistralrs_core::input_audio_data_url(data, format)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    } else {
        value[kind]["url"]
            .as_str()
            .ok_or_else(|| PyValueError::new_err("Media requires a URL"))?
            .to_owned()
    };
    Ok((kind, url))
}

#[pyclass(frozen)]
#[derive(Clone, Debug)]
pub struct DecisionQuestion {
    question: CoreQuestion,
    spoken: Option<String>,
}

impl DecisionQuestion {
    fn parse(mut value: Value) -> PyResult<Self> {
        let spoken = if value["instructions"].is_object() {
            let (kind, url) = media_url(&value["instructions"])?;
            if !matches!(kind, "audio_url" | "input_audio") {
                return Err(PyValueError::new_err(
                    "Spoken instructions require audio_url or input_audio",
                ));
            }
            value["instructions"] = Value::String(String::new());
            Some(url)
        } else {
            None
        };
        let question: CoreQuestion = serde_json::from_value(value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if spoken.is_none() {
            question
                .validate()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        Ok(Self { question, spoken })
    }

    fn resolve(mut self) -> PyApiResult<CoreQuestion> {
        if let Some(source) = self.spoken {
            self.question
                .set_instructions(DecisionInstructions::Audio(parse_audio_url(&source)?));
        }
        Ok(self.question)
    }
}

#[pymethods]
impl DecisionQuestion {
    #[new]
    #[pyo3(signature = (kind, instructions, *, criteria=None, multiple=false, range=None, step=None, unit=None, max=None))]
    fn new(
        kind: String,
        instructions: &Bound<'_, PyAny>,
        criteria: Option<&Bound<'_, PyAny>>,
        multiple: bool,
        range: Option<[f64; 2]>,
        step: Option<f64>,
        unit: Option<String>,
        max: Option<usize>,
    ) -> PyResult<Self> {
        Self::parse(json!({
            "type": kind, "instructions": from_python(instructions)?,
            "criteria": criteria.map(from_python).transpose()?, "multiple": multiple,
            "range": range, "step": step, "unit": unit, "max": max,
        }))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut value = serde_json::to_value(&self.question)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if let Some(url) = &self.spoken {
            value["instructions"] = json!({"type": "audio_url", "audio_url": {"url": url}});
        }
        to_python(py, &value)
    }
}

#[pyclass(frozen)]
#[derive(Clone, Debug)]
pub struct DecisionRequest {
    state: Value,
    questions: IndexMap<String, DecisionQuestion>,
    images: Vec<String>,
    media: Vec<Value>,
    set_size: Option<usize>,
}

#[pymethods]
impl DecisionRequest {
    #[new]
    #[pyo3(signature = (state, questions, *, images=None, media=None, set_size=None))]
    fn new(
        state: &Bound<'_, PyAny>,
        questions: &Bound<'_, PyDict>,
        images: Option<Vec<String>>,
        media: Option<&Bound<'_, PyAny>>,
        set_size: Option<usize>,
    ) -> PyResult<Self> {
        let questions = questions
            .iter()
            .map(|(name, question)| {
                let value = match question.extract::<PyRef<'_, DecisionQuestion>>() {
                    Ok(question) => question.clone(),
                    Err(_) => DecisionQuestion::parse(from_python(&question)?)?,
                };
                Ok((name.extract::<String>()?, value))
            })
            .collect::<PyResult<IndexMap<_, _>>>()?;
        let media = media
            .map(from_python)
            .transpose()?
            .unwrap_or_else(|| json!([]));
        Ok(Self {
            state: from_python(state)?,
            questions,
            images: images.unwrap_or_default(),
            media: serde_json::from_value(media)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            set_size,
        })
    }
}

impl DecisionRequest {
    pub(crate) fn resolve(self) -> PyApiResult<CoreRequest> {
        let mut media = self
            .images
            .iter()
            .map(|url| parse_image_url(url).map(DecisionMedia::Image))
            .collect::<PyApiResult<Vec<_>>>()?;
        for source in self.media {
            let (kind, url) = media_url(&source).map_err(crate::util::PyApiErr)?;
            media.push(match kind {
                "image_url" => DecisionMedia::Image(parse_image_url(&url)?),
                "audio_url" | "input_audio" => DecisionMedia::Audio(parse_audio_url(&url)?),
                "video_url" => DecisionMedia::Video(parse_video_url(&url)?),
                _ => return Err("Unsupported decision media type".into()),
            });
        }
        let questions = self
            .questions
            .into_iter()
            .map(|(name, question)| question.resolve().map(|question| (name, question)))
            .collect::<PyApiResult<IndexMap<_, _>>>()?;
        let request = CoreRequest {
            state: self.state,
            questions,
            media,
            set_size: self.set_size,
        };
        request.validate()?;
        Ok(request)
    }
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct DecisionAnswer {
    value: Value,
}

#[pymethods]
impl DecisionAnswer {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(value) = self.value.get(name) {
            return to_python(py, value);
        }
        if matches!(
            (self.value["type"].as_str(), name),
            (Some("choice"), "choice" | "choices" | "confidence")
                | (Some("text" | "date"), "path")
                | (
                    Some("number"),
                    "interval" | "text" | "span" | "found" | "path"
                )
        ) {
            return Ok(py.None());
        }
        Err(PyAttributeError::new_err(format!(
            "This decision answer has no {name} field"
        )))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.value)
    }
}

#[pyclass(frozen, get_all)]
#[derive(Clone)]
pub struct DecisionUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[pyclass(frozen, get_all)]
#[derive(Clone)]
pub struct DecisionResponse {
    answers: IndexMap<String, DecisionAnswer>,
    usage: DecisionUsage,
}

impl TryFrom<CoreResponse> for DecisionResponse {
    type Error = serde_json::Error;

    fn try_from(result: CoreResponse) -> Result<Self, Self::Error> {
        Ok(Self {
            answers: result
                .answers
                .into_iter()
                .map(|(name, answer)| {
                    serde_json::to_value(answer).map(|value| (name, DecisionAnswer { value }))
                })
                .collect::<Result<_, _>>()?,
            usage: DecisionUsage {
                input_tokens: result.usage.input_tokens,
                output_tokens: result.usage.output_tokens,
            },
        })
    }
}

#[pymethods]
impl DecisionResponse {
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let answers = PyDict::new(py);
        for (name, answer) in &self.answers {
            answers.set_item(name, answer.to_dict(py)?)?;
        }
        let response = PyDict::new(py);
        response.set_item("answers", answers)?;
        response.set_item(
            "usage",
            to_python(
                py,
                &json!({
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                }),
            )?,
        )?;
        Ok(response.into_any().unbind())
    }
}
