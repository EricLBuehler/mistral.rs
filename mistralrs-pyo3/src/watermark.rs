use candle_core::{Device, Tensor};
use mistralrs_core::{Watermark, WatermarkEvidence};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAnyMethods, PyDict},
};

#[pyclass]
#[derive(Clone, Debug)]
pub struct WatermarkConfig {
    inner: mistralrs_core::WatermarkConfig,
}

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn evidence(py: Python<'_>, result: WatermarkEvidence) -> PyResult<Py<PyAny>> {
    let json = serde_json::to_string(&result).map_err(invalid)?;
    Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
}

fn device(name: &str) -> PyResult<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Device::new_cuda(0).map_err(invalid),
        "metal" => Device::new_metal(0).map_err(invalid),
        _ => Err(invalid("device must be cpu, cuda, or metal")),
    }
}

#[pymethods]
impl WatermarkConfig {
    #[new]
    #[pyo3(signature = (key, *, scheme="synthid", **parameters))]
    fn new(
        py: Python<'_>,
        key: String,
        scheme: &str,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut value = if let Some(parameters) = parameters {
            let json: String = py
                .import("json")?
                .call_method1("dumps", (parameters,))?
                .extract()?;
            serde_json::from_str::<serde_json::Value>(&json).map_err(invalid)?
        } else {
            serde_json::json!({})
        };
        value["key"] = key.into();
        value["scheme"] = scheme.into();
        let inner: mistralrs_core::WatermarkConfig =
            serde_json::from_value(value).map_err(invalid)?;
        inner.validate().map_err(invalid)?;
        Ok(Self { inner })
    }

    #[getter]
    fn scheme(&self) -> &str {
        self.inner.scheme()
    }

    #[pyo3(signature = (tokens, prompt_len=0, eos_token_ids=None))]
    fn detect(
        &self,
        py: Python<'_>,
        tokens: Vec<u32>,
        prompt_len: usize,
        eos_token_ids: Option<Vec<u32>>,
    ) -> PyResult<Py<PyAny>> {
        let watermark = Watermark::new(&self.inner).map_err(invalid)?;
        evidence(
            py,
            watermark
                .detect(&tokens, prompt_len, &eos_token_ids.unwrap_or_default())
                .map_err(invalid)?,
        )
    }

    #[pyo3(signature = (embeddings, prompt_len=0, *, device_name="cpu"))]
    fn detect_embeddings(
        &self,
        py: Python<'_>,
        embeddings: Vec<Vec<f32>>,
        prompt_len: usize,
        device_name: &str,
    ) -> PyResult<Py<PyAny>> {
        let watermark = Watermark::new(&self.inner).map_err(invalid)?;
        if device_name == "cpu" {
            return evidence(
                py,
                watermark
                    .detect_embeddings(&embeddings, prompt_len)
                    .map_err(invalid)?,
            );
        }
        let tensor = Tensor::new(embeddings, &device(device_name)?).map_err(invalid)?;
        evidence(
            py,
            watermark
                .detect_embeddings_tensor(&tensor, prompt_len)
                .map_err(invalid)?,
        )
    }

    #[pyo3(signature = (previous, candidate))]
    fn accepts_embedding(&self, previous: Vec<f32>, candidate: Vec<f32>) -> PyResult<bool> {
        let watermark = Watermark::new(&self.inner).map_err(invalid)?;
        watermark
            .semstamp()
            .map_err(invalid)?
            .accepts(&previous, &candidate)
            .map_err(invalid)
    }
}

#[derive(Clone, Debug, FromPyObject)]
pub(crate) enum WatermarkArg {
    Config(WatermarkConfig),
    Synthid(crate::requests::SynthIdTextWatermarkConfig),
}

impl WatermarkArg {
    pub(crate) fn to_core(&self) -> mistralrs_core::WatermarkConfig {
        match self {
            Self::Config(config) => config.inner.clone(),
            Self::Synthid(config) => config.inner.clone().into(),
        }
    }
}
