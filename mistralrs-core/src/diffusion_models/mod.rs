pub(crate) mod clip;
pub(crate) mod flux;
pub(crate) mod processor;
pub(crate) mod response;
pub(crate) mod t5;

macro_rules! generate_repr {
    ($t:ident) => {
        #[cfg(feature = "pyo3_macros")]
        #[pyo3::pymethods]
        impl $t {
            fn __repr__(&self) -> String {
                format!("{self:#?}")
            }
        }
    };
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
}

generate_repr!(DiffusionGenerationParams);

impl Default for DiffusionGenerationParams {
    /// Image dimensions will be 720x1280.
    fn default() -> Self {
        Self {
            height: 720,
            width: 1280,
        }
    }
}
