pub(crate) mod clip;
pub(crate) mod flux;
pub(crate) mod processor;
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
    /// Number of denoising steps. If None, uses model default.
    #[serde(default)]
    pub num_steps: Option<usize>,
    /// Guidance scale (CFG). If None, uses model default.
    #[serde(default)]
    pub guidance_scale: Option<f64>,
    /// Negative prompt for CFG (Flux2 only). If None, uses empty string.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Emit preview images every N steps when streaming image generation.
    #[serde(default)]
    pub preview_interval: Option<usize>,
}

generate_repr!(DiffusionGenerationParams);

impl Default for DiffusionGenerationParams {
    /// Image dimensions will be 720x1280.
    fn default() -> Self {
        Self {
            height: 720,
            width: 1280,
            num_steps: None,
            guidance_scale: None,
            negative_prompt: None,
            preview_interval: None,
        }
    }
}
