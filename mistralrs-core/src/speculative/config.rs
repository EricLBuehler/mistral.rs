use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub enum SpeculativeConfig {
    Off,
    Mtp(MtpConfig),
}

#[derive(Clone, Debug)]
pub struct MtpConfig {
    pub model: ModelSource,
    pub n_predict: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum ModelSource {
    Hf {
        id: String,
        revision: Option<String>,
    },
    Path {
        path: PathBuf,
    },
}

impl ModelSource {
    pub fn from_cli(value: impl Into<String>) -> Self {
        let value = value.into();
        let path = PathBuf::from(&value);
        if path.exists() || value.starts_with('.') || value.starts_with('/') {
            Self::Path { path }
        } else {
            Self::Hf {
                id: value,
                revision: None,
            }
        }
    }

    pub fn as_path(&self) -> candle_core::Result<&Path> {
        match self {
            Self::Path { path } => Ok(path.as_path()),
            Self::Hf { id, .. } => candle_core::bail!(
                "Gemma4 MTP HF model sources are not wired yet; pass a local --mtp-model path for `{id}`"
            ),
        }
    }
}
