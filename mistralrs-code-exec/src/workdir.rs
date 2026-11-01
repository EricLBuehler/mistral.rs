use std::path::{Path, PathBuf};

pub(crate) enum SessionDir {
    Configured(PathBuf),
    Temp(tempfile::TempDir),
}

impl SessionDir {
    // configured dirs persist by design; temp dirs are removed when the owning session drops
    pub(crate) fn create(configured: Option<PathBuf>, temp_prefix: &str) -> anyhow::Result<Self> {
        match configured {
            Some(path) => {
                std::fs::create_dir_all(&path)?;
                Ok(Self::Configured(path))
            }
            None => Ok(Self::Temp(
                tempfile::Builder::new().prefix(temp_prefix).tempdir()?,
            )),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        match self {
            Self::Configured(path) => path,
            Self::Temp(dir) => dir.path(),
        }
    }
}
