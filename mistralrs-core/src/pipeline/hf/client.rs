use std::path::PathBuf;

use hf_hub::{
    progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler},
    HFClient, HFClientBuilder, HFClientSync, HFError, HFRepository, HFRepositorySync,
    RepoTypeModel,
};
use indicatif::{ProgressBar, ProgressStyle};

use super::{is_hf_hub_offline, resolve_hf_cache};
use crate::{pipeline::TokenSource, utils::tokens::get_token};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const DOWNLOAD_PROGRESS_TEMPLATE: &str =
    "{msg} {wide_bar} {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}";

#[derive(Clone)]
pub struct HubClient {
    inner: HFClientSync,
    progress: bool,
    cache: PathBuf,
}

#[derive(Clone)]
pub struct HubRepo {
    inner: HFRepositorySync<RepoTypeModel>,
    revision: String,
    progress: bool,
    cache: PathBuf,
}

pub(crate) struct AsyncHubRepo {
    inner: HFRepository<RepoTypeModel>,
    revision: String,
}

pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn client_builder(token_source: &TokenSource, cache: PathBuf) -> anyhow::Result<HFClientBuilder> {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
    let mut builder = HFClient::builder().endpoint(endpoint).cache_dir(cache);
    if let Some(token) = get_token(token_source)? {
        builder = builder.token(token);
    }
    Ok(builder)
}

impl HubClient {
    pub(super) fn new(
        token_source: &TokenSource,
        progress: bool,
        cache: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let cache = resolve_hf_cache(cache);
        Ok(Self {
            inner: client_builder(token_source, cache.clone())?.build_sync()?,
            progress,
            cache,
        })
    }

    pub fn repo(&self, model_id: impl Into<String>, revision: impl Into<String>) -> HubRepo {
        let model_id = model_id.into();
        let (owner, name) = hf_hub::split_id(&model_id);
        HubRepo {
            inner: self.inner.model(owner, name),
            revision: revision.into(),
            progress: self.progress,
            cache: self.cache.clone(),
        }
    }
}

impl HubRepo {
    pub fn get(&self, file: &str) -> Result<PathBuf, HFError> {
        let cached = self
            .inner
            .download_file()
            .filename(file)
            .revision(&self.revision)
            .local_files_only(true)
            .send();
        match cached {
            Err(HFError::LocalEntryNotFound { .. } | HFError::EntryNotFound { .. })
                if !is_hf_hub_offline() => {}
            cached => return cached,
        }
        self.inner
            .download_file()
            .filename(file)
            .revision(&self.revision)
            .maybe_progress(
                self.progress
                    .then(|| Progress::from(DownloadProgress::new(file))),
            )
            .send()
    }

    fn cached_files(&self) -> Result<Vec<String>, HFError> {
        Ok(super::offline_snapshot_files_at(
            std::path::Path::new(&self.inner.repo_path()),
            &self.revision,
            &self.cache,
        )?)
    }

    pub fn files(&self) -> Result<Vec<String>, HFError> {
        if is_hf_hub_offline() {
            return self.cached_files();
        }
        let info = self.inner.info().revision(&self.revision).send()?;
        Ok(info
            .siblings
            .unwrap_or_default()
            .into_iter()
            .map(|file| file.rfilename)
            .collect())
    }
}

impl AsyncHubRepo {
    pub(super) fn new(
        token_source: &TokenSource,
        cache: Option<PathBuf>,
        model_id: &str,
        revision: &str,
    ) -> anyhow::Result<Self> {
        let (owner, name) = hf_hub::split_id(model_id);
        Ok(Self {
            inner: client_builder(token_source, resolve_hf_cache(cache))?
                .build()?
                .model(owner, name),
            revision: revision.to_owned(),
        })
    }

    pub(super) async fn get(&self, file: &str) -> Result<PathBuf, HFError> {
        let cached = self
            .inner
            .download_file()
            .filename(file)
            .revision(&self.revision)
            .local_files_only(true)
            .send()
            .await;
        match cached {
            Err(HFError::LocalEntryNotFound { .. } | HFError::EntryNotFound { .. })
                if !is_hf_hub_offline() => {}
            cached => return cached,
        }
        self.inner
            .download_file()
            .filename(file)
            .revision(&self.revision)
            .send()
            .await
    }

    fn cached_files(&self) -> Result<Vec<String>, HFError> {
        Ok(super::offline_snapshot_files_at(
            std::path::Path::new(&self.inner.repo_path()),
            &self.revision,
            self.inner.client().cache_dir(),
        )?)
    }

    pub(super) async fn files(&self) -> Result<Vec<String>, HFError> {
        if is_hf_hub_offline() {
            return self.cached_files();
        }
        let info = self.inner.info().revision(&self.revision).send().await?;
        Ok(info
            .siblings
            .unwrap_or_default()
            .into_iter()
            .map(|file| file.rfilename)
            .collect())
    }
}

struct DownloadProgress {
    bar: ProgressBar,
}

impl DownloadProgress {
    fn new(file: &str) -> Self {
        let bar = ProgressBar::new(0);
        bar.set_style(ProgressStyle::with_template(DOWNLOAD_PROGRESS_TEMPLATE).unwrap());
        bar.set_message(file.to_owned());
        Self { bar }
    }
}

impl ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => {
                self.bar.set_length(*total_bytes)
            }
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                if let Some(file) = files.first() {
                    self.bar.set_length(file.total_bytes);
                    self.bar.set_position(file.bytes_completed);
                }
            }
            ProgressEvent::Download(DownloadEvent::Complete) => self.bar.finish_and_clear(),
            _ => {}
        }
    }
}

impl Drop for DownloadProgress {
    fn drop(&mut self) {
        self.bar.finish_and_clear();
    }
}

#[cfg(test)]
mod tests;
