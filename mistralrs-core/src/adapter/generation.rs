use std::{fmt, str::FromStr};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;

const ADAPTER_GENERATION_BYTES: usize = 32;
const ADAPTER_GENERATION_HEX_LEN: usize = ADAPTER_GENERATION_BYTES * 2;
const ADAPTER_GENERATION_DOMAIN: &[u8] = b"mistral.rs dynamic LoRA generation v2\0";
const DEFAULT_ADAPTER_REVISION: &str = "main";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
/// Content hash identifying one immutable adapter generation.
pub struct AdapterGenerationId([u8; ADAPTER_GENERATION_BYTES]);

impl AdapterGenerationId {
    /// Returns the raw SHA-256 generation bytes.
    pub fn as_bytes(&self) -> &[u8; ADAPTER_GENERATION_BYTES] {
        &self.0
    }

    /// Creates a generation ID from raw SHA-256 bytes.
    pub const fn from_bytes(bytes: [u8; ADAPTER_GENERATION_BYTES]) -> Self {
        Self(bytes)
    }

    pub(crate) fn from_adapter_digests(
        config_digest: [u8; ADAPTER_GENERATION_BYTES],
        weights_digest: [u8; ADAPTER_GENERATION_BYTES],
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(ADAPTER_GENERATION_DOMAIN);
        hasher.update(b"config\0");
        hasher.update(config_digest);
        hasher.update(b"\0weights\0");
        hasher.update(weights_digest);
        Self(hasher.finalize().into())
    }
}

impl fmt::Display for AdapterGenerationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("adapter generation must be exactly 64 hexadecimal characters")]
/// Error returned when a generation ID is not a 64-character hexadecimal string.
pub struct AdapterGenerationParseError;

impl FromStr for AdapterGenerationId {
    type Err = AdapterGenerationParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != ADAPTER_GENERATION_HEX_LEN || !value.is_ascii() {
            return Err(AdapterGenerationParseError);
        }
        let mut bytes = [0u8; ADAPTER_GENERATION_BYTES];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex(pair[0]).ok_or(AdapterGenerationParseError)?;
            let low = decode_hex(pair[1]).ok_or(AdapterGenerationParseError)?;
            bytes[index] = high << 4 | low;
        }
        Ok(Self(bytes))
    }
}

fn decode_hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

impl Serialize for AdapterGenerationId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AdapterGenerationId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
/// Alias and source used to preload a LoRA adapter.
pub struct LoraAdapterSpec {
    /// Stable request-facing name for the adapter.
    pub alias: String,
    /// Hugging Face repository or local adapter directory.
    pub source: String,
    /// Hugging Face revision for a remote adapter repository.
    #[serde(default)]
    pub revision: Option<String>,
    /// Parent-model lineage metadata accepted by vLLM-compatible configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_name: Option<String>,
}

impl LoraAdapterSpec {
    /// Creates an adapter preload specification.
    pub fn new(alias: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            alias: alias.into().trim().to_string(),
            source: source.into().trim().to_string(),
            revision: None,
            base_model_name: None,
        }
    }

    /// Selects a Hugging Face revision for this adapter repository.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into().trim().to_string());
        self
    }

    /// Records parent-model lineage metadata for the adapter.
    pub fn with_base_model_name(mut self, base_model_name: impl Into<String>) -> Self {
        self.base_model_name = Some(base_model_name.into().trim().to_string());
        self
    }

    /// Returns the adapter revision, defaulting independently to `main`.
    pub fn revision(&self) -> &str {
        self.revision
            .as_deref()
            .map(str::trim)
            .unwrap_or(DEFAULT_ADAPTER_REVISION)
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("LoRA adapter must use `alias=source` with nonempty alias and source")]
/// Error returned when an adapter specification does not use `alias=source`.
pub struct LoraAdapterSpecParseError;

impl FromStr for LoraAdapterSpec {
    type Err = LoraAdapterSpecParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (alias, source) = value.split_once('=').ok_or(LoraAdapterSpecParseError)?;
        let alias = alias.trim();
        let source = source.trim();
        if alias.is_empty() || source.is_empty() {
            return Err(LoraAdapterSpecParseError);
        }
        Ok(Self::new(alias, source))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
/// Metadata for a loaded adapter alias and its current immutable generation.
pub struct LoraAdapterInfo {
    /// Stable request-facing name for the adapter.
    pub alias: String,
    /// Repository or directory from which the adapter was loaded.
    pub source: String,
    /// Requested Hugging Face revision for a remote preload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Content-derived generation pinned by admitted requests.
    pub generation: AdapterGenerationId,
    /// Maximum rank used by the adapter configuration.
    pub rank: usize,
    /// Resident bytes occupied by this generation's adapter tensors.
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
/// A loaded adapter together with the base model that owns its runtime.
pub struct LoraAdapterRoute {
    /// Base model ID used to dispatch requests for this adapter.
    pub model_id: String,
    /// Loaded adapter metadata.
    pub adapter: LoraAdapterInfo,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
/// Residency and lease state for one immutable adapter generation.
pub struct LoraResidentGenerationInfo {
    pub generation: AdapterGenerationId,
    pub aliases: Vec<String>,
    pub rank: usize,
    pub bytes: u64,
    pub retired: bool,
    pub active_leases: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_id_is_content_addressed_and_serializes_as_hex() {
        let generation = |config: &[u8], weights: &[u8]| {
            AdapterGenerationId::from_adapter_digests(
                Sha256::digest(config).into(),
                Sha256::digest(weights).into(),
            )
        };
        let first = generation(b"config", b"weights");
        let second = generation(b"config", b"weights");
        assert_eq!(first, second);
        assert_eq!(first.to_string().parse(), Ok(first));
        assert_eq!(
            serde_json::from_str::<AdapterGenerationId>(&serde_json::to_string(&first).unwrap())
                .unwrap(),
            first
        );

        let changed = generation(b"config", b"weights changed");
        assert_ne!(first, changed);
    }

    #[test]
    fn adapter_revision_defaults_to_main_and_can_be_overridden() {
        let default = LoraAdapterSpec::new("code", "org/code-lora");
        assert_eq!(default.revision(), "main");
        assert_eq!(default.revision, None);

        let deserialized: LoraAdapterSpec =
            serde_json::from_str(r#"{"alias":"code","source":"org/code-lora"}"#).unwrap();
        assert_eq!(deserialized.revision(), "main");

        let pinned = default.with_revision(" refs/pr/7 ");
        assert_eq!(pinned.revision(), "refs/pr/7");
        assert_eq!(pinned.revision.as_deref(), Some("refs/pr/7"));
    }
}
