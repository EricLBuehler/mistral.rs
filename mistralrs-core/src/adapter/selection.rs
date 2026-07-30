use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{registry::AdapterLease, AdapterGenerationId, DynamicLoraRuntime, LoraAdapterError};

#[derive(Clone)]
/// Adapter selected for one request, by alias or immutable generation.
pub struct AdapterSelection(AdapterSelectionState);

#[derive(Clone, Debug)]
enum AdapterSelectionState {
    Alias(String),
    Generation(AdapterGenerationId),
    Pinned(AdapterLease),
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum AdapterSelectionWire {
    Alias(String),
    Generation { generation: AdapterGenerationId },
}

impl AdapterSelection {
    /// Selects the generation currently registered under an alias.
    pub fn alias(alias: impl Into<String>) -> Self {
        Self(AdapterSelectionState::Alias(
            alias.into().trim().to_string(),
        ))
    }

    /// Selects an exact resident adapter generation.
    pub fn generation(generation: AdapterGenerationId) -> Self {
        Self(AdapterSelectionState::Generation(generation))
    }

    /// Returns the exact generation after a selection has been resolved.
    pub fn resolved_generation(&self) -> Option<AdapterGenerationId> {
        match &self.0 {
            AdapterSelectionState::Alias(_) => None,
            AdapterSelectionState::Generation(generation) => Some(*generation),
            AdapterSelectionState::Pinned(lease) => Some(lease.generation()),
        }
    }

    pub(crate) fn is_pinned(&self) -> bool {
        matches!(self.0, AdapterSelectionState::Pinned(_))
    }

    pub(crate) fn pin(&mut self, runtime: &DynamicLoraRuntime) -> Result<(), LoraAdapterError> {
        let lease = match &self.0 {
            AdapterSelectionState::Alias(alias) => runtime.resolve_alias(alias)?,
            AdapterSelectionState::Generation(generation) => {
                runtime.resolve_generation(*generation)?
            }
            AdapterSelectionState::Pinned(_) => return Ok(()),
        };
        self.0 = AdapterSelectionState::Pinned(lease);
        Ok(())
    }

    pub(crate) fn lease(&self) -> Option<&AdapterLease> {
        match &self.0 {
            AdapterSelectionState::Pinned(lease) => Some(lease),
            AdapterSelectionState::Alias(_) | AdapterSelectionState::Generation(_) => None,
        }
    }
}

impl fmt::Debug for AdapterSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            AdapterSelectionState::Alias(alias) => f.debug_tuple("Alias").field(alias).finish(),
            AdapterSelectionState::Generation(generation) => {
                f.debug_tuple("Generation").field(generation).finish()
            }
            AdapterSelectionState::Pinned(lease) => {
                f.debug_tuple("Pinned").field(&lease.generation()).finish()
            }
        }
    }
}

impl From<String> for AdapterSelection {
    fn from(alias: String) -> Self {
        Self::alias(alias)
    }
}

impl From<&str> for AdapterSelection {
    fn from(alias: &str) -> Self {
        Self::alias(alias)
    }
}

impl Serialize for AdapterSelection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = match &self.0 {
            AdapterSelectionState::Alias(alias) => AdapterSelectionWire::Alias(alias.clone()),
            AdapterSelectionState::Generation(generation) => AdapterSelectionWire::Generation {
                generation: *generation,
            },
            AdapterSelectionState::Pinned(lease) => AdapterSelectionWire::Generation {
                generation: lease.generation(),
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AdapterSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match AdapterSelectionWire::deserialize(deserializer)? {
            AdapterSelectionWire::Alias(alias) => Self::alias(alias),
            AdapterSelectionWire::Generation { generation } => Self::generation(generation),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_state_is_unambiguous() {
        let alias = AdapterSelection::alias("production");
        let alias_json = serde_json::to_string(&alias).unwrap();
        assert_eq!(alias_json, r#""production""#);

        let generation = AdapterGenerationId::from_bytes([7; 32]);
        let selected = AdapterSelection::generation(generation);
        let generation_json = serde_json::to_string(&selected).unwrap();
        assert_eq!(
            generation_json,
            format!(r#"{{"generation":"{generation}"}}"#)
        );
        let decoded: AdapterSelection = serde_json::from_str(&generation_json).unwrap();
        assert_eq!(decoded.resolved_generation(), Some(generation));
    }
}
