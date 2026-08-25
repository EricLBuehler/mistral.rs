use std::str::FromStr;

use anyhow::{ensure, Context, Result};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct HfConfigOverrides(Value);

impl<'de> Deserialize<'de> for HfConfigOverrides {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

impl HfConfigOverrides {
    pub fn new(value: Value) -> Result<Self> {
        ensure!(
            value.is_object(),
            "HF config overrides must be a JSON object"
        );
        Ok(Self(value))
    }

    pub fn apply(&self, config: &str) -> Result<String> {
        let mut config: Value =
            serde_json::from_str(config).context("failed to parse model config.json")?;
        ensure!(
            config.is_object(),
            "model config.json must be a JSON object"
        );
        merge_json(&mut config, &self.0);
        serde_json::to_string(&config).context("failed to serialize overridden model config")
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

impl FromStr for HfConfigOverrides {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let value = serde_json::from_str(value).context("invalid JSON for --hf-overrides")?;
        Self::new(value)
    }
}

fn merge_json(target: &mut Value, overrides: &Value) {
    match (target, overrides) {
        (Value::Object(target), Value::Object(overrides)) => {
            for (key, value) in overrides {
                match target.get_mut(key) {
                    Some(target) => merge_json(target, value),
                    None => {
                        target.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (target, value) => *target = value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursively_merges_nested_objects() -> Result<()> {
        let overrides: HfConfigOverrides = r#"{
            "text_config": {
                "max_position_embeddings": 131072,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 4.0
                }
            },
            "architectures": ["OverriddenArchitecture"]
        }"#
        .parse()?;
        let merged = overrides.apply(
            r#"{
                "text_config": {
                    "hidden_size": 4096,
                    "max_position_embeddings": 32768,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 1000000
                    }
                },
                "architectures": ["OriginalArchitecture"]
            }"#,
        )?;
        let merged: Value = serde_json::from_str(&merged)?;

        assert_eq!(merged["text_config"]["hidden_size"], 4096);
        assert_eq!(merged["text_config"]["max_position_embeddings"], 131072);
        assert_eq!(
            merged["text_config"]["rope_parameters"]["rope_type"],
            "yarn"
        );
        assert_eq!(
            merged["text_config"]["rope_parameters"]["rope_theta"],
            1000000
        );
        assert_eq!(merged["text_config"]["rope_parameters"]["factor"], 4.0);
        assert_eq!(merged["architectures"][0], "OverriddenArchitecture");
        Ok(())
    }

    #[test]
    fn rejects_non_object_overrides() {
        assert!(HfConfigOverrides::from_str("[]").is_err());
        assert!(HfConfigOverrides::from_str("null").is_err());
    }
}
