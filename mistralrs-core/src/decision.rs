mod cache;
mod clm;
mod inference;
#[cfg(test)]
mod tests;

pub(crate) use clm::{ClmConfig, ClmHeads};
pub(crate) use inference::ClmInference;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_DECISION_CHOICES: usize = 255;
pub const MAX_DECISION_SCORE_LEVELS: usize = 10;
pub const CLM_MAX_TOKENS: usize = 2048;
const MAX_TEMPERATURE: f64 = 100.0;
pub const CLM_MODEL_ID: &str = "Contrastive-LM/CLM-v0.1-8B";

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DecisionValidationError(pub String);

fn invalid(message: impl Into<String>) -> anyhow::Error {
    DecisionValidationError(message.into()).into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DecisionRequest {
    pub model: String,
    pub state: Value,
    pub questions: IndexMap<String, DecisionQuestion>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

fn default_temperature() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum DecisionQuestion {
    Noul {
        #[serde(default)]
        instructions: Value,
        #[serde(default)]
        criteria: Option<IndexMap<String, Value>>,
    },
    Choice {
        #[serde(default)]
        instructions: Value,
        criteria: IndexMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum DecisionAnswer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: IndexMap<String, f64>,
    },
    Score {
        score: f64,
        confidence: f64,
        probabilities: IndexMap<String, f64>,
        legend: IndexMap<String, String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub billing_units: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DecisionResponse {
    pub model: String,
    pub answers: IndexMap<String, DecisionAnswer>,
    pub usage: DecisionUsage,
}

pub(crate) struct DecisionPair {
    pub state: String,
    pub keys: Vec<String>,
    pub candidates: Vec<String>,
}

fn is_text(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
}

impl DecisionRequest {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !is_text(&self.state) {
            return Err(invalid("state must be a string, object, or array"));
        }
        if self.questions.is_empty() {
            return Err(invalid("questions must not be empty"));
        }
        if !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.temperature > MAX_TEMPERATURE
        {
            return Err(invalid("temperature must be in (0, 100]"));
        }
        for (id, question) in &self.questions {
            question
                .validate()
                .map_err(|e| invalid(format!("question {id:?}: {e}")))?;
        }
        Ok(())
    }

    pub(crate) fn pairs(&self) -> anyhow::Result<Vec<DecisionPair>> {
        self.validate()?;
        Ok(self
            .questions
            .values()
            .map(|q| q.pair(&self.state))
            .collect())
    }
}

impl DecisionQuestion {
    fn instructions(&self) -> &Value {
        match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        if !self.instructions().is_null() && !is_text(self.instructions()) {
            return Err(invalid("instructions must be a string, object, or array"));
        }
        match self {
            Self::Noul {
                criteria: Some(criteria),
                ..
            } => {
                if criteria.iter().any(|(key, value)| {
                    !matches!(key.as_str(), "true" | "false") || !is_text(value)
                }) {
                    return Err(invalid("noul criteria must describe true and/or false"));
                }
            }
            Self::Choice { criteria, .. } => {
                if criteria.is_empty() || criteria.len() > MAX_DECISION_CHOICES {
                    return Err(invalid(format!(
                        "choice requires 1..={MAX_DECISION_CHOICES} options"
                    )));
                }
                if criteria.values().any(|v| !v.is_null() && !is_text(v)) {
                    return Err(invalid(
                        "choice descriptions must be strings, objects, arrays, or null",
                    ));
                }
            }
            Self::Score { criteria, .. }
                if !(2..=MAX_DECISION_SCORE_LEVELS).contains(&criteria.len())
                    || criteria.iter().any(|v| !is_text(v)) =>
            {
                return Err(invalid(format!(
                    "score requires 2..={MAX_DECISION_SCORE_LEVELS} text or structured levels"
                )));
            }
            _ => (),
        }
        Ok(())
    }

    fn pair(&self, state: &Value) -> DecisionPair {
        let state = to_text(state, 0);
        let instructions = to_text(self.instructions(), 0);
        let state = state.trim();
        let instructions = instructions.trim();
        let state = match (state.is_empty(), instructions.is_empty()) {
            (false, false) => format!("{state}\n\n{instructions}"),
            _ => format!("{state}{instructions}"),
        };
        let (keys, candidates) = match self {
            Self::Choice { criteria, .. } => criteria
                .iter()
                .map(|(key, value)| {
                    let text = if value.is_null() || value.as_str() == Some("") {
                        key.clone()
                    } else {
                        to_text(value, 0)
                    };
                    (key.clone(), text)
                })
                .unzip(),
            Self::Score { criteria, .. } => criteria
                .iter()
                .enumerate()
                .map(|(i, value)| (i.to_string(), to_text(value, 0)))
                .unzip(),
            Self::Noul { criteria, .. } => ["false", "true"]
                .into_iter()
                .map(|key| {
                    let description = criteria
                        .as_ref()
                        .and_then(|c| c.get(key))
                        .filter(|v| !v.is_null() && v.as_str() != Some(""))
                        .map(|v| to_text(v, 0))
                        .unwrap_or_else(|| {
                            if instructions.is_empty() {
                                key.to_string()
                            } else if key == "true" {
                                format!("Yes. This is true: {instructions}")
                            } else {
                                format!("No. This is false: {instructions}")
                            }
                        });
                    (key.to_string(), format!("{key}: {description}"))
                })
                .unzip(),
        };
        DecisionPair {
            state,
            keys,
            candidates,
        }
    }

    pub(crate) fn answer(&self, keys: &[String], logits: &[f32]) -> anyhow::Result<DecisionAnswer> {
        anyhow::ensure!(
            keys.len() == logits.len() && !logits.is_empty(),
            "invalid CLM score shape"
        );
        anyhow::ensure!(
            logits.iter().all(|x| x.is_finite()),
            "CLM returned non-finite scores"
        );
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let weights: Vec<_> = logits.iter().map(|x| (*x as f64 - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let probs: Vec<_> = weights.iter().map(|x| x / total).collect();
        let mut best = 0;
        for i in 1..probs.len() {
            if probs[i] > probs[best] {
                best = i;
            }
        }
        let confidence = if probs.len() == 1 {
            1.0
        } else {
            let rest_count =
                f64::from(u32::try_from(probs.len() - 1).expect("validated option count"));
            (probs[best] - (1.0 - probs[best]) / rest_count).clamp(0.0, 1.0)
        };
        let probabilities = keys.iter().cloned().zip(probs.iter().copied()).collect();
        Ok(match self {
            Self::Noul { .. } => DecisionAnswer::Noul { noul: probs[1] },
            Self::Choice { .. } => DecisionAnswer::Choice {
                choice: keys[best].clone(),
                confidence,
                probabilities,
            },
            Self::Score { criteria, .. } => DecisionAnswer::Score {
                score: probs
                    .iter()
                    .zip(0_u32..)
                    .map(|(p, i)| f64::from(i) * p)
                    .sum(),
                confidence,
                probabilities,
                legend: keys
                    .iter()
                    .cloned()
                    .zip(criteria.iter().map(|v| to_text(v, 0)))
                    .collect(),
            },
        })
    }
}

fn to_text(value: &Value, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let nested = |value: &Value| match value {
        Value::Object(x) => !x.is_empty(),
        Value::Array(x) => !x.is_empty(),
        _ => false,
    };
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| {
                if nested(value) {
                    format!("{pad}{key}:\n{}", to_text(value, indent + 2))
                } else {
                    format!("{pad}{key}: {}", to_text(value, 0))
                }
            })
            .collect::<Vec<_>>()
            .join(if indent == 0 { "\n\n" } else { "\n" }),
        Value::Array(values) => values
            .iter()
            .map(|value| {
                if nested(value) {
                    format!("{pad}-\n{}", to_text(value, indent + 2))
                } else {
                    format!("{pad}- {}", to_text(value, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => value.to_string(),
    }
}
