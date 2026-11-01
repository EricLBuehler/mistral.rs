use anyhow::{ensure, Result};
use image::DynamicImage;
use indexmap::IndexMap;
use mistralrs_audio::AudioInput;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::VideoInput;

const MAX_BOX_COUNT: usize = 64;
const GRID_STEP_RELATIVE_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Debug)]
pub enum DecisionMedia {
    Image(DynamicImage),
    Audio(AudioInput),
    Video(VideoInput),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum DecisionInstructions {
    Text(String),
    #[serde(skip)]
    Audio(AudioInput),
}

impl From<String> for DecisionInstructions {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for DecisionInstructions {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum DecisionQuestion {
    Noul {
        instructions: DecisionInstructions,
        #[serde(default)]
        criteria: Option<IndexMap<String, Value>>,
    },
    Choice {
        instructions: DecisionInstructions,
        criteria: IndexMap<String, Option<String>>,
        #[serde(default)]
        multiple: bool,
    },
    Score {
        instructions: DecisionInstructions,
        criteria: Vec<String>,
    },
    Text {
        instructions: DecisionInstructions,
    },
    Number {
        instructions: DecisionInstructions,
        #[serde(default)]
        range: Option<[f64; 2]>,
        #[serde(default)]
        step: Option<f64>,
        #[serde(default)]
        unit: Option<String>,
    },
    Date {
        instructions: DecisionInstructions,
    },
    Box {
        instructions: DecisionInstructions,
    },
    Boxes {
        instructions: DecisionInstructions,
        #[serde(default)]
        max: Option<usize>,
    },
}

impl DecisionQuestion {
    pub fn instructions(&self) -> &DecisionInstructions {
        match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Text { instructions }
            | Self::Number { instructions, .. }
            | Self::Date { instructions }
            | Self::Box { instructions }
            | Self::Boxes { instructions, .. } => instructions,
        }
    }

    pub fn set_instructions(&mut self, value: DecisionInstructions) {
        match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Text { instructions }
            | Self::Number { instructions, .. }
            | Self::Date { instructions }
            | Self::Box { instructions }
            | Self::Boxes { instructions, .. } => *instructions = value,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self.instructions() {
            DecisionInstructions::Text(text) => ensure!(
                !text.trim().is_empty(),
                "Decision instructions must not be empty"
            ),
            DecisionInstructions::Audio(audio) => ensure!(
                !audio.samples.is_empty(),
                "Spoken decision instructions must not be empty"
            ),
        }
        match self {
            Self::Choice { criteria, .. } => {
                ensure!(!criteria.is_empty(), "Decision choice requires options")
            }
            Self::Score { criteria, .. } => ensure!(
                (2..=10).contains(&criteria.len()),
                "Decision score requires 2 to 10 ordered levels"
            ),
            Self::Number { range, step, .. } => {
                if let Some([low, high]) = range {
                    ensure!(
                        low.is_finite() && high.is_finite() && low < high,
                        "Decision number range must be finite and increasing"
                    );
                }
                if let Some(step) = step {
                    ensure!(
                        range.is_some() && step.is_finite() && *step > 0.0,
                        "Decision number step requires a range and must be positive"
                    );
                    let [low, high] = range.unwrap();
                    let count = (high - low) / step;
                    ensure!(
                        count.is_finite()
                            && (count - count.round_ties_even()).abs()
                                <= GRID_STEP_RELATIVE_TOLERANCE * count.abs(),
                        "Decision number step must divide its range"
                    );
                }
            }
            Self::Boxes { max: Some(max), .. } => ensure!(
                (1..=MAX_BOX_COUNT).contains(max),
                "Decision boxes max must be between 1 and {MAX_BOX_COUNT}"
            ),
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub state: Value,
    pub questions: IndexMap<String, DecisionQuestion>,
    #[serde(skip)]
    pub media: Vec<DecisionMedia>,
    #[serde(default)]
    pub set_size: Option<usize>,
}

impl DecisionRequest {
    pub fn new(state: impl Into<Value>) -> Self {
        Self {
            state: state.into(),
            questions: IndexMap::new(),
            media: Vec::new(),
            set_size: None,
        }
    }

    pub fn with_question(mut self, name: impl Into<String>, question: DecisionQuestion) -> Self {
        self.questions.insert(name.into(), question);
        self
    }

    pub fn with_media(mut self, media: DecisionMedia) -> Self {
        self.media.push(media);
        self
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.questions.is_empty(),
            "Decision request requires questions"
        );
        ensure!(
            self.set_size != Some(0),
            "Decision set_size must be positive"
        );
        let pictures = self
            .media
            .iter()
            .filter(|media| matches!(media, DecisionMedia::Image(_)))
            .count();
        for (name, question) in &self.questions {
            ensure!(
                !name.is_empty(),
                "Decision question names must not be empty"
            );
            question.validate()?;
            if matches!(
                question,
                DecisionQuestion::Box { .. } | DecisionQuestion::Boxes { .. }
            ) {
                ensure!(
                    pictures == 1,
                    "Decision box questions require exactly one picture"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DecisionBox {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DecisionFoundBox {
    #[serde(rename = "box")]
    pub bounds: DecisionBox,
    pub found: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum DecisionAnswer {
    Noul {
        noul: f64,
    },
    Choice {
        #[serde(skip_serializing_if = "Option::is_none")]
        choice: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        choices: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
        probabilities: IndexMap<String, f64>,
    },
    Score {
        score: f64,
        confidence: f64,
        probabilities: IndexMap<String, f64>,
        legend: IndexMap<String, String>,
    },
    Text {
        text: Option<String>,
        span: Option<[usize; 2]>,
        found: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    Number {
        number: Option<f64>,
        unit: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        interval: Option<[f64; 2]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        span: Option<[usize; 2]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        found: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    Date {
        date: Option<String>,
        text: Option<String>,
        span: Option<[usize; 2]>,
        found: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    Box {
        #[serde(rename = "box")]
        bounds: Option<DecisionBox>,
        found: f64,
        confidence: f64,
    },
    Boxes {
        boxes: Vec<DecisionFoundBox>,
        count: usize,
        at_least: bool,
        presence: Vec<f64>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DecisionResponse {
    pub answers: IndexMap<String, DecisionAnswer>,
    pub usage: DecisionUsage,
}
