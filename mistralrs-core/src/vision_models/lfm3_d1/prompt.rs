use std::{collections::BTreeSet, ops::Range};

use anyhow::{bail, ensure, Context, Result};
use tokenizers::{Encoding, Tokenizer};

use crate::decision::{DecisionInstructions, DecisionMedia, DecisionQuestion, DecisionRequest};

use super::config::DecisionConfig;

const DEFAULT_BOX_COUNT: usize = 16;
const GRID_SIZE: usize = 128;
const IM_START: &str = "<|im_start|>";
const BOS: &str = "<|startoftext|>";
const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n<think></think>";
const EDGES: [&str; 4] = ["top", "bottom", "left", "right"];
const COORDINATE_SCALE: usize = 1000;
const NUMBER_TOLERANCE: f64 = 1e-6;
const SPECIAL_BREAK: &str = "\u{200b}";

#[derive(Clone, Debug)]
pub(super) struct Grid {
    pub values: Vec<f64>,
    pub numeric: bool,
    pub codes: Vec<String>,
    pub groups: Vec<Vec<u32>>,
}

impl Grid {
    #[allow(clippy::cast_precision_loss)]
    fn new(
        tokenizer: &Tokenizer,
        config: &DecisionConfig,
        low: f64,
        high: f64,
        size: usize,
        scale: usize,
    ) -> Result<Self> {
        ensure!(
            size >= 2,
            "Decision numeric grid requires at least two values"
        );
        let step = (high - low) / (size - 1) as f64;
        let values = (0..size)
            .map(|index| {
                if index + 1 == size {
                    high
                } else {
                    low + index as f64 * step
                }
            })
            .collect::<Vec<_>>();
        if config.d1_numbers {
            let mut groups = Vec::with_capacity(size);
            for value in &values {
                let scaled = value * scale as f64;
                let rounded = scaled.round_ties_even();
                let label = format!("{rounded:.0}");
                if (scaled - rounded).abs() > NUMBER_TOLERANCE.max(scaled.abs() * 1e-9)
                    || encode(tokenizer, &label)?.len() != 1
                {
                    break;
                }
                groups.push(forms(tokenizer, &[label.clone(), format!(" {label}")])?);
            }
            if groups.len() == size {
                return Ok(Self {
                    values,
                    numeric: true,
                    codes: Vec::new(),
                    groups,
                });
            }
        }
        if scale == COORDINATE_SCALE {
            return Self::new(tokenizer, config, 0.0, 1.0, GRID_SIZE, 1);
        }
        let labels = (0..size).map(|index| index.to_string()).collect::<Vec<_>>();
        let codes = if size <= 10 {
            labels
        } else {
            aliases(tokenizer, &labels, config.d1_codes)?
        };
        let groups = codes
            .iter()
            .map(|code| {
                if size <= 10 {
                    forms(tokenizer, std::slice::from_ref(code))
                } else {
                    forms(tokenizer, &[code.clone(), format!(" {code}")])
                }
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            values,
            numeric: false,
            codes,
            groups,
        })
    }

    pub(super) fn step(&self) -> f64 {
        self.values[1] - self.values[0]
    }
}

#[derive(Clone, Debug)]
pub(super) enum Slot {
    Code(Vec<Vec<u32>>),
    Pointer,
}

#[derive(Clone, Debug)]
pub(super) struct PlannedQuestion {
    pub name: String,
    pub question: DecisionQuestion,
    pub slots: Vec<Slot>,
    pub grid: Option<Grid>,
}

#[derive(Clone, Debug)]
struct Escaped {
    text: String,
    boundaries: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Part {
    Literal(String),
    Media(usize),
    State(Escaped),
    Marker(String),
}

#[derive(Clone, Debug)]
pub(crate) struct DecisionPrompt {
    pub text: String,
    pub media: Vec<DecisionMedia>,
    prefix: Vec<Part>,
    suffixes: Vec<Vec<Part>>,
    questions: Vec<PlannedQuestion>,
    state_text: String,
    state_json: Option<serde_json::Value>,
    box_edges: String,
}

#[derive(Clone, Debug)]
pub(crate) struct DecisionPlan {
    pub ids: Vec<u32>,
    pub state_len: usize,
    pub question_ids: Vec<i32>,
    pub slot_positions: Vec<usize>,
    pub(super) questions: Vec<PlannedQuestion>,
    pub(super) state_text: String,
    pub(super) state_json: Option<serde_json::Value>,
    pub(super) state_offsets: Vec<Option<Range<usize>>>,
    pub(super) box_edges: String,
}

pub struct DecisionReadout {
    pub positions: Vec<usize>,
    pub token_ids: Vec<u32>,
}

impl DecisionPlan {
    pub(crate) fn code_readout(&self) -> DecisionReadout {
        let mut positions = Vec::new();
        let mut token_ids = BTreeSet::new();
        for (slot, &position) in self
            .questions
            .iter()
            .flat_map(|question| &question.slots)
            .zip(&self.slot_positions)
        {
            if let Slot::Code(groups) = slot {
                positions.push(position);
                token_ids.extend(groups.iter().flatten().copied());
            }
        }
        DecisionReadout {
            positions,
            token_ids: token_ids.into_iter().collect(),
        }
    }
}

impl DecisionPrompt {
    pub(crate) fn prepare<'a>(
        request: &'a DecisionRequest,
        tokenizer: &'a Tokenizer,
        config: &'a DecisionConfig,
    ) -> Result<impl Iterator<Item = Result<Self>> + 'a> {
        if !config.d1_calibration_complete {
            for (name, question) in &request.questions {
                ensure!(
                    matches!(
                        question,
                        DecisionQuestion::Noul { .. }
                            | DecisionQuestion::Choice { multiple: false, .. }
                            | DecisionQuestion::Score { .. }
                    ),
                    "Decision question `{name}` requires full calibration. Standalone d1 GGUF supports only noul, single choice, and score. Load with --tok-model-id LiquidAI/d1-2B (or LiquidAI/d1-Omni-2B for Omni) to enable the full decision API."
                );
            }
        }
        request.validate()?;
        config.d1_calibration.validate()?;
        ensure!(
            config.d1_attention == "set",
            "LFM3 decisions require the checkpoint's set attention"
        );
        ensure!(
            (1..=2).contains(&config.d1_prompt),
            "Unsupported decision prompt version"
        );
        ensure!(
            matches!(config.d1_box_edges.as_str(), "median" | "interp"),
            "Unsupported decision box readout"
        );
        ensure!(
            tokenizer.token_to_id(BOS).is_some(),
            "LFM3 decision tokenizer is missing its BOS token"
        );
        let size = request.set_size.unwrap_or(config.d1_set_size);
        ensure!(size > 0, "Decision set size must be positive");
        let state_text = match &request.state {
            serde_json::Value::Null => String::new(),
            serde_json::Value::String(text) => text.clone(),
            value => serde_json::to_string_pretty(value)?,
        };
        let state_json =
            (!request.state.is_null() && !request.state.is_string()).then(|| request.state.clone());
        Ok((0..request.questions.len())
            .step_by(size)
            .map(move |start| {
                let group = request.questions.get_range(start..).unwrap();
                let mut media = request.media.clone();
                let mut prefix = vec![Part::Literal(format!("{BOS}{IM_START}user\n"))];
                prefix.extend((0..media.len()).map(Part::Media));
                if !request.state.is_null() {
                    prefix.push(Part::State(plain(tokenizer, &state_text)?));
                    prefix.push(Part::Literal("\n\n\nQUESTION:\n".to_owned()));
                }
                let mut suffixes = Vec::new();
                let mut plans = Vec::new();
                for (index, (name, question)) in group.iter().take(size).enumerate() {
                    let (mut parts, plan) =
                        render_question(name, question, tokenizer, config, &mut media)?;
                    if index > 0 && !request.state.is_null() {
                        parts.insert(0, Part::Literal("QUESTION:\n".to_owned()));
                    }
                    suffixes.push(parts);
                    plans.push(plan);
                }
                let text = prefix
                    .iter()
                    .chain(suffixes.iter().flatten())
                    .map(|part| initial_part(part, &media))
                    .collect::<String>();
                Ok(Self {
                    text,
                    media,
                    prefix,
                    suffixes,
                    questions: plans,
                    state_text: state_text.clone(),
                    state_json: state_json.clone(),
                    box_edges: config.d1_box_edges.clone(),
                })
            }))
    }

    pub(crate) fn plan(
        &self,
        tokenizer: &Tokenizer,
        expanded_text: &str,
    ) -> Result<(DecisionPlan, Vec<(usize, usize)>)> {
        let mut cursor = 0;
        let mut state = None;
        let mut ignored = Vec::new();
        walk_parts(
            &self.prefix,
            &self.media,
            expanded_text,
            &mut cursor,
            &mut ignored,
            &mut state,
        )?;
        let prefix_end = cursor;
        let mut ranges = Vec::new();
        let mut markers = Vec::new();
        for parts in &self.suffixes {
            let start = cursor;
            let mut ends = Vec::new();
            walk_parts(
                parts,
                &self.media,
                expanded_text,
                &mut cursor,
                &mut ends,
                &mut state,
            )?;
            ranges.push(start..cursor);
            markers.push(ends);
        }
        ensure!(
            cursor == expanded_text.len(),
            "Unexpected text after expanded decision prompt"
        );
        let first_end = ranges[0].end;
        let first = encoding(tokenizer, &expanded_text[..first_end])?;
        let first_suffix_len = encode(tokenizer, &expanded_text[prefix_end..first_end])?.len();
        let state_len = first
            .len()
            .checked_sub(first_suffix_len)
            .context("Decision state token boundary is invalid")?;
        let mut ids = first.get_ids().to_vec();
        let mut offsets = first.get_offsets().to_vec();
        let mut question_ids = vec![-1; state_len];
        question_ids.resize(ids.len(), 1);
        for (index, range) in ranges.iter().enumerate().skip(1) {
            let own = encoding(tokenizer, &expanded_text[range.clone()])?;
            ids.extend_from_slice(own.get_ids());
            offsets.extend(
                own.get_offsets()
                    .iter()
                    .map(|&(a, b)| (a + range.start, b + range.start)),
            );
            question_ids.resize(ids.len(), i32::try_from(index + 1)?);
        }
        let mut slot_positions = Vec::new();
        for (range, ends) in ranges.iter().zip(markers) {
            if ends.is_empty() {
                slot_positions.push(
                    offsets
                        .iter()
                        .rposition(|&(a, b)| a < range.end && b <= range.end)
                        .context("Decision answer slot is missing")?,
                );
            } else {
                for end in ends {
                    let position = offsets
                        .iter()
                        .position(|&(a, b)| a < end && end <= b)
                        .context("Decision marker has no token")?;
                    let (_, stop) = offsets[position];
                    ensure!(
                        expanded_text[end..stop].trim().is_empty(),
                        "Decision marker merged with following text"
                    );
                    slot_positions.push(position);
                }
            }
        }
        let mut state_offsets = vec![None; ids.len()];
        if let Some((start, escaped)) = state {
            let stop = start + escaped.text.len();
            for (index, &(a, b)) in offsets.iter().enumerate() {
                let (lo, hi) = (a.max(start), b.min(stop));
                if lo < hi
                    && expanded_text[a..lo].trim().is_empty()
                    && expanded_text[hi..b].trim().is_empty()
                {
                    let (left, right) = (
                        escaped.boundaries[lo - start],
                        escaped.boundaries[hi - start],
                    );
                    if left < right && !self.state_text[left..right].trim().is_empty() {
                        state_offsets[index] = Some(left..right);
                    }
                }
            }
        }
        ensure!(
            state_offsets.first().is_some_and(Option::is_none),
            "Decision position zero must be the null pointer"
        );
        Ok((
            DecisionPlan {
                ids,
                state_len,
                question_ids,
                slot_positions,
                questions: self.questions.clone(),
                state_text: self.state_text.clone(),
                state_json: self.state_json.clone(),
                state_offsets,
                box_edges: self.box_edges.clone(),
            },
            offsets,
        ))
    }
}

fn initial_part(part: &Part, media: &[DecisionMedia]) -> String {
    match part {
        Part::Literal(text) | Part::Marker(text) => text.clone(),
        Part::State(state) => state.text.clone(),
        Part::Media(index) => match media[*index] {
            DecisionMedia::Image(_) => format!("{VISION_START}<|image|>{VISION_END}"),
            DecisionMedia::Audio(_) => format!("{VISION_START}<|audio|>{VISION_END}"),
            DecisionMedia::Video(_) => format!("{VISION_START}<|video|>{VISION_END}"),
        },
    }
}

fn walk_parts(
    parts: &[Part],
    media: &[DecisionMedia],
    text: &str,
    cursor: &mut usize,
    markers: &mut Vec<usize>,
    state: &mut Option<(usize, Escaped)>,
) -> Result<()> {
    for part in parts {
        match part {
            Part::Media(index) => consume_media(&media[*index], text, cursor)?,
            _ => {
                let literal = match part {
                    Part::Literal(text) | Part::Marker(text) => text,
                    Part::State(value) => &value.text,
                    _ => unreachable!(),
                };
                ensure!(
                    text[*cursor..].starts_with(literal),
                    "Media expansion changed decision prompt text"
                );
                if let Part::State(value) = part {
                    *state = Some((*cursor, value.clone()));
                }
                *cursor += literal.len();
                if matches!(part, Part::Marker(_)) {
                    markers.push(*cursor);
                }
            }
        }
    }
    Ok(())
}

fn consume_media(media: &DecisionMedia, text: &str, cursor: &mut usize) -> Result<()> {
    let (token, frames) = match media {
        DecisionMedia::Image(_) => ("<|image|>", 1),
        DecisionMedia::Audio(_) => ("<|audio|>", 1),
        DecisionMedia::Video(video) => ("<|video|>", video.frames.len().div_ceil(2)),
    };
    let video = matches!(media, DecisionMedia::Video(_));
    if video {
        ensure!(
            text[*cursor..].starts_with(VISION_START),
            "Decision video start delimiter is missing"
        );
        *cursor += VISION_START.len();
    }
    for _ in 0..frames {
        if video {
            let rest = &text[*cursor..];
            let end = rest
                .find(" seconds>")
                .context("Decision video timestamp is missing")?;
            ensure!(
                rest.starts_with('<') && rest[1..end].parse::<f64>().is_ok(),
                "Invalid decision video timestamp"
            );
            *cursor += end + " seconds>".len();
        }
        ensure!(
            text[*cursor..].starts_with(VISION_START),
            "Decision media start delimiter is missing"
        );
        *cursor += VISION_START.len();
        let start = *cursor;
        while text[*cursor..].starts_with(token) {
            *cursor += token.len();
        }
        ensure!(
            *cursor > start && text[*cursor..].starts_with(VISION_END),
            "Decision media expansion is empty or malformed"
        );
        *cursor += VISION_END.len();
    }
    if video {
        ensure!(
            text[*cursor..].starts_with(VISION_END),
            "Decision video end delimiter is missing"
        );
        *cursor += VISION_END.len();
    }
    Ok(())
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(encoding(tokenizer, text)?.get_ids().to_vec())
}
fn encoding(tokenizer: &Tokenizer, text: &str) -> Result<Encoding> {
    tokenizer.encode(text, false).map_err(anyhow::Error::msg)
}

fn forms(tokenizer: &Tokenizer, labels: &[String]) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for label in labels {
        let own = encode(tokenizer, label)?;
        if own.len() == 1 && !ids.contains(&own[0]) {
            ids.push(own[0]);
        }
    }
    ensure!(
        !ids.is_empty(),
        "Decision option has no single-token spelling"
    );
    Ok(ids)
}

fn binary(tokenizer: &Tokenizer) -> Result<Slot> {
    Ok(Slot::Code(
        [vec!["yes", "Yes", "YES"], vec!["no", "No", "NO"]]
            .into_iter()
            .map(|labels| {
                forms(
                    tokenizer,
                    &labels.into_iter().map(str::to_owned).collect::<Vec<_>>(),
                )
            })
            .collect::<Result<_>>()?,
    ))
}

fn aliases(tokenizer: &Tokenizer, labels: &[String], scheme: usize) -> Result<Vec<String>> {
    let letters = ('A'..='Z').map(|ch| ch.to_string()).collect::<Vec<_>>();
    let pairs = ('A'..='Z')
        .flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")))
        .collect::<Vec<_>>();
    let desired = if labels.iter().all(|label| {
        label.trim().chars().count() == 1 && label.trim().chars().all(char::is_alphabetic)
    }) {
        labels.iter().map(|label| label.trim().to_owned()).collect()
    } else if labels.len() <= letters.len() {
        letters[..labels.len()].to_vec()
    } else if scheme == 2 {
        letters
            .iter()
            .chain(&pairs)
            .take(labels.len())
            .cloned()
            .collect()
    } else {
        (0..labels.len())
            .map(|index| format!("{index:02}"))
            .collect()
    };
    ensure!(
        desired.len() == labels.len(),
        "Too many decision choice options"
    );
    let fallback = letters
        .into_iter()
        .chain((0..100).map(|index| format!("{index:02}")))
        .chain(('a'..='z').map(|ch| ch.to_string()))
        .chain((0..200).map(|index| format!("#{index}")))
        .chain(pairs)
        .collect::<Vec<_>>();
    let mut used = Vec::new();
    desired
        .iter()
        .map(|code| {
            for candidate in std::iter::once(code).chain(&fallback) {
                let ids = encode(tokenizer, candidate)?;
                if ids.len() == 1 && !used.contains(&ids[0]) {
                    used.push(ids[0]);
                    return Ok(candidate.clone());
                }
            }
            bail!("No distinct single-token code for decision option")
        })
        .collect()
}

fn plain(tokenizer: &Tokenizer, text: &str) -> Result<Escaped> {
    let mut specials = tokenizer
        .get_added_tokens_decoder()
        .into_values()
        .filter(|token| {
            token.special && token.content.len() > 1 && !token.content.contains(SPECIAL_BREAK)
        })
        .map(|token| token.content)
        .collect::<Vec<_>>();
    specials.sort_by_key(|value| std::cmp::Reverse(value.len()));
    let mut escaped = Escaped {
        text: text.to_owned(),
        boundaries: (0..=text.len()).collect(),
    };
    if specials.is_empty() {
        return Ok(escaped);
    }
    let pattern = regex::Regex::new(
        &specials
            .iter()
            .map(|token| regex::escape(token))
            .collect::<Vec<_>>()
            .join("|"),
    )?;
    loop {
        let points = pattern
            .find_iter(&escaped.text)
            .map(|found| found.start() + found.as_str().chars().next().unwrap().len_utf8())
            .collect::<Vec<_>>();
        if points.is_empty() {
            break;
        }
        for point in points.into_iter().rev() {
            escaped.text.insert_str(point, SPECIAL_BREAK);
            let original = escaped.boundaries[point];
            escaped.boundaries.splice(
                point..point,
                std::iter::repeat_n(original, SPECIAL_BREAK.len()),
            );
        }
    }
    Ok(escaped)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn render_question(
    name: &str,
    question: &DecisionQuestion,
    tokenizer: &Tokenizer,
    config: &DecisionConfig,
    media: &mut Vec<DecisionMedia>,
) -> Result<(Vec<Part>, PlannedQuestion)> {
    let mut parts = match question.instructions() {
        DecisionInstructions::Text(text) => vec![Part::Literal(plain(tokenizer, text)?.text)],
        DecisionInstructions::Audio(audio) => {
            let index = media.len();
            media.push(DecisionMedia::Audio(audio.clone()));
            vec![Part::Media(index)]
        }
    };
    let mut slots = Vec::new();
    let mut names = Vec::new();
    let mut grid = None;
    let body = match question {
        DecisionQuestion::Noul { criteria, .. } => {
            slots.push(binary(tokenizer)?);
            let extra = criteria
                .as_ref()
                .filter(|criteria| !criteria.is_empty())
                .map(|criteria| {
                    format!(
                        "\nYes: {}\nNo: {}",
                        criteria
                            .get("true")
                            .map(value_text)
                            .unwrap_or_else(|| "None".into()),
                        criteria
                            .get("false")
                            .map(value_text)
                            .unwrap_or_else(|| "None".into())
                    )
                })
                .unwrap_or_default();
            format!("{extra}\n\nReply with yes or no only.")
        }
        DecisionQuestion::Choice {
            criteria, multiple, ..
        } => {
            if *multiple {
                let mut lines = Vec::new();
                for (label, description) in criteria {
                    let quoted = serde_json::to_string(label)?;
                    lines.push(format!(
                        "{quoted}: {}",
                        description
                            .as_deref()
                            .filter(|text| !text.is_empty())
                            .unwrap_or(label)
                    ));
                    names.push(quoted);
                    slots.push(binary(tokenizer)?);
                }
                format!(
                    "\n\nOptions:\n{}\n\nReply with yes or no for each option.",
                    lines.join("\n")
                )
            } else {
                let codes = aliases(
                    tokenizer,
                    &criteria.keys().cloned().collect::<Vec<_>>(),
                    config.d1_codes,
                )?;
                slots.push(Slot::Code(
                    codes
                        .iter()
                        .map(|code| forms(tokenizer, &[code.clone(), format!(" {code}")]))
                        .collect::<Result<_>>()?,
                ));
                let lines = codes
                    .iter()
                    .zip(criteria)
                    .map(|(code, (label, description))| {
                        format!(
                            "{code} {}",
                            description
                                .clone()
                                .filter(|text| !text.is_empty())
                                .unwrap_or_else(|| label.replace('_', " "))
                        )
                    })
                    .collect::<Vec<_>>();
                format!(
                    "\n\nOptions:\n{}\n\nReply with the option code only.",
                    lines.join("\n")
                )
            }
        }
        DecisionQuestion::Score { criteria, .. } => {
            slots.push(Slot::Code(
                (0..criteria.len())
                    .map(|index| forms(tokenizer, &[index.to_string()]))
                    .collect::<Result<_>>()?,
            ));
            format!(
                "\n\n{}\n\nReply with a single digit 0-{} only.",
                criteria
                    .iter()
                    .enumerate()
                    .map(|(index, label)| format!("{index} {label}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                criteria.len() - 1
            )
        }
        DecisionQuestion::Number {
            range: Some([low, high]),
            step,
            unit,
            ..
        } => {
            let size = step
                .map(|step| {
                    (((high - low) / step).round_ties_even() + 1.0).min(GRID_SIZE as f64) as usize
                })
                .unwrap_or(GRID_SIZE);
            let own = Grid::new(tokenizer, config, *low, *high, size, 1)?;
            slots.push(Slot::Code(own.groups.clone()));
            let unit = unit
                .as_ref()
                .map(|unit| format!(" {unit}"))
                .unwrap_or_default();
            let (low, high, step) = (general(*low), general(*high), general(own.step()));
            let body = if own.numeric {
                format!("\n\nEstimate from {low} to {high}{unit}, in steps of {step}. Reply with the number only.")
            } else {
                format!("\n\nEstimate from {low} to {high}{unit} on {size} equally spaced values, step {step}: {} is {low}, {} is {high}. Reply with the code only.", own.codes.first().unwrap(), own.codes.last().unwrap())
            };
            grid = Some(own);
            body
        }
        DecisionQuestion::Text { .. }
        | DecisionQuestion::Date { .. }
        | DecisionQuestion::Number { range: None, .. } => {
            let (kind, unit) = match question {
                DecisionQuestion::Text { .. } => ("text", None),
                DecisionQuestion::Date { .. } => ("date", None),
                DecisionQuestion::Number { unit, .. } => ("number", unit.as_ref()),
                _ => unreachable!(),
            };
            slots.extend([Slot::Pointer, Slot::Pointer]);
            names.extend(["start".into(), "end".into()]);
            format!(
                "\n\nRead the {kind}{} from the state, or nothing if it is absent.",
                unit.map(|unit| format!(" (unit: {unit})"))
                    .unwrap_or_default()
            )
        }
        DecisionQuestion::Box { .. } | DecisionQuestion::Boxes { .. } => {
            let own = Grid::new(
                tokenizer,
                config,
                0.0,
                0.999,
                COORDINATE_SCALE,
                COORDINATE_SCALE,
            )?;
            let legend = if own.numeric {
                "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, left and right from its left edge.".to_owned()
            } else {
                format!("Each edge is a code on a {}-value scale: {} is 0 (top or left), {} is 1 (bottom or right). Values are equally spaced.", own.values.len(), own.codes.first().unwrap(), own.codes.last().unwrap())
            };
            let multiple = matches!(question, DecisionQuestion::Boxes { .. });
            let count = match question {
                DecisionQuestion::Boxes { max, .. } => max.unwrap_or(DEFAULT_BOX_COUNT),
                _ => 1,
            };
            ensure!(
                !multiple || config.d1_prompt != 1 || count <= DEFAULT_BOX_COUNT,
                "This checkpoint supports at most {DEFAULT_BOX_COUNT} boxes"
            );
            for index in 1..=count {
                names.push(if multiple {
                    format!("present {index}")
                } else {
                    "present".into()
                });
                slots.push(binary(tokenizer)?);
                for edge in EDGES {
                    names.push(if multiple {
                        format!("{edge} {index}")
                    } else {
                        edge.to_owned()
                    });
                    slots.push(Slot::Code(own.groups.clone()));
                }
            }
            let body = if multiple {
                format!("\n\nEvery one of them, {}in reading order (left edge, then top). Present k is yes when there are at least k. {legend} Reply with yes or no at each present, then its top, bottom, left and right.", if config.d1_prompt == 1 { "16 at most, " } else { "" })
            } else {
                format!("\n\nPresence is yes or no. {legend} Reply with yes or no at present, then one {} at top, bottom, left and right.", if own.numeric { "number" } else { "code" })
            };
            grid = Some(own);
            body
        }
    };
    parts.push(Part::Literal(format!(
        "{}{ASSISTANT}",
        plain(tokenizer, &body)?.text
    )));
    for (index, marker) in names.iter().enumerate() {
        if index > 0 {
            parts.push(Part::Literal("\n".into()));
        }
        parts.push(Part::Marker(format!("{}:", plain(tokenizer, marker)?.text)));
    }
    if let DecisionQuestion::Boxes { max, .. } = question {
        if max.unwrap_or(DEFAULT_BOX_COUNT) < DEFAULT_BOX_COUNT || config.d1_prompt > 1 {
            let text = parts
                .iter()
                .map(|part| initial_part(part, media))
                .collect::<String>()
                + "\n";
            let last = encode(tokenizer, &text)?
                .pop()
                .context("Decision box prompt is empty")?;
            if tokenizer
                .decode(&[last], false)
                .map_err(anyhow::Error::msg)?
                .trim_end()
                .ends_with(':')
            {
                parts.push(Part::Literal("\n".into()));
            }
        }
    }
    Ok((
        parts,
        PlannedQuestion {
            name: name.to_owned(),
            question: question.clone(),
            slots,
            grid,
        },
    ))
}

fn value_text(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| match value {
            serde_json::Value::Null => "None".into(),
            serde_json::Value::Bool(true) => "True".into(),
            serde_json::Value::Bool(false) => "False".into(),
            _ => value.to_string(),
        })
}

fn general(value: f64) -> String {
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent = exponent.parse::<i32>().unwrap();
    if !(-4..6).contains(&exponent) {
        format!(
            "{}e{exponent:+03}",
            mantissa.trim_end_matches('0').trim_end_matches('.')
        )
    } else {
        let precision = usize::try_from((5 - exponent).max(0)).unwrap();
        let text = format!("{value:.precision$}");
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text
        }
    }
}
