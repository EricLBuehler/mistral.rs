use std::{collections::VecDeque, ops::Range};

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use indexmap::IndexMap;

use super::{
    calibration::DecisionCalibration,
    parsing::{parse_date, parse_number, written_unit},
    prompt::{DecisionPlan, Grid, PlannedQuestion, Slot},
};
use crate::decision::{DecisionAnswer, DecisionBox, DecisionFoundBox, DecisionQuestion};

const TEXT_SPAN_LIMIT: usize = 512;
const VALUE_SPAN_LIMIT: usize = 32;
const BOX_CONFIDENCE_DRAWS: u32 = 512;
const BOX_SLOTS: usize = 5;
const BOX_CONFIDENCE_IOU: f64 = 0.5;
const FOUND_THRESHOLD: f64 = 0.5;

impl DecisionPlan {
    pub(crate) fn decode(
        &self,
        hidden: &Tensor,
        code_logits: &Tensor,
        calibration: &DecisionCalibration,
    ) -> Result<IndexMap<String, DecisionAnswer>> {
        let rows = self.code_probabilities(code_logits, calibration)?;
        let mut row = 0;
        let mut slot = 0;
        let mut answers = IndexMap::new();
        for question in &self.questions {
            let answer = if matches!(question.slots.first(), Some(Slot::Pointer)) {
                self.read_span(
                    question,
                    hidden,
                    &self.slot_positions[slot..slot + 2],
                    calibration,
                )?
            } else {
                let count = question.slots.len();
                let answer = answer_codes(
                    question,
                    &rows[row..row + count],
                    calibration,
                    &self.box_edges,
                )?;
                row += count;
                answer
            };
            slot += question.slots.len();
            answers.insert(question.name.clone(), answer);
        }
        Ok(answers)
    }

    fn code_probabilities(
        &self,
        logits: &Tensor,
        calibration: &DecisionCalibration,
    ) -> Result<Vec<Vec<f64>>> {
        let code_slots = self
            .questions
            .iter()
            .flat_map(|question| {
                question.slots.iter().filter_map(move |slot| match slot {
                    Slot::Code(groups) => Some((question, groups)),
                    Slot::Pointer => None,
                })
            })
            .collect::<Vec<_>>();
        if code_slots.is_empty() {
            return Ok(Vec::new());
        }
        let token_ids = self.code_readout().token_ids;
        let gathered = logits
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        code_slots
            .into_iter()
            .zip(gathered)
            .map(|((question, groups), values)| {
                let temperature = match &question.question {
                    DecisionQuestion::Noul { .. } => calibration.temperature("noul", groups.len()),
                    DecisionQuestion::Choice {
                        multiple: false, ..
                    } => calibration.temperature("choice", groups.len()),
                    DecisionQuestion::Score { .. } => {
                        calibration.temperature("score", groups.len())
                    }
                    _ => 1.0,
                };
                let scores = groups
                    .iter()
                    .map(|group| {
                        let score = group
                            .iter()
                            .map(|token| values[token_ids.binary_search(token).unwrap()])
                            .fold(f32::NEG_INFINITY, f32::max);
                        f64::from(score) / temperature
                    })
                    .collect::<Vec<_>>();
                softmax(&scores)
            })
            .collect()
    }

    fn read_span(
        &self,
        question: &PlannedQuestion,
        hidden: &Tensor,
        positions: &[usize],
        calibration: &DecisionCalibration,
    ) -> Result<DecisionAnswer> {
        let limit = if matches!(question.question, DecisionQuestion::Text { .. }) {
            TEXT_SPAN_LIMIT
        } else {
            VALUE_SPAN_LIMIT
        };
        let width = self
            .state_offsets
            .iter()
            .rposition(Option::is_some)
            .map_or(1, |index| index + 1);
        let indices = Tensor::from_vec(
            positions
                .iter()
                .map(|&position| u32::try_from(position))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            positions.len(),
            hidden.device(),
        )?;
        let queries = hidden.index_select(&indices, 0)?.to_dtype(DType::F32)?;
        let states = hidden.narrow(0, 0, width)?.to_dtype(DType::F32)?;
        let scale = f64::from(u32::try_from(hidden.dim(1)?)?).sqrt().recip();
        let logits = queries
            .matmul(&states.t()?)?
            .affine(scale, 0.0)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .enumerate()
                    .map(|(index, score)| {
                        if index == 0 || self.state_offsets[index].is_some() {
                            f64::from(score)
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let raw_found = 1.0 - softmax(&logits[0])?[0];
        let kind = match question.question {
            DecisionQuestion::Text { .. } => "text",
            DecisionQuestion::Date { .. } => "date",
            _ => "number",
        };
        let found = calibration.through(&format!("found:{kind}"), raw_found);
        let mut text = None;
        let mut span = None;
        let mut path = None;
        // The reference gates the raw pointer before applying its calibrated presence threshold.
        if raw_found >= FOUND_THRESHOLD && found >= FOUND_THRESHOLD && width > 1 {
            let (first, last) = best_span(&logits[0], &logits[1], limit);
            if let (Some(first), Some(last)) =
                (&self.state_offsets[first], &self.state_offsets[last])
            {
                let mut bounds = first.start..last.end;
                let trimmed = self.state_text[bounds.clone()].trim();
                bounds.start += self.state_text[bounds.clone()].len()
                    - self.state_text[bounds.clone()].trim_start().len();
                bounds.end = bounds.start + trimmed.len();
                let paths = self
                    .state_json
                    .as_ref()
                    .map(|value| json_paths(&self.state_text, value))
                    .unwrap_or_default();
                let overlaps = paths
                    .iter()
                    .filter(|(range, _)| range.start < bounds.end && bounds.start < range.end)
                    .collect::<Vec<_>>();
                if let [(range, name)] = overlaps.as_slice() {
                    bounds.start = bounds.start.max(range.start);
                    bounds.end = bounds.end.min(range.end);
                    path = Some(name.clone());
                }
                let selected = &self.state_text[bounds.clone()];
                text = Some(if path.is_some() {
                    serde_json::from_str::<String>(&format!("\"{selected}\""))
                        .unwrap_or_else(|_| selected.to_owned())
                } else {
                    selected.to_owned()
                });
                span = Some([
                    self.state_text[..bounds.start].chars().count(),
                    self.state_text[..bounds.end].chars().count(),
                ]);
            }
        }
        Ok(match &question.question {
            DecisionQuestion::Text { .. } => DecisionAnswer::Text {
                text,
                span,
                found,
                path,
            },
            DecisionQuestion::Date { .. } => DecisionAnswer::Date {
                date: text
                    .as_deref()
                    .and_then(|text| parse_date(text, &self.state_text)),
                text,
                span,
                found,
                path,
            },
            DecisionQuestion::Number { unit, .. } => {
                let unit = unit
                    .clone()
                    .or_else(|| text.as_deref().and_then(written_unit));
                DecisionAnswer::Number {
                    number: text
                        .as_deref()
                        .and_then(|text| parse_number(text, &self.state_text, unit.as_deref())),
                    unit,
                    interval: None,
                    text,
                    span,
                    found: Some(found),
                    path,
                }
            }
            _ => unreachable!(),
        })
    }
}

fn softmax(scores: &[f64]) -> Result<Vec<f64>> {
    ensure!(
        !scores
            .iter()
            .any(|score| score.is_nan() || *score == f64::INFINITY),
        "Decision logits are nonfinite"
    );
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(maximum.is_finite(), "Decision options have no finite logit");
    let weights = scores
        .iter()
        .map(|score| (score - maximum).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

fn best_span(start: &[f64], end: &[f64], limit: usize) -> (usize, usize) {
    let mut window = VecDeque::new();
    let mut added = 1;
    let mut best = f64::NEG_INFINITY;
    let mut selected = (1, 1);
    for (first, probability) in start.iter().enumerate().skip(1) {
        while added < end.len().min(first + limit) {
            while window.back().is_some_and(|&index| end[index] < end[added]) {
                window.pop_back();
            }
            window.push_back(added);
            added += 1;
        }
        while window.front().is_some_and(|&index| index < first) {
            window.pop_front();
        }
        let last = *window.front().unwrap();
        let score = probability + end[last];
        if score > best {
            best = score;
            selected = (first, last);
        }
    }
    selected
}

#[allow(clippy::cast_precision_loss)]
fn answer_codes(
    question: &PlannedQuestion,
    rows: &[Vec<f64>],
    calibration: &DecisionCalibration,
    box_edges: &str,
) -> Result<DecisionAnswer> {
    let probabilities = &rows[0];
    Ok(match &question.question {
        DecisionQuestion::Noul { .. } => DecisionAnswer::Noul {
            noul: probabilities[0],
        },
        DecisionQuestion::Choice {
            criteria, multiple, ..
        } => {
            if *multiple {
                let probabilities = criteria
                    .keys()
                    .cloned()
                    .zip(
                        rows.iter()
                            .map(|row| calibration.through("multiple", row[0])),
                    )
                    .collect::<IndexMap<_, _>>();
                let choices = probabilities
                    .iter()
                    .filter_map(|(label, &value)| {
                        (value >= FOUND_THRESHOLD).then_some(label.clone())
                    })
                    .collect();
                DecisionAnswer::Choice {
                    choice: None,
                    choices: Some(choices),
                    confidence: None,
                    probabilities,
                }
            } else {
                let mut best = 0;
                for index in 1..probabilities.len() {
                    if probabilities[index] > probabilities[best] {
                        best = index;
                    }
                }
                DecisionAnswer::Choice {
                    choice: Some(criteria.get_index(best).unwrap().0.clone()),
                    choices: None,
                    confidence: Some(probabilities[best]),
                    probabilities: criteria
                        .keys()
                        .cloned()
                        .zip(probabilities.iter().copied())
                        .collect(),
                }
            }
        }
        DecisionQuestion::Score { criteria, .. } => DecisionAnswer::Score {
            score: probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum(),
            confidence: probabilities.iter().copied().fold(0.0, f64::max),
            probabilities: probabilities
                .iter()
                .enumerate()
                .map(|(index, &value)| (index.to_string(), value))
                .collect(),
            legend: criteria
                .iter()
                .enumerate()
                .map(|(index, value)| (index.to_string(), value.clone()))
                .collect(),
        },
        DecisionQuestion::Number {
            range: Some(_),
            unit,
            ..
        } => {
            let grid = question.grid.as_ref().unwrap();
            DecisionAnswer::Number {
                number: Some(quantile(
                    grid,
                    probabilities,
                    calibration.level("estimate", 0.5),
                )),
                unit: unit.clone(),
                interval: Some([
                    quantile(grid, probabilities, calibration.level("estimate", 0.1)),
                    quantile(grid, probabilities, calibration.level("estimate", 0.9)),
                ]),
                text: None,
                span: None,
                found: None,
                path: None,
            }
        }
        DecisionQuestion::Box { .. } => {
            let found = calibration.through("found:box", probabilities[0]);
            let bounds = (found >= FOUND_THRESHOLD)
                .then(|| median_box(question.grid.as_ref().unwrap(), &rows[1..], box_edges));
            let confidence = bounds.as_ref().map_or(0.0, |bounds| {
                calibration.through(
                    "box",
                    box_confidence(question.grid.as_ref().unwrap(), &rows[1..], bounds),
                )
            });
            DecisionAnswer::Box {
                bounds,
                found,
                confidence,
            }
        }
        DecisionQuestion::Boxes { .. } => {
            let grid = question.grid.as_ref().unwrap();
            let mut presence = Vec::new();
            let mut previous: f64 = 1.0;
            for block in rows.as_chunks::<BOX_SLOTS>().0 {
                previous = previous.min(calibration.through("found:boxes", block[0][0]));
                presence.push(previous);
            }
            let mut boxes = Vec::new();
            for (&found, block) in presence.iter().zip(rows.as_chunks::<BOX_SLOTS>().0) {
                if found < FOUND_THRESHOLD {
                    break;
                }
                let bounds = median_box(grid, &block[1..], box_edges);
                let confidence =
                    calibration.through("boxes", box_confidence(grid, &block[1..], &bounds));
                boxes.push(DecisionFoundBox {
                    bounds,
                    found,
                    confidence,
                });
            }
            boxes.sort_by(|a, b| {
                a.bounds
                    .left
                    .total_cmp(&b.bounds.left)
                    .then(a.bounds.top.total_cmp(&b.bounds.top))
            });
            DecisionAnswer::Boxes {
                count: boxes.len(),
                at_least: !boxes.is_empty() && boxes.len() == presence.len(),
                boxes,
                presence,
            }
        }
        _ => unreachable!(),
    })
}

fn quantile(grid: &Grid, probabilities: &[f64], level: f64) -> f64 {
    let total = probabilities.iter().sum::<f64>();
    let mut cumulative = 0.0;
    for (&value, probability) in grid.values.iter().zip(probabilities) {
        cumulative += probability / total;
        if cumulative >= level {
            return value;
        }
    }
    *grid.values.last().unwrap()
}

fn median_box(grid: &Grid, rows: &[Vec<f64>], method: &str) -> DecisionBox {
    let edges = rows
        .iter()
        .map(|probabilities| {
            if method != "interp" {
                return quantile(grid, probabilities, 0.5);
            }
            let total = probabilities.iter().sum::<f64>();
            let mut mass = 0.0;
            for (&value, probability) in grid.values.iter().zip(probabilities) {
                let share = probability / total;
                if share > 0.0 && mass + share >= 0.5 {
                    return (value - grid.step() / 2.0 + (0.5 - mass) / share * grid.step())
                        .clamp(grid.values[0], *grid.values.last().unwrap());
                }
                mass += share;
            }
            *grid.values.last().unwrap()
        })
        .collect::<Vec<_>>();
    rectangle(grid, &edges)
}

fn rectangle(grid: &Grid, edges: &[f64]) -> DecisionBox {
    let ordered = |a: f64, b: f64| {
        if a < b {
            return (a, b);
        }
        let half = grid.step() / 2.0;
        let middle = ((a + b) / 2.0).clamp(half, 1.0 - half);
        (middle - half, middle + half)
    };
    let (top, bottom) = ordered(edges[0], edges[1]);
    let (left, right) = ordered(edges[2], edges[3]);
    DecisionBox {
        top,
        bottom,
        left,
        right,
    }
}

fn box_confidence(grid: &Grid, rows: &[Vec<f64>], bounds: &DecisionBox) -> f64 {
    let cdfs = rows
        .iter()
        .map(|row| {
            let total = row.iter().sum::<f64>();
            let mut value = 0.0;
            row.iter()
                .map(|probability| {
                    value += probability / total;
                    value
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut hits = 0u32;
    for draw in 1..=BOX_CONFIDENCE_DRAWS {
        let edges = [2u32, 3, 5, 7]
            .iter()
            .zip(&cdfs)
            .map(|(&base, cdf)| {
                let mut index = draw;
                let mut value = 0.0;
                let mut scale = 1.0;
                while index > 0 {
                    scale /= f64::from(base);
                    value += f64::from(index % base) * scale;
                    index /= base;
                }
                grid.values[cdf
                    .partition_point(|&point| point < value)
                    .min(grid.values.len() - 1)]
            })
            .collect::<Vec<_>>();
        let sample = rectangle(grid, &edges);
        let overlap = (sample.bottom.min(bounds.bottom) - sample.top.max(bounds.top)).max(0.0)
            * (sample.right.min(bounds.right) - sample.left.max(bounds.left)).max(0.0);
        let area =
            |bounds: &DecisionBox| (bounds.bottom - bounds.top) * (bounds.right - bounds.left);
        if overlap / (area(&sample) + area(bounds) - overlap) >= BOX_CONFIDENCE_IOU {
            hits += 1;
        }
    }
    f64::from(hits) / f64::from(BOX_CONFIDENCE_DRAWS)
}

fn json_paths(text: &str, value: &serde_json::Value) -> Vec<(Range<usize>, String)> {
    fn skip(text: &str, cursor: &mut usize) {
        while text
            .as_bytes()
            .get(*cursor)
            .is_some_and(u8::is_ascii_whitespace)
        {
            *cursor += 1;
        }
    }
    fn string(text: &str, cursor: &mut usize) -> Range<usize> {
        *cursor += 1;
        let start = *cursor;
        while text.as_bytes()[*cursor] != b'"' {
            if text.as_bytes()[*cursor] == b'\\' {
                *cursor += 1;
            }
            *cursor += 1;
        }
        let end = *cursor;
        *cursor += 1;
        start..end
    }
    fn walk(
        text: &str,
        value: &serde_json::Value,
        cursor: &mut usize,
        path: &str,
        output: &mut Vec<(Range<usize>, String)>,
    ) {
        skip(text, cursor);
        match value {
            serde_json::Value::String(_) => output.push((string(text, cursor), path.to_owned())),
            serde_json::Value::Object(object) => {
                *cursor += 1;
                for (index, (name, value)) in object.iter().enumerate() {
                    if index > 0 {
                        skip(text, cursor);
                        *cursor += 1;
                    }
                    skip(text, cursor);
                    string(text, cursor);
                    skip(text, cursor);
                    *cursor += 1;
                    walk(
                        text,
                        value,
                        cursor,
                        &format!("{path}/{}", name.replace('~', "~0").replace('/', "~1")),
                        output,
                    );
                }
                skip(text, cursor);
                *cursor += 1;
            }
            serde_json::Value::Array(array) => {
                *cursor += 1;
                for (index, value) in array.iter().enumerate() {
                    if index > 0 {
                        skip(text, cursor);
                        *cursor += 1;
                    }
                    walk(text, value, cursor, &format!("{path}/{index}"), output);
                }
                skip(text, cursor);
                *cursor += 1;
            }
            _ => {
                while text
                    .as_bytes()
                    .get(*cursor)
                    .is_some_and(|byte| !byte.is_ascii_whitespace() && !b",]}".contains(byte))
                {
                    *cursor += 1;
                }
            }
        }
    }
    let mut output = Vec::new();
    walk(text, value, &mut 0, "", &mut output);
    output
}
