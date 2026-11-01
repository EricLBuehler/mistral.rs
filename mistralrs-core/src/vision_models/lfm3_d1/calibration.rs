use std::collections::HashMap;

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct DecisionCalibration {
    pub default: f64,
    #[serde(default)]
    pub temperatures: HashMap<String, f64>,
    #[serde(default)]
    pub tables: HashMap<String, Vec<[f64; 2]>>,
}

impl DecisionCalibration {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            std::iter::once(&self.default)
                .chain(self.temperatures.values())
                .all(|temperature| temperature.is_finite() && *temperature > 0.0),
            "Decision calibration temperatures must be finite and positive"
        );
        for (name, knots) in &self.tables {
            ensure!(
                knots.iter().flatten().all(|value| value.is_finite())
                    && knots
                        .windows(2)
                        .all(|pair| pair[0][0] <= pair[1][0] && pair[0][1] <= pair[1][1]),
                "Decision calibration table {name} must be finite and monotone"
            );
        }
        Ok(())
    }

    pub(crate) fn temperature(&self, kind: &str, count: usize) -> f64 {
        let key = match kind {
            "noul" => "noul".to_owned(),
            "score" => format!("score:{count}"),
            _ => format!(
                "choice:{}",
                match count {
                    0..=3 => "small",
                    4..=8 => "mid",
                    9..=26 => "large",
                    _ => "inventory",
                }
            ),
        };
        self.temperatures.get(&key).copied().unwrap_or(self.default)
    }

    pub(crate) fn through(&self, name: &str, value: f64) -> f64 {
        interpolate(
            self.tables.get(name).map(Vec::as_slice).unwrap_or_default(),
            value,
            false,
        )
    }

    pub(crate) fn level(&self, name: &str, value: f64) -> f64 {
        interpolate(
            self.tables.get(name).map(Vec::as_slice).unwrap_or_default(),
            value,
            true,
        )
    }
}

fn interpolate(knots: &[[f64; 2]], value: f64, inverse: bool) -> f64 {
    if knots.is_empty() {
        return value;
    }
    let x = usize::from(inverse);
    let y = usize::from(!inverse);
    let index = knots.partition_point(|knot| knot[x] < value);
    if index == 0 {
        return knots[0][y];
    }
    if index == knots.len() {
        return knots[index - 1][y];
    }
    let (a, b) = (knots[index - 1], knots[index]);
    a[y] + (value - a[x]) * (b[y] - a[y]) / (b[x] - a[x])
}
