use std::{fs::File, path::Path};

use csv::Reader;
use serde::Deserialize;

pub struct AnyMoeTrainingResult {
    pub steps: usize,
    /// One for each gating layer
    pub final_loss: Vec<f32>,
}

#[derive(Deserialize, Debug)]
pub struct AnyMoeTrainingInputRow {
    pub prompt: String,
    pub expert: usize,
    pub image_urls: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
pub struct AnyMoeTrainingInputs {
    rows: Vec<AnyMoeTrainingInputRow>,
}

impl AnyMoeTrainingInputs {
    /// From a CSV file with the mandatory columns `prompt` (String), `expert` (usize), and the optional
    /// column `image_urls` (`Vec<String>`).
    pub fn from_csv<P: AsRef<Path>>(file: P) -> anyhow::Result<Self> {
        let file = File::open(file)?;
        let mut reader = Reader::from_reader(file);
        let mut rows = Vec::new();
        for result in reader.deserialize() {
            let row: AnyMoeTrainingInputRow = result?;
            rows.push(row);
        }
        Ok(Self { rows })
    }

    /// From a JSON file with the top-level key being `rows` (array), which contains objects with the
    /// keys `prompt` (String), `expert` (usize), `image_urls` (Option<Vec<String>>).
    pub fn from_json<P: AsRef<Path>>(file: P) -> anyhow::Result<Self> {
        let file = File::open(file)?;
        Ok(serde_json::from_reader(file)?)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn into_inner(self) -> Vec<AnyMoeTrainingInputRow> {
        self.rows
    }
}
