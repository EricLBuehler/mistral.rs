use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Slice {
    pub effective_params: f64,
    pub ffn_hidden_dimensions: Vec<usize>,
    pub layers_skipped: Option<Vec<usize>>,
}

#[derive(Debug)]
pub struct MatformerConfig {
    pub slices: HashMap<String, Slice>,
}

#[derive(Debug, Clone)]
pub struct MatformerSliceConfig {
    pub slice_name: String,
    pub config: Arc<MatformerConfig>,
}

impl MatformerSliceConfig {
    pub fn new(slice_name: String, config: Arc<MatformerConfig>) -> Self {
        Self { slice_name, config }
    }

    pub fn get_slicing(&self) -> Option<&Slice> {
        self.config.get_slicing(&self.slice_name)
    }
}

#[derive(Debug, Deserialize)]
struct CsvRecord {
    name: String,
    #[serde(rename = "# Layers")]
    #[allow(dead_code)]
    num_layers: u32,
    #[serde(rename = "# Effective Params (B)")]
    effective_params: f64,
    #[serde(rename = "MMLU PT accuracy")]
    #[allow(dead_code)]
    mmlu_accuracy: String,
    #[serde(rename = "FFN Hidden Dims")]
    ffn_hidden_dims: String,
    #[serde(rename = "Layers Skipped")]
    layers_skipped: Option<String>,
}

impl MatformerConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path).with_context(|| {
            format!("Failed to open matformer config file: {:?}", path.as_ref())
        })?;
        let reader = BufReader::new(file);

        let mut rdr = csv::Reader::from_reader(reader);
        let mut slices = HashMap::new();

        for result in rdr.deserialize() {
            let record: CsvRecord = result.context("Failed to parse CSV record")?;

            let ffn_hidden_dimensions = parse_ffn_hidden_dims(&record.ffn_hidden_dims)
                .with_context(|| format!("Failed to parse FFN hidden dims for {}", record.name))?;

            let layers_skipped = record
                .layers_skipped
                .as_ref()
                .filter(|s| !s.is_empty())
                .map(|s| parse_layers_skipped(s))
                .transpose()
                .with_context(|| format!("Failed to parse layers skipped for {}", record.name))?;

            let slicing = Slice {
                effective_params: record.effective_params,
                ffn_hidden_dimensions,
                layers_skipped,
            };

            slices.insert(record.name, slicing);
        }

        Ok(MatformerConfig { slices })
    }

    pub fn get_slicing(&self, name: &str) -> Option<&Slice> {
        self.slices.get(name)
    }
}

fn parse_ffn_hidden_dims(s: &str) -> Result<Vec<usize>> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        anyhow::bail!("FFN hidden dims must be enclosed in brackets");
    }

    let inner = &s[1..s.len() - 1];
    let parts: Vec<&str> = inner.split(',').collect();

    let mut dimensions = Vec::with_capacity(parts.len());
    for part in parts {
        let dim = evaluate_expression(part.trim())
            .with_context(|| format!("Failed to evaluate expression: {part}"))?;
        dimensions.push(dim);
    }

    Ok(dimensions)
}

fn parse_layers_skipped(s: &str) -> Result<Vec<usize>> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        anyhow::bail!("Layers skipped must be enclosed in brackets");
    }

    let inner = &s[1..s.len() - 1];
    let parts: Vec<&str> = inner.split(',').collect();

    let mut layers = Vec::with_capacity(parts.len());
    for part in parts {
        let layer = part
            .trim()
            .parse::<usize>()
            .with_context(|| format!("Failed to parse layer number: {part}"))?;
        layers.push(layer);
    }

    Ok(layers)
}

fn evaluate_expression(expr: &str) -> Result<usize> {
    let expr = expr.trim();

    // Handle simple number (with potential underscores)
    if let Ok(num) = expr.replace('_', "").parse::<usize>() {
        return Ok(num);
    }

    // Handle multiplication expressions like "2_048 * 4"
    if expr.contains('*') {
        let parts: Vec<&str> = expr.split('*').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid multiplication expression: {}", expr);
        }

        let left = parts[0]
            .trim()
            .replace('_', "")
            .parse::<usize>()
            .with_context(|| format!("Failed to parse left operand: {}", parts[0]))?;
        let right = parts[1]
            .trim()
            .parse::<usize>()
            .with_context(|| format!("Failed to parse right operand: {}", parts[1]))?;

        return Ok(left * right);
    }

    anyhow::bail!("Unsupported expression format: {}", expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_expression() {
        assert_eq!(evaluate_expression("2048").unwrap(), 2048);
        assert_eq!(evaluate_expression("2_048").unwrap(), 2048);
        assert_eq!(evaluate_expression("2_048 * 4").unwrap(), 8192);
        assert_eq!(evaluate_expression("2048 * 8").unwrap(), 16384);
    }

    #[test]
    fn test_parse_ffn_hidden_dims() {
        let dims = parse_ffn_hidden_dims("[2_048 * 4, 2_048 * 8, 2048 * 6]").unwrap();
        assert_eq!(dims, vec![8192, 16384, 12288]);
    }

    #[test]
    fn test_parse_layers_skipped() {
        let layers = parse_layers_skipped("[20, 21, 22]").unwrap();
        assert_eq!(layers, vec![20, 21, 22]);
    }
}
