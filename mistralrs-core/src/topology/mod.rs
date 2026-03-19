use std::{fs, io::Read, ops::Range, path::Path};

use candle_core::Device;
use indexmap::IndexMap;
use itertools::Itertools;
use mistralrs_quant::IsqType;
use regex::Regex;
use serde::Deserialize;

use crate::parse_isq_value;

const DEVICE_PATTERN: &str = r"^(cpu|cuda\[(\d+)\]|metal\[(\d+)\])$";

#[derive(Deserialize)]
pub struct DeserLayerTopology {
    isq: Option<String>,
    device: Option<String>,
}

#[derive(Deserialize)]
pub struct DeserTopology(IndexMap<String, DeserLayerTopology>);

#[derive(Clone, Debug)]
pub struct LayerTopology {
    pub isq: Option<IsqType>,
    pub device: Option<Device>,
}

#[derive(PartialEq, Eq, Debug)]
struct CustomRange {
    start: usize,
    end: usize,
    index: usize,
}

impl From<CustomRange> for Range<usize> {
    fn from(value: CustomRange) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

impl Ord for CustomRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order based on end position followed by declaration order so later ranges override
        self.end
            .cmp(&other.end)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for CustomRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub layers: Vec<Option<LayerTopology>>,
    pub patterns: Vec<(Regex, LayerTopology)>,
}

impl Topology {
    /// Create an empty topology.
    pub fn empty() -> Self {
        Topology {
            layers: Vec::new(),
            patterns: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Topology {
            layers: vec![None; cap],
            patterns: Vec::new(),
        }
    }

    pub fn is_dummy_device_map(&self) -> bool {
        self.layers
            .iter()
            .all(|l| l.is_none() || l.as_ref().is_some_and(|l| l.device.is_none()))
            && self
                .patterns
                .iter()
                .all(|(_, topo)| topo.device.as_ref().is_none())
    }

    pub fn with_range(mut self, range: Range<usize>, layer: LayerTopology) -> Self {
        if self.layers.len() < range.end {
            self.layers
                .extend(vec![None; range.end - self.layers.len()]);
        }
        for i in range.start..range.end {
            self.layers[i] = Some(layer.clone());
        }
        self
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(topology: &str) -> anyhow::Result<Self> {
        let deser: DeserTopology = serde_saphyr::from_str(topology)?;
        let device_regex = Regex::new(DEVICE_PATTERN)?;

        let mut range_layers = Vec::new();
        let mut pattern_layers = Vec::new();
        for (index, (selector, DeserLayerTopology { isq, device })) in
            deser.0.into_iter().enumerate()
        {
            let parsed_isq = if let Some(isq) = isq {
                Some(parse_isq_value(&isq, None).map_err(anyhow::Error::msg)?)
            } else {
                None
            };

            let parsed_device = if let Some(device) = device {
                let Some(captures) = device_regex.captures(&device) else {
                    anyhow::bail!(
                        "Device specifier must match regex {DEVICE_PATTERN}. Examples: `cpu`, `cuda[ORD]`, `metal[ORD]`"
                    );
                };
                let device = if let Some(val) = captures.get(2).or(captures.get(3)) {
                    let ord = val.as_str().parse::<usize>()?;
                    let device = device.split('[').collect::<Vec<_>>()[0];
                    match device {
                        "cuda" => Device::new_cuda(ord)?,
                        "metal" => Device::new_metal(ord)?,
                        _ => unreachable!(),
                    }
                } else {
                    Device::Cpu
                };

                Some(device)
            } else {
                None
            };

            if selector.starts_with('/') && selector.ends_with('/') && selector.len() >= 2 {
                let pattern = &selector[1..selector.len() - 1];
                let regex = Regex::new(pattern)
                    .map_err(|err| anyhow::anyhow!("Invalid topology regex `{pattern}`: {err}"))?;
                pattern_layers.push((
                    regex,
                    LayerTopology {
                        isq: parsed_isq,
                        device: parsed_device,
                    },
                ));
                continue;
            }

            let (start, end) = if selector.contains('-') {
                // Range (inclusive, exclusive)
                let Some((start, end)) = selector.splitn(2, '-').collect_tuple() else {
                    anyhow::bail!("Topology range segment must follow the format START-END")
                };
                (start.parse::<usize>()?, end.parse::<usize>()?)
            } else {
                // Single layer here
                let layer = selector.parse::<usize>()?;
                (layer, layer + 1)
            };

            if end <= start {
                anyhow::bail!("Topology range end must be > start, got {end} <= {start}");
            }
            let range = CustomRange { start, end, index };

            range_layers.push((
                range,
                LayerTopology {
                    isq: parsed_isq,
                    device: parsed_device,
                },
            ));
        }
        // Sort so that we increase in end points
        range_layers.sort_by(|(r1, _), (r2, _)| r1.cmp(r2));

        let capacity = range_layers.iter().map(|(r, _)| r.end).max().unwrap_or(0);
        let mut this = if capacity == 0 {
            Self::empty()
        } else {
            Self::with_capacity(capacity)
        };
        for (range, layer) in range_layers {
            for i in range.start..range.end {
                this.layers[i] = Some(layer.clone());
            }
        }
        this.patterns = pattern_layers;
        Ok(this)
    }

    pub fn from_reader<R: Read>(mut reader: R) -> anyhow::Result<Self> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_str(&buf)
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let buf = fs::read_to_string(path)?;
        Self::from_str(&buf)
    }

    pub fn from_option_path<P: AsRef<Path>>(path: Option<P>) -> anyhow::Result<Option<Self>> {
        if let Some(path) = path {
            let buf = fs::read_to_string(path)?;
            Ok(Some(Self::from_str(&buf)?))
        } else {
            Ok(None)
        }
    }

    pub fn layer_for(&self, layer: usize) -> Option<&LayerTopology> {
        self.layers.get(layer).and_then(|lt| lt.as_ref())
    }

    pub fn match_for_name(&self, name: &str) -> Option<LayerTopology> {
        for (regex, layer) in self.patterns.iter().rev() {
            if regex.is_match(name) {
                return Some(layer.clone());
            }
        }
        None
    }

    pub fn pattern_overrides(&self) -> Vec<(Regex, LayerTopology)> {
        self.patterns
            .iter()
            .rev()
            .map(|(regex, topo)| (regex.clone(), topo.clone()))
            .collect()
    }

    pub fn requires_post_quantization(&self) -> bool {
        self.layers.iter().any(|layer| {
            layer
                .as_ref()
                .is_some_and(|layer| layer.isq.is_some() || layer.device.is_some())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_isq(topology: &Topology, layer: usize) -> Option<IsqType> {
        topology
            .layer_for(layer)
            .and_then(|lt| lt.isq.as_ref().copied())
    }

    #[test]
    fn highest_end_range_overrides_lower_end() {
        let yaml = "0-4:\n  isq: Q4K\n2-6:\n  isq: Q6K\n";
        let topology = Topology::from_str(yaml).expect("topology parses");

        assert_eq!(layer_isq(&topology, 0), Some(IsqType::Q4K));
        assert_eq!(layer_isq(&topology, 2), Some(IsqType::Q6K));
        assert_eq!(layer_isq(&topology, 5), Some(IsqType::Q6K));
    }

    #[test]
    fn later_range_with_same_end_wins() {
        let yaml = "0-4:\n  isq: Q4K\n2-4:\n  isq: Q3K\n";
        let topology = Topology::from_str(yaml).expect("topology parses");

        assert_eq!(layer_isq(&topology, 1), Some(IsqType::Q4K));
        assert_eq!(layer_isq(&topology, 2), Some(IsqType::Q3K));
        assert_eq!(layer_isq(&topology, 3), Some(IsqType::Q3K));
    }

    #[test]
    fn regex_overrides_respect_declaration_order() {
        let yaml = r#"'/ffn\./':
  isq: Q4K
'/ffn\.weight$/':
  isq: Q6K
"#;
        let topology = Topology::from_str(yaml).expect("topology parses");

        let match_exact = topology
            .match_for_name("model.layers.2.ffn.weight")
            .expect("regex match");
        assert_eq!(match_exact.isq, Some(IsqType::Q6K));

        let overrides = topology.pattern_overrides();
        assert_eq!(overrides.len(), 2);
        assert_eq!(overrides[0].0.as_str(), "ffn\\.weight$");
        assert_eq!(overrides[1].0.as_str(), "ffn\\.");
    }

    #[test]
    fn match_for_name_returns_none_when_unmatched() {
        let yaml = "0-2:\n  isq: Q4K\n";
        let topology = Topology::from_str(yaml).expect("topology parses");
        assert!(topology.match_for_name("transformer.wte.weight").is_none());
    }
}
