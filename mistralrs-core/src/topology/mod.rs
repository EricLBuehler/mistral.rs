use std::{collections::HashMap, fs, io::Read, ops::Range, path::Path};

use candle_core::Device;
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
pub struct DeserTopology(HashMap<String, DeserLayerTopology>);

#[derive(Clone, Debug)]
pub struct LayerTopology {
    pub isq: Option<IsqType>,
    pub device: Option<Device>,
}

#[derive(PartialEq, Eq, Debug)]
struct CustomRange {
    start: usize,
    end: usize,
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
        // Order based on end position
        self.end.cmp(&other.end)
    }
}

impl PartialOrd for CustomRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub struct Topology(pub Vec<Option<LayerTopology>>);

impl Topology {
    /// Create an empty topology.
    pub fn empty() -> Self {
        Topology(Vec::new())
    }

    pub fn with_capacity(cap: usize) -> Self {
        Topology(vec![None; cap])
    }

    pub fn is_dummy_device_map(&self) -> bool {
        self.0
            .iter()
            .all(|l| l.is_none() || l.as_ref().is_some_and(|l| l.device.is_none()))
    }

    pub fn with_range(mut self, range: Range<usize>, layer: LayerTopology) -> Self {
        if self.0.len() < range.end {
            self.0.extend(vec![None; range.end - self.0.len()]);
        }
        for i in range.start..range.end {
            self.0[i] = Some(layer.clone());
        }
        self
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(topology: &str) -> anyhow::Result<Self> {
        let deser: DeserTopology = serde_yaml::from_str(topology)?;
        let device_regex = Regex::new(DEVICE_PATTERN)?;

        let mut layers = Vec::new();
        for (range, DeserLayerTopology { isq, device }) in deser.0 {
            // Parse isq
            let (start, end) = if range.contains('-') {
                // Range (inclusive, exclusive)
                let Some((start, end)) = range.splitn(2, '-').collect_tuple() else {
                    anyhow::bail!("Topology range segment must follow the format START-END")
                };
                (start.parse::<usize>()?, end.parse::<usize>()?)
            } else {
                // Single layer here
                let layer = range.parse::<usize>()?;
                (layer, layer + 1)
            };

            if end <= start {
                anyhow::bail!("Topology range end must be > start, got {end} <= {start}");
            }
            let range = CustomRange { start, end };
            let isq = if let Some(isq) = isq {
                Some(parse_isq_value(&isq).map_err(anyhow::Error::msg)?)
            } else {
                None
            };

            // Parse device
            let device = if let Some(device) = device {
                let Some(captures) = device_regex.captures(&device) else {
                    anyhow::bail!("Device specifier must match regex {DEVICE_PATTERN}. Examples: `cpu`, `cuda[ORD]`, `metal[ORD]`");
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

            let layer_topo = LayerTopology { isq, device };
            layers.push((range, layer_topo));
        }
        // Sort so that we increase in end points
        layers.sort_by(|(r1, _), (r2, _)| r1.cmp(r2));

        let mut this = Self::with_capacity(layers.last().unwrap().0.end);
        for (range, layer) in layers {
            for i in range.start..range.end {
                this.0[i] = Some(layer.clone());
            }
        }
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
}
