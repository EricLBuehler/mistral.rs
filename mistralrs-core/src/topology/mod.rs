use std::{collections::HashMap, fs, io::Read, ops::Range, path::Path};

use itertools::Itertools;
use mistralrs_quant::IsqType;
use serde::Deserialize;

use crate::parse_isq_value;

#[derive(Deserialize)]
pub struct DeserLayerTopology {
    isq: Option<String>,
}

#[derive(Deserialize)]
pub struct DeserTopology(HashMap<String, DeserLayerTopology>);

#[derive(Clone, Debug)]
pub struct LayerTopology {
    pub isq: Option<IsqType>,
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

    /// Add an topology item (which will be expanded to cover all elements) of a range.
    /// Any overlapping items will be overwritten. Padding automatically occurs if gaps occur.
    pub fn with_range(mut self, range: Range<usize>, topo: LayerTopology) -> Self {
        self.add_from_range(range, topo);
        self
    }

    fn add_from_range(&mut self, range: Range<usize>, topo: LayerTopology) {
        let n_repeat = 0..=range.end - range.start;
        if range.start == 0 && self.0.is_empty() {
            // Simple case, starting out
            self.0
                .extend(n_repeat.into_iter().map(|_| Some(topo.clone())));
        } else if range.end >= self.0.len() && range.start > self.0.len() {
            // Adding new layers. Add Nones to pad
            self.0.extend(vec![None; range.start - self.0.len()]);
            self.0
                .extend(n_repeat.into_iter().map(|_| Some(topo.clone())));
        } else if range.end >= self.0.len() && range.start < self.0.len() {
            // Replacing some layers at least but the range exceeds
            self.0.extend(vec![None; range.end - self.0.len()]);
            self.0
                .extend(n_repeat.into_iter().map(|_| Some(topo.clone())));
        } else {
            assert!(self.0.len() > range.end);
            self.0.splice(
                range.clone(),
                n_repeat.into_iter().map(|_| Some(topo.clone())),
            );
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(topology: &str) -> anyhow::Result<Self> {
        let deser: DeserTopology = serde_yaml::from_str(topology)?;

        let mut layers = Vec::new();
        for (range, DeserLayerTopology { isq }) in deser.0 {
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
            let layer_topo = LayerTopology { isq };
            layers.push((range, layer_topo));
        }
        // Sort so that we increase in end points
        layers.sort_by(|(r1, _), (r2, _)| r1.cmp(r2));

        let mut this = Self::empty();
        for (range, layer) in layers {
            this.add_from_range(range.into(), layer);
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
