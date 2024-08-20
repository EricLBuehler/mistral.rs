use std::{collections::HashMap, ops::Range};

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

#[derive(Clone)]
pub struct LayerTopology {
    isq: Option<IsqType>,
}

pub struct Topology(Vec<Option<LayerTopology>>);

impl Topology {
    fn new() -> Self {
        Topology(Vec::new())
    }

    fn add_from_range(&mut self, range: Range<usize>, topo: LayerTopology) {
        if range.start == 0 && self.0.len() == 0 {
            // Simple case, starting out
            self.0.extend(range.into_iter().map(|_| Some(topo.clone())));
        } else if range.end >= self.0.len() && range.start > self.0.len() {
            // Adding new layers. Add Nones to pad
            self.0.extend(vec![None; self.0.len() - range.start]);
            self.0.extend(range.into_iter().map(|_| Some(topo.clone())));
        } else if range.end >= self.0.len() && range.start < self.0.len() {
            // Replacing some layers at least but the range exceeds
            self.0.extend(vec![None; self.0.len() - (range.end - 1)]);
            self.0.extend(range.into_iter().map(|_| Some(topo.clone())));
        } else {
            assert!(self.0.len() > range.end);
            self.0
                .splice(range.clone(), range.into_iter().map(|_| Some(topo.clone())));
        }
    }

    pub fn from_str(topology: &str) -> anyhow::Result<Self> {
        let deser: DeserTopology = serde_yaml::from_str(topology)?;
        let mut this = Topology::new();

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
            let range = Range { start, end };
            let isq = if let Some(isq) = isq {
                Some(parse_isq_value(&isq).map_err(|e| anyhow::Error::msg(e))?)
            } else {
                None
            };
            let layer_topo = LayerTopology { isq };
            this.add_from_range(range, layer_topo);
        }

        Ok(this)
    }
}
