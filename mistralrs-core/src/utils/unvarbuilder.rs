use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use candle_core::Tensor;
use candle_nn::{Embedding, LayerNorm, Linear};
use mistralrs_quant::QuantMethod;

use crate::layers::RmsNorm;

pub trait ToTensors {
    /// Tensor names to tensors
    fn to_tensors(&self) -> HashMap<String, Tensor>;
}

impl ToTensors for Embedding {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([("weight".to_string(), self.embeddings().clone())])
    }
}

impl ToTensors for RmsNorm {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([("weight".to_string(), self.weight().clone())])
    }
}

impl ToTensors for LayerNorm {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([
            ("weight".to_string(), self.weight().clone()),
            ("bias".to_string(), self.bias().clone()),
        ])
    }
}

impl ToTensors for Linear {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.weight().clone());
        if let Some(bias) = self.bias() {
            map.insert("bias".to_string(), bias.clone());
        }
        map
    }
}

impl ToTensors for Arc<dyn QuantMethod> {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        let (w, b) = match self.unquant_weight_bias() {
            Some(x) => x,
            None => return HashMap::new(),
        };
        let mut map = HashMap::new();
        map.insert("weight".to_string(), w);
        if let Some(bias) = b {
            map.insert("bias".to_string(), bias.clone());
        }
        map
    }
}

pub struct UnVarBuilder {
    data: Arc<RwLock<HashMap<String, Tensor>>>,
    path: Vec<String>,
}

impl UnVarBuilder {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            path: Vec::new(),
        }
    }

    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
        }
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        self.push_prefix(s)
    }

    pub fn add<T: ToTensors>(&self, item: &T) {
        let mut data = self.data.write().expect("Write failed!");
        let path = self.path.join(".");
        data.extend(
            item.to_tensors()
                .into_iter()
                .map(|(n, t)| (format!("{path}.{n}"), t))
                .collect::<Vec<(_, _)>>(),
        );
    }

    pub fn to_safetensors(&self) -> Vec<(String, Tensor)> {
        let data = self.data.read().expect("Read failed!");
        data.iter().map(|(p, t)| (p.clone(), t.clone())).collect()
    }
}
