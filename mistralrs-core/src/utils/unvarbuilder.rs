use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use candle_core::{quantized::QMatMul, Tensor};
use candle_nn::{Conv2d, Embedding, LayerNorm, Linear};
use itertools::Itertools;
use mistralrs_quant::QuantMethod;

use crate::layers::{F32RmsNorm, GemmaRmsNorm, QLinear, RmsNorm, ScaledEmbedding};

pub trait ToTensors {
    /// Tensor names to tensors
    fn to_tensors(&self) -> HashMap<String, Tensor>;
}

impl ToTensors for Embedding {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([("weight".to_string(), self.embeddings().clone())])
    }
}

impl ToTensors for ScaledEmbedding {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([("weight".to_string(), self.embeddings().clone())])
    }
}

impl ToTensors for RmsNorm {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([("weight".to_string(), self.weight().clone())])
    }
}

impl ToTensors for GemmaRmsNorm {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([("weight".to_string(), self.original_weight().clone())])
    }
}

impl ToTensors for F32RmsNorm {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        HashMap::from_iter([("weight".to_string(), self.weight().clone())])
    }
}

impl ToTensors for LayerNorm {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.weight().clone());
        if let Some(bias) = self.bias() {
            map.insert("bias".to_string(), bias.clone());
        }
        map
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

impl ToTensors for Conv2d {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.weight().clone());
        if let Some(bias) = self.bias() {
            map.insert("bias".to_string(), bias.clone());
        }
        map
    }
}

impl ToTensors for candle_nn::Conv1d {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.weight().clone());
        if let Some(bias) = self.bias() {
            map.insert("bias".to_string(), bias.clone());
        }
        map
    }
}

impl ToTensors for QLinear {
    fn to_tensors(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        match self.inner_ref() {
            QMatMul::Tensor(w) | QMatMul::TensorF16(w) => {
                map.insert("weight".to_string(), w.clone());
                if let Some(bias) = self.bias() {
                    map.insert("bias".to_string(), bias.clone());
                }
            }
            QMatMul::QTensor(_) => return HashMap::new(),
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

    pub fn path(&self) -> String {
        self.path.iter().filter(|p| !p.trim().is_empty()).join(".")
    }

    pub fn add<T: ToTensors>(&self, item: &T) {
        let mut data = self.data.write().expect("Write failed!");
        let path = self.path();
        data.extend(
            item.to_tensors()
                .into_iter()
                .map(|(n, t)| (format!("{path}.{n}"), t))
                .collect::<Vec<(_, _)>>(),
        );
    }

    pub fn add_tensor<S: ToString>(&self, s: S, v: Tensor) {
        let mut data = self.data.write().expect("Write failed!");
        let mut path = self.path.clone();
        path.push(s.to_string());
        data.insert(
            path.into_iter().filter(|p| !p.trim().is_empty()).join("."),
            v,
        );
    }

    pub fn extend(&self, other: Vec<(String, Tensor)>) {
        let mut data = self.data.write().expect("Write failed!");
        let path = self.path();
        data.extend(
            other
                .into_iter()
                .map(|(n, t)| {
                    (
                        if path.is_empty() {
                            n
                        } else {
                            format!("{path}.{n}")
                        },
                        t,
                    )
                })
                .collect::<Vec<(_, _)>>(),
        );
    }

    pub fn to_safetensors(&self) -> Vec<(String, Tensor)> {
        let data = self.data.read().expect("Read failed!");
        data.iter().map(|(p, t)| (p.clone(), t.clone())).collect()
    }
}
