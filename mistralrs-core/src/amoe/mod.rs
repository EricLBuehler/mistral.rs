use std::{
    fs::File,
    path::Path,
    sync::{Arc, RwLock},
};

use candle_core::{quantized::QMatMul, DType, Device, Result, Tensor, Var, D};
use candle_nn::{linear, Linear, ModuleT, VarBuilder, VarMap};
use csv::Reader;
use serde::{Deserialize, Serialize};

use crate::serde_default_fn;

pub struct AnyMoeTrainingResult {
    pub steps: usize,
    /// One for each gating layer
    pub final_loss: Vec<f32>,
}

#[derive(Deserialize, Debug)]
pub struct AnyMoeTrainingInputRow {
    pub prompt: String,
    pub expert: usize,
}

pub struct AnyMoeTrainingInputs(pub Vec<AnyMoeTrainingInputRow>);

impl AnyMoeTrainingInputs {
    pub fn from_csv<P: AsRef<Path>>(file: P) -> anyhow::Result<Self> {
        let file = File::open(file)?;
        let mut reader = Reader::from_reader(file);
        let mut rows = Vec::new();
        for result in reader.deserialize() {
            let row: AnyMoeTrainingInputRow = result?;
            rows.push(row);
        }
        Ok(Self(rows))
    }
}

/// Implemented by the base model of an AnyMoe.
pub trait AnyMoeBaseModelMixin {
    fn get_vars(&self) -> Vec<Vec<Var>> {
        self.get_mlps()
            .iter()
            .map(|mlp| mlp.get_vars())
            .collect::<Vec<_>>()
    }
    fn done_training(&mut self) {
        let _ = self
            .get_mlps_mut()
            .iter_mut()
            .map(|mlp| mlp.done_training())
            .collect::<Vec<_>>();
    }
    fn trainable_params(&self) -> usize {
        self.get_mlps()
            .iter()
            .map(|mlp| mlp.trainable_params())
            .sum()
    }
    fn take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        self.get_mlps_mut()
            .iter_mut()
            .map(|mlp| mlp.take_cached_gating_output())
            .collect::<Vec<_>>()
    }

    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<VarBuilder>,
        config: AnyMoeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<()>;
    fn get_mlps(&self) -> Vec<&dyn MlpLayer>;
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>>;
}

pub trait MlpLayer: Send + Sync + AnyMoeTrainableLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn get_isq_tensors(&mut self) -> Vec<&mut QMatMul>;
    fn clone(&self) -> Box<dyn MlpLayer>;
}

pub trait AnyMoeTrainableLayer {
    fn get_vars(&self) -> Vec<Var> {
        vec![]
    }
    fn done_training(&mut self) {}
    fn trainable_params(&self) -> usize {
        0
    }
    fn take_cached_gating_output(&mut self) -> Tensor {
        panic!("Gating output is not applicable to this layer.")
    }
}

serde_default_fn!(f64, default_lr, 1e-3);
serde_default_fn!(usize, default_epochs, 100);
serde_default_fn!(usize, default_bs, 4);

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct AnyMoeConfig {
    pub hidden_size: usize,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_bs")]
    pub batch_size: usize,
}

impl AnyMoeConfig {
    pub fn default(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            lr: 1e-3,
            epochs: 100,
            batch_size: 4,
        }
    }
}

#[derive(Clone)]
pub struct MoeGate {
    lin: Linear,
}

impl ModuleT for MoeGate {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let hidden_states = xs.apply(&self.lin)?;
        if train {
            candle_nn::ops::softmax(&hidden_states, D::Minus1)
        } else {
            candle_nn::ops::softmax_last_dim(&hidden_states)
        }
    }
}

pub struct MoeMlp {
    experts: Vec<Box<dyn MlpLayer>>,
    gate: MoeGate,
    training: bool,
    vars: Vec<Var>,
    gating_output: Arc<RwLock<Option<Tensor>>>,
}

impl MoeMlp {
    /// Create a new MoeMlp layer. By default this is in training mode.
    pub fn new(
        experts: Vec<Box<dyn MlpLayer>>,
        config: AnyMoeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let n_experts = experts.len();
        let var_map = VarMap::new();

        let vb = VarBuilder::from_varmap(&var_map, dtype, dev);
        let vb = vb.pp("moe_gate");

        let lin = linear(config.hidden_size, n_experts, vb)?;

        let vars = var_map.all_vars();
        if vars.is_empty() {
            candle_core::bail!("No vars to train in MoeMlp, perhaps there are no layers?");
        }
        Ok(Self {
            experts,
            gate: MoeGate { lin },
            training: true,
            vars,
            gating_output: Arc::new(RwLock::new(None)),
        })
    }
}

impl AnyMoeTrainableLayer for MoeMlp {
    fn done_training(&mut self) {
        self.training = false;
    }
    fn trainable_params(&self) -> usize {
        self.gate.lin.weight().elem_count() + self.gate.lin.bias().unwrap().elem_count()
    }
    fn get_vars(&self) -> Vec<Var> {
        self.vars.clone()
    }
    fn take_cached_gating_output(&mut self) -> Tensor {
        self.gating_output.read().unwrap().clone().take().unwrap()
    }
}

impl MlpLayer for MoeMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // ^ [b, s, h]
        let gate = self.gate.forward_t(xs, self.training)?;
        // ^ [b, s, n_e]
        // Mean across the sequence dimension
        let mut gate = gate.mean(1)?;
        // ^ [b, n_e]

        *self.gating_output.write().unwrap() = Some(gate.clone());

        // Detach to not track grads for the entire model
        gate = gate.detach();

        let gate_expanded = gate
            .permute((1, 0))?
            .unsqueeze(D::Minus1)?
            .unsqueeze(D::Minus1)?;
        // ^ [b, n_e] -> [n_e, b, 1, 1]
        let mut expert_outputs = Vec::new();
        for expert in &self.experts {
            expert_outputs.push(expert.forward(xs)?);
        }
        let stacked_outputs = Tensor::stack(&expert_outputs, 0)?;
        // ^ [n_e, b, s, h]
        let weighted_outputs = stacked_outputs.broadcast_mul(&gate_expanded)?;
        weighted_outputs.sum(0)
    }

    fn get_isq_tensors(&mut self) -> Vec<&mut QMatMul> {
        if self.training {
            unreachable!("Should not be applying ISQ before training is complete.");
        }

        let mut accum = Vec::new();
        for expert in &mut self.experts {
            accum.extend(expert.get_isq_tensors());
        }
        accum
    }

    fn clone(&self) -> Box<dyn MlpLayer> {
        let mut experts = Vec::new();
        for e in &self.experts {
            experts.push((*e).clone());
        }
        Box::new(Self {
            experts,
            gate: self.gate.clone(),
            training: self.training.clone(),
            vars: self.vars.clone(),
            gating_output: self.gating_output.clone(),
        })
    }
}