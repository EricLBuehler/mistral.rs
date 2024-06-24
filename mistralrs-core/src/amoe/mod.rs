use std::sync::{Arc, RwLock};

use candle_core::{quantized::QMatMul, DType, Device, Result, Tensor, Var, D};
use candle_nn::{linear, Linear, ModuleT, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

pub struct AnyMoeTrainingResult {
    pub steps: usize,
    /// One for each gating layer
    pub final_loss: Vec<f32>,
}

pub struct AnyMoeTrainingInputs(pub Vec<(String, usize)>);

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
    fn get_cached_gating_outputs(&self) -> Vec<Tensor> {
        self.get_mlps()
            .iter()
            .map(|mlp| mlp.get_cached_gating_output())
            .collect::<Vec<_>>()
    }

    fn create_anymoe_layers(
        self,
        additional_vbs: Vec<VarBuilder>,
        config: AnyMoeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self>
    where
        Self: Sized;
    fn get_mlps(&self) -> Vec<&dyn MlpLayer>;
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>>;
}

pub trait MlpLayer: Send + Sync + AnyMoeTrainableLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn get_isq_tensors(&mut self) -> Vec<&mut QMatMul>;
}

pub trait AnyMoeTrainableLayer {
    fn get_vars(&self) -> Vec<Var> {
        vec![]
    }
    fn done_training(&mut self) {}
    fn trainable_params(&self) -> usize {
        0
    }
    fn get_cached_gating_output(&self) -> Tensor {
        panic!("Gating output is not applicable to this layer.")
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct AnyMoeConfig {
    pub hidden_size: usize,
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

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

        let vars = var_map.all_vars();
        if vars.is_empty() {
            candle_core::bail!("No vars to train in MoeMlp, perhaps there are no layer?");
        }
        Ok(Self {
            experts,
            gate: MoeGate {
                lin: linear(config.hidden_size, n_experts, vb)?,
            },
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
    fn get_cached_gating_output(&self) -> Tensor {
        self.gating_output.read().unwrap().clone().unwrap()
    }
}

impl MlpLayer for MoeMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // ^ [b, s, h]
        let gate = self.gate.forward_t(xs, self.training)?;
        // ^ [b, s, n_e]
        // Mean across the sequence dimension
        let gate = gate.mean(1)?;
        // ^ [b, n_e]

        *self.gating_output.write().unwrap() = Some(gate.clone());

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
}
