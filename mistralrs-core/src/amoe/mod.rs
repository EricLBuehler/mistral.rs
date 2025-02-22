use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, RwLock},
};

use candle_core::{safetensors, DType, Device, Result, Tensor, Var, D};
use candle_nn::{Linear, ModuleT, VarMap};
use mistralrs_quant::{QuantMethod, ShardedSafeTensors, ShardedVarBuilder};
use serde::{Deserialize, Serialize};

mod inputs;
mod macros;
pub use inputs::{AnyMoeTrainingInputRow, AnyMoeTrainingInputs, AnyMoeTrainingResult};
use tracing::info;

use crate::{
    layers::{linear, Activation},
    ops::{TopKLastDimOp, TopKOutput},
    serde_default_fn,
};

/// Implemented by the base model of an AnyMoe.
pub trait AnyMoeBaseModelMixin {
    fn get_vars(&self) -> Vec<Vec<Var>> {
        self.get_mlps()
            .iter()
            .filter(|mlp| mlp.is_moe_layer())
            .map(|mlp| mlp.get_vars())
            .collect::<Vec<_>>()
    }
    fn finish_training(&mut self, gate_model_id: Option<String>) -> Result<()> {
        let mut out = HashMap::new();
        for mlp in self
            .get_mlps_mut()
            .iter_mut()
            .filter(|mlp| mlp.is_moe_layer())
        {
            let out_accum = if gate_model_id.is_some() {
                Some(&mut out)
            } else {
                None
            };
            mlp.finish_training(out_accum);
        }
        if let Some(gate_model_id) = gate_model_id {
            if !Path::new(&gate_model_id).exists() {
                fs::create_dir_all(&gate_model_id)?;
            }
            let save_path = Path::new(&gate_model_id).join("gate.safetensors");
            safetensors::save(&out, &save_path)?;
            info!("Saved gating layers to `{}`", save_path.display());
        }
        Ok(())
    }
    fn trainable_params(&self) -> usize {
        self.get_mlps()
            .iter()
            .filter(|mlp| mlp.is_moe_layer())
            .map(|mlp| mlp.trainable_params())
            .sum()
    }
    fn take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        self.get_mlps_mut()
            .iter_mut()
            .filter(|mlp| mlp.is_moe_layer())
            .map(|mlp| mlp.take_cached_gating_output())
            .collect::<Vec<_>>()
    }

    #[allow(clippy::too_many_arguments)]
    fn create_anymoe_layers(
        &mut self,
        _additional_vbs: Vec<ShardedVarBuilder>,
        _config: AnyMoeConfig,
        (_prefix, _mlp): (String, String),
        _layers: Vec<usize>,
        _expert_type: AnyMoeExpertType,
        _gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        candle_core::bail!("Model does not support AnyMoE layers");
    }
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        panic!("Model does not support AnyMoE layers");
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        panic!("Model does not support AnyMoE layers");
    }
    fn amoe_supported(&self) -> bool {
        false
    }
}

pub trait MlpLayer: Send + Sync + AnyMoeTrainableLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>>;
    fn clone(&self) -> Box<dyn MlpLayer>;
    /// WARNING: The deltas are not a struct but are instead assumed to
    /// be correctly ordered! for that model and it's implementation details
    fn get_params(&self) -> &[usize];
    fn hidden_act(&self) -> Activation;
    fn is_moe_layer(&self) -> bool {
        false
    }
    /// This is for LoRA experts and completes the merging process.
    /// WARNING: The deltas are not a struct but are instead assumed to
    /// be correctly ordered! for that model and it's implementation details
    fn new_added_delta(&self, _deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>>;
    fn dtype_device(&self) -> (DType, Device);
}

pub trait AnyMoeTrainableLayer {
    fn get_vars(&self) -> Vec<Var> {
        vec![]
    }
    fn finish_training(&mut self, _out: Option<&mut HashMap<String, Tensor>>) {}
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
serde_default_fn!(bool, default_true, true);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum AnyMoeExpertType {
    #[serde(rename = "fine_tuned")]
    FineTuned,
    #[serde(rename = "lora_adapter")]
    LoraAdapter {
        rank: usize,
        alpha: f64,
        target_modules: Vec<String>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AnyMoeConfig {
    pub hidden_size: usize,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_bs")]
    pub batch_size: usize,
    pub expert_type: AnyMoeExpertType,
    pub gate_model_id: Option<String>,
    #[serde(default = "default_true")]
    pub training: bool,
    /// If `training == true`, `loss_csv_path` will not save anything.
    /// Otherwise, this will save a .csv loss file here.
    pub loss_csv_path: Option<String>,
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
    layer_idx: usize,
}

impl MoeMlp {
    /// Create a new MoeMlp layer. By default this is in training mode.
    pub fn new(
        experts: Vec<Box<dyn MlpLayer>>,
        config: AnyMoeConfig,
        dtype: DType,
        dev: &Device,
        layer: usize,
        gate_vb: Option<&ShardedVarBuilder>,
    ) -> Result<Self> {
        let n_experts = experts.len();
        let var_map = VarMap::new();

        let inference = gate_vb.is_some();
        let empty_map = ShardedSafeTensors::wrap(Box::new(var_map.clone()), dtype, dev.clone());
        let vb = gate_vb.unwrap_or(&empty_map);
        let vb = vb
            .pp("moe_gate")
            .pp(layer)
            .set_device(dev.clone())
            .set_dtype(dtype);

        let lin = linear(config.hidden_size, n_experts, vb)?;

        let vars = var_map.all_vars();
        if vars.is_empty() && !inference {
            candle_core::bail!("No vars to train in MoeMlp, perhaps there are no layers?");
        }
        Ok(Self {
            experts,
            gate: MoeGate { lin },
            training: true,
            vars,
            gating_output: Arc::new(RwLock::new(None)),
            layer_idx: layer,
        })
    }
}

impl AnyMoeTrainableLayer for MoeMlp {
    fn finish_training(&mut self, out: Option<&mut HashMap<String, Tensor>>) {
        self.training = false;
        let w = self.gate.lin.weight().detach();
        let b = self.gate.lin.bias().map(|b| b.detach());
        self.gate = MoeGate {
            lin: Linear::new(w.clone(), b.clone()),
        };
        if let Some(out) = out {
            out.insert(format!("moe_gate.{}.weight", self.layer_idx), w);
            if let Some(b) = b {
                out.insert(format!("moe_gate.{}.bias", self.layer_idx), b);
            }
        }
    }
    fn trainable_params(&self) -> usize {
        let mut sum = 0;
        if self.gate.lin.weight().is_variable() {
            sum += self.gate.lin.weight().elem_count();
        }
        if self.gate.lin.bias().as_ref().unwrap().is_variable() {
            sum += self.gate.lin.bias().unwrap().elem_count();
        }
        sum
    }
    fn get_vars(&self) -> Vec<Var> {
        self.vars.clone()
    }
    fn take_cached_gating_output(&mut self) -> Tensor {
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

        // Gate with topk 1 to get the highest ranked expert
        let TopKOutput { values: _, indices } = gate.topk(1)?;

        if self.training {
            *self.gating_output.write().unwrap() = Some(gate.clone());
        }

        let mut expert_outputs = Vec::new();
        for expert in &self.experts {
            expert_outputs.push(expert.forward(xs)?);
        }
        let stacked_outputs = Tensor::stack(&expert_outputs, 1)?;
        // ^ [b, n_e s, h]
        let (b, _e, s, h) = stacked_outputs.dims4()?;
        let indices = indices.reshape((b, 1, 1, 1))?.expand((b, 1, s, h))?;
        let gathered_outputs = stacked_outputs
            .contiguous()?
            .gather(&indices.contiguous()?, 1)?;
        gathered_outputs.squeeze(1)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        if self.training {
            unreachable!("Should not be applying ISQ before training is complete.");
        }

        let mut accum = Vec::new();
        for expert in &mut self.experts {
            accum.extend(expert.get_isq_layers());
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
            training: self.training,
            vars: self.vars.clone(),
            gating_output: self.gating_output.clone(),
            layer_idx: self.layer_idx,
        })
    }

    fn get_params(&self) -> &[usize] {
        self.experts[0].get_params()
    }

    fn hidden_act(&self) -> Activation {
        self.experts[0].hidden_act()
    }

    fn is_moe_layer(&self) -> bool {
        true
    }

    fn new_added_delta(&self, _deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        unreachable!()
    }

    fn dtype_device(&self) -> (DType, Device) {
        (
            self.gate.lin.weight().dtype(),
            self.gate.lin.weight().device().clone(),
        )
    }
}
