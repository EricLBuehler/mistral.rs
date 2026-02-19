use std::{collections::HashMap, iter::zip, ops::Mul, sync::Arc};

use candle_core::{quantized::QMatMul, DType, Module, Result, Tensor};
use candle_nn::Linear;
use either::Either;
use mistralrs_quant::{
    GgufMatMul, QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear,
};

use crate::layers::MatMul;

use super::{
    apply_scalings_to_x, get_maybe_topk_scalings, make_adapter, Adapter, LinearLayerLike,
    LoraConfig, LoraLinearConfig, Merge, Ordering,
};

#[derive(Debug)]
pub struct QLoraLinear {
    old: Arc<dyn QuantMethod>,
    a_adapters: Either<Vec<Linear>, (Tensor, Vec<Linear>)>,
    b_adapters: Either<Vec<Linear>, (Tensor, Vec<Linear>)>,
    scale_adapters: Vec<f64>,
    layer_n: usize,
    merged: bool,
    adapters: HashMap<String, Adapter>,
}

/// Specialized QLoRA for no bias
impl QLoraLinear {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        old: QMatMul,
        linear_config: &LoraLinearConfig,
        config: &[((String, String), LoraConfig)],
        vb: &ShardedVarBuilder,
        ordering: &Ordering,
        prefix: String,
        count: &mut usize,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let target_modules = &config.first().map(|c| &c.1.target_modules);
        for (_, cfg) in config {
            if target_modules
                .as_ref()
                .is_some_and(|target_modules| &cfg.target_modules != *target_modules)
            {
                candle_core::bail!("Expected all target modules to be the same.");
            }
        }

        let old: Arc<dyn QuantMethod> = match old {
            QMatMul::QTensor(q) => Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: q,
                b: None,
            })?),
            QMatMul::TensorF16(t) | QMatMul::Tensor(t) => Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(t, None)),
            )?),
        };

        let module = prefix.split('.').next_back().unwrap();
        if target_modules.is_some_and(|target_modules| !target_modules.contains(module)) {
            return Ok(Self {
                old,
                a_adapters: Either::Left(vec![]),
                b_adapters: Either::Left(vec![]),
                scale_adapters: vec![],
                layer_n: usize::MAX,
                merged: false,
                adapters: HashMap::default(),
            });
        }

        *count += 1;

        let mut a_adapters = Vec::with_capacity(config.len());
        let mut b_adapters = Vec::with_capacity(config.len());
        let mut scale_adapters = Vec::with_capacity(config.len());
        let vb = vb.pp(prefix.clone());
        let a_vb = vb.pp("lora_A".to_string());
        let b_vb = vb.pp("lora_B".to_string());
        let mut state = None;
        let mut all_same = true;
        let mut adapters = HashMap::new();
        for ((name_id, adapter_name), cfg) in config.iter() {
            let a_pp = a_vb.pp(name_id);
            let b_pp = b_vb.pp(name_id);
            let adapter = make_adapter(a_pp, b_pp, cfg, linear_config)?;
            a_adapters.push(adapter.a.clone());
            b_adapters.push(adapter.b.clone());
            scale_adapters.push(adapter.scale);
            if state.is_some_and(|x| {
                x == (
                    cfg.rank,
                    linear_config.in_features,
                    linear_config.out_features,
                    cfg.alpha,
                    cfg.dropout,
                )
            }) || state.is_none()
            {
                state = Some((
                    cfg.rank,
                    linear_config.in_features,
                    linear_config.out_features,
                    cfg.alpha,
                    cfg.dropout,
                ));
            } else {
                all_same = false;
            }
            adapters.insert(adapter_name.clone(), adapter);
        }

        if let Some(preload_adapters) = preload_adapters {
            all_same = false;
            for (name, (vb, cfg)) in preload_adapters {
                let a_vb = vb.set_prefix(a_vb.prefix());
                let b_vb = vb.set_prefix(b_vb.prefix());
                let adapter = make_adapter(a_vb, b_vb, cfg, linear_config)?;
                adapters.insert(name.clone(), adapter);
            }
        }

        let layer = if let Some(ref layers) = ordering.layers {
            *layers.get(&prefix).unwrap()
        } else {
            0
        };

        if all_same {
            let a_adapters_stack = Tensor::cat(
                &a_adapters
                    .iter()
                    .map(|x| x.weight().unsqueeze(0))
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            let b_adapters_stack = Tensor::cat(
                &b_adapters
                    .iter()
                    .map(|x| x.weight().unsqueeze(0))
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            #[allow(clippy::cast_possible_truncation)]
            let scale_adapters_t = Tensor::from_vec(
                scale_adapters.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                (scale_adapters.len(), 1, 1),
                a_adapters_stack.device(),
            )?
            .to_dtype(a_adapters_stack.dtype())?;
            let a_adapters_stack = a_adapters_stack.broadcast_mul(&scale_adapters_t)?;
            Ok(QLoraLinear {
                old,
                a_adapters: Either::Right((a_adapters_stack.clone(), a_adapters)),
                b_adapters: Either::Right((b_adapters_stack.clone(), b_adapters)),
                scale_adapters,
                layer_n: layer,
                merged: false,
                adapters,
            })
        } else {
            Ok(QLoraLinear {
                old,
                a_adapters: Either::Left(a_adapters),
                b_adapters: Either::Left(b_adapters),
                scale_adapters,
                layer_n: layer,
                merged: false,
                adapters,
            })
        }
    }
}

impl Merge for QLoraLinear {
    fn get_delta_weight(&self, adapter: usize) -> Result<Tensor> {
        match (&self.a_adapters, &self.b_adapters) {
            (Either::Left(a), Either::Left(b)) | (Either::Right((_, a)), Either::Right((_, b))) => {
                let w_a = a[adapter].weight();
                let w_b = b[adapter].weight();

                MatMul.matmul(w_b, w_a)? * self.scale_adapters[adapter]
            }
            _ => unreachable!("Both adapters must be Either::Left or Either::Right."),
        }
    }

    fn merge_weights(&mut self) -> Result<()> {
        let mut w_base_layer: Option<Tensor> = None;
        for adapter in 0..self.scale_adapters.len() {
            if let Some(w_base_layer) = &mut w_base_layer {
                *w_base_layer = (&*w_base_layer + &self.get_delta_weight(adapter)?)?;
            } else {
                w_base_layer = Some(self.get_delta_weight(adapter)?)
            }
        }
        self.old
            .add_delta_w(w_base_layer.as_ref().expect("Found no adapters to merge."))?;
        self.merged = true;
        Ok(())
    }
}

impl LinearLayerLike for QLoraLinear {
    fn quant_inner(&mut self) -> &mut Arc<dyn QuantMethod> {
        &mut self.old
    }
    fn bias(&self) -> Option<&Tensor> {
        None
    }
    fn weight(&self) -> &Tensor {
        unimplemented!()
    }
    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F32)
    }
    fn lora_forward(
        &self,
        input: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        //No fan_in_fan_out so no weight.transpose(0,1)
        let mut result = self.old.forward(input)?;
        if self.merged {
            return Ok(result);
        }

        if self
            .a_adapters
            .as_ref()
            .left()
            .is_some_and(|x| x.is_empty())
            || (is_scaling_pass.is_some_and(|x| x == 0.))
        {
            return Ok(result);
        }

        let scalings =
            scalings.map(|scalings| get_maybe_topk_scalings(scalings, self.layer_n).unwrap());
        if self.a_adapters.is_left()
            || scalings
                .as_ref()
                .is_some_and(|scalings| scalings.dims3().unwrap().1 != 1)
        {
            let a_adapters = if self.a_adapters.is_right() {
                self.a_adapters.as_ref().unwrap_right().1.clone()
            } else {
                self.a_adapters.as_ref().unwrap_left().clone()
            };
            let b_adapters = if self.b_adapters.is_right() {
                self.b_adapters.as_ref().unwrap_right().1.clone()
            } else {
                self.b_adapters.as_ref().unwrap_left().clone()
            };
            for (i, (adapter_a, (adapter_b, adapter_scale))) in
                zip(a_adapters, zip(b_adapters, &self.scale_adapters)).enumerate()
            {
                let input_new = if let Some(scalings) = &scalings {
                    apply_scalings_to_x(input.clone(), scalings, i)?
                } else {
                    input.clone()
                };

                let res = adapter_b
                    .forward(&adapter_a.forward(&input_new)?)?
                    .mul(*adapter_scale)?
                    .mul(global_scaling_weight)?;
                result = (result + res)?;
            }
            Ok(result)
        } else {
            let adapter_a = &self.a_adapters.as_ref().unwrap_right().0;
            let adapter_b = &self.b_adapters.as_ref().unwrap_right().0;
            let adapter_scales = &self.scale_adapters;
            let n_adapters = adapter_scales.len();
            let adapter_a = if let Some(scalings) = scalings.as_ref() {
                let scalings = scalings
                    .squeeze(0)?
                    .squeeze(0)?
                    .unsqueeze(1)?
                    .unsqueeze(1)?;
                adapter_a
                    .broadcast_mul(&scalings)?
                    .mul(global_scaling_weight)?
            } else {
                adapter_a.clone().mul(global_scaling_weight)?
            };

            let (b, s, h) = input.dims3()?;
            let input = input.reshape((b * s, h))?;
            let out = adapter_a.broadcast_matmul(&input.t()?)?;
            let out = adapter_b.broadcast_matmul(&out)?;
            let o_h = out.dims()[1];
            let out = out.reshape((n_adapters, b, s, o_h))?;
            let out = out.sum(0)?;
            out + result
        }
    }
    fn is_lora(&self) -> bool {
        !self.adapters.is_empty()
    }
}
