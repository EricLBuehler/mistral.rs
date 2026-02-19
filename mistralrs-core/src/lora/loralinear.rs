use std::{collections::HashMap, iter::zip, ops::Mul, sync::Arc};

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::Linear;
use either::Either;
use mistralrs_quant::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};

use crate::layers::MatMul;

use super::{
    apply_scalings_to_x, get_maybe_topk_scalings, make_adapter, Adapter, LinearLayerLike,
    LoraConfig, LoraLinearConfig, Merge,
};

pub struct LoraLinear {
    old: Arc<dyn QuantMethod>,
    a_adapters: Either<Vec<Linear>, (Tensor, Vec<Linear>)>,
    b_adapters: Either<Vec<Linear>, (Tensor, Vec<Linear>)>,
    scale_adapters: Vec<f64>,
    layer_n: usize,
    merged: bool,
    adapters: HashMap<String, Adapter>,
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        linear_config: &LoraLinearConfig,
        config: &[((String, String), LoraConfig)],
        vb: &ShardedVarBuilder,
        layer_n: usize,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let mut a_adapters = Vec::with_capacity(config.len());
        let mut b_adapters = Vec::with_capacity(config.len());
        let mut scale_adapters = Vec::with_capacity(config.len());
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
            Ok(LoraLinear {
                old: Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                    Linear::new(old.weight().clone(), old.bias().cloned()),
                ))?),
                a_adapters: Either::Right((a_adapters_stack.clone(), a_adapters)),
                b_adapters: Either::Right((b_adapters_stack, b_adapters)),
                scale_adapters,
                layer_n,
                merged: false,
                adapters,
            })
        } else {
            Ok(LoraLinear {
                old: Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                    Linear::new(old.weight().clone(), old.bias().cloned()),
                ))?),
                a_adapters: Either::Left(a_adapters),
                b_adapters: Either::Left(b_adapters),
                scale_adapters,
                layer_n,
                merged: false,
                adapters,
            })
        }
    }
}

impl Merge for LoraLinear {
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

impl LinearLayerLike for LoraLinear {
    fn quant_inner(&mut self) -> &mut Arc<dyn QuantMethod> {
        &mut self.old
    }
    fn bias(&self) -> Option<&Tensor> {
        unreachable!()
    }
    fn weight(&self) -> &Tensor {
        unreachable!()
    }
    fn quantized_act_type(&self) -> Option<DType> {
        self.old.quantized_act_type()
    }
    fn lora_forward(
        &self,
        input: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let mut result = self.old.forward(input)?;

        if self.merged {
            return Ok(result);
        }

        if is_scaling_pass.is_some_and(|x| x == 0.) {
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
            //No fan_in_fan_out so no weight.transpose(0,1)
            for (i, (adapter_a, (adapter_b, adapter_scale))) in
                zip(a_adapters, zip(b_adapters, &self.scale_adapters)).enumerate()
            {
                let input_new = input.to_dtype(adapter_a.weight().dtype())?;
                let input_new = if let Some(scalings) = &scalings {
                    apply_scalings_to_x(input_new, scalings, i)?
                } else {
                    input_new
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
