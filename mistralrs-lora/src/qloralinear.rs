use std::{iter::zip, ops::Mul};

use candle_core::{quantized::QMatMul, DType, Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};

use crate::{
    apply_scalings_to_x, get_maybe_topk_scalings, LinearLayerLike, LoraConfig, LoraLinearConfig,
    Ordering,
};

#[derive(Debug)]
pub struct QLoraLinear {
    old: QMatMul,
    a_adapters: Vec<Linear>,
    b_adapters: Vec<Linear>,
    scale_adapters: Vec<f64>,
    dropout_adapters: Vec<Option<Dropout>>,
    layer_n: usize,
}

impl QLoraLinear {
    pub fn new(
        old: QMatMul,
        linear_config: &LoraLinearConfig,
        config: &[(String, LoraConfig)],
        vb: &VarBuilder,
        ordering: &Ordering,
        prefix: String,
        count: &mut usize,
    ) -> Result<Self> {
        let target_modules = &config[0].1.target_modules;
        for (_, cfg) in config {
            if &cfg.target_modules != target_modules {
                candle_core::bail!("Expected all target modules to be the same.");
            }
        }

        let module = prefix.split('.').last().unwrap();
        if !target_modules.contains(module) {
            return Ok(Self {
                old,
                a_adapters: vec![],
                b_adapters: vec![],
                scale_adapters: vec![],
                dropout_adapters: vec![],
                layer_n: usize::MAX,
            });
        }

        *count += 1;

        let mut a_adapters = Vec::with_capacity(config.len());
        let mut b_adapters = Vec::with_capacity(config.len());
        let mut scale_adapters = Vec::with_capacity(config.len());
        let mut dropout_adapters = Vec::with_capacity(config.len());
        let vb = vb.pp(prefix.clone());
        dbg!(&prefix);
        let a_vb = vb.pp("lora_A".to_string());
        let b_vb = vb.pp("lora_B".to_string());
        for (name, cfg) in config.iter() {
            let a_pp = a_vb.pp(name);
            assert!(a_pp.contains_tensor("weight"));
            let a = a_pp.get_with_hints(
                (cfg.rank, linear_config.in_features),
                "weight",
                init::DEFAULT_KAIMING_NORMAL,
            )?.to_dtype(DType::F32)?;
            let b_pp = b_vb.pp(name);
            assert!(b_pp.contains_tensor("weight"));
            let b =
                b_pp.get_with_hints((linear_config.out_features, cfg.rank), "weight", init::ZERO)?.to_dtype(DType::F32)?;
            a_adapters.push(Linear::new(a, None));
            b_adapters.push(Linear::new(b, None));
            scale_adapters.push(if cfg.rank > 0 {
                cfg.alpha / cfg.rank as f64
            } else {
                1.0
            });
            dropout_adapters.push(cfg.dropout.map(Dropout::new));
        }
        let name = prefix.split("lora_A").last().unwrap();
        let layer = *ordering.layers.get(name).unwrap();

        Ok(QLoraLinear {
            old,
            a_adapters,
            b_adapters,
            scale_adapters,
            dropout_adapters,
            layer_n: layer,
        })
    }
}

impl LinearLayerLike for QLoraLinear {
    fn bias(&self) -> Option<&Tensor> {
        None
    }
    fn weight(&self) -> &Tensor {
        unimplemented!()
    }
    fn shape(&self) -> &Shape {
        unimplemented!()
    }
    fn lora_forward(
        &self,
        input: &Tensor,
        scalings: Tensor,
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        //No fan_in_fan_out so no weight.transpose(0,1)
        let mut result = self.old.forward(input)?;
        if self.a_adapters.is_empty() {
            return Ok(result);
        }
        let scalings = get_maybe_topk_scalings(scalings, self.layer_n)?;

        for (i, (adapter_a, (adapter_b, (adapter_scale, adapter_dropout)))) in zip(
            &self.a_adapters,
            zip(
                &self.b_adapters,
                zip(&self.scale_adapters, &self.dropout_adapters),
            ),
        )
        .enumerate()
        {
            let mut input_new = apply_scalings_to_x(input.clone(), &scalings, i)?;

            input_new = if let Some(ref dropout) = adapter_dropout {
                dropout.forward(&input_new, true)?
            } else {
                input_new.clone()
            };
            let res = adapter_b
                .forward(&adapter_a.forward(&input_new)?)?
                .mul(*adapter_scale)?
                .mul(global_scaling_weight)?;
            result = (result + res)?;
        }
        Ok(result)
    }
}
