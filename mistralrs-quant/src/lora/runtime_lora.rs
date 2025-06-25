use std::sync::Arc;

use candle_core::{Context, Result, Tensor};
use regex::Regex;

use crate::{
    get_applied_loras,
    lora::{get_adapter_delta, load_adapter, InstantiatedLoraAdapter},
    AppliedLoraKind, LoraAdapter, QuantMethod, QuantizedSerde, Shard, ShardedVarBuilder,
};

#[derive(Debug)]
pub struct RuntimeLoraLayer {
    base: Arc<dyn QuantMethod>,
    adapters: Vec<InstantiatedLoraAdapter>,
}

impl RuntimeLoraLayer {
    fn apply_lora(&self, mut result: Tensor) -> Result<Tensor> {
        for adapter in &self.adapters {
            let lora_out = result.matmul(&adapter.a.t()?)?.matmul(&adapter.b.t()?)?;
            let scaled_lora = (lora_out * adapter.scale as f64)?;
            result = (result + scaled_lora)?;
        }

        Ok(result)
    }

    fn apply_delta_weight(&self, mut weight: Tensor) -> Result<Tensor> {
        for adapter in &self.adapters {
            weight = get_adapter_delta(adapter)?;
        }

        Ok(weight)
    }
}

impl QuantMethod for RuntimeLoraLayer {
    fn new(_: crate::QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("RuntimeLoraLayer::new is not implemented");
    }

    fn add_delta_w(
        &self,
        delta: &candle_core::Tensor,
    ) -> candle_core::Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self {
            base: self.base.add_delta_w(delta)?,
            adapters: self.adapters.clone(),
        }))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: crate::QuantizeOntoGuard,
    ) -> candle_core::Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self {
            base: self
                .base
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?,
            adapters: self.adapters.clone(),
        }))
    }

    fn begin_track_stats(&mut self) -> candle_core::Result<()> {
        Arc::get_mut(&mut self.base)
            .context("Failed to get &mut to weight")?
            .begin_track_stats()
    }

    fn end_track_stats(&self) -> candle_core::Result<candle_core::Tensor> {
        self.base.end_track_stats()
    }

    fn dequantize_w(&self) -> candle_core::Result<candle_core::Tensor> {
        let mut w = self.base.dequantize_w()?;
        w = self.apply_delta_weight(w)?;
        Ok(w)
    }

    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        self.base.dtype_and_device()
    }

    fn is_distributed(&self) -> Option<crate::DistributedKind> {
        self.base.is_distributed()
    }

    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        self.base.quantized_act_type()
    }

    fn unquant_weight_bias(&self) -> Option<(candle_core::Tensor, Option<candle_core::Tensor>)> {
        match self.base.unquant_weight_bias() {
            Some((mut w, b)) => {
                w = self.apply_delta_weight(w).unwrap();
                Some((w, b))
            }
            None => None,
        }
    }

    fn forward(&self, a: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let mut result = self.base.forward(a)?;
        result = self.apply_lora(result)?;
        Ok(result)
    }

    fn gather_forward(
        &self,
        a: &candle_core::Tensor,
        indices: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        let mut result = self.base.gather_forward(a, indices)?;
        result = self.apply_lora(result)?;
        Ok(result)
    }
}

impl QuantizedSerde for RuntimeLoraLayer {
    fn name(&self) -> &'static str {
        self.base.name()
    }

    fn isq_serde_supported(&self) -> bool {
        true
    }

    fn serialize(&self) -> candle_core::Result<std::borrow::Cow<[u8]>> {
        self.base.serialize()
    }

    fn serialize_with_bias(
        &self,
        bias: Option<candle_core::Tensor>,
    ) -> candle_core::Result<std::borrow::Cow<[u8]>> {
        self.base.serialize_with_bias(bias)
    }

    fn deserialize(
        _data: std::borrow::Cow<[u8]>,
        _device: &candle_core::Device,
        _comm: &Arc<crate::Comm>,
        _guard: crate::QuantizeOntoGuard,
    ) -> candle_core::Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        candle_core::bail!("RuntimeLoraLayer::deserialize is not implemented.")
    }

    fn deserialize_ext_bias(
        _data: std::borrow::Cow<[u8]>,
        _device: &candle_core::Device,
        _guard: crate::QuantizeOntoGuard,
    ) -> candle_core::Result<(Arc<dyn QuantMethod>, Option<candle_core::Tensor>)>
    where
        Self: Sized,
    {
        candle_core::bail!("RuntimeLoraLayer::deserialize_ext_bias is not implemented.")
    }
}

pub fn maybe_wrap_runtime_lora(
    base: Arc<dyn QuantMethod>,
    vb: &ShardedVarBuilder,
    in_dim: usize,
    out_dim: usize,
    shard: Shard,
) -> Result<Arc<dyn QuantMethod>> {
    let Some(applied_loras) = get_applied_loras() else {
        return Ok(base);
    };

    if !matches!(applied_loras.kind, AppliedLoraKind::Runtime) {
        return Ok(base);
    }

    let mut adapters = Vec::new();

    for LoraAdapter { config, weights } in applied_loras.adapters {
        let target_modules = config
            .target_modules
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("|");
        let regex = Regex::new(&target_modules).map_err(candle_core::Error::msg)?;
        if !regex.is_match(&vb.prefix()) {
            continue;
        }

        // Handle base_model.model things from peft
        let weights = if weights
            .pp("base_model.model")
            .pp(vb.prefix())
            .contains_tensor("lora_A.weight")
        {
            weights.pp("base_model.model").pp(vb.prefix())
        } else {
            weights.pp(vb.prefix())
        };

        let adapter = load_adapter(in_dim, out_dim, None, weights, shard, &config)?;

        adapters.push(adapter);
    }

    Ok(Arc::new(RuntimeLoraLayer { base, adapters }))
}
