use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    DummyLayerInfo, IsqPlanParams, IsqRequest, IsqType, QuantMethod, QuantMethodConfig,
    QuantizeOntoGuard, QuantizedSerde, ShardedVarBuilder, UqffTensor,
};

use super::{add_delta, current_lora_execution, LoraLinearSpec, LoraSiteHandle, LoraSiteKey};

#[derive(Debug)]
pub(crate) struct DynamicLoraLinear {
    base: Arc<dyn QuantMethod>,
    site: Arc<LoraSiteHandle>,
    runtime_id: super::LoraRuntimeId,
}

impl DynamicLoraLinear {
    fn new(
        base: Arc<dyn QuantMethod>,
        site: Arc<LoraSiteHandle>,
        runtime_id: super::LoraRuntimeId,
    ) -> Self {
        Self {
            base,
            site,
            runtime_id,
        }
    }

    fn site_is_active(&self) -> bool {
        current_lora_execution(self.runtime_id)
            .is_some_and(|execution| execution.site_is_active(&self.site).unwrap_or(true))
    }
}

pub fn maybe_wrap_dynamic_lora(
    vb: &ShardedVarBuilder,
    base: Arc<dyn QuantMethod>,
    spec: LoraLinearSpec,
) -> Result<Arc<dyn QuantMethod>> {
    maybe_wrap_dynamic_lora_with_key(vb, base, LoraSiteKey::new(vb.prefix()), spec)
}

pub(crate) fn maybe_wrap_dynamic_lora_with_key(
    vb: &ShardedVarBuilder,
    base: Arc<dyn QuantMethod>,
    key: LoraSiteKey,
    spec: LoraLinearSpec,
) -> Result<Arc<dyn QuantMethod>> {
    let Some(registry) = vb.lora_registry() else {
        return Ok(base);
    };
    let site = registry.register(key, spec, vb.dtype(), vb.device().clone())?;
    Ok(Arc::new(DynamicLoraLinear::new(
        base,
        site,
        registry.runtime_id(),
    )))
}

impl QuantizedSerde for DynamicLoraLinear {
    fn name(&self) -> &'static str {
        self.base.name()
    }

    fn isq_serde_supported(&self) -> bool {
        self.base.isq_serde_supported()
    }

    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        self.base.serialize_uqff(prefix, ty)
    }
}

impl QuantMethod for DynamicLoraLinear {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("DynamicLoraLinear requires an existing base linear")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.base.dequantize_w()
    }

    fn forward_raw(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.base.forward(input)?;
        let Some(execution) = current_lora_execution(self.runtime_id) else {
            return Ok(output);
        };
        if !execution.site_is_active(&self.site)? {
            return Ok(output);
        }
        add_delta(&execution, &self.site, input, output)
    }

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        if let Some(execution) = current_lora_execution(self.runtime_id) {
            if execution.site_is_active(&self.site)? {
                candle_core::bail!("dynamic LoRA does not support embedding linears");
            }
        }
        self.base.embedding_forward_raw(ids)
    }

    fn gather_forward_raw(&self, input: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if let Some(execution) = current_lora_execution(self.runtime_id) {
            if execution.site_is_active(&self.site)? {
                candle_core::bail!("dynamic LoRA does not support gather-forward linears");
            }
        }
        self.base.gather_forward_raw(input, indices)
    }

    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        if self.site_is_active() {
            None
        } else {
            self.base.get_qtensor()
        }
    }

    fn afq_inner(&self) -> Option<crate::AfqInner> {
        if self.site_is_active() {
            None
        } else {
            self.base.afq_inner()
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        if self.site_is_active() {
            None
        } else {
            self.base.quantized_act_type()
        }
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        self.base.dtype_and_device()
    }

    fn plan_isq(&self, request: &IsqRequest) -> Result<IsqPlanParams> {
        self.base.plan_isq(request)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let base = self.base.add_delta_w(delta)?;
        Ok(Arc::new(Self::new(
            base,
            self.site.clone(),
            self.runtime_id,
        )))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let base =
            self.base
                .clone()
                .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        Ok(Arc::new(Self::new(
            base,
            self.site.clone(),
            self.runtime_id,
        )))
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        self.base.unquant_weight_bias()
    }

    fn is_dynamic_lora_active(&self) -> bool {
        self.site_is_active()
    }

    fn preserve_dynamic_lora(&self, replacement: Arc<dyn QuantMethod>) -> Arc<dyn QuantMethod> {
        Arc::new(Self::new(replacement, self.site.clone(), self.runtime_id))
    }

    fn has_bias(&self) -> bool {
        self.base.has_bias()
    }

    fn begin_track_stats(&self) -> Result<()> {
        self.base.begin_track_stats()
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.base.end_track_stats()
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.base.stats_snapshot()
    }

    fn process_routed_stats(&self, input: &Tensor, ids: &Tensor) -> Result<()> {
        self.base.process_routed_stats(input, ids)
    }

    fn is_distributed(&self) -> Option<crate::DistributedKind> {
        self.base.is_distributed()
    }

    fn dummy_info(&self) -> Option<DummyLayerInfo> {
        self.base.dummy_info()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use std::collections::HashMap;
    use std::fmt;

    use candle_core::{Device, Tensor};
    use candle_nn::Linear;

    use super::*;
    use crate::{
        with_lora_execution, LoraExecution, LoraLayerRegistry, LoraWeights, UnquantLinear,
    };

    struct ProbeLayer {
        weight: Tensor,
        afq: crate::AfqInner,
        qtensor: Arc<candle_core::quantized::QTensor>,
    }

    impl ProbeLayer {
        fn new() -> Result<Self> {
            let weight = Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?;
            Ok(Self {
                weight: weight.clone(),
                afq: crate::AfqInner {
                    w_q: Tensor::zeros((1, 1), DType::U8, &Device::Cpu)?,
                    scales: Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                    biases: Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                    bias: None,
                    bits: crate::AfqBits::Four,
                    group_size: crate::AfqGroupSize::Low,
                },
                qtensor: Arc::new(candle_core::quantized::QTensor::quantize(
                    &weight,
                    candle_core::quantized::GgmlDType::Q8_0,
                )?),
            })
        }
    }

    impl fmt::Debug for ProbeLayer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("ProbeLayer").finish()
        }
    }

    impl QuantizedSerde for ProbeLayer {
        fn name(&self) -> &'static str {
            "probe"
        }

        fn isq_serde_supported(&self) -> bool {
            false
        }

        fn serialize_uqff(&self, _prefix: &str, _ty: IsqType) -> Result<Vec<UqffTensor>> {
            Ok(Vec::new())
        }
    }

    impl QuantMethod for ProbeLayer {
        fn new(_method: QuantMethodConfig) -> Result<Self> {
            Self::new()
        }

        fn dequantize_w(&self) -> Result<Tensor> {
            Ok(self.weight.clone())
        }

        fn forward_raw(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.clone())
        }

        fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
            Some(self.qtensor.clone())
        }

        fn afq_inner(&self) -> Option<crate::AfqInner> {
            Some(self.afq.clone())
        }

        fn quantized_act_type(&self) -> Option<DType> {
            Some(DType::F16)
        }

        fn dtype_and_device(&self) -> (DType, Device) {
            (self.weight.dtype(), self.weight.device().clone())
        }

        fn plan_isq(&self, _request: &IsqRequest) -> Result<IsqPlanParams> {
            candle_core::bail!("probe layer cannot be quantized")
        }

        fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
            candle_core::bail!("probe layer cannot apply static deltas")
        }

        fn apply_isq(
            self: Arc<Self>,
            _dtype: Option<IsqType>,
            _device: Device,
            _n_quantized: &AtomicUsize,
            _imatrix_weight: Option<Vec<f32>>,
            _guard: QuantizeOntoGuard,
        ) -> Result<Arc<dyn QuantMethod>> {
            Ok(self)
        }

        fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
            Some((self.weight.clone(), None))
        }
    }

    #[test]
    fn wrapper_is_inert_without_an_execution_scope() -> Result<()> {
        let device = Device::Cpu;
        let registry = LoraLayerRegistry::new();
        let site = registry.register(
            LoraSiteKey::new("proj"),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;
        let base: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?, None),
            ))?);
        let layer = DynamicLoraLinear::new(base, site.clone(), registry.runtime_id());
        let input = Tensor::new(&[[1f32, 2.]], &device)?;
        assert_eq!(layer.forward(&input)?.to_vec2::<f32>()?, vec![vec![1., 2.]]);

        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        execution.insert(
            &site,
            0,
            LoraWeights::new(
                Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
                Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
                2.0,
            )?,
        )?;
        let output = with_lora_execution(Some(Arc::new(execution)), || layer.forward(&input))?;
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![3., 6.]]);
        Ok(())
    }

    #[test]
    fn embedding_delegates_unless_its_site_is_active() -> Result<()> {
        let device = Device::Cpu;
        let registry = LoraLayerRegistry::new();
        let embedding_site = registry.register(
            LoraSiteKey::new("embed_tokens"),
            LoraLinearSpec::replicated(2, 3),
            DType::F32,
            device.clone(),
        )?;
        let other_site = registry.register(
            LoraSiteKey::new("other"),
            LoraLinearSpec::replicated(2, 3),
            DType::F32,
            device.clone(),
        )?;
        registry.finalize()?;
        let base: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(
                Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?,
                None,
            )),
        )?);
        let layer = DynamicLoraLinear::new(base, embedding_site.clone(), registry.runtime_id());
        let ids = Tensor::new(&[2u32, 0], &device)?;
        let expected = vec![vec![5f32, 6.], vec![1., 2.]];

        assert_eq!(
            layer.embedding_forward_raw(&ids)?.to_vec2::<f32>()?,
            expected
        );

        let weights = || {
            LoraWeights::new(
                Tensor::zeros((1, 2), DType::F32, &device)?,
                Tensor::zeros((3, 1), DType::F32, &device)?,
                1.0,
            )
        };
        let mut other_execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        other_execution.insert(&other_site, 0, weights()?)?;
        let output = with_lora_execution(Some(Arc::new(other_execution)), || {
            layer.embedding_forward_raw(&ids)
        })?;
        assert_eq!(output.to_vec2::<f32>()?, expected);

        let mut embedding_execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        embedding_execution.insert(&embedding_site, 0, weights()?)?;
        let error = with_lora_execution(Some(Arc::new(embedding_execution)), || {
            layer.embedding_forward_raw(&ids)
        })
        .expect_err("active embedding LoRA should fail");
        assert!(error
            .to_string()
            .contains("dynamic LoRA does not support embedding linears"));
        Ok(())
    }

    #[test]
    fn fast_path_capabilities_are_hidden_only_for_the_targeted_site() -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let targeted = registry.register(
            LoraSiteKey::new("targeted"),
            LoraLinearSpec::replicated(32, 1),
            DType::F32,
            Device::Cpu,
        )?;
        let untargeted = registry.register(
            LoraSiteKey::new("untargeted"),
            LoraLinearSpec::replicated(32, 1),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;
        let targeted_layer = DynamicLoraLinear::new(
            Arc::new(ProbeLayer::new()?),
            targeted.clone(),
            registry.runtime_id(),
        );
        let untargeted_layer = DynamicLoraLinear::new(
            Arc::new(ProbeLayer::new()?),
            untargeted,
            registry.runtime_id(),
        );

        assert_eq!(targeted_layer.quantized_act_type(), Some(DType::F16));
        assert!(targeted_layer.afq_inner().is_some());
        assert!(targeted_layer.unquant_weight_bias().is_some());
        assert!(targeted_layer.get_qtensor().is_some());

        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        execution.insert(
            &targeted,
            0,
            LoraWeights::new(
                Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?,
                Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                1.0,
            )?,
        )?;
        with_lora_execution(Some(Arc::new(execution)), || {
            assert!(targeted_layer.is_dynamic_lora_active());
            assert_eq!(targeted_layer.quantized_act_type(), None);
            assert!(targeted_layer.afq_inner().is_none());
            assert!(targeted_layer.unquant_weight_bias().is_some());
            assert!(!untargeted_layer.is_dynamic_lora_active());
            assert_eq!(untargeted_layer.quantized_act_type(), Some(DType::F16));
            assert!(untargeted_layer.afq_inner().is_some());
            assert!(untargeted_layer.unquant_weight_bias().is_some());
            assert!(targeted_layer.get_qtensor().is_none());
            assert!(untargeted_layer.get_qtensor().is_some());
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn site_placement_comes_from_the_var_builder() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            return Ok(());
        }
        let registry = Arc::new(LoraLayerRegistry::new());
        let vb = crate::ShardedSafeTensors::wrap_with_dummy_regexes(
            HashMap::<String, Tensor>::new(),
            DType::F16,
            device.clone(),
            None,
        )
        .with_lora_registry(registry.clone())
        .pp("proj");
        let base: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?, None),
            ))?);

        maybe_wrap_dynamic_lora(&vb, base, LoraLinearSpec::replicated(2, 2))?;
        let site = registry.sites().pop().expect("registered LoRA site");
        assert_eq!(site.device().location(), device.location());
        assert_eq!(site.activation_dtype(), DType::F16);
        Ok(())
    }
}
