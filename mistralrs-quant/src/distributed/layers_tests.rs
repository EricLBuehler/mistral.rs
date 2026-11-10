use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use regex::Regex;

use super::{
    distributed, validate_tp_head_layout, ColumnParallelLayer, PackedOutputLayout, ReplicatedLayer,
    RowParallelLayer,
};
use crate::{
    create_isq_executor, set_immediate_isq_config, ActivationQuantizationScheme, Comm,
    Fp8ActivationScheme, Id, ImmediateIsqConfig, ImmediateIsqOverride, IsqCaptureMode, IsqConsumer,
    IsqExecutorConfig, IsqPlanParams, IsqRequest, IsqType, LoraLayerRegistry, LoraLinearSpec,
    QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedActivation, QuantizedConfig,
    QuantizedSerde, QuantizedWeightSource, Shard, ShardedSafeTensors, UnquantLinear,
};

#[test]
fn modelopt_nvfp4_experts_preserve_scales_and_excluded_projections() -> candle_core::Result<()> {
    const HIDDEN: usize = 16;
    const INTERMEDIATE: usize = 16;
    const EXPERTS: usize = 2;

    let device = Device::Cpu;
    let config = serde_json::from_value::<QuantizedConfig>(serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "exclude_modules": [
            "model.layers.0.mlp.experts.*.up_proj",
            "model.layers.0.mlp.shared_experts"
        ]
    }))
    .map_err(candle_core::Error::msg)?;
    let mut tensors = HashMap::new();
    for (expert, global_scale) in [2f32, 4.].into_iter().enumerate() {
        for projection in ["gate_proj", "down_proj"] {
            let prefix = format!("model.layers.0.mlp.experts.{expert}.{projection}");
            tensors.insert(
                format!("{prefix}.weight"),
                Tensor::full(0x22u8, (INTERMEDIATE, HIDDEN / 2), &device)?,
            );
            tensors.insert(
                format!("{prefix}.weight_scale"),
                Tensor::ones((INTERMEDIATE, 1), DType::F8E4M3, &device)?,
            );
            tensors.insert(
                format!("{prefix}.weight_scale_2"),
                Tensor::new(global_scale, &device)?,
            );
        }
        tensors.insert(
            format!("model.layers.0.mlp.experts.{expert}.up_proj.weight"),
            Tensor::full(global_scale + 1., (INTERMEDIATE, HIDDEN), &device)?,
        );
    }
    tensors.insert(
        "model.layers.0.mlp.shared_experts.gate_proj.weight".to_string(),
        Tensor::full(7f32, (INTERMEDIATE, HIDDEN), &device)?,
    );
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device).pp("model.layers.0.mlp");
    let shared = crate::fp8_config::checkpoint_linear_b(
        HIDDEN,
        INTERMEDIATE,
        &config,
        false,
        Shard::default(),
        vb.pp("shared_experts.gate_proj"),
    )?;
    let experts =
        super::PreQuantizedExperts::new(HIDDEN, INTERMEDIATE, EXPERTS, &Some(config), vb)?;
    for projection in [&experts.fused_gate_proj, &experts.fused_down_proj] {
        let values = projection
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        for (values, expected) in values.iter().zip([2f32, 4.]) {
            assert!(values.iter().flatten().all(|value| *value == expected));
        }
    }
    let up = experts.fused_up_proj.dequantize_w()?.to_vec3::<f32>()?;
    for (values, expected) in up.iter().zip([3f32, 5.]) {
        assert!(values.iter().flatten().all(|value| *value == expected));
    }
    assert!(shared
        .dequantize_w()?
        .to_vec2::<f32>()?
        .iter()
        .flatten()
        .all(|value| *value == 7.));
    Ok(())
}

#[test]
fn compressed_tensors_nvfp4_experts_use_reciprocal_global_scales() -> candle_core::Result<()> {
    const HIDDEN: usize = 16;
    const EXPERTS: usize = 2;

    let device = Device::Cpu;
    let config = serde_json::from_value::<QuantizedConfig>(serde_json::json!({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "dynamic": false,
                    "symmetric": true
                },
                "input_activations": null
            }
        }
    }))
    .map_err(candle_core::Error::msg)?;
    let mut tensors = HashMap::new();
    for (expert, global_scale) in [2f32, 4.].into_iter().enumerate() {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let prefix = format!("model.layers.0.mlp.experts.{expert}.{projection}");
            tensors.insert(
                format!("{prefix}.weight_packed"),
                Tensor::full(0x22u8, (HIDDEN, HIDDEN / 2), &device)?,
            );
            tensors.insert(
                format!("{prefix}.weight_scale"),
                Tensor::ones((HIDDEN, 1), DType::F8E4M3, &device)?,
            );
            tensors.insert(
                format!("{prefix}.weight_global_scale"),
                Tensor::new(&[global_scale], &device)?,
            );
        }
    }
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device).pp("model.layers.0.mlp");
    let experts = super::PreQuantizedExperts::new(HIDDEN, HIDDEN, EXPERTS, &Some(config), vb)?;
    for projection in [
        &experts.fused_gate_proj,
        &experts.fused_up_proj,
        &experts.fused_down_proj,
    ] {
        let values = projection
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        for (values, expected) in values.iter().zip([0.5f32, 0.25]) {
            assert!(values.iter().flatten().all(|value| *value == expected));
        }
    }
    Ok(())
}

#[test]
fn nvfp4_tensor_parallel_shards_preserve_fused_global_scale_groups() -> candle_core::Result<()> {
    const INPUT: usize = 32;
    const OUTPUT: usize = 4;
    const WORLD_SIZE: usize = 2;

    let device = Device::Cpu;
    let config = Some(
        serde_json::from_value::<QuantizedConfig>(serde_json::json!({
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "tensor_group",
                        "group_size": 16,
                        "dynamic": false,
                        "symmetric": true
                    },
                    "input_activations": null
                }
            }
        }))
        .map_err(candle_core::Error::msg)?,
    );
    let tensors = HashMap::from([
        (
            "gate_up_proj.weight_packed".to_string(),
            Tensor::full(0x22u8, (OUTPUT, INPUT / 2), &device)?,
        ),
        (
            "gate_up_proj.weight_scale".to_string(),
            Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.], [7., 8.]], &device)?
                .to_dtype(DType::F8E4M3)?,
        ),
        (
            "gate_up_proj.weight_global_scale".to_string(),
            Tensor::new(&[2f32, 4.], &device)?,
        ),
    ]);
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device.clone()).pp("gate_up_proj");
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 1, WORLD_SIZE)?);
    let column = ColumnParallelLayer::new_with_shard(
        INPUT,
        OUTPUT,
        &config,
        false,
        &comm,
        Shard::Simple {
            dim: 0,
            rank: 1,
            world_size: WORLD_SIZE,
        },
        vb.clone(),
    )?;
    let column = column
        .dequantize_w()?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    assert_eq!(column.len(), OUTPUT / WORLD_SIZE);
    for (values, expected) in column.iter().zip([[1.25f32, 1.5], [1.75, 2.]]) {
        for (block, expected) in values
            .as_chunks::<{ INPUT / WORLD_SIZE }>()
            .0
            .iter()
            .zip(expected)
        {
            assert!(block.iter().all(|value| *value == expected));
        }
    }
    // CPU communicators always report one rank, so pass the input shard directly.
    let row = crate::fp8_config::checkpoint_linear_b(
        INPUT,
        OUTPUT,
        config.as_ref().unwrap(),
        false,
        Shard::Simple {
            dim: 1,
            rank: 1,
            world_size: WORLD_SIZE,
        },
        vb,
    )?;
    let row = row.dequantize_w()?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    assert_eq!(row.len(), OUTPUT);
    for (values, expected) in row.iter().zip([1f32, 2., 1.5, 2.]) {
        assert_eq!(values.len(), INPUT / WORLD_SIZE);
        assert!(values.iter().all(|value| *value == expected));
    }
    Ok(())
}

const FP8_MOE_HIDDEN: usize = 128;
const FP8_MOE_INTERMEDIATE: usize = 64;
const FP8_MOE_EXPERTS: usize = 3;
const FP8_MOE_TOKENS: usize = 4;
const FP8_MOE_TOPK: usize = 2;
const FP8_MOE_IDS: [u32; FP8_MOE_TOKENS * FP8_MOE_TOPK] = [2, 0, 1, 2, 0, 1, 2, 1];

enum Fp8ScaleFixture {
    Tensor,
    Channel,
}

fn fp8_expert_values(expert: usize, projection: usize, out: usize, inp: usize) -> Vec<f32> {
    (0..out * inp)
        .map(|i| (((i * 7 + expert * 13 + projection * 29) % 41) as f32 - 20.0) * 0.5)
        .collect()
}

fn per_tensor_fp8_experts_load_and_gather(
    config: serde_json::Value,
    scales: Fp8ScaleFixture,
    device: &Device,
) -> candle_core::Result<()> {
    let config =
        serde_json::from_value::<QuantizedConfig>(config).map_err(candle_core::Error::msg)?;
    let cpu = Device::Cpu;
    let mut tensors = HashMap::new();
    let mut dense = HashMap::new();
    for expert in 0..FP8_MOE_EXPERTS {
        for (projection, (name, out, inp)) in [
            ("gate_proj", FP8_MOE_INTERMEDIATE, FP8_MOE_HIDDEN),
            ("up_proj", FP8_MOE_INTERMEDIATE, FP8_MOE_HIDDEN),
            ("down_proj", FP8_MOE_HIDDEN, FP8_MOE_INTERMEDIATE),
        ]
        .into_iter()
        .enumerate()
        {
            let prefix = format!("model.layers.0.mlp.experts.{expert}.{name}");
            let weight = Tensor::from_vec(
                fp8_expert_values(expert, projection, out, inp),
                (out, inp),
                &cpu,
            )?
            .to_dtype(DType::F8E4M3)?;
            let scale = match scales {
                Fp8ScaleFixture::Tensor => {
                    Tensor::new(0.01f32 * (expert + projection + 1) as f32, &cpu)?
                }
                Fp8ScaleFixture::Channel => Tensor::from_vec(
                    (0..out)
                        .map(|row| 0.001 * (row + expert + projection + 1) as f32)
                        .collect::<Vec<_>>(),
                    (out, 1),
                    &cpu,
                )?,
            };
            let row_scale = match scales {
                Fp8ScaleFixture::Tensor => scale.reshape((1, 1))?,
                Fp8ScaleFixture::Channel => scale.clone(),
            };
            dense.insert(
                (expert, name),
                weight.to_dtype(DType::F32)?.broadcast_mul(&row_scale)?,
            );
            tensors.insert(format!("{prefix}.weight"), weight);
            tensors.insert(format!("{prefix}.weight_scale"), scale);
            tensors.insert(format!("{prefix}.input_scale"), Tensor::new(0.05f32, &cpu)?);
        }
    }
    let dir = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
    let path = dir.path().join("model.safetensors");
    candle_core::safetensors::save(&tensors, &path)?;
    let vb = unsafe {
        ShardedSafeTensors::sharded(&[&path], DType::BF16, device, None, Arc::new(|_| true))?
    }
    .pp("model.layers.0.mlp");
    let experts = super::PreQuantizedExperts::new(
        FP8_MOE_HIDDEN,
        FP8_MOE_INTERMEDIATE,
        FP8_MOE_EXPERTS,
        &Some(config),
        vb,
    )?;

    let ids = Tensor::from_vec(FP8_MOE_IDS.to_vec(), (FP8_MOE_TOKENS, FP8_MOE_TOPK), device)?;
    let x = Tensor::from_vec(
        (0..FP8_MOE_TOKENS * FP8_MOE_HIDDEN)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect::<Vec<_>>(),
        (FP8_MOE_TOKENS, FP8_MOE_HIDDEN),
        &cpu,
    )?;
    for (name, layer) in [
        ("gate_proj", &experts.fused_gate_proj),
        ("up_proj", &experts.fused_up_proj),
    ] {
        let got = layer
            .gather_forward(
                &x.reshape((FP8_MOE_TOKENS, 1, FP8_MOE_HIDDEN))?
                    .to_dtype(DType::BF16)?
                    .to_device(device)?,
                &ids,
            )?
            .to_device(&cpu)?
            .to_dtype(DType::F32)?;
        let mut rows = Vec::new();
        for (route, &expert) in FP8_MOE_IDS.iter().enumerate() {
            rows.push(
                x.narrow(0, route / FP8_MOE_TOPK, 1)?
                    .matmul(&dense[&(expert as usize, name)].t()?)?,
            );
        }
        let expected =
            Tensor::cat(&rows, 0)?.reshape((FP8_MOE_TOKENS, FP8_MOE_TOPK, FP8_MOE_INTERMEDIATE))?;
        let err = (got - &expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = expected.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err <= 0.02 * scale, "{name}: max err {err} vs {scale}");
    }
    let down = experts
        .fused_down_proj
        .dequantize_w()?
        .to_device(&cpu)?
        .to_dtype(DType::F32)?;
    for expert in 0..FP8_MOE_EXPERTS {
        let err = (down.get(expert)? - &dense[&(expert, "down_proj")])?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(err <= 0.05, "down_proj expert {expert}: max err {err}");
    }
    Ok(())
}

fn fp8_moe_fixture_configs() -> Vec<(serde_json::Value, Fp8ScaleFixture)> {
    vec![
        (
            serde_json::json!({"quant_method": "modelopt", "quant_algo": "FP8"}),
            Fp8ScaleFixture::Tensor,
        ),
        (
            serde_json::json!({
                "quant_method": "modelopt",
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
            }),
            Fp8ScaleFixture::Channel,
        ),
        (
            serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "config_groups": {"group_0": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": false, "num_bits": 8, "strategy": "tensor",
                        "symmetric": true, "type": "float"},
                    "input_activations": {"dynamic": false, "num_bits": 8,
                        "strategy": "tensor", "symmetric": true, "type": "float"}
                }}
            }),
            Fp8ScaleFixture::Tensor,
        ),
        (
            serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "config_groups": {"group_0": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": false, "num_bits": 8, "strategy": "channel",
                        "symmetric": true, "type": "float"},
                    "input_activations": {"dynamic": true, "num_bits": 8,
                        "strategy": "token", "symmetric": true, "type": "float"}
                }}
            }),
            Fp8ScaleFixture::Channel,
        ),
    ]
}

#[test]
fn per_tensor_and_channel_fp8_experts_load_cpu() -> candle_core::Result<()> {
    for (config, scales) in fp8_moe_fixture_configs() {
        per_tensor_fp8_experts_load_and_gather(config, scales, &Device::Cpu)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn per_tensor_and_channel_fp8_experts_load_cuda() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    for (config, scales) in fp8_moe_fixture_configs() {
        per_tensor_fp8_experts_load_and_gather(config, scales, &device)?;
    }
    Ok(())
}

#[test]
fn checkpoint_experts_reject_mixed_precision_within_projection() -> candle_core::Result<()> {
    let config = serde_json::from_value::<QuantizedConfig>(serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "exclude_modules": ["model.layers.0.mlp.experts.1.gate_proj"]
    }))
    .map_err(candle_core::Error::msg)?;
    let vb =
        ShardedSafeTensors::wrap(HashMap::new(), DType::F32, Device::Cpu).pp("model.layers.0.mlp");
    let error = super::PreQuantizedExperts::new(16, 16, 2, &Some(config), vb)
        .unwrap_err()
        .to_string();
    assert!(error.contains("require the same quantization format and activation scheme"));
    Ok(())
}

#[derive(Debug)]
struct SharedActivationWeight {
    output: Tensor,
    active_lora: bool,
    tracking_stats: bool,
}

impl QuantizedSerde for SharedActivationWeight {
    fn name(&self) -> &'static str {
        "shared-activation-weight"
    }
}

impl QuantMethod for SharedActivationWeight {
    fn new(_method: QuantMethodConfig) -> candle_core::Result<Self> {
        candle_core::bail!("test weight cannot be constructed from a quantization config")
    }

    fn dequantize_w(&self) -> candle_core::Result<Tensor> {
        Ok(self.output.clone())
    }

    fn forward_raw(&self, _a: &Tensor) -> candle_core::Result<Tensor> {
        Ok(self.output.clone())
    }

    fn activation_quantization_scheme(&self) -> Option<ActivationQuantizationScheme> {
        Some(ActivationQuantizationScheme {
            dtype: DType::F8E4M3,
            scale_encoding: crate::ActivationScaleEncoding::F32,
            block_shape: [1, 4],
        })
    }

    fn quantize_activation(&self, a: &Tensor) -> candle_core::Result<QuantizedActivation> {
        let scheme = self.activation_quantization_scheme().unwrap();
        QuantizedActivation::new(
            Tensor::zeros(a.dims(), scheme.dtype, a.device())?,
            Tensor::ones((a.elem_count() / 4, 1), DType::F32, a.device())?,
            a.dims().to_vec(),
            a.dtype(),
            scheme,
        )
    }

    fn forward_glu(
        &self,
        input: crate::GluInput<'_>,
        activation: crate::GluActivationType,
    ) -> candle_core::Result<Option<Tensor>> {
        assert!(
            !self.active_lora && !self.tracking_stats,
            "guarded projection reached fused quantization"
        );
        let Some((gate, value)) = input.halves()? else {
            return Ok(None);
        };
        if gate.shape() != value.shape() || !matches!(activation, crate::GluActivationType::Relu) {
            return Ok(None);
        }
        self.forward_quantized(&self.quantize_activation(&gate)?)
            .map(Some)
    }

    fn forward_quantized(&self, _a: &QuantizedActivation) -> candle_core::Result<Tensor> {
        Ok(self.output.clone())
    }

    fn is_dynamic_lora_active(&self) -> bool {
        self.active_lora
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.tracking_stats.then_some((1, 1))
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.output.dtype(), self.output.device().clone())
    }

    fn plan_isq(&self, _request: &IsqRequest) -> candle_core::Result<IsqPlanParams> {
        candle_core::bail!("test weight cannot be quantized")
    }

    fn add_delta_w(&self, _delta: &Tensor) -> candle_core::Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("test weight cannot apply deltas")
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> candle_core::Result<Arc<dyn QuantMethod>> {
        Ok(self)
    }
}

struct DenseWeightSource;

impl QuantizedWeightSource for DenseWeightSource {
    fn contains(&self, name: &str) -> bool {
        name == "model.embed_tokens.weight"
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        _shard: Shard,
    ) -> candle_core::Result<Option<std::sync::Arc<dyn QuantMethod>>> {
        if key != "model.embed_tokens" {
            return Ok(None);
        }
        let weight = Tensor::zeros((64, 32), DType::F32, device)?;
        Ok(Some(std::sync::Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(candle_nn::Linear::new(weight, None)),
        )?)))
    }

    fn load_optional_tensor(
        &self,
        _name: &str,
        _device: &Device,
    ) -> candle_core::Result<Option<Tensor>> {
        Ok(None)
    }

    fn shard_alignment(&self, _key: &str) -> candle_core::Result<usize> {
        Ok(1)
    }

    fn pack_factor(&self, _dtype: DType) -> candle_core::Result<usize> {
        Ok(1)
    }

    fn pack_factor_for(&self, _key: &str, _dtype: DType) -> candle_core::Result<Option<usize>> {
        Ok(Some(1))
    }
}

struct MatformerWeightSource;

impl QuantizedWeightSource for MatformerWeightSource {
    fn contains(&self, name: &str) -> bool {
        matches!(name, "row.weight" | "column.weight" | "replicated.weight")
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        _shard: Shard,
    ) -> candle_core::Result<Option<std::sync::Arc<dyn QuantMethod>>> {
        let shape = match key {
            "row" => (5, 8),
            "column" => (8, 4),
            "replicated" => (6, 4),
            _ => return Ok(None),
        };
        let weight = Tensor::zeros(shape, DType::F32, device)?;
        Ok(Some(std::sync::Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(candle_nn::Linear::new(weight, None)),
        )?)))
    }

    fn load_optional_tensor(
        &self,
        name: &str,
        device: &Device,
    ) -> candle_core::Result<Option<Tensor>> {
        let size = match name {
            "row.bias" => 5,
            "column.bias" => 8,
            "replicated.bias" => 6,
            _ => return Ok(None),
        };
        Tensor::zeros(size, DType::F32, device).map(Some)
    }

    fn shard_alignment(&self, _key: &str) -> candle_core::Result<usize> {
        Ok(1)
    }

    fn pack_factor(&self, _dtype: DType) -> candle_core::Result<usize> {
        Ok(1)
    }

    fn pack_factor_for(&self, _key: &str, _dtype: DType) -> candle_core::Result<Option<usize>> {
        Ok(Some(1))
    }
}

fn install_immediate(ty: IsqType, overrides: Vec<ImmediateIsqOverride>) {
    let ty = Some(ty);
    let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
    let promoted = Regex::new(r"^model\.embed_tokens\.weight$").unwrap();
    set_immediate_isq_config(
        ImmediateIsqConfig::new(ty, vec![promoted.clone()], IsqCaptureMode::Immediate)
            .with_promoted_predicates(vec![promoted])
            .with_overrides(overrides),
        executor,
    );
}

fn tracked_from_linear() -> crate::TrackedModule {
    let vb = ShardedSafeTensors::wrap(
        HashMap::<String, candle_core::Tensor>::new(),
        candle_core::DType::F32,
        candle_core::Device::Cpu,
    )
    .pp("model")
    .pp("embed_tokens");
    let tracker = vb.tracker().clone();
    let linear = candle_nn::Linear::new(
        candle_core::Tensor::zeros(
            (64, 256),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap(),
        None,
    );
    let _ = ReplicatedLayer::from_linear(linear, vb).unwrap();
    let tracked = tracker.get()[0].clone();
    crate::clear_immediate_isq();
    tracked
}

#[test]
fn distributed_raw_weight_hooks_require_transparent_wrappers() -> candle_core::Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};

    let device = Device::Cpu;
    let tensor = Tensor::ones((32, 32), DType::F32, &device)?;
    let quantized = Arc::new(crate::GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(QTensor::quantize(&tensor, GgmlDType::Q4_0)?),
        b: None,
    })?) as Arc<dyn QuantMethod>;
    let dense = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
        candle_nn::Linear::new(tensor, None),
    ))?) as Arc<dyn QuantMethod>;
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    for weight in [quantized, dense] {
        for biased in [false, true] {
            let bias = biased
                .then(|| Tensor::ones(32, DType::F32, &device))
                .transpose()?;
            let row = RowParallelLayer {
                weight: weight.clone(),
                bias: bias.clone(),
                all_reduce: distributed::SumAllReduce::new(&comm),
            };
            let column = ColumnParallelLayer {
                weight: weight.clone(),
                bias,
            };
            for layer in [&row as &dyn QuantMethod, &column] {
                assert_eq!(
                    layer.get_qtensor().is_some(),
                    !biased && weight.get_qtensor().is_some()
                );
                assert_eq!(
                    layer.unquant_forward_weight().is_some(),
                    !biased && weight.unquant_forward_weight().is_some()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn distributed_wrappers_preserve_shared_activation_forwarding() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let output = Tensor::from_vec(vec![1f32, 2.], (1, 2), &device)?;
    let weight = Arc::new(SharedActivationWeight {
        output,
        active_lora: false,
        tracking_stats: false,
    }) as Arc<dyn QuantMethod>;
    let bias = Tensor::from_vec(vec![3f32, 4.], (2,), &device)?;
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    let activation_input = Tensor::zeros((1, 4), DType::BF16, &device)?;

    let row = RowParallelLayer {
        weight: weight.clone(),
        bias: Some(bias.clone()),
        all_reduce: distributed::SumAllReduce::new(&comm),
    };
    let column = ColumnParallelLayer {
        weight: weight.clone(),
        bias: Some(bias),
    };
    let replicated = ReplicatedLayer(weight);

    let expected_scheme = ActivationQuantizationScheme {
        dtype: DType::F8E4M3,
        scale_encoding: crate::ActivationScaleEncoding::F32,
        block_shape: [1, 4],
    };
    for layer in [&row as &dyn QuantMethod, &column, &replicated] {
        assert_eq!(
            layer.activation_quantization_scheme(),
            Some(expected_scheme)
        );
    }

    let glu = crate::GluInput::Split {
        gate: &activation_input,
        value: &activation_input,
    };
    for layer in [&row as &dyn QuantMethod, &column, &replicated] {
        let output = layer
            .forward_glu(glu, crate::GluActivationType::Relu)?
            .expect("wrapper must preserve optional fused GLU capability");
        assert_eq!(
            output.to_vec2::<f32>()?,
            layer.forward(&activation_input)?.to_vec2::<f32>()?
        );
        assert!(layer
            .forward_glu(glu, crate::GluActivationType::Gelu)?
            .is_none());
    }

    let activation = row.quantize_activation(&activation_input)?;
    assert_eq!(
        row.forward_quantized(&activation)?.to_vec2::<f32>()?,
        vec![vec![4., 6.]]
    );
    assert_eq!(
        column.forward_quantized(&activation)?.to_vec2::<f32>()?,
        vec![vec![4., 6.]]
    );
    assert_eq!(
        replicated
            .forward_quantized(&activation)?
            .to_vec2::<f32>()?,
        vec![vec![1., 2.]]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn fused_glu_wrapper_guards_preserve_lora_and_stats_inputs() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let input = Tensor::zeros((2, 4), DType::BF16, &device)?;
    let output = Tensor::zeros((2, 2), DType::BF16, &device)?;
    for (active_lora, tracking_stats) in [(true, false), (false, true)] {
        let weight = Arc::new(SharedActivationWeight {
            output: output.clone(),
            active_lora,
            tracking_stats,
        }) as Arc<dyn QuantMethod>;
        let column = ColumnParallelLayer { weight, bias: None };
        let glu = crate::GluInput::Split {
            gate: &input,
            value: &input,
        };
        assert!(crate::forward_fused_glu(&column, glu, crate::GluActivationType::Relu)?.is_none());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn packed_checkpoint_aligned_dense_exclusions_keep_logical_slices() -> candle_core::Result<()> {
    const COLUMNS: usize = 64;
    const ROWS: [usize; 2] = [32, 3];
    const LOGICAL_ROWS: usize = 35;
    const PADDED_ROWS: usize = 48;
    let device = Device::new_cuda(0)?;
    let names = ["down", "inject"];
    let specs = ROWS.map(|rows| LoraLinearSpec::replicated(COLUMNS, rows));
    let config = Some(
        serde_json::from_value::<QuantizedConfig>(serde_json::json!({
            "quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
            "exclude_modules": ["model.down", "model.inject"]
        }))
        .map_err(candle_core::Error::msg)?,
    );
    for dtype in [DType::F16, DType::BF16] {
        let tensors = HashMap::from([
            (
                "model.down.weight".to_string(),
                Tensor::full(0.25f32, (ROWS[0], COLUMNS), &device)?.to_dtype(dtype)?,
            ),
            (
                "model.inject.weight".to_string(),
                Tensor::full(0.5f32, (ROWS[1], COLUMNS), &device)?.to_dtype(dtype)?,
            ),
        ]);
        let builder = ShardedSafeTensors::wrap(tensors, dtype, device.clone()).pp("model");
        let group = ReplicatedLayer::new_packed(&specs, &names, &config, false, builder)?.unwrap();
        assert_eq!(group.rows_per_rank, ROWS);
        assert_eq!(
            group.packed.dequantize_w()?.dims(),
            &[LOGICAL_ROWS, COLUMNS]
        );
        for tokens in [1, 17] {
            let input = Tensor::ones((2, tokens, COLUMNS), dtype, &device)?;
            let output = group.packed.forward(&input)?;
            assert_eq!(output.dims(), &[2, tokens, LOGICAL_ROWS]);
            let mut offset = 0;
            for (projection, &rows) in group.constituents.iter().zip(&group.rows_per_rank) {
                let view = output.narrow(2, offset, rows)?;
                assert_eq!(view.stride(), &[tokens * PADDED_ROWS, PADDED_ROWS, 1]);
                assert_eq!(
                    view.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    projection
                        .forward(&input)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
                offset += rows;
            }
        }
    }
    Ok(())
}

#[test]
fn packed_checkpoint_mixed_and_incompatible_groups_keep_fallback() -> candle_core::Result<()> {
    const COLUMNS: usize = 32;
    const ROWS: usize = 4;
    let device = Device::Cpu;
    let specs = [
        LoraLinearSpec::replicated(COLUMNS, ROWS),
        LoraLinearSpec::replicated(COLUMNS, ROWS),
    ];
    let mut tensors = HashMap::new();
    for name in ["left", "right"] {
        tensors.insert(
            format!("model.block.{name}.weight"),
            Tensor::ones((ROWS, COLUMNS), DType::F32, &device)?,
        );
        tensors.insert(
            format!("model.block.{name}.weight_scale"),
            Tensor::ones((ROWS, 2), DType::U8, &device)?,
        );
    }
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device).pp("model.block");
    for value in [
        serde_json::json!({
            "quant_method": "modelopt", "quant_algo": "MXFP8", "group_size": 32,
            "exclude_modules": ["model.block.left"]
        }),
        serde_json::json!({
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.block.left": {"quant_algo": "NVFP4", "group_size": 16},
                "model.block.right": {"quant_algo": "MXFP8", "group_size": 32}
            }
        }),
        serde_json::json!({
            "quant_method": "compressed-tensors", "format": "float-quantized",
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"dynamic": false, "num_bits": 8, "strategy": "channel",
                            "symmetric": true, "type": "float"}
            }},
            "ignore": ["model.block.left"]
        }),
    ] {
        let config = Some(
            serde_json::from_value::<QuantizedConfig>(value).map_err(candle_core::Error::msg)?,
        );
        for names in [["left", "right"], ["right", "left"]] {
            assert!(
                ReplicatedLayer::new_packed(&specs, &names, &config, false, vb.clone())?.is_none()
            );
        }
    }
    Ok(())
}

#[test]
fn packed_column_shards_each_projection_before_concatenating() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let matrix = |rows: usize, base: f32| {
        Tensor::from_vec(
            (0..rows)
                .flat_map(|row| [base + row as f32, base + row as f32 + 0.5])
                .collect::<Vec<_>>(),
            (rows, 2),
            &device,
        )
    };
    let tensors = HashMap::from([
        ("q.weight".to_string(), matrix(8, 0.)?),
        ("k.weight".to_string(), matrix(4, 100.)?),
        ("v.weight".to_string(), matrix(4, 200.)?),
    ]);
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device.clone());
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 1, 2)?);
    let shards = [
        Shard::Simple {
            dim: 0,
            rank: 1,
            world_size: 2,
        },
        Shard::Offset {
            dim: 0,
            offset: 0,
            len: 2,
        },
        Shard::Offset {
            dim: 0,
            offset: 2,
            len: 2,
        },
    ];
    let group = ColumnParallelLayer::new_packed(
        2,
        &[8, 4, 4],
        &["q", "k", "v"],
        &None,
        false,
        &comm,
        Some(&shards),
        vb,
    )?
    .expect("compatible projections should pack");

    assert_eq!(group.rows_per_rank, [4, 2, 2]);
    assert_eq!(
        group.packed.dequantize_w()?.to_vec2::<f32>()?,
        vec![
            vec![4., 4.5],
            vec![5., 5.5],
            vec![6., 6.5],
            vec![7., 7.5],
            vec![100., 100.5],
            vec![101., 101.5],
            vec![202., 202.5],
            vec![203., 203.5],
        ]
    );

    let input = Tensor::new(&[[2f32, -1.]], &device)?;
    let packed_output = group.packed.forward(&input)?;
    let mut offset = 0;
    for (projection, &rows) in group.constituents.iter().zip(&group.rows_per_rank) {
        assert_eq!(
            projection.forward(&input)?.to_vec2::<f32>()?,
            packed_output.narrow(1, offset, rows)?.to_vec2::<f32>()?
        );
        offset += rows;
    }
    Ok(())
}

#[test]
fn packed_output_layout_stays_within_tp_shards_and_preserves_canonical_weights(
) -> candle_core::Result<()> {
    let device = Device::Cpu;
    let layout = PackedOutputLayout::rank_local_interleaved_to_grouped(4, &[1, 1], 2)?;
    assert_eq!(layout.runtime_to_canonical(), &[0, 2, 1, 3, 4, 6, 5, 7]);
    let weight = Tensor::from_vec(
        (0..8).flat_map(|row| [row as f32, 0.]).collect::<Vec<_>>(),
        (8, 2),
        &device,
    )?;
    let registry = Arc::new(LoraLayerRegistry::new());
    let vb = ShardedSafeTensors::wrap(
        HashMap::from([("q.weight".to_string(), weight)]),
        DType::F32,
        device.clone(),
    )
    .with_lora_registry(registry.clone());
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 1, 2)?);
    let q_shard = Shard::Simple {
        dim: 0,
        rank: 1,
        world_size: 2,
    };
    let group = ColumnParallelLayer::new_packed_with_output_layouts(
        2,
        &[8],
        &["q"],
        &[layout],
        &None,
        false,
        &comm,
        Some(&[q_shard]),
        vb,
    )?
    .expect("rank-local row permutation should pack");

    assert_eq!(
        group.packed.dequantize_w()?.to_vec2::<f32>()?,
        vec![vec![4., 0.], vec![6., 0.], vec![5., 0.], vec![7., 0.]]
    );
    assert_eq!(
        group.constituents[0].dequantize_w()?.to_vec2::<f32>()?,
        vec![vec![4., 0.], vec![5., 0.], vec![6., 0.], vec![7., 0.]]
    );
    assert_eq!(
        group.constituents[0]
            .forward(&Tensor::new(&[[1f32, 0.]], &device)?)?
            .to_vec2::<f32>()?,
        vec![vec![4., 6., 5., 7.]]
    );
    assert_eq!(
        registry.sites()[0].spec().output_runtime_to_canonical(),
        Some(&[0, 2, 1, 3, 4, 6, 5, 7][..])
    );
    let uqff_layer = group.constituents[0].clone().apply_isq(
        None,
        device,
        &AtomicUsize::new(0),
        None,
        QuantizeOntoGuard::new().with_consumer(IsqConsumer::UqffWrite),
    )?;
    assert!(uqff_layer.isq_serde_supported());
    assert_eq!(
        uqff_layer.dequantize_w()?.to_vec2::<f32>()?,
        vec![vec![4., 0.], vec![5., 0.], vec![6., 0.], vec![7., 0.]]
    );
    Ok(())
}

#[test]
fn packed_output_layout_permutates_fp8_scale_blocks() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let scales = Tensor::new(&[[1f32, 1.], [2., 2.], [3., 3.], [4., 4.]], &device)?;
    let tensors = HashMap::from([
        (
            "q.weight".to_string(),
            Tensor::ones((8, 4), DType::F8E4M3, &device)?,
        ),
        ("q.weight_scale_inv".to_string(), scales),
    ]);
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device.clone());
    let config = Some(QuantizedConfig::Fp8 {
        weight_block_size: Some(vec![2, 2]),
        activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        fmt: Some("e4m3fn".to_string()),
        modules_to_not_convert: Vec::new(),
    });
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    let layout = PackedOutputLayout::rank_local_interleaved_to_grouped(2, &[2, 2], 1)?;
    let group = ColumnParallelLayer::new_packed_with_output_layouts(
        4,
        &[8],
        &["q"],
        &[layout],
        &config,
        false,
        &comm,
        None,
        vb,
    )?
    .expect("block-preserving FP8 output layout should pack");

    let physical = group
        .packed
        .dequantize_w()?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    assert_eq!(
        physical.iter().map(|row| row[0]).collect::<Vec<_>>(),
        vec![1., 1., 3., 3., 2., 2., 4., 4.]
    );
    let canonical = group.constituents[0]
        .dequantize_w()?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    assert_eq!(
        canonical.iter().map(|row| row[0]).collect::<Vec<_>>(),
        vec![1., 1., 2., 2., 3., 3., 4., 4.]
    );
    Ok(())
}

#[test]
fn packed_output_layout_falls_back_if_fp8_blocks_are_split() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let tensors = HashMap::from([
        (
            "q.weight".to_string(),
            Tensor::ones((4, 4), DType::F8E4M3, &device)?,
        ),
        (
            "q.weight_scale_inv".to_string(),
            Tensor::ones((2, 2), DType::F32, &device)?,
        ),
    ]);
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device.clone());
    let config = Some(QuantizedConfig::Fp8 {
        weight_block_size: Some(vec![2, 2]),
        activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        fmt: Some("e4m3".to_string()),
        modules_to_not_convert: Vec::new(),
    });
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    let layout = PackedOutputLayout::rank_local_interleaved_to_grouped(2, &[1, 1], 1)?;
    assert!(ColumnParallelLayer::new_packed_with_output_layouts(
        4,
        &[4],
        &["q"],
        &[layout],
        &config,
        false,
        &comm,
        None,
        vb,
    )?
    .is_none());
    Ok(())
}

#[test]
fn packed_projection_missing_weights_falls_back() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let vb = ShardedSafeTensors::wrap(HashMap::new(), DType::F32, device.clone());
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    assert!(ColumnParallelLayer::new_packed(
        4,
        &[4, 4],
        &["gate", "up"],
        &None,
        false,
        &comm,
        None,
        vb,
    )?
    .is_none());
    Ok(())
}

#[test]
fn packed_blockwise_fp8_preserves_scale_rows_and_outputs() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let tensors = HashMap::from([
        (
            "gate.weight".to_string(),
            Tensor::ones((4, 4), DType::F8E4M3, &device)?,
        ),
        (
            "gate.weight_scale_inv".to_string(),
            Tensor::new(&[[2f32, 3.], [5., 7.]], &device)?,
        ),
        (
            "up.weight".to_string(),
            Tensor::ones((4, 4), DType::F8E4M3, &device)?,
        ),
        (
            "up.weight_scale_inv".to_string(),
            Tensor::new(&[[11f32, 13.], [17., 19.]], &device)?,
        ),
    ]);
    let vb = ShardedSafeTensors::wrap(tensors, DType::BF16, device.clone());
    let config = Some(QuantizedConfig::Fp8 {
        weight_block_size: Some(vec![2, 2]),
        activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        fmt: Some("e4m3".to_string()),
        modules_to_not_convert: Vec::new(),
    });
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    let group = ColumnParallelLayer::new_packed(
        4,
        &[4, 4],
        &["gate", "up"],
        &config,
        false,
        &comm,
        None,
        vb.clone(),
    )?
    .expect("aligned FP8 projections should pack");
    let gate = ColumnParallelLayer::new(4, 4, &config, false, &comm, vb.pp("gate"))?;
    let up = ColumnParallelLayer::new(4, 4, &config, false, &comm, vb.pp("up"))?;
    let expected_weight = Tensor::cat(&[&gate.dequantize_w()?, &up.dequantize_w()?], 0)?;
    assert_eq!(
        group
            .packed
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?,
        expected_weight.to_dtype(DType::F32)?.to_vec2::<f32>()?
    );

    let input = Tensor::new(&[[1f32, 2., 3., 4.]], &device)?.to_dtype(DType::BF16)?;
    let packed_output = group.packed.forward(&input)?.to_dtype(DType::F32)?;
    for (index, projection) in group.constituents.iter().enumerate() {
        assert_eq!(
            projection
                .forward(&input)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?,
            packed_output.narrow(1, index * 4, 4)?.to_vec2::<f32>()?
        );
    }
    Ok(())
}

#[test]
fn packed_blockwise_fp8_falls_back_on_unaligned_boundary() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let tensors = HashMap::from([
        (
            "first.weight".to_string(),
            Tensor::ones((3, 4), DType::F8E4M3, &device)?,
        ),
        (
            "first.weight_scale_inv".to_string(),
            Tensor::ones((2, 2), DType::F32, &device)?,
        ),
        (
            "second.weight".to_string(),
            Tensor::ones((4, 4), DType::F8E4M3, &device)?,
        ),
        (
            "second.weight_scale_inv".to_string(),
            Tensor::ones((2, 2), DType::F32, &device)?,
        ),
    ]);
    let vb = ShardedSafeTensors::wrap(tensors, DType::BF16, device.clone());
    let config = Some(QuantizedConfig::Fp8 {
        weight_block_size: Some(vec![2, 2]),
        activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        fmt: Some("e4m3".to_string()),
        modules_to_not_convert: Vec::new(),
    });
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    assert!(ColumnParallelLayer::new_packed(
        4,
        &[3, 4],
        &["first", "second"],
        &config,
        false,
        &comm,
        None,
        vb,
    )?
    .is_none());
    Ok(())
}

#[test]
fn replicated_from_linear_uses_sensitive_default() {
    install_immediate(IsqType::AFQ4, Vec::new());
    assert_eq!(tracked_from_linear().ty, Some(IsqType::AFQ6));
}

#[test]
fn replicated_from_linear_preserves_explicit_type() {
    install_immediate(
        IsqType::AFQ4,
        vec![ImmediateIsqOverride {
            predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
            layer_range: None,
            ty: Some(IsqType::AFQ2),
            device: None,
        }],
    );
    assert_eq!(tracked_from_linear().ty, Some(IsqType::AFQ2));
}

#[test]
fn replicated_from_linear_promotes_q4k_sensitive_default() {
    install_immediate(IsqType::Q4K, Vec::new());
    assert_eq!(tracked_from_linear().ty, Some(IsqType::Q6K));
}

#[test]
fn replicated_from_linear_promotes_q6k_sensitive_default() {
    install_immediate(IsqType::Q6K, Vec::new());
    assert_eq!(tracked_from_linear().ty, Some(IsqType::Q8_0));
}

#[test]
fn replicated_from_linear_preserves_explicit_q_type() {
    install_immediate(
        IsqType::Q4K,
        vec![ImmediateIsqOverride {
            predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
            layer_range: None,
            ty: Some(IsqType::Q4_0),
            device: None,
        }],
    );
    assert_eq!(tracked_from_linear().ty, Some(IsqType::Q4_0));
}

#[test]
fn replicated_weight_source_is_tracked_for_immediate_isq() {
    install_immediate(IsqType::Q8_0, Vec::new());
    let vb = ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
        .with_weight_source(std::sync::Arc::new(DenseWeightSource))
        .pp("model")
        .pp("embed_tokens");
    let tracker = vb.tracker().clone();
    let layer = ReplicatedLayer::new(32, 64, &None, false, vb).unwrap();
    crate::clear_immediate_isq();

    layer
        .forward_raw(&Tensor::zeros((1, 32), DType::F32, &Device::Cpu).unwrap())
        .unwrap();
    assert_eq!(tracker.get().len(), 1);
    assert_eq!(tracker.get()[0].key, "model.embed_tokens");
}

#[test]
fn weight_source_precedes_checkpoint_quantization_for_parallel_linears() {
    let config = Some(QuantizedConfig::GptqAwq {
        bits: 4,
        group_size: 128,
        checkpoint_format: None,
        is_awq: true,
    });
    let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).unwrap());
    let vb = || {
        ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
            .with_weight_source(std::sync::Arc::new(DenseWeightSource))
            .pp("model")
            .pp("embed_tokens")
    };

    let layers = [
        ReplicatedLayer::new(32, 64, &config, false, vb()).unwrap(),
        ColumnParallelLayer::new(32, 64, &config, false, &comm, vb()).unwrap(),
        RowParallelLayer::new(32, 64, &config, false, &comm, vb()).unwrap(),
    ];
    for layer in layers {
        assert_eq!(layer.name(), "unquant-linear");
        assert_eq!(
            layer
                .forward_raw(&Tensor::zeros((1, 32), DType::F32, &Device::Cpu).unwrap())
                .unwrap()
                .dims(),
            &[1, 64]
        );
    }
}

#[test]
fn matformer_factories_transform_weight_source_layers() -> candle_core::Result<()> {
    let ty = Some(IsqType::Q8_0);
    let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
    set_immediate_isq_config(
        ImmediateIsqConfig::new(
            ty,
            vec![Regex::new(r"^(row|column|replicated)\.weight$").unwrap()],
            IsqCaptureMode::CaptureMatches,
        ),
        executor,
    );
    let config = Some(QuantizedConfig::GptqAwq {
        bits: 4,
        group_size: 128,
        checkpoint_format: None,
        is_awq: true,
    });
    let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1)?);
    let vb = |prefix| {
        let vb =
            ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
                .with_weight_source(std::sync::Arc::new(MatformerWeightSource))
                .pp(prefix);
        let tracker = vb.tracker().clone();
        (vb, tracker)
    };

    let (row_vb, row_tracker) = vb("row");
    let row = RowParallelLayer::new_matformer(4, 5, 8, &config, true, &comm, row_vb)?;
    let (column_vb, column_tracker) = vb("column");
    let column = ColumnParallelLayer::new_matformer(4, 4, 8, &config, true, &comm, column_vb)?;
    let kept = Tensor::new(&[0u32, 2], &Device::Cpu)?;
    let (replicated_vb, replicated_tracker) = vb("replicated");
    let replicated = ReplicatedLayer::new_layers_matformer_indices(
        4,
        6,
        Some(&kept),
        3,
        &config,
        true,
        replicated_vb,
    )?;
    crate::clear_immediate_isq();

    for tracker in [row_tracker, column_tracker, replicated_tracker] {
        let tracked = tracker.get();
        assert_eq!(tracked.len(), 1);
        assert!(tracked[0].shard.is_none());
    }

    for (layer, input, output) in [(row, 4, 5), (column, 4, 4), (replicated, 4, 4)] {
        assert_eq!(
            layer
                .forward_raw(&Tensor::zeros((1, input), DType::F32, &Device::Cpu)?)?
                .dims(),
            &[1, output]
        );
    }
    Ok(())
}

#[test]
fn spec_aware_layers_register_exact_feature_maps() -> candle_core::Result<()> {
    let replicated_registry = std::sync::Arc::new(LoraLayerRegistry::new());
    let replicated_vb = ShardedSafeTensors::wrap(
        HashMap::from([(
            "replicated.weight".to_string(),
            Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?,
        )]),
        DType::F32,
        Device::Cpu,
    )
    .with_lora_registry(replicated_registry.clone())
    .pp("replicated");
    let replicated_spec =
        LoraLinearSpec::replicated(4, 3).with_output_runtime_to_canonical(vec![2, 0, 1])?;
    ReplicatedLayer::new_with_lora_spec(replicated_spec.clone(), &None, false, replicated_vb)?;
    assert_eq!(replicated_registry.sites()[0].spec(), &replicated_spec);

    let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1)?);
    let row_registry = std::sync::Arc::new(LoraLayerRegistry::new());
    let row_vb = ShardedSafeTensors::wrap(
        HashMap::from([(
            "row.weight".to_string(),
            Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?,
        )]),
        DType::F32,
        Device::Cpu,
    )
    .with_lora_registry(row_registry.clone())
    .pp("row");
    let row_spec = LoraLinearSpec::row(
        4,
        3,
        Shard::Simple {
            dim: 1,
            rank: 0,
            world_size: 1,
        },
    )
    .with_input_runtime_to_canonical(vec![0, 2, 1, 3])?;
    RowParallelLayer::new_with_lora_spec(row_spec.clone(), &None, false, &comm, row_vb)?;
    assert_eq!(row_registry.sites()[0].spec(), &row_spec);
    Ok(())
}

#[test]
fn spec_aware_layers_reject_wrong_parallelism() -> candle_core::Result<()> {
    let vb = ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu);
    let row_spec = LoraLinearSpec::row(
        4,
        3,
        Shard::Simple {
            dim: 1,
            rank: 0,
            world_size: 1,
        },
    );
    assert!(ReplicatedLayer::new_with_lora_spec(row_spec, &None, false, vb.clone()).is_err());

    let comm = std::sync::Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1)?);
    assert!(RowParallelLayer::new_with_lora_spec(
        LoraLinearSpec::replicated(4, 3),
        &None,
        false,
        &comm,
        vb,
    )
    .is_err());
    Ok(())
}

#[test]
fn tp_head_layout_accepts_partitioned_kv_heads() {
    validate_tp_head_layout(48, 12, 3).unwrap();
}

#[test]
fn tp_head_layout_accepts_replicated_kv_heads() {
    validate_tp_head_layout(32, 2, 4).unwrap();
}

#[test]
fn tp_head_layout_rejects_attention_head_remainder() {
    let err = validate_tp_head_layout(40, 8, 6).unwrap_err();
    assert!(err.to_string().contains("attention heads (40)"));
}

#[test]
fn tp_head_layout_rejects_partitioned_kv_remainder() {
    let err = validate_tp_head_layout(30, 8, 3).unwrap_err();
    assert!(err.to_string().contains("KV heads (8)"));
}

#[test]
fn tp_head_layout_rejects_replicated_kv_remainder() {
    let err = validate_tp_head_layout(24, 2, 3).unwrap_err();
    assert!(err.to_string().contains("Tensor parallel size (3)"));
}
