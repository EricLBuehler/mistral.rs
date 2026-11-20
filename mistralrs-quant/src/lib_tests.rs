use std::collections::HashMap;

use super::*;

#[derive(Debug)]
struct SharedActivationProbe(Option<f32>, ActivationScaleEncoding);

impl QuantizedSerde for SharedActivationProbe {
    fn name(&self) -> &'static str {
        "shared-activation-probe"
    }
}

impl QuantMethod for SharedActivationProbe {
    fn new(_method: QuantMethodConfig) -> Result<Self> {
        Ok(Self(None, ActivationScaleEncoding::F32))
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        Tensor::zeros((1, 1), DType::F32, &Device::Cpu)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        Ok(a.clone())
    }

    fn activation_quantization_scheme(&self) -> Option<ActivationQuantizationScheme> {
        Some(ActivationQuantizationScheme {
            dtype: DType::F8E4M3,
            scale_encoding: self.1,
            block_shape: [1, 4],
        })
    }

    fn activation_calibration(&self) -> ActivationCalibration {
        self.0
            .map_or(ActivationCalibration::Dynamic, |global_scale| {
                ActivationCalibration::Static {
                    global_scale,
                    source: None,
                }
            })
    }

    fn quantize_activation(&self, _a: &Tensor) -> Result<QuantizedActivation> {
        panic!("unsupported activation dtype reached the quantizer")
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F8E4M3, Device::Cpu)
    }

    fn plan_isq(&self, _request: &IsqRequest) -> Result<IsqPlanParams> {
        candle_core::bail!("probe cannot be quantized")
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("probe cannot apply deltas")
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
}

#[test]
fn shared_activation_requires_matching_global_calibration() -> Result<()> {
    let input = Tensor::zeros((2, 4), DType::BF16, &Device::Cpu)?;
    let first = SharedActivationProbe(Some(0.25), ActivationScaleEncoding::F32);
    let second = SharedActivationProbe(Some(0.5), ActivationScaleEncoding::F32);
    assert!(forward_group(&input, &[&first, &second], GroupKind::Dense)?.is_none());
    Ok(())
}

#[test]
fn shared_activation_requires_matching_scale_encoding() -> Result<()> {
    let input = Tensor::zeros((2, 4), DType::BF16, &Device::Cpu)?;
    let first = SharedActivationProbe(None, ActivationScaleEncoding::F32);
    let second = SharedActivationProbe(None, ActivationScaleEncoding::E8M0);
    assert!(forward_group(&input, &[&first, &second], GroupKind::Dense)?.is_none());
    Ok(())
}

#[test]
fn mxfp8_activation_preserves_e8m0_scale_encoding() -> Result<()> {
    let scheme = mxfp8::MXFP8_ACTIVATION_SCHEME;
    let quantized = Tensor::zeros((2, 64), DType::F8E4M3, &Device::Cpu)?;
    let scales = Tensor::full(127u8, (2, 2), &Device::Cpu)?;
    let activation = QuantizedActivation::new(
        quantized.clone(),
        scales.clone(),
        vec![1, 2, 64],
        DType::BF16,
        scheme,
    )?;
    assert_eq!(activation.scales().dtype(), DType::U8);
    assert_eq!(
        activation.scheme().scale_encoding,
        ActivationScaleEncoding::E8M0
    );
    assert!(QuantizedActivation::new(
        quantized.clone(),
        scales.to_dtype(DType::F32)?,
        vec![2, 64],
        DType::BF16,
        scheme,
    )
    .is_err());
    let ordinary = ActivationQuantizationScheme {
        scale_encoding: ActivationScaleEncoding::F32,
        ..scheme
    };
    assert!(
        QuantizedActivation::new(quantized, scales, vec![2, 64], DType::BF16, ordinary,).is_err()
    );
    Ok(())
}

#[test]
fn packed_nvfp4_activation_keeps_logical_shape_and_rejects_wrong_scales() -> Result<()> {
    const ROWS: usize = 2;
    let source = Tensor::zeros((ROWS, 1, NVFP4_BLOCK_SIZE), DType::BF16, &Device::Cpu)?;
    let packed = Tensor::zeros((ROWS, NVFP4_BLOCK_SIZE / 2), DType::U8, &Device::Cpu)?;
    let scales = Tensor::ones((ROWS, 1), DType::F8E4M3, &Device::Cpu)?;
    let activation = QuantizedActivation::new_nvfp4(packed.clone(), scales.clone(), &source, 0.25)?;
    assert_eq!(activation.source_shape(), source.dims());
    assert_eq!(activation.quantized().dims(), &[ROWS, NVFP4_BLOCK_SIZE / 2]);
    assert_eq!(activation.global_scale(), Some(0.25));
    assert!(QuantizedActivation::new_nvfp4(
        packed.clone(),
        scales.to_dtype(DType::F32)?,
        &source,
        0.25
    )
    .is_err());
    assert!(QuantizedActivation::new_nvfp4(packed, scales, &source, f32::NAN).is_err());
    Ok(())
}

#[test]
fn shared_activation_falls_back_for_unsupported_input_dtype() -> Result<()> {
    let input = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
    let method = SharedActivationProbe(None, ActivationScaleEncoding::F32);
    assert!(forward_group(&input, &[&method, &method], GroupKind::Dense)?.is_none());
    Ok(())
}

#[test]
fn fused_glu_quantization_defaults_to_fallback() -> Result<()> {
    let input = Tensor::zeros((2, 16), DType::BF16, &Device::Cpu)?;
    let method = SharedActivationProbe(None, ActivationScaleEncoding::F32);
    let glu = GluInput::Split {
        gate: &input,
        value: &input,
    };
    assert!(method.forward_glu(glu, GluActivationType::Relu)?.is_none());
    Ok(())
}

#[test]
fn fused_quantized_glu_falls_back_off_cuda() -> Result<()> {
    let gate = Tensor::zeros((3, 4), DType::BF16, &Device::Cpu)?;
    let value = Tensor::ones((3, 4), DType::BF16, &Device::Cpu)?;
    let method = SharedActivationProbe(None, ActivationScaleEncoding::F32);
    assert!(forward_fused_glu(
        &method,
        GluInput::Split {
            gate: &gate,
            value: &value,
        },
        GluActivationType::Sigmoid,
    )?
    .is_none());
    Ok(())
}

#[test]
fn group_major_scale_rows_follow_physical_bucket() -> Result<()> {
    let alignment = NonZeroUsize::new(4).unwrap();
    assert_eq!(aligned_activation_scale_rows(1, alignment)?, 4);
    assert_eq!(aligned_activation_scale_rows(5, alignment)?, 8);
    assert_eq!(aligned_activation_scale_rows(16, alignment)?, 16);
    assert!(aligned_activation_scale_rows(usize::MAX, alignment).is_err());
    Ok(())
}

#[test]
fn quantized_activation_validates_flattened_shape_and_scale_grid() -> Result<()> {
    let scheme = ActivationQuantizationScheme {
        dtype: DType::F8E4M3,
        scale_encoding: crate::ActivationScaleEncoding::F32,
        block_shape: [1, 4],
    };
    let quantized = Tensor::zeros((6, 8), DType::F8E4M3, &Device::Cpu)?;
    let scales = Tensor::zeros((6, 2), DType::F32, &Device::Cpu)?;
    let activation = QuantizedActivation::new(
        quantized.clone(),
        scales,
        vec![2, 3, 8],
        DType::BF16,
        scheme,
    )?;
    assert_eq!(activation.source_shape(), &[2, 3, 8]);
    assert_eq!(activation.source_dtype(), DType::BF16);
    assert_eq!(activation.scheme(), scheme);
    assert_eq!(activation.scale_layout(), ActivationScaleLayout::RowMajor);

    let wrong_scales = Tensor::zeros((3, 2), DType::F32, &Device::Cpu)?;
    assert!(
        QuantizedActivation::new(quantized, wrong_scales, vec![2, 3, 8], DType::BF16, scheme,)
            .is_err()
    );
    Ok(())
}

#[test]
fn quantized_activation_validates_group_major_scale_layout() -> Result<()> {
    let scheme = ActivationQuantizationScheme {
        dtype: DType::F8E4M3,
        scale_encoding: crate::ActivationScaleEncoding::F32,
        block_shape: [1, 128],
    };
    let layout = ActivationScaleLayout::GroupMajor {
        row_alignment: std::num::NonZeroUsize::new(4).unwrap(),
    };
    let quantized = Tensor::zeros((5, 256), DType::F8E4M3, &Device::Cpu)?;
    let scales = Tensor::zeros((2, 8), DType::F32, &Device::Cpu)?;
    let activation = QuantizedActivation::new_with_scale_layout(
        quantized.clone(),
        scales,
        vec![5, 256],
        DType::BF16,
        scheme,
        layout,
    )?;
    assert_eq!(activation.scale_layout(), layout);
    let fused = FusedRmsNormQuantized::new(
        Tensor::zeros((5, 256), DType::BF16, &Device::Cpu)?,
        activation,
    )?;
    assert_eq!(fused.residual().dims(), &[5, 256]);
    assert_eq!(fused.activation().scales().dims(), &[2, 8]);

    let wrong_scales = Tensor::zeros((5, 2), DType::F32, &Device::Cpu)?;
    assert!(QuantizedActivation::new_with_scale_layout(
        quantized.clone(),
        wrong_scales,
        vec![5, 256],
        DType::BF16,
        scheme,
        layout,
    )
    .is_err());
    let row_blocked_scheme = ActivationQuantizationScheme {
        dtype: DType::F8E4M3,
        scale_encoding: crate::ActivationScaleEncoding::F32,
        block_shape: [2, 128],
    };
    assert!(QuantizedActivation::new_with_scale_layout(
        quantized,
        Tensor::zeros((2, 8), DType::F32, &Device::Cpu)?,
        vec![5, 256],
        DType::BF16,
        row_blocked_scheme,
        layout,
    )
    .is_err());
    Ok(())
}

struct DenseWeightSource;

impl QuantizedWeightSource for DenseWeightSource {
    fn contains(&self, name: &str) -> bool {
        name == "foo.weight"
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        _shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        if key != "foo" {
            return Ok(None);
        }
        let weight = Tensor::zeros((3, 2), DType::F32, device)?;
        Ok(Some(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None)),
        )?)))
    }

    fn load_optional_tensor(&self, _name: &str, _device: &Device) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn shard_alignment(&self, _key: &str) -> Result<usize> {
        Ok(1)
    }

    fn pack_factor(&self, _dtype: DType) -> Result<usize> {
        Ok(1)
    }

    fn pack_factor_for(&self, _key: &str, _dtype: DType) -> Result<Option<usize>> {
        Ok(Some(1))
    }
}

#[cfg(feature = "cuda")]
struct RecordingWeightSource {
    load_devices: Arc<std::sync::Mutex<Vec<bool>>>,
}

#[cfg(feature = "cuda")]
impl QuantizedWeightSource for RecordingWeightSource {
    fn contains(&self, name: &str) -> bool {
        name == "foo.weight"
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        _shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        if key != "foo" {
            return Ok(None);
        }
        self.load_devices.lock().unwrap().push(device.is_cpu());
        let weight = Tensor::zeros((3, 2), DType::F32, device)?;
        Ok(Some(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None)),
        )?)))
    }

    fn load_optional_tensor(&self, _name: &str, _device: &Device) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn shard_alignment(&self, _key: &str) -> Result<usize> {
        Ok(1)
    }

    fn pack_factor(&self, _dtype: DType) -> Result<usize> {
        Ok(1)
    }

    fn pack_factor_for(&self, _key: &str, _dtype: DType) -> Result<Option<usize>> {
        Ok(Some(1))
    }
}

fn empty_vb(make_dummy_regexes: Option<Vec<&str>>) -> ShardedVarBuilder {
    let backend: HashMap<String, Tensor> = HashMap::new();
    let make_dummy_regexes = make_dummy_regexes.map(|regexes| {
        Arc::new(
            regexes
                .into_iter()
                .map(Regex::new)
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
        )
    });
    ShardedSafeTensors::wrap_with_dummy_regexes(
        backend,
        DType::F32,
        Device::Cpu,
        make_dummy_regexes,
    )
}

fn immediate_params(
    ty: Option<IsqType>,
    overrides: Vec<ImmediateIsqOverride>,
) -> ImmediateIsqParams {
    let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
    ImmediateIsqParams {
        guard: QuantizeOntoGuard::new(),
        ty,
        predicates: vec![Regex::new(r"\.weight$").unwrap()],
        promoted_predicates: vec![Regex::new(r"^model\.embed_tokens\.(?:weight|bias)$").unwrap()],
        overrides,
        executor,
        capture: IsqCaptureMode::Immediate,
    }
}

#[test]
fn sensitive_tensor_policy_promotes_expected_types() {
    for ty in [IsqType::AFQ2, IsqType::AFQ3, IsqType::AFQ4] {
        assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::AFQ6);
    }
    for ty in [IsqType::AFQ6, IsqType::AFQ8] {
        assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::AFQ8);
    }
    for ty in [
        IsqType::Q2K,
        IsqType::Q3K,
        IsqType::Q4K,
        IsqType::Q4_0,
        IsqType::Q4_1,
    ] {
        assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::Q6K);
    }
    for ty in [
        IsqType::Q5K,
        IsqType::Q6K,
        IsqType::Q8K,
        IsqType::Q5_0,
        IsqType::Q5_1,
        IsqType::Q8_0,
        IsqType::Q8_1,
    ] {
        assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::Q8_0);
    }
    assert_eq!(IsqType::HQQ4.promote_for_sensitive_tensor(), IsqType::HQQ4);
}

#[test]
fn ggml_pack_factor_uses_a_safe_integer_lower_bound() {
    let dtype = DType::BF16;
    let cases = [
        (IsqType::Q4K, GgmlDType::Q4K, 512, 144, 4, 128, 3, 170),
        (IsqType::Q6K, GgmlDType::Q6K, 512, 210, 3, 170, 2, 256),
        (IsqType::Q8_0, GgmlDType::Q8_0, 64, 34, 2, 32, 1, 64),
    ];

    for (ty, quantized_dtype, dense_bytes, packed_bytes, ceil_factor, ceil_bytes, factor, bytes) in
        cases
    {
        let actual_dense_bytes = dtype.size_in_bytes() * quantized_dtype.block_size();
        let actual_packed_bytes = quantized_dtype.type_size();
        assert_eq!(actual_dense_bytes, dense_bytes);
        assert_eq!(actual_packed_bytes, packed_bytes);
        assert_eq!(
            actual_dense_bytes.div_ceil(actual_packed_bytes),
            ceil_factor
        );
        assert_eq!(actual_dense_bytes / ceil_factor, ceil_bytes);
        assert!(ceil_bytes < actual_packed_bytes);
        assert_eq!(ty.pack_factor(dtype), factor);
        assert_eq!(actual_dense_bytes / factor, bytes);
        assert!(bytes >= actual_packed_bytes);
    }
}

#[test]
fn ggml_and_f8q8_pack_factors_never_undercount_block_storage() {
    let ggml_types = [
        (IsqType::Q4_0, GgmlDType::Q4_0),
        (IsqType::Q4_1, GgmlDType::Q4_1),
        (IsqType::Q5_0, GgmlDType::Q5_0),
        (IsqType::Q5_1, GgmlDType::Q5_1),
        (IsqType::Q8_0, GgmlDType::Q8_0),
        (IsqType::Q8_1, GgmlDType::Q8_1),
        (IsqType::Q2K, GgmlDType::Q2K),
        (IsqType::Q3K, GgmlDType::Q3K),
        (IsqType::Q4K, GgmlDType::Q4K),
        (IsqType::Q5K, GgmlDType::Q5K),
        (IsqType::Q6K, GgmlDType::Q6K),
        (IsqType::Q8K, GgmlDType::Q8K),
    ];

    for dtype in [DType::F16, DType::BF16, DType::F32] {
        for (ty, quantized_dtype) in ggml_types {
            let estimated_bytes =
                quantized_dtype.block_size() / ty.pack_factor(dtype) * dtype.size_in_bytes();
            assert!(estimated_bytes >= quantized_dtype.type_size());
        }

        let f8q8_bytes = f8q8::QK8_0 / IsqType::F8Q8.pack_factor(dtype) * dtype.size_in_bytes();
        assert!(f8q8_bytes >= std::mem::size_of::<f8q8::BlockF8Q8>());

        let mxfp4_bytes =
            mxfp4::MXFP4_BLOCK_SIZE / IsqType::MXFP4.pack_factor(dtype) * dtype.size_in_bytes();
        let packed_mxfp4_bytes =
            mxfp4::MXFP4_BLOCK_SIZE * mxfp4::N_BITS / u8::BITS as usize + DType::U8.size_in_bytes();
        assert!(mxfp4_bytes >= packed_mxfp4_bytes);
    }
}

#[test]
fn affine_pack_factors_include_group_metadata() {
    let afq_types = [
        (IsqType::AFQ2, 2),
        (IsqType::AFQ3, 3),
        (IsqType::AFQ4, 4),
        (IsqType::AFQ6, 6),
        (IsqType::AFQ8, 8),
    ];
    for (dtype, expected) in [
        (DType::F16, [5, 4, 3, 2, 1]),
        (DType::BF16, [5, 4, 3, 2, 1]),
        (DType::F32, [8, 6, 5, 4, 3]),
    ] {
        for ((ty, bits), expected_factor) in afq_types.into_iter().zip(expected) {
            let group_size = AfqGroupSize::Low as usize;
            let packed_bytes = (group_size * bits).div_ceil(u8::BITS as usize)
                + AFFINE_METADATA_TENSOR_COUNT * dtype.size_in_bytes();
            assert_eq!(ty.pack_factor(dtype), expected_factor);
            assert!(group_size / ty.pack_factor(dtype) * dtype.size_in_bytes() >= packed_bytes);
        }
    }

    for (dtype, hqq4, hqq8) in [(DType::F16, 3, 1), (DType::BF16, 3, 1), (DType::F32, 6, 3)] {
        for (ty, bits, expected_factor) in [(IsqType::HQQ4, 4, hqq4), (IsqType::HQQ8, 8, hqq8)] {
            let packed_bytes = (hqq::ISQ_HQQ_GROUP_SIZE * bits).div_ceil(u8::BITS as usize)
                + AFFINE_METADATA_TENSOR_COUNT * DType::F32.size_in_bytes();
            assert_eq!(ty.pack_factor(dtype), expected_factor);
            assert!(
                hqq::ISQ_HQQ_GROUP_SIZE / ty.pack_factor(dtype) * dtype.size_in_bytes()
                    >= packed_bytes
            );
        }
    }
}

#[test]
fn configured_pack_factors_use_their_storage_contracts() {
    let afq = QuantizedConfig::Afq {
        bits: 4,
        group_size: 64,
    };
    assert_eq!(afq.pack_factor(DType::F32), 6);

    let legacy_mxfp4 = QuantizedConfig::GptqAwq {
        bits: 40,
        group_size: 32,
        checkpoint_format: None,
        is_awq: false,
    };
    assert_eq!(legacy_mxfp4.pack_factor(DType::BF16), 3);
    assert_eq!(legacy_mxfp4.pack_factor(DType::F32), 6);

    let bnb8 = QuantizedConfig::Bitsandbytes {
        bnb_4bit_quant_type: None,
    };
    assert_eq!(bnb8.pack_factor(DType::BF16), 1);
}

#[test]
fn fp8_config_preserves_checkpoint_semantics() {
    let config: QuantizedConfig = serde_json::from_str(
        r#"{
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "modules_to_not_convert": ["model.embed_tokens", "lm_head"],
                "weight_block_size": [128, 128]
            }"#,
    )
    .unwrap();

    let QuantizedConfig::Fp8 {
        weight_block_size,
        activation_scheme,
        fmt,
        modules_to_not_convert,
    } = config
    else {
        panic!("expected FP8 config")
    };
    assert_eq!(weight_block_size, Some(vec![128, 128]));
    assert_eq!(activation_scheme, Some(Fp8ActivationScheme::Dynamic));
    assert_eq!(fmt.as_deref(), Some("e4m3"));
    assert_eq!(modules_to_not_convert, ["model.embed_tokens", "lm_head"]);

    let legacy: QuantizedConfig =
        serde_json::from_str(r#"{"quant_method":"fp8","modules_to_not_convert":null}"#).unwrap();
    let QuantizedConfig::Fp8 {
        activation_scheme,
        fmt,
        modules_to_not_convert,
        ..
    } = legacy
    else {
        panic!("expected FP8 config")
    };
    assert_eq!(activation_scheme, None);
    assert_eq!(fmt, None);
    assert!(modules_to_not_convert.is_empty());
}

#[test]
fn fp8_config_deserializes_compressed_tensors_and_model_opt() {
    let compressed: QuantizedConfig = serde_json::from_value(serde_json::json!({
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "dynamic": false,
                    "num_bits": 8,
                    "strategy": "channel",
                    "symmetric": true,
                    "type": "float"
                },
                "input_activations": null
            }
        }
    }))
    .unwrap();
    assert!(matches!(
        &compressed,
        QuantizedConfig::CompressedTensors { .. }
    ));
    assert_eq!(
        compressed
            .resolve_fp8("model.layers.0.mlp.up_proj")
            .unwrap()
            .unwrap()
            .activation,
        Fp8ActivationMode::None
    );
    let serialized = serde_json::to_value(&compressed).unwrap();
    assert_eq!(serialized["quant_method"], "compressed-tensors");
    assert_eq!(serialized["format"], "float-quantized");
    assert!(serialized.get("config").is_none());
    let round_trip: QuantizedConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        round_trip
            .resolve_fp8("model.layers.0.mlp.up_proj")
            .unwrap(),
        compressed
            .resolve_fp8("model.layers.0.mlp.up_proj")
            .unwrap()
    );

    let model_opt: QuantizedConfig = serde_json::from_value(serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
        "producer": {"name": "modelopt", "version": "test"}
    }))
    .unwrap();
    assert!(matches!(&model_opt, QuantizedConfig::ModelOpt { .. }));
    assert_eq!(
        model_opt
            .resolve_fp8("model.layers.0.mlp.up_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Channel
    );
    let serialized = serde_json::to_value(&model_opt).unwrap();
    assert_eq!(serialized["quant_method"], "modelopt");
    assert_eq!(serialized["producer"]["version"], "test");
    assert!(serialized.get("config").is_none());
    let round_trip: QuantizedConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        round_trip
            .resolve_fp8("model.layers.0.mlp.up_proj")
            .unwrap(),
        model_opt.resolve_fp8("model.layers.0.mlp.up_proj").unwrap()
    );
}

#[test]
fn native_fp8_merges_exclusion_aliases_and_rejects_other_formats() {
    let config: QuantizedConfig = serde_json::from_value(serde_json::json!({
        "quant_method": "fp8",
        "fmt": "e4m3",
        "modules_to_not_convert": ["lm_head"],
        "ignored_layers": ["model.embed_tokens"]
    }))
    .unwrap();
    assert!(config.resolve_fp8("lm_head").unwrap().is_none());
    assert!(config.resolve_fp8("model.embed_tokens").unwrap().is_none());
    assert!(
        serde_json::from_value::<QuantizedConfig>(serde_json::json!({
            "quant_method": "fp8",
            "fmt": "e5m2"
        }))
        .is_err()
    );
}

#[test]
fn legacy_model_opt_file_can_be_merged_with_embedded_config() {
    let external = serde_json::json!({
        "quantization": {
            "quant_algo": "FP8",
            "exclude_modules": ["old_head"]
        }
    });
    let embedded = serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
        "ignore": ["lm_head"]
    });
    let config = QuantizedConfig::from_modelopt_configs(Some(&embedded), &external).unwrap();
    assert!(config.resolve_fp8("lm_head").unwrap().is_none());
    assert_eq!(
        config
            .resolve_fp8("model.layers.0.self_attn.q_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Channel
    );
}

#[test]
fn immediate_isq_promotion_requires_an_explicit_predicate() {
    let params = immediate_params(Some(IsqType::AFQ4), Vec::new());
    let promoted = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
    assert_eq!(promoted.ty, Some(IsqType::AFQ6));
    assert!(promoted.promote_default);

    let lookalike = resolve_immediate_isq(&params, "vision.embed_tokens.weight").unwrap();
    assert_eq!(lookalike.ty, Some(IsqType::AFQ4));
    assert!(!lookalike.promote_default);
}

#[test]
fn immediate_isq_explicit_type_wins_over_sensitive_default() {
    let params = immediate_params(
        Some(IsqType::AFQ4),
        vec![ImmediateIsqOverride {
            predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
            layer_range: None,
            ty: Some(IsqType::AFQ2),
            device: None,
        }],
    );

    let matched = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
    assert_eq!(matched.ty, Some(IsqType::AFQ2));
    assert!(matched.promote_default);
}

#[test]
fn immediate_isq_uses_sensitive_default_only_for_sensitive_tensors() {
    for (base, sensitive) in [
        (IsqType::AFQ6, IsqType::AFQ8),
        (IsqType::Q4K, IsqType::Q6K),
        (IsqType::Q6K, IsqType::Q8_0),
    ] {
        let params = immediate_params(Some(base), Vec::new());
        let embedding = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
        assert_eq!(embedding.ty, Some(sensitive));
        let projection =
            resolve_immediate_isq(&params, "model.layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(projection.ty, Some(base));
    }
}

#[test]
fn immediate_isq_device_only_override_keeps_sensitive_default() {
    let params = immediate_params(
        Some(IsqType::AFQ4),
        vec![ImmediateIsqOverride {
            predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
            layer_range: None,
            ty: None,
            device: Some(Device::Cpu),
        }],
    );

    let matched = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
    assert_eq!(matched.ty, Some(IsqType::AFQ6));
}

#[test]
fn capture_all_records_promotion_without_pinning_a_type() {
    let mut params = immediate_params(Some(IsqType::AFQ4), Vec::new());
    params.capture = IsqCaptureMode::CaptureAll;

    let promoted = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
    assert_eq!(promoted.ty, None);
    assert!(promoted.promote_default);

    let regular = resolve_immediate_isq(&params, "model.layers.0.mlp.down_proj.weight").unwrap();
    assert_eq!(regular.ty, None);
    assert!(!regular.promote_default);
}

#[test]
fn missing_linear_weight_outside_uqff_errors() {
    let err = linear_no_bias(2, 3, &None, empty_vb(None).pp("foo")).unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains("Missing required tensor(s)"));
    assert!(msg.contains("foo.weight"));
    assert!(msg.contains("UQFF"));
}

#[test]
fn missing_uqff_placeholder_creates_contextual_dummy() -> Result<()> {
    let layer = linear_no_bias(
        2,
        3,
        &None,
        empty_vb(Some(vec![r"^foo\.weight$"])).pp("foo"),
    )?;

    let info = layer.dummy_info().unwrap();
    assert_eq!(layer.name(), "dummy");
    assert_eq!(info.context, "linear_no_bias");
    assert_eq!(info.prefix, "foo");
    assert_eq!(info.missing_tensors, vec!["foo.weight"]);

    let input = Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?;
    let err = layer.forward_raw(&input).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("forward pass"));
    assert!(msg.contains("foo.weight"));
    assert!(msg.contains("temporary UQFF placeholders"));

    Ok(())
}

#[test]
fn weight_source_linear_participates_in_immediate_isq_tracking() -> Result<()> {
    let ty = Some(IsqType::Q8_0);
    let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
    set_immediate_isq_config(
        ImmediateIsqConfig::new(
            ty,
            vec![Regex::new(r"^foo\.weight$").unwrap()],
            IsqCaptureMode::CaptureMatches,
        ),
        executor,
    );

    let vb = empty_vb(None)
        .with_weight_source(Arc::new(DenseWeightSource))
        .pp("foo");
    let tracker = vb.tracker().clone();
    let layer = linear_no_bias(2, 3, &None, vb);
    clear_immediate_isq();

    layer?.forward_raw(&Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)?;
    assert_eq!(tracker.get().len(), 1);
    assert_eq!(tracker.get()[0].key, "foo");
    assert_eq!(tracker.get()[0].ty, ty);
    Ok(())
}

#[test]
fn weight_source_precedes_checkpoint_quantization_for_plain_linears() -> Result<()> {
    let config = Some(QuantizedConfig::GptqAwq {
        bits: 4,
        group_size: 128,
        checkpoint_format: None,
        is_awq: true,
    });
    let vb = || {
        empty_vb(None)
            .with_weight_source(Arc::new(DenseWeightSource))
            .pp("foo")
    };
    for layer in [
        linear_no_bias(2, 3, &config, vb())?,
        linear(2, 3, &config, vb())?,
    ] {
        assert_eq!(layer.name(), "unquant-linear");
        assert_eq!(
            layer
                .forward_raw(&Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)?
                .dims(),
            &[1, 3]
        );
    }
    Ok(())
}

#[test]
fn uqff_source_precedes_checkpoint_quantization_for_plain_linears() -> Result<()> {
    let dir = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
    let path = dir.path().join("source-first.uqff");
    let mut tensors = uqff_version_tensors();
    for (prefix, bias) in [("no_bias", false), ("with_bias", true)] {
        let bias = if bias {
            Some(Tensor::zeros(3, DType::F32, &Device::Cpu)?)
        } else {
            None
        };
        let layer = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            Tensor::zeros((3, 2), DType::F32, &Device::Cpu)?,
            bias,
        )))?;
        tensors.extend(layer.serialize_uqff(prefix, IsqType::Q4K)?);
    }
    ::safetensors::serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name(), tensor)),
        None,
        &path,
    )
    .map_err(candle_core::Error::wrap)?;

    let reader = Arc::new(UqffReader::open(&[path])?);
    let config = Some(QuantizedConfig::GptqAwq {
        bits: 4,
        group_size: 128,
        checkpoint_format: None,
        is_awq: true,
    });
    let vb = |prefix| empty_vb(None).with_uqff_reader(reader.clone()).pp(prefix);
    for layer in [
        linear_no_bias(2, 3, &config, vb("no_bias"))?,
        linear(2, 3, &config, vb("with_bias"))?,
    ] {
        assert_eq!(layer.name(), "unquant-linear");
        assert_eq!(
            layer
                .forward_raw(&Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)?
                .dims(),
            &[1, 3]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn weight_source_stages_immediate_isq_on_cpu() -> Result<()> {
    let Ok(device) = Device::new_cuda(0) else {
        return Ok(());
    };
    let load_devices = Arc::new(std::sync::Mutex::new(Vec::new()));
    let source = Arc::new(RecordingWeightSource {
        load_devices: load_devices.clone(),
    });
    let vb = ShardedSafeTensors::wrap(HashMap::new(), DType::F32, device)
        .with_weight_source(source)
        .pp("foo");

    let ty = Some(IsqType::Q8_0);
    let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
    set_immediate_isq_config(
        ImmediateIsqConfig::new(
            ty,
            vec![Regex::new(r"^foo\.weight$").unwrap()],
            IsqCaptureMode::CaptureMatches,
        ),
        executor,
    );
    let tracker = vb.tracker().clone();
    linear_no_bias(2, 3, &None, vb.clone())?;
    tracker.get()[0].ct.resolve()?;
    clear_immediate_isq();

    linear_no_bias(2, 3, &None, vb)?;
    assert_eq!(&*load_devices.lock().unwrap(), &[true, false]);
    Ok(())
}
