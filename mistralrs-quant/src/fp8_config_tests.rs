use std::collections::HashMap;

use super::*;
use serde_json::json;

fn ct_args(strategy: &str) -> Value {
    json!({
        "dynamic": false,
        "num_bits": 8,
        "strategy": strategy,
        "symmetric": true,
        "type": "float"
    })
}

fn ct_nvfp4_args(dynamic: Value) -> Value {
    json!({
        "dynamic": dynamic,
        "num_bits": 4,
        "strategy": "tensor_group",
        "group_size": 16,
        "symmetric": true,
        "type": "float",
        "scale_dtype": "torch.float8_e4m3fn"
    })
}

#[test]
fn compressed_tensors_nvfp4_preserves_activation_contract_and_scale_convention() {
    for (input, activation, scale_name) in [
        (Value::Null, Nvfp4ActivationMode::None, None),
        (
            ct_nvfp4_args(json!("local")),
            Nvfp4ActivationMode::DynamicBlock,
            Some("input_global_scale"),
        ),
    ] {
        let config = CheckpointQuantConfig::compressed_tensors(&json!({
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_nvfp4_args(json!(false)),
                    "input_activations": input
                }
            },
            "ignore": ["lm_head"]
        }))
        .unwrap();
        let Some(CheckpointLinearSpec::Nvfp4(spec)) = config.resolve_checked("layer").unwrap()
        else {
            panic!("expected NVFP4");
        };
        assert_eq!(spec.activation, activation);
        assert_eq!(spec.global_scale, ScaleConvention::Quantize);
        assert_eq!(spec.scale_names.weight, "weight_packed");
        assert_eq!(spec.scale_names.block_scale, "weight_scale");
        assert_eq!(spec.scale_names.global_scale, "weight_global_scale");
        assert_eq!(spec.scale_names.activation_scale, scale_name);
        assert!(config
            .resolve_fp8("layer")
            .unwrap_err()
            .contains("uses NVFP4"));
        assert_eq!(config.resolve_checked("lm_head").unwrap(), None);
    }
}

#[test]
fn compressed_tensors_mixed_fp8_nvfp4_targets_and_fused_conflicts() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {"targets": ["Linear"], "weights": ct_args("channel")},
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "targets": ["re:.*mlp\\.(gate|up)_proj$"],
                "weights": ct_nvfp4_args(json!(false))
            },
            "group_2": {
                "targets": ["model.layers.1.mlp.up_proj"],
                "weights": ct_args("tensor")
            }
        }
    }))
    .unwrap();
    assert!(matches!(
        config
            .resolve_checked("model.layers.0.mlp.gate_up_proj")
            .unwrap(),
        Some(CheckpointLinearSpec::Nvfp4(_))
    ));
    assert!(matches!(
        config
            .resolve_checked("model.layers.0.self_attn.q_proj")
            .unwrap(),
        Some(CheckpointLinearSpec::Fp8(_))
    ));
    assert!(config
        .resolve_checked("model.layers.1.mlp.gate_up_proj")
        .unwrap_err()
        .contains("different schemes"));
}

#[test]
fn compressed_tensors_nvfp4_rejects_incompatible_quantization_contracts() {
    let base = json!({
        "format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": ct_nvfp4_args(json!(false)),
            "input_activations": ct_nvfp4_args(json!("local"))
        }}
    });
    for (kind, field, value, expected) in [
        ("weights", "group_size", json!(32), "group_size=16"),
        ("weights", "dynamic", json!(true), "static weights"),
        (
            "weights",
            "scale_dtype",
            json!("torch.float8_e8m0fnu"),
            "E4M3FN",
        ),
        ("weights", "symmetric", json!(false), "symmetric weights"),
        (
            "input_activations",
            "dynamic",
            json!(true),
            "calibrated global scale",
        ),
        (
            "input_activations",
            "num_bits",
            json!(8),
            "4-bit float activations",
        ),
    ] {
        let mut value_config = base.clone();
        value_config["config_groups"]["group_0"][kind][field] = value;
        assert!(CheckpointQuantConfig::compressed_tensors(&value_config)
            .unwrap_err()
            .contains(expected));
    }
}

#[test]
fn model_opt_nvfp4_preserves_activation_contract_and_scale_convention() {
    for (algorithm, activation, scale_name) in [
        (
            "NVFP4",
            Nvfp4ActivationMode::DynamicBlock,
            Some("input_scale"),
        ),
        ("W4A16_NVFP4", Nvfp4ActivationMode::None, None),
    ] {
        let config = CheckpointQuantConfig::model_opt(&json!({
            "quantization": {
                "quant_algo": algorithm,
                "group_size": 16,
                "exclude_modules": ["lm_head"]
            }
        }))
        .unwrap();
        let Some(CheckpointLinearSpec::Nvfp4(spec)) = config.resolve_checked("layer").unwrap()
        else {
            panic!("expected NVFP4");
        };
        assert_eq!(spec.activation, activation);
        assert_eq!(spec.global_scale, ScaleConvention::Dequantize);
        assert_eq!(spec.scale_names.weight, "weight");
        assert_eq!(spec.scale_names.global_scale, "weight_scale_2");
        assert_eq!(spec.scale_names.activation_scale, scale_name);
        assert_eq!(config.resolve_checked("lm_head").unwrap(), None);
    }
    assert!(CheckpointQuantConfig::model_opt(&json!({
        "quant_algo": "NVFP4", "group_size": 32
    }))
    .unwrap_err()
    .contains("group_size=16"));
}

#[test]
fn model_opt_mixed_expert_container_targets_preserve_exact_overrides() {
    let config = CheckpointQuantConfig::model_opt(&json!({
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.language_model.layers.0.mlp.experts": {
                "quant_algo": "W4A16_NVFP4", "group_size": 16
            },
            "model.language_model.layers.0.mlp.experts.1.up_proj": {"quant_algo": "FP8"},
            "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}
        },
        "ignore": ["*.experts.2.*"]
    }))
    .unwrap();
    assert!(matches!(
        config
            .resolve_checked("model.layers.0.mlp.experts.0.gate_up_proj")
            .unwrap(),
        Some(CheckpointLinearSpec::Nvfp4(_))
    ));
    assert!(matches!(
        config
            .resolve_checked("model.layers.0.mlp.experts.1.up_proj")
            .unwrap(),
        Some(CheckpointLinearSpec::Fp8(_))
    ));
    assert!(config
        .resolve_checked("model.layers.0.mlp.experts.1.gate_up_proj")
        .unwrap_err()
        .contains("different schemes"));
    assert_eq!(
        config
            .resolve_checked("model.layers.0.mlp.experts.2.gate_proj")
            .unwrap(),
        None
    );
    assert_eq!(
        config
            .resolve_checked("model.layers.0.mlp.experts_extra.0.gate_proj")
            .unwrap(),
        None
    );
}

#[test]
fn compressed_tensors_resolves_channel_dynamic_and_ignore() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("channel"),
                "input_activations": {
                    "dynamic": true,
                    "num_bits": 8,
                    "strategy": "token",
                    "symmetric": true,
                    "type": "float"
                }
            }
        },
        "ignore": ["lm_head"]
    }))
    .unwrap();

    let spec = config
        .resolve_fp8("model.layers.0.self_attn.q_proj")
        .unwrap()
        .unwrap();
    assert_eq!(spec.weight_scale, Fp8WeightScaleLayout::Channel);
    assert_eq!(spec.activation, Fp8ActivationMode::DynamicToken);
    assert_eq!(spec.scale_names.weight[0], "weight_scale");
    assert_eq!(config.resolve_fp8("lm_head").unwrap(), None);
}

#[test]
fn compressed_tensors_resolves_block_and_regex_targets() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$"],
                "weights": {
                    "block_structure": [128, 128],
                    "dynamic": false,
                    "num_bits": 8,
                    "strategy": "block",
                    "symmetric": true,
                    "type": "float"
                },
                "input_activations": {
                    "dynamic": true,
                    "group_size": 128,
                    "num_bits": 8,
                    "strategy": "group",
                    "symmetric": true,
                    "type": "float"
                }
            },
            "group_1": {
                "targets": ["Linear"],
                "weights": {"dynamic": false, "num_bits": 4, "type": "float"}
            }
        }
    }))
    .unwrap();

    let spec = config
        .resolve_fp8("model.layers.2.self_attn.q_proj")
        .unwrap()
        .unwrap();
    assert_eq!(spec.weight_scale, Fp8WeightScaleLayout::Block([128, 128]));
    assert_eq!(spec.activation, Fp8ActivationMode::DynamicBlock(128));
    assert!(config.resolve_fp8("model.layers.2.mlp.gate_proj").is_err());
}

#[test]
fn compressed_tensors_resolves_w8a16() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("tensor"),
                "input_activations": null
            }
        }
    }))
    .unwrap();
    let spec = config
        .resolve_fp8("model.layers.0.mlp.down_proj")
        .unwrap()
        .unwrap();
    assert_eq!(spec.activation, Fp8ActivationMode::None);
    assert_eq!(spec.weight_scale, Fp8WeightScaleLayout::Tensor);
}

#[test]
fn compressed_tensors_exact_target_precedes_linear_class() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("tensor")
            },
            "group_1": {
                "targets": ["model.layers.0.self_attn.q_proj"],
                "weights": ct_args("channel")
            }
        }
    }))
    .unwrap();
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
fn compressed_tensors_non_fp8_target_shadows_linear_fp8() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("channel")
            },
            "group_1": {
                "targets": ["model.layers.0.mlp.down_proj"],
                "output_activations": {
                    "dynamic": false,
                    "num_bits": 8,
                    "type": "int"
                },
                "weights": {
                    "dynamic": false,
                    "num_bits": 4,
                    "strategy": "group",
                    "symmetric": true,
                    "type": "int"
                }
            }
        }
    }))
    .unwrap();
    assert!(config.resolve_fp8("model.layers.0.mlp.down_proj").is_err());
    assert!(config
        .resolve_fp8("model.layers.0.mlp.down_proj")
        .unwrap_err()
        .contains("4-bit `int` weights"));
    assert!(config
        .resolve_fp8("model.layers.0.mlp.up_proj")
        .unwrap()
        .is_some());
}

#[test]
fn compressed_tensors_uses_natural_group_order() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "config_groups": {
            "group_10": {
                "targets": ["re:.*q_proj$"],
                "weights": ct_args("channel")
            },
            "group_2": {
                "targets": ["re:.*q_proj$"],
                "weights": ct_args("tensor")
            }
        }
    }))
    .unwrap();
    assert_eq!(
        config
            .resolve_fp8("model.layers.0.self_attn.q_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Tensor
    );
}

#[test]
fn compressed_tensors_fused_projection_propagates_unsupported_constituent() {
    let config = CheckpointQuantConfig::compressed_tensors(&json!({
        "config_groups": {
            "group_0": {
                "targets": ["model.layers.0.mlp.gate_proj"],
                "weights": ct_args("tensor")
            },
            "group_1": {
                "targets": ["model.layers.0.mlp.up_proj"],
                "weights": {
                    "dynamic": false,
                    "num_bits": 4,
                    "strategy": "tensor_group",
                    "symmetric": true,
                    "type": "float"
                }
            }
        }
    }))
    .unwrap();
    let error = config
        .resolve_fp8("model.layers.0.mlp.gate_up_proj")
        .unwrap_err();
    assert!(error.contains("4-bit `float` weights"));
}

#[test]
fn compressed_tensors_rejects_output_activation_quantization() {
    let error = CheckpointQuantConfig::compressed_tensors(&json!({
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("tensor"),
                "output_activations": ct_args("tensor")
            }
        }
    }))
    .unwrap_err();
    assert!(error.contains("output activation"));
}

#[test]
fn compressed_tensors_rejects_asymmetric_fp8_activations() {
    let error = CheckpointQuantConfig::compressed_tensors(&json!({
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("tensor"),
                "input_activations": {
                    "dynamic": true,
                    "num_bits": 8,
                    "strategy": "token",
                    "symmetric": false,
                    "type": "float"
                }
            }
        }
    }))
    .unwrap_err();
    assert!(error.contains("symmetric activations"));
}

#[test]
fn compressed_tensors_rejects_block_activations_with_non_block_weights() {
    let error = CheckpointQuantConfig::compressed_tensors(&json!({
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("channel"),
                "input_activations": {
                    "dynamic": true,
                    "group_size": 128,
                    "num_bits": 8,
                    "strategy": "group",
                    "symmetric": true,
                    "type": "float"
                }
            }
        }
    }))
    .unwrap_err();
    assert!(error.contains("blockwise weights for blockwise activations"));
}

#[test]
fn model_opt_resolves_all_fp8_algorithms() {
    for (algorithm, weight, activation) in [
        (
            "FP8",
            Fp8WeightScaleLayout::Tensor,
            Fp8ActivationMode::StaticTensor,
        ),
        (
            "FP8_PER_CHANNEL_PER_TOKEN",
            Fp8WeightScaleLayout::Channel,
            Fp8ActivationMode::DynamicToken,
        ),
        (
            "fp8_pb_wo",
            Fp8WeightScaleLayout::Block([128, 128]),
            Fp8ActivationMode::None,
        ),
    ] {
        let config = CheckpointQuantConfig::model_opt(&json!({
            "quant_method": "modelopt",
            "quant_algo": algorithm
        }))
        .unwrap();
        let spec = config
            .resolve_fp8("model.layers.0.mlp.gate_proj")
            .unwrap()
            .unwrap();
        assert_eq!(spec.weight_scale, weight);
        assert_eq!(spec.activation, activation);
        assert_eq!(spec.scale_names.weight[0], "weight_scale");
    }
}

#[test]
fn model_opt_mixed_precision_resolves_per_layer() {
    let config = CheckpointQuantConfig::model_opt(&json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": ["mtp*"],
        "quantized_layers": {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
            "model.layers.0.self_attn.k_proj": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
            },
            "model.layers.0.mlp.gate_proj": {"quant_algo": "UNQUANTIZED"}
        }
    }))
    .unwrap();

    assert_eq!(
        config
            .resolve_fp8("model.layers.0.self_attn.q_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Tensor
    );
    assert_eq!(
        config
            .resolve_fp8("model.layers.0.self_attn.k_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Channel
    );
    assert_eq!(
        config.resolve_fp8("model.layers.0.mlp.gate_proj").unwrap(),
        None
    );
    assert_eq!(
        config.resolve_fp8("mtp.layers.0.self_attn.q_proj").unwrap(),
        None
    );
}

#[test]
fn model_opt_mixed_precision_resolves_fused_projection_constituents() {
    let config = CheckpointQuantConfig::model_opt(&json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
            "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
            "model.layers.0.self_attn.q_proj": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
            },
            "model.layers.0.self_attn.k_proj": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
            },
            "model.layers.0.self_attn.v_proj": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
            },
            "model.layers.1.mlp.gate_proj": {"quant_algo": "FP8"},
            "model.layers.1.mlp.up_proj": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
            },
            "model.layers.2.mlp.gate_up_proj": {"quant_algo": "FP8_PB_WO"},
            "model.layers.2.mlp.gate_proj": {"quant_algo": "FP8"},
            "model.layers.2.mlp.up_proj": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
            }
        }
    }))
    .unwrap();

    assert_eq!(
        config
            .resolve_fp8("model.layers.0.mlp.gate_up_proj")
            .unwrap()
            .unwrap()
            .activation,
        Fp8ActivationMode::StaticTensor
    );
    assert_eq!(
        config
            .resolve_fp8("model.layers.0.self_attn.qkv_proj")
            .unwrap()
            .unwrap()
            .activation,
        Fp8ActivationMode::DynamicToken
    );
    assert!(config
        .resolve_fp8("model.layers.1.mlp.gate_up_proj")
        .unwrap_err()
        .contains("different schemes"));
    assert_eq!(
        config
            .resolve_fp8("model.layers.2.mlp.gate_up_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Block([128, 128])
    );
}

#[test]
fn model_opt_mixed_precision_rejects_unsupported_algorithms() {
    let error = CheckpointQuantConfig::model_opt(&json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "W4A8_NVFP4_FP8"}
        }
    }))
    .unwrap_err();
    assert!(error.contains("W4A8_NVFP4_FP8"));
}

#[test]
fn model_opt_mixed_precision_requires_a_supported_layer() {
    let error = CheckpointQuantConfig::model_opt(&json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "UNQUANTIZED"}
        }
    }))
    .unwrap_err();
    assert!(error.contains("no supported FP8 or NVFP4 layers"));
}

#[test]
fn legacy_model_opt_quantization_object_is_supported() {
    let config = CheckpointQuantConfig::model_opt(&json!({
        "producer": {"name": "modelopt"},
        "quantization": {
            "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
            "exclude_modules": ["lm_head"]
        }
    }))
    .unwrap();
    assert_eq!(config.resolve_fp8("lm_head").unwrap(), None);
    assert_eq!(
        config
            .resolve_fp8("model.layers.0.mlp.up_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Channel
    );
    for prefix in [
        "model.vision_tower.encoder.layers.0.mlp.fc1",
        "model.vision_model.encoder.layers.0.mlp.fc1",
        "model.vit_large_projector.linear_1",
    ] {
        assert_eq!(config.resolve_fp8(prefix).unwrap(), None);
    }
}

#[test]
fn embedded_model_opt_fields_override_legacy_file() {
    let external = json!({
        "producer": {"name": "modelopt"},
        "quantization": {
            "quant_algo": "FP8",
            "exclude_modules": ["old_head"]
        }
    });
    let embedded = json!({
        "quant_method": "modelopt",
        "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
        "ignore": ["lm_head"]
    });
    let config = CheckpointQuantConfig::model_opt_merged(Some(&embedded), &external).unwrap();
    assert_eq!(config.resolve_fp8("lm_head").unwrap(), None);
    assert_eq!(
        config
            .resolve_fp8("model.layers.0.mlp.up_proj")
            .unwrap()
            .unwrap()
            .weight_scale,
        Fp8WeightScaleLayout::Channel
    );
}

#[test]
fn scale_layout_normalizes_singleton_checkpoint_dimensions() {
    let device = candle_core::Device::Cpu;
    let tensor = Tensor::zeros(1, DType::F16, &device).unwrap();
    let tensor = Fp8WeightScaleLayout::Tensor
        .normalize(tensor, [896, 896])
        .unwrap();
    assert!(tensor.dims().is_empty());
    assert_eq!(tensor.dtype(), DType::F32);

    let channel = Tensor::zeros((896, 1), DType::BF16, &device).unwrap();
    let channel = Fp8WeightScaleLayout::Channel
        .normalize(channel, [896, 896])
        .unwrap();
    assert_eq!(channel.dims(), [896]);
    assert_eq!(channel.dtype(), DType::F32);

    let blocks = Tensor::zeros((7, 1, 7, 1), DType::F32, &device).unwrap();
    let blocks = Fp8WeightScaleLayout::Block([128, 128])
        .normalize(blocks, [896, 896])
        .unwrap();
    assert_eq!(blocks.dims(), [7, 7]);

    for degenerate in [
        Tensor::zeros((1, 1), DType::F32, &device).unwrap(),
        Tensor::zeros((1, 1, 1, 1), DType::F32, &device).unwrap(),
    ] {
        assert_eq!(
            Fp8WeightScaleLayout::Block([128, 128])
                .normalize(degenerate, [64, 64])
                .unwrap()
                .dims(),
            [1, 1]
        );
    }

    assert!(Fp8WeightScaleLayout::Tensor
        .normalize(Tensor::zeros((1, 1), DType::F32, &device).unwrap(), [1, 1])
        .is_err());
    assert!(Fp8WeightScaleLayout::Channel
        .normalize(
            Tensor::zeros((1, 896, 1), DType::F32, &device).unwrap(),
            [896, 896]
        )
        .is_err());
    assert!(Fp8WeightScaleLayout::Block([128, 128])
        .normalize(
            Tensor::zeros((1, 7, 7, 1), DType::F32, &device).unwrap(),
            [896, 896]
        )
        .is_err());
    assert!(Fp8WeightScaleLayout::Block([0, 128])
        .logical_shape([896, 896])
        .is_err());

    let spec = Fp8LinearSpec::new(
        CheckpointDialect::ModelOpt,
        Fp8WeightScaleLayout::Tensor,
        Fp8ActivationMode::StaticTensor,
    );
    assert!(spec
        .normalize_activation_scale(Tensor::zeros(1, DType::F16, &device).unwrap())
        .unwrap()
        .dims()
        .is_empty());
    assert!(spec
        .normalize_activation_scale(Tensor::zeros((1, 1), DType::F32, &device).unwrap())
        .is_err());
}

#[test]
fn native_config_canonicalizes_language_model_prefixes() {
    let config = CheckpointQuantConfig::native(
        None,
        None,
        Some("e4m3"),
        &["model.language_model.embed_tokens".to_string()],
    )
    .unwrap();
    assert_eq!(config.resolve_fp8("model.embed_tokens").unwrap(), None);
    assert!(config
        .resolve_fp8("model.layers.0.mlp.up_proj")
        .unwrap()
        .is_some());
    assert!(CheckpointQuantConfig::native(None, None, Some("e5m2"), &[]).is_err());
}

#[test]
fn duplicate_scale_aliases_are_rejected() {
    let scale = Tensor::zeros((), DType::F32, &candle_core::Device::Cpu).unwrap();
    let vb = crate::ShardedSafeTensors::wrap(
        HashMap::from([
            ("layer.weight_scale".to_string(), scale.clone()),
            ("layer.weight_scale_inv".to_string(), scale),
        ]),
        DType::BF16,
        candle_core::Device::Cpu,
    )
    .pp("layer");
    let config = CheckpointQuantConfig::native(None, None, None, &[]).unwrap();
    let spec = config.resolve_fp8("layer").unwrap().unwrap();
    assert!(spec.scale_names.weight_name(&vb).is_err());
}

#[test]
fn native_missing_scale_falls_back_only_without_an_exclusion_policy() -> CandleResult<()> {
    let device = candle_core::Device::Cpu;
    let vb = crate::ShardedSafeTensors::wrap(
        HashMap::from([(
            "layer.weight".to_string(),
            Tensor::ones((2, 2), DType::F32, &device)?,
        )]),
        DType::F32,
        device,
    )
    .pp("layer");
    let permissive = crate::QuantizedConfig::Fp8 {
        weight_block_size: None,
        activation_scheme: None,
        fmt: None,
        modules_to_not_convert: Vec::new(),
    };
    let layer = checkpoint_linear_b(2, 2, &permissive, false, Default::default(), vb.clone())?;
    assert_eq!(layer.dequantize_w()?.dtype(), DType::F32);

    let explicit = crate::QuantizedConfig::Fp8 {
        weight_block_size: None,
        activation_scheme: None,
        fmt: None,
        modules_to_not_convert: vec!["lm_head".to_string()],
    };
    let error = checkpoint_linear_b(2, 2, &explicit, false, Default::default(), vb)
        .expect_err("a non-excluded native FP8 module must have a scale");
    assert!(error.to_string().contains("missing FP8 weight scale"));
    Ok(())
}

#[test]
fn native_excluded_module_rejects_a_stray_scale() -> CandleResult<()> {
    let device = candle_core::Device::Cpu;
    let vb = crate::ShardedSafeTensors::wrap(
        HashMap::from([
            (
                "layer.weight".to_string(),
                Tensor::ones((2, 2), DType::F32, &device)?,
            ),
            (
                "layer.weight_scale".to_string(),
                Tensor::ones((), DType::F32, &device)?,
            ),
        ]),
        DType::F32,
        device,
    )
    .pp("layer");
    let config = crate::QuantizedConfig::Fp8 {
        weight_block_size: None,
        activation_scheme: None,
        fmt: None,
        modules_to_not_convert: vec!["layer".to_string()],
    };
    let error = checkpoint_linear_b(2, 2, &config, false, Default::default(), vb)
        .expect_err("an excluded native module must not retain a scale");
    assert!(error
        .to_string()
        .contains("unexpectedly has a weight scale"));
    Ok(())
}

#[test]
fn compressed_tensors_target_requires_a_scale() -> CandleResult<()> {
    let device = candle_core::Device::Cpu;
    let vb = crate::ShardedSafeTensors::wrap(
        HashMap::from([(
            "layer.weight".to_string(),
            Tensor::ones((2, 2), DType::F32, &device)?,
        )]),
        DType::F32,
        device,
    )
    .pp("layer");
    let config: crate::QuantizedConfig = serde_json::from_value(json!({
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": ct_args("tensor")
            }
        }
    }))
    .unwrap();
    let error = checkpoint_linear_b(2, 2, &config, false, Default::default(), vb)
        .expect_err("a compressed-tensors FP8 target must have a scale");
    assert!(error.to_string().contains("missing FP8 weight scale"));
    Ok(())
}

#[test]
fn model_opt_fused_tensor_scales_follow_projection_chunks() -> CandleResult<()> {
    use float8::F8E4M3;

    const N: usize = 8;
    const K: usize = 4;

    let device = candle_core::Device::Cpu;
    let weights = Tensor::from_vec(vec![F8E4M3::from_f32(1.0); N * K], (N, K), &device)?;
    let vb = crate::ShardedSafeTensors::wrap(
        HashMap::from([
            (
                "model.layers.0.mlp.gate_up_proj.weight".to_string(),
                weights,
            ),
            (
                "model.layers.0.mlp.gate_up_proj.weight_scale".to_string(),
                Tensor::from_vec(vec![2f32, 3.0], 2, &device)?,
            ),
            (
                "model.layers.0.mlp.gate_up_proj.input_scale".to_string(),
                Tensor::from_vec(vec![0.5f32, 0.75], 2, &device)?,
            ),
        ]),
        DType::BF16,
        device,
    )
    .pp("model.layers.0.mlp.gate_up_proj");
    let config: crate::QuantizedConfig = serde_json::from_value(json!({
        "quant_method": "modelopt",
        "quant_algo": "FP8"
    }))
    .unwrap();

    for (rank, expected) in [(0, 2f32), (1, 2f32), (2, 3f32), (3, 3f32)] {
        let layer = checkpoint_linear_b(
            K,
            N,
            &config,
            false,
            crate::Shard::Simple {
                dim: 0,
                rank,
                world_size: 4,
            },
            vb.clone(),
        )?;
        let weight = layer
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert_eq!(weight.len(), N / 4);
        assert!(weight.iter().flatten().all(|value| *value == expected));
    }
    Ok(())
}

#[test]
fn fused_tensor_scale_shards_require_exact_aligned_partitions() -> CandleResult<()> {
    for rank in 0..6 {
        assert_eq!(
            tensor_scale_shard(
                [12, 4],
                &[3],
                crate::Shard::Simple {
                    dim: 0,
                    rank,
                    world_size: 6,
                },
                Some(3),
            )?,
            crate::Shard::Offset {
                dim: 0,
                offset: rank / 2,
                len: 1,
            }
        );
    }
    assert_eq!(
        tensor_scale_shard(
            [12, 4],
            &[3],
            crate::Shard::Offset {
                dim: 0,
                offset: 8,
                len: 4,
            },
            Some(3),
        )?,
        crate::Shard::Offset {
            dim: 0,
            offset: 2,
            len: 1,
        }
    );
    assert!(tensor_scale_shard([12, 4], &[2], Default::default(), Some(3)).is_err());
    assert!(tensor_scale_shard([12, 4], &[3], Default::default(), Some(3)).is_err());
    assert!(tensor_scale_shard([12, 4], &[3], Default::default(), None).is_err());
    assert!(tensor_scale_shard(
        [12, 4],
        &[3],
        crate::Shard::Simple {
            dim: 0,
            rank: 0,
            world_size: 0,
        },
        Some(3),
    )
    .is_err());
    assert!(tensor_scale_shard(
        [12, 4],
        &[3],
        crate::Shard::Offset {
            dim: 0,
            offset: 2,
            len: 4,
        },
        Some(3),
    )
    .is_err());
    Ok(())
}
#[test]
fn model_opt_mxfp8_and_csf_keep_distinct_encodings() {
    let config = CheckpointQuantConfig::model_opt(&serde_json::json!({
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.vision_model.proj": {"quant_algo": "MXFP8", "group_size": 32},
                "model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"}
            }
        })).unwrap();
    assert_eq!(
        config.resolve_checked("model.vision_model.proj").unwrap(),
        Some(CheckpointLinearSpec::Mxfp8(Mxfp8LinearSpec {
            group_size: 32
        }))
    );
    let Some(CheckpointLinearSpec::Nvfp4(spec)) = config
        .resolve_checked("model.layers.0.mlp.experts.7.gate_proj")
        .unwrap()
    else {
        panic!("expected NVFP4 expert")
    };
    assert_eq!(spec.weight_scale_encoding, Nvfp4ScaleEncoding::Csf);
    assert!(CheckpointQuantConfig::model_opt(
        &serde_json::json!({"quant_algo": "MXFP8", "group_size": 16})
    )
    .is_err());
    assert!(CheckpointQuantConfig::model_opt(
        &serde_json::json!({"quant_algo": "MXFP8", "weight_scale_encoding": "csf"})
    )
    .is_err());
    assert!(CheckpointQuantConfig::model_opt(
        &serde_json::json!({"quant_algo": "NVFP4", "weight_scale_encoding": "other"})
    )
    .is_err());
    let legacy =
        CheckpointQuantConfig::model_opt(&serde_json::json!({"quant_algo": "FP8"})).unwrap();
    assert_eq!(
        legacy.resolve_checked("model.vision_model.proj").unwrap(),
        None
    );
}

#[test]
fn model_opt_external_csf_encoding_survives_merge() {
    let config = CheckpointQuantConfig::model_opt_merged(
        None,
        &serde_json::json!({
            "quant_algo": "NVFP4", "weight_scale_encoding": "csf"
        }),
    )
    .unwrap();
    let Some(CheckpointLinearSpec::Nvfp4(spec)) = config.resolve_checked("layer").unwrap() else {
        panic!("expected NVFP4")
    };
    assert_eq!(spec.weight_scale_encoding, Nvfp4ScaleEncoding::Csf);
}
