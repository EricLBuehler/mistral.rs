use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_quant::{
    linear_no_bias, CheckpointLinearSpec, Nvfp4ActivationMode, Nvfp4Layer, Nvfp4LayerParts,
    Nvfp4LinearSpec, QuantMethod, QuantizedConfig, Shard, ShardedSafeTensors,
};
use serde_json::json;

const INPUT_DIM: usize = 32;
const OUTPUT_DIM: usize = 4;
const PREFIX: &str = "model.layers.0.mlp.gate_up_proj";
const FP4_PACKED: [u8; 8] = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
const FP4_DECODED: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const BLOCK_SCALES: [f32; 8] = [1.0, 2.0, 0.5, 4.0, 3.0, 1.5, 0.25, 8.0];
const GLOBAL_SCALE: f32 = 0.25;
const INPUT_SCALE: f32 = 0.25;
const FIXTURE_TOLERANCE: f32 = 0.00001;
const TIE_INPUT: [f32; 16] = [
    0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 6.0, -6.0,
];

#[derive(Clone, Copy)]
enum Format {
    ModelOpt,
    CompressedTensors,
}

impl Format {
    fn config(self, a4: bool) -> Result<QuantizedConfig> {
        let value = match self {
            Self::ModelOpt => json!({
                "quant_method": "modelopt",
                "quant_algo": if a4 { "NVFP4" } else { "W4A16_NVFP4" },
                "group_size": 16
            }),
            Self::CompressedTensors => {
                let weights = json!({
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "dynamic": false,
                    "symmetric": true
                });
                let mut input = weights.clone();
                input["dynamic"] = json!("local");
                json!({
                    "quant_method": "compressed-tensors",
                    "format": "nvfp4-pack-quantized",
                    "config_groups": {"group_0": {
                        "targets": ["Linear"],
                        "weights": weights,
                        "input_activations": if a4 { input } else { serde_json::Value::Null }
                    }}
                })
            }
        };
        serde_json::from_value(value).map_err(candle_core::Error::msg)
    }

    fn spec(self, a4: bool) -> Result<Nvfp4LinearSpec> {
        match self.config(a4)?.resolve_checkpoint(PREFIX)? {
            Some(CheckpointLinearSpec::Nvfp4(spec)) => Ok(spec),
            _ => candle_core::bail!("expected NVFP4 configuration"),
        }
    }

    fn globals(self, dequant: &[f32]) -> Vec<f32> {
        dequant
            .iter()
            .map(|value| match self {
                Self::ModelOpt => *value,
                Self::CompressedTensors => value.recip(),
            })
            .collect()
    }

    fn tensors(self, a4: bool) -> Result<HashMap<String, Tensor>> {
        let spec = self.spec(a4)?;
        let device = &Device::Cpu;
        let mut tensors = HashMap::from([
            (
                format!("{PREFIX}.{}", spec.scale_names.weight),
                Tensor::from_vec(
                    FP4_PACKED.repeat(INPUT_DIM * OUTPUT_DIM / (2 * FP4_PACKED.len())),
                    (OUTPUT_DIM, INPUT_DIM / 2),
                    device,
                )?,
            ),
            (
                format!("{PREFIX}.{}", spec.scale_names.block_scale),
                Tensor::from_vec(
                    BLOCK_SCALES.into_iter().map(F8E4M3::from_f32).collect(),
                    (OUTPUT_DIM, 2),
                    device,
                )?,
            ),
            (
                format!("{PREFIX}.{}", spec.scale_names.global_scale),
                Tensor::from_vec(self.globals(&[GLOBAL_SCALE]), 1, device)?,
            ),
        ]);
        if let Some(name) = spec.scale_names.activation_scale {
            tensors.insert(
                format!("{PREFIX}.{name}"),
                Tensor::from_vec(self.globals(&[INPUT_SCALE]), 1, device)?,
            );
        }
        Ok(tensors)
    }

    fn load(self, a4: bool, shard: Shard, tensors: HashMap<String, Tensor>) -> Result<Nvfp4Layer> {
        Nvfp4Layer::load(
            INPUT_DIM,
            OUTPUT_DIM,
            self.spec(a4)?,
            false,
            shard,
            ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp(PREFIX),
        )
    }
}

fn expected_weights(globals: [f32; OUTPUT_DIM]) -> Vec<Vec<f32>> {
    globals
        .into_iter()
        .enumerate()
        .map(|(row, global)| {
            FP4_DECODED
                .into_iter()
                .map(|value| value * BLOCK_SCALES[2 * row] * global)
                .chain(
                    FP4_DECODED
                        .into_iter()
                        .map(|value| value * BLOCK_SCALES[2 * row + 1] * global),
                )
                .collect()
        })
        .collect()
}

#[test]
fn checkpoint_dialects_decode_identically_with_reciprocal_globals() -> Result<()> {
    let expected = expected_weights([GLOBAL_SCALE; OUTPUT_DIM]);
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let layer = linear_no_bias(
            INPUT_DIM,
            OUTPUT_DIM,
            &Some(format.config(false)?),
            ShardedSafeTensors::wrap(format.tensors(false)?, DType::F32, Device::Cpu).pp(PREFIX),
        )?;
        assert_eq!(layer.dequantize_w()?.to_vec2::<f32>()?, expected);
    }
    Ok(())
}

#[test]
fn w4a16_ignores_stale_modelopt_input_scale() -> Result<()> {
    let mut tensors = Format::ModelOpt.tensors(false)?;
    tensors.insert(
        format!("{PREFIX}.input_scale"),
        Tensor::new(f32::NAN, &Device::Cpu)?,
    );
    let layer = Format::ModelOpt.load(false, Shard::default(), tensors)?;
    let mut input = vec![0.0; INPUT_DIM];
    input[..TIE_INPUT.len()].copy_from_slice(&TIE_INPUT);
    let output = layer.forward(&Tensor::from_vec(input, (1, INPUT_DIM), &Device::Cpu)?)?;
    assert_eq!(
        output.to_vec2::<f32>()?,
        vec![vec![19.59375, 9.796875, 58.78125, 4.8984375]]
    );
    Ok(())
}

#[test]
fn w4a4_rounds_ties_to_even_and_keeps_zero_blocks_finite() -> Result<()> {
    let mut input = vec![0.0; 2 * INPUT_DIM];
    input[..TIE_INPUT.len()].copy_from_slice(&TIE_INPUT);
    let input = Tensor::from_vec(input, (2, 1, INPUT_DIM), &Device::Cpu)?;
    let expected = vec![
        vec![vec![18.5, 9.25, 55.5, 4.625]],
        vec![vec![0.0; OUTPUT_DIM]],
    ];
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let layer = format.load(true, Shard::default(), format.tensors(true)?)?;
        assert_eq!(layer.forward(&input)?.to_vec3::<f32>()?, expected);
    }
    Ok(())
}

#[test]
fn logical_input_shards_slice_packed_values_and_block_scales_together() -> Result<()> {
    let expected = expected_weights([GLOBAL_SCALE; OUTPUT_DIM]);
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        for shard in [
            Shard::Simple {
                dim: 1,
                rank: 1,
                world_size: 2,
            },
            Shard::Offset {
                dim: 1,
                offset: 16,
                len: 16,
            },
        ] {
            let layer = format.load(false, shard, format.tensors(false)?)?;
            let actual = layer.dequantize_w()?.to_vec2::<f32>()?;
            let expected: Vec<Vec<_>> = expected.iter().map(|row| row[16..].to_vec()).collect();
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}

#[test]
fn fused_projection_globals_follow_output_shards() -> Result<()> {
    let globals = [GLOBAL_SCALE, GLOBAL_SCALE / 2.0];
    let expected = expected_weights([globals[0], globals[0], globals[1], globals[1]]);
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let mut tensors = format.tensors(false)?;
        let name = format.spec(false)?.scale_names.global_scale;
        tensors.insert(
            format!("{PREFIX}.{name}"),
            Tensor::from_vec(format.globals(&globals), 2, &Device::Cpu)?,
        );
        let layer = format.load(false, Shard::default(), tensors.clone())?;
        assert_eq!(layer.dequantize_w()?.to_vec2::<f32>()?, expected);
        let layer = format.load(
            false,
            Shard::Simple {
                dim: 0,
                rank: 1,
                world_size: 2,
            },
            tensors,
        )?;
        assert_eq!(layer.dequantize_w()?.to_vec2::<f32>()?, expected[2..]);
    }
    Ok(())
}

#[test]
fn input_shards_must_preserve_nvfp4_groups() -> Result<()> {
    for shard in [
        Shard::Offset {
            dim: 1,
            offset: 8,
            len: 16,
        },
        Shard::Simple {
            dim: 1,
            rank: 0,
            world_size: 4,
        },
    ] {
        let error = Format::ModelOpt
            .load(false, shard, Format::ModelOpt.tensors(false)?)
            .unwrap_err();
        assert!(error.to_string().contains("align to 16"));
    }
    Ok(())
}

#[test]
fn missing_checkpoint_scales_are_errors_for_each_dialect() -> Result<()> {
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let spec = format.spec(true)?;
        for name in [
            spec.scale_names.block_scale,
            spec.scale_names.global_scale,
            spec.scale_names.activation_scale.unwrap(),
        ] {
            let mut tensors = format.tensors(true)?;
            tensors.remove(&format!("{PREFIX}.{name}"));
            let error = format.load(true, Shard::default(), tensors).unwrap_err();
            assert!(error.to_string().contains(name));
        }
    }
    Ok(())
}

#[test]
fn malformed_checkpoint_scales_are_rejected() -> Result<()> {
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let spec = format.spec(true)?;
        for global in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let mut tensors = format.tensors(true)?;
            tensors.insert(
                format!("{PREFIX}.{}", spec.scale_names.global_scale),
                Tensor::new(global, &Device::Cpu)?,
            );
            assert!(format
                .load(true, Shard::default(), tensors)
                .unwrap_err()
                .to_string()
                .contains("finite and positive"));
        }
        let mut tensors = format.tensors(true)?;
        tensors.insert(
            format!("{PREFIX}.{}", spec.scale_names.block_scale),
            Tensor::zeros((OUTPUT_DIM, 1), DType::F8E4M3, &Device::Cpu)?,
        );
        assert!(format.load(true, Shard::default(), tensors).is_err());
        let mut tensors = format.tensors(true)?;
        tensors.insert(
            format!("{PREFIX}.{}", spec.scale_names.global_scale),
            Tensor::ones(3, DType::F32, &Device::Cpu)?,
        );
        assert!(format
            .load(true, Shard::default(), tensors)
            .unwrap_err()
            .to_string()
            .contains("weight rows"));
        let mut tensors = format.tensors(true)?;
        tensors.insert(
            format!("{PREFIX}.{}", spec.scale_names.activation_scale.unwrap()),
            Tensor::ones(2, DType::F32, &Device::Cpu)?,
        );
        assert!(format
            .load(true, Shard::default(), tensors)
            .unwrap_err()
            .to_string()
            .contains("input scale"));
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct ProjectionFixture {
    prefix: String,
    in_dim: usize,
    out_dim: usize,
    input: Vec<f32>,
    expected: Vec<f32>,
}

#[test]
#[ignore = "requires scripts/fetch_nvfp4_test_fixtures.py output in MISTRALRS_NVFP4_FIXTURE_DIR"]
fn real_checkpoint_projections_match_external_references() -> Result<()> {
    let directory = std::env::var("MISTRALRS_NVFP4_FIXTURE_DIR")
        .map(std::path::PathBuf::from)
        .map_err(candle_core::Error::msg)?;
    for (stem, format, a4) in [
        ("modelopt-qwen3.6-expert-down", Format::ModelOpt, false),
        ("ct-moe-expert0-down", Format::CompressedTensors, true),
    ] {
        let manifest: ProjectionFixture =
            serde_json::from_slice(&std::fs::read(directory.join(format!("{stem}.json")))?)
                .map_err(candle_core::Error::msg)?;
        let tensors = candle_core::safetensors::load(
            directory.join(format!("{stem}.safetensors")),
            &Device::Cpu,
        )?;
        let layer = linear_no_bias(
            manifest.in_dim,
            manifest.out_dim,
            &Some(format.config(a4)?),
            ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp(&manifest.prefix),
        )?;
        let input = Tensor::from_vec(manifest.input, (1, manifest.in_dim), &Device::Cpu)?;
        let actual = layer.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), manifest.expected.len());
        for (index, (actual, expected)) in actual.into_iter().zip(manifest.expected).enumerate() {
            assert!(
                (actual - expected).abs() <= FIXTURE_TOLERANCE * expected.abs().max(1.0),
                "{stem} output[{index}]={actual}, expected {expected}"
            );
        }
    }
    Ok(())
}

const GATHER_EXPECTED_ROW: [f32; OUTPUT_DIM] = [18.5, 9.25, 55.5, 4.625];

fn gather_checkpoint_stack(format: Format) -> Result<Nvfp4Layer> {
    let mut layers = Vec::new();
    for multiplier in [1.0, 2.0] {
        let mut tensors = format.tensors(true)?;
        let name = format.spec(true)?.scale_names.global_scale;
        tensors.insert(
            format!("{PREFIX}.{name}"),
            Tensor::from_vec(
                format.globals(&[GLOBAL_SCALE * multiplier]),
                1,
                &Device::Cpu,
            )?,
        );
        layers.push(format.load(true, Shard::default(), tensors)?);
    }
    Nvfp4Layer::stack(layers)
}

fn gather_input(multipliers: &[f32]) -> Vec<f32> {
    multipliers
        .iter()
        .flat_map(|multiplier| {
            TIE_INPUT
                .into_iter()
                .map(move |value| value * multiplier)
                .chain(std::iter::repeat_n(0.0, INPUT_DIM - TIE_INPUT.len()))
        })
        .collect()
}

fn gather_expected(multipliers: &[f32]) -> Vec<f32> {
    multipliers
        .iter()
        .flat_map(|multiplier| GATHER_EXPECTED_ROW.map(|value| value * multiplier))
        .collect()
}

#[test]
fn cpu_checkpoint_gather_broadcasts_shared_token_rows() -> Result<()> {
    let ids = Tensor::new(&[[0u32, 1], [1, 0]], &Device::Cpu)?;
    let input = Tensor::from_vec(gather_input(&[1.0, -1.0]), (2, INPUT_DIM), &Device::Cpu)?;
    let expected = gather_expected(&[1.0, 2.0, -2.0, -1.0]);
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let layer = gather_checkpoint_stack(format)?;
        for input in [input.clone(), input.unsqueeze(1)?] {
            let output = layer.gather_forward(&input, &ids)?;
            assert_eq!(output.dims(), [2, 2, OUTPUT_DIM]);
            assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, expected);
        }
        let output = layer.gather_forward(
            &input.reshape((1, 2, 1, 1, INPUT_DIM))?,
            &ids.reshape((1, 2, 2))?,
        )?;
        assert_eq!(output.dims(), [1, 2, 2, 1, OUTPUT_DIM]);
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, expected);
    }
    Ok(())
}

#[test]
fn cpu_checkpoint_gather_preserves_distinct_routed_rows() -> Result<()> {
    let ids = Tensor::new(&[[0u32, 1], [1, 0]], &Device::Cpu)?;
    let input = Tensor::from_vec(
        gather_input(&[1.0, -1.0, 2.0, -2.0]),
        (2, 2, INPUT_DIM),
        &Device::Cpu,
    )?;
    let expected = gather_expected(&[1.0, -2.0, 4.0, -2.0]);
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let layer = gather_checkpoint_stack(format)?;
        let output = layer.gather_forward(&input, &ids)?;
        assert_eq!(output.dims(), [2, 2, OUTPUT_DIM]);
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, expected);
        for input in [
            input.reshape((1, 2, 2, INPUT_DIM))?,
            input.reshape((1, 2, 2, 1, INPUT_DIM))?,
        ] {
            let output = layer.gather_forward(&input, &ids.reshape((1, 2, 2))?)?;
            assert_eq!(output.dims(), [1, 2, 2, 1, OUTPUT_DIM]);
            assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, expected);
        }
    }
    Ok(())
}

fn layer_parts() -> Result<Nvfp4LayerParts> {
    let mut tensors = Format::ModelOpt.tensors(true)?;
    Ok(Nvfp4LayerParts {
        weight: tensors.remove(&format!("{PREFIX}.weight")).unwrap(),
        scales: tensors.remove(&format!("{PREFIX}.weight_scale")).unwrap(),
        global_scales: Tensor::full(GLOBAL_SCALE, OUTPUT_DIM, &Device::Cpu)?,
        input_scale: Some(Tensor::new(INPUT_SCALE, &Device::Cpu)?),
        activation: Nvfp4ActivationMode::DynamicBlock,
        bias: None,
        dtype: DType::F32,
    })
}

#[test]
fn public_parts_reject_inconsistent_activation_and_bias_contracts() -> Result<()> {
    let mut parts = layer_parts()?;
    parts.activation = Nvfp4ActivationMode::None;
    assert!(Nvfp4Layer::from_parts(parts)
        .unwrap_err()
        .to_string()
        .contains("W4A16"));
    let mut parts = layer_parts()?;
    parts.input_scale = None;
    assert!(Nvfp4Layer::from_parts(parts)
        .unwrap_err()
        .to_string()
        .contains("W4A4"));
    for bias in [
        Tensor::zeros(OUTPUT_DIM + 1, DType::F32, &Device::Cpu)?,
        Tensor::zeros(OUTPUT_DIM, DType::BF16, &Device::Cpu)?,
    ] {
        let mut parts = layer_parts()?;
        parts.bias = Some(bias);
        assert!(Nvfp4Layer::from_parts(parts)
            .unwrap_err()
            .to_string()
            .contains("bias"));
    }
    let mut parts = layer_parts()?;
    parts.dtype = DType::U8;
    assert!(Nvfp4Layer::from_parts(parts)
        .unwrap_err()
        .to_string()
        .contains("output dtype"));
    Ok(())
}

#[test]
fn public_parts_reject_invalid_global_values_and_empty_weights() -> Result<()> {
    for value in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        for input in [false, true] {
            let mut parts = layer_parts()?;
            if input {
                parts.input_scale = Some(Tensor::new(value, &Device::Cpu)?);
            } else {
                parts.global_scales = Tensor::full(value, OUTPUT_DIM, &Device::Cpu)?;
            }
            assert!(Nvfp4Layer::from_parts(parts)
                .unwrap_err()
                .to_string()
                .contains("finite and positive"));
        }
    }
    for shape in [(0, INPUT_DIM / 2), (OUTPUT_DIM, 0)] {
        let mut parts = layer_parts()?;
        parts.weight = Tensor::zeros(shape, DType::U8, &Device::Cpu)?;
        assert!(Nvfp4Layer::from_parts(parts)
            .unwrap_err()
            .to_string()
            .contains("nonzero"));
    }
    Ok(())
}

#[test]
fn expert_stacks_reject_inconsistent_bias_presence() -> Result<()> {
    for biased_first in [false, true] {
        let mut layers = Vec::new();
        for biased in [biased_first, !biased_first] {
            let mut parts = layer_parts()?;
            if biased {
                parts.bias = Some(Tensor::zeros(OUTPUT_DIM, DType::F32, &Device::Cpu)?);
            }
            layers.push(Nvfp4Layer::from_parts(parts)?);
        }
        assert!(Nvfp4Layer::stack(layers)
            .unwrap_err()
            .to_string()
            .contains("bias presence"));
    }
    Ok(())
}

#[test]
fn reference_keeps_global_scale_precision_until_output_conversion() -> Result<()> {
    const ROWS: usize = 11;
    const WEIGHT_GLOBAL: f32 = 0.00314159;
    const ACTIVATION_GLOBAL: f32 = 0.01113;
    let values = (0..ROWS * INPUT_DIM)
        .map(|i| ((i * 11 % 47) as f32 - 23.0) / 13.0)
        .collect::<Vec<_>>();
    for activation in [Nvfp4ActivationMode::None, Nvfp4ActivationMode::DynamicBlock] {
        for dtype in [DType::BF16, DType::F16] {
            let make_layer = |dtype| {
                let mut parts = layer_parts()?;
                parts.dtype = dtype;
                parts.activation = activation;
                parts.global_scales = Tensor::full(WEIGHT_GLOBAL, OUTPUT_DIM, &Device::Cpu)?;
                parts.input_scale = (activation == Nvfp4ActivationMode::DynamicBlock)
                    .then(|| Tensor::new(ACTIVATION_GLOBAL, &Device::Cpu))
                    .transpose()?;
                Nvfp4Layer::from_parts(parts)
            };
            let input = Tensor::from_vec(values.clone(), (ROWS, 1, 1, 1, INPUT_DIM), &Device::Cpu)?
                .to_dtype(dtype)?;
            let expected = make_layer(DType::F32)?
                .forward(&input.to_dtype(DType::F32)?)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?;
            let actual = make_layer(dtype)?.forward(&input)?.to_dtype(DType::F32)?;
            assert_eq!(actual.dims(), [ROWS, 1, 1, 1, OUTPUT_DIM]);
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
    }
    Ok(())
}

#[test]
fn reference_normalizes_global_before_fp4_midpoint_rounding() -> Result<()> {
    // Separate divisions round below 0.75, whereas a combined divisor lands exactly on the midpoint.
    const GLOBAL_BITS: u32 = 0x3b08_8889;
    const MIDPOINT_INPUT_BITS: u32 = 0x3b00_0000;
    const MAX_INPUT_BITS: u32 = 0x3c80_0000;
    const EXPECTED_OUTPUT_BITS: u32 = 0x3aaa_aaab;
    let mut parts = layer_parts()?;
    let mut packed = vec![0u8; OUTPUT_DIM * INPUT_DIM / 2];
    for row in packed.chunks_mut(INPUT_DIM / 2) {
        row[0] = 0x20;
    }
    parts.weight = Tensor::from_vec(packed, (OUTPUT_DIM, INPUT_DIM / 2), &Device::Cpu)?;
    parts.scales = Tensor::ones((OUTPUT_DIM, 2), DType::F8E4M3, &Device::Cpu)?;
    parts.global_scales = Tensor::ones(OUTPUT_DIM, DType::F32, &Device::Cpu)?;
    parts.input_scale = Some(Tensor::new(f32::from_bits(GLOBAL_BITS), &Device::Cpu)?);
    let layer = Nvfp4Layer::from_parts(parts)?;
    let mut input = vec![0f32; INPUT_DIM];
    input[1] = f32::from_bits(MIDPOINT_INPUT_BITS);
    input[2] = f32::from_bits(MAX_INPUT_BITS);
    let output = layer.forward(&Tensor::from_vec(input, (1, INPUT_DIM), &Device::Cpu)?)?;
    assert_eq!(
        output.to_vec2::<f32>()?,
        vec![vec![f32::from_bits(EXPECTED_OUTPUT_BITS); OUTPUT_DIM]]
    );
    Ok(())
}

#[test]
fn invalid_input_shapes_fail_and_empty_batches_keep_output_shapes() -> Result<()> {
    let layer = Nvfp4Layer::from_parts(layer_parts()?)?;
    for shape in [vec![], vec![INPUT_DIM], vec![1, 0], vec![1, INPUT_DIM / 2]] {
        let input = Tensor::zeros(shape, DType::F32, &Device::Cpu)?;
        assert!(layer
            .forward(&input)
            .unwrap_err()
            .to_string()
            .contains("activation shape"));
    }
    let empty = Tensor::zeros((3, 0, INPUT_DIM), DType::F32, &Device::Cpu)?;
    assert_eq!(layer.forward(&empty)?.dims(), [3, 0, OUTPUT_DIM]);
    let experts = gather_checkpoint_stack(Format::ModelOpt)?;
    let empty = Tensor::zeros((0, INPUT_DIM), DType::F32, &Device::Cpu)?;
    let ids = Tensor::zeros((0, 2), DType::U32, &Device::Cpu)?;
    assert_eq!(
        experts.gather_forward(&empty, &ids)?.dims(),
        [0, 2, OUTPUT_DIM]
    );
    let input = Tensor::zeros((1, INPUT_DIM), DType::F32, &Device::Cpu)?;
    let ids = Tensor::zeros((1, 2), DType::I64, &Device::Cpu)?;
    assert!(experts
        .gather_forward(&input, &ids)
        .unwrap_err()
        .to_string()
        .contains("indices must be U32"));
    Ok(())
}

#[test]
fn stacked_checkpoint_shards_preserve_expert_scale_boundaries() -> Result<()> {
    const EXPERTS: usize = 3;
    for format in [Format::ModelOpt, Format::CompressedTensors] {
        let spec = format.spec(true)?;
        let mut tensors = format.tensors(true)?;
        for name in [spec.scale_names.weight, spec.scale_names.block_scale] {
            let key = format!("{PREFIX}.{name}");
            let tensor = tensors.remove(&key).unwrap();
            tensors.insert(key, Tensor::stack(&[&tensor; EXPERTS], 0)?);
        }
        let globals = [GLOBAL_SCALE, GLOBAL_SCALE * 2.0, GLOBAL_SCALE * 4.0];
        tensors.insert(
            format!("{PREFIX}.{}", spec.scale_names.global_scale),
            Tensor::from_vec(format.globals(&globals), EXPERTS, &Device::Cpu)?,
        );
        tensors.insert(
            format!("{PREFIX}.{}", spec.scale_names.activation_scale.unwrap()),
            Tensor::from_vec(
                format.globals(&[INPUT_SCALE, INPUT_SCALE * 2.0, INPUT_SCALE * 4.0]),
                EXPERTS,
                &Device::Cpu,
            )?,
        );
        let expected = Tensor::from_vec(
            globals
                .into_iter()
                .flat_map(|global| expected_weights([global; OUTPUT_DIM]).into_iter().flatten())
                .collect(),
            (EXPERTS, OUTPUT_DIM, INPUT_DIM),
            &Device::Cpu,
        )?;
        for (dim, offset, len) in [(0, 1, 1), (1, 1, 2), (2, 16, 16)] {
            let layer = Nvfp4Layer::load_stacked(
                EXPERTS,
                INPUT_DIM,
                OUTPUT_DIM,
                spec,
                false,
                Shard::Offset { dim, offset, len },
                ShardedSafeTensors::wrap(tensors.clone(), DType::F32, Device::Cpu).pp(PREFIX),
            )?;
            assert_eq!(
                layer.dequantize_w()?.to_vec3::<f32>()?,
                expected.narrow(dim, offset, len)?.to_vec3::<f32>()?
            );
        }
        tensors.insert(
            format!("{PREFIX}.{}", spec.scale_names.global_scale),
            Tensor::from_vec(format.globals(&globals[..2]), 2, &Device::Cpu)?,
        );
        assert!(Nvfp4Layer::load_stacked(
            EXPERTS,
            INPUT_DIM,
            OUTPUT_DIM,
            spec,
            false,
            Shard::default(),
            ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp(PREFIX),
        )
        .unwrap_err()
        .to_string()
        .contains("weight rows"));
    }
    Ok(())
}
