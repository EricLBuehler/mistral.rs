use std::path::PathBuf;

use candle_core::{DType, Device};
use mistralrs_core::{
    decision::{DecisionRequest, DecisionResponse, DecisionValidationError, CLM_MAX_TOKENS},
    AutoLoaderBuilder, DeviceMapSetting, ModelCategory, TokenSource,
};
use serde_json::Value;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clm")
}

fn assert_close(actual: &Value, expected: &Value, tolerance: f64) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            assert!(
                (a.as_f64().unwrap() - e.as_f64().unwrap()).abs() < tolerance,
                "{a} != {e}"
            );
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len());
            for (key, value) in e {
                assert_close(&a[key], value, tolerance);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

async fn check_clm(
    device: Device,
    tolerance: f64,
    mapper: DeviceMapSetting,
    dtype: DType,
) -> anyhow::Result<()> {
    let model_id = fixture().to_string_lossy().into_owned();
    let loader = AutoLoaderBuilder::new(
        Default::default(),
        Default::default(),
        Default::default(),
        None,
        None,
        model_id.clone(),
        false,
        None,
    )
    .build();
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::None,
        &dtype,
        &device,
        true,
        mapper.clone(),
        None,
        None,
    )?;
    let pipeline = pipeline.lock().await;
    assert_eq!(pipeline.category(), ModelCategory::Decision);
    assert_eq!(pipeline.get_metadata().max_seq_len, CLM_MAX_TOKENS);
    let reference: Value = serde_json::from_str(include_str!("fixtures/clm/reference.json"))?;
    let request: DecisionRequest = serde_json::from_value(reference["request"].clone())?;
    let response: DecisionResponse = pipeline.decide(&request)?;
    assert_eq!(response.model, model_id);
    assert_eq!(
        response.usage.input_tokens,
        reference["input_tokens"].as_u64().unwrap() as usize
    );
    assert_eq!(response.usage.output_tokens, 0);
    assert_eq!(response.usage.billing_units, request.questions.len());
    assert_close(
        &serde_json::to_value(&response.answers)?,
        &reference["answers"],
        tolerance,
    );

    let cached = pipeline.decide(&request)?;
    assert_eq!(cached.usage.input_tokens, 0);
    assert_eq!(cached.usage.billing_units, request.questions.len());
    assert_close(
        &serde_json::to_value(&cached.answers)?,
        &reference["answers"],
        tolerance,
    );

    for (key, question) in &request.questions {
        let mut single = request.clone();
        single.questions = [(key.clone(), question.clone())].into_iter().collect();
        let response = pipeline.decide(&single)?;
        assert_eq!(response.usage.input_tokens, 0);
        assert_close(
            &serde_json::to_value(&response.answers[key])?,
            &reference["answers"][key],
            tolerance,
        );
    }
    let mut changed_state = request.clone();
    changed_state.state = Value::String("technical invoice invoice".to_string());
    let first = pipeline.decide(&changed_state)?;
    assert!(first.usage.input_tokens > 0);
    let pairs = changed_state
        .questions
        .values()
        .map(|q| {
            let mut single = changed_state.clone();
            single.questions = [("q".to_string(), q.clone())].into_iter().collect();
            pipeline.decide(&single)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    assert!(pairs.iter().all(|r| r.usage.input_tokens == 0));

    let independent = loader.load_model_from_hf(
        None,
        TokenSource::None,
        &dtype,
        &device,
        true,
        mapper,
        None,
        None,
    )?;
    let independent = independent.lock().await;
    independent.decide(&request)?;
    let mut shared = request.clone();
    shared.state = Value::String("invoice technical ".repeat(24));
    let combined = pipeline.decide(&shared)?;
    assert!(combined.usage.input_tokens > 0);
    for (key, question) in &shared.questions {
        let mut single = shared.clone();
        single.questions = [(key.clone(), question.clone())].into_iter().collect();
        let separate = independent.decide(&single)?;
        assert!(separate.usage.input_tokens > 0);
        assert_close(
            &serde_json::to_value(&combined.answers[key])?,
            &serde_json::to_value(&separate.answers[key])?,
            tolerance,
        );
    }
    let mut too_long = request;
    too_long.state = Value::String("invoice ".repeat(CLM_MAX_TOKENS + 1));
    assert!(pipeline
        .decide(&too_long)
        .unwrap_err()
        .is::<DecisionValidationError>());
    Ok(())
}

#[tokio::test]
async fn clm_cpu_matches_pytorch_encoder_and_heads() -> anyhow::Result<()> {
    check_clm(Device::Cpu, 2e-5, DeviceMapSetting::dummy(), DType::F32).await
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn clm_metal_matches_pytorch_encoder_and_heads() -> anyhow::Result<()> {
    check_clm(
        Device::new_metal(0)?,
        2e-4,
        DeviceMapSetting::dummy(),
        DType::F32,
    )
    .await
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn clm_metal_cpu_offloading_preserves_bf16_decisions() -> anyhow::Result<()> {
    let mapper = DeviceMapSetting::Map(mistralrs_core::DeviceMapMetadata::from_num_device_layers(
        vec![],
    ));
    check_clm(Device::new_metal(0)?, 1e-2, mapper, DType::BF16).await
}

#[cfg(feature = "cuda")]
#[tokio::test]
async fn clm_cuda_matches_pytorch_encoder_and_heads() -> anyhow::Result<()> {
    check_clm(
        Device::new_cuda(0)?,
        2e-4,
        DeviceMapSetting::dummy(),
        DType::F32,
    )
    .await
}
