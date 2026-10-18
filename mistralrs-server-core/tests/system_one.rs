use std::{num::NonZeroUsize, path::PathBuf, time::Duration};

use candle_core::{DType, Device};
use mistralrs_core::{
    AutoLoaderBuilder, DefaultSchedulerMethod, DeviceMapSetting, MistralRsBuilder, SchedulerConfig,
    TokenSource,
};
use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;
use serde_json::{json, Value};

#[tokio::test]
async fn system_one_http_contract_and_model_routing() -> anyhow::Result<()> {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../mistralrs-core/tests/fixtures/clm");
    let loader = AutoLoaderBuilder::new(
        Default::default(),
        Default::default(),
        Default::default(),
        None,
        None,
        fixture.to_string_lossy().into_owned(),
        false,
        None,
    )
    .build();
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::None,
        &DType::F32,
        &Device::Cpu,
        true,
        DeviceMapSetting::dummy(),
        None,
        None,
    )?;
    let scheduler = SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(NonZeroUsize::new(1).unwrap()),
    };
    let state = MistralRsBuilder::new(pipeline, scheduler, false, None)
        .with_model_id("clm-test")
        .build()
        .await;
    state
        .register_model_alias("clm-alias", "clm-test")
        .map_err(anyhow::Error::msg)?;
    let router = MistralRsServerRouterBuilder::new()
        .with_mistralrs(state.clone())
        .with_include_swagger_routes(false)
        .build()
        .await?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let reference: Value = serde_json::from_str(include_str!(
        "../../mistralrs-core/tests/fixtures/clm/reference.json"
    ))?;
    let mut request = reference["request"].clone();
    for model in ["default", "clm-test", "clm-alias"] {
        request["model"] = json!(model);
        let response = client
            .post(format!("{base_url}/v1/systemone"))
            .json(&request)
            .send()
            .await?;
        assert_eq!(response.status(), 200);
        let response: Value = response.json().await?;
        assert_eq!(response["model"], fixture.to_string_lossy().as_ref());
        assert_eq!(response["answers"].as_object().unwrap().len(), 4);
        assert_eq!(response["answers"]["urgent"]["type"], "noul");
        assert_eq!(response["answers"]["team"]["type"], "choice");
        assert_eq!(response["answers"]["anger"]["type"], "score");
        assert_eq!(response["usage"]["output_tokens"], 0);
        if model == "default" {
            assert_eq!(response["usage"]["input_tokens"], reference["input_tokens"]);
        } else {
            assert_eq!(response["usage"]["input_tokens"], 0);
        }
    }
    let models: Value = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(models["object"], "list");
    assert!(models["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["id"] == "clm-test"));
    assert!(models["models"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["name"] == "clm-test"));

    request["model"] = json!("missing");
    assert_eq!(
        client
            .post(format!("{base_url}/v1/systemone"))
            .json(&request)
            .send()
            .await?
            .status(),
        404
    );
    request["model"] = json!("default");
    request["questions"] = json!({});
    assert_eq!(
        client
            .post(format!("{base_url}/v1/systemone"))
            .json(&request)
            .send()
            .await?
            .status(),
        400
    );
    request["questions"] = json!({"q":{"type":"noul"}});
    assert_eq!(
        client
            .post(format!("{base_url}/v1/systemone"))
            .json(&request)
            .send()
            .await?
            .status(),
        200
    );
    request["state"] = json!("invoice ".repeat(2049));
    assert_eq!(
        client
            .post(format!("{base_url}/v1/systemone"))
            .json(&request)
            .send()
            .await?
            .status(),
        400
    );
    for (route, request) in [
        ("embeddings", json!({"model":"default","input":"invoice"})),
        (
            "chat/completions",
            json!({"model":"default","messages":[{"role":"user","content":"invoice"}]}),
        ),
    ] {
        assert_eq!(
            client
                .post(format!("{base_url}/v1/{route}"))
                .json(&request)
                .send()
                .await?
                .status(),
            400
        );
    }
    if let Ok(python) = std::env::var("MISTRALRS_TEST_SYSTEM_ONE_PYTHON") {
        let output = tokio::process::Command::new(python)
            .arg(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../scripts/test_system_one_clients.py"),
            )
            .env("SYSTEM_ONE_BASE_URL", &base_url)
            .env("SYSTEM_ONE_MODEL", "clm-test")
            .env_remove("SYSTEM_ONE_API_KEY")
            .output()
            .await?;
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    drop(client);
    let _ = stop.send(());
    server.await??;
    state.shutdown().await.map_err(anyhow::Error::msg)?;
    Ok(())
}
