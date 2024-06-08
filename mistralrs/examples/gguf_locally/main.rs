use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    Constraint, Device, DeviceMapMetadata, GGUFLoaderBuilder, GGUFSpecificConfig, MistralRs,
    MistralRsBuilder, NormalRequest, Request, RequestMessage, Response, SamplingParams,
    SchedulerMethod, TokenSource,
};

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    // We do not use any files from HF servers here, and instead load the
    // chat template from the specified file, and the tokenizer and model from a
    // local GGUF file at the path `.`
    let loader = GGUFLoaderBuilder::new(
        GGUFSpecificConfig { repeat_last_n: 64 },
        Some("chat_templates/mistral.json".to_string()),
        None,
        ".".to_string(),
        "mistral-7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
    )
    .build();
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        None,
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
        None,
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build())
}

fn main() -> anyhow::Result<()> {
    let mistralrs = setup()?;

    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        messages: RequestMessage::Completion {
            text: "Hello! My name is ".to_string(),
            echo_prompt: false,
            best_of: 1,
        },
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        adapters: None,
    });
    mistralrs.get_sender().blocking_send(request)?;

    let response = rx.blocking_recv().unwrap();
    match response {
        Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
        _ => unreachable!(),
    }
    Ok(())
}
